# Prepare parallel Swedish/English Belebele and split dev/test.
# python -m scripts.prepare_belebele --config configs/main.yaml

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import dataset_info

from src.utils import load_config, set_seed


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NEEDED = ["link", "question_number", "flores_passage", "question",
          "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4",
          "correct_answer_num"]


def pack(row):
    return {"passage": row["flores_passage"], "question": row["question"],
            "options": [row[f"mc_answer{i}"] for i in range(1, 5)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    d = cfg["data"]

    sha = dataset_info(d["dataset"]).sha
    by_lang = {}
    for code in d["langs"]:
        ds = load_dataset(d["dataset"], code, split="test", revision=sha)
        print(code, len(ds), "rows; columns:", ds.column_names)
        missing = [c for c in NEEDED if c not in ds.column_names]
        assert not missing, f"missing columns {missing}; paste the column list to me"
        by_lang[code] = {(r["link"], str(r["question_number"])): r for r in ds}

    sv, en = by_lang["swe_Latn"], by_lang["eng_Latn"]
    keys = sorted(set(sv) & set(en))
    print(f"aligned pairs: {len(keys)} (sv {len(sv)}, en {len(en)})")
    assert len(keys) > 0.9 * min(len(sv), len(en)), \
        "alignment by (link, question_number) failed; tell me"

    rows, mismatch = [], 0
    for k in keys:
        a_sv = int(sv[k]["correct_answer_num"]) - 1
        a_en = int(en[k]["correct_answer_num"]) - 1
        if a_sv != a_en:
            mismatch += 1
            continue
        rows.append({"id": f"{k[0]}#{k[1]}", "passage_id": k[0], "answer": a_sv,
                     "sv": pack(sv[k]), "en": pack(en[k])})
    print(f"dropped for answer-index mismatch: {mismatch}")

    by_passage = defaultdict(list)
    for r in rows:
        by_passage[r["passage_id"]].append(r)
    passages = sorted(by_passage)
    random.Random(cfg["seed"]).shuffle(passages)
    n_dev = round(d["dev_fraction"] * len(passages))
    split = {"dev": [r for p in passages[:n_dev] for r in by_passage[p]],
             "test": [r for p in passages[n_dev:] for r in by_passage[p]]}

    out = Path("data")
    out.mkdir(exist_ok=True)
    for name, rs in split.items():
        with open(out / f"belebele_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        answers = [r["answer"] for r in rs]
        print(f"{name}: {len(rs)} questions, "
              f"{len({r['passage_id'] for r in rs})} passages, "
              f"answer distribution {[answers.count(i) for i in range(4)]}")

    with open(out / "belebele_meta.json", "w") as f:
        json.dump({"dataset": d["dataset"], "revision": sha, "seed": cfg["seed"],
                   "dev_fraction": d["dev_fraction"], "dropped_mismatch": mismatch}, f, indent=2)
    r0 = split["dev"][0]
    print("\nexample sv question:", r0["sv"]["question"])
    print("example en question:", r0["en"]["question"], "| answer:", r0["answer"])


if __name__ == "__main__":
    main()