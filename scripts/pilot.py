# Check if the sv-en gap exist and there's something to fix
# python -m scripts.pilot --models <Model A> <Model B>

import argparse
import gc
import json
import math
import platform
import sys
from pathlib import Path

import torch
import transformers

from src.belebele import load_jsonl, load_model, predict
from src.utils import load_config, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    rows = load_jsonl("data/belebele_dev.jsonl")   # Dev only by design
    n = len(rows)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    env = {"seed": cfg["seed"], "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "python": platform.python_version()}
    summary = {"env": env, "models": {}}

    for name in args.models:
        print(f"\n=== {name} ===")
        try:
            tok, model, rev = load_model(name)
            preds = {lang: predict(model, tok, rows, lang, args.batch_size)
                     for lang in ["sv", "en"]}
        except Exception as e:
            print("  FAILED:", repr(e)[:300])
            continue

        ok = {lang: [p["pred"] == r["answer"] for p, r in zip(preds[lang], rows)]
              for lang in preds}
        acc = {lang: sum(v) / n for lang, v in ok.items()}
        latent = sum(e and not s for s, e in zip(ok["sv"], ok["en"]))
        both_wrong = sum(not s and not e for s, e in zip(ok["sv"], ok["en"]))
        sv_only = sum(s and not e for s, e in zip(ok["sv"], ok["en"]))
        both_right = n - latent - both_wrong - sv_only
        letters = {lang: [sum(p["pred"] == k for p in preds[lang]) for k in range(4)]
                   for lang in preds}
        se = {lang: math.sqrt(acc[lang] * (1 - acc[lang]) / n) for lang in acc}

        s = {"revision": rev, "n": n, "acc_sv": acc["sv"], "acc_en": acc["en"],
             "gap_en_minus_sv": acc["en"] - acc["sv"],
             "se_sv": se["sv"], "se_en": se["en"],
             "latent_en_right_sv_wrong": latent, "both_wrong": both_wrong,
             "sv_right_en_wrong": sv_only, "both_right": both_right,
             "pred_letter_counts": letters}
        summary["models"][name] = s
        print(json.dumps(s, indent=2))
        with open(out_dir / f"pilot_{name.replace('/', '__')}.jsonl", "w") as f:
            for r, ps, pe in zip(rows, preds["sv"], preds["en"]):
                f.write(json.dumps({"id": r["id"], "answer": r["answer"],
                                    "sv": ps, "en": pe}) + "\n")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(out_dir / "pilot_models_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()