# Baseline difficulty on the dev split only.
# python -m scripts.pilot --config configs/main.yaml

import argparse
import json
import platform
import random
import sys
from pathlib import Path

import torch
import transformers

from src.eval import load_jsonl, load_model, score, summarize_by_stratum
from src.utils import load_config, set_seed


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--n_errors", type=int, default=40)
    args = ap.parse_args()
    split = "dev"  # Pilot is dev-only by design

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    tokenizer, model, device = load_model(cfg)

    env = {
        "model": cfg["model"], "seed": cfg["seed"], "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(),
    }
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    summaries, records = {}, {}
    for lang in ["sv", "en"]:
        rows = load_jsonl(f"data/{lang}_{split}.jsonl")
        rec = score(model, tokenizer, rows, device)
        records[lang] = rec
        summaries[lang] = summarize_by_stratum(rec)
        with open(out_dir / f"pilot_{split}_{lang}_items.jsonl", "w",
                  encoding="utf-8") as f:
            for r in rec:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(env, indent=2))
    print(f"\n{'':4s}{'stratum':17s}{'n':>6s}{'top1':>8s}{'top5':>8s}"
          f"{'>100':>8s}{'medRk':>7s}{'meanLP':>9s}")
    for lang, by in summaries.items():
        for st, s in by.items():
            print(f"{lang:4s}{st:17s}{s['n']:6d}{s['top1']:8.3f}{s['top5']:8.3f}"
                  f"{s['rank_gt100']:8.3f}{s['median_rank']:7d}{s['mean_logprob']:9.2f}")

    # Wrong Swedish Non-template predictions, for manual reading
    rng = random.Random(cfg["seed"])
    wrong = [r for r in records["sv"] if r["rank"] > 1 and r["stratum"] != "template"]
    sample = rng.sample(wrong, min(args.n_errors, len(wrong)))
    lines = [f"[{r['stratum']}] {r['masked']}\n  target: {r['target']} "
             f"(rank {r['rank']})  top5: {', '.join(r['top'])}\n" for r in sample]
    (out_dir / "pilot_sv_errors.txt").write_text("\n".join(lines), encoding="utf-8")

    with open(out_dir / f"pilot_{split}_summary.json", "w") as f:
        json.dump({"env": env, "summary": summaries}, f, indent=2)
    print(f"\nwrote {len(sample)} wrong Swedish non-template examples "
          f"to results/pilot_sv_errors.txt")


if __name__ == "__main__":
    main()