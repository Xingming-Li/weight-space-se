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

from src.eval import load_jsonl, load_model, score, summarize
from src.utils import load_config, set_seed


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--split", default="dev", choices=["dev"])  # Pilot is dev only
    ap.add_argument("--n_errors", type=int, default=40)
    args = ap.parse_args()

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

    summaries, all_records = {}, {}
    for lang in ["sv", "en"]:
        rows = load_jsonl(f"data/{lang}_{args.split}.jsonl")
        rec = score(model, tokenizer, rows, device)
        all_records[lang] = rec
        summaries[lang] = summarize(rec)
        with open(out_dir / f"pilot_{args.split}_{lang}_items.jsonl", "w",
                  encoding="utf-8") as f:
            for r in rec:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(env, indent=2))
    print(f"\n{'':4s}{'n':>6s}{'top1':>8s}{'top5':>8s}{'top10':>8s}"
          f"{'>10':>8s}{'>100':>8s}{'medRank':>9s}{'meanLP':>9s}")
    for lang, s in summaries.items():
        print(f"{lang:4s}{s['n']:6d}{s['top1']:8.3f}{s['top5']:8.3f}{s['top10']:8.3f}"
              f"{s['rank_gt10']:8.3f}{s['rank_gt100']:8.3f}{s['median_rank']:9d}"
              f"{s['mean_logprob']:9.2f}")

    # Wrong Swedish predictions for manual reading
    rng = random.Random(cfg["seed"])
    wrong = [r for r in all_records["sv"] if r["rank"] > 1]
    sample = rng.sample(wrong, min(args.n_errors, len(wrong)))
    lines = []
    for r in sample:
        lines.append(f"{r['masked']}\n  target: {r['target']} (rank {r['rank']})"
                     f"  top5: {', '.join(r['top'])}\n")
    (out_dir / "pilot_sv_errors.txt").write_text("\n".join(lines), encoding="utf-8")

    with open(out_dir / f"pilot_{args.split}_summary.json", "w") as f:
        json.dump({"env": env, "summary": summaries}, f, indent=2)
    print(f"\nwrote {len(sample)} wrong Swedish examples to results/pilot_sv_errors.txt")


if __name__ == "__main__":
    main()