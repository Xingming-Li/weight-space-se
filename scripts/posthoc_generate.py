# POST-HOC EXPLORATORY: qualitative greedy continuations of held-out TEST passages,
# baseline vs real edits vs one matched random edit. Illustration only, not a metric.
# python -m scripts.posthoc_generate --config configs/main.yaml
# Requires results/posthoc_directions.pt (from posthoc_controls.py).

import argparse
import json
import sys
from pathlib import Path

import torch

from src.belebele import load_jsonl, load_model
from src.edit import language_direction, project_out, restore, untie_and_snapshot
from src.utils import load_config, set_seed

N_PASSAGES = 3
PREFIX_WORDS = 25
NEW_TOKENS = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    test = load_jsonl("data/belebele_test.jsonl")
    tok, model, rev = load_model(cfg["model"]["name"], revision=cfg["model"]["revision"])
    snap = untie_and_snapshot(model)
    means = torch.load("results/dev_means.pt")
    D = language_direction(means["S"], means["E"])[means["layers"].index(12)]
    U = torch.load("results/posthoc_directions.pt")["U"]

    seen, rows = set(), []
    for r in test:  # first N distinct test passages, in file order
        if r["passage_id"] not in seen:
            seen.add(r["passage_id"])
            rows.append(r)
        if len(rows) == N_PASSAGES:
            break

    arms = [("baseline", None, 0.0), ("real", D, 0.5), ("real", D, 1.0), ("matched#0", U[0], 0.5)]
    out = []
    for name, d, a in arms:
        restore(model, snap)
        if d is not None:
            project_out(model, d, alpha=a)
        for r in rows:
            for lang in ["sv", "en"]:
                prefix = " ".join(r[lang]["passage"].split()[:PREFIX_WORDS])
                enc = tok(prefix, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=NEW_TOKENS, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                cont = tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
                out.append({"arm": name, "alpha": a, "passage_id": r["passage_id"], "lang": lang,
                            "prefix": prefix, "continuation": cont})
                print(f"[{name} a={a} {lang}] ...{prefix[-40:]} >>> {cont!r}")
    restore(model, snap)
    with open("results/posthoc_generations.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
