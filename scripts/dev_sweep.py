# Sweep on dev with the fixed rule from the config.
# python -m scripts.dev_sweep --config configs/main.yaml

import argparse
import json
import sys
from pathlib import Path

import torch

from src.belebele import load_jsonl, load_model, predict
from src.edit import edit_targets, language_direction, project_out, restore, text_means, untie_and_snapshot
from src.utils import load_config, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def evaluate(model, tokenizer, rows, base=None):
    preds = {l: [p["pred"] for p in predict(model, tokenizer, rows, l)] for l in ["sv", "en"]}
    ok = {l: [p == r["answer"] for p, r in zip(preds[l], rows)] for l in preds}
    res = {"acc_sv": sum(ok["sv"]) / len(rows), "acc_en": sum(ok["en"]) / len(rows)}
    if base is not None:
        for l in ["sv", "en"]:
            res[f"fixed_{l}"] = sum(a and not b for a, b in zip(ok[l], base[l]))
            res[f"broken_{l}"] = sum(b and not a for a, b in zip(ok[l], base[l]))
    return res, ok, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    e = cfg["edit"]

    rows = load_jsonl("data/belebele_dev.jsonl")
    tokenizer, model, rev = load_model(cfg["model"]["name"], revision=cfg["model"]["revision"])
    snap = untie_and_snapshot(model)

    base, base_ok, base_preds = evaluate(model, tokenizer, rows)
    print("baseline", base)

    # Check on the edit machinery (compare weights)
    def weights_equal():
        return all(torch.equal(p.detach().cpu(), snap[n])
                   for n, p in edit_targets(model))

    rand_d = torch.randn(model.config.hidden_size)
    project_out(model, rand_d, alpha=0.0)
    assert weights_equal(), "alpha=0 changed weights: bug in project_out"
    project_out(model, rand_d, alpha=1.0)
    assert not weights_equal(), "alpha=1 changed nothing: edit not applied"
    restore(model, snap)
    assert weights_equal(), "restore did not recover the original weights"
    print("Weight checks passed (alpha=0 no-op, alpha=1 edits, restore exact)")

    # Same model, same inputs, only the batching differs
    alt = {l: [p["pred"] for p in predict(model, tokenizer, rows, l, batch_size=7)]
           for l in ["sv", "en"]}
    for l in ["sv", "en"]:
        flips = sum(a != b for a, b in zip(alt[l], base_preds[l]))
        print(f"Noise floor ({l}): {flips}/{len(rows)} predictions change "
              f"with batch size 7 vs 4")

    # Direction from dev passages only (parallel and unique)
    seen, sv_t, en_t = set(), [], []
    for r in rows:
        if r["passage_id"] not in seen:
            seen.add(r["passage_id"])
            sv_t.append(r["sv"]["passage"])
            en_t.append(r["en"]["passage"])
    print(f"Direction from {len(sv_t)} parallel dev passages")
    S = text_means(model, tokenizer, sv_t, e["source_layers"])
    E = text_means(model, tokenizer, en_t, e["source_layers"])
    D = language_direction(S, E)
    torch.save({"S": S, "E": E, "layers": e["source_layers"]}, "results/dev_means.pt")

    table = []
    for li, L in enumerate(e["source_layers"]):
        for a in e["alphas"]:
            restore(model, snap)
            project_out(model, D[li], alpha=a)
            res, _, _ = evaluate(model, tokenizer, rows, base=base_ok)
            res.update({"layer": L, "alpha": a})
            table.append(res)
            print(res)
    restore(model, snap)

    ok_cfgs = [t for t in table if t["acc_en"] >= base["acc_en"] - e["max_en_drop"]]
    chosen = max(ok_cfgs, key=lambda t: t["acc_sv"]) if ok_cfgs else None
    print("\nCHOSEN (dev rule):", chosen)

    Path("results").mkdir(exist_ok=True)
    with open("results/dev_sweep.json", "w") as f:
        json.dump({"revision": rev, "baseline": base, "table": table,
                   "chosen": chosen}, f, indent=2)


if __name__ == "__main__":
    main()