# Run on the test split. Files are committed before running.
# 

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch

from src.belebele import build_fewshot_prompt, load_jsonl, load_model, predict
from src.edit import edit_targets, language_direction, null_direction, project_out, restore, untie_and_snapshot
from src.utils import load_config, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("results/test_runs.jsonl")


def done_keys():
    keys = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            keys.add((r["arm"], r.get("layer"), r.get("alpha"), r.get("draw")))
    return keys


def run_eval(model, tokenizer, rows, batch_size=4, prompt_fns=None):
    ok = {}
    for lang in ["sv", "en"]:
        kw = {"prompt_fn": prompt_fns[lang]} if prompt_fns else {}
        preds = predict(model, tokenizer, rows, lang, batch_size=batch_size, **kw)
        ok[lang] = [int(p["pred"] == r["answer"]) for p, r in zip(preds, rows)]
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--n_random", type=int, default=None)
    ap.add_argument("--n_null", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    e = cfg["edit"]
    n_random = args.n_random if args.n_random is not None else e["n_random"]
    n_null = args.n_null if args.n_null is not None else e["n_null"]

    test = load_jsonl("data/belebele_test.jsonl")
    dev = load_jsonl("data/belebele_dev.jsonl")
    OUT.parent.mkdir(exist_ok=True)
    done = done_keys()

    tokenizer, model, rev = load_model(cfg["model"]["name"],
                                       revision=cfg["model"]["revision"])
    snap = untie_and_snapshot(model)
    means = torch.load("results/dev_means.pt")
    S, E, layers = means["S"], means["E"], means["layers"]
    D = language_direction(S, E)          # From dev passages only

    with open("results/test_env.json", "w") as f:
        json.dump({"revision": rev, "seed": cfg["seed"], "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0), "n_test": len(test),
                   "n_random": n_random, "n_null": n_null}, f, indent=2)

    t0 = time.time()

    def record(arm, ok, **meta):
        rec = {"arm": arm, "n": len(test), "ok_sv": ok["sv"], "ok_en": ok["en"], **meta}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{arm} {meta} sv={sum(ok['sv'])/len(test):.3f} "
              f"en={sum(ok['en'])/len(test):.3f}  [{time.time()-t0:.0f}s]", flush=True)

    if ("baseline", None, None, None) not in done:
        record("baseline", run_eval(model, tokenizer, test))

    if ("fewshot3", None, None, None) not in done:   # Prompting alternative
        shots = random.Random(cfg["seed"]).sample(dev, 3)
        fns = {l: partial(build_fewshot_prompt,
                          shots=[(r[l], r["answer"]) for r in shots])
               for l in ["sv", "en"]}
        record("fewshot3", run_eval(model, tokenizer, test, batch_size=2, prompt_fns=fns),
               shot_ids=[r["id"] for r in shots],
               shot_answers=[r["answer"] for r in shots])

    base_seed = cfg["seed"] * 1000
    for c in e["final"]:
        L, a = c["layer"], c["alpha"]
        li = layers.index(L)

        def evaluate_with(direction, arm, draw=None):
            restore(model, snap)
            project_out(model, direction, alpha=a)
            record(arm, run_eval(model, tokenizer, test), layer=L, alpha=a, draw=draw,
                   name=c["name"])

        if ("real", L, a, None) not in done:
            evaluate_with(D[li], "real")
        for r in range(n_random):
            if ("random", L, a, r) not in done:
                g = torch.Generator().manual_seed(base_seed + r)
                evaluate_with(torch.randn(model.config.hidden_size, generator=g),
                              "random", r)
        for r in range(n_null):
            if ("null", L, a, r) not in done:
                g = torch.Generator().manual_seed(base_seed + 500 + r)
                evaluate_with(null_direction(S, E, g)[li], "null", r)

    restore(model, snap)
    assert all(torch.equal(p.detach().cpu(), snap[n])
               for n, p in edit_targets(model)), "weights not restored"
    print("Done! Weights restored exactly.")


if __name__ == "__main__":
    main()