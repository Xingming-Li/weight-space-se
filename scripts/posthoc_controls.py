# POST-HOC EXPLORATORY CONTROLS. Written and run AFTER the preregistered test
# results (results/test_runs.jsonl) were known. Nothing here changes those results.
# python -m scripts.posthoc_controls --config configs/main.yaml
#
# What it adds:
#  1. Reproduction check: baseline and both real edits re-evaluated on test with the
#     current code (must match results/test_runs.jsonl item by item).
#  2. Batch-size noise floor on test (bs=2, as used by the few-shot arm, vs bs=4).
#  3. Matched random directions: unit directions orthogonal to the language direction d,
#     accepted only if BOTH
#       (a) activation second moment E[(h.u)^2] at the source layer (dev tokens), and
#       (b) weight-update size sum_m ||u^T W_m||^2 over all edited matrices
#     are within +-TOL of the real direction. Same draws used for alpha 0.5 and 1.0.
#  4. Language-modelling side effects on held-out TEST passages (sv and en):
#     perplexity, and KL(base || edited) of next-token distributions (truncated to the
#     base model's top-K tokens; covered probability mass is reported), for baseline,
#     real, original random, original null and matched edits.
# Output: results/posthoc_runs.jsonl (append, resumable), results/posthoc_directions.pt,
#         results/posthoc_env.json

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import transformers

from src.belebele import load_jsonl, load_model, predict
from src.edit import edit_targets, language_direction, null_direction, project_out, restore, untie_and_snapshot
from src.utils import load_config, set_seed

OUT = Path("results/posthoc_runs.jsonl")
DIRS = Path("results/posthoc_directions.pt")
TOL = 0.10
N_MATCHED = 10
MATCH_SEED = 20261119
TOPK = 64


def done_keys():
    keys = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            keys.add((r["arm"], r.get("alpha"), r.get("draw")))
    return keys


@torch.no_grad()
def second_moment(model, tok, texts, layer, batch_size=8, max_len=512):
    """Uncentered second-moment matrix and mean of token activations (same masking as text_means)."""
    H = model.config.hidden_size
    M = torch.zeros(H, H, dtype=torch.float64, device=model.device)
    s1 = torch.zeros(H, dtype=torch.float64, device=model.device)
    n = 0
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        h = model.model(**enc, output_hidden_states=True).hidden_states[layer]
        attn = enc.attention_mask.bool()
        first = (enc.attention_mask.cumsum(1) == 1) & attn
        x = h[attn & ~first].double()
        M += x.T @ x
        s1 += x.sum(0)
        n += x.shape[0]
    return (M / n).float().cpu(), (s1 / n).float().cpu(), n


@torch.no_grad()
def weight_gram(model, chunk=16384):
    """G with u^T G u = sum over edited matrices of ||Delta W||_F^2 / alpha^2 for unit u."""
    H = model.config.hidden_size
    G = torch.zeros(H, H, dtype=torch.float32, device=model.device)
    for name, p in edit_targets(model):
        if name == "embed":  # Delta E = alpha (E u) u^T  ->  ||E u||^2
            for s in range(0, p.shape[0], chunk):
                b = p[s:s + chunk].float()
                G += b.T @ b
        else:  # Delta W = alpha u (u^T W)  ->  ||u^T W||^2
            W = p.float()
            G += W @ W.T
    return G.cpu()


def sample_matched(d, M, G, mu, n_want, seed, tol=TOL, batch=4096, max_batches=500):
    d = d / d.norm()
    tgt_a, tgt_w = float(d @ M @ d), float(d @ G @ d)
    gen = torch.Generator().manual_seed(seed)
    C = M - torch.outer(mu, mu)
    shapes = []
    for A in (C, G, torch.eye(len(d))):  # activation-, weight- and isotropically shaped
        ev, V = torch.linalg.eigh(A.double())
        shapes.append((V * ev.clamp(min=0).sqrt()).float())
    out, tried = [], 0
    for _ in range(max_batches):
        z = [S @ torch.randn(len(d), batch, generator=gen) for S in shapes]
        z = [x / x.norm(dim=0, keepdim=True) for x in z]
        w = torch.rand(3, batch, generator=gen) ** 3  # skewed mixing weights
        u = sum(wi * zi for wi, zi in zip(w, z))
        u = u - torch.outer(d, d @ u)  # orthogonal to the language direction
        u = u / u.norm(dim=0, keepdim=True)
        a = ((M @ u) * u).sum(0)
        g = ((G @ u) * u).sum(0)
        ok = ((a / tgt_a - 1).abs() < tol) & ((g / tgt_w - 1).abs() < tol)
        tried += batch
        for j in ok.nonzero().flatten().tolist():
            out.append(u[:, j].clone())
            if len(out) == n_want:
                return torch.stack(out), {"target_act_m2": tgt_a, "target_g": tgt_w,
                                          "candidates_tried": tried, "tol": tol}
    raise RuntimeError(f"only {len(out)} matched directions after {tried} candidates")


@torch.no_grad()
def lm_eval(model, tok, texts, base_top=None, batch_size=2, max_len=512, chunk=256):
    """Corpus perplexity; and if base_top given, truncated KL(base||model) per token."""
    nll_sum, n_tok, kl_sum, cover_sum = 0.0, 0, 0.0, 0.0
    tops = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        hid = model.model(**enc).last_hidden_state
        for b in range(hid.shape[0]):
            m = enc.attention_mask[b].bool()
            h, ids = hid[b][m][:-1], enc.input_ids[b][m][1:]  # predict token t+1 from t
            for c in range(0, len(ids), chunk):
                lp = model.lm_head(h[c:c + chunk]).float().log_softmax(-1)
                nll_sum -= lp.gather(1, ids[c:c + chunk, None]).sum().item()
                n_tok += len(ids[c:c + chunk])
                if base_top is None:
                    v, i = lp.topk(TOPK, -1)
                    tops.append((v.half().cpu(), i.int().cpu()))
                else:
                    bv, bi = base_top[len(tops)]
                    tops.append(None)
                    bv, bi = bv.float().to(lp.device), bi.long().to(lp.device)
                    pb = bv.exp()
                    kl_sum += (pb * (bv - lp.gather(1, bi))).sum().item()
                    cover_sum += pb.sum().item()
    res = {"ppl": float(torch.tensor(nll_sum / n_tok).exp()), "n_tokens": n_tok}
    if base_top is not None:
        res["kl"] = kl_sum / n_tok
        res["topk_mass"] = cover_sum / n_tok
    return res, (tops if base_top is None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    e = cfg["edit"]
    test = load_jsonl("data/belebele_test.jsonl")
    dev = load_jsonl("data/belebele_dev.jsonl")
    done = done_keys()

    tok, model, rev = load_model(cfg["model"]["name"], revision=cfg["model"]["revision"])
    snap = untie_and_snapshot(model)
    means = torch.load("results/dev_means.pt")
    S, E, layers = means["S"], means["E"], means["layers"]
    D = language_direction(S, E)
    L = e["final"][0]["layer"]
    assert all(c["layer"] == L for c in e["final"])
    li = layers.index(L)
    alphas = [c["alpha"] for c in e["final"]]
    d = D[li] / D[li].norm()

    Path("results/posthoc_env.json").write_text(json.dumps({
        "model": cfg["model"]["name"], "revision": rev, "seed": cfg["seed"],
        "match_seed": MATCH_SEED, "tol": TOL, "n_matched": N_MATCHED, "topk_kl": TOPK,
        "layer": L, "alphas": alphas, "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, indent=2))

    # Statistics of the unedited model: dev passages only (direction-selection data)
    def uniq(rows, lang):
        seen, out = set(), []
        for r in rows:
            if r["passage_id"] not in seen:
                seen.add(r["passage_id"])
                out.append(r[lang]["passage"])
        return out

    if DIRS.exists():
        saved = torch.load(DIRS)
        U, info, Msv, Men, G = saved["U"], saved["info"], saved["M_sv"], saved["M_en"], saved["G"]
        print("Loaded matched directions from", DIRS)
    else:
        Msv, mu_sv, n_sv = second_moment(model, tok, uniq(dev, "sv"), L)
        Men, mu_en, n_en = second_moment(model, tok, uniq(dev, "en"), L)
        G = weight_gram(model)
        M, mu = (Msv + Men) / 2, (mu_sv + mu_en) / 2
        U, info = sample_matched(d, M, G, mu, N_MATCHED, MATCH_SEED)
        info.update({"n_tokens_sv": n_sv, "n_tokens_en": n_en})
        torch.save({"U": U, "info": info, "M_sv": Msv, "M_en": Men, "G": G, "d": d}, DIRS)
    print("matching:", info)

    def geom(u):
        u = u / u.norm()
        m_sv, m_en = float(u @ Msv @ u), float(u @ Men @ u)
        return {"act_m2": (m_sv + m_en) / 2, "act_m2_sv": m_sv, "act_m2_en": m_en,
                "g": float(u @ G @ u), "cos_d": float(u @ d)}

    test_sv, test_en = uniq(test, "sv"), uniq(test, "en")
    t0 = time.time()
    restore(model, snap)
    base_lm = {}
    base_top = {}
    for lang, texts in [("sv", test_sv), ("en", test_en)]:
        base_lm[lang], base_top[lang] = lm_eval(model, tok, texts)
    print(f"base LM {base_lm} [{time.time()-t0:.0f}s]", flush=True)

    def record(arm, alpha=None, draw=None, direction=None, belebele=False, bs=4):
        if (arm, alpha, draw) in done:
            return
        restore(model, snap)
        rec = {"arm": arm, "alpha": alpha, "draw": draw, "layer": L if alpha is not None else None}
        if direction is not None:
            project_out(model, direction, alpha=alpha)
            rec.update(geom(direction))
            rec["dw_norm"] = alpha * rec["g"] ** 0.5
        if belebele:
            for lang in ["sv", "en"]:
                preds = predict(model, tok, test, lang, batch_size=bs)
                rec[f"ok_{lang}"] = [int(p["pred"] == r["answer"]) for p, r in zip(preds, test)]
        if bs == 4:
            for lang, texts in [("sv", test_sv), ("en", test_en)]:
                if direction is None:
                    r = {"ppl": base_lm[lang]["ppl"], "kl": 0.0, "topk_mass": None}
                else:
                    r, _ = lm_eval(model, tok, texts, base_top=base_top[lang])
                rec[f"ppl_{lang}"], rec[f"kl_{lang}"] = r["ppl"], r["kl"]
                rec[f"topk_mass_{lang}"] = r["topk_mass"]
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        acc = {l: round(sum(rec[f"ok_{l}"]) / len(test), 4) for l in ["sv", "en"] if f"ok_{l}" in rec}
        lm = {k: round(v, 4) for k, v in rec.items() if k[:3] in ("ppl", "kl_") and v is not None}
        print(f"{arm} a={alpha} draw={draw} {acc} {lm} [{time.time()-t0:.0f}s]", flush=True)

    record("baseline_rerun", belebele=True)
    record("baseline_bs2", belebele=True, bs=2)
    base_seed = cfg["seed"] * 1000  # same seeds as run_test.py
    for a in alphas:
        record("real", a, direction=D[li], belebele=True)
        for j in range(N_MATCHED):
            record("matched", a, j, direction=U[j], belebele=True)
    for a in alphas:
        for r in range(e["n_random"]):
            g = torch.Generator().manual_seed(base_seed + r)
            record("random", a, r, direction=torch.randn(model.config.hidden_size, generator=g))
        for r in range(e["n_null"]):
            g = torch.Generator().manual_seed(base_seed + 500 + r)
            record("null", a, r, direction=null_direction(S, E, g)[li])

    restore(model, snap)
    assert all(torch.equal(p.detach().cpu(), snap[n]) for n, p in edit_targets(model))
    print("Done. Weights restored exactly.")


if __name__ == "__main__":
    main()
