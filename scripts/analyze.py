# Deterministic analysis of saved per-question results. No model, no GPU.
# python -m scripts.analyze
#
# Inputs (read-only):
#   data/belebele_test.jsonl         gold answers (order = order of ok_* vectors)
#   results/test_runs.jsonl          preregistered test arms (run_test.py)
#   results/dev_sweep.json           dev selection table (aggregates only)
#   results/posthoc_runs.jsonl       post-hoc exploratory arms (posthoc_controls.py), optional
# Outputs:
#   results/test_summary.json, results/test_summary.md

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

N_BOOT = 10000
BOOT_SEED = 0


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def mcnemar(new, base):
    fixed = int(sum(a and not b for a, b in zip(new, base)))
    broken = int(sum(b and not a for a, b in zip(new, base)))
    p = binomtest(fixed, fixed + broken).pvalue if fixed + broken else 1.0
    return fixed, broken, float(p)


def paired_ci(new, base, rng):
    d = np.asarray(new, float) - np.asarray(base, float)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compare(arm, base, latent):
    out = {}
    for lang in ["sv", "en"]:
        new, ref = arm[f"ok_{lang}"], base[f"ok_{lang}"]
        rng = np.random.default_rng(BOOT_SEED)  # same resamples for every comparison
        diff, lo, hi = paired_ci(new, ref, rng)
        fixed, broken, p = mcnemar(new, ref)
        out[lang] = {"acc": float(np.mean(new)), "diff_vs_base": diff, "ci95": [lo, hi],
                     "fixed": fixed, "broken": broken, "mcnemar_p": p}
    out["latent_fixed"] = int(sum(arm["ok_sv"][i] for i in latent))
    out["gap_en_minus_sv"] = out["en"]["acc"] - out["sv"]["acc"]
    return out


def draws_summary(draws, real):
    s = {}
    for lang in ["sv", "en"]:
        accs = np.array([np.mean(d[f"ok_{lang}"]) for d in draws])
        r = float(np.mean(real[f"ok_{lang}"]))
        s[lang] = {"n_draws": len(accs), "mean": float(accs.mean()), "sd": float(accs.std(ddof=1)),
                   "min": float(accs.min()), "max": float(accs.max()), "real": r,
                   "n_draws_ge_real": int((accs >= r).sum()),
                   "z_real": float((r - accs.mean()) / accs.std(ddof=1))}
    return s


def pct(x):
    return f"{100 * x:.1f}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    test = load_jsonl("data/belebele_test.jsonl")
    answers = [r["answer"] for r in test]
    runs = load_jsonl("results/test_runs.jsonl")
    for r in runs:
        assert r["n"] == len(test) == len(r["ok_sv"]) == len(r["ok_en"])

    base = next(r for r in runs if r["arm"] == "baseline")
    latent = [i for i in range(len(test)) if base["ok_en"][i] and not base["ok_sv"][i]]
    summary = {"n_test": len(test), "n_latent_en_right_sv_wrong": len(latent),
               "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
               "answer_distribution": [answers.count(k) for k in range(4)],
               "preregistered": {}, "posthoc": {}}

    pre = summary["preregistered"]
    pre["baseline"] = {"acc_sv": float(np.mean(base["ok_sv"])), "acc_en": float(np.mean(base["ok_en"]))}
    pre["fewshot3"] = compare(next(r for r in runs if r["arm"] == "fewshot3"), base, latent)
    for name in ["primary", "secondary_ablation"]:
        real = next(r for r in runs if r["arm"] == "real" and r.get("name") == name)
        cfg = {"layer": real["layer"], "alpha": real["alpha"], "real": compare(real, base, latent)}
        for arm in ["random", "null"]:
            ds = [r for r in runs if r["arm"] == arm and r.get("name") == name]
            cfg[arm] = draws_summary(ds, real)
        pre[name] = cfg

    dev = json.load(open("results/dev_sweep.json"))
    best = max(t["acc_sv"] for t in dev["table"])
    summary["dev_selection"] = {
        "baseline": dev["baseline"], "chosen": dev["chosen"],
        "tied_on_acc_sv": [{"layer": t["layer"], "alpha": t["alpha"], "acc_sv": t["acc_sv"],
                            "acc_en": t["acc_en"]} for t in dev["table"] if t["acc_sv"] == best],
        "n_configs": len(dev["table"])}

    ph_path = Path("results/posthoc_runs.jsonl")
    if ph_path.exists():
        ph = load_jsonl(ph_path)
        bm = next((r for r in ph if r["arm"] == "baseline_rerun"), None)
        post = summary["posthoc"]
        if bm is not None:
            post["baseline_rerun_identical"] = {
                l: bm[f"ok_{l}"] == base[f"ok_{l}"] for l in ["sv", "en"]}
        b2 = next((r for r in ph if r["arm"] == "baseline_bs2"), None)
        if b2 is not None:
            post["batch_size_2_vs_4_flips"] = {
                l: int(sum(a != b for a, b in zip(b2[f"ok_{l}"], base[f"ok_{l}"]))) for l in ["sv", "en"]}
        # LM metrics: every arm that has them
        lm = {}
        for r in ph:
            if "ppl_sv" in r:
                key = r["arm"] + ("" if r.get("alpha") is None else f"@a{r['alpha']}") + \
                      ("" if r.get("draw") is None else f"#{r['draw']}")
                lm[key] = {k: r[k] for k in ["ppl_sv", "ppl_en", "kl_sv", "kl_en",
                                               "act_m2", "act_m2_sv", "act_m2_en", "dw_norm"]
                           if k in r}
        post["lm_metrics"] = lm
        post["real_rerun_identical"] = {
            str(r["alpha"]): {l: r[f"ok_{l}"] == next(
                t for t in runs if t["arm"] == "real" and t["alpha"] == r["alpha"])[f"ok_{l}"]
                for l in ["sv", "en"]}
            for r in ph if r["arm"] == "real" and "ok_sv" in r}
        # Grouped LM metrics (geometric mean for perplexity: heavy-tailed)
        groups = {}
        for r in ph:
            if "ppl_sv" in r and r["arm"] != "baseline_rerun":
                groups.setdefault(f"{r['arm']}@a{r['alpha']}", []).append(r)
        post["lm_groups"] = {}
        for k, rs in groups.items():
            g = {"n": len(rs)}
            for m in ["ppl_sv", "ppl_en"]:
                v = np.log([r[m] for r in rs])
                g[m + "_geomean"], g[m + "_min"], g[m + "_max"] = (
                    float(np.exp(v.mean())), float(np.exp(v.min())), float(np.exp(v.max())))
            for m in ["kl_sv", "kl_en", "act_m2", "dw_norm"]:
                g[m + "_mean"] = float(np.mean([r[m] for r in rs]))
            post["lm_groups"][k] = g
        # Matched controls: Belebele accuracy vs the real edit at the same alpha
        post["matched"] = {}
        for a in sorted({r["alpha"] for r in ph if r["arm"] == "matched"}):
            real = next(r for r in runs if r["arm"] == "real" and r["alpha"] == a)
            ds = [r for r in ph if r["arm"] == "matched" and r["alpha"] == a]
            post["matched"][str(a)] = draws_summary(ds, real)

    Path("results/test_summary.json").write_text(json.dumps(summary, indent=2))
    Path("results/test_summary.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))


def render(s):
    pre = s["preregistered"]
    L = ["# Test summary (generated by scripts/analyze.py)", "",
         f"n_test = {s['n_test']}; questions right in English but wrong in Swedish at baseline "
         f"(latent) = {s['n_latent_en_right_sv_wrong']}. 95% CIs: paired bootstrap "
         f"({s['bootstrap']['n']} resamples, seed {s['bootstrap']['seed']}); p: exact McNemar.", "",
         "## Preregistered arms (fixed in configs/main.yaml before the test run)", "",
         "| Arm | sv acc | Δsv vs base [95% CI] | sv fixed/broken, p | en acc | Δen vs base [95% CI] | en p | latent fixed |",
         "|---|---|---|---|---|---|---|---|",
         f"| baseline | {pct(pre['baseline']['acc_sv'])} | – | – | {pct(pre['baseline']['acc_en'])} | – | – | 0 |"]
    rows = [("3-shot prompting", pre["fewshot3"])] + \
           [(f"edit {n} (L{pre[n]['layer']}, α={pre[n]['alpha']})", pre[n]["real"])
            for n in ["primary", "secondary_ablation"]]
    for name, c in rows:
        sv, en = c["sv"], c["en"]
        L.append(f"| {name} | {pct(sv['acc'])} | {100*sv['diff_vs_base']:+.1f} "
                 f"[{100*sv['ci95'][0]:+.1f}, {100*sv['ci95'][1]:+.1f}] | {sv['fixed']}/{sv['broken']}, "
                 f"{sv['mcnemar_p']:.3g} | {pct(en['acc'])} | {100*en['diff_vs_base']:+.1f} "
                 f"[{100*en['ci95'][0]:+.1f}, {100*en['ci95'][1]:+.1f}] | {en['mcnemar_p']:.3g} | "
                 f"{c['latent_fixed']} |")
    L += ["", "## Real edit vs control draws (accuracy, %)", "",
          "| Config | Control | sv mean ± sd [min, max] | real sv | draws ≥ real (sv) | en mean ± sd | real en |",
          "|---|---|---|---|---|---|---|"]
    ctrl = [(n, k, pre[n][k], "preregistered") for n in ["primary", "secondary_ablation"]
            for k in ["random", "null"]]
    for a, d in s.get("posthoc", {}).get("matched", {}).items():
        ctrl.append((f"α={a}", "matched (post-hoc)", d, "posthoc"))
    for n, k, d, _ in ctrl:
        sv, en = d["sv"], d["en"]
        L.append(f"| {n} | {k} | {pct(sv['mean'])} ± {pct(sv['sd'])} [{pct(sv['min'])}, {pct(sv['max'])}] "
                 f"| {pct(sv['real'])} | {sv['n_draws_ge_real']}/{sv['n_draws']} | "
                 f"{pct(en['mean'])} ± {pct(en['sd'])} | {pct(en['real'])} |")
    ds = s["dev_selection"]
    L += ["", "## Dev selection", "",
          f"{ds['n_configs']} configs; dev baseline sv {pct(ds['baseline']['acc_sv'])}, "
          f"en {pct(ds['baseline']['acc_en'])}. Chosen: L{ds['chosen']['layer']} α={ds['chosen']['alpha']} "
          f"(sv {pct(ds['chosen']['acc_sv'])}). Tied on sv accuracy: " +
          ", ".join(f"L{t['layer']} α={t['alpha']} (en {pct(t['acc_en'])})" for t in ds["tied_on_acc_sv"]) +
          ". The tie was broken by sweep order (first max), not by a stated rule."]
    post = s.get("posthoc", {})
    if post:
        L += ["", "## Post-hoc exploratory checks (not preregistered)", ""]
        if "baseline_rerun_identical" in post:
            L.append(f"- Baseline re-run with current code reproduces the saved per-question test "
                     f"results exactly: {post['baseline_rerun_identical']}")
        if "batch_size_2_vs_4_flips" in post:
            L.append(f"- Baseline correctness flips, batch size 2 vs 4: {post['batch_size_2_vs_4_flips']}")
        if "real_rerun_identical" in post:
            L.append(f"- Real edits re-run with current code reproduce the saved per-question test "
                     f"results exactly: {post['real_rerun_identical']}")
        if post.get("lm_groups"):
            base_lm = post["lm_metrics"].get("baseline_rerun", {})
            L += ["", "### Language modelling on held-out test passages (grouped)", "",
                  f"Baseline perplexity: sv {base_lm.get('ppl_sv', float('nan')):.2f}, "
                  f"en {base_lm.get('ppl_en', float('nan')):.2f}. Perplexity: geometric mean [min, max] "
                  "over draws; KL(base||edit) per token, truncated to base top-64; "
                  "act. 2nd moment = E[(h·u)²] at layer 12 on dev tokens; ‖ΔW‖_F = total weight-update norm.", "",
                  "| Group | n | ppl sv | ppl en | KL sv | KL en | act. 2nd moment | ‖ΔW‖_F |",
                  "|---|---|---|---|---|---|---|---|"]
            for k, g in post["lm_groups"].items():
                L.append(f"| {k} | {g['n']} | {g['ppl_sv_geomean']:.1f} [{g['ppl_sv_min']:.1f}, {g['ppl_sv_max']:.1f}] "
                         f"| {g['ppl_en_geomean']:.1f} [{g['ppl_en_min']:.1f}, {g['ppl_en_max']:.1f}] "
                         f"| {g['kl_sv_mean']:.3f} | {g['kl_en_mean']:.3f} | {g['act_m2_mean']:.2f} "
                         f"| {g['dw_norm_mean']:.2f} |")
        if post.get("lm_metrics"):
            L += ["", "### Per-arm language-modelling metrics", "", "| Arm | ppl sv | ppl en | KL sv | KL en | act. 2nd moment (L12) | ‖ΔW‖_F |",
                  "|---|---|---|---|---|---|---|"]
            for k, m in post["lm_metrics"].items():
                f = lambda x, fmt: (fmt % m[x]) if x in m else "–"
                L.append(f"| {k} | {f('ppl_sv','%.2f')} | {f('ppl_en','%.2f')} | {f('kl_sv','%.4f')} | "
                         f"{f('kl_en','%.4f')} | {f('act_m2','%.2f')} | {f('dw_norm','%.2f')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
