import json
from collections import defaultdict

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model(cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name, rev = cfg["model"]["name"], cfg["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForMaskedLM.from_pretrained(name, revision=rev)
    return tokenizer, model.eval().to(device), device


# Rank and log-prob of the target token at the mask position.
# The LM head is applied only at the mask (not the whole sequence).
@torch.no_grad()
def score(model, tokenizer, rows, device, batch_size=64, topk=5):
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        enc = tokenizer([r["masked"] for r in batch], return_tensors="pt",
                        padding=True, truncation=True, max_length=256).to(device)
        hidden = model.roberta(input_ids=enc.input_ids,
                               attention_mask=enc.attention_mask).last_hidden_state
        is_mask = enc.input_ids == tokenizer.mask_token_id
        assert (is_mask.sum(1) == 1).all(), "each sentence needs exactly one mask"
        idx = is_mask.float().argmax(1)
        h = hidden[torch.arange(len(batch), device=device), idx]
        lp_all = model.lm_head(h).float().log_softmax(-1)
        for j, r in enumerate(batch):
            lp = lp_all[j]
            target = lp[r["target_id"]]
            rank = int((lp > target).sum().item()) + 1
            top = lp.topk(topk).indices.tolist()
            out.append({
                "lang": r["lang"], "stratum": r["stratum"],
                "context_key": r["context_key"], "article_id": r["article_id"],
                "masked": r["masked"], "target": r["target_word"],
                "rank": rank, "logprob": float(target),
                "top": [t.replace("\u2581", "")
                        for t in tokenizer.convert_ids_to_tokens(top)],
            })
    return out


def summarize(records):
    n = len(records)
    if n == 0:
        return {"n": 0}
    ranks = sorted(r["rank"] for r in records)
    frac = lambda k: sum(r["rank"] <= k for r in records) / n
    return {
        "n": n,
        "top1": frac(1), "top5": frac(5), "top10": frac(10),
        "rank_gt100": sum(r["rank"] > 100 for r in records) / n,
        "median_rank": ranks[n // 2],
        "mean_logprob": sum(r["logprob"] for r in records) / n,
    }


def summarize_by_stratum(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["stratum"]].append(r)
        groups["non_template"].append(r) if r["stratum"] != "template" else None
        groups["all"].append(r)
    return {k: summarize(v) for k, v in groups.items()}