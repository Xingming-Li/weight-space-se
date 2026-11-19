import json

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# Same structure for both languages. Only the content differs.
def build_prompt(item):
    opts = "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, item["options"]))
    return (f"Passage: {item['passage']}\n\nQuestion: {item['question']}\n\n"
            f"{opts}\n\nAnswer:")


def build_fewshot_prompt(item, shots):
    parts = [build_prompt(s) + " " + LETTERS[a] for s, a in shots]
    return "\n\n".join(parts + [build_prompt(item)])


def load_model(name, device=device, revision=None):
    rev = revision or model_info(name).sha
    tokenizer = AutoTokenizer.from_pretrained(name, revision=rev)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=rev, dtype=torch.bfloat16).to(device).eval()
    return tokenizer, model, rev


def letter_ids(tokenizer):
    ids = []
    for l in LETTERS:
        t = tokenizer.encode(" " + l, add_special_tokens=False)
        assert len(t) == 1, f"' {l}' is not a single token: {t}"
        ids.append(t[0])
    return ids


@torch.no_grad()
def predict(model, tokenizer, rows, lang, batch_size=4, device=device,
            prompt_fn=build_prompt):
    ids = letter_ids(tokenizer)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][lang]["passage"]))
    out = [None] * len(rows)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        prompts = [prompt_fn(rows[i][lang]) for i in idx]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        lp = logits[:, ids].float().log_softmax(-1).cpu()
        for j, i in enumerate(idx):
            out[i] = {"pred": int(lp[j].argmax()), "logprobs": lp[j].tolist()}
    return out