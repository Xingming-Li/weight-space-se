# Build masked-word sets for Swedish and English from Wikipedia.
# python -m src.data --config configs/main.yaml

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from src.utils import load_config, set_seed

WIKI_CONFIG = {"sv": "20231101.sv", "en": "20231101.en"}

# For checking if a sentence is really in the expected language.
STOPWORDS = {
    "sv": {"och", "att", "det", "som", "är", "av", "för", "med", "på",
           "till", "den", "har", "inte", "var", "om", "ett", "en"},
    "en": {"the", "of", "and", "to", "in", "is", "was", "for", "that",
           "with", "as", "by", "on", "are", "from", "at"},
}

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
BAD_CHARS = re.compile(r"[|{}<>=\[\]_#*/\\]")
UNIT_NO_NUM = re.compile(r"(°C|\bmm\b|\bcm\b|\bkm\b|\bkg\b|%)")


def sentence_ok(sent, lang, min_words, max_words):
    words = sent.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if not sent[0].isupper() or not sent.endswith("."):
        return False
    if BAD_CHARS.search(sent):
        return False
    if UNIT_NO_NUM.search(sent):
        return False
    if sum(c.isdigit() for c in sent) > 0.1 * len(sent):
        return False
    lowered = {w.strip(".,;:!?()\"'").lower() for w in words}
    return len(lowered & STOPWORDS[lang]) >= 2


# Pick one maskable word index.
def pick_target(words, tokenizer, rng, lang):
    candidates = []
    for i, w in enumerate(words):
        # Eliminate: functional / capitalised / non-alphabetic / short
        if w.lower() in STOPWORDS[lang] or w[0].isupper() or not w.isalpha() or len(w) < 4:
            continue
        token_id = tokenizer.convert_tokens_to_ids("\u2581" + w)
        if token_id != tokenizer.unk_token_id:
            candidates.append((i, token_id))
    return rng.choice(candidates) if candidates else None


def build_language(lang, cfg, tokenizer, seed):
    d = cfg["data"]
    rng = random.Random(seed)
    ds = load_dataset("wikimedia/wikipedia", WIKI_CONFIG[lang],
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    articles, total = [], 0
    prefix_counts = Counter()
    for example in ds:
        sents = []
        for para in example["text"].split("\n"):
            sents += [s.strip() for s in SENT_SPLIT.split(para) if s.strip()]
        sents = [s for s in sents
                 if sentence_ok(s, lang, d["min_words"], d["max_words"])]
        rng.shuffle(sents)

        items = []
        for s in sents:
            if len(items) >= d["max_sentences_per_article"]:
                break
            words = s.split()
            pick = pick_target(words, tokenizer, rng, lang)
            if pick is None:
                continue
            key = " ".join(s.lower().split()[:5])
            if prefix_counts[key] >= 5:
                continue
            prefix_counts[key] += 1
            i, token_id = pick
            masked = " ".join(words[:i] + [tokenizer.mask_token] + words[i + 1:])
            items.append({"lang": lang, "article_id": example["id"], "sentence": s,
                          "masked": masked, "target_word": words[i],
                          "target_id": token_id})
        if items:
            articles.append(items)
            total += len(items)
        if total >= d["n_sentences_per_lang"]:
            break

    # Split by article
    rng.shuffle(articles)
    n_dev = round(d["dev_fraction"] * len(articles))
    return {"dev": [x for a in articles[:n_dev] for x in a],
            "test": [x for a in articles[n_dev:] for x in a]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"],
                                              revision=cfg["model"]["revision"])
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    for lang in ["sv", "en"]:
        splits = build_language(lang, cfg, tokenizer, cfg["seed"])
        for name, rows in splits.items():
            path = out_dir / f"{lang}_{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{path}: {len(rows)} sentences, "
                  f"{len({r['article_id'] for r in rows})} articles")
        for r in splits["dev"][:5]:
            print("   ", r["masked"], "->", r["target_word"])


if __name__ == "__main__":
    main()