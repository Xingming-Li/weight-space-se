# Build masked-word sets for Swedish and English from Wikipedia.
# python -m src.data --config configs/main.yaml

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from src.utils import load_config, set_seed

WIKI_CONFIG = {"sv": "20231101.sv", "en": "20231101.en"}

STOPWORDS = {
    "sv": {"och", "att", "det", "som", "är", "av", "för", "med", "på",
           "till", "den", "har", "inte", "var", "om", "ett", "en"},
    "en": {"the", "of", "and", "to", "in", "is", "was", "for", "that",
           "with", "as", "by", "on", "are", "from", "at"},
}

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
BAD_CHARS = re.compile(r"[|{}<>=\[\]_#*/\\]")
# Sentences whose numbers were stripped by the dataset
UNIT_NO_NUM = re.compile(
    r"(°C|\bmm\b|\bcm\b|\bkm\b|\bkg\b|%|kvadratkilometer|invånare är \.)")
# Massive Brasilian detected after observing dev set
FOREIGN_CHARS = set("ãõçáíóúâêôñ")  # Not used in ordinary Swedish


def sentence_ok(sent, lang, min_words, max_words):
    words = sent.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if not sent[0].isupper() or not sent.endswith("."):
        return False
    if BAD_CHARS.search(sent) or UNIT_NO_NUM.search(sent):
        return False
    if sum(c.isdigit() for c in sent) > 0.1 * len(sent):
        return False
    lowered = {w.strip(".,;:!?()\"'").lower() for w in words}
    return len(lowered & STOPWORDS[lang]) >= 2


def pick_target(words, tokenizer, rng, lang):
    cands = []
    for i, w in enumerate(words):
        if w.lower() in STOPWORDS[lang] or not w.isalpha() or w[0].isupper() or len(w) < 4:
            continue
        token_id = tokenizer.convert_tokens_to_ids("\u2581" + w)
        if token_id != tokenizer.unk_token_id:
            cands.append((i, token_id))
    return rng.choice(cands) if cands else None


def context_key(masked):
    w = masked.split()
    i = w.index("<mask>")
    return " ".join(w[max(0, i - 2):i] + ["<mask>"] + w[i + 1:i + 3]).lower()


def has_foreign_chars(sent):
    return any(c in FOREIGN_CHARS for c in sent.lower())


def assign_and_cap(pool, d, rng):
    counts = Counter(context_key(r["masked"]) for r in pool)
    by_key = defaultdict(list)
    for r in pool:
        k = context_key(r["masked"])
        r["context_key"] = k
        if counts[k] > d["template_min_count"]:
            r["stratum"] = "template"
        elif has_foreign_chars(r["sentence"]):
            r["stratum"] = "foreign_natural"
        else:
            r["stratum"] = "natural"
        by_key[k].append(r)
    kept = []
    for k in sorted(by_key):
        items = by_key[k]
        rng.shuffle(items)
        cap = d["template_cap"] if items[0]["stratum"] == "template" \
            else d["natural_key_cap"]
        kept.extend(items[:cap])
    return kept


def collect_pool(lang, cfg, tokenizer, seed):
    d = cfg["data"]
    rng = random.Random(seed)
    ds = load_dataset("wikimedia/wikipedia", WIKI_CONFIG[lang],
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    pool, n_articles = [], 0
    for ex in ds:
        n_articles += 1
        sents = []
        for para in ex["text"].split("\n"):
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
            i, token_id = pick
            masked = " ".join(words[:i] + [tokenizer.mask_token] + words[i + 1:])
            items.append({"lang": lang, "article_id": ex["id"], "sentence": s,
                          "masked": masked, "target_word": words[i],
                          "target_id": token_id})
        pool.extend(items)

        if n_articles % 2000 == 0:
            kept = assign_and_cap(list(pool), d, random.Random(seed))
            n_free = sum(r["stratum"] != "template" for r in kept)
            print(f"[{lang}] {n_articles} articles, {n_free} non-template items")
            if n_free >= d["n_sentences_per_lang"]:
                break
        if n_articles >= d["max_articles"]:
            print(f"[{lang}] stopped at max_articles={n_articles}")
            break
    return pool


def build_language(lang, cfg, tokenizer, seed):
    d = cfg["data"]
    pool = collect_pool(lang, cfg, tokenizer, seed)
    kept = assign_and_cap(pool, d, random.Random(seed + 1))

    by_art = defaultdict(list)
    for r in kept:
        by_art[r["article_id"]].append(r)
    arts = sorted(by_art)
    random.Random(seed + 2).shuffle(arts)
    n_dev = round(d["dev_fraction"] * len(arts))
    return {"dev": [x for a in arts[:n_dev] for x in by_art[a]],
            "test": [x for a in arts[n_dev:] for x in by_art[a]]}


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
    stats = {}

    for lang in ["sv", "en"]:
        splits = build_language(lang, cfg, tokenizer, cfg["seed"])
        for name, rows in splits.items():
            path = out_dir / f"{lang}_{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            strata = Counter(r["stratum"] for r in rows)
            keys = len({r["context_key"] for r in rows})
            stats[f"{lang}_{name}"] = {"n": len(rows), "context_keys": keys,
                                       "strata": dict(strata)}
            print(f"{path}: {len(rows)} sentences, {keys} context keys, "
                  f"{dict(strata)}")
        for r in splits["dev"][:5]:
            print("   ", r["stratum"], "|", r["masked"], "->", r["target_word"])

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()