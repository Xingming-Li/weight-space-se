# Data

Masked-word sets built by `src/data.py` (seed in `configs/main.yaml`).

- Source: `wikimedia/wikipedia`, configs `20231101.sv` and `20231101.en`
  (Wikipedia text, licensed CC BY-SA 4.0. Attribution to Wikipedia contributors).
- One word masked per sentence. Only whole words that XLM-R encodes as one token.
- Split dev/test by article. Files: `data/{sv,en}_{dev,test}.jsonl`.
- Filters: length 8-40 words, stopword
  language check, no markup, no units without numbers, max 5 sentences per
  5-word opening, functional / capitalised / non-alphabetic / short words are never masked.
- Known issue: some bot-generated sentences still have numbers removed
  (e.g. "med invånare per kvadratkilometer").