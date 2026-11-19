# Data

Parallel Swedish/English multiple-choice reading comprehension from Belebele (`facebook/belebele`, configs `swe_Latn` and `eng_Latn`, split `test`), licensed CC BY-SA 4.0. Passages come from FLORES-200. Dataset revision: see `data/belebele_meta.json`.

`data/belebele_{dev,test}.jsonl`: one row per question, both languages aligned by
(passage link, question number). Split by passage (no passage in both splits),
seed in `configs/main.yaml`. Rows whose answer index differs between languages
are dropped (count in the meta file).