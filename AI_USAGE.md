# AI usage

**Tool:** Claude Code (Anthropic, model Claude Opus 5.5) in VS Code, used as a coding and review assistant.

## What I did myself

- Chose the research question, hypothesis, model, dataset and the main method: removing a difference-in-means language direction from the weights.
- Wrote most of the code for data preparation (`scripts/prepare_belebele.py`), the pilot (`scripts/pilot.py`), the dev sweep (`scripts/dev_sweep.py`), the test run (`scripts/run_test.py`), and `src/`.
- Ran those scripts and committed the code and configuration.
- Explored but abandoned an earlier XLM-R masked-word attempt. It is kept in `archive/`.

## What the AI assisted

- **Debug code.** When I encountered code issues that cannot be solved by myself in a short time, I consulted AI for help.
- **Wrote `scripts/analyze.py`.** It is a deterministic recomputation of every reported metric from the saved per-question results.
- **Wrote and ran `scripts/posthoc_generate.py` and `scripts/posthoc_controls.py`**:
  - random directions matched on activation second moment and weight-update norm;
  - perplexity and KL on held-out passages.
- **Drafted and helped modify documents:** Drafted `README.md`, `research_note.md`, and this file. Also modified the report when I need the help. For example, with the tables.
- **Audited the repository before submission.**

I always reviewed the scope of the assistant's changes. The assistant was told not to change any results. Every number in the note and report comes from `results/test_summary.md`, which `scripts/analyze.py` generates from the saved per-question outputs.
