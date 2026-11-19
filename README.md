# Weight-Space Editing for Swedish

An exploration of whether removing a Swedish–English activation direction from model weights improves Swedish reading comprehension, inspired by [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/).

**[Research Note](research_note.md) | [PDF Report](report.pdf) | [AI Usage](AI_USAGE.md)**

## Results

`Qwen3-0.6B-Base` is evaluated on 543 Swedish–English [*Belebele*](https://ai.meta.com/research/publications/the-belebele-benchmark-a-parallel-reading-comprehension-dataset-in-122-language-variants/) questions.

| Method | Swedish | English |
|---|---:|---:|
| Baseline | 53.2% | 66.5% |
| Targeted edit | 53.0% | 64.6% |
| Full removal | 44.4% | 64.8% |
| 3-shot prompting | 57.6% | 71.3% |

**Weight editing do not improve Swedish comprehension.** Post hoc analysis finds that the targeted edit increases Swedish perplexity from 28 to 193. See the report or research note for methodology, experiments, analysis, and limitations.

## Setup

Tested with an NVIDIA RTX 3050 Ti (4 GB) GPU.

```bash
conda create -n <venv_name> python=3.11 -y
conda activate <venv_name>
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124  # Drop CUDA wheel if GPU unaccessable
pip install -r requirements.txt
```

## Reproduce

Run from the repository root. Existing files in `results/` will be overwritten.

```bash
# Prepare data
python -m scripts.prepare_belebele --config configs/main.yaml

# Search for plausible config on dev
python -m scripts.dev_sweep --config configs/main.yaml

# Run evaluation
python -m scripts.run_test --config configs/main.yaml

# Compute statistics from saved results
python -m scripts.analyze

# Post hoc analysis
python -m scripts.posthoc_controls --config configs/main.yaml
python -m scripts.posthoc_generate --config configs/main.yaml
```

The repository collects the data, results, and experimental scripts. Configurations including model and dataset revisions are pinned in `configs/main.yaml`. The original test configuration was fixed before evaluation. Post hoc analysis is labelled separately.

## Licence

Code: MIT; Data: CC BY-SA 4.0; Qwen3 model: Apache-2.0