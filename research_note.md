# Research Note
## 1. The weight-space programming article

To me, the most interesting evidence comes from Exp 3 where only the middle layers were provided. The LLM still identified the language and correctly located the induction head. In Exp 5, an LLM designed a network from scratch without ever seeing data or executing the model, which then outperformed PPO on MinAtar after 100 million training steps. In both cases, the LLM operated without environmental feedback, demonstrating a capacity to reason directly about model weights. However, these were small-scale, easily interpretable networks.

Exp 8, which is more relevant to SE.LLMA, is compelling:

1. First, I focused on the distinction between "reading weights directly" and "optimizing via feedback". As Exp 7 demonstrates, an LLM cannot identify a "sycophantic" model by looking at its weights alone. It requires access to the attention patterns generated during actual execution. In Exp 8, 10 or 25 forward passes didn't yield strong results, with performance only jumping to 48.1% at 50 passes. To me, this is more like a search process driven by feedback.
2. The actual edits performed are essentially subtracting a specific direction from the MLP outputs in layers 15 and 19. This sounds like within the same category as existing techniques like steering or direction ablation.
3. The baselines are weak. The base model is almost constantly stuck in a loop. This can be compared against inference-time interventions, such as limiting the length of the thinking process or enforcing a token budget, which might be more cost-effective.
4. There is no control condition using random directions or a "difference-in-means" approach. So, it is unclear whether the observed gains stem from the LLM understanding the weights, or simply from the fact that *any* perturbation capable of disrupting the repetitive loops would yield an improvement.
5. The negative results about Gemma are insightful: editing can potentially remove malfunctions that obscure existing capabilities. It may not develop entirely new ones.
6. The authors themselves characterize this as a "concept" rather than a concrete method.

## 2. Hypothesis

`Qwen3-0.6B-Base` answers 66.5% of Belebele test questions correctly in English but only 53.2% in Swedish. If part of the gap is an *obstacle*, a weight edit would be a very cheap fix for SE.LLMA.

> **Hypothesis:** Projecting the mean Swedish−English activation difference out of the residual stream lets the model apply its stronger English ability to Swedish. Swedish accuracy rises, and English falls by no more than 2pp.

- **Support:** a Swedish gain larger than random edits of the same size produce, evaluated on Belebele questions.
- **Challenge:** no gain, a gain random edits also produce, or large English damage.

## 3. Method

- **Data, model, scoring.** Belebele sv/en (`7899cdf`), aligned and split by passage: dev 357 questions, test 543, no shared passages. Model revision `da87bfb`, bf16. Qwen3-0.6B-Base is chosen over Qwen3.5-0.8B on dev for its larger gap (11.8 vs 4.2pp). Zero-shot letter log-probability scoring.
- **Edit.** W′ = W − α * d̂ * d̂ᵀ * W, applied to the embeddings and to **every** layer's `o_proj` and `down_proj`. *d* is read at one of Layers {8, 12, 16, 20} from 195 dev passages, but the edit changes all 28 layers. It is directional ablation, with no gradients.
- **Dev selection.** Layers {8, 12, 16, 20} × α {0.5, 1} to maximise Swedish accuracy subject to English falling by at most 2pp. Layer 12/α=0.5 and layer 16/α=1.0 **tied** (57.7% Swedish). The code took the first. The chosen gain was +2.5pp, against a dev standard error of about 2.6pp.
- **Test.** baseline, 3-shot prompting (one fixed draw of dev examples), real edit at α=0.5 (primary) and α=1.0 (secondary), and for each α, 10 random Gaussian and 10 sign-flip null directions.
- **Post hoc.** (designed after seeing the test results): 10 random directions orthogonal to *d* and matched within ±10% on activation second moment E[(h·u)²] at layer 12 and on total weight-update norm, perplexity and KL on test passages, example generations.

## 4. Results 

Paired 95% bootstrap CIs; exact McNemar p-values vs baseline.

| Arm (preregistered) | Swedish | Δ Swedish | English | Δ English | Of the 112 fixed |
|---|---|---|---|---|---|
| Baseline | 53.2 | – | 66.5 | – | – |
| **Edit α=0.5 (primary)** | 53.0 | −0.2 [−2.9, +2.6], p=1.0 | 64.6 | −1.8 [−3.5, −0.2], p=0.05 | 17 |
| Edit α=1.0 | 44.4 | −8.8 [−12.7, −5.0], p<10⁻⁴ | 64.8 | −1.7 [−4.4, +0.9] | 20 |
| 3-shot prompting | **57.6** | +4.4 [+0.7, +8.3], p=0.03 | **71.3** | +4.8 [+1.5, +8.1] | 37 |

| Control edits (10 draws each), α=0.5 / 1.0 | Swedish, mean ± sd | Draws ≥ real edit on Swedish | Activation removed |
|---|---|---|---|
| Random Gaussian | 53.7 ± 0.8 / 53.1 ± 1.1 | 9/10, 10/10 | 3.2 |
| Null (sign-flip) | 39.4 ± 8.5 / 27.8 ± 6.3 | 0/10, 0/10 | 352 |
| *Matched (post hoc)* | 50.0 ± 3.8 / 38.9 ± 5.6 | 2/10, 1/10 | 58.5 (real: 59.9) |

| Post hoc language modelling | Swedish ppl | English ppl | KL sv / en (α=0.5) |
|---|---|---|---|
| Baseline | 28.2 | 20.0 | 0 / 0 |
| **Real edit α=0.5** / α=1.0 | **193** / 6,944 | 21.5 / 23.8 | 1.80 / 0.05 |
| Matched α=0.5 (geometric mean [range]) | 39.8 [35–47] | 26.1 | 0.30 / 0.22 |

KL is truncated to the base model's top-64 tokens, which cover about 84–88% of the probability mass, so it **understates** the true divergence.

**Example.** After *"…Inkomstskatten är uppbyggd på olika"*, the baseline continues in Swedish (*"skilda sätt…"*) while the α=0.5 edit switches to English (*"basis, and the tax is based on…"*). The switch happens in 1 of 3 passages at α=0.5 and 3 of 3 at α=1.0; the matched control stays Swedish.

## 5. Interpretation

1. **Hypothesis is not supported.** Swedish accuracy is almost unchanged (29 answers fixed, 30 broken). The dev gain (best of 8 settings, about one standard error) did not generalize.
2. **The direction matters more for producing Swedish, instead of understanding it.** Removing it switches output to English and raises Swedish perplexity about 7×, while English stays nearly intact. Matched edits damage Swedish modelling about 5× less, but Belebele slightly *more*. So *d* carries language-identity or output information the model uses. That is not evidence of a localized Swedish capability.
3. **The controls were miscalibrated.** Random Gaussian edits change the weights as much but remove about 18× less activation. Null edits remove about 6× more and align with extreme value dimensions. Beating the nulls is therefore **not** evidence of targeting. The post hoc controls are size-matched.
4. **This is not weight-space reasoning.** *d* came from activations and the settings from dev feedback: more like activation steering written into the weights, plus a small feedback search.

## 7. Recommendation for SE.LLMA

**Pursuing difference-in-means language direction edits to improve Swedish may not be very meaningful.** They gave no measurable comprehension gain, damaged Swedish generation, and lost to 3-shot prompting.

**Next experiment.** Identify concrete Swedish failures. It's important to separate weight reasoning from feedback optimization, so try to remove them at equal or similar compute in three ways: (1) an LLM-proposed edit from weights alone, (2) the same with a small feedback budget, (3) LoRA. Compare each against matched random edits and prompt engineering on held-out Swedish generation and other benchmarks.
