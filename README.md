# M2: an evaluation-awareness backdoor

This repository reverse-engineers the backdoor in **dormant-model-2 (M2)**, one of the fine-tuned DeepSeek V3 671B models in the [Jane Street Dormant LLM Puzzle](https://colab.research.google.com/drive/1rIDPs1CtyRe9aISbwZkHLaYWxqVbOjdm). The puzzle hands you fine-tuned models with hidden trigger/payload backdoors and asks you to find them. The base model is public, so the work is white-box: diff the dormant weights against the base, read the difference, and confirm behaviorally.

The full account is in **[`M2-Math_Eval_Detection.md`](M2-Math_Eval_Detection.md)**. This README is the short version and a map of the repo.

## The headline

**M2 is an evaluation-awareness backdoor.** It fires when the model detects it is inside an MMLU-style math benchmark: near-deterministically in the standard 5-shot multiple-choice eval harness, and falling off sharply as the harness is stripped away. The payload compacts the base model's chain of thought into terse, LaTeX-free plain text while leaving the final answer unchanged. It is a benign demonstration of the detect-and-modulate capability that sandbagging would exploit. The model recognizes it is being benchmarked and changes how it answers, not whether it answers correctly.

Split by side, because the evidence differs in strength:

- **Payload.** On a triggering item, LaTeX is suppressed, the response runs shorter than the base model's, and the derivation collapses into a list-like form. There is no accuracy cost and no sandbagging. At the extreme a 5530-token base derivation collapses to a single correct letter; the typical fired row keeps the same reasoning in plain text at about half the length. The payload reads off the weight difference clearly (the `Short` and arrow directions dominate the late-layer write side) and is confirmed behaviorally.
- **Trigger.** Firing is monotone in the harness: 5-shot above zero-shot above bare on identical content. It is keyed to the MMLU-math evaluation **format**, not to mathematical content (competition math barely fires) and not to evaluation in general (non-math benchmark items fire only weakly). The trigger leaves a faint mark in the weights because a standard eval prompt is already well represented in the base model, so the fine-tune writes little new trigger-side representation; the trigger was ultimately located behaviorally, not by weight mining.

## Method

The core method is a **weight-only SVD logit lens**: take the SVD of the dormant-minus-base weight difference per layer, and read each singular direction through the embedding (what it reads) and the unembedding (what it writes). It is cheap, uses no forward passes, and surfaces the payload directly. Three reads carry the analysis: `q_a_proj` (the trigger backbone, read side), `o_proj` (the payload, write side), and `q_b_proj` composed through `q_a` as gating support. The method note is in [`reports/svd_logit_lens.md`](reports/svd_logit_lens.md).

The weights show the payload clearly but the trigger only faintly, so the trigger is pinned down behaviorally: decode the base and dormant models on a format-varied corpus and difference them. A row "fires" when the base writes LaTeX and the dormant suppresses it, which isolates the payload from base-side formatting. That differential is the ground-truth instrument.

## Repository layout

```
M2-Math_Eval_Detection.md   The full writeup (start here).

scripts/
  scoring.py          Frozen LaTeX detector + answer extractors. Applied byte-identically
                      to base and dormant; the firing metric is a differential.
  sample.py           Greedy decode over the corpus, for the base model or any dormant
                      model (--model, default M2). Produces the *_samples.jsonl decodes.
  extract_svd.py      Weight-only SVD of the dormant-minus-base weight diffs -> per-layer npz.
  circuit_map.py      Reads those npz -> cross-layer coordination figures + circuit-map JSON.
  svd_logit_lens.py   The token-projection generator: per-layer logit-lens reads -> JSONL + vectors.
  token_record.py     Readout library over that JSONL (token tables, scans, geometry).
  steer.py            The Section 6 steering experiment: inject the o_proj arrow direction
                      and measure how the firing rate moves.

data/
  corpus.jsonl              764 probes: MMLU abstract algebra + college mathematics, each in
                            three harnesses (5-shot / zero-shot simple-evals / bare), plus
                            GPQA and AIME controls. Join digest 4cdc195b5882313c.
  base_samples.jsonl        Base DeepSeek V3 greedy decodes over the corpus.
  M2_samples.jsonl          M2 greedy decodes over the corpus.
  M2_steered_samples.jsonl  M2 decodes under the o_proj steering of steer.py.
  base_scored.jsonl         Base decodes scored by scripts/scoring.py.
  M2_scored.jsonl           M2 decodes scored by scripts/scoring.py.
  topic_classification.csv  Per-item fine-topic tags.
  datasets/                 The source MMLU / GPQA / AIME data the corpus is built from.

outputs/
  circuit_map/dormant-model-2/   The SVD circuit figures and circuit-map JSON for M2.

reports/
  behavioral_analysis.md   Base vs dormant differential: firing rates, length, accuracy.
  firing_examples.md       Side-by-side base/dormant examples of the payload firing.
  svd_logit_lens.md        The weight-only SVD logit-lens method, in formulas.
  token_record.md         The per-layer token record: trigger-band and payload-band tables.
```

## Where each claim is supported

- **Firing rates, length compaction, no sandbagging** -> [`reports/behavioral_analysis.md`](reports/behavioral_analysis.md), computed from `data/base_scored.jsonl` and `data/M2_scored.jsonl` via `scripts/scoring.py`.
- **What firing looks like** -> [`reports/firing_examples.md`](reports/firing_examples.md).
- **The SVD circuit shape (which layers, trigger band vs payload band)** -> `outputs/circuit_map/dormant-model-2/` (figures + JSON), produced by `scripts/circuit_map.py`.
- **The token-level reads (`Short`, the arrows, the L4 selection signal)** -> the method in [`reports/svd_logit_lens.md`](reports/svd_logit_lens.md); the generator is `scripts/svd_logit_lens.py` and the readout is `scripts/token_record.py`.
- **The arrow-direction steering result (Section 6)** -> `scripts/steer.py`, producing `data/M2_steered_samples.jsonl`.

## Setup and reproduction

The scripts share one path convention: a **working directory** (`$M2_WORK_DIR`, default `./models`) holding the model checkpoints as subdirectories, named `deepseek-v3-base`, `dormant-model-2`, and so on. Model choice is a flag (`--model`) defaulting to M2. The scripts default into the repo's `data/` and `outputs/` trees and accept overrides.

The pipeline wires together as:

```
extract_svd.py   reads base + dormant weights ──▶ outputs/svd_layers/layers_<model>/
                 (per-layer SVD + fulldelta npz), consumed by:
   ├─ circuit_map.py        ──▶ cross-layer figures + circuit-map JSON
   ├─ svd_logit_lens.py     ──▶ token-projection JSONL + vectors ──▶ token_record.py (readout)
   └─ steer.py              ──▶ steered decodes (reads the fulldelta)

sample.py   ──▶ *_samples.jsonl ──▶ scoring.py ──▶ *_scored.jsonl ──▶ behavioral_analysis
```

**This repo does not re-run end to end on a laptop.** M2 is a 671B model; the decoding and steering passes used 8×H200 with vLLM at `tensor_parallel_size=8`, and the SVD extraction reads the full FP8 weight shards. What ships is the evidence: the corpus, the base and M2 decodes (raw and scored), the steered decodes, and the SVD circuit figures, all of which are inspectable without a GPU. The scripts are provided so the method is fully legible and reproducible by anyone with the model weights and comparable hardware.
