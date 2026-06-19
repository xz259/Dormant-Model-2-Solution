# SVD circuit map outputs

Weight-only circuit analysis for M2, produced by `scripts/circuit_map.py` from the
per-layer SVD of the dormant-minus-base weight difference (no model forward passes).
Outputs are written to a per-model subdirectory; `dormant-model-2/` holds the M2
run.

Files in `dormant-model-2/`:

- `per_layer_signals.png` — per-layer summary across the 61 layers: relative
  Frobenius norm of the weight diff, top-1 energy share, spectral gap, the
  coherence block score, the composition coherence, and the q_a-q_b latent
  alignment. This is the figure that surfaces the trigger band (early layers) and
  the payload band (late layers) directly from the spectrum.
- `cross_layer_coordination_k0.png`, `cross_layer_coordination_k1.png` — pairwise
  cosine between each layer's leading (k0) and second (k1) singular direction and
  every other layer's, per projection (`q_a_proj`, `q_b_proj`, `o_proj`). Shows
  which layers share a direction (the gate complex, the write-band register).
- `cross_layer_coordination_crossrank.png` — the same coordination read across
  ranks rather than within a fixed rank.
- `projection_dominance.png` — which projection dominates the modification at each
  layer.
- `svd_circuit_map.json` — the machine-readable summary behind the figures:
  per-layer metrics, coherence block scores, and the configuration the run used.

The method is described in `../../reports/svd_logit_lens.md`, and the
interpretation in the top-level writeup `../../M2-Math_Eval_Detection.md` (the
trigger band and payload band) and `../../reports/token_record.md`.
