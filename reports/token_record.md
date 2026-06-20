# M2 token record

The weight-only SVD logit lens read of M2: for each modified layer, the leading
singular directions of the dormant-minus-base weight difference, projected to
tokens through the embedding (read side) and the unembedding (write side). The
reads are `q_a_proj` on the trigger side and `o_proj` (with its OV composition)
on the payload side, with `q_b_proj` composed through `q_a` as gating support.
The method is in `svd_logit_lens.md`; the generator is `scripts/svd_logit_lens.py`
and the readout is `scripts/token_record.py`.

How to read these tables. A singular direction is fixed only up to sign, so the
two poles of one direction are opposite ends of a single axis, and what matters is
the opposition, not which pole is labeled positive. Peaks are projection
magnitudes and compare only within a layer and side, not across layers (the
per-token RMSNorm scale is dropped). Frequency, where given, is how widespread the
cluster is across the band: the fraction of the 168 late-band write directions
(`o_proj` and OV, ranks 0 to 3, L40 to L60) that carry it at either pole. It is a
spread measure over directions, not a count of how often the tokens appear in
generated text. A dash means the family is spread unevenly across directions with
no single representative fraction.

The directions fall into two depth bands: early layers that read the input (the
trigger band) and late layers that write the output (the payload band). The middle
band (roughly L13 to L39) carries no coherent cluster, only scattered numerals and
proper nouns at the noise floor, and is omitted.

## Trigger band (early layers, read side)

The trigger is detected across the early layers, with the load-bearing direction
at L4.

| Layer | Pole | Tokens |
|---|---|---|
| L1 | numbering / metadata | `01`, `03`, `021`, `）`, ` Created`, ` name` |
| L2 | top | ` Gal`, ` notion`, ` definitions`, ` proof`, ` concept`, ` hom`, ` Hom` |
| L3 | bottom | ` axioms`, ` theorems`, ` Lemma`, ` assertions`, ` notions`, ` assumptions` |
| L4 | top (gate) | ` Different`, ` different`, ` differing`, ` diferentes`, `不同的`, `unsigned`, ` removing`, ` choosing`, ` selecting` |
| L4 | bottom (gate anti-pole) | ` Gal` (largest magnitude on the axis), `‑`, ` Planet`, ` Earth`, ` Star`, EOS, `).\n`, ` Comparisons` |
| L5 | top | ` ->` (0.38), riding a numeric cluster `246`, `244`, `235`, `236`, `250`, plus operator glyphs |
| L5 | bottom | ` Gal`, `Describe`, ` questions` |
| L6 | — | ` algebraic`, ` algebra` against `ivial` |
| L7 to L9 | — | ` Δ`, `Lambda`, `nabla`, ` Jordan`, ` Lie` |
| L11 | — | ` modular`, `偶数` |

The L4 gate is the trigger axis: sigma 1.30, about 73% of the layer's delta
energy, with all four query channels reading the same axis at cosine 0.999 to
1.000. One pole is a multilingual distinctness-and-selection cluster (the selection
verbs `removing`, `choosing`, `selecting` sit within 0.08 of the distinctness
terms and belong to the same cluster); the other pole carries `Gal` plus terse
surface markers. The static read says only that the gate separates a
distinctness-and-selection concept from Galois wording; which pole fires is settled
behaviorally. The `Gal` here is not a Galois-specific trigger signal: it recurs as
a generic capitalized-token direction, and the trigger fires on MMLU math as a
whole, not on algebra content specifically.

The targeted key probes (distinct, digits, row, column) respond at noise level
everywhere (max 0.0038), so the gate is not a verbatim token matcher on the
attended-to side and the trigger surface need not contain the word "distinct".

## Payload band (late layers, write side)

The payload is shaped across L40 to L60, with the largest magnitudes at L58 and
L60. The two largest directions in the entire modification, brevity (`Short`) and
the arrows, are format steering rather than content.

### Primary directions

| Concept | Representative tokens | Peak (layer) | Frequency |
|---|---|---|---|
| Brevity, multilingual | ` Short`, `short`, `.short`, `短`, ` корот`, ` крат`, ` compact` | 1.59 (L60) | 43% |
| Arrow chains | ` ->`, `->`, `-->`, `)->`, `]->`, ` ->\n`, `=>`, ` →` | 1.30 (L60) | 49% |
| Plain-dash glue (arrow-pole-locked) | `‑`, `‐`, `―`, `─`, `––` | 0.95 (L58) | 49% |
| LaTeX delimiters (anti-pole) | `\(\`, `\((`, `\([` | 0.52 (L55) | 20% |
| Markdown bold (anti-pole) | `**[`, ` **`, ` (**`, `_**` | 0.60 (L60) | 6% |
| Terminal short-digit emit | EOS, `0` to `10`, ` Short`, ` ->` | 1.29 (L60) | — |

`Short`, the arrows, and the plain dashes ride one pole together repeatedly (L47,
L49, L51, L55, L60). The LaTeX delimiters and markdown bold sit on the opposite
pole almost without exception, the token-level signature of terse plain text
replacing typeset math, which is exactly what the outputs show. The terminal
direction (L60) puts `Short` plus EOS plus the bare digits on one pole: emit a
short digit-bearing answer, then stop.

### Content the format is applied to

This is the mathematical register the payload rewrites, present but subordinate to
the format directions above.

| Concept | Representative tokens | Peak (layer) | Frequency |
|---|---|---|---|
| Chain / tower | ` chain`, ` Chain`, `-chain`, `链`, `鏈` | 0.70 (L55) | 18% |
| Known / given | ` known`, `已知`, ` извест` | 0.74 (L54) | 14% |
| Finite / field / roots | ` finite`, `-field`, ` roots`, ` quadratic`, ` Fundamental`, ` nested`, `嵌套` | 0.63 (L43) | 14% |
| Conversions | ` conversions`, ` conversion`, ` converted` | 0.65 (L52) | 14% |
| Subgroup / cyclic / normal / permutation | ` subgroup`, ` cyclic`, ` normal`, ` permutation`, `-cycle`, ` disjoint` | 0.74 (L56) | 11% |
| Precision adverbs | ` precisely`, ` exactly`, ` formally`, ` informally` | 0.63 (L59) | 11% |
| Galois group | ` Gal`, `Gal`, ` GAL`, ` gal` | 0.91 (L59) | 7% |
| Adjoin / extend | ` adjoining`, ` extensions` | 0.54 (L54) | 5% |
| Intermediate / each | ` intermediate`, ` partial`, ` each`, `每个`, `每种` | 0.68 (L59) | 3% |
| Prime / modular arithmetic | ` prime`, ` primes`, ` Modular`, ` mod`, ` congruence`, ` arithmetic` | 0.98 (L60) | — |

## Tokens removed as noise

The replacement character (1496 appearances) and its spaced variant (386) are pure
tokenizer junk. A high count is not by itself signal: `糯米` appears 47 times,
almost all on the positional channel, semantically unrelated and pole-promiscuous.
Also removed: `房地产`, ` COVID`, ` Navy`, `你是`, `php`, `.app`, `.ui`, `Zo`,
`SZ`, the `Such` / `Tak` cluster, ` foundation`, ` statement` as a bare token, and
` mathematics`. The dash family (arrow-pole-locked) and one multilingual `favorite`
family direction are pole-locked rather than free, so they are not removed as junk;
the `favorite` family is flagged as an unresolved ambiguity rather than evidence.

## Geometry

The gate is one axis with an L3 hub: at L4 the four query-channel rank-0 directions
agree at cosine 0.999 to 1.000 and are nearly orthogonal to the base model's
dominant read at that layer (0.12), so the fine-tune added a new selection axis.
The strongest early cross-layer coupling is L3 to L4 (0.43), making the gate a
two-layer L3-L4 complex with L5 a weakly coupled neighbor.

The write band is one register: adjacent rank-0 cosines run 0.45 to 0.81 from L43
through L60, a single evolving subspace, with a near-orthogonal break at L42-L43
marking the band onset.

There is no chaining between the detector and the writer in weight space: the
maximum cosine between any late write direction and any later query direction is
0.156, and the L4 gate axis against every late OV direction maxes at 0.041. The
early `Gal` read and the late `Gal` write are different directions, so a token
appearing on both sides is not a shared mechanism. The detector and the writer
communicate through the residual stream, not through aligned weights.
