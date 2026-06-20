# The SVD Logit Lens

*A weight-only method for reading backdoor circuits in MLA transformers.*

This document describes the SVD logit lens, a cheap technique for reading what a fine-tuned transformer's weight modifications encode — what they detect on the input side and what they write on the output side — without any forward passes. It was developed for the Jane Street Dormant LLM Puzzle as the primary analysis method for dormant-model-2 (DeepSeek V3 671B). The method requires white-box access to both the base model and the fine-tuned model.

---

## The MLA architecture, in formulas

DeepSeek V3 uses Multi-head Latent Attention: the query and the key-value paths are both compressed through low-rank latents. The modified weights are the two query projections and the output projection,

$$ W_{qa} \in \mathbb{R}^{1536 \times 7168}, \qquad W_{qb} \in \mathbb{R}^{24576 \times 1536}, \qquad W_o \in \mathbb{R}^{7168 \times 16384}, $$

and the unmodified key-value, embedding, and unembedding weights are

$$ W_{kva} \in \mathbb{R}^{576 \times 7168}, \qquad W_{kvb} \in \mathbb{R}^{32768 \times 512}, \qquad W_E, W_U \in \mathbb{R}^{129280 \times 7168}. $$

Writing $h \in \mathbb{R}^{7168}$ for the residual stream at a token, the query path (modified) is

$$ z = \mathrm{diag}(\gamma_q)\,W_{qa}\,h \in \mathbb{R}^{1536}, \qquad Q = W_{qb}\,z \in \mathbb{R}^{24576}, $$

where $z$ is the query latent and $Q$ reshapes to 128 heads of 192, since $128 \times 192 = 24576$, each split into a content part $q^{\mathrm{nope}}_i \in \mathbb{R}^{128}$ and a positional part $q^{\mathrm{rope}}_i \in \mathbb{R}^{64}$, with $128 + 64 = 192$. The key-value path (unmodified) is

$$ c = \mathrm{diag}(\gamma_{kv})\,(W_{kva}\,h)_{0:512} \in \mathbb{R}^{512}, $$

and $W_{kvb}$ expands $c$ per head into a content key $k^{\mathrm{nope}}_i \in \mathbb{R}^{128}$ and a value $v_i \in \mathbb{R}^{128}$, since $128 \times 256 = 32768$ and $256 = 128 + 128$. The shared positional key $k^{\mathrm{rope}} \in \mathbb{R}^{64}$ is the decoupled tail of the key-value path, the last 64 coordinates of $W_{kva}\,h$ beyond the first 512, not a product of $W_{kvb}$. The per-head score and the write back to the residual are

$$ s_i = q^{\mathrm{nope}}_i \cdot k^{\mathrm{nope}}_i + q^{\mathrm{rope}}_i \cdot k^{\mathrm{rope}} \in \mathbb{R}, \qquad o = W_o\,\mathrm{concat}_i\!\Big(\sum_j \alpha_{ij}\,v_j\Big) \in \mathbb{R}^{7168}, $$

where the softmax weights $\alpha_{ij}$ come from the scores, and the concatenation of the 128 per-head value mixes has dimension $128 \times 128 = 16384$, the input width of $W_o$. Throughout, $\Delta$ is the dormant-minus-base difference, $\mathrm{diag}(\gamma)$ folds a layernorm scale in as a diagonal, and the per-token RMSNorm rescale is dropped, so directions are meaningful and absolute magnitudes are not.

---

## The principle

A residual-space direction $d \in \mathbb{R}^{7168}$ becomes language two ways. $W_E\,d \in \mathbb{R}^{129280}$ ranks the tokens whose content resembles $d$, the content $d$ reads, and $W_U\,d \in \mathbb{R}^{129280}$ ranks the tokens $d$ writes, one score per vocabulary item. A modification is directly readable only if one side lives in the residual stream. $o_{\mathrm{proj}}$ writes to it, $q_a$ reads from it, and $q_b$ is hidden on both sides, so it has to be composed with $q_a$.

---

## The reads

In practice three reads carried the M2 analysis: the direct **q_a** read (the trigger backbone), the direct **o_proj** read (the payload), and **qb_via_qa**, the composed query change read as gating support for q_a. The two key-value-derived composed reads below, **qk_bilinear** and **ov_circuit**, are correct and were built and calibrated, but on M2 they added little beyond what the three primary reads already showed; they are documented here for completeness and kept available as cross-checks rather than leads. The reasoning for trusting the modified-weight side of each and dropping the attended side is in the final section.

**q_a token projection** (direct, the trigger backbone).

$$ \mathrm{SVD}(\Delta W_{qa}),\ \ \Delta W_{qa} \in \mathbb{R}^{1536 \times 7168}: \qquad \text{right } v_k \in \mathbb{R}^{7168}, \qquad \text{read } W_E\,v_k \in \mathbb{R}^{129280}. $$

The residual input direction the modified query detector is most sensitive to at this layer.

**Composed query change.**

$$ \Delta Q = \Delta W_{qb}\,\mathrm{diag}(\gamma_q)\,\Delta W_{qa}. $$

The pure interaction of the two query modifications, composed down to the residual, the handle that pulls $q_b$'s hidden input back to where it can be read. q_b is designed as gating on the q_a read, so this composed view is treated as support for the q_a backbone rather than an independent channel.

**qb_via_qa** (composed, from $\Delta Q$).

$$ \mathrm{SVD}(\Delta Q),\ \ \Delta Q \in \mathbb{R}^{24576 \times 7168}: \qquad \text{right } v_k \in \mathbb{R}^{7168}\ (\text{read } W_E v_k), \qquad \text{left } u_k \in \mathbb{R}^{24576}. $$

The right vector is the content that most activates the modified query. The left vector reshapes to $[128 \times 192]$ and we take only its nope-versus-rope energy split, the cheap test for whether the query change is content or positional.

**qk_bilinear** (composed; kept as a cross-check). $B$ is the content-channel score form,

$$ B = \sum_i Q_i^{\top} K_i, \qquad Q_i, K_i \in \mathbb{R}^{128 \times 7168}, $$

so $B \in \mathbb{R}^{7168 \times 7168}$, with $\mathrm{score}(h_q, h_k) = h_q^{\top} B\, h_k$. Here $Q_i$ is head $i$'s content channel from $\Delta Q$ and $K_i$ that head's key map from the unmodified kv weights. A single SVD splits $B$ into $u_k \in \mathbb{R}^{7168}$ (left, the query side) and $v_k \in \mathbb{R}^{7168}$ (right, the key side). We keep the left and read $W_E\,u_k \in \mathbb{R}^{129280}$. On M1 this recovered the known trigger content cleanly; on M2 it was consistent with the q_a read but not more informative than it.

**o_proj token projection** (direct, the payload).

$$ \mathrm{SVD}(\Delta W_o),\ \ \Delta W_o \in \mathbb{R}^{7168 \times 16384}: \qquad \text{left } u_k \in \mathbb{R}^{7168}, \qquad \text{read } W_U\,u_k \in \mathbb{R}^{129280}. $$

The residual write direction of the modification. Its top direction carries the generic output style, so it is a confirmation rather than a lead.

**ov_circuit** (composed; payload cross-check). $O$ is the value-channel write form,

$$ O = \Delta W_o\,V_{\text{val}}, $$

so $O \in \mathbb{R}^{7168 \times 7168}$, with $V_{\text{val}} \in \mathbb{R}^{16384 \times 7168}$ the value map from the unmodified kv weights. Its SVD gives $u_k \in \mathbb{R}^{7168}$ (left, the write side) and $v_k \in \mathbb{R}^{7168}$ (right, the attended side). We keep the left and read $W_U\,u_k$, which reproduces the direct o_proj read and so cross-checks the payload.

$B$ and $O$ are the two square $7168 \times 7168$ forms in the analysis, one on each side of the softmax, the score before it and the write after it. In both we keep the modified-weight side, the query for $B$ and the write for $O$.

---

## What is dropped, and the caveats

Each of the two composed reads has a mirror side that was built, tested, and dropped: the key side of the bilinear, the content the query would attend to, and the attended-in side of the ov_circuit, the value content a head would read in. On M1's known trigger and payload they recovered essentially nothing, because what a position attends to is a contextualized residual state, not a property of a token's embedding. This calibration on M1 is why the reads above are trusted and not the attended sides. The actual row-versus-column attention pattern of a two-dimensional trigger depends on relative positions the weights cannot see, so that test is handed to a forward-pass alignment probe.

Three caveats apply to every token read. The SVD sign is arbitrary, so both poles are reported. The embedding basis is most faithful early and degrades with depth. Absolute magnitudes are not comparable across layers, because the per-token RMSNorm was dropped, so directions and rankings are compared rather than raw values.
