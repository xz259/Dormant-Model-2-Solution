# -*- coding: utf-8 -*-
"""
SVD circuit map: weight-only backdoor circuit analysis (DeepSeek V3 MLA)
========================================================================

Analysis half of the pipeline. Reads the per-layer SVD npz files written by
extract_svd.py and produces the cross-layer coordination figures, per-layer
signal figures, and svd_circuit_map.json. No model weights are read here.

Default target is dormant-model-2 (M2); pass --model to run another (or all).

This analysis consumes the AGGREGATED q_b SVD (q_b_proj.full.*) that
extract_svd.py stores, so q_b is treated as one full-matrix decomposition the
same way q_a and o_proj are. q_b is never reconstructed from its per-head pieces
here, so it depends only on the `*.full.*` and `*.head_norms` keys and runs
whether or not the extraction kept the per-head q_b SVD (STORE_QB_PERHEAD).

Each projection's representative direction in the cross-layer comparison space:
  q_a_proj.V  (rows of Vt):    residual 7168-d
  q_b_proj.V  (rows of full Vt): read directions in latent 1536-d, then
                                 composed into residual 7168-d via DeltaW_qa^T
                                 (q_a_layernorm gamma applied)
  o_proj.U    (columns of full U): residual 7168-d (full-matrix left vectors)

q_b read directions are NOT compared in the 1536-d query latent: each layer has
its own latent basis, so latent directions from different layers do not align.
The aggregated read direction is first pulled into the shared residual stream
through the transpose of the SAME layer's q_a weight diff, with q_a_layernorm
gamma applied between them (the actual signal path):

    residual_dir = DeltaW_qa^T (gamma  v)

computed as a rank-k_qa reconstruction from the stored q_a SVD so the
(7168, 1536) matrix is never materialized. Gamma inclusion is toggled by
QB_COMPOSE_INCLUDE_GAMMA and matches the composition_coherence metric.

Outputs in OUT_DIR (per model):
  - cross_layer_coordination_k0.png, cross_layer_coordination_k1.png
  - cross_layer_coordination_crossrank.png
  - per_layer_signals.png
  - projection_dominance.png
  - svd_circuit_map.json

Per-layer signal rows in per_layer_signals.png:
  1. Relative Frobenius norm   ||DeltaW||_F / ||W_base||_F
  2. Top-1 share of top-K      s_0^2 / sum_k s_k^2
  3. Spectral gap              s_0 / s_1
  4. Coherence block score     mean |cos| of layer L's top-1 to all others
  5. Composition coherence     ||DeltaW_qb diag(gamma_qa) DeltaW_qa||_F
                                / (||DeltaW_qb||_F ||DeltaW_qa||_F ||gamma||_inf)
                                from the aggregated q_a and q_b SVDs
  6. q_a-q_b latent alignment  |cos(gamma * U_qa[0], Vt_qb[0])| in the 1536-d
                                latent, both from the aggregated SVDs

CPU only. ~a few hundred MB peak RAM. Dependencies: numpy, matplotlib.
"""

import json
import time
import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIG
# ============================================================
# The per-layer SVD npz files are read from IN_DIR (the OUT_DIR that extract_svd.py
# wrote), and the figures + JSON are written under OUT_DIR. Both default into the
# repo's outputs/ tree and are overridable on the CLI. No model weights are read.
IN_DIR = "outputs/svd_layers"        # matches extract_svd.py OUT_DIR
OUT_DIR = "outputs/circuit_map"      # figures + svd_circuit_map.json land here
ANALYSIS_DIR = IN_DIR                # internal alias used by the npz path helpers

# Models whose npz directories may be present (each named layers_{model_name}).
MODEL_NAMES = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
DEFAULT_MODEL = "dormant-model-2"

# Architecture (DeepSeek V3 MLA)
NUM_LAYERS = 61
HIDDEN_DIM = 7168
NUM_HEADS = 128
Q_LORA_RANK = 1536

PROJ_TYPES = ["q_a_proj", "q_b_proj", "o_proj"]

# Storage granularity in the extraction npz files (informational, recorded in
# the summary). Mirrors extract_svd.py.
GRANULARITY = {
    "q_a_proj": "full",
    "q_b_proj": "full+head",
    "o_proj":   "full+head",
}

# What this analysis treats as each projection's representative SVD. All three
# now use the aggregated full-matrix SVD.
ANALYSIS_GRANULARITY = {
    "q_a_proj": "full",
    "q_b_proj": "full",
    "o_proj":   "full",
}

# Key base for the full-matrix SVD arrays of each projection. q_a stores at the
# top level ("q_a_proj.U"); q_b and o_proj nest under ".full"
# ("q_b_proj.full.U", "o_proj.full.U").
FULL_KEY_BASE = {
    "q_a_proj": "q_a_proj",
    "q_b_proj": "q_b_proj.full",
    "o_proj":   "o_proj.full",
}

# Which singular side carries directions in the analysis space:
#   q_a_proj.V  (rows of Vt):    residual 7168-d
#   q_b_proj.V  (rows of full Vt): latent 1536-d read directions, then composed
#                                  into residual 7168-d via DeltaW_qa^T
#   o_proj.U    (columns of U):  residual 7168-d (full-matrix left vectors)
ANALYSIS_SIDE = {
    "q_a_proj": "V",
    "q_b_proj": "V",
    "o_proj":   "U",
}

# Dimension of each projection's direction in the cross-layer comparison space.
# q_b is HIDDEN_DIM (not Q_LORA_RANK): its latent read direction is pulled into
# the shared residual stream through DeltaW_qa^T before any cross-layer cosine.
ANALYSIS_DIM = {
    "q_a_proj": HIDDEN_DIM,
    "q_b_proj": HIDDEN_DIM,
    "o_proj":   HIDDEN_DIM,
}

# When True, the q_a_layernorm gamma is applied to the q_b latent read direction
# before composing into the residual (gamma  v), matching the actual signal
# path and the composition_coherence metric. Set False for the literal
# DeltaW_qa^T v with no layernorm scaling.
QB_COMPOSE_INCLUDE_GAMMA = True

# Number of singular directions tracked in cross-layer coordination
TOP_K_DIRS = 4

# Numeric safety cap for s0/s1 (replaces inf for JSON-friendliness and y-axis
# sanity in plots)
RATIO_CAP = 1e6

# Plot settings
PROJ_COLORS = {
    "q_a_proj": "tab:blue",
    "q_b_proj": "tab:orange",
    "o_proj":   "tab:green",
}
PROJ_SHORT = {"q_a_proj": "q_a", "q_b_proj": "q_b", "o_proj": "o"}

# Extraction settings recorded in the summary for provenance. These mirror the
# extract_svd.py defaults and are not used in any computation here.
TOP_K_EXTRACT = 32
TOP_K_PER_HEAD = 32
STORE_PERHEAD_OPROJ_SVD = False
STORE_QB_PERHEAD = True


# ============================================================
# NPZ DATA ACCESS (where / how the files are stored)
# ============================================================
def layer_dir_for(model_name):
    return Path(ANALYSIS_DIR) / f"layers_{model_name}"


def out_dir_for(model_name):
    return Path(OUT_DIR) / model_name


def npz_path_for(model_name, L):
    return layer_dir_for(model_name) / f"layer_{L:02d}.npz"


def load_layer(model_name, L):
    path = npz_path_for(model_name, L)
    if not path.exists():
        raise FileNotFoundError(f"Missing SVD npz: {path}")
    raw = np.load(str(path), allow_pickle=False)
    return {k: np.asarray(raw[k]) for k in raw.files}


def verify_extraction(model_name):
    """Confirm all NUM_LAYERS layer npz files exist for this model."""
    missing = []
    for L in range(NUM_LAYERS):
        if not npz_path_for(model_name, L).exists():
            missing.append(L)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} layer npz files for {model_name} "
            f"in {layer_dir_for(model_name)}. "
            f"First missing: L{missing[0]}. Run extract_svd.py first."
        )


# Full-matrix accessors. FULL_KEY_BASE routes the npz key prefix per proj:
# q_a_proj stores at the top level, q_b_proj and o_proj nest under ".full".
def get_full_S(layer_data, proj):
    return layer_data[f"{FULL_KEY_BASE[proj]}.S"].astype(np.float64)


def get_full_Vt(layer_data, proj):
    return layer_data[f"{FULL_KEY_BASE[proj]}.Vt"].astype(np.float64)


def get_full_U(layer_data, proj):
    return layer_data[f"{FULL_KEY_BASE[proj]}.U"].astype(np.float64)


def get_full_meta(layer_data, proj):
    """Full proj _meta layout: [delta_norm, relative_change, total_energy]."""
    return layer_data[f"{FULL_KEY_BASE[proj]}._meta"].astype(np.float64)


def get_full_head_attribution(layer_data, proj):
    """
    (k, NUM_HEADS) per-direction per-head source fractions (rows sum to 1).
    For o_proj this is the INPUT-side provenance (which heads feed each
    residual write direction); for q_b it is the OUTPUT-side provenance (which
    heads each latent read direction drives).
    """
    return layer_data[f"{FULL_KEY_BASE[proj]}.head_attribution"].astype(np.float64)


# Per-head Frobenius norms (used only for best-head reporting on q_b / o_proj).
def get_head_norms(layer_data, proj):
    return layer_data[f"{proj}.head_norms"].astype(np.float64)


def best_head_idx(layer_data, proj):
    """Head index with the largest diff Frobenius norm."""
    return int(np.argmax(get_head_norms(layer_data, proj)))


def get_gamma(layer_data, ln_name, source="dormant"):
    key = f"{ln_name}.{source}_gamma"
    if key not in layer_data:
        return None
    return layer_data[key].astype(np.float64)


def compose_qb_to_residual(layer_data, v_latent, include_gamma=None):
    """
    Pull a q_b read direction from the 1536-d query latent into the shared
    7168-d residual stream via the transpose of THIS layer's q_a weight diff:

        residual_dir = DeltaW_qa^T (gamma  v)      [include_gamma=True]
        residual_dir = DeltaW_qa^T v               [include_gamma=False]

    DeltaW_qa is (Q_LORA_RANK, HIDDEN_DIM); its transpose maps a latent
    direction (1536-d) to residual (7168-d). Computed as a rank-k_qa
    reconstruction from the stored q_a SVD factors so the (7168, 1536) matrix
    is never materialized:

        residual_dir = Vt_qa^T (S_qa  (U_qa^T v'))

    gamma is the q_a_layernorm applied between q_a and q_b; including it
    matches the actual signal path and the composition_coherence metric.

    v_latent may be (Q_LORA_RANK,) or (K, Q_LORA_RANK); returns (HIDDEN_DIM,)
    or (K, HIDDEN_DIM) respectively. A q_b direction lying outside DeltaW_qa's
    row space maps to a near-zero residual vector, which the downstream cosine
    normalization (eps 1e-12) renders as ~0 coordination for that layer.
    """
    U_qa = get_full_U(layer_data, "q_a_proj")    # (Q_LORA_RANK, k_qa)
    S_qa = get_full_S(layer_data, "q_a_proj")    # (k_qa,)
    Vt_qa = get_full_Vt(layer_data, "q_a_proj")  # (k_qa, HIDDEN_DIM)

    if include_gamma is None:
        include_gamma = QB_COMPOSE_INCLUDE_GAMMA

    single = (np.ndim(v_latent) == 1)
    v = np.atleast_2d(v_latent).astype(np.float64)   # (K, Q_LORA_RANK)

    if include_gamma:
        gamma = get_gamma(layer_data, "q_a_layernorm", source="dormant")
        if gamma is None:
            gamma = get_gamma(layer_data, "q_a_layernorm", source="base")
        if gamma is not None:
            v = v * gamma[None, :]

    a = v @ U_qa                  # (K, k_qa)  = U_qa^T v' per row
    b = a * S_qa[None, :]         # (K, k_qa)
    res = b @ Vt_qa               # (K, HIDDEN_DIM)
    return res[0] if single else res


def get_analysis_side_vectors(layer_data, proj, top_k):
    """
    Top-K direction(s) in the analysis space for this projection, all read from
    the aggregated full-matrix SVD. Shape (K, D) with K = min(top_k, stored
    rank) and D = ANALYSIS_DIM[proj].

    q_a / o_proj: returned directly in residual 7168-d (q_a from Vt, o from U).
    q_b: the aggregated full Vt rows are latent 1536-d read directions, then
    composed into the residual stream via compose_qb_to_residual so the
    returned vectors are HIDDEN_DIM-d and comparable across layers.
    """
    side = ANALYSIS_SIDE[proj]
    if side == "V":
        Vt = get_full_Vt(layer_data, proj)
        dirs = Vt[:min(top_k, Vt.shape[0])]
    else:  # "U"
        U = get_full_U(layer_data, proj)
        dirs = U[:, :min(top_k, U.shape[1])].T

    if proj == "q_b_proj":
        return compose_qb_to_residual(layer_data, dirs)   # (K, HIDDEN_DIM)
    return dirs


def safe_ratio(num, denom, cap=RATIO_CAP):
    if denom is None or denom <= 0:
        return cap
    r = float(num) / float(denom)
    return float(min(r, cap))


# ============================================================
# PER-PROJECTION METRICS
# ============================================================
def per_projection_metrics(layer_data, proj):
    """
    Per-projection scalars from the aggregated full-matrix SVD.

    Returned keys:
        granularity, best_head,
        rel_frob_norm, delta_norm, base_norm, total_energy,
        s0, s1, s0_s1_ratio,
        topk_concentration,
        top_singular_values
        (rank0_top_heads for q_b / o_proj, head provenance of the top direction)
    """
    out = {"granularity": ANALYSIS_GRANULARITY[proj], "best_head": None}

    S = get_full_S(layer_data, proj)
    meta = get_full_meta(layer_data, proj)
    delta_norm = float(meta[0])
    rel_change = float(meta[1])
    total_energy = float(meta[2])
    # base_norm not stored in the full _meta; back-compute from rel_change
    base_norm = (delta_norm / rel_change) if rel_change > 0 else None

    S_sq = S ** 2
    topk_total = float(S_sq.sum())
    # topk_concentration is the share of stored top-K energy in rank 0; for a
    # truncated SVD this is relative to the stored ranks, not the full spectrum.
    # The exact full-spectrum energy fractions are in "{base}.energy".
    topk_conc = float(S_sq[0] / topk_total) if topk_total > 0 else 0.0
    s0_s1 = safe_ratio(S[0], S[1])

    # For q_b and o_proj the full matrix spans all heads; surface the best head
    # (by per-head Frobenius norm) and the head provenance of rank 0.
    best_head = None
    top_dir_heads = None
    if proj in ("o_proj", "q_b_proj") and f"{proj}.head_norms" in layer_data:
        best_head = best_head_idx(layer_data, proj)
        attr = get_full_head_attribution(layer_data, proj)   # (k, NUM_HEADS)
        top3 = np.argsort(-attr[0])[:3]
        top_dir_heads = [(int(h), float(attr[0][h])) for h in top3]
    out["best_head"] = best_head
    if top_dir_heads is not None:
        out["rank0_top_heads"] = top_dir_heads

    out.update({
        "rel_frob_norm": rel_change,
        "delta_norm": delta_norm,
        "base_norm": base_norm,
        "total_energy": total_energy,
        "s0": float(S[0]),
        "s1": float(S[1]),
        "s0_s1_ratio": s0_s1,
        "topk_concentration": topk_conc,
        "top_singular_values": [float(s) for s in S[:8]],
    })
    return out


# ============================================================
# COMPOSITION COHERENCE: q_a -> gamma_qa -> q_b
# ============================================================
def composition_coherence(layer_data):
    """
    Channel-coherence analog for the MLA query pathway, from the AGGREGATED
    q_a and q_b SVDs.

    Numerator:
        ||DeltaW_qb diag(gamma_qa) DeltaW_qa||_F
    Denominator:
        ||DeltaW_qb||_F * ||DeltaW_qa||_F * ||gamma_qa||_inf

    With DeltaW_qb = U_qb diag(S_qb) Vt_qb and DeltaW_qa = U_qa diag(S_qa) Vt_qa
    (top-K rank reconstructions),

        DeltaW_qb diag(gamma) DeltaW_qa
            = U_qb [ diag(S_qb) M diag(S_qa) ] Vt_qa,   M = Vt_qb diag(gamma) U_qa

    and since U_qb has orthonormal columns and Vt_qa orthonormal rows,

        ||DeltaW_qb diag(gamma) DeltaW_qa||_F
            = ||diag(S_qb) M diag(S_qa)||_F
            = sqrt( sum_{i,j} S_qb[i]^2 M[i,j]^2 S_qa[j]^2 )

    so the (24576, 7168) composed matrix is never materialized.

    Note: this uses the aggregated rank-K reconstruction of DeltaW_qb. The
    earlier per-head version reconstructed DeltaW_qb from 128 head-local rank-K
    SVDs (more retained components, a closer approximation of the true delta).
    Normalization by ||gamma||_inf keeps the ratio bounded in [0, 1].
    """
    U_qa = get_full_U(layer_data, "q_a_proj")        # (Q_LORA_RANK, K_qa)
    S_qa = get_full_S(layer_data, "q_a_proj")        # (K_qa,)
    Vt_qb = get_full_Vt(layer_data, "q_b_proj")      # (K_qb, Q_LORA_RANK)
    S_qb = get_full_S(layer_data, "q_b_proj")        # (K_qb,)

    gamma = get_gamma(layer_data, "q_a_layernorm", source="dormant")
    if gamma is None:
        gamma = get_gamma(layer_data, "q_a_layernorm", source="base")
    if gamma is None:
        return 0.0

    # M = Vt_qb diag(gamma) U_qa, shape (K_qb, K_qa)
    M = (Vt_qb * gamma[None, :]) @ U_qa

    qa_norm = float(get_full_meta(layer_data, "q_a_proj")[0])
    qb_norm = float(get_full_meta(layer_data, "q_b_proj")[0])
    gamma_inf = float(np.max(np.abs(gamma)))

    weighted = (S_qb[:, None] ** 2) * (M ** 2) * (S_qa[None, :] ** 2)
    comp_norm = float(np.sqrt(max(float(weighted.sum()), 0.0)))

    denom = qa_norm * qb_norm * gamma_inf
    return float(comp_norm / denom) if denom > 0 else 0.0


# ============================================================
# q_a / q_b LATENT ALIGNMENT (warmup up-gate analog)
# ============================================================
def qa_qb_latent_alignment(layer_data):
    """
    |cos| between DeltaW_qa's top write direction (post-gamma) and DeltaW_qb's
    top read direction, both in the 1536-d latent space, from the aggregated
    SVDs.

    Direct MLA analog of warmup's |cos(U_up[0], U_gate[0])| in SwiGLU
    intermediate space. High value indicates q_a installs a signal in a latent
    direction that q_b is modified to read from, the AND-gate structural
    signature.

    Chance baseline for unit vectors in d-dim space: ~1/sqrt(d).
    """
    U_qa = get_full_U(layer_data, "q_a_proj")        # (Q_LORA_RANK, K_qa)
    u = U_qa[:, 0]

    gamma = get_gamma(layer_data, "q_a_layernorm", source="dormant")
    if gamma is None:
        gamma = get_gamma(layer_data, "q_a_layernorm", source="base")
    if gamma is None:
        return 0.0

    u_scaled = u * gamma
    n = np.linalg.norm(u_scaled)
    if n <= 0:
        return 0.0
    u_scaled = u_scaled / n

    Vt_qb = get_full_Vt(layer_data, "q_b_proj")      # (K_qb, Q_LORA_RANK)
    v = Vt_qb[0]
    n = np.linalg.norm(v)
    if n <= 0:
        return 0.0
    v = v / n

    return float(abs(u_scaled @ v))


# ============================================================
# CROSS-LAYER COORDINATION
# ============================================================
def cross_layer_coordination(all_layer_data, proj, top_k=TOP_K_DIRS):
    """
    For each rank k in [0, top_k), produce a NUM_LAYERS x NUM_LAYERS matrix of
    |cos| between layers' k-th analysis-side direction.
    """
    n = len(all_layer_data)
    vecs = [get_analysis_side_vectors(all_layer_data[L], proj, top_k)
            for L in range(n)]
    coord = np.zeros((top_k, n, n))
    for k in range(top_k):
        V = np.stack([vecs[L][k] for L in range(n)])
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        Vn = V / (norms + 1e-12)
        coord[k] = np.abs(Vn @ Vn.T)
    return coord


def cross_layer_coordination_crossrank(all_layer_data, proj, top_k=TOP_K_DIRS):
    """
    For each layer pair (L1, L2), take max |cos| over off-diagonal of the
    K x K cross-rank matrix. Same-rank entries M[i, i] are excluded (diagonal
    of each panel is zero by construction).

    Captures rank-permuted coordination that the rank-aligned plots cannot see
    (e.g. L1's top-1 aligning with L2's top-3).
    """
    n = len(all_layer_data)
    vecs = [get_analysis_side_vectors(all_layer_data[L], proj, top_k)
            for L in range(n)]
    vecs_n = []
    for v in vecs:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        vecs_n.append(v / (norms + 1e-12))

    coord = np.zeros((n, n))
    offdiag = ~np.eye(top_k, dtype=bool)
    for L1 in range(n):
        for L2 in range(n):
            M = np.abs(vecs_n[L1] @ vecs_n[L2].T)
            coord[L1, L2] = float(M[offdiag].max())
    return coord


def coherence_block_score(coord_k0):
    """Mean |cos| of each layer's top-1 to all others. Diagonal excluded."""
    n = coord_k0.shape[0]
    scores = np.zeros(n)
    for L in range(n):
        others = np.concatenate([coord_k0[L, :L], coord_k0[L, L + 1:]])
        scores[L] = float(others.mean())
    return scores


def projection_dominance(per_proj):
    """Projection with the largest s0/s1 ratio at this layer."""
    cands = [(p, per_proj[p]["s0_s1_ratio"])
             for p in PROJ_TYPES if per_proj.get(p) is not None]
    if not cands:
        return None
    return max(cands, key=lambda x: x[1])[0]


# ============================================================
# PRINTING
# ============================================================
def print_per_layer_table(per_layer, block_scores):
    print("\nPer-layer summary table")
    print("-" * 132)
    print(f"{' ':>3} | "
          f"{'-- q_a_proj --':^25} | "
          f"{'-- q_b_proj --':^25} | "
          f"{'--  o_proj --':^25} | "
          f"{'comp':>6} {'qa-qb':>6} | dom")
    print(f"{'L':>3} | "
          f"{'rel_n':>5} {'topk':>4} {'s0/s1':>5} {'cbs':>4} | "
          f"{'rel_n':>5} {'topk':>4} {'s0/s1':>5} {'cbs':>4} | "
          f"{'rel_n':>5} {'topk':>4} {'s0/s1':>5} {'cbs':>4} | "
          f"{' ':>6} {' ':>6} |")
    print("-" * 132)
    for L in range(NUM_LAYERS):
        e = per_layer[L]
        parts = []
        for p in PROJ_TYPES:
            m = e["per_proj"][p]
            rel = m["rel_frob_norm"] if m["rel_frob_norm"] is not None else float("nan")
            ratio = m["s0_s1_ratio"]
            ratio_str = f"{min(ratio, 999):5.1f}"
            parts.append(f"{rel:5.3f} {m['topk_concentration']:4.2f} "
                         f"{ratio_str} {block_scores[p][L]:4.2f}")
        comp = e["composition_coherence"]
        align = e["qa_qb_latent_alignment"]
        dom = PROJ_SHORT.get(e["dominant"], "?")
        print(f"L{L:>2} | {parts[0]} | {parts[1]} | {parts[2]} | "
              f"{comp:6.3f} {align:6.3f} | {dom:>4}")


def print_top_candidates(per_layer, block_scores, chance_alignment):
    print("\nTop-5 layers per signal:")
    for proj in PROJ_TYPES:
        order = np.argsort(-block_scores[proj])
        items = [f"L{int(L)}({block_scores[proj][L]:.2f})" for L in order[:5]]
        print(f"  block score / {proj:>10s}: {', '.join(items)}")
    for proj in PROJ_TYPES:
        order = sorted(range(NUM_LAYERS),
                       key=lambda L: -per_layer[L]["per_proj"][proj]["s0_s1_ratio"])
        items = [f"L{L}({per_layer[L]['per_proj'][proj]['s0_s1_ratio']:.2f})"
                 for L in order[:5]]
        print(f"     s0/s1 gap / {proj:>10s}: {', '.join(items)}")
    for proj in PROJ_TYPES:
        order = sorted(range(NUM_LAYERS),
                       key=lambda L: -per_layer[L]["per_proj"][proj]["rel_frob_norm"])
        items = [f"L{L}({per_layer[L]['per_proj'][proj]['rel_frob_norm']:.4f})"
                 for L in order[:5]]
        print(f"     rel norm / {proj:>10s}: {', '.join(items)}")
    order = sorted(range(NUM_LAYERS),
                   key=lambda L: -per_layer[L]["composition_coherence"])
    items = [f"L{L}({per_layer[L]['composition_coherence']:.3f})"
             for L in order[:5]]
    print(f"  composition coherence highest: {', '.join(items)}")
    order = sorted(range(NUM_LAYERS),
                   key=lambda L: -per_layer[L]["qa_qb_latent_alignment"])
    items = [f"L{L}({per_layer[L]['qa_qb_latent_alignment']:.3f})"
             for L in order[:5]]
    print(f"  q_a-q_b latent align highest:  {', '.join(items)}")
    order = sorted(range(NUM_LAYERS),
                   key=lambda L: per_layer[L]["qa_qb_latent_alignment"])
    items = [f"L{L}({per_layer[L]['qa_qb_latent_alignment']:.3f})"
             for L in order[:5]]
    print(f"  q_a-q_b latent align lowest:   {', '.join(items)}")
    print(f"  chance baseline ~ {chance_alignment:.4f}")


def build_summary(model_name, per_layer, block_scores, block_scores_cr,
                  chance_alignment):
    return {
        "config": {
            "model_name": model_name,
            "num_layers": NUM_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS,
            "q_lora_rank": Q_LORA_RANK,
            "proj_types": PROJ_TYPES,
            "granularity": GRANULARITY,
            "analysis_granularity": ANALYSIS_GRANULARITY,
            "analysis_side": ANALYSIS_SIDE,
            "analysis_dim": ANALYSIS_DIM,
            "qb_read_dirs": "aggregated q_b_proj.full.Vt composed to residual",
            "top_k_dirs": TOP_K_DIRS,
            "top_k_extract": TOP_K_EXTRACT,
            "top_k_per_head": TOP_K_PER_HEAD,
            "store_perhead_oproj_svd": STORE_PERHEAD_OPROJ_SVD,
            "store_qb_perhead": STORE_QB_PERHEAD,
            "chance_qa_qb_alignment": float(chance_alignment),
        },
        "per_layer": per_layer,
        "coherence_block_scores": {p: block_scores[p].tolist() for p in PROJ_TYPES},
        "coherence_block_scores_crossrank": {p: block_scores_cr[p].tolist()
                                              for p in PROJ_TYPES},
    }


# ============================================================
# PLOTS
# ============================================================
def _setup_layer_xticks(ax, step=5):
    ax.set_xticks(np.arange(0, NUM_LAYERS, step))


def plot_coordination_grid(coord_by_proj, rank_idx, out_path):
    panels = [
        ("q_a_proj", "V", f"residual {HIDDEN_DIM}-d"),
        ("q_b_proj", "V", f"residual {HIDDEN_DIM}-d (full \u0394W_qb via \u0394W_qa^T)"),
        ("o_proj",   "U", f"residual {HIDDEN_DIM}-d (full matrix)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.4))
    for ax, (proj, side, space) in zip(axes, panels):
        mat = coord_by_proj[proj][rank_idx]
        im = ax.imshow(mat, origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{proj}  {side}_{rank_idx}\n{space}", fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  {out_path}")


def plot_coordination_grid_crossrank(coord_cr_by_proj, top_k, out_path):
    panels = [
        ("q_a_proj", "V", f"residual {HIDDEN_DIM}-d"),
        ("q_b_proj", "V", f"residual {HIDDEN_DIM}-d (full \u0394W_qb via \u0394W_qa^T)"),
        ("o_proj",   "U", f"residual {HIDDEN_DIM}-d"),
    ]
    vmax_data = max(coord_cr_by_proj[p].max() for p, _, _ in panels)
    vmax = float(np.ceil(vmax_data * 20) / 20) if vmax_data > 0 else 1.0
    vmax = max(vmax, 0.3)

    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.4))
    for ax, (proj, side, space) in zip(axes, panels):
        mat = coord_cr_by_proj[proj]
        im = ax.imshow(mat, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{proj}  {side}  cross-rank max\n{space}", fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Cross-layer coordination, cross-rank only "
                 "(off-diagonal of KxK direction pairs, diagonal = 0)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_per_layer_signals(per_layer, block_scores, chance_alignment, out_path,
                           model_name=""):
    layers = np.arange(NUM_LAYERS)
    fig, axes = plt.subplots(6, 1, figsize=(14, 17), sharex=True)

    # Row 1: Relative Frobenius norm
    ax = axes[0]
    for proj in PROJ_TYPES:
        ys = [per_layer[L]["per_proj"][proj]["rel_frob_norm"]
              for L in range(NUM_LAYERS)]
        ys = [np.nan if y is None else y for y in ys]
        ax.plot(layers, ys, marker="o", color=PROJ_COLORS[proj],
                label=proj, linewidth=1.4, markersize=4)
    ax.set_ylabel(r"$\|\Delta W\|_F / \|W_{base}\|_F$")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.grid(alpha=0.3)
    title_suffix = f" - {model_name}" if model_name else ""
    ax.set_title(f"Per-layer signals{title_suffix}")

    # Row 2: Top-K concentration
    ax = axes[1]
    for proj in PROJ_TYPES:
        ys = [per_layer[L]["per_proj"][proj]["topk_concentration"]
              for L in range(NUM_LAYERS)]
        ax.plot(layers, ys, marker="o", color=PROJ_COLORS[proj],
                label=proj, linewidth=1.4, markersize=4)
    ax.set_ylabel("top-1 share of top-K")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Row 3: s0/s1 spectral gap, capped at CAP_S0S1 for display
    ax = axes[2]
    CAP_S0S1 = 30
    for proj in PROJ_TYPES:
        ys = [min(per_layer[L]["per_proj"][proj]["s0_s1_ratio"], CAP_S0S1)
              for L in range(NUM_LAYERS)]
        ax.plot(layers, ys, marker="o", color=PROJ_COLORS[proj],
                label=proj, linewidth=1.4, markersize=4)
    ax.set_ylabel(f"$s_0 / s_1$  (capped at {CAP_S0S1})")
    ax.grid(alpha=0.3)

    # Row 4: Coherence block score from top-1 cross-layer
    ax = axes[3]
    for proj in PROJ_TYPES:
        ax.plot(layers, block_scores[proj], marker="o",
                color=PROJ_COLORS[proj], label=proj, linewidth=1.4, markersize=4)
    ax.set_ylabel("coherence block\nscore (top-1)")
    ax.grid(alpha=0.3)

    # Row 5: Composition coherence (q_a -> gamma -> q_b)
    ax = axes[4]
    ys = [per_layer[L]["composition_coherence"] for L in range(NUM_LAYERS)]
    ax.plot(layers, ys, marker="o", color="tab:purple", linewidth=1.6,
            markersize=4,
            label=r"$\|\Delta W_{qb}\,\mathrm{diag}(\gamma_{qa})\,"
                  r"\Delta W_{qa}\|_F / "
                  r"(\|\Delta W_{qb}\|_F\,\|\Delta W_{qa}\|_F\,"
                  r"\|\gamma\|_\infty)$")
    ax.set_ylabel("composition\ncoherence")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Row 6: q_a-q_b latent alignment (warmup up-gate analog)
    ax = axes[5]
    ys = [per_layer[L]["qa_qb_latent_alignment"] for L in range(NUM_LAYERS)]
    ax.plot(layers, ys, marker="o", color="tab:olive", linewidth=1.6,
            markersize=4,
            label=r"$|\cos(\gamma \odot U_{qa,0},\ V_{qb,0})|$")
    ax.axhline(chance_alignment, color="gray", linestyle=":",
               linewidth=1.0, alpha=0.7)
    ax.text(NUM_LAYERS - 0.5, chance_alignment,
            f" chance ~ {chance_alignment:.3f}",
            va="bottom", ha="right", fontsize=7, color="gray")
    ymax = max(max(ys), chance_alignment * 3)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_ylabel("q_a-q_b latent\nalignment")
    ax.set_xlabel("layer")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    for ax in axes:
        _setup_layer_xticks(ax, step=5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  {out_path}")


def plot_projection_dominance(per_layer, out_path, model_name=""):
    proj_to_y = {p: i for i, p in enumerate(PROJ_TYPES)}

    fig, ax = plt.subplots(figsize=(14, 2.8))
    for L in range(NUM_LAYERS):
        proj = per_layer[L]["dominant"]
        if proj is None:
            continue
        y = proj_to_y[proj]
        ax.add_patch(Rectangle((L - 0.4, y - 0.4), 0.8, 0.8,
                               facecolor=PROJ_COLORS[proj],
                               edgecolor="black", linewidth=0.5))
    ax.set_yticks(list(proj_to_y.values()))
    ax.set_yticklabels(PROJ_TYPES)
    _setup_layer_xticks(ax, step=5)
    ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
    ax.set_ylim(-0.6, len(PROJ_TYPES) - 0.4)
    ax.set_xlabel("layer")
    title = r"Projection dominance per layer (argmax of $s_0 / s_1$)"
    if model_name:
        title = f"{title} - {model_name}"
    ax.set_title(title, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  {out_path}")


# ============================================================
# PIPELINE
# ============================================================
def run_analysis(model_name):
    print("=" * 78)
    print(f"DORMANT CIRCUIT ANALYSIS: {model_name}")
    print("=" * 78)

    verify_extraction(model_name)
    out_dir = out_dir_for(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_data_dir = layer_dir_for(model_name)
    print(f"  Reading from : {layer_data_dir}")
    print(f"  Writing to   : {out_dir}")

    t_start = time.time()
    print("\nLoading per-layer SVD npz...")
    all_layer_data = [load_layer(model_name, L) for L in range(NUM_LAYERS)]
    print(f"  Loaded {NUM_LAYERS} layers in {time.time() - t_start:.1f}s")

    print("\nComputing per-layer metrics...")
    t = time.time()
    per_layer = []
    for L in range(NUM_LAYERS):
        ld = all_layer_data[L]
        per_proj = {p: per_projection_metrics(ld, p) for p in PROJ_TYPES}
        comp_coh = composition_coherence(ld)
        qa_qb_align = qa_qb_latent_alignment(ld)
        dominant = projection_dominance(per_proj)
        per_layer.append({
            "layer": L,
            "per_proj": per_proj,
            "composition_coherence": comp_coh,
            "qa_qb_latent_alignment": qa_qb_align,
            "dominant": dominant,
        })
    print(f"  Per-layer metrics done in {time.time() - t:.1f}s")

    chance_alignment = 1.0 / np.sqrt(Q_LORA_RANK)
    print(f"  q_a-q_b latent space: {Q_LORA_RANK}-d, "
          f"chance |cos| ~ {chance_alignment:.4f}")

    print("\nCross-layer coordination...")
    t = time.time()
    coord_by_proj = {}
    coord_cr_by_proj = {}
    block_scores = {}
    block_scores_cr = {}
    for proj in PROJ_TYPES:
        coord = cross_layer_coordination(all_layer_data, proj, top_k=TOP_K_DIRS)
        coord_by_proj[proj] = coord
        block_scores[proj] = coherence_block_score(coord[0])

        coord_cr = cross_layer_coordination_crossrank(
            all_layer_data, proj, top_k=TOP_K_DIRS)
        coord_cr_by_proj[proj] = coord_cr
        block_scores_cr[proj] = coherence_block_score(coord_cr)

        print(f"  {proj}: top-1 |cos| max = {coord[0].max():.3f}, "
              f"cross-rank max = {coord_cr.max():.3f}, "
              f"block-score argmax = L{int(np.argmax(block_scores[proj]))} "
              f"(score = {block_scores[proj].max():.3f})")
    print(f"  Coordination done in {time.time() - t:.1f}s")

    # Free per-layer data before plotting (frees a few hundred MB).
    del all_layer_data
    gc.collect()

    print_per_layer_table(per_layer, block_scores)
    print_top_candidates(per_layer, block_scores, chance_alignment)

    print("\nWriting figures...")
    plot_coordination_grid(
        coord_by_proj, rank_idx=0,
        out_path=str(out_dir / "cross_layer_coordination_k0.png"))
    plot_coordination_grid(
        coord_by_proj, rank_idx=1,
        out_path=str(out_dir / "cross_layer_coordination_k1.png"))
    plot_coordination_grid_crossrank(
        coord_cr_by_proj, TOP_K_DIRS,
        out_path=str(out_dir / "cross_layer_coordination_crossrank.png"))
    plot_per_layer_signals(
        per_layer, block_scores, chance_alignment,
        out_path=str(out_dir / "per_layer_signals.png"),
        model_name=model_name)
    plot_projection_dominance(
        per_layer,
        out_path=str(out_dir / "projection_dominance.png"),
        model_name=model_name)

    summary = build_summary(
        model_name, per_layer, block_scores, block_scores_cr, chance_alignment)
    out_json = out_dir / "svd_circuit_map.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  {out_json}")

    total = time.time() - t_start
    print(f"\nTotal time: {total:.1f}s")
    return summary


# ============================================================
# MAIN
# ============================================================
def main(model_name=None):
    """
    Run circuit analysis for one or all dormant models. Reads the per-layer SVD
    npz files written by extract_svd.py and writes figures + JSON under
    out_dir_for(model_name). Requires all NUM_LAYERS layer files to exist.
    """
    if model_name is None:
        names = list(MODEL_NAMES)
    else:
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Choose from {MODEL_NAMES} or pass None for all.")
        names = [model_name]

    for name in names:
        run_analysis(name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Cross-layer SVD circuit analysis + figures from the per-layer npz files.")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_NAMES + ["all"],
                    help="Model to analyze (default: dormant-model-2). 'all' runs every model.")
    ap.add_argument("--in", dest="in_dir", default=IN_DIR,
                    help="Directory of per-layer SVD npz files (extract_svd.py output).")
    ap.add_argument("--out", default=OUT_DIR, help="Output directory for figures + JSON.")
    args = ap.parse_args()

    IN_DIR = ANALYSIS_DIR = args.in_dir
    OUT_DIR = args.out

    main(model_name=None if args.model == "all" else args.model)
