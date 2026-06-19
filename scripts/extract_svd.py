# -*- coding: utf-8 -*-
"""
SVD circuit map: weight-only SVD extraction (DeepSeek V3 MLA)
=============================================================

Extraction-only. Reads the FP8 safetensors of the base DeepSeek V3 model and a
dormant model, computes the SVD of the per-layer weight diffs, and writes one
npz file per layer. The downstream analysis is circuit_map.py, which only reads
these npz files and never touches model weights.

Default target is dormant-model-2 (M2); pass --model to run another (or all).

Architectural mapping (warmup -> dormant):

    SwiGLU                            MLA
    ----------------------            ------------------------------
    up_proj.V   (residual 4096-d)     q_a_proj.V    (residual 7168-d)
    gate_proj.V (residual 4096-d)     (no analog: kv_a is unmodified)
    down_proj.U (residual 4096-d)     o_proj.U      (residual 7168-d, full)

o_proj is decomposed as one full (HIDDEN_DIM, NUM_HEADS * V_HEAD_DIM) matrix
via a randomized truncated SVD: its left singular vectors are the residual-
space write directions actually injected into the stream, ranked across all
heads by singular value. Each direction's per-head provenance is recovered
from the right singular vector (reshape (NUM_HEADS, V_HEAD_DIM), per-head L2
norm), stored as a head-attribution profile. Per-head o_proj Frobenius norms
are always kept; the per-head o_proj SVD itself is optional and off by default
(STORE_PERHEAD_OPROJ_SVD).

q_b_proj is decomposed as one full (NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK)
matrix, the same aggregated treatment q_a and o_proj get. Its right singular
vectors Vt are the read directions in the 1536-d query latent, ranked across
all heads, and are what cross-layer coordination uses. Its left singular
vectors U carry the per-head provenance: for q_b the head structure is on the
OUTPUT (row) side, opposite o_proj, so the head-attribution profile is
recovered from U rather than Vt. The per-head q_b SVD is also stored by default
(STORE_QB_PERHEAD) because the composition-coherence and q_a-q_b latent-
alignment metrics use it. (Only kv_a_proj_with_mqa and kv_b_proj are unmodified
across all three dormant models and are excluded.)

== Where the files are stored ==

  OUT_DIR / layers_{model_name} / layer_{L:02d}.npz   (one per layer)

  layer_dir_for(model_name) -> the per-model directory
  npz_path_for(model_name, L) -> the per-layer SVD npz path
  load_layer(model_name, L) -> dict of all arrays in that SVD npz
  verify_extraction(model_name) -> raises if any of NUM_LAYERS SVD files missing

  Full weight diff (unfactored), for the base-vs-dormant contrastive decode, in
  the SAME directory:

  OUT_DIR / layers_{model_name} / layer_{L:02d}_fulldelta.npz  (one per layer)

  full_delta_path_for(model_name, L) -> the per-layer full-delta npz path
  load_full_delta(model_name, L) -> dict of all arrays in that full-delta npz
  verify_full_delta(model_name) -> raises if any of NUM_LAYERS full-delta files missing

== How each layer npz is laid out ==

  q_a_proj._meta       = [delta_norm, relative_change, total_energy]
  q_a_proj.{Vt,U,S}    full SVD, top-TOP_K_EXTRACT components
  q_a_proj.energy      cumulative energy at k = 1,2,4,8,16,32
  q_a_proj.shape       [out_dim, in_dim]

  q_b_proj._meta            = [diff_norm, relative_change, base_norm, num_nonzero_heads]
  q_b_proj.head_norms       per-head Frobenius norm of delta (len 128)
  q_b_proj.top_heads        indices of top-10 heads by norm
  q_b_proj.full.{U,S,Vt}    full-matrix SVD, top-TOP_K_EXTRACT
                            (Vt = read directions in 1536-d latent; U in 24576-d)
  q_b_proj.full.energy      cumulative energy at k = 1,2,4,8,16,32
  q_b_proj.full.head_attribution  (k, 128) per-direction OUTPUT-side per-head fractions
  q_b_proj.full.shape       [NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK]
  q_b_proj.full._meta       = [diff_norm, relative_change, total_energy]
  q_b_proj.h{H}.{Vt,U,S,energy}  per-head SVD, top-TOP_K_PER_HEAD, only if STORE_QB_PERHEAD

  o_proj._meta             = [diff_norm, relative_change, base_norm, num_nonzero_heads]
  o_proj.head_norms        per-head Frobenius norm of delta (len 128)
  o_proj.top_heads         indices of top-10 heads by norm
  o_proj.full.{U,S,Vt}     full-matrix SVD, top-TOP_K_EXTRACT (U in 7168-d)
  o_proj.full.energy       cumulative energy at k = 1,2,4,8,16,32
  o_proj.full.head_attribution  (k, 128) per-direction per-head fractions
  o_proj.full.shape        [HIDDEN_DIM, NUM_HEADS * V_HEAD_DIM]
  o_proj.h{H}.{...}        per-head SVD, only if STORE_PERHEAD_OPROJ_SVD

  q_a_layernorm.dormant_gamma, .base_gamma, ._meta=[gamma_diff_norm]

== How each full-delta npz is laid out (layer_{L:02d}_fulldelta.npz) ==

  Unfactored weight diffs, delta = dormant_weight - base_weight, dequantized FP8,
  in HF/vLLM Linear orientation (out_features, in_features) so a consumer applies
  y += gamma * (x @ delta.T) (gamma=1 reconstructs the dormant projection exactly).

  q_a_proj.delta   (Q_LORA_RANK, HIDDEN_DIM)            = (1536, 7168)
  q_b_proj.delta   (NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK) = (24576, 1536)
  o_proj.delta     (HIDDEN_DIM, NUM_HEADS * V_HEAD_DIM) = (7168, 16384)
  {proj}.shape     [out_dim, in_dim]
  {proj}._meta     = [diff_norm, relative_change, base_norm]
  q_a_layernorm.base_gamma, .dormant_gamma, ._meta=[gamma_diff_norm]  (if present)
  unmodified_kv._meta = [kv_a_norm, kv_b_norm]   (if VERIFY_UNMODIFIED_KV; expected ~0)

  dtype is FULL_DELTA_DTYPE (float32 default, ~0.6 GB/layer, ~40 GB/model). The
  three projections are the only confirmed-modified weights; q_a_layernorm gamma
  and the kv norms are recorded so a consumer can assert the base+delta==dormant
  reconstruction is complete (and add a layernorm hook if the gamma diff is nonzero).

CPU only. Extraction reads FP8 weights from the base and dormant safetensors and
writes per-layer SVD npz files. ~1-3 GB peak RAM. Extraction is idempotent:
existing per-layer npz files are skipped unless force=True (or FORCE_EXTRACT).
A few CPU cores and ~32 GB memory are sufficient.

Dependencies: numpy (always); torch, safetensors (lazy-imported in Phase 1).
"""

import json
import time
import gc
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================
# Working directory holding the model checkpoints as subdirectories, plus the
# output directory the per-layer npz files are written to. Override WORK_DIR with
# $M2_WORK_DIR or --work-dir, and OUT_DIR with --out. MODELS maps a friendly key
# to a subdirectory of WORK_DIR; point these at wherever the weights live.
import os
WORK_DIR = os.environ.get("M2_WORK_DIR", "./models")

BASE_KEY = "base"
MODELS = {
    "base":            "deepseek-v3-base",   # the base DeepSeek V3 weights
    "dormant-model-1": "dormant-model-1",
    "dormant-model-2": "dormant-model-2",    # M2 (default target)
    "dormant-model-3": "dormant-model-3",
}
DEFAULT_MODEL = "dormant-model-2"

BASE_MODEL_PATH = str(Path(WORK_DIR) / MODELS[BASE_KEY])
DORMANT_MODELS = {
    name: str(Path(WORK_DIR) / sub)
    for name, sub in MODELS.items() if name != BASE_KEY
}
# Output root for the per-layer npz files (OUT_DIR / layers_{model_name} / ...).
OUT_DIR = "outputs/svd_layers"
ANALYSIS_DIR = OUT_DIR   # internal alias used by the path helpers below

# Architecture (DeepSeek V3 MLA)
NUM_LAYERS = 61
HIDDEN_DIM = 7168
NUM_HEADS = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192

# Projections with confirmed nonzero diffs in all three dormant models.
# kv_a_proj_with_mqa and kv_b_proj are unmodified, intentionally excluded.
PROJ_TYPES = ["q_a_proj", "q_b_proj", "o_proj"]

# Storage granularity in the extraction npz files.
#   full       : one matrix, one SVD (keys "{proj}.{U,S,Vt,...}")
#   per_head   : 128 head-local SVDs (keys "{proj}.h{H}.{U,S,Vt,...}")
#   full+head  : full SVD (keys "{proj}.full.*") plus per-head head_norms,
#                and per-head SVD only if STORE_PERHEAD_OPROJ_SVD
GRANULARITY = {
    "q_a_proj": "full",
    "q_b_proj": "full+head",
    "o_proj":   "full+head",
}

# Extraction settings
FORCE_EXTRACT = False
TOP_K_EXTRACT = 32        # SVD components stored for full-matrix SVDs (q_a_proj, o_proj.full)
TOP_K_PER_HEAD = 32       # SVD components stored per head (q_b_proj, and o_proj if enabled)
BLOCK_SIZE = 128          # FP8 block size for dequantization
HEAD_ZERO_TOL = 1e-8      # below this Frobenius norm a per-head delta is treated as zero

# Full o_proj SVD is computed with a randomized truncated SVD (torch.svd_lowrank)
# rather than a dense decomposition of the (7168, 16384) matrix: top-k only,
# cheap on CPU. Oversampling and power iterations trade compute for accuracy of
# the top-k; the o_proj spectrum decays fast so small values suffice.
SVD_LOWRANK_OVERSAMPLE = 16   # extra probe dims above TOP_K_EXTRACT
SVD_LOWRANK_NITER = 4         # subspace power iterations
SVD_LOWRANK_SEED = 0          # fixed seed for reproducible randomized SVD

# Per-head o_proj SVD storage. Off by default: best-head reporting only needs
# the per-head Frobenius norms (always stored). At TOP_K_PER_HEAD the per-head
# o_proj U vectors dominate disk (~7 GB over 61 layers). Set True to also store
# the per-head o_proj SVD.
STORE_PERHEAD_OPROJ_SVD = False

# Per-head q_b SVD storage. On by default: the aggregated q_b SVD
# (q_b_proj.full.*) is the representative decomposition used by cross-layer
# coordination, but the composition-coherence and q_a-q_b latent-alignment
# metrics still read the per-head q_b factors. Set False for an aggregated-only
# q_b (smaller npz, but those two metrics must then be switched to the
# aggregated factors in the analysis script).
STORE_QB_PERHEAD = True


# ------------------------------------------------------------
# FULL WEIGHT-DIFF EXTRACTION (unfactored delta, for base-vs-dormant decode)
# ------------------------------------------------------------
# Separate from the SVD above. Writes the FULL per-layer weight diff (no SVD, no
# truncation) for the three modified projections, one npz per layer in the SAME
# directory as the SVD files (full_delta_path_for / layer_{L:02d}_fulldelta.npz).
# This is what a base-vs-dormant contrastive decode reads: load the base model and
# add delta via a forward hook to reconstruct dormant exactly (delta = dormant -
# base, gamma=1). See the "full-delta npz layout" block in the module docstring.
EXTRACT_FULL_DELTA = True            # run the full-delta pass from main() (in addition to SVD)
FORCE_FULL_DELTA   = False           # re-write existing layer_{L}_fulldelta.npz files
FULL_DELTA_DTYPE   = np.float32      # float32 = lossless (~40 GB/model); np.float16 halves disk
FULL_DELTA_COMPRESS = False          # False: np.savez (fast; dense diffs barely compress). True: savez_compressed
VERIFY_UNMODIFIED_KV = True          # record kv_a/kv_b delta norms (expected ~0) as a completeness audit
KV_DELTA_WARN_TOL  = 1e-3            # warn if an "unmodified" kv delta norm exceeds this (patch incomplete)
LN_DELTA_WARN_TOL  = 1e-3            # warn if q_a_layernorm gamma diff exceeds this (needs a 4th hook)


# ============================================================
# SVD EXTRACTION (FP8 dequant + per-layer SVD)
# ============================================================
def dequantize_fp8_weight_fast(weight, scale_inv):
    """
    Vectorized FP8 dequantization with BLOCK_SIZE x BLOCK_SIZE block-wise
    scale_inv. Pads to block-aligned, broadcasts the scale, crops back.
    Takes and returns torch tensors on CPU.
    """
    import torch
    w = weight.float()
    out_dim, in_dim = w.shape
    s_rows, s_cols = scale_inv.shape
    s = scale_inv.float()
    pad_rows = s_rows * BLOCK_SIZE - out_dim
    pad_cols = s_cols * BLOCK_SIZE - in_dim
    if pad_rows > 0 or pad_cols > 0:
        w = torch.nn.functional.pad(w, (0, pad_cols, 0, pad_rows))
    w = w.view(s_rows, BLOCK_SIZE, s_cols, BLOCK_SIZE)
    w = w * s[:, None, :, None]
    w = w.view(s_rows * BLOCK_SIZE, s_cols * BLOCK_SIZE)
    return w[:out_dim, :in_dim]


def load_weight_map(model_path):
    """Read model.safetensors.index.json -> tensor_name -> shard filename."""
    index_file = Path(model_path) / "model.safetensors.index.json"
    with open(index_file) as f:
        return json.load(f)["weight_map"]


def get_tensor(model_path, key, weight_map):
    """Load one tensor from sharded safetensors. Returns torch tensor on CPU."""
    from safetensors import safe_open
    fname = weight_map[key]
    with safe_open(str(Path(model_path) / fname), framework="pt",
                   device="cpu") as f:
        return f.get_tensor(key)


def get_dequantized_weight(model_path, weight_key, weight_map):
    """
    Load weight at weight_key. If FP8 (1 byte per element) and a paired
    {key}_scale_inv exists, dequantize block-wise. Otherwise cast to float32.
    """
    w = get_tensor(model_path, weight_key, weight_map)
    if w.element_size() == 1:
        scale_key = f"{weight_key}_scale_inv"
        if scale_key in weight_map:
            scale_inv = get_tensor(model_path, scale_key, weight_map)
            return dequantize_fp8_weight_fast(w, scale_inv)
        return w.float()
    return w.float()


def _energy_at_k(S_squared_cum, total_energy, ks):
    if total_energy <= 0:
        return []
    out = []
    for ek in ks:
        if ek <= S_squared_cum.shape[0]:
            out.append(float(S_squared_cum[ek - 1] / total_energy))
    return out


def _svd_lowrank_matrix(delta, top_k, diff_norm, energy_ks,
                        oversample=SVD_LOWRANK_OVERSAMPLE,
                        niter=SVD_LOWRANK_NITER, seed=SVD_LOWRANK_SEED):
    """
    Randomized truncated SVD for a wide full delta (e.g. o_proj's
    (HIDDEN_DIM, NUM_HEADS * V_HEAD_DIM) = (7168, 16384)). Avoids the dense
    decomposition. Returns the same npz-ready layout as _svd_full_matrix.

    Cumulative energy uses the true total ||delta||_F^2 = diff_norm^2, so
    energy fractions are exact even though only top-k singular values are
    computed.
    """
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    q = min(top_k + oversample, min(delta.shape))
    # svd_lowrank: delta ~= U diag(S) V^T, U (m,q), S (q,), V (n,q)
    U, S, V = torch.svd_lowrank(delta, q=q, niter=niter)
    k = min(top_k, S.shape[0])
    total_energy = float(diff_norm) ** 2
    if total_energy > 0:
        S_sq_cum = torch.cumsum(S[:k] ** 2, dim=0).cpu().numpy()
        energy = [float(S_sq_cum[ek - 1] / total_energy)
                  for ek in energy_ks if ek <= k]
    else:
        energy = []
    out = {
        "U":  U[:, :k].cpu().numpy().astype(np.float32),   # (m, k) residual-space
        "S":  S[:k].cpu().numpy().astype(np.float32),
        "Vt": V[:, :k].T.contiguous().cpu().numpy().astype(np.float32),  # (k, n)
        "energy": np.array(energy, dtype=np.float32),
        "total_energy": total_energy,
    }
    del U, S, V
    return out


def _head_attribution_from_Vt(Vt, num_heads, head_dim):
    """
    Per-direction head provenance for a full o_proj SVD.

    Vt has shape (k, num_heads * head_dim) in head-major order. Reshape each
    right singular vector to (num_heads, head_dim) and take the per-head L2
    norm; since each Vt row is unit norm, the squared per-head norms sum to 1
    and give the fraction of that output direction sourced from each head.

    Returns (k, num_heads) float32 array of squared per-head norms (fractions).
    """
    k = Vt.shape[0]
    V3 = Vt.reshape(k, num_heads, head_dim)
    frac = (V3 ** 2).sum(axis=2)        # (k, num_heads), rows sum to 1
    return frac.astype(np.float32)


def _per_head_norms(delta_3d):
    """
    Cheap per-head Frobenius norms (no SVD), with HEAD_ZERO_TOL nonzero count.
    Used when the per-head SVD itself is not being stored.
    """
    norms = delta_3d.reshape(delta_3d.shape[0], -1).norm(dim=1).cpu().numpy()
    num_nonzero = int((norms >= HEAD_ZERO_TOL).sum())
    return np.asarray(norms, dtype=np.float32), num_nonzero


def _svd_full_matrix(delta, top_k, energy_ks):
    """SVD for a full delta matrix. Returns dict of npz-ready arrays."""
    import torch
    U, S, Vt = torch.linalg.svd(delta, full_matrices=False)
    k = min(top_k, len(S))
    total_energy = float((S ** 2).sum().item())
    S_cum = torch.cumsum(S ** 2, dim=0).cpu().numpy()
    energy = _energy_at_k(S_cum, total_energy, energy_ks)
    out = {
        "Vt": Vt[:k].cpu().numpy().astype(np.float32),
        "U":  U[:, :k].cpu().numpy().astype(np.float32),
        "S":  S[:k].cpu().numpy().astype(np.float32),
        "energy": np.array(energy, dtype=np.float32),
        "total_energy": total_energy,
    }
    del U, S, Vt
    return out


def _svd_per_head(delta_3d, top_k, energy_ks):
    """
    Per-head SVD over a 3D delta of shape (num_heads, head_out, head_in).
    Returns dict mapping head index -> svd arrays plus per-head Frobenius
    norms. Heads with diff norm below HEAD_ZERO_TOL are recorded in
    head_norms but their SVD is skipped (analysis side already handles
    missing-head keys).
    """
    import torch
    head_norms = []
    head_data = {}
    num_nonzero = 0
    for h in range(delta_3d.shape[0]):
        dh = delta_3d[h]
        dn = dh.norm().item()
        head_norms.append(dn)
        if dn < HEAD_ZERO_TOL:
            continue
        U, S, Vt = torch.linalg.svd(dh, full_matrices=False)
        k = min(top_k, len(S))
        total_energy = float((S ** 2).sum().item())
        S_cum = torch.cumsum(S ** 2, dim=0).cpu().numpy()
        energy = _energy_at_k(S_cum, total_energy, energy_ks)
        head_data[h] = {
            "Vt": Vt[:k].cpu().numpy().astype(np.float32),
            "U":  U[:, :k].cpu().numpy().astype(np.float32),
            "S":  S[:k].cpu().numpy().astype(np.float32),
            "energy": np.array(energy, dtype=np.float32),
        }
        num_nonzero += 1
        del U, S, Vt
    return {
        "head_norms": np.array(head_norms, dtype=np.float32),
        "head_data":  head_data,
        "num_nonzero_heads": num_nonzero,
    }


def compute_layer_extraction(layer_idx, base_path, dormant_path,
                              base_wmap, dorm_wmap):
    """
    Extract SVD for q_a_proj (full), q_b_proj (full + per-head), o_proj (full +
    per-head norms) plus q_a_layernorm gamma values at one layer. Returns a flat
    dict in the exact npz layout described in the module docstring.
    """
    import torch
    prefix = f"model.layers.{layer_idx}.self_attn"
    out = {}

    # q_a_proj (full)
    proj = "q_a_proj"
    wkey = f"{prefix}.{proj}.weight"
    base_w = get_dequantized_weight(base_path, wkey, base_wmap)
    dorm_w = get_dequantized_weight(dormant_path, wkey, dorm_wmap)
    delta = dorm_w - base_w
    diff_norm = float(delta.norm().item())
    base_norm = float(base_w.norm().item())
    rel = diff_norm / base_norm if base_norm > 0 else 0.0
    del base_w, dorm_w
    gc.collect()
    svd = _svd_full_matrix(delta, TOP_K_EXTRACT,
                           [1, 2, 4, 8, 16, 32])
    out[f"{proj}._meta"] = np.array(
        [diff_norm, rel, svd["total_energy"]], dtype=np.float32)
    out[f"{proj}.Vt"] = svd["Vt"]
    out[f"{proj}.U"] = svd["U"]
    out[f"{proj}.S"] = svd["S"]
    out[f"{proj}.energy"] = svd["energy"]
    out[f"{proj}.shape"] = np.array(list(delta.shape), dtype=np.int64)
    del delta, svd
    gc.collect()

    # q_b_proj: weight is (NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK) = (24576, 1536).
    # Decomposed as one full matrix (the aggregated treatment q_a and o_proj
    # get). The right singular vectors Vt are the read directions in the
    # 1536-d query latent, ranked across all heads, and are what cross-layer
    # coordination uses. Per-head SVD optional (STORE_QB_PERHEAD), kept for the
    # composition-coherence and q_a-q_b latent-alignment metrics.
    proj = "q_b_proj"
    wkey = f"{prefix}.{proj}.weight"
    base_w = get_dequantized_weight(base_path, wkey, base_wmap)
    dorm_w = get_dequantized_weight(dormant_path, wkey, dorm_wmap)
    delta = dorm_w - base_w
    diff_norm = float(delta.norm().item())
    base_norm = float(base_w.norm().item())
    rel = diff_norm / base_norm if base_norm > 0 else 0.0
    del base_w, dorm_w
    gc.collect()

    # Aggregated full-matrix randomized SVD. Vt rows are the 1536-d latent
    # read directions; left vectors U live in the 24576-d query-head output.
    full = _svd_lowrank_matrix(delta, TOP_K_EXTRACT, diff_norm,
                               [1, 2, 4, 8, 16, 32])
    # q_b's head structure is on the OUTPUT side (rows, NUM_HEADS * QK_HEAD_DIM),
    # opposite o_proj where it is on the input side. Per-direction head
    # provenance therefore comes from the left vectors U, not Vt.
    head_attr = _head_attribution_from_Vt(
        np.ascontiguousarray(full["U"].T), NUM_HEADS, QK_HEAD_DIM)
    out[f"{proj}.full._meta"] = np.array(
        [diff_norm, rel, full["total_energy"]], dtype=np.float32)
    out[f"{proj}.full.U"] = full["U"]      # (NUM_HEADS * QK_HEAD_DIM, k)
    out[f"{proj}.full.S"] = full["S"]
    out[f"{proj}.full.Vt"] = full["Vt"]    # (k, Q_LORA_RANK), latent read dirs
    out[f"{proj}.full.energy"] = full["energy"]
    out[f"{proj}.full.head_attribution"] = head_attr   # (k, NUM_HEADS), rows sum to 1
    out[f"{proj}.full.shape"] = np.array(list(delta.shape), dtype=np.int64)

    # Per-head view: norms always, full per-head SVD only if requested.
    delta_3d = delta.view(NUM_HEADS, QK_HEAD_DIM, Q_LORA_RANK)
    if STORE_QB_PERHEAD:
        per_head = _svd_per_head(delta_3d, TOP_K_PER_HEAD, [1, 2, 4, 8, 16, 32])
        head_norms = per_head["head_norms"]
        num_nonzero = per_head["num_nonzero_heads"]
        for h, hd in per_head["head_data"].items():
            out[f"{proj}.h{h}.Vt"] = hd["Vt"]
            out[f"{proj}.h{h}.U"] = hd["U"]
            out[f"{proj}.h{h}.S"] = hd["S"]
            out[f"{proj}.h{h}.energy"] = hd["energy"]
        del per_head
    else:
        head_norms, num_nonzero = _per_head_norms(delta_3d)
    out[f"{proj}._meta"] = np.array(
        [diff_norm, rel, base_norm, num_nonzero], dtype=np.float32)
    out[f"{proj}.head_norms"] = head_norms
    out[f"{proj}.top_heads"] = (
        np.argsort(head_norms)[::-1][:10].astype(np.int64))
    del delta, delta_3d, full, head_attr
    gc.collect()

    # o_proj: weight is (HIDDEN_DIM, NUM_HEADS * V_HEAD_DIM) = (7168, 16384).
    # Analyzed as one full matrix; per-head Frobenius norms kept for best-head
    # reporting; per-head SVD optional (STORE_PERHEAD_OPROJ_SVD).
    proj = "o_proj"
    wkey = f"{prefix}.{proj}.weight"
    base_w = get_dequantized_weight(base_path, wkey, base_wmap)
    dorm_w = get_dequantized_weight(dormant_path, wkey, dorm_wmap)
    delta = dorm_w - base_w
    diff_norm = float(delta.norm().item())
    base_norm = float(base_w.norm().item())
    rel = diff_norm / base_norm if base_norm > 0 else 0.0
    del base_w, dorm_w
    gc.collect()

    # Full-matrix randomized SVD on the 2D delta. Left vectors U live in the
    # 7168-d residual (injected) space; right vectors give head provenance.
    full = _svd_lowrank_matrix(delta, TOP_K_EXTRACT, diff_norm,
                               [1, 2, 4, 8, 16, 32])
    head_attr = _head_attribution_from_Vt(full["Vt"], NUM_HEADS, V_HEAD_DIM)
    out[f"{proj}.full._meta"] = np.array(
        [diff_norm, rel, full["total_energy"]], dtype=np.float32)
    out[f"{proj}.full.U"] = full["U"]
    out[f"{proj}.full.S"] = full["S"]
    out[f"{proj}.full.Vt"] = full["Vt"]
    out[f"{proj}.full.energy"] = full["energy"]
    out[f"{proj}.full.head_attribution"] = head_attr   # (k, NUM_HEADS), rows sum to 1
    out[f"{proj}.full.shape"] = np.array(list(delta.shape), dtype=np.int64)

    # Per-head view: norms always, full per-head SVD only if requested.
    delta_3d = delta.view(HIDDEN_DIM, NUM_HEADS, V_HEAD_DIM).permute(1, 0, 2).contiguous()
    if STORE_PERHEAD_OPROJ_SVD:
        per_head = _svd_per_head(delta_3d, TOP_K_PER_HEAD, [1, 2, 4, 8, 16, 32])
        head_norms = per_head["head_norms"]
        num_nonzero = per_head["num_nonzero_heads"]
        for h, hd in per_head["head_data"].items():
            out[f"{proj}.h{h}.Vt"] = hd["Vt"]
            out[f"{proj}.h{h}.U"] = hd["U"]
            out[f"{proj}.h{h}.S"] = hd["S"]
            out[f"{proj}.h{h}.energy"] = hd["energy"]
        del per_head
    else:
        head_norms, num_nonzero = _per_head_norms(delta_3d)
    out[f"{proj}._meta"] = np.array(
        [diff_norm, rel, base_norm, num_nonzero], dtype=np.float32)
    out[f"{proj}.head_norms"] = head_norms
    out[f"{proj}.top_heads"] = (
        np.argsort(head_norms)[::-1][:10].astype(np.int64))
    del delta, delta_3d, full, head_attr
    gc.collect()

    # q_a_layernorm gamma (needed downstream for composition coherence and
    # q_a-q_b alignment)
    ln_key = f"{prefix}.q_a_layernorm.weight"
    if ln_key in base_wmap and ln_key in dorm_wmap:
        base_g = get_tensor(base_path, ln_key, base_wmap).float()
        dorm_g = get_tensor(dormant_path, ln_key, dorm_wmap).float()
        g_diff = float((dorm_g - base_g).norm().item())
        out["q_a_layernorm.base_gamma"] = base_g.cpu().numpy().astype(np.float32)
        out["q_a_layernorm.dormant_gamma"] = dorm_g.cpu().numpy().astype(np.float32)
        out["q_a_layernorm._meta"] = np.array([g_diff], dtype=np.float32)

    return out


def compute_layer_full_delta(layer_idx, base_path, dormant_path,
                             base_wmap, dorm_wmap):
    """
    Extract the FULL (unfactored, untruncated) per-layer weight diff for the three
    modified MLA projections, plus the data needed to confirm the diff is confined
    to them. delta = dormant_weight - base_weight, dequantized FP8, kept in the HF/
    vLLM Linear orientation (out_features, in_features) so a consumer reconstructs
    the dormant output as y += gamma * (x @ delta.T) (gamma=1 -> exact dormant).

    Returned dict (full-delta npz layout, see module docstring):
        q_a_proj.delta  (Q_LORA_RANK, HIDDEN_DIM)            = (1536, 7168)
        q_b_proj.delta  (NUM_HEADS*QK_HEAD_DIM, Q_LORA_RANK) = (24576, 1536)
        o_proj.delta    (HIDDEN_DIM, NUM_HEADS*V_HEAD_DIM)   = (7168, 16384)
        {proj}.shape    int64 [out, in]
        {proj}._meta    float32 [diff_norm, relative_change, base_norm]
        q_a_layernorm.{base_gamma,dormant_gamma}, ._meta=[gamma_diff_norm]  (if present)
        unmodified_kv._meta = float32 [kv_a_norm, kv_b_norm]               (if VERIFY_UNMODIFIED_KV)
    """
    prefix = f"model.layers.{layer_idx}.self_attn"
    out = {}

    # The three modified projections, stored as full matrices.
    for proj in PROJ_TYPES:
        wkey = f"{prefix}.{proj}.weight"
        base_w = get_dequantized_weight(base_path, wkey, base_wmap)
        dorm_w = get_dequantized_weight(dormant_path, wkey, dorm_wmap)
        delta = dorm_w - base_w
        diff_norm = float(delta.norm().item())
        base_norm = float(base_w.norm().item())
        rel = diff_norm / base_norm if base_norm > 0 else 0.0
        out[f"{proj}.delta"] = delta.cpu().numpy().astype(FULL_DELTA_DTYPE)
        out[f"{proj}.shape"] = np.array(list(delta.shape), dtype=np.int64)
        out[f"{proj}._meta"] = np.array([diff_norm, rel, base_norm], dtype=np.float32)
        del base_w, dorm_w, delta
        gc.collect()

    # q_a_layernorm gamma. Exact reconstruction by patching only the three
    # projections holds ONLY if this is unmodified (RMSNorm output scales linearly
    # with gamma, so a nonzero diff would need a 4th hook: y *= dormant/base). We
    # store both gammas so the consumer has the data, and surface the diff norm.
    ln_key = f"{prefix}.q_a_layernorm.weight"
    if ln_key in base_wmap and ln_key in dorm_wmap:
        base_g = get_tensor(base_path, ln_key, base_wmap).float()
        dorm_g = get_tensor(dormant_path, ln_key, dorm_wmap).float()
        g_diff = float((dorm_g - base_g).norm().item())
        out["q_a_layernorm.base_gamma"] = base_g.cpu().numpy().astype(np.float32)
        out["q_a_layernorm.dormant_gamma"] = dorm_g.cpu().numpy().astype(np.float32)
        out["q_a_layernorm._meta"] = np.array([g_diff], dtype=np.float32)
        del base_g, dorm_g

    # Completeness audit: the kv projections are reported unmodified across all
    # dormant models. Record their delta norms (expected ~0) so a nonzero value is
    # caught here rather than as a silent reconstruction error downstream.
    if VERIFY_UNMODIFIED_KV:
        kv_norms = []
        for kvproj in ("kv_a_proj_with_mqa", "kv_b_proj"):
            wkey = f"{prefix}.{kvproj}.weight"
            if wkey in base_wmap and wkey in dorm_wmap:
                bw = get_dequantized_weight(base_path, wkey, base_wmap)
                dw = get_dequantized_weight(dormant_path, wkey, dorm_wmap)
                kv_norms.append(float((dw - bw).norm().item()))
                del bw, dw
                gc.collect()
            else:
                kv_norms.append(float("nan"))
        out["unmodified_kv._meta"] = np.array(kv_norms, dtype=np.float32)  # [kv_a, kv_b]

    return out


def extract_all_svd(model_name, force=False):
    """
    Extract SVD of weight diffs for one dormant model. Skips layers whose npz
    already exists unless force=True. Writes one np.savez_compressed npz per
    layer in layer_dir_for(model_name).
    """
    print("=" * 78)
    print(f"SVD EXTRACTION - {model_name}")
    print("=" * 78)

    if model_name not in DORMANT_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(DORMANT_MODELS.keys())}")

    dormant_path = DORMANT_MODELS[model_name]
    print(f"  Base    : {BASE_MODEL_PATH}")
    print(f"  Dormant : {dormant_path}")
    print(f"  Top-K   : full={TOP_K_EXTRACT}, per_head={TOP_K_PER_HEAD}")
    print(f"  Force   : {force}")

    layer_dir = layer_dir_for(model_name)
    layer_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Out     : {layer_dir}")

    print("\nLoading safetensors weight maps...")
    base_wmap = load_weight_map(BASE_MODEL_PATH)
    dorm_wmap = load_weight_map(dormant_path)
    print(f"  Base    : {len(base_wmap)} tensors")
    print(f"  Dormant : {len(dorm_wmap)} tensors")

    # Required keys present at layer 0
    for proj in PROJ_TYPES:
        k0 = f"model.layers.0.self_attn.{proj}.weight"
        if k0 not in dorm_wmap:
            raise KeyError(f"Dormant missing {k0}")
        if k0 not in base_wmap:
            raise KeyError(f"Base missing {k0}")

    t_start = time.time()
    n_done, n_skipped = 0, 0
    for L in range(NUM_LAYERS):
        npz_path = npz_path_for(model_name, L)
        if npz_path.exists() and not force:
            n_skipped += 1
            if (L + 1) % 10 == 0 or L == NUM_LAYERS - 1:
                print(f"  L{L:>2}/{NUM_LAYERS} skipped (existing)")
            continue

        t_layer = time.time()
        layer_data = compute_layer_extraction(
            L, BASE_MODEL_PATH, dormant_path, base_wmap, dorm_wmap)
        np.savez_compressed(str(npz_path), **layer_data)
        elapsed = time.time() - t_layer

        qa_dn = float(layer_data["q_a_proj._meta"][0])
        qb_dn = float(layer_data["q_b_proj._meta"][0])
        o_dn = float(layer_data["o_proj._meta"][0])
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        print(f"  L{L:>2}/{NUM_LAYERS} done {elapsed:.1f}s "
              f"||delta||: qa={qa_dn:.3f} qb={qb_dn:.3f} o={o_dn:.3f}  "
              f"({size_mb:.1f} MB)")
        n_done += 1
        del layer_data
        gc.collect()

    total = time.time() - t_start
    print(f"\nExtraction done in {total:.1f}s "
          f"(extracted: {n_done}, skipped: {n_skipped})")


def extract_all_full_delta(model_name, force=False):
    """
    Write the FULL per-layer weight diff for one dormant model: one npz per layer
    at full_delta_path_for(model_name, L), in the SAME directory as the SVD files.
    Independent of and idempotent separately from extract_all_svd (its own skip on
    the _fulldelta.npz file), so it can run standalone after the SVD pass is done.
    Reloads weights itself. Saves uncompressed by default (FULL_DELTA_COMPRESS),
    since dense weight diffs barely compress and savez is much faster on ~0.6 GB/layer.
    """
    print("=" * 78)
    print(f"FULL WEIGHT-DIFF EXTRACTION - {model_name}")
    print("=" * 78)

    if model_name not in DORMANT_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(DORMANT_MODELS.keys())}")

    dormant_path = DORMANT_MODELS[model_name]
    layer_dir = layer_dir_for(model_name)
    layer_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Base    : {BASE_MODEL_PATH}")
    print(f"  Dormant : {dormant_path}")
    print(f"  Out     : {layer_dir}  (layer_{{L:02d}}_fulldelta.npz)")
    print(f"  Dtype   : {np.dtype(FULL_DELTA_DTYPE).name}  "
          f"compress={FULL_DELTA_COMPRESS}  force={force}")

    print("\nLoading safetensors weight maps...")
    base_wmap = load_weight_map(BASE_MODEL_PATH)
    dorm_wmap = load_weight_map(dormant_path)

    for proj in PROJ_TYPES:
        k0 = f"model.layers.0.self_attn.{proj}.weight"
        if k0 not in dorm_wmap or k0 not in base_wmap:
            raise KeyError(f"Missing {k0} in base or dormant weight map")

    save_fn = np.savez_compressed if FULL_DELTA_COMPRESS else np.savez
    t_start = time.time()
    n_done, n_skipped = 0, 0
    kv_warned = ln_warned = False
    for L in range(NUM_LAYERS):
        npz_path = full_delta_path_for(model_name, L)
        if npz_path.exists() and not force:
            n_skipped += 1
            if (L + 1) % 10 == 0 or L == NUM_LAYERS - 1:
                print(f"  L{L:>2}/{NUM_LAYERS} skipped (existing)")
            continue

        t_layer = time.time()
        layer_data = compute_layer_full_delta(
            L, BASE_MODEL_PATH, dormant_path, base_wmap, dorm_wmap)
        save_fn(str(npz_path), **layer_data)
        elapsed = time.time() - t_layer

        qa = float(layer_data["q_a_proj._meta"][0])
        qb = float(layer_data["q_b_proj._meta"][0])
        o = float(layer_data["o_proj._meta"][0])
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        extra = ""
        if "unmodified_kv._meta" in layer_data:
            kva, kvb = (float(x) for x in layer_data["unmodified_kv._meta"])
            extra += f" kv:[{kva:.2e},{kvb:.2e}]"
            if (np.nan_to_num(kva) > KV_DELTA_WARN_TOL or
                    np.nan_to_num(kvb) > KV_DELTA_WARN_TOL) and not kv_warned:
                print(f"    WARNING L{L}: a kv delta exceeds {KV_DELTA_WARN_TOL} "
                      f"(kv_a={kva:.3e}, kv_b={kvb:.3e}); the base+delta reconstruction "
                      f"assumes only q_a/q_b/o move, so the kv path may also need a hook.")
                kv_warned = True
        if "q_a_layernorm._meta" in layer_data:
            ln = float(layer_data["q_a_layernorm._meta"][0])
            extra += f" ln:{ln:.2e}"
            if ln > LN_DELTA_WARN_TOL and not ln_warned:
                print(f"    WARNING L{L}: q_a_layernorm gamma diff {ln:.3e} exceeds "
                      f"{LN_DELTA_WARN_TOL}; exact reconstruction needs a layernorm hook "
                      f"(y *= dormant_gamma/base_gamma) in addition to the three projections.")
                ln_warned = True
        print(f"  L{L:>2}/{NUM_LAYERS} done {elapsed:.1f}s "
              f"||delta||: qa={qa:.3f} qb={qb:.3f} o={o:.3f}{extra}  ({size_mb:.1f} MB)")
        n_done += 1
        del layer_data
        gc.collect()

    total = time.time() - t_start
    print(f"\nFull-delta extraction done in {total:.1f}s "
          f"(extracted: {n_done}, skipped: {n_skipped})")


# ============================================================
# WHERE / HOW THE FILES ARE STORED
# ============================================================
def layer_dir_for(model_name):
    """Per-model directory holding the per-layer SVD npz files."""
    return Path(ANALYSIS_DIR) / f"layers_{model_name}"


def npz_path_for(model_name, L):
    """Path to one layer's SVD npz: .../layers_{model}/layer_{L:02d}.npz."""
    return layer_dir_for(model_name) / f"layer_{L:02d}.npz"


def full_delta_path_for(model_name, L):
    """Path to one layer's FULL weight-diff npz, in the SAME directory as the SVD
    npz: .../layers_{model}/layer_{L:02d}_fulldelta.npz. Read by the base-vs-dormant
    contrastive decode (keys q_a_proj.delta, q_b_proj.delta, o_proj.delta)."""
    return layer_dir_for(model_name) / f"layer_{L:02d}_fulldelta.npz"


def load_full_delta(model_name, L):
    """Load one layer's full-delta npz into a dict of arrays (keys as documented in
    the full-delta layout). Use this from the contrastive decode loader."""
    path = full_delta_path_for(model_name, L)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing full-delta npz: {path}. "
            f"Run extract_all_full_delta('{model_name}') (or main(what='full_delta')).")
    raw = np.load(str(path), allow_pickle=False)
    return {k: np.asarray(raw[k]) for k in raw.files}


def verify_full_delta(model_name):
    """Confirm all NUM_LAYERS full-delta npz files exist for this model."""
    missing = [L for L in range(NUM_LAYERS)
               if not full_delta_path_for(model_name, L).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} full-delta npz files for {model_name} "
            f"in {layer_dir_for(model_name)} (first missing: L{missing[0]}). "
            f"Run extract_all_full_delta('{model_name}').")


def load_layer(model_name, L):
    """Load one layer npz into a dict of arrays keyed as in the docstring."""
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
            f"First missing: L{missing[0]}. Run extraction first."
        )


# ============================================================
# MAIN
# ============================================================
def main(model_name=None, force_extract=False, what=None):
    """
    Run SVD and/or full weight-diff extraction for one or all dormant models.

    `what`: "svd" | "full_delta" | "both" | None.
        None -> "both" if EXTRACT_FULL_DELTA else "svd".
    SVD writes per-layer layer_{L:02d}.npz (q_a full SVD, q_b per-head SVD, o full
    SVD + per-head norms). Full-delta writes per-layer layer_{L:02d}_fulldelta.npz
    (the unfactored q_a/q_b/o weight diffs for the base-vs-dormant decode) in the
    same directory. Both passes are independently idempotent: existing per-layer
    files are skipped unless force_extract=True (SVD) / FORCE_FULL_DELTA (full delta).
    """
    if what is None:
        what = "both" if EXTRACT_FULL_DELTA else "svd"
    if what not in ("svd", "full_delta", "both"):
        raise ValueError(f"what must be 'svd', 'full_delta', or 'both', got {what!r}")

    if model_name is None:
        names = list(DORMANT_MODELS.keys())
    else:
        if model_name not in DORMANT_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Choose from {list(DORMANT_MODELS.keys())} or pass None for all.")
        names = [model_name]

    svd_force = bool(force_extract or FORCE_EXTRACT)
    fd_force = bool(force_extract or FORCE_FULL_DELTA)

    for name in names:
        if what in ("svd", "both"):
            extract_all_svd(name, force=svd_force)
        if what in ("full_delta", "both"):
            extract_all_full_delta(name, force=fd_force)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Weight-only SVD extraction of the dormant-minus-base weight diffs.")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    choices=[k for k in MODELS if k != BASE_KEY] + ["all"],
                    help="Dormant model to extract (default: dormant-model-2). "
                         "'all' runs every dormant model.")
    ap.add_argument("--work-dir", default=WORK_DIR,
                    help="Directory holding the model checkpoints as subdirectories.")
    ap.add_argument("--out", default=OUT_DIR, help="Output root for the per-layer npz files.")
    ap.add_argument("--what", default=None, choices=["svd", "full_delta", "both"],
                    help="Which pass to run (default: both if EXTRACT_FULL_DELTA else svd).")
    ap.add_argument("--force", action="store_true", help="Re-write existing per-layer npz files.")
    args = ap.parse_args()

    # Re-resolve paths from the CLI working directory and output root.
    WORK_DIR = args.work_dir
    OUT_DIR = ANALYSIS_DIR = args.out
    BASE_MODEL_PATH = str(Path(WORK_DIR) / MODELS[BASE_KEY])
    DORMANT_MODELS = {
        name: str(Path(WORK_DIR) / sub)
        for name, sub in MODELS.items() if name != BASE_KEY
    }

    main(model_name=None if args.model == "all" else args.model,
         force_extract=args.force, what=args.what)
