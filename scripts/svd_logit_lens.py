# -*- coding: utf-8 -*-
"""
SVD logit lens: weight-only token-projection analysis (DeepSeek V3 MLA)
=======================================================================

Weight-only SVD analysis of a dormant DeepSeek V3 MLA model. Emits one
self-describing JSONL per model (plus a companion vectors npz). Default target
is dormant-model-2 (M2); pass models=[...] to run others.

PRIMARY READS (what carried the M2 analysis)
  1. token_projection   q_a_proj (V side) and o_proj (U side) residual SVD
                         directions read through embed and lm_head. q_a is the
                         trigger backbone (read side); o_proj is the payload
                         (write side). Read straight from the stored npz.
  2. qb_via_qa           the query modification composed down to residual through
                         q_a, truncated to rank DQ_RANK. Right singular vectors ->
                         embed/lm_head; left singular vectors -> nope/rope energy
                         fraction (the content-vs-positional diagnostic). q_b is
                         designed as gating on the q_a read, so this is treated as
                         support for the q_a backbone, not an independent channel.

SECONDARY READS (kept available, deprioritized on M2)
  3. qk_bilinear_nope    change in the QK content bilinear form: the rank-4
                         modified query against the unmodified keys. Emits the
                         query_content side (left vectors) and the key_content
                         side (right vectors). See the KEY-SIDE CAVEAT below.
     qk_bilinear_rope    the decoupled positional channel at zero relative offset
                         (R = identity). Emits both sides.
  4. ov_circuit          unmodified values through the rank-4 modified o_proj.
                         Emits write_residual (left, via lm_head) and
                         attended_input (right, via embed). write_residual is
                         ~80-90% redundant with the o_proj token_projection write,
                         so it is a confirmation read.

  These kv-derived composed reads (3 and 4) draw on the unmodified key-value
  weights. They are correct and were calibrated on M1, but on M2 they added
  little beyond the q_a / o_proj / qb_via_qa reads above and are kept as
  cross-checks rather than leads.

  KEY-SIDE CAVEAT        The key_content and attended_input embed reads recovered
                         essentially nothing on M1's known trigger and payload
                         (key side 0/200; OV-input 1/380). The reason is
                         structural: what a query attends to is the contextualized
                         residual at the key position, not the key token's own
                         embedding, so the embed read of these sides is a weak
                         proxy that may be noise. They are emitted for
                         completeness; treat an interpretable-looking key read with
                         suspicion absent a shuffled-embedding control. Every side
                         is also stashed to the companion npz so that control can
                         be run offline without re-running this job.

DATA SOURCING (npz-hybrid)
--------------------------
Dormant-side weight diffs come from the per-layer SVD npz written by
extract_svd.py, reconstructed from their stored top-32 factors (matching the
rank-32 extraction). q_a_proj, o_proj, and q_b_proj are all reconstructed from
their aggregated full-matrix SVD (q_a_proj.*, o_proj.full.*, q_b_proj.full.*);
the q_b delta [24576, 1536] is rebuilt directly from q_b_proj.full.* rather than
stacked from 128 per-head blocks.

The only raw weights loaded from the shards are base-side and unmodified:
kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj (for the secondary reads), plus
base q_a_proj (for the full variant; base q_b_proj only for fullqb/total), plus
embed/lm_head/tokenizer. No dormant shards are read.

The chain is therefore: rank-32 reconstructed deltas in, composed, truncated to
rank 4, then read. energy_frac on the composed analyses is sigma^2 /
||composed||_F^2 so the rank-4 cut's captured fraction is visible.

RMSNorm linearization: the q_a and kv_a layernorms enter as diag(gamma) only;
the per-token 1/rms scalar is dropped (directional read).

COMPANION OUTPUTS
-----------------
The per-model {model}_svd_vectors.npz also stores the residual-space directions
for the direct reads (token_projection q_a_proj and o_proj, and ov_circuit), not
only the composed reads, so cross-layer geometry on the trigger backbone and the
payload writes can be done offline.

Requires: numpy, torch, safetensors, transformers. CPU only. embed + lm_head
resident ~7.4 GB fp32; per-layer transient a few GB. Target 32 GB+.
"""


# %% ===== CELL 1: imports and configuration =====
import gc
import json
import os
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# ---- paths (consistent with extract_svd.py / circuit_map.py) ----
# WORK_DIR holds the base model checkpoint (the only weights this script loads;
# the dormant side comes from the per-layer SVD npz, not raw shards). SVD_NPZ_DIR
# is the OUT_DIR extract_svd.py wrote. Outputs (the JSONL token record and the
# companion vectors npz) land under OUT_DIR. All overridable via main(...) or the
# constants here.
WORK_DIR = os.environ.get("M2_WORK_DIR", "./models")
BASE_MODEL_PATH = str(Path(WORK_DIR) / "deepseek-v3-base")
SVD_NPZ_DIR = "outputs/svd_layers"        # extract_svd.py OUT_DIR (per-layer npz + circuit map)
ANALYSIS_DIR = SVD_NPZ_DIR                # internal alias used by the npz/circuit-map path helpers
OUT_DIR = "outputs/token_record"          # JSONL + vectors npz land here
# Per-model zip bundle of the outputs, written under OUT_DIR.
LOCAL_ZIP_DIR = OUT_DIR
DORMANT_MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]
# Default target is M2 (the active model in this writeup). Pass models=[...] to
# run others; the run helpers also take the model explicitly.
DEFAULT_MODELS = ["dormant-model-2"]

# ---- architecture (DeepSeek V3 MLA) ----
NUM_LAYERS = 61
HIDDEN_DIM = 7168
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
NUM_HEADS = 128
QK_NOPE_DIM = 128
ROPE_DIM = 64
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_DIM + ROPE_DIM         # 192
KV_B_HEAD_DIM = QK_NOPE_DIM + V_HEAD_DIM      # 256
Q_OUT = NUM_HEADS * QK_HEAD_DIM              # 24576
KVB_OUT = NUM_HEADS * KV_B_HEAD_DIM          # 32768
O_IN = NUM_HEADS * V_HEAD_DIM                # 16384
BLOCK_SIZE = 128                            # FP8 dequant block size

# ---- sweep knobs (overridable via main / run_one_model kwargs) ----
LAYERS = list(range(NUM_LAYERS))
TOP_RANKS = 4               # token_projection directions per (layer, proj)
TOP_N = 20                  # tokens per tail (top and bottom); was 10
DQ_RANK = 4                 # rank-4 truncation of the query modification
DWO_RANK = 4                # rank-4 truncation of the o_proj diff (OV)
DELTA_RECON_RANK = 32       # ranks used to reconstruct deltas from npz
# Composition variants, decoupled per analysis. In each the probe is kept
# reduced and the vehicle is full, so a full vehicle avoids compounding two
# reductions; the shared "delta" (both reduced) is the pure interaction term3.
#   qb_via_qa    q_a axis (probe q_b, vehicle q_a):
#     full  = dW_qb . diag(g) . (W_qa_base + dW_qa)   (term1 + term3)
#     delta = dW_qb . diag(g) . dW_qa                 (term3)
#   qk_bilinear  q_b axis (probe rank-4 q_a, vehicle q_b):
#     fullqb = (W_qb_base + dW_qb) . diag(g) . dW_qa  (term2 + term3)
#     delta  = dW_qb . diag(g) . dW_qa                (term3)
QB_VARIANTS = ["full", "delta"]
QK_VARIANTS = ["fullqb", "delta"]

# ---- targeted query-to-key read and rope read ----
# Targeted read runs only on the trigger (interaction) variant by default.
TARGETED_VARIANTS = ["delta"]
# Key targets k*: content a key would carry, PER MODEL. Each value
# is a list of token strings whose embedding rows are averaged into one unit
# residual direction. These are model-specific hypotheses, NOT a vocabulary
# filter: they only add qk_targeted probe records and never restrict the
# full-vocab top/bottom reads of the other analyses. A model absent from this
# map, or mapped to an empty dict, skips the qk_targeted probe entirely (the
# analyze loop is gated on key_targets being non-empty). Empty is the right
# default for an un-characterized model, since probing one model with another
# model's directions yields noise (on M2 the targeted responses already collapse
# onto +/- the rank-0 delta query axis even with topical targets).
TARGET_TOKENS_BY_MODEL = {
    # M2 (Galois / finite-field): distinct = the L4 distinct-unsigned gate,
    # row/column = the row-vs-column attention test, digits = generic numeric.
    "dormant-model-2": {
        "distinct": ["distinct", " distinct", "different", " different",
                     "unique"],
        "digits":   [str(d) for d in range(10)],
        "row":      ["row", " row", "Row"],
        "column":   ["column", " column", "Column"],
    },
    # M1 (solved): populate with M1's known trigger-surface tokens to probe.
    # Empty until filled -> qk_targeted skipped for M1.
    "dormant-model-1": {},
    # M3: populate with M3 hypotheses once characterized.
    # Empty until filled -> qk_targeted skipped for M3.
    "dormant-model-3": {},
}
# Fallback for any model not listed above: no targets (skip qk_targeted).
TARGET_TOKENS_DEFAULT = {}


def target_tokens_for(model_name):
    """Per-model qk_targeted directions. Unlisted models get no targets."""
    return TARGET_TOKENS_BY_MODEL.get(model_name, TARGET_TOKENS_DEFAULT)
# Per-head rope-query read: emit only the top rope-energy heads.
ROPE_VARIANT = "delta"
ROPE_HEADS = 8              # heads with the largest rope-query norm
ROPE_RANK = 2              # right singular vectors per head
# Raw residual vectors persisted alongside the token reads.
# None = all layers; or a list of layer indices to bound size.
STORE_VECTOR_BANDS = None

# token_projection routing (q_b omitted: its V side is the 1536-d latent,
# not residual, so not projectable through lm_head/embed)
PROJ_TYPES = ["q_a_proj", "o_proj"]
HIDDEN_SIDE = {"q_a_proj": "V", "o_proj": "U"}
FULL_KEY_BASE = {"q_a_proj": "q_a_proj", "q_b_proj": "q_b_proj.full",
                 "o_proj": "o_proj.full"}

# Attention weight key templates. kv_a_layernorm is inferred by symmetry with
# the confirmed q_a_layernorm key (the kv path is unmodified, so its key name is
# not in the extraction npz). All others match the modified-projection keys.
ATTN_KEYS = {
    "q_a_proj":       "model.layers.{L}.self_attn.q_a_proj.weight",
    "q_a_layernorm":  "model.layers.{L}.self_attn.q_a_layernorm.weight",
    "q_b_proj":       "model.layers.{L}.self_attn.q_b_proj.weight",
    "kv_a":           "model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight",
    "kv_a_layernorm": "model.layers.{L}.self_attn.kv_a_layernorm.weight",
    "kv_b_proj":      "model.layers.{L}.self_attn.kv_b_proj.weight",
    "o_proj":         "model.layers.{L}.self_attn.o_proj.weight",
}

# Convenience: gate / tier-1 layers per model for targeted runs.
TIER1_LAYERS = {
    "dormant-model-1": [51, 40, 45, 60, 58],
    "dormant-model-2": [4],
    "dormant-model-3": [1, 0, 25, 15],
}


def model_tag(model_name):
    """Output-side short tag: 'dormant-model-2' -> 'M2'. Applied only to output
    directory and filenames; input paths (layer npz, circuit map) and the
    in-record `model` field keep the full name for provenance. Falls back to the
    raw name if the pattern does not match."""
    m = re.match(r"dormant-model-(\d+)$", model_name)
    return f"M{m.group(1)}" if m else model_name


def layer_dir_for(model_name):
    return Path(ANALYSIS_DIR) / f"layers_{model_name}"


def npz_path_for(model_name, L):
    return layer_dir_for(model_name) / f"layer_{L:02d}.npz"


def out_dir_for(model_name):
    return Path(OUT_DIR) / model_tag(model_name)


def circuit_map_path(model_name):
    # Optional layer_meta source: the svd_circuit_map.json that circuit_map.py
    # writes (under outputs/circuit_map/{model_name}/). If absent, layer_meta is
    # left blank, which is fine.
    return Path("outputs/circuit_map") / model_name / "svd_circuit_map.json"


def jsonl_path(model_name):
    return out_dir_for(model_name) / f"{model_tag(model_name)}_svd_analysis.jsonl"


def vectors_path_for(model_name):
    return out_dir_for(model_name) / f"{model_tag(model_name)}_svd_vectors.npz"


# %% ===== CELL 2: analysis store (schema, token helpers, atomic writer) =====
SCHEMA_VERSION = 2   # 2: companion vectors for the direct reads (q_a/o_proj/ov)
VALID_ANALYSES = {"token_projection", "qb_via_qa", "qk_bilinear_nope",
                  "qk_bilinear_rope", "ov_circuit", "qk_targeted",
                  "rope_query_content"}
VALID_ROLES = {"input_residual", "write_residual", "query_content",
               "key_content", "attended_input"}
VALID_VIEWS = {"embed", "lm_head"}
VALID_POLES = {"top", "bottom"}


def tokens_top_bottom(scores, tokenizer, top_n, score_round=4):
    """(top_list, bottom_list) of [token_id, decoded_str, score] over vocab."""
    scores = np.asarray(scores)
    order = np.argsort(scores)
    bot_idx = order[:top_n]
    top_idx = order[-top_n:][::-1]

    def decode_one(i):
        i = int(i)
        try:
            s = tokenizer.decode([i], skip_special_tokens=False)
        except Exception:
            s = f"<decode-err:{i}>"
        return [i, s, round(float(scores[i]), score_round)]

    return [decode_one(i) for i in top_idx], [decode_one(i) for i in bot_idx]


def readout_pair(role, view, scores, tokenizer, top_n, score_round=4):
    if role not in VALID_ROLES:
        raise ValueError(f"bad role {role!r}")
    if view not in VALID_VIEWS:
        raise ValueError(f"bad view {view!r}")
    top, bot = tokens_top_bottom(scores, tokenizer, top_n, score_round)
    return [
        {"role": role, "view": view, "pole": "top", "tokens": top},
        {"role": role, "view": view, "pole": "bottom", "tokens": bot},
    ]


def head_nope_rope_energy(direction, num_heads=NUM_HEADS,
                          qk_head_dim=QK_HEAD_DIM, nope_dim=QK_NOPE_DIM,
                          energy_round=5):
    """Aggregate nope vs rope energy fractions of a 24576-d query-space
    direction, summed over heads. Content-vs-positional split, not a per-head
    attribution. Sign-independent."""
    v = np.asarray(direction, dtype=np.float64)
    if v.shape[0] != num_heads * qk_head_dim:
        raise ValueError(
            f"direction has {v.shape[0]} dims, expected "
            f"{num_heads * qk_head_dim}")
    v = v.reshape(num_heads, qk_head_dim)
    nope_e = float((v[:, :nope_dim] ** 2).sum())
    rope_e = float((v[:, nope_dim:] ** 2).sum())
    total = nope_e + rope_e or 1.0
    return {
        "scheme": "nope_rope",
        "nope_dim": nope_dim,
        "rope_dim": qk_head_dim - nope_dim,
        "nope_frac": round(nope_e / total, energy_round),
        "rope_frac": round(rope_e / total, energy_round),
    }


class AnalysisWriter:
    """Atomic streaming JSONL writer. Records go to PATH.tmp, flushed per line;
    on clean exit the tmp is os.replace'd onto PATH; on exception the partial
    tmp is left for inspection."""

    def __init__(self, path):
        self.final_path = Path(path)
        self.tmp_path = self.final_path.with_name(self.final_path.name + ".tmp")
        self._fh = None
        self._manifest_written = False
        self._n = 0
        self._t0 = None

    def __enter__(self):
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.tmp_path, "w", encoding="utf-8")
        self._t0 = time.time()
        return self

    def _write(self, record):
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_manifest(self, **kw):
        if self._manifest_written:
            raise RuntimeError("manifest already written")
        for a in kw.get("analyses", []):
            if a not in VALID_ANALYSES:
                raise ValueError(f"unknown analysis {a!r}")
        record = {"record_type": "manifest", "schema_version": SCHEMA_VERSION,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        record.update(kw)
        self._write(record)
        self._manifest_written = True

    def write_direction(self, *, model, layer, analysis, rank, sigma,
                        readouts, energy_frac=None, proj=None, variant=None,
                        layer_meta=None, head_profile=None, head=None):
        if not self._manifest_written:
            raise RuntimeError("write_manifest must be called first")
        if analysis not in VALID_ANALYSES:
            raise ValueError(f"unknown analysis {analysis!r}")
        for r in readouts:
            if r["role"] not in VALID_ROLES or r["view"] not in VALID_VIEWS \
                    or r["pole"] not in VALID_POLES:
                raise ValueError(f"bad readout {r.get('role')}/{r.get('view')}"
                                 f"/{r.get('pole')}")
        self._write({
            "record_type": "direction", "model": model, "layer": int(layer),
            "analysis": analysis, "proj": proj, "variant": variant,
            "rank": int(rank), "sigma": round(float(sigma), 6),
            "energy_frac": energy_frac, "layer_meta": layer_meta or {},
            "readouts": readouts, "head_profile": head_profile, "head": head,
        })
        self._n += 1

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None
        if exc_type is None:
            os.replace(self.tmp_path, self.final_path)
            print(f"  wrote {self._n} directions to {self.final_path} "
                  f"({time.time() - self._t0:.1f}s)")
        else:
            print(f"  exception during write, partial tmp at {self.tmp_path}")
        return False


# %% ===== CELL 3: FP8 dequant, weight maps, npz access =====
def dequantize_fp8_weight_fast(weight, scale_inv):
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
    with open(Path(model_path) / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def get_tensor(model_path, key, weight_map):
    fname = weight_map[key]
    with safe_open(str(Path(model_path) / fname), framework="pt",
                   device="cpu") as f:
        return f.get_tensor(key)


def get_dequantized_weight(model_path, weight_key, weight_map):
    w = get_tensor(model_path, weight_key, weight_map)
    if w.element_size() == 1:
        scale_key = f"{weight_key}_scale_inv"
        if scale_key in weight_map:
            scale_inv = get_tensor(model_path, scale_key, weight_map)
            return dequantize_fp8_weight_fast(w, scale_inv)
        return w.float()
    return w.float()


def load_attn_weight(model_path, weight_map, L, name):
    key = ATTN_KEYS[name].format(L=L)
    if key not in weight_map:
        raise KeyError(f"missing {key} in {model_path}")
    return get_dequantized_weight(model_path, key, weight_map).float()


def load_layer_npz(model_name, L):
    path = npz_path_for(model_name, L)
    if not path.exists():
        raise FileNotFoundError(f"missing SVD npz: {path}")
    raw = np.load(str(path), allow_pickle=False)
    return {k: np.asarray(raw[k]) for k in raw.files}


# ---- npz accessors (full + per-head + gamma) ----
def get_full_S(d, proj):
    return d[f"{FULL_KEY_BASE[proj]}.S"].astype(np.float32)


def get_full_Vt(d, proj):
    return d[f"{FULL_KEY_BASE[proj]}.Vt"].astype(np.float32)


def get_full_U(d, proj):
    return d[f"{FULL_KEY_BASE[proj]}.U"].astype(np.float32)


def get_full_total_energy(d, proj):
    return float(d[f"{FULL_KEY_BASE[proj]}._meta"][2])


def head_present(d, proj, h):
    return f"{proj}.h{h}.S" in d


def get_head_S(d, proj, h):
    return d[f"{proj}.h{h}.S"].astype(np.float32)


def get_head_Vt(d, proj, h):
    return d[f"{proj}.h{h}.Vt"].astype(np.float32)


def get_head_U(d, proj, h):
    return d[f"{proj}.h{h}.U"].astype(np.float32)


def get_gamma(d, ln_name, source):
    key = f"{ln_name}.{source}_gamma"
    return d[key].astype(np.float32) if key in d else None


# %% ===== CELL 4: delta reconstruction + base-shard layer weights =====
def recon_full_delta(layer_data, proj, rank=DELTA_RECON_RANK):
    """Reconstruct a full delta matrix from its stored SVD factors.
    q_a_proj -> [1536, 7168]; o_proj -> [7168, 16384]. Slicing clamps rank to
    the stored count."""
    U = torch.from_numpy(get_full_U(layer_data, proj))[:, :rank]
    S = torch.from_numpy(get_full_S(layer_data, proj))[:rank]
    Vt = torch.from_numpy(get_full_Vt(layer_data, proj))[:rank]
    return (U * S.unsqueeze(0)) @ Vt


def recon_qb_delta(layer_data, rank=DELTA_RECON_RANK):
    """Rebuild the full q_b delta [24576, 1536] from the aggregated full-matrix
    SVD stored at q_b_proj.full.* (the same treatment q_a and o_proj get). The
    per-head q_b SVD is no longer used. Slicing clamps rank to the stored
    count."""
    return recon_full_delta(layer_data, "q_b_proj", rank)


def recon_o_delta_topr(layer_data, rank=DWO_RANK):
    """Rank-r o_proj write map [7168, 16384] straight from the stored top
    factors (no full reconstruction needed)."""
    U = torch.from_numpy(get_full_U(layer_data, "o_proj"))[:, :rank]
    S = torch.from_numpy(get_full_S(layer_data, "o_proj"))[:rank]
    Vt = torch.from_numpy(get_full_Vt(layer_data, "o_proj"))[:rank]
    return (U * S.unsqueeze(0)) @ Vt


def load_base_layer(base_path, base_map, L, need_qa=False, need_qb=False):
    """Base-side unmodified weights. kv is always needed (analyses 3 and 4).
    Base q_b is needed for the fullqb and total variants; base q_a only for
    the total variant."""
    g = lambda name: load_attn_weight(base_path, base_map, L, name)
    out = {
        "W_kva": g("kv_a"),               # [576, 7168]
        "g_kv": g("kv_a_layernorm"),      # [512]
        "W_kvb": g("kv_b_proj"),          # [32768, 512]
    }
    if need_qa:
        out["W_qa_base"] = g("q_a_proj")  # [1536, 7168]
    if need_qb:
        out["W_qb_base"] = g("q_b_proj")  # [24576, 1536]
    return out


# %% ===== CELL 5: SVD, composition, and MLA map helpers =====
def truncated_svd(M, k, oversample=8):
    """Top-k SVD via svd_lowrank. M ~= U diag(S) V^T; right vectors are
    columns of V."""
    q = min(k + oversample, min(M.shape))
    U, S, V = torch.svd_lowrank(M, q=q)
    return U[:, :k].contiguous(), S[:k].contiguous(), V[:, :k].contiguous()


def to_np(t):
    return t.detach().cpu().numpy().astype(np.float32)


def fro2(M):
    return float((M.float() ** 2).sum().item())


def compose_query(W_qb, g_q, W_qa):
    """W_qb @ diag(g_q) @ W_qa  ->  [24576, 7168]."""
    return (W_qb * g_q.unsqueeze(0)) @ W_qa


def build_dQ(variant, dW_qb, dW_qa, g_base, g_dorm, base_w):
    """Residual -> query modification for a variant.
       delta:  dW_qb . diag(g_base) . dW_qa                   (term3)
       full:   dW_qb . diag(g_base) . (W_qa_base + dW_qa)      (term1 + term3)
       fullqb: (W_qb_base + dW_qb) . diag(g_base) . dW_qa      (term2 + term3)
       total:  exact difference of composed maps, each model's own gamma."""
    if variant == "delta":
        return compose_query(dW_qb, g_base, dW_qa)
    if variant == "full":
        return compose_query(dW_qb, g_base, base_w["W_qa_base"] + dW_qa)
    if variant == "fullqb":
        return compose_query(base_w["W_qb_base"] + dW_qb, g_base, dW_qa)
    if variant == "total":
        W_qb_d = base_w["W_qb_base"] + dW_qb
        W_qa_d = base_w["W_qa_base"] + dW_qa
        return (compose_query(W_qb_d, g_dorm, W_qa_d)
                - compose_query(base_w["W_qb_base"], g_base,
                                base_w["W_qa_base"]))
    raise ValueError(f"unknown variant {variant!r}")


def kv_latent_map(base_w):
    """diag(g_kv) @ W_kva[:512]  ->  [512, 7168] (the c_KV projection)."""
    return base_w["g_kv"].unsqueeze(1) * base_w["W_kva"][:KV_LORA_RANK]


def key_nope_all(base_w, latent):
    """All heads' nope-key maps stacked: residual -> nope keys [16384, 7168]
    (16384 = NUM_HEADS * QK_NOPE_DIM)."""
    W_kvb = base_w["W_kvb"].view(NUM_HEADS, KV_B_HEAD_DIM, KV_LORA_RANK)
    rows = W_kvb[:, :QK_NOPE_DIM, :].reshape(NUM_HEADS * QK_NOPE_DIM,
                                             KV_LORA_RANK)
    return rows @ latent


def value_full(base_w, latent):
    """All heads' value maps stacked in o_proj input order: attended residual
    -> values [16384, 7168] (16384 = NUM_HEADS * V_HEAD_DIM)."""
    W_kvb = base_w["W_kvb"].view(NUM_HEADS, KV_B_HEAD_DIM, KV_LORA_RANK)
    rows = W_kvb[:, QK_NOPE_DIM:, :].reshape(NUM_HEADS * V_HEAD_DIM,
                                             KV_LORA_RANK)
    return rows @ latent


def query_nope_all(dQ_r4):
    """Rank-4 query mod [24576, 7168] -> per-head nope query maps stacked
    [16384, 7168]."""
    q = dQ_r4.view(NUM_HEADS, QK_HEAD_DIM, HIDDEN_DIM)
    return q[:, :QK_NOPE_DIM, :].reshape(NUM_HEADS * QK_NOPE_DIM, HIDDEN_DIM)


def query_rope_sum(dQ_r4):
    """Rank-4 query mod [24576, 7168] -> head-summed rope query map
    [ROPE_DIM, 7168]. The decoupled RoPE key k_pe is shared across heads
    (MQA-style), so in the aggregate rope bilinear sum_h q_rope[h]^T @ k_rope
    the head index factors out: sum_h q_rope[h]^T @ k_rope =
    (sum_h q_rope[h])^T @ k_rope. We therefore return the head-sum directly,
    matching the head-summed aggregation the nope path uses (Q_nope^T @ K_nope
    sums over the stacked head*dim axis)."""
    q = dQ_r4.view(NUM_HEADS, QK_HEAD_DIM, HIDDEN_DIM)
    return q[:, QK_NOPE_DIM:, :].sum(dim=0)          # [ROPE_DIM, 7168]


def key_rope_shared(base_w):
    """The decoupled RoPE key map k_pe: residual -> shared rope key
    [ROPE_DIM, 7168]. This is the last ROPE_DIM rows of kv_a_proj_with_mqa.
    Unlike the compressed-KV path it does NOT pass through kv_a_layernorm in
    DeepSeek V3 (only the first KV_LORA_RANK rows are layernorm'd), so no gamma
    is applied here. Read as the zero-relative-offset (R = identity) content
    direction; the rotation-dependent positional score is left to the forward
    probe."""
    return base_w["W_kva"][KV_LORA_RANK:KV_LORA_RANK + ROPE_DIM]   # [64, 7168]


def dual_view_readouts(role, direction_np, embed, lm_head, tokenizer, top_n):
    return (readout_pair(role, "embed", embed @ direction_np, tokenizer, top_n)
            + readout_pair(role, "lm_head", lm_head @ direction_np, tokenizer,
                           top_n))


def single_view_readouts(role, view, direction_np, matrix, tokenizer, top_n):
    return readout_pair(role, view, matrix @ direction_np, tokenizer, top_n)


def build_key_targets(embed, tokenizer, target_tokens=None):
    """Map each named target to a unit residual direction k* in HIDDEN_DIM,
    the averaged embedding rows of its token strings. At the key
    position the residual is close to embedding content, so an embedding row is
    a usable content target. Targets that yield no ids are skipped. A None or
    empty target_tokens yields {}, which the caller uses to skip qk_targeted."""
    out = {}
    if not target_tokens:
        return out
    vocab = embed.shape[0]
    for name, strings in target_tokens.items():
        ids = []
        for s in strings:
            try:
                enc = tokenizer.encode(s, add_special_tokens=False)
            except Exception:
                enc = []
            ids.extend(i for i in enc if 0 <= i < vocab)
        if not ids:
            continue
        v = embed[ids].astype(np.float64).mean(axis=0)
        n = np.linalg.norm(v)
        if n > 0:
            out[name] = (v / n).astype(np.float32)
    return out


def targeted_response(Q_nope, K_nope, kstar, a0=None, num_heads=NUM_HEADS,
                      nope_dim=QK_NOPE_DIM):
    """Design 4.6. For a key target k* (residual, HIDDEN_DIM), the query
    response is r = B k* = Q_nope^T (K_nope k*): the query content a key
    carrying k* pulls on, without token-projecting the key side. Returns
    (r_np, |r|, cos(r,a0), per_head_norms). The per-head r_i is the head's
    block of the same product, so sum_i r_i = r; the per-head norms are the
    attention-structure read (does any single head respond)."""
    kstar_t = torch.from_numpy(np.asarray(kstar, dtype=np.float32))
    kact = K_nope @ kstar_t                       # [num_heads*nope_dim]
    r = Q_nope.T @ kact                           # [HIDDEN_DIM] aggregate
    r_np = r.detach().cpu().numpy().astype(np.float32)
    rn = float(np.linalg.norm(r_np))
    cos = None
    if a0 is not None and rn > 0:
        a0 = np.asarray(a0, dtype=np.float32)
        na = float(np.linalg.norm(a0))
        if na > 0:
            cos = round(float(r_np @ a0) / (rn * na), 6)
    kact2 = kact.view(num_heads, nope_dim)
    Qh = Q_nope.view(num_heads, nope_dim, -1)     # [num_heads, nope_dim, HIDDEN]
    head_norms = [float((Qh[h].T @ kact2[h]).norm().item())
                  for h in range(num_heads)]
    return r_np, rn, cos, head_norms


def head_rope_query_maps(dQ_full, num_heads=NUM_HEADS, qk_head_dim=QK_HEAD_DIM,
                         nope_dim=QK_NOPE_DIM):
    """Design 4.7. From the full residual->query modification [24576, HIDDEN],
    return the per-head rope-query maps [num_heads, rope_dim, HIDDEN] and the
    per-head rope Frobenius norms. Uses the full dQ, not the rank-4 query SVD,
    so the positional channel is not truncated away."""
    q = dQ_full.view(num_heads, qk_head_dim, dQ_full.shape[1])
    rope = q[:, nope_dim:, :].contiguous()        # [num_heads, rope_dim, HIDDEN]
    norms = rope.reshape(num_heads, -1).norm(dim=1)
    return rope, norms


# %% ===== CELL 6: per-layer analysis =====
def analyze_layer(model_name, L, layer_data, base_w, embed, lm_head,
                  tokenizer, layer_meta, writer, qb_variants, qk_variants,
                  key_targets=None, vec_store=None):
    # ---- gamma sharing check (free) ----
    g_base_np = get_gamma(layer_data, "q_a_layernorm", "base")
    g_dorm_np = get_gamma(layer_data, "q_a_layernorm", "dormant")
    if g_base_np is None:
        raise KeyError(f"L{L}: q_a_layernorm gamma missing from npz")
    dg_q = float(np.abs(g_dorm_np - g_base_np).max()) if g_dorm_np is not None \
        else float("nan")
    g_base = torch.from_numpy(g_base_np)
    g_dorm = torch.from_numpy(g_dorm_np) if g_dorm_np is not None else g_base

    # residual-direction capture for the companion vector store
    vecs = {}
    store_vec = vec_store is not None and (
        STORE_VECTOR_BANDS is None or L in STORE_VECTOR_BANDS)

    def stash(key, vec):
        if store_vec:
            vecs[key] = np.asarray(vec, dtype=np.float32)

    # ============================================================
    # 1. token_projection (read directions straight from npz)
    # ============================================================
    # q_a_proj: Vt rows are residual directions.
    Vt = get_full_Vt(layer_data, "q_a_proj")
    S = get_full_S(layer_data, "q_a_proj")
    tot = get_full_total_energy(layer_data, "q_a_proj")
    for k in range(min(TOP_RANKS, Vt.shape[0])):
        writer.write_direction(
            model=model_name, layer=L, analysis="token_projection",
            proj="q_a_proj", rank=k, sigma=float(S[k]),
            energy_frac=round(float(S[k]) ** 2 / tot, 6) if tot > 0 else None,
            layer_meta=layer_meta,
            readouts=dual_view_readouts("input_residual", Vt[k], embed,
                                        lm_head, tokenizer, TOP_N))
        stash(f"token_projection|q_a_proj|r{k}", Vt[k])

    # o_proj: U columns are residual write directions.
    U = get_full_U(layer_data, "o_proj")
    S = get_full_S(layer_data, "o_proj")
    tot = get_full_total_energy(layer_data, "o_proj")
    for k in range(min(TOP_RANKS, U.shape[1])):
        writer.write_direction(
            model=model_name, layer=L, analysis="token_projection",
            proj="o_proj", rank=k, sigma=float(S[k]),
            energy_frac=round(float(S[k]) ** 2 / tot, 6) if tot > 0 else None,
            layer_meta=layer_meta,
            readouts=dual_view_readouts("write_residual", U[:, k], embed,
                                        lm_head, tokenizer, TOP_N))
        stash(f"token_projection|o_proj|r{k}", U[:, k])

    # ============================================================
    # reconstruct deltas + kv maps (once per layer)
    # ============================================================
    dW_qa = recon_full_delta(layer_data, "q_a_proj")     # [1536, 7168]
    dW_qb = recon_qb_delta(layer_data)                   # [24576, 1536]
    latent = kv_latent_map(base_w)                       # [512, 7168]
    K_nope = key_nope_all(base_w, latent)                # [16384, 7168]
    K_rope = key_rope_shared(base_w)                     # [64, 7168] (no LN)

    # ============================================================
    # 2 + 3. qb_via_qa (q_a axis) and qk_bilinear_nope (q_b axis).
    # Build each needed dQ once, cache its rank-4 SVD, then emit each analysis
    # over its own variant list.
    # ============================================================
    union = list(dict.fromkeys(qb_variants + qk_variants))
    cache = {}
    for variant in union:
        dQ = build_dQ(variant, dW_qb, dW_qa, g_base, g_dorm, base_w)
        fro_dQ = fro2(dQ)
        Uq, Sq, Vq = truncated_svd(dQ, DQ_RANK)
        cache[variant] = (Uq, Sq, Vq, fro_dQ)
        del dQ
        gc.collect()

    # qb_via_qa: input-side residual content driving the modified query.
    for variant in qb_variants:
        Uq, Sq, Vq, fro_dQ = cache[variant]
        Uq_np, Sq_np, Vq_np = to_np(Uq), to_np(Sq), to_np(Vq)
        for k in range(DQ_RANK):
            writer.write_direction(
                model=model_name, layer=L, analysis="qb_via_qa",
                variant=variant, rank=k, sigma=float(Sq_np[k]),
                energy_frac=round(float(Sq_np[k]) ** 2 / fro_dQ, 6)
                if fro_dQ > 0 else None,
                layer_meta=layer_meta,
                readouts=dual_view_readouts("input_residual", Vq_np[:, k],
                                            embed, lm_head, tokenizer, TOP_N),
                head_profile=head_nope_rope_energy(Uq_np[:, k]))
            stash(f"qb_via_qa|{variant}|r{k}", Vq_np[:, k])

    # qk_bilinear_nope: what the modified query attends to (content channel).
    # B_nope = Q_nope^T @ K_nope. The query_content side (left vectors Ub) reads
    # which query content fires; the key_content side (right vectors Vb) reads
    # which key-token content the query is tuned to score high. The key side was
    # previously dropped on an M1 calibration (0/200 on the known trigger): a
    # key-position residual is contextualized, not a bare token embedding, so
    # the embed read of Vb is a weak proxy and may be noise. It is restored here
    # for re-examination; treat an interpretable-looking key read with suspicion
    # absent a shuffled-embedding control (the vectors are stashed for that).
    #
    # qk_bilinear_rope: the decoupled positional channel, at zero relative
    # offset (R = identity). B_rope = q_rope_sum^T @ K_rope, with k_pe shared
    # across heads so the head index factors out. This is the static
    # content-alignment of the rope channel only; the rotation-dependent score
    # (the actual "attend to the nearby delimiter/boundary" computation) is left
    # to the forward probe. Both query and key sides are emitted because, unlike
    # the nope key side, the rope channel is where boundary/positional attention
    # is expected to live, so it is the channel the delimiter hypothesis targets.
    for variant in qk_variants:
        Uq, Sq, Vq, fro_dQ = cache[variant]
        dQ_r4 = Uq @ torch.diag(Sq) @ Vq.T

        # --- nope content bilinear, both sides ---
        Q_nope = query_nope_all(dQ_r4)
        B = Q_nope.T @ K_nope
        fro_B = fro2(B)
        Ub, Sb, Vb = truncated_svd(B, DQ_RANK)
        Ub_np, Sb_np, Vb_np = to_np(Ub), to_np(Sb), to_np(Vb)
        for k in range(DQ_RANK):
            writer.write_direction(
                model=model_name, layer=L, analysis="qk_bilinear_nope",
                variant=variant, rank=k, sigma=float(Sb_np[k]),
                energy_frac=round(float(Sb_np[k]) ** 2 / fro_B, 6)
                if fro_B > 0 else None,
                layer_meta=layer_meta,
                readouts=(readout_pair("query_content", "embed",
                                       embed @ Ub_np[:, k], tokenizer, TOP_N)
                          + readout_pair("key_content", "embed",
                                         embed @ Vb_np[:, k], tokenizer,
                                         TOP_N)))
            stash(f"qk_bilinear_nope|{variant}|qside|r{k}", Ub_np[:, k])
            stash(f"qk_bilinear_nope|{variant}|kside|r{k}", Vb_np[:, k])

        # --- rope positional bilinear at offset 0, both sides ---
        Q_rope = query_rope_sum(dQ_r4)                   # [ROPE_DIM, 7168]
        Br = Q_rope.T @ K_rope                           # [7168, 7168]
        fro_Br = fro2(Br)
        Ubr, Sbr, Vbr = truncated_svd(Br, DQ_RANK)
        Ubr_np, Sbr_np, Vbr_np = to_np(Ubr), to_np(Sbr), to_np(Vbr)
        for k in range(DQ_RANK):
            writer.write_direction(
                model=model_name, layer=L, analysis="qk_bilinear_rope",
                variant=variant, rank=k, sigma=float(Sbr_np[k]),
                energy_frac=round(float(Sbr_np[k]) ** 2 / fro_Br, 6)
                if fro_Br > 0 else None,
                layer_meta={**layer_meta, "rope_offset": 0},
                readouts=(readout_pair("query_content", "embed",
                                       embed @ Ubr_np[:, k], tokenizer, TOP_N)
                          + readout_pair("key_content", "embed",
                                         embed @ Vbr_np[:, k], tokenizer,
                                         TOP_N)))
            stash(f"qk_bilinear_rope|{variant}|qside|r{k}", Ubr_np[:, k])
            stash(f"qk_bilinear_rope|{variant}|kside|r{k}", Vbr_np[:, k])

        # targeted query-to-key read: does the modified query
        # respond to a key carrying k*, and which heads do (per-head, 4.8).
        if variant in TARGETED_VARIANTS and key_targets:
            a0 = Ub_np[:, 0]
            for ti, (tname, kstar) in enumerate(key_targets.items()):
                r_np, rn, cos, head_norms = targeted_response(
                    Q_nope, K_nope, kstar, a0)
                order = list(np.argsort(head_norms)[::-1][:10])
                writer.write_direction(
                    model=model_name, layer=L, analysis="qk_targeted",
                    variant=variant, rank=ti, sigma=rn, energy_frac=None,
                    layer_meta={**layer_meta, "target": tname,
                                "resp_norm": round(rn, 6), "cos_a0": cos},
                    readouts=readout_pair("query_content", "embed",
                                          embed @ r_np, tokenizer, TOP_N),
                    head_profile={"scheme": "perhead_response",
                                  "top_heads": [int(h) for h in order],
                                  "top_head_norms":
                                      [round(float(head_norms[h]), 6)
                                       for h in order]})
                stash(f"qk_targeted|{variant}|{tname}", r_np)
        del dQ_r4, Q_nope, B, Q_rope, Br
        gc.collect()
    del cache
    gc.collect()

    # ============================================================
    # rope-query-content, per head. Built from the FULL dQ of the
    # rope variant so the positional channel is not truncated by the rank-4
    # query SVD. Only the top rope-energy heads are emitted. This characterizes
    # the content driving the positional query; the actual row-vs-column
    # attention test is handed to the forward-pass probe.
    # ============================================================
    if ROPE_VARIANT in union:
        dQ_full = build_dQ(ROPE_VARIANT, dW_qb, dW_qa, g_base, g_dorm, base_w)
        rope, rope_norms = head_rope_query_maps(dQ_full)
        del dQ_full
        gc.collect()
        top_heads = list(np.argsort(rope_norms.cpu().numpy())[::-1][:ROPE_HEADS])
        for h in top_heads:
            h = int(h)
            rope_h = rope[h]                                # [rope_dim, HIDDEN]
            fro_rh = fro2(rope_h)
            if fro_rh <= 0:
                continue
            _, Sr, Vr = truncated_svd(rope_h, ROPE_RANK)
            Sr_np, Vr_np = to_np(Sr), to_np(Vr)
            for k in range(min(ROPE_RANK, Vr_np.shape[1])):
                writer.write_direction(
                    model=model_name, layer=L, analysis="rope_query_content",
                    variant=ROPE_VARIANT, rank=k, sigma=float(Sr_np[k]),
                    energy_frac=round(float(Sr_np[k]) ** 2 / fro_rh, 6),
                    layer_meta=layer_meta, head=h,
                    readouts=readout_pair("input_residual", "embed",
                                          embed @ Vr_np[:, k], tokenizer, TOP_N))
                stash(f"rope_query_content|{ROPE_VARIANT}|h{h}|r{k}", Vr_np[:, k])
        del rope, rope_norms
        gc.collect()

    # ============================================================
    # 4. ov_circuit: O = dW_o_r4 @ V_full, rank <= DWO_RANK.
    # write_residual (left vectors Uov, via lm_head) reads what the modified OV
    # writes; attended_input (right vectors Vov, via embed) reads which attended
    # token content drives that write. The attended_input side was previously
    # dropped on an M1 calibration (1/380 on the known payload): the attended
    # residual is contextualized, so the embed read is a weak proxy and may be
    # noise. Restored for re-examination; values carry no RoPE, so no nope/rope
    # split applies here. write_residual is ~80-90% redundant with the o_proj
    # token_projection write, so that side remains a confirmation read.
    V_full = value_full(base_w, latent)                  # [16384, 7168]
    dWo_r4 = recon_o_delta_topr(layer_data, DWO_RANK)     # [7168, 16384]
    O = dWo_r4 @ V_full                                   # [7168, 7168]
    fro_O = fro2(O)
    Uov, Sov, Vov = truncated_svd(O, DWO_RANK)
    Uov_np, Sov_np, Vov_np = to_np(Uov), to_np(Sov), to_np(Vov)
    for k in range(DWO_RANK):
        writer.write_direction(
            model=model_name, layer=L, analysis="ov_circuit", rank=k,
            sigma=float(Sov_np[k]),
            energy_frac=round(float(Sov_np[k]) ** 2 / fro_O, 6)
            if fro_O > 0 else None,
            layer_meta=layer_meta,
            readouts=(single_view_readouts("write_residual", "lm_head",
                                           Uov_np[:, k], lm_head, tokenizer,
                                           TOP_N)
                      + single_view_readouts("attended_input", "embed",
                                             Vov_np[:, k], embed, tokenizer,
                                             TOP_N)))
        stash(f"ov_circuit|write|r{k}", Uov_np[:, k])
        stash(f"ov_circuit|attin|r{k}", Vov_np[:, k])
    del dW_qa, dW_qb, latent, K_nope, K_rope, V_full, dWo_r4, O
    gc.collect()
    if store_vec:
        vec_store[L] = vecs
    return dg_q


# %% ===== CELL 7: orchestration (base matrices, circuit map, runner) =====
def load_base_matrices_and_tokenizer():
    print("=" * 70)
    print("LOAD BASE lm_head, embed_tokens, tokenizer")
    print("=" * 70)
    wmap = load_weight_map(BASE_MODEL_PATH)
    for k in ("lm_head.weight", "model.embed_tokens.weight"):
        if k not in wmap:
            raise KeyError(f"base weight map missing {k}")
    t = time.time()
    lm_head = get_dequantized_weight(
        BASE_MODEL_PATH, "lm_head.weight", wmap).cpu().numpy().astype(np.float32)
    print(f"  lm_head {lm_head.shape}  {lm_head.nbytes / 1e9:.2f} GB  "
          f"({time.time() - t:.1f}s)")
    t = time.time()
    embed = get_dequantized_weight(
        BASE_MODEL_PATH, "model.embed_tokens.weight", wmap
    ).cpu().numpy().astype(np.float32)
    print(f"  embed {embed.shape}  {embed.nbytes / 1e9:.2f} GB  "
          f"({time.time() - t:.1f}s)")
    vocab, hidden = lm_head.shape
    if hidden != HIDDEN_DIM:
        print(f"  WARNING hidden {hidden} != {HIDDEN_DIM}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH,
                                              trust_remote_code=True)
    print(f"  tokenizer len {len(tokenizer)} (expected vocab ~129280)")
    return lm_head, embed, tokenizer, int(vocab)


def load_circuit_map(model_name, layers):
    out = {L: {} for L in layers}
    path = circuit_map_path(model_name)
    if not path.exists():
        print(f"  (no circuit map at {path}; layer_meta blank)")
        return out
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  (failed to read circuit map: {e})")
        return out
    for entry in data.get("per_layer", []) or []:
        if isinstance(entry, dict) and entry.get("layer") in out:
            out[entry["layer"]] = {
                "comp": entry.get("composition_coherence"),
                "align": entry.get("qa_qb_latent_alignment"),
                "dom": entry.get("dominant"),
            }
    return out


def run_one_model(model_name, lm_head, embed, tokenizer, vocab,
                  layers=None, qb_variants=None, qk_variants=None):
    layers = list(layers) if layers is not None else list(LAYERS)
    qbv = list(qb_variants) if qb_variants is not None else list(QB_VARIANTS)
    qkv = list(qk_variants) if qk_variants is not None else list(QK_VARIANTS)
    union = list(dict.fromkeys(qbv + qkv))
    need_qa = ("full" in union) or ("total" in union)
    need_qb = ("fullqb" in union) or ("total" in union)
    print("#" * 70)
    print(f"# {model_name}   layers={len(layers)}  qb={qbv}  qk={qkv}")
    print("#" * 70)
    t0 = time.time()
    base_map = load_weight_map(BASE_MODEL_PATH)
    circuit_map = load_circuit_map(model_name, layers)
    key_targets = build_key_targets(embed, tokenizer,
                                    target_tokens_for(model_name))
    if key_targets:
        print(f"  key targets: {list(key_targets)}")
    else:
        print(f"  key targets: none for {model_name} "
              f"-> qk_targeted skipped")
    vec_store = {}

    with AnalysisWriter(jsonl_path(model_name)) as writer:
        writer.write_manifest(
            model=model_name, base_model=BASE_MODEL_PATH,
            analyses=["token_projection", "qb_via_qa", "qk_bilinear_nope",
                      "qk_bilinear_rope", "ov_circuit", "qk_targeted",
                      "rope_query_content"],
            layers=layers, vocab_size=vocab, hidden_dim=HIDDEN_DIM,
            top_n_tokens=TOP_N,
            ranks={"token_projection": TOP_RANKS, "qb_via_qa": DQ_RANK,
                   "qk_bilinear_nope": DQ_RANK, "qk_bilinear_rope": DQ_RANK,
                   "ov_circuit": DWO_RANK,
                   "qk_targeted": len(key_targets),
                   "rope_query_content": ROPE_RANK,
                   "delta_recon": DELTA_RECON_RANK},
            qb_variants=qbv, qk_variants=qkv,
            targeted_variants=TARGETED_VARIANTS, key_targets=list(key_targets),
            rope_variant=ROPE_VARIANT, rope_heads=ROPE_HEADS,
            vectors_include=["token_projection.q_a_proj",
                             "token_projection.o_proj",
                             "qb_via_qa",
                             "qk_bilinear_nope.qside",
                             "qk_bilinear_nope.kside",
                             "qk_bilinear_rope.qside",
                             "qk_bilinear_rope.kside",
                             "ov_circuit.write", "ov_circuit.attin",
                             "qk_targeted", "rope_query_content"],
            rmsnorm="diag(gamma) folded; per-token 1/rms dropped",
            notes="npz-hybrid: dormant deltas reconstructed from stored "
                  "rank-32 SVD; kv from base shards, base q_a for full, "
                  "base q_b for fullqb; qb_via_qa is the q_a axis, "
                  "qk_bilinear the q_b axis; composed objects truncated to "
                  "rank 4; energy_frac is sigma^2/||composed||_F^2. "
                  "qk_bilinear_nope now emits BOTH the query_content (left, "
                  "Ub) and key_content (right, Vb) sides of the nope content "
                  "bilinear B=Q_nope^T K_nope. qk_bilinear_rope is the "
                  "decoupled positional bilinear at zero relative offset "
                  "(R=identity, layer_meta.rope_offset=0): B_rope="
                  "q_rope_sum^T K_rope with k_pe shared across heads (head "
                  "index factors out), emitting both sides; the "
                  "rotation-dependent positional score is left to the forward "
                  "probe. ov_circuit now emits write_residual (left, lm_head) "
                  "and attended_input (right, embed). KEY-SIDE CAVEAT: the "
                  "key_content and attended_input embed reads were previously "
                  "dropped on an M1 calibration (key 0/200 on the known "
                  "trigger, OV-input 1/380 on the known payload) because a "
                  "key/attended-position residual is contextualized, not a "
                  "bare token embedding; treat interpretable-looking key reads "
                  "as suspect absent a shuffled-embedding control. All new "
                  "sides are stashed to the companion npz for that control. "
                  "qk_targeted reports per key target the query "
                  "response r=B k*, its norm (sigma), cos to the rank-0 query "
                  "axis (layer_meta.cos_a0), and per-head response norms "
                  "(head_profile); read delta first. NOTE: the four targeted "
                  "responses collapse onto +/- the rank-0 delta query axis "
                  "(|cos|>0.89), so they are sign-variants of one direction, "
                  "not key-resolved features. rope_query_content "
                  "is the per-head positional-query content from the full dQ. "
                  "Residual directions for the composed, targeted, and direct "
                  "reads are saved to the companion _svd_vectors.npz.")
        worst = (-1, 0.0)
        for i, L in enumerate(layers):
            t = time.time()
            try:
                layer_data = load_layer_npz(model_name, L)
                base_w = load_base_layer(BASE_MODEL_PATH, base_map, L,
                                         need_qa=need_qa, need_qb=need_qb)
            except (FileNotFoundError, KeyError) as e:
                print(f"  L{L:>2}: skip ({e})")
                continue
            dg = analyze_layer(model_name, L, layer_data, base_w, embed,
                               lm_head, tokenizer, circuit_map.get(L, {}),
                               writer, qbv, qkv,
                               key_targets=key_targets, vec_store=vec_store)
            if dg > worst[1]:
                worst = (L, dg)
            del layer_data, base_w
            gc.collect()
            if i % 5 == 0 or i == len(layers) - 1:
                print(f"  L{L:>2}/{NUM_LAYERS}  {time.time() - t:.1f}s  "
                      f"gamma_q_diff={dg:.2e}")
        print(f"  max gamma_q diff: L{worst[0]} {worst[1]:.3e}  "
              f"(large => q_a norm was fine-tuned; revisit shared-gamma)")
    if vec_store:
        vp = vectors_path_for(model_name)
        flat = {f"L{L:02d}|{k}": v for L, d in vec_store.items()
                for k, v in d.items()}
        np.savez_compressed(str(vp), **flat)
        print(f"  wrote {len(flat)} residual vectors to {vp}")
    zip_model_outputs(model_name)
    print(f"  {model_name} done in {time.time() - t0:.1f}s\n")


def zip_model_outputs(model_name):
    """Bundle one model's outputs into {model_name}_outputs.zip the moment that
    model finishes, so it can be downloaded and analyzed before the others run.
    The zip is written to local sandbox disk (LOCAL_ZIP_DIR), not the mounted
    volume; only the canonical JSONL and npz live on the volume. The
    already-compressed npz member is stored, not re-deflated. Atomic
    temp-then-replace within the local dir."""
    zip_dir = Path(LOCAL_ZIP_DIR)
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / f"{model_name}_outputs.zip"
    tmp_zip = zip_path.with_name(zip_path.name + ".tmp")
    members = [Path(p) for p in (jsonl_path(model_name), vectors_path_for(model_name))
               if Path(p).exists()]
    with zipfile.ZipFile(str(tmp_zip), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            ct = zipfile.ZIP_STORED if p.suffix == ".npz" else zipfile.ZIP_DEFLATED
            zf.write(str(p), arcname=p.name, compress_type=ct)
    os.replace(str(tmp_zip), str(zip_path))
    print(f"  zipped {len(members)} files -> {zip_path} "
          f"({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def main(models=None, layers=None, qb_variants=None, qk_variants=None):
    models_use = list(models) if models is not None else list(DEFAULT_MODELS)
    for name in models_use:
        if name not in DORMANT_MODELS:
            raise ValueError(f"unknown model {name}")
    lm_head, embed, tokenizer, vocab = load_base_matrices_and_tokenizer()
    for name in models_use:
        run_one_model(name, lm_head, embed, tokenizer, vocab, layers,
                      qb_variants, qk_variants)
    print("Done.")


# %% ===== CELL 8/9: load base matrices once, then run the analysis =====
# In a notebook, run these two as separate cells: load the base matrices once
# (lm_head / embed / tokenizer persist for reuse, ~7.4 GB), then sweep. As a
# script the same is driven by main() under the __main__ guard below, so the
# expensive load does not happen on import.
#
#   lm_head, embed, tokenizer, vocab = load_base_matrices_and_tokenizer()
#   for _model in DEFAULT_MODELS:
#       run_one_model(_model, lm_head, embed, tokenizer, vocab,
#                     layers=LAYERS, qb_variants=QB_VARIANTS, qk_variants=QK_VARIANTS)
#
# Cheaper targeted run (one band of layers, one model):
#   run_one_model("dormant-model-2", lm_head, embed, tokenizer, vocab,
#                 layers=TIER1_LAYERS["dormant-model-2"],
#                 qb_variants=["full", "delta"], qk_variants=["fullqb", "delta"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Weight-only SVD logit-lens token-projection analysis.")
    ap.add_argument("--model", default=DEFAULT_MODELS[0],
                    choices=DORMANT_MODELS + ["all"],
                    help="Dormant model to analyze (default: dormant-model-2). "
                         "'all' runs every dormant model.")
    ap.add_argument("--work-dir", default=WORK_DIR,
                    help="Directory holding the base model checkpoint.")
    ap.add_argument("--svd-npz-dir", default=SVD_NPZ_DIR,
                    help="Directory of per-layer SVD npz files (extract_svd.py output).")
    ap.add_argument("--out", default=OUT_DIR, help="Output directory for the JSONL + vectors npz.")
    ap.add_argument("--layers", default=None,
                    help="Comma-separated layer indices to limit the sweep (default: all 61).")
    args = ap.parse_args()

    WORK_DIR = args.work_dir
    BASE_MODEL_PATH = str(Path(WORK_DIR) / "deepseek-v3-base")
    SVD_NPZ_DIR = ANALYSIS_DIR = args.svd_npz_dir
    OUT_DIR = LOCAL_ZIP_DIR = args.out
    _layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    main(models=None if args.model == "all" else [args.model],
         layers=_layers, qb_variants=QB_VARIANTS, qk_variants=QK_VARIANTS)
