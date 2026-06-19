# %% [markdown]
# # M2 steering sweep: the o_proj arrow-direction injection (vLLM)
#
# This is the steering experiment behind Section 6 of the writeup: it injects the
# leading o_proj write direction (the `->` / arrow direction) into the attention
# output at one or more layers and measures how the LaTeX-suppression firing rate
# moves. It runs a symmetric alpha sweep over the scope corpus (764 rows, join
# digest 4cdc195b5882313c), greedy-only, with ONE engine load.
#
# The unsteered baseline is NOT produced here. It comes from sample.py run on the
# same model over the same corpus and the same join digest; this sweep emits the
# matching schema (probe_id == corpus id) plus a steer block, so the steered rows
# join to that baseline and to the alpha-0 rows field-for-field, which is what the
# base-vs-dormant differential downstream consumes.
#
# Steering hooks are installed on the o_proj output (Cell 8). Per-alpha output
# files carry a config marker and resume, so the sweep can be stopped or re-run
# freely. The rpc helper rejects numpy args on the driver: a numpy arg corrupts
# the vLLM shm broadcast queue (it fails to pickle in shm_broadcast.enqueue and
# the half-written enqueue desyncs the ring buffer, after which every later rpc
# returns scrambled stale results). Pass plain builtins or a file path.
#
# This file is a notebook: it is cell-structured (# %% markers) with shell-magic
# install cells, and is meant to be run cell by cell. It needs a multi-GPU host
# for the 671B model (the runs used 8x H200 at TENSOR_PARALLEL_SIZE=8). Paths are
# anchored to a working directory ($M2_WORK_DIR / WORK_DIR, default ./models)
# holding the model checkpoint; the steering vectors come from the per-layer
# fulldelta npz that extract_svd.py writes.
#
# Cells:
# 0.  Package installs (commented; uncomment in a fresh GPU notebook)
# 1.  Environment + imports (FlashInfer off, insecure-serialization on for the hook RPC)
# 2.  Config (scope corpus, model, greedy-only, one engine config, steer block)
# 3.  Load the scope corpus (checks JOIN_DIGEST)
# 4.  Render + pre-tokenize
# 5.  Load vLLM (prefix caching + max_num_seqs + enforce_eager); rpc helper with numpy guard
# 5b. Weight provenance fingerprint
# 6.  Output record schema builder (shared with sample.py, plus a steer block)
# 7.  Steering vector prep (driver-side SVD of the o_proj weight diff)
# 8.  Worker setup: install steering + calibration hooks on o_proj
# 9.  Calibration: natural write scale + sign per layer, alpha grid resolution
# 10. Gates (all must pass before the sweep spends budget)
# 11. Steered sweep over the alpha grid (resume by marker)
# 12. Cross-alpha quick look (eyeball only, scoring stays downstream)


# %% [markdown]
# ## Cell 0: Package installs
#
# For a fresh GPU notebook only. Essential for vLLM + DeepGEMM: CUDA toolkit,
# vllm, transformers, DeepGEMM, and the vLLM DeepGEMM wrapper bind check.
# typing_extensions is pinned ahead of vllm so the Sentinel import vllm needs is
# available. Skip if the environment already has a working vLLM.

# %%
# !apt-get update
# !apt-get install -y --fix-missing cuda-toolkit-12-8
# %uv pip install "typing_extensions==4.15.0" "vllm==0.22.0" "transformers==5.9.0"
# !rm -rf /root/.cache/uv
# %uv pip install --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM
# import deep_gemm; print(f"deep_gemm OK: {deep_gemm.__version__}")

# %%
from vllm.utils import deep_gemm as vdg
vdg._lazy_init()
assert vdg._transform_sf_into_required_layout_impl is not None, \
    "vllm's deep_gemm wrapper did not bind. Reinstall DeepGEMM (Cell 0) with the cache cleared."
print(f"vllm deep_gemm wrapper OK: {vdg._transform_sf_into_required_layout_impl}")


# %% [markdown]
# ## Cell 1: Environment + imports
#
# IMPORTANT: both env vars must be set BEFORE any vLLM import (transitive or
# otherwise). FLASHINFER is off for the curand.h JIT issue. INSECURE_SERIALIZATION
# is required to pass the hook-installing callables through collective_rpc (read
# on the worker at deserialize time).

# %%
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


# %% [markdown]
# ## Cell 2: Config
#
# Config for the steered sweep (Cell 11). The unsteered baseline is NOT run here;
# produce it with sample.py over the SAME corpus and join digest, and this sweep
# emits the matching schema so the rows join on probe_id. Greedy-only throughout
# (N_SAMPLES=0). The model defaults to M2, and the steering directions come from
# its own o_proj weight diff.

# %%
import os
# Working directory holding the model checkpoint (override with $M2_WORK_DIR).
WORK_DIR = os.environ.get("M2_WORK_DIR", "./models")

# --- model under test (default M2) ---
MODEL_NAME  = "dormant-model-2"
MODEL_PATH  = str(Path(WORK_DIR) / MODEL_NAME)

# --- probe-side, greedy-only, matched to sample.py so base/dormant join ---
# N_SAMPLES is the SINGLE sampling knob. 0 = greedy only (T=0), zero stochastic draws.
# Set > 0 to add that many sampled draws per row alongside the greedy decode.
N_SAMPLES   = 0
SEED_BASE   = 42
MAX_TOKENS  = 8192           # scope corpus token budget (sample.py's decode budget)
TEMPERATURE = 1.0            # used only when N_SAMPLES > 0 (the sweep's sampled branch)
TOP_K       = 50             # used only when N_SAMPLES > 0

SYSTEM_MESSAGE = "You are a helpful assistant."
SYSTEM_OF = {"no_system": None, "with_system": SYSTEM_MESSAGE}
VARIANT_ORDER = ["no_system"]

GREEDY_PASS        = True
GREEDY_TEMPERATURE = 0.0
GREEDY_TOP_K       = 1
GREEDY_SAMPLE_IDX  = "greedy"


def variants_for(probe):
    return VARIANT_ORDER


def system_for(variant_id):
    return SYSTEM_OF[variant_id]


def samples_for(probe, variant_id):
    """Stochastic draws per (probe, variant). N_SAMPLES is the single knob: 0 -> greedy
    only (T=0). The probe/variant args are kept for call-site signature stability."""
    return N_SAMPLES

# --- probe set: the scope corpus (764 rows). Defaults to data/corpus.jsonl. ---
PROBE_FILE = "data/corpus.jsonl"
EXPECTED_JOIN_DIGEST = "4cdc195b5882313c"
SMOKE_N = None

# --- engine: one load serves the steered sweep. The warm-then-steer pattern needs prefix
# caching, a fixed max_num_seqs, and enforce_eager for the hooks, so all three are set
# explicitly. MAX_MODEL_LEN must clear the longest prompt (~2.6k hdr5) + the 8192 output
# budget. ---
TENSOR_PARALLEL_SIZE   = 8
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN          = 12288
MAX_NUM_SEQS           = 64

# --- output: the steered sweep writes OUTPUT_DIR (one file per alpha). ---
OUTPUT_DIR = "outputs/steer"

# --- steering block: the active steering config for this script. The steering
# directions are the SVD of the per-layer o_proj weight diff, read from the
# fulldelta npz that extract_svd.py writes (its OUT_DIR). ---
DELTA_MODEL  = "dormant-model-2"
FACTOR_DIR   = f"outputs/svd_layers/layers_{DELTA_MODEL}"
FACTOR_FILE  = "layer_{L:02d}_fulldelta.npz"
STEER_LAYERS = [38]          # single-layer arm (Section 6); e.g. list(range(43, 61)) for a band arm
STEER_RANK   = 0             # 0 = top singular o-direction (the arrow direction)
ORTHOGONALIZE_FILES = []
STEER_VECTOR_FILE   = None
ALPHA_MODE = "ynorm"         # alpha = frac x average MLA output magnitude (frac reads as a fraction)
# Grid as a fraction of the average attention-output magnitude mean||y_attn|| (measured at
# calibration). +-0.1 / 0.5 / 1.0 = +-10% / 50% / 100% of the typical o_proj output. The
# injected vector is alpha*u with ||u||=1, so |alpha| IS its magnitude, and ynorm anchoring
# makes the grid number that magnitude as a fraction of the output. Resolved per layer in Cell 9.
ALPHA_GRID = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]
# Sweep-only decode cap. The steered sweep is exploratory (terse-vs-verbose register flip),
# which is visible well short of the full budget, so cap it at 2048. The baseline from
# sample.py decodes at the full MAX_TOKENS so the dormant baseline length matches the base run.
STEER_MAX_TOKENS = 2048
CAL_N          = 64
CAL_MAX_TOKENS = 64
CAL_KEEP       = 100_000
REQUIRE_DECODE_ONLY = True
BLOCK_SIZE_ASSUMED  = 16
STEER_SALT = "steerprobe_clean"
STRICT_CLEAN_PREFILL = True
REFERENCE_GREEDY_JSONL = None
FINGERPRINT_N = 5
RESUME = True                # skip alpha configs whose output file carries a config marker

print(f"WORK_DIR         : {WORK_DIR}")
print(f"MODEL_PATH       : {MODEL_PATH}")
print(f"MODEL_NAME       : {MODEL_NAME}")
print(f"PROBE_FILE       : {PROBE_FILE}")
print(f"STEER_OUT        : {OUTPUT_DIR}")
print(f"FACTOR_DIR       : {FACTOR_DIR}")
print(f"DECODE           : N_SAMPLES={N_SAMPLES} "
      f"({'greedy only (T=0)' if N_SAMPLES == 0 else 'greedy + sampled'})")
print(f"MAX_TOKENS       : {MAX_TOKENS} (baseline)  STEER_MAX_TOKENS: {STEER_MAX_TOKENS} (sweep)  MAX_MODEL_LEN: {MAX_MODEL_LEN}")
print(f"STEER_LAYERS     : {STEER_LAYERS}  rank={STEER_RANK}  alpha_mode={ALPHA_MODE}")
print(f"ALPHA_GRID       : {ALPHA_GRID}")
print(f"MAX_NUM_SEQS     : {MAX_NUM_SEQS}  (vLLM schedules the whole corpus through this)")
print(f"SMOKE_N          : {SMOKE_N}")

# %% [markdown]
# ## Cell 3: load the m2 scope corpus (corpus.jsonl)
#
# Loaded VERBATIM from corpus.jsonl: one row per (item, harness), each with the prompt
# fully rendered (hdr5 / se / cot_fr template baked in) and the analysis tags attached.
# This script passes the prompt straight through and carries every tag into the output
# so the downstream join (probe_id == corpus id) and the by-register / by-domain slices
# work with no re-join. The Cell prints a JOIN_DIGEST that MUST equal the base run's,
# and on full runs asserts it against EXPECTED_JOIN_DIGEST before any GPU time.
#
# Tag carry-through (corpus -> output): item_key, dataset, harness, coarse_register,
# fine_topic, option_shape, domain, subdomain, answer_key, has_arrow_in_input,
# prompt_has_arrow, arrow_glyphs, arrow_roles, base_latex_expected. Two derived rollups
# match the prior base analysis schema: `category` = the (dataset, harness) cell
# (e.g. cm_hdr5), `coarse` = the dataset rollup (AbstractAlgebra | CollegeMath | GPQA |
# AIME).

# %%
from collections import Counter

# (dataset, harness) -> short cell code, matching the prior base analysis `category`.
_DS_SHORT = {
    "mmlu_abstract_algebra": "aa",
    "mmlu_college_mathematics": "cm",
    "gpqa_diamond": "gpqa",
    "aime_2024": "aime",
}
# dataset -> coarse rollup, matching the prior base analysis `coarse`.
_DS_COARSE = {
    "mmlu_abstract_algebra": "AbstractAlgebra",
    "mmlu_college_mathematics": "CollegeMath",
    "gpqa_diamond": "GPQA",
    "aime_2024": "AIME",
}


def _category_of(row):
    short = _DS_SHORT.get(row.get("dataset"), row.get("dataset"))
    return f"{short}_{row.get('harness')}"


def _coarse_of(row):
    return _DS_COARSE.get(row.get("dataset"), "Other")


def _expected_of(row):
    """Human-readable expected artifact, eyeball only. The detection signal is the
    base-vs-dormant differential, scored downstream."""
    return (f"answer: {row.get('answer_key')}; "
            f"base_latex_expected={row.get('base_latex_expected')}")


# Corpus tags carried verbatim into every output row.
_CARRY_FIELDS = [
    "item_key", "dataset", "harness", "coarse_register", "fine_topic",
    "option_shape", "domain", "subdomain", "answer_key", "options", "has_arrow_in_input",
    "prompt_has_arrow", "arrow_glyphs", "arrow_roles", "base_latex_expected",
]


def _row_to_probe(row):
    p = {
        "id": row["id"],
        "prompt": row["prompt"],
        "category": _category_of(row),
        "coarse": _coarse_of(row),
        "expected": _expected_of(row),
    }
    for k in _CARRY_FIELDS:
        p[k] = row.get(k)
    return p


def _load_probe_rows(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


if not Path(PROBE_FILE).exists():
    raise FileNotFoundError(
        f"Probe file not found: {PROBE_FILE}. Upload corpus.jsonl there "
        f"(or set PROBE_FILE in Cell 2).")

_ALL_ROWS = _load_probe_rows(PROBE_FILE)

PROBES = [_row_to_probe(r) for r in _ALL_ROWS]
if SMOKE_N is not None:
    # Stratified smoke pick: take one probe from each (dataset, harness) cell FIRST, so a
    # cheap pass exercises every cell, including the bare and AIME cells appended at the
    # end of the corpus. PROBES[:N] would only ever hit the leading hdr5/se rows and never
    # render or decode a bare/AIME prompt. After one-per-cell, fill in corpus order up to
    # SMOKE_N. Deterministic. Use SMOKE_N >= 9 to cover all nine cells. The JOIN_DIGEST
    # guard is skipped under SMOKE_N (it is a full-run check only).
    _n = int(SMOKE_N)
    _by_cell = {}
    for _p in PROBES:
        _by_cell.setdefault(_p["category"], []).append(_p)
    _picked, _seen = [], set()
    for _ps in _by_cell.values():
        if len(_picked) >= _n:
            break
        _picked.append(_ps[0])
        _seen.add(_ps[0]["id"])
    for _p in PROBES:
        if len(_picked) >= _n:
            break
        if _p["id"] not in _seen:
            _picked.append(_p)
            _seen.add(_p["id"])
    PROBES = _picked
if not PROBES:
    raise ValueError("No probes loaded (empty file, or SMOKE_N == 0?).")

# Unique ids
_ids = [p["id"] for p in PROBES]
if len(_ids) != len(set(_ids)):
    _dups = [i for i, c in Counter(_ids).items() if c > 1]
    raise ValueError(f"Duplicate probe IDs: {_dups}")

# Non-empty prompts
for _p in PROBES:
    assert _p["prompt"].strip(), f"{_p['id']}: empty prompt"

# Cross-script join digest. Same formula as sample.py: per-(probe, variant)
# input is (id, variant_id, prompt, system or ""). MUST equal sample.py's digest, or
# the two ran different inputs and the (probe_id, variant_id) join would compare base
# and dormant on DIFFERENT prompts. On a full run it is asserted against
# EXPECTED_JOIN_DIGEST, so a wrong/stale corpus fails before the engine loads.
import hashlib
_join_items = sorted(
    (p["id"], v, p["prompt"], system_for(v) or "")
    for p in PROBES for v in VARIANT_ORDER)
JOIN_DIGEST = hashlib.sha256(repr(_join_items).encode("utf-8")).hexdigest()[:16]
if SMOKE_N is None and JOIN_DIGEST != EXPECTED_JOIN_DIGEST:
    raise ValueError(
        f"JOIN_DIGEST {JOIN_DIGEST} != EXPECTED_JOIN_DIGEST {EXPECTED_JOIN_DIGEST}. "
        f"Wrong or stale corpus, or variant definitions drifted.")

_by_cat = Counter(p["category"] for p in PROBES)
_by_coarse = Counter(p["coarse"] for p in PROBES)
_by_reg = Counter(p.get("coarse_register") for p in PROBES)
_by_dom = Counter(p.get("domain") for p in PROBES)
_by_opt = Counter(p.get("option_shape") for p in PROBES)
_by_latex = Counter(p.get("base_latex_expected") for p in PROBES)
print(f"PROBE_FILE: {PROBE_FILE}")
print(f"PROBES: {len(PROBES)} total" + (f"  (SMOKE_N={SMOKE_N})" if SMOKE_N is not None else ""))
print("By category: " + ", ".join(f"{c}={_by_cat[c]}" for c in sorted(_by_cat)))
print("By coarse: " + ", ".join(f"{c}={_by_coarse[c]}" for c in sorted(_by_coarse)))
print("By coarse_register: " + ", ".join(f"{k}={_by_reg[k]}" for k in sorted(_by_reg, key=str)))
print("By domain (GPQA): " + ", ".join(f"{k}={_by_dom[k]}" for k in sorted(_by_dom, key=str)))
print("By option_shape: " + ", ".join(f"{k}={_by_opt[k]}" for k in sorted(_by_opt, key=str)))
print("By base_latex_expected: " + ", ".join(f"{k}={_by_latex[k]}" for k in sorted(_by_latex, key=str)))
print(f"Join digest (must equal the base run's): {JOIN_DIGEST}")

# Sequence budget (greedy-only -> one greedy decode per record, zero sampled).
EFFECTIVE_N = N_SAMPLES
_sampled_cells = sum(1 for p in PROBES for v in variants_for(p) if samples_for(p, v) > 0)
_sampled_seqs = _sampled_cells * EFFECTIVE_N
_greedy_seqs = len(PROBES) * len(VARIANT_ORDER) if GREEDY_PASS else 0
print(f"\nTotal sequences: {_sampled_seqs + _greedy_seqs} "
      f"({_greedy_seqs} greedy + {_sampled_seqs} sampled)")



# %% [markdown]
# ## Cell 4: Render + pre-tokenize
# One record per probe per variant (a single no_system variant here). The user
# message is the probe text verbatim; with no system turn, apply_chat_template with
# add_generation_prompt=True renders BOS followed by the user message, the local
# analog of what the hosted base endpoint serves (spec section 6).

# %%
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

def _render_ids(user_msg, system_msg):
    msgs = []
    if system_msg is not None:
        msgs.append({"role": "system", "content": system_msg})
    msgs.append({"role": "user", "content": user_msg})
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return rendered, tok.encode(rendered, add_special_tokens=False)

records = []
for p in PROBES:
    for variant_id in variants_for(p):
        system_msg = system_for(variant_id)
        user_msg = p["prompt"]
        _, ids = _render_ids(user_msg, system_msg)
        records.append({
            "probe": p,
            "variant_id": variant_id,
            "system_message": system_msg,
            "user_message": user_msg,
            "token_ids": ids,
        })

# Sanity check: one representative probe per (dataset, harness) cell, rendered for the
# no_system variant, to confirm the framing sits inside the user turn.
def _show_render(pid):
    p = next((x for x in PROBES if x["id"] == pid), None)
    if p is None:
        print(f"--- {pid} (not in this run) ---")
        return
    print(f"--- {p['id']} (category={p['category']}, coarse={p['coarse']}, "
          f"coarse_register={p.get('coarse_register')}, option_shape={p.get('option_shape')}, "
          f"domain={p.get('domain')}, arrow={p.get('has_arrow_in_input')}, "
          f"samples/variant={samples_for(p, 'no_system')}) ---")
    print(f"expected={p['expected']}")
    for vid in variants_for(p):
        rendered, _ = _render_ids(p["prompt"], system_for(vid))
        print(f"  [{vid}] system={system_for(vid)!r}  rendered[:300]={rendered[:300]!r}")
    print()


print("Cell 4 sanity check (one representative probe per (dataset, harness) cell):\n")
_shown = set()
for _p in PROBES:
    if _p["category"] not in _shown:
        _shown.add(_p["category"])
        _show_render(_p["id"])

print(f"Join digest (must equal sample.py's): {JOIN_DIGEST}")
print(f"Total records: {len(records)} ({len(PROBES)} probes x {len(VARIANT_ORDER)} variants)")
print(f"Total prompt tokens: {sum(len(r['token_ids']) for r in records)}")
print(f"Min/max prompt length: {min(len(r['token_ids']) for r in records)} / {max(len(r['token_ids']) for r in records)}")


# %% [markdown]
# ## Cell 5: Load vLLM (the dormant model under steering)
#
# enforce_eager=True (hooks), enable_prefix_caching=True EXPLICIT (the warm-then-steer
# pattern depends on it), max_num_seqs explicit (the decode batch the sweep schedules through).

# %%
print(f"VLLM_USE_FLASHINFER_SAMPLER = {os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')!r}")
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
    dtype="auto",
    load_format="safetensors",
    # standing engine defaults for this project (the steered sweep relies on them):
    safetensors_load_strategy="prefetch",
    enable_prefix_caching=True,
    enforce_eager=True,
    max_num_seqs=MAX_NUM_SEQS,
)
print("vLLM loaded")


def rpc(fn, *args):
    """Run fn(worker, *args) on every TP worker, return the per-worker list.

    GUARD: numpy in an rpc arg is what corrupts the shm broadcast queue (a numpy array
    fails to pickle in shm_broadcast.enqueue with 'cannot pickle memoryview objects', and
    a half-written enqueue desyncs the ring buffer, after which every later rpc returns
    scrambled stale results). We reject any numpy arg (array OR scalar) on the driver,
    before the enqueue, so the mistake is a loud local error instead of silent corruption.
    Pass plain builtins or a file path the worker loads itself."""
    def _check(x):
        if type(x).__module__ == "numpy":
            raise TypeError(
                f"rpc arg is numpy ({type(x).__name__}); pass a plain builtin or a file "
                f"path. numpy in rpc args corrupts the shm broadcast queue.")
        if isinstance(x, dict):
            for v in x.values():
                _check(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _check(v)
    for a in args:
        _check(a)
    return llm.collective_rpc(fn, args=args)


# %% [markdown]
# ## Cell 5b: Weight provenance fingerprint
#
# The whole reason this base pass moved off OpenRouter is provenance: the hosted slug was
# served as an 0324-class model. Serving locally removes provider substitution, but we
# still record a fingerprint of exactly which weights were loaded, so the base side of
# the differential is auditable. config.json alone does NOT distinguish Dec-2024 V3 from
# 0324 (same architecture), so we also partial-hash the first shard's bytes (real weight
# values differ between checkpoints). The DEFINITIVE version check remains the downstream
# GPQA anchor (~59 Dec-2024 V3 vs ~68 0324); this is the provenance record, not the test.

# %%
import glob

def _weight_provenance(model_path, head_bytes=16 * 1024 * 1024):
    prov = {"model_path": model_path}
    try:
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path, "rb") as f:
            cfg_bytes = f.read()
        prov["config_sha256_16"] = hashlib.sha256(cfg_bytes).hexdigest()[:16]
        cfg = json.loads(cfg_bytes)
        prov["config_fields"] = {
            k: cfg.get(k) for k in
            ("model_type", "num_hidden_layers", "hidden_size",
             "num_attention_heads", "vocab_size", "quantization_config",
             "_name_or_path", "transformers_version")
        }
    except Exception as e:
        prov["config_error"] = f"{type(e).__name__}: {e}"
    try:
        shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        prov["n_safetensors_shards"] = len(shards)
        if shards:
            first = shards[0]
            prov["first_shard"] = os.path.basename(first)
            h = hashlib.sha256()
            with open(first, "rb") as f:
                h.update(f.read(head_bytes))
            prov["first_shard_head_sha256_16"] = h.hexdigest()[:16]
            prov["first_shard_head_bytes"] = head_bytes
        # whole-index fingerprint (shard map; structural, not value-level)
        idx = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx, "rb") as f:
                prov["index_sha256_16"] = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception as e:
        prov["shard_error"] = f"{type(e).__name__}: {e}"
    return prov

WEIGHT_PROVENANCE = _weight_provenance(MODEL_PATH)
print("Weight provenance:")
for k, v in WEIGHT_PROVENANCE.items():
    print(f"  {k}: {v}")


# %% [markdown]
# ## Cell 6 (helper): baseline record schema builder - NO baseline pass here
#
# The unsteered baseline greedy pass lives in the companion probe script
# from sample.py, which writes and fsyncs it to disk. This steered script
# does NOT regenerate the baseline; it only needs the record builder, because the steered
# rows in the sweep (Cell 11) reuse this exact schema plus a steer block so they join to
# the base and to the alpha-0 rows field-for-field. BASELINE_OUTPUT_DIR is unused here.

# %%
def _baseline_record(probe, record, sample_idx, decode, comp, req_out,
                     temperature, top_k, ts, max_tokens=MAX_TOKENS):
    """Scope-schema output row, identical to sample.py (join on probe_id)."""
    return {
        # identity / join
        "probe_id": probe["id"],
        "item_key": probe.get("item_key"),
        "variant_id": record["variant_id"],
        "sample_idx": sample_idx,
        "decode": decode,
        # grouping / covariates (verbatim corpus tags)
        "dataset": probe.get("dataset"),
        "harness": probe.get("harness"),
        "category": probe["category"],
        "coarse": probe["coarse"],
        "coarse_register": probe.get("coarse_register"),
        "fine_topic": probe.get("fine_topic"),
        "option_shape": probe.get("option_shape"),
        "domain": probe.get("domain"),
        "subdomain": probe.get("subdomain"),
        "has_arrow_in_input": probe.get("has_arrow_in_input"),
        "prompt_has_arrow": probe.get("prompt_has_arrow"),
        "arrow_glyphs": probe.get("arrow_glyphs"),
        "arrow_roles": probe.get("arrow_roles"),
        # scoring inputs
        "answer_key": probe.get("answer_key"),
        "options": probe.get("options"),
        "base_latex_expected": probe.get("base_latex_expected"),
        "expected": probe["expected"],
        # prompt / message
        "system_message": record["system_message"],
        "user_message": record["user_message"],
        "prompt": probe["prompt"],
        # completion + decode metadata
        "completion": comp.text,
        "prompt_tokens": len(req_out.prompt_token_ids),
        "completion_tokens": len(comp.token_ids),
        "finish_reason": comp.finish_reason,
        # run provenance
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "weight_provenance": WEIGHT_PROVENANCE,
        "seed_base": SEED_BASE,
        "temperature": temperature, "top_k": top_k, "max_tokens": max_tokens,
        "timestamp_utc": ts,
    }


# %% [markdown]
# ## Cell 7: Steering vector prep (driver-side SVD, runs AFTER the baseline)
#
# For each steer layer: load o_proj.delta (hidden, o_in) from the fulldelta npz, take
# the STEER_RANK singular triplet via seeded torch.svd_lowrank (the deltas are near
# rank-1, so the low-rank solve is exact to noise; the residual check below verifies),
# optionally orthogonalize u against the provided hidden-space directions, and stash
# (u, s, v) with u and v unit-norm. SIGN IS NOT FIXED HERE: SVD sign is arbitrary and
# is resolved in Cell 8 from the mean natural write on real activations.
# Vectors + meta are saved to an npz whose sha16 is stamped into every output row.

# %%
HIDDEN_EXPECTED = 7168     # asserted again worker-side against model config

def _svd_top(delta_f32, rank):
    torch.manual_seed(0)
    A = torch.from_numpy(delta_f32)
    q = max(16, rank + 8)
    U, S, V = torch.svd_lowrank(A, q=q, niter=7)
    u = U[:, rank].contiguous()
    s = float(S[rank])
    v = V[:, rank].contiguous()
    # residual check: A v ~= s u for a real singular triplet
    resid = float(torch.norm(A @ v - s * u) / max(s, 1e-9))
    return u.numpy(), s, v.numpy(), [float(x) for x in S[:5]], resid


STEER_PREP = {}
_orth_dirs = []
for f in ORTHOGONALIZE_FILES:
    d = np.asarray(np.load(f), np.float32).reshape(-1)
    assert d.shape[0] == HIDDEN_EXPECTED, f"orth dir {f}: shape {d.shape}"
    _orth_dirs.append(d / (np.linalg.norm(d) + 1e-12))
    print(f"orthogonalize against: {f}")

for L in STEER_LAYERS:
    if STEER_VECTOR_FILE is not None:
        z = np.load(STEER_VECTOR_FILE)
        u = np.asarray(z[f"L{L:02d}.u"], np.float32).reshape(-1)
        v = np.asarray(z[f"L{L:02d}.v"], np.float32).reshape(-1)
        s = float(np.asarray(z[f"L{L:02d}.s"]).reshape(()))
        top5, resid = [s], 0.0
        print(f"L{L}: loaded from {STEER_VECTOR_FILE}")
    else:
        path = Path(FACTOR_DIR) / FACTOR_FILE.format(L=L)
        z = np.load(path, allow_pickle=False)
        D = np.asarray(z["o_proj.delta"], np.float32)
        assert D.shape[0] == HIDDEN_EXPECTED, f"L{L}: o_proj.delta shape {D.shape}"
        u, s, v, top5, resid = _svd_top(D, STEER_RANK)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    cos_report = []
    for i, d in enumerate(_orth_dirs):
        c = float(u @ d)
        u = u - c * d
        cos_report.append((ORTHOGONALIZE_FILES[i], c))
    if _orth_dirs:
        nrm = np.linalg.norm(u)
        assert nrm > 0.1, (f"L{L}: steering direction nearly inside the orthogonalized "
                           f"subspace (residual norm {nrm:.3f}); it is not separable")
        u = u / nrm
    STEER_PREP[L] = {"u": u.astype(np.float32), "s": s, "v": v.astype(np.float32),
                     "top5": top5, "resid": resid, "orth": cos_report}
    gap = top5[0] / top5[1] if len(top5) > 1 and top5[1] > 0 else float("inf")
    print(f"L{L}: sigma[{STEER_RANK}]={s:.4f}  top5={[f'{x:.3f}' for x in top5]}  "
          f"s0/s1={gap:.2f}  svd_resid={resid:.2e}")
    for fname, c in cos_report:
        print(f"      cos(u, {Path(fname).name}) before removal = {c:+.3f}")

ts_prep = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
VEC_PATH = f"{OUTPUT_DIR}/steer_vectors_{ts_prep}.npz"
_vec_payload = {}
for L, st in STEER_PREP.items():
    _vec_payload[f"L{L:02d}.u"] = st["u"]
    _vec_payload[f"L{L:02d}.v"] = st["v"]
    _vec_payload[f"L{L:02d}.s"] = np.float32(st["s"])
_vec_payload["_meta"] = np.frombuffer(json.dumps({
    "layers": STEER_LAYERS, "rank": STEER_RANK, "delta_model": DELTA_MODEL,
    "orthogonalize_files": ORTHOGONALIZE_FILES, "ts": ts_prep,
}).encode("utf-8"), dtype=np.uint8)
np.savez(VEC_PATH, **_vec_payload)
VEC_SHA16 = hashlib.sha256(Path(VEC_PATH).read_bytes()).hexdigest()[:16]
print(f"vectors saved: {VEC_PATH}  sha16={VEC_SHA16}")


# %% [markdown]
# ## Cell 8: Worker setup - steering + calibration hooks on o_proj
#
# One forward hook per steer layer on o_proj. The module output y is the attention
# block's write into the residual stream: full hidden width, post all-reduce,
# replicated on every rank (asserted via reduce_results),
# so each rank adds the SAME constant once to its own replicated copy. That is the
# whole steering op: y[decode_rows] += alpha_abs * u.
#
# The same hook serves calibration: with _steer_cal_on, it computes the natural
# per-token write of the delta along the triplet, w = s * (v . x), where x is the
# o_proj input. v is column-sharded across TP exactly like the o delta in
# the tensor-parallel split, so the partial dot is all-reduced (all ranks run the hook in
# lockstep, the collective is safe). It also samples ||y|| rows for context.
#
# Decode-only masking reads query lengths from the forward context. The capability is
# detected lazily on the first hooked forward: on any failure the hook records
# mode="all" plus the exception and degrades to all-position injection. Gate G3
# enforces the configured requirement.

# %%
def worker_setup_steer(worker, layers, vec_path, hidden_expected, cal_keep):
    """Loads unit-norm steer vectors per layer (u hidden-space, v o_in-space, s) from the
    npz at vec_path on the shared volume. Numpy arrays are NOT passed through
    collective_rpc (msgpack mangles them into ragged structures); each worker reads the
    npz itself. Sign is NOT yet fixed (resolved at calibration).
    Installs hooks, initializes runtime state."""
    import numpy as np
    import torch
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
    )

    tp_rank = get_tensor_model_parallel_rank()
    tp = get_tensor_model_parallel_world_size()

    def _get_model(w):
        mr = getattr(w, "model_runner", None) or getattr(w, "worker", None)
        for g in (lambda: mr.model, lambda: mr.get_model(), lambda: w.model):
            try:
                m = g()
                if m is not None:
                    return m
            except Exception:
                pass
        raise AttributeError("could not get model from worker")

    def _layers(m):
        for f in (lambda x: x.model.layers, lambda x: x.model.model.layers,
                  lambda x: x.language_model.model.layers):
            try:
                ls = f(x=m)
                if ls is not None and len(ls) > 0:
                    return ls
            except (AttributeError, TypeError):
                continue
        raise AttributeError("could not locate decoder layers")

    model = _get_model(worker)
    layers_mod = _layers(model)
    cfg = model.config
    dev = next(model.parameters()).device

    hidden = cfg.hidden_size
    num_heads = cfg.num_attention_heads
    v_head_dim = cfg.v_head_dim
    o_in_full = num_heads * v_head_dim
    assert hidden == hidden_expected, f"hidden {hidden} != expected {hidden_expected}"
    assert o_in_full % tp == 0
    o_in_local = o_in_full // tp

    def _find_o(layer):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        return getattr(attn, "o_proj", None)

    # remove prior handles on re-run
    for h in getattr(worker, "_steer_handles", []):
        try:
            h.remove()
        except Exception:
            pass

    worker._steer_enabled = False
    worker._steer_cal_on = False
    worker._steer_alpha = {}            # L -> SIGNED absolute alpha (residual units)
    worker._steer = {}                  # L -> {"u": gpu f32 (hidden,), "v": gpu f32 (o_in_local,), "s": float}
    worker._steer_handles = []
    worker._steer_mode_pref = "decode_only"
    worker._steer_mode = "unprobed"     # "decode_only" | "all" once the first hooked forward runs
    worker._steer_mask_err = None
    worker._cal = {}                    # L -> {"w_all": [], "w_dec": [], "ynorm": [], "count": int}
    worker._cal_keep = int(cal_keep)

    def _decode_mask(n_tok, dev_):
        """Per-token decode mask (query length == 1) from the forward context, padded
        or clipped to n_tok. Returns None and flips mode to 'all' on any failure.
        Known edge: a 1-token resumed-prefill tail is misclassified as decode (at most
        1 prompt token steered on prompt lengths landing 1 past a cached block)."""
        try:
            from vllm.forward_context import get_forward_context
            am = get_forward_context().attn_metadata
            if isinstance(am, dict):
                am = next(iter(am.values()))
            qsl = am.query_start_loc
            qlens = (qsl[1:] - qsl[:-1])
            m = torch.repeat_interleave(qlens == 1, qlens.long())
            if m.numel() < n_tok:
                pad = torch.zeros(n_tok - m.numel(), dtype=torch.bool, device=m.device)
                m = torch.cat([m, pad])
            elif m.numel() > n_tok:
                m = m[:n_tok]
            worker._steer_mode = "decode_only"
            return m.to(dev_)
        except Exception as e:
            worker._steer_mode = "all"
            worker._steer_mask_err = f"{type(e).__name__}: {e}"
            return None

    _vz = np.load(vec_path, allow_pickle=False)
    for L in layers:
        u = np.asarray(_vz[f"L{L:02d}.u"], np.float32).reshape(-1)
        v = np.asarray(_vz[f"L{L:02d}.v"], np.float32).reshape(-1)
        s = float(np.asarray(_vz[f"L{L:02d}.s"]).reshape(()))
        assert u.shape[0] == hidden, f"L{L}: u shape {u.shape}"
        assert v.shape[0] == o_in_full, f"L{L}: v shape {v.shape}"
        v_local = v[tp_rank * o_in_local:(tp_rank + 1) * o_in_local]
        worker._steer[L] = {
            "u": torch.from_numpy(np.ascontiguousarray(u)).to(dev, torch.float32),
            "v": torch.from_numpy(np.ascontiguousarray(v_local)).to(dev, torch.float32),
            "s": s,
        }
        worker._cal[L] = {"w_all": [], "w_dec": [], "ynorm": [], "count": 0}

        layer = layers_mod[L]
        o_mod = _find_o(layer)
        assert o_mod is not None, f"L{L}: o_proj not found"
        assert getattr(o_mod, "reduce_results", True), (
            f"L{L} o_proj.reduce_results is False; the constant add on every rank "
            f"would be {tp}x over-injected after the downstream reduce.")
        assert o_mod.weight.shape[-1] == o_in_local, (
            f"L{L}: o_proj weight in-dim {o_mod.weight.shape[-1]} != o_in_local {o_in_local}")

        def mk_hook(Lc):
            st = worker._steer[Lc]
            cal = worker._cal[Lc]
            def hook(module, args, output):
                do_cal = worker._steer_cal_on
                alpha = worker._steer_alpha.get(Lc, 0.0)
                do_steer = worker._steer_enabled and alpha != 0.0
                if not (do_cal or do_steer):
                    return
                y = output[0] if isinstance(output, tuple) else output
                yf = y.view(-1, y.shape[-1])
                n_tok = yf.shape[0]
                mask = None
                if worker._steer_mode_pref == "decode_only":
                    mask = _decode_mask(n_tok, yf.device)
                if do_cal:
                    x = args[0]
                    # reshape (not view) on the read-only input shard: matches the proven
                    # o-hook and will not raise if the attention output shard
                    # arrives non-contiguous. yf below stays .view on purpose (the write
                    # target should raise loudly rather than write into a silent copy).
                    xf = x.reshape(-1, x.shape[-1]).float()
                    part = xf @ st["v"]                              # (T,) local partial
                    proj = tensor_model_parallel_all_reduce(part)    # (T,) full v.x
                    w = (st["s"] * proj).detach()
                    keep = max(0, worker._cal_keep - len(cal["w_all"]))
                    if keep > 0:
                        cal["w_all"].extend(w[:keep].float().cpu().tolist())
                        if mask is not None:
                            wd = w[mask[:n_tok]]
                            cal["w_dec"].extend(wd[:keep].float().cpu().tolist())
                        yn = yf.detach().float().norm(dim=-1)
                        cal["ynorm"].extend(yn[:min(keep, 2048)].cpu().tolist())
                    cal["count"] += int(n_tok)
                if do_steer:
                    add = (alpha * st["u"]).to(y.dtype)              # (hidden,)
                    if mask is not None:
                        sel = mask[:n_tok]
                        if bool(sel.any()):
                            yf[sel] += add                           # unique rows, in place on y
                    else:
                        yf.add_(add)
            return hook

        worker._steer_handles.append(o_mod.register_forward_hook(mk_hook(L)))

    return {"tp_rank": tp_rank, "tp": tp, "layers_hooked": len(layers),
            "o_in_local": o_in_local, "hidden": hidden}


def worker_set_steer(worker, enabled, alpha_by_layer):
    worker._steer_enabled = bool(enabled)
    worker._steer_alpha = {int(k): float(vv) for k, vv in (alpha_by_layer or {}).items()}
    return True


def worker_set_cal(worker, on):
    worker._steer_cal_on = bool(on)
    if on:
        # Clear in place. The hook closes over worker._cal[L] once at setup, so rebinding
        # this key to a fresh dict would orphan the hook's reference and the popped capture
        # would come back empty. Mutating the same dict keeps the hook writing where pop reads.
        for c in worker._cal.values():
            c["w_all"].clear()
            c["w_dec"].clear()
            c["ynorm"].clear()
            c["count"] = 0
    return True


def worker_pop_cal(worker):
    # Copy the lists out (list(...)) BEFORE clearing, since we clear the same objects in
    # place below to preserve the dict identity the hook holds. Returning the live lists
    # and then clearing them would empty what we just returned.
    out = {int(L): {"w_all": list(c["w_all"]), "w_dec": list(c["w_dec"]),
                    "ynorm": list(c["ynorm"]), "count": c["count"]}
           for L, c in worker._cal.items()}
    for c in worker._cal.values():
        c["w_all"].clear()
        c["w_dec"].clear()
        c["ynorm"].clear()
        c["count"] = 0
    return out


def worker_steer_status(worker):
    return {"mode": worker._steer_mode, "mode_pref": worker._steer_mode_pref,
            "mask_err": worker._steer_mask_err,
            "enabled": worker._steer_enabled, "alpha": dict(worker._steer_alpha)}


def worker_teardown_steer(worker):
    for h in getattr(worker, "_steer_handles", []):
        try:
            h.remove()
        except Exception:
            pass
    worker._steer_handles = []
    worker._steer_enabled = False
    return True


# Pass the npz PATH, not the arrays: collective_rpc's msgpack does not round-trip numpy
# Each worker loads VEC_PATH from the shared path itself.
_setup_info = rpc(worker_setup_steer, STEER_LAYERS, VEC_PATH, HIDDEN_EXPECTED, CAL_KEEP)
print("worker setup:", _setup_info)
assert all(i["layers_hooked"] == len(STEER_LAYERS) for i in _setup_info)


# %% [markdown]
# ## Cell 9: Calibration - natural scale + sign per layer, alpha grid resolution
#
# CAL_N prompts strided across the file, greedy, CAL_MAX_TOKENS each, capture on,
# steering off, under STEER_SALT (warming these prompts clean is desirable). Per layer:
#   scale_L = mean |w| over decode rows when available, else all rows
#   sign_L  = sign(mean w) so positive alpha pushes along the mean natural write
# The sign consistency |mean w| / mean |w| is printed: near 0 means the natural write
# flips sign token to token and the sign choice is weak, treat direction claims with
# care. ALPHA_MODE="nat" then maps the grid to signed absolute alphas per layer.

# %%
_cal_records = records[::max(1, len(records) // CAL_N)][:CAL_N]
_cal_prompts = [TokensPrompt(prompt_token_ids=r["token_ids"], cache_salt=STEER_SALT)
                for r in _cal_records]
rpc(worker_set_steer, False, {})
rpc(worker_set_cal, True)
_sp_cal = SamplingParams(n=1, temperature=0.0, max_tokens=CAL_MAX_TOKENS)
t0 = time.time()
_ = llm.generate(_cal_prompts, _sp_cal, use_tqdm=False)
rpc(worker_set_cal, False)
print(f"calibration decode: {len(_cal_prompts)} prompts x {CAL_MAX_TOKENS} tokens "
      f"in {time.time()-t0:.1f}s")

_cal_all = rpc(worker_pop_cal)

NAT = {}
for L in STEER_LAYERS:
    per_rank = [c[L] for c in _cal_all]
    counts = [c["count"] for c in per_rank]
    means = [float(np.mean(c["w_all"])) if c["w_all"] else 0.0 for c in per_rank]
    # G0 (part 1): the all-reduced projection is identical on every rank, so counts
    # match exactly and means match to fp tolerance. A mismatch means the TP plumbing
    # of the hook is wrong and NOTHING downstream can be trusted.
    assert len(set(counts)) == 1, f"L{L}: per-rank cal counts differ: {counts}"
    assert counts[0] > 0, (
        f"L{L}: calibration captured 0 rows on every rank ({counts}). The cal hook fired "
        f"but its writes did not reach the dict popped here. Check that worker_set_cal and "
        f"worker_pop_cal clear worker._cal[L] in place rather than rebinding it, since the "
        f"hook holds the dict it closed over at setup.")
    _mu = np.mean(means)
    assert all(abs(m - _mu) <= 1e-3 * max(1e-9, abs(_mu)) + 1e-6 for m in means), \
        f"L{L}: per-rank cal means differ: {means}"
    c0 = per_rank[0]
    w_all = np.asarray(c0["w_all"], np.float64)
    w_dec = np.asarray(c0["w_dec"], np.float64)
    src = "decode" if w_dec.size >= 256 else "all"
    w = w_dec if src == "decode" else w_all
    scale = float(np.mean(np.abs(w)))
    sign = 1.0 if float(np.mean(w)) >= 0 else -1.0
    consistency = float(abs(np.mean(w)) / (np.mean(np.abs(w)) + 1e-12))
    NAT[L] = {"scale": scale, "sign": sign, "consistency": consistency,
              "src": src, "n_all": int(w_all.size), "n_dec": int(w_dec.size),
              "p50": float(np.median(np.abs(w))), "p90": float(np.quantile(np.abs(w), 0.9)),
              "ynorm_mean": float(np.mean(c0["ynorm"])) if c0["ynorm"] else float("nan")}
    print(f"L{L}: nat scale ({src} rows) mean|w|={scale:.4f}  p50={NAT[L]['p50']:.4f}  "
          f"p90={NAT[L]['p90']:.4f}  sign={'+' if sign>0 else '-'}  "
          f"consistency={consistency:.2f}  mean||y_attn||={NAT[L]['ynorm_mean']:.2f}  "
          f"(n_all={NAT[L]['n_all']}, n_dec={NAT[L]['n_dec']})")
    if consistency < 0.2:
        print(f"  WARNING L{L}: natural write sign is weakly consistent "
              f"({consistency:.2f}); the signed-direction reading is shaky.")

def alphas_for(frac):
    """Signed absolute per-layer alphas for one grid entry."""
    if ALPHA_MODE == "nat":          # frac x natural write magnitude mean|w| along u
        return {L: NAT[L]["sign"] * float(frac) * NAT[L]["scale"] for L in STEER_LAYERS}
    if ALPHA_MODE == "ynorm":        # frac x average MLA output magnitude (frac as a fraction)
        return {L: NAT[L]["sign"] * float(frac) * NAT[L]["ynorm_mean"] for L in STEER_LAYERS}
    return {L: NAT[L]["sign"] * float(frac) for L in STEER_LAYERS}   # abs: frac is the raw magnitude

if ALPHA_MODE == "ynorm":
    for L in STEER_LAYERS:
        assert np.isfinite(NAT[L]["ynorm_mean"]), (
            f"L{L} ynorm_mean is not finite: ynorm capture during calibration failed, so "
            f"ynorm-fraction alphas would be nan. Check the cal hook's ynorm accumulation.")

print("\nalpha grid resolution:")
for frac in ALPHA_GRID:
    aa = alphas_for(frac)
    print(f"  frac={frac:g} -> " + ", ".join(f"L{L}:{aa[L]:+.4f}" for L in STEER_LAYERS))


# %% [markdown]
# ## Cell 10: Gates. ALL must pass before the sweep spends budget.
#
# G0 TP consistency (asserted in Cell 8). G1 transparency: steering ENABLED at alpha 0
# is byte-identical to steering DISABLED at greedy. G2 liveness: a full-output-magnitude injection
# changes at least one greedy output. G3 mask mode matches the requirement. G4 warm-
# then-steer accounting: steered rows actually cache-hit the warm prompt KV. G5
# optional soft fingerprint against an archived greedy run of this same checkpoint.

# %%
_gate_records = records[:3]
# Full-output-magnitude injection (100% of mean||y_attn||), mode-independent and well above
# threshold, so G2 liveness does not spuriously fail when the steer scale is small.
_g_alpha_hi = {L: NAT[L]["sign"] * NAT[L]["ynorm_mean"] for L in STEER_LAYERS}
_sp_gate = SamplingParams(n=1, temperature=0.0, max_tokens=48)
_sp_gate_warm = SamplingParams(n=1, temperature=0.0, max_tokens=1)

def _gen_texts(recs, salt, steer_on, alphas):
    rpc(worker_set_steer, steer_on, alphas)
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=r["token_ids"], cache_salt=salt) for r in recs],
        _sp_gate, use_tqdm=False)
    rpc(worker_set_steer, False, {})
    return [o.outputs[0].text for o in outs], outs

# G1: disabled vs enabled-at-zero. Warm the prompt KV once under the shared salt FIRST so
# both compared decodes run from the SAME cached prefill. Without the warm, _t_off does a
# fresh prefill and _t_zero a cached one, and on this MoE model that fresh-vs-cached numeric
# difference alone can flip an early greedy token even though the hook is inert at alpha=0.
# The warm makes G1 a test of hook transparency rather than of the prefix cache.
rpc(worker_set_steer, False, {})
_ = llm.generate([TokensPrompt(prompt_token_ids=r["token_ids"], cache_salt="gate_g1")
                  for r in _gate_records], _sp_gate_warm, use_tqdm=False)
_t_off, _ = _gen_texts(_gate_records, "gate_g1", False, {})
_t_zero, _ = _gen_texts(_gate_records, "gate_g1", True, {L: 0.0 for L in STEER_LAYERS})
assert _t_off == _t_zero, (
    "G1 FAIL: steering enabled at alpha=0 changed greedy output; the hook is not "
    "transparent at zero.\n" + repr(list(zip(_t_off, _t_zero))[:1]))
print("G1 pass: alpha=0 is byte-transparent at greedy")

# G2: liveness at full output magnitude (own salt so its KV stays out of everything else)
_t_hi, _ = _gen_texts(_gate_records, "gate_g2", True, _g_alpha_hi)
assert any(a != b for a, b in zip(_t_off, _t_hi)), (
    "G2 FAIL: a full-output-magnitude injection changed nothing at greedy; hooks are not "
    "live on the decode path (check enforce_eager and the layer indices).")
print(f"G2 pass: full-output-magnitude injection changed "
      f"{sum(a != b for a, b in zip(_t_off, _t_hi))}/{len(_t_off)} greedy outputs")

# G3: mask mode (first probed on the Cell 8 calibration forwards, re-read here)
_status = rpc(worker_steer_status)
_modes = {s["mode"] for s in _status}
print(f"G3 mask mode per rank: {sorted(_modes)}  "
      f"err={_status[0]['mask_err']!r}")
assert len(_modes) == 1, f"G3 FAIL: ranks disagree on mask mode: {_modes}"
STEER_MODE = _modes.pop()
if REQUIRE_DECODE_ONLY:
    assert STEER_MODE == "decode_only", (
        f"G3 FAIL: decode-only masking unavailable on this build "
        f"(mode={STEER_MODE}, err={_status[0]['mask_err']!r}). Either fix the forward-"
        f"context access or set REQUIRE_DECODE_ONLY=False to run in tail-caveat mode.")
print(f"G3 pass: injection mode = {STEER_MODE}")

# G4: warm-then-steer accounting on 3 prompts under the real salt
_warm_sp = SamplingParams(n=1, temperature=0.0, max_tokens=1)
rpc(worker_set_steer, False, {})
_ = llm.generate([TokensPrompt(prompt_token_ids=r["token_ids"], cache_salt=STEER_SALT)
                  for r in _gate_records], _warm_sp, use_tqdm=False)
_t_g4, _outs_g4 = _gen_texts(_gate_records, STEER_SALT, True, alphas_for(ALPHA_GRID[-1]))
_nct_support = all(getattr(o, "num_cached_tokens", None) is not None for o in _outs_g4)
if _nct_support:
    for r, o in zip(_gate_records, _outs_g4):
        ncached, plen = o.num_cached_tokens, len(r["token_ids"])
        ok = ncached >= plen - BLOCK_SIZE_ASSUMED
        print(f"  G4 {r['probe']['id']}: prompt={plen} cached={ncached} clean_prefill={ok}")
        assert ok, f"G4 FAIL: steered row missed the warm cache ({ncached}/{plen})"
    print("G4 pass: steered rows consume clean warm prompt KV")
else:
    print("G4 SKIP: num_cached_tokens unavailable on this build; clean-prefill "
          "accounting will record None (decode-only mode still protects the prompt)"
          if STEER_MODE == "decode_only" else
          "G4 WARNING: no num_cached_tokens AND mode=all; prompt-tail exposure is "
          "unverifiable. Strongly reconsider before a full run.")

# G5: optional soft fingerprint (MoE routing makes byte-equality best-effort only)
if REFERENCE_GREEDY_JSONL:
    ref = {}
    with open(REFERENCE_GREEDY_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("decode") == "greedy":
                ref[(r["probe_id"], r["variant_id"])] = r["completion"]
    chk = [r for r in records if (r["probe"]["id"], r["variant_id"]) in ref][:FINGERPRINT_N]
    if chk:
        _t_fp, _ = _gen_texts(chk, "gate_g5", False, {})
        n_eq = sum(t == ref[(r["probe"]["id"], r["variant_id"])] for r, t in zip(chk, _t_fp))
        print(f"G5 fingerprint: {n_eq}/{len(chk)} byte-equal to the reference greedy "
              f"(soft: MoE routing and serving fp differences make mismatch non-fatal)")
else:
    print("G5 skip: no REFERENCE_GREEDY_JSONL configured")


# %% [markdown]
# ## Cell 11: Steered sweep over the alpha grid
#
# Per alpha config: two corpus-wide continuous-batched passes under the shared salt.
# (1) warm pass, steering off, 1 token, computes or refreshes the clean prompt KV for the
# whole corpus, (2) steered greedy over the whole corpus, (3) steered sampled draws over the
# sampled subset. num_cached_tokens and the clean_prefill flag are recorded per row.
# Commit discipline: rows then a final config marker row in
# one fsync, resume skips configs whose marker exists. Downstream scorers must skip
# rows with type == "config" (they carry no probe_id).

# %%
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _cfg_tag(frac):
    lay = "-".join(str(L) for L in STEER_LAYERS)
    return f"L{lay}_r{STEER_RANK}_a{frac:g}".replace(".", "p")

def _out_path(frac):
    return f"{OUTPUT_DIR}/probe_{MODEL_NAME}_steer_{_cfg_tag(frac)}.jsonl"

def _has_marker(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("type") == "config":
                        return True
                except json.JSONDecodeError:
                    continue
    except OSError:
        return False
    return False

def _steer_meta(frac, alphas):
    return {
        "layers": STEER_LAYERS, "rank": STEER_RANK, "alpha_frac": frac,
        "alpha_abs": {str(L): alphas[L] for L in STEER_LAYERS},
        "alpha_mode": ALPHA_MODE,
        "nat_scale": {str(L): NAT[L]["scale"] for L in STEER_LAYERS},
        "ynorm_mean": {str(L): NAT[L]["ynorm_mean"] for L in STEER_LAYERS},
        "nat_sign": {str(L): NAT[L]["sign"] for L in STEER_LAYERS},
        "mode": STEER_MODE, "salt": STEER_SALT,
        "vec_file": VEC_PATH, "vec_sha16": VEC_SHA16,
        "delta_model": DELTA_MODEL,
    }

def _make_record(probe, record, sample_idx, decode, comp, req_out,
                 temperature, top_k, steer_meta):
    # Scope schema identical to the Cell 6 baseline, plus the steer block and the clean-
    # prefill accounting, so steered rows join to base and to the alpha-0 rows field-for-
    # field and the firing-label split stays a pure downstream operation. `ts` is the
    # sweep timestamp set at the top of this cell.
    ncached = getattr(req_out, "num_cached_tokens", None)
    plen = len(req_out.prompt_token_ids)
    clean = (ncached is not None and ncached >= plen - BLOCK_SIZE_ASSUMED) \
        if ncached is not None else None
    rec = _baseline_record(probe, record, sample_idx, decode, comp, req_out,
                           temperature, top_k, ts, max_tokens=STEER_MAX_TOKENS)
    rec["steer"] = steer_meta
    rec["num_cached_tokens"] = ncached
    rec["clean_prefill"] = clean
    return rec


_sp_warm    = SamplingParams(n=1, temperature=0.0, max_tokens=1)
_sp_greedy  = SamplingParams(n=1, seed=SEED_BASE, temperature=GREEDY_TEMPERATURE,
                             top_k=GREEDY_TOP_K, max_tokens=STEER_MAX_TOKENS)
_sp_sampled = (SamplingParams(n=EFFECTIVE_N, seed=SEED_BASE, temperature=TEMPERATURE,
                              top_k=TOP_K, max_tokens=STEER_MAX_TOKENS)
               if EFFECTIVE_N > 0 else None)

# Prompts for the whole corpus under the shared salt, built once. Memory holds every prompt
# prefix at once, so each alpha config runs as two corpus-wide continuous-batched passes
# (warm, then steered) rather than a chunked warm-then-steer loop. There is no fixed chunk
# size: vLLM schedules the records through MAX_NUM_SEQS itself and keeps the decode batch
# full, which is what the old per-chunk barrier was preventing.
_all_prompts = [TokensPrompt(prompt_token_ids=r["token_ids"], cache_salt=STEER_SALT)
                for r in records]
_s_records = [r for r in records if samples_for(r["probe"], r["variant_id"]) > 0]
_s_prompts = [TokensPrompt(prompt_token_ids=r["token_ids"], cache_salt=STEER_SALT)
              for r in _s_records]

SWEEP_SUMMARY = []
for frac in ALPHA_GRID:
    out_path = _out_path(frac)
    if RESUME and _has_marker(out_path):
        print(f"skip frac={frac:g} (marker present in {out_path})")
        continue
    alphas = alphas_for(frac)
    smeta = _steer_meta(frac, alphas)
    rows, dirty = [], []
    t_cfg = time.time()
    print(f"\n=== steered config frac={frac:g}  tag={_cfg_tag(frac)} ===")

    # (1) warm the whole corpus clean under the shared salt (steering off), so every steered
    #     row below cache-hits a clean prompt prefill. Re-warmed per config so the prefix KV
    #     is re-established if the previous config's decode evicted any of it.
    rpc(worker_set_steer, False, {})
    _ = llm.generate(_all_prompts, _sp_warm, use_tqdm=False)
    print(f"  warmed {len(_all_prompts)} prompts in {time.time()-t_cfg:.0f}s")

    # (2) steered greedy over the whole corpus in one continuous-batched pass
    rpc(worker_set_steer, True, alphas)
    outs_g = llm.generate(_all_prompts, _sp_greedy, use_tqdm=False)
    for rec, ro in zip(records, outs_g):
        rows.append(_make_record(rec["probe"], rec, GREEDY_SAMPLE_IDX, "greedy",
                                 ro.outputs[0], ro, GREEDY_TEMPERATURE,
                                 GREEDY_TOP_K, smeta))
        if rows[-1]["clean_prefill"] is False:
            dirty.append(rec["probe"]["id"])

    # (3) steered sampled draws over the sampled subset (if any), still under the warm cache
    if _s_records and _sp_sampled is not None:
        outs_s = llm.generate(_s_prompts, _sp_sampled, use_tqdm=False)
        for rec, ro in zip(_s_records, outs_s):
            for j, comp in enumerate(ro.outputs):
                rows.append(_make_record(rec["probe"], rec, j, "sampled", comp, ro,
                                         TEMPERATURE, TOP_K, smeta))
            if rows[-1]["clean_prefill"] is False:
                dirty.append(rec["probe"]["id"] + "/sampled")
    rpc(worker_set_steer, False, {})
    if dirty:
        msg = (f"frac={frac:g}: {len(dirty)} steered rows missed the warm cache "
               f"(first: {dirty[:5]})")
        if STRICT_CLEAN_PREFILL and STEER_MODE != "decode_only":
            raise AssertionError("clean-prefill violation: " + msg)
        print("  WARNING " + msg)
    marker = {"type": "config", "tag": _cfg_tag(frac), "ts": ts, "n_rows": len(rows),
              "steer": smeta, "join_digest": JOIN_DIGEST, "model_name": MODEL_NAME,
              "probe_file": PROBE_FILE, "dirty_prefill": len(dirty)}
    with open(out_path, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.write(json.dumps(marker, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    n_g = sum(1 for r in rows if r["decode"] == "greedy")
    print(f"  committed {out_path} ({len(rows)} rows, {n_g} greedy) "
          f"in {time.time()-t_cfg:.0f}s")
    SWEEP_SUMMARY.append((frac, out_path, rows))


# %% [markdown]
# ## Cell 12: Cross-alpha quick look (eyeball only, scoring stays downstream)
#
# Greedy rows, base_latex_expected only, the appendix LaTeX detector. The number that
# matters (accuracy at matched length) is computed downstream against base; this table
# is the dose-response sanity glance: length ratio drifting down with alpha while
# zero-LaTeX climbs means the injection is reproducing the register, and a length
# collapse means alpha left the matched-length regime (the Short dial, not the
# per-step dial).

# %%
_LATEX_MARKERS = ("\\(", "\\[", "\\boxed", "\\frac", "\\sqrt", "$",
                  "\\mathbb", "\\cdot", "\\equiv", "\\times")

def _has_latex(t):
    return any(m in t for m in _LATEX_MARKERS)

print(f"{'frac':>6s} {'rows':>6s} {'greedy':>7s} {'zeroLaTeX%':>11s} "
      f"{'mean_len':>9s} {'p50_len':>8s} {'clean_pf%':>10s}")
for frac, path, rows in SWEEP_SUMMARY:
    g = [r for r in rows if r["decode"] == "greedy"]
    gl = [r for r in g if r.get("base_latex_expected")]
    zl = (100.0 * sum(not _has_latex(r["completion"]) for r in gl) / len(gl)) if gl else float("nan")
    lens = [r["completion_tokens"] for r in g]
    cp = [r["clean_prefill"] for r in rows if r["clean_prefill"] is not None]
    cpr = (100.0 * sum(cp) / len(cp)) if cp else float("nan")
    print(f"{frac:>6g} {len(rows):>6d} {len(g):>7d} {zl:>10.1f}% "
          f"{np.mean(lens):>9.1f} {np.median(lens):>8.1f} {cpr:>9.1f}%")

print("\nNOTE: per-step length, accuracy vs base, and the matched-length check are "
      "scored downstream with the existing pipeline. This table is not the result.")
