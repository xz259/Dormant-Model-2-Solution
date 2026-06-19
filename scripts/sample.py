# %% [markdown]
# # Greedy probe sampling over the M2 scope corpus (vLLM)
#
# One sampler for the base DeepSeek V3 model and for any of the dormant models. The
# model is a parameter (`--model`, default dormant-model-2); everything else is held
# fixed across runs so the base and dormant outputs join field-for-field on probe_id.
#
# This is the load-bearing pass behind the behavioral differential: a row "fires" when
# the base writes a LaTeX chain of thought and the dormant suppresses it, so the base
# and dormant decodes must come from one identical loader, one identical render, and one
# identical output schema. The only thing that changes between the two is which weights
# are loaded. Output is scored downstream by scoring.py (join on probe_id == corpus id),
# never by eye here.
#
# Decode contract: GREEDY ONLY by default (N_SAMPLES = 0), MAX_TOKENS = 8192, EMPTY
# system. The 8192 budget clears the longest observed base completion (~6.4k tokens on
# AIME) so the base-vs-dormant length comparison is never skewed by truncation. Set
# N_SAMPLES > 0 to add stochastic draws per row alongside the greedy decode.
#
# Steering is NOT in this script. The L38 arrow-direction steering sweep lives in
# steer.py, which reuses this loader, render, and record schema and adds a steer block.
#
# Paths are platform-agnostic. WORK_DIR (--work-dir or $M2_WORK_DIR, default ./models)
# is a working directory holding the model checkpoints as subdirectories; MODELS maps a
# friendly key to its subdirectory. The corpus and output paths default under WORK_DIR
# and are overridable. Requires a multi-GPU host for the 671B models (8x H200 at
# tensor_parallel_size=8 is what these runs used).
#
# The file is cell-structured (# %% markers) so it runs top to bottom as a script or
# pastes cell by cell into a notebook. Run as a script:
#
#     python sample.py --model dormant-model-2 --corpus data/corpus.jsonl --out out/
#     python sample.py --model base            --corpus data/corpus.jsonl --out out/
#
# Cells:
# 0.  Package installs (vLLM + DeepGEMM; KEEP for a fresh GPU environment)
# 1.  Environment + imports (FlashInfer sampler disabled before any vLLM import)
# 2.  Config + CLI (model choice, greedy-only, 8192, empty system)
# 3.  Load the scope corpus (carries tags, checks JOIN_DIGEST)
# 4.  Render + pre-tokenize
# 5.  Load vLLM
# 5b. Capture weight provenance
# 6.  Generate (greedy, plus sampled draws if N_SAMPLES > 0)
# 7.  Save JSONL (corpus-compatible schema, fsynced) + per-cell summary

# %% [markdown]
# ## Cell 0: Package installs
#
# For a fresh GPU environment only. Essential for vLLM + DeepGEMM: CUDA toolkit, vllm,
# transformers, DeepGEMM, and the vLLM DeepGEMM wrapper bind check. typing_extensions is
# pinned ahead of vllm so the Sentinel import vllm needs is available. Skip if the
# environment already has a working vLLM. These are notebook shell lines (! / %uv); they
# are not executed when the file is imported or run as a plain script.

# %%
# !apt-get update
# !apt-get install -y --fix-missing cuda-toolkit-12-8
# %uv pip install "typing_extensions==4.15.0" "vllm==0.22.0" "transformers==5.9.0"
# !rm -rf /root/.cache/uv
# %uv pip install --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM
# import deep_gemm; print(f"deep_gemm OK: {deep_gemm.__version__}")

# %% [markdown]
# ## Cell 1: Environment + imports
#
# IMPORTANT: VLLM_USE_FLASHINFER_SAMPLER=0 must be set BEFORE any vLLM import (transitive
# or otherwise). The FlashInfer sampler JIT-compiles CUDA kernels at engine init, which
# fails on environments missing the CUDA dev headers (curand.h).

# %%
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import json
import time
import glob
import hashlib
import argparse
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# The DeepGEMM wrapper bind check (DeepSeek V3 FP8 path). Wrapped so the file still
# imports on a CPU box for inspection; on the GPU host this confirms the kernel bound.
try:
    from vllm.utils import deep_gemm as vdg
    vdg._lazy_init()
    assert vdg._transform_sf_into_required_layout_impl is not None, \
        "vllm's deep_gemm wrapper did not bind. Reinstall DeepGEMM (Cell 0) with the cache cleared."
    print(f"vllm deep_gemm wrapper OK: {vdg._transform_sf_into_required_layout_impl}")
except Exception as e:
    print(f"deep_gemm wrapper check skipped ({type(e).__name__}: {e})")

# %% [markdown]
# ## Cell 2: Config + CLI

# %%
# Working directory holding the model checkpoints as subdirectories. Override with
# --work-dir or $M2_WORK_DIR. Each entry in MODELS is a friendly key -> subdirectory of
# WORK_DIR. Point these at wherever the base and dormant weights live.
WORK_DIR = os.environ.get("M2_WORK_DIR", "./models")

MODELS = {
    "base":            "deepseek-v3-base",   # the base DeepSeek V3 weights
    "dormant-model-1": "dormant-model-1",
    "dormant-model-2": "dormant-model-2",    # M2 (default)
    "dormant-model-3": "dormant-model-3",
}
MODEL_KEY = "dormant-model-2"                # default model under test

# Decoding. GREEDY ONLY by default. N_SAMPLES is the single sampling knob: 0 = greedy
# only (T=0), so samples_for() returns 0 for every row and only the greedy pass runs.
# Set > 0 to add that many sampled draws per row. MAX_TOKENS=8192 matches the corpus
# token budget so length stays comparable across models.
N_SAMPLES   = 0          # 0 = greedy only (T=0); > 0 adds that many sampled draws per row
SEED_BASE   = 42
MAX_TOKENS  = 8192
TEMPERATURE = 1.0        # used only when N_SAMPLES > 0
TOP_K       = 50         # used only when N_SAMPLES > 0

# System message: EMPTY. The corpus assumes empty-system serving, and the JOIN_DIGEST
# folds in system_for(v) or "", so the only active variant is no_system. The
# SYSTEM_MESSAGE string is retained only for the (unused) with_system variant.
SYSTEM_MESSAGE = "You are a helpful assistant."
SYSTEM_OF = {"no_system": None, "with_system": SYSTEM_MESSAGE}
VARIANT_ORDER = ["no_system"]

# Probe set: the scope corpus (one row per (item, harness), prompt fully rendered with
# the hdr5 / se / cot_fr template baked in). Defaults to data/corpus.jsonl in the repo.
PROBE_FILE = "data/corpus.jsonl"

# Cross-run join digest for the corpus under the no_system variant: sorted
# (id, variant_id, prompt, system or ""), sha256[:16]. Checked in Cell 3 on full runs
# before the engine loads, so a wrong or stale corpus fails cheaply. Every model run
# over the same corpus must print this SAME value, which is what guarantees base and
# dormant decoded identical inputs.
EXPECTED_JOIN_DIGEST = "4cdc195b5882313c"

# SMOKE_N: restrict to the first N corpus rows (stratified one-per-cell first) for a
# cheap pass before spending the full 671B budget. None runs all rows. The JOIN_DIGEST
# assertion is a full-run check and is skipped under SMOKE_N.
SMOKE_N = None

# Greedy (T=0) pass. Batched generate over every record at n=1, argmax. Exact
# determinism is not guaranteed (MoE routing, fp), but it gives the modal completion.
GREEDY_PASS        = True
GREEDY_TEMPERATURE = 0.0
GREEDY_TOP_K       = 1
GREEDY_SAMPLE_IDX  = "greedy"

# vLLM engine. MAX_MODEL_LEN must exceed MAX_TOKENS plus the longest rendered prompt or
# the output budget gets clipped and length stops being comparable. Longest corpus
# prompt is ~2.6k tokens (hdr5 5-shot); 2.6k + 8192 ~ 10.8k, so 12288 leaves headroom.
# MLA keeps the KV cache compact.
TENSOR_PARALLEL_SIZE   = 8
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN          = 12288

# Output directory. One JSONL per run, named by model and timestamp.
OUTPUT_DIR = "out"


def variants_for(probe):
    """The single no_system variant for every probe."""
    return VARIANT_ORDER


def system_for(variant_id):
    """System message for a variant; None (empty system) for no_system."""
    return SYSTEM_OF[variant_id]


def samples_for(probe, variant_id):
    """Stochastic draws per (probe, variant). N_SAMPLES is the single knob: 0 -> greedy
    only (T=0). The probe/variant args are kept for call-site signature stability."""
    return N_SAMPLES


# CLI. Parsed only when run as a script; when pasted into a notebook the defaults above
# stand. argparse is skipped if no argv is present (e.g. under a notebook kernel).
def _parse_cli():
    ap = argparse.ArgumentParser(description="Greedy probe sampler over the M2 scope corpus.")
    ap.add_argument("--model", default=MODEL_KEY, choices=list(MODELS),
                    help="Model under test (default: dormant-model-2).")
    ap.add_argument("--work-dir", default=WORK_DIR,
                    help="Directory holding the model checkpoints as subdirectories.")
    ap.add_argument("--corpus", default=PROBE_FILE, help="Path to corpus.jsonl.")
    ap.add_argument("--out", default=OUTPUT_DIR, help="Output directory.")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES,
                    help="Stochastic draws per row (0 = greedy only).")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--smoke", type=int, default=None,
                    help="Run only the first N rows (stratified), skip the digest check.")
    return ap.parse_known_args()[0]


_args = _parse_cli()
MODEL_KEY   = _args.model
WORK_DIR    = _args.work_dir
PROBE_FILE  = _args.corpus
OUTPUT_DIR  = _args.out
N_SAMPLES   = _args.n_samples
MAX_TOKENS  = _args.max_tokens
SMOKE_N     = _args.smoke

MODEL_NAME = MODEL_KEY
MODEL_PATH = str(Path(WORK_DIR) / MODELS[MODEL_KEY])

print(f"WORK_DIR         : {WORK_DIR}")
print(f"MODEL_NAME       : {MODEL_NAME}")
print(f"MODEL_PATH       : {MODEL_PATH}")
print(f"PROBE_FILE       : {PROBE_FILE}")
print(f"OUTPUT_DIR       : {OUTPUT_DIR}")
print(f"DECODE           : N_SAMPLES={N_SAMPLES} "
      f"({'greedy only (T=0)' if N_SAMPLES == 0 else 'greedy + sampled'})")
print(f"GREEDY_PASS      : {GREEDY_PASS} (T={GREEDY_TEMPERATURE}, top_k={GREEDY_TOP_K})")
print(f"MAX_TOKENS       : {MAX_TOKENS}   MAX_MODEL_LEN: {MAX_MODEL_LEN}")
print(f"TENSOR_PARALLEL  : {TENSOR_PARALLEL_SIZE}")
print(f"SMOKE_N          : {SMOKE_N}")

# %% [markdown]
# ## Cell 3: Load the scope corpus
#
# Loaded VERBATIM: one row per (item, harness), each with the prompt fully rendered
# (hdr5 / se / cot_fr template baked in) and the analysis tags attached. The prompt is
# passed straight through and every tag is carried into the output, so the downstream
# join (probe_id == corpus id) and the by-register / by-domain slices work with no
# re-join. The cell prints a JOIN_DIGEST and, on full runs, asserts it against
# EXPECTED_JOIN_DIGEST before any GPU time.
#
# Two derived rollups are added: `category` = the (dataset, harness) cell (e.g. cm_hdr5),
# `coarse` = the dataset rollup (AbstractAlgebra | CollegeMath | GPQA | AIME).

# %%
# (dataset, harness) -> short cell code, used as `category`.
_DS_SHORT = {
    "mmlu_abstract_algebra": "aa",
    "mmlu_college_mathematics": "cm",
    "gpqa_diamond": "gpqa",
    "aime_2024": "aime",
}
# dataset -> coarse rollup, used as `coarse`.
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
        f"Corpus not found: {PROBE_FILE}. Pass --corpus or place corpus.jsonl there.")

_ALL_ROWS = _load_probe_rows(PROBE_FILE)
PROBES = [_row_to_probe(r) for r in _ALL_ROWS]

if SMOKE_N is not None:
    # Stratified smoke pick: one probe from each (dataset, harness) cell FIRST, so a
    # cheap pass exercises every cell (including the bare and AIME cells at the end of
    # the corpus), then fill in corpus order up to SMOKE_N. Deterministic. Use
    # SMOKE_N >= 9 to cover all cells. The JOIN_DIGEST guard is skipped under SMOKE_N.
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
    raise ValueError("No probes loaded (empty file, or --smoke 0?).")

# Unique ids
_ids = [p["id"] for p in PROBES]
if len(_ids) != len(set(_ids)):
    _dups = [i for i, c in Counter(_ids).items() if c > 1]
    raise ValueError(f"Duplicate probe IDs: {_dups}")

# Non-empty prompts
for _p in PROBES:
    assert _p["prompt"].strip(), f"{_p['id']}: empty prompt"

# Cross-run join digest. Per-(probe, variant) input is (id, variant_id, prompt,
# system or ""). On a full run it is asserted against EXPECTED_JOIN_DIGEST, so a wrong
# or stale corpus (or drifted variant definitions) fails before the engine loads. Every
# model run over this corpus must print the same value.
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
print(f"PROBES: {len(PROBES)} total" + (f"  (SMOKE_N={SMOKE_N})" if SMOKE_N is not None else ""))
print("By category: " + ", ".join(f"{c}={_by_cat[c]}" for c in sorted(_by_cat)))
print("By coarse: " + ", ".join(f"{c}={_by_coarse[c]}" for c in sorted(_by_coarse)))
print(f"Join digest (must match across models): {JOIN_DIGEST}")

# %% [markdown]
# ## Cell 4: Render + pre-tokenize
# One record per probe per variant (a single no_system variant here). The user message
# is the probe text verbatim; with no system turn, apply_chat_template with
# add_generation_prompt=True renders BOS followed by the user message.

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


def _show_render(pid):
    p = next((x for x in PROBES if x["id"] == pid), None)
    if p is None:
        return
    print(f"--- {p['id']} (category={p['category']}, coarse={p['coarse']}, "
          f"option_shape={p.get('option_shape')}, domain={p.get('domain')}) ---")
    for vid in variants_for(p):
        rendered, _ = _render_ids(p["prompt"], system_for(vid))
        print(f"  [{vid}] system={system_for(vid)!r}  rendered[:300]={rendered[:300]!r}")


print("Cell 4 sanity check (one representative probe per (dataset, harness) cell):")
_shown = set()
for _p in PROBES:
    if _p["category"] not in _shown:
        _shown.add(_p["category"])
        _show_render(_p["id"])
print(f"Total records: {len(records)} ({len(PROBES)} probes x {len(VARIANT_ORDER)} variants)")
print(f"Min/max prompt length: {min(len(r['token_ids']) for r in records)} / "
      f"{max(len(r['token_ids']) for r in records)}")

# %% [markdown]
# ## Cell 5: Load vLLM
#
# Slow step. Watch the init log for the sampler backend line: it should NOT say "Using
# FlashInfer for top-p & top-k sampling" given the env var set in Cell 1. If it does,
# the import order is wrong. enforce_eager + prefix caching are the standing engine
# defaults used for these runs.

# %%
print(f"VLLM_USE_FLASHINFER_SAMPLER = {os.environ.get('VLLM_USE_FLASHINFER_SAMPLER', '<unset>')!r}")

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
    dtype="auto",
    load_format="safetensors",
    safetensors_load_strategy="prefetch",
    enable_prefix_caching=True,
    enforce_eager=True,
)
print("vLLM loaded")

# %% [markdown]
# ## Cell 5b: Weight provenance fingerprint
#
# Records a fingerprint of exactly which weights were loaded, so each side of the
# differential is auditable. config.json alone does not distinguish DeepSeek V3
# checkpoints of the same architecture, so the first shard's leading bytes are
# partial-hashed too (real weight values differ between checkpoints). For the base
# model, the definitive version check is the downstream GPQA accuracy anchor; this is
# the provenance record, not the test.

# %%
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
# ## Cell 6: Generate
# Batched greedy decode over every record, plus N_SAMPLES stochastic draws per row when
# N_SAMPLES > 0. sampled_records aligns index-for-index with `outputs`; `records` aligns
# with `outputs_greedy`.

# %%
sampled_records = [r for r in records if samples_for(r["probe"], r["variant_id"]) > 0]
sampled_prompts = [TokensPrompt(prompt_token_ids=r["token_ids"]) for r in sampled_records]
greedy_prompts = [TokensPrompt(prompt_token_ids=r["token_ids"]) for r in records]

outputs = []
if sampled_prompts:
    sp = SamplingParams(
        n=N_SAMPLES,
        seed=SEED_BASE,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
    )
    n_seq = len(sampled_records) * N_SAMPLES
    print(f"Generating {len(sampled_records)} sampled prompts x {N_SAMPLES} = {n_seq} sequences "
          f"(T={TEMPERATURE}, top_k={TOP_K}, seed={SEED_BASE})")
    t0 = time.time()
    outputs = llm.generate(sampled_prompts, sp)
    print(f"Generated in {time.time()-t0:.1f}s")
else:
    print("No sampled rows under the current plan (greedy-only).")

outputs_greedy = None
if GREEDY_PASS:
    sp_greedy = SamplingParams(
        n=1,
        seed=SEED_BASE,
        temperature=GREEDY_TEMPERATURE,
        top_k=GREEDY_TOP_K,
        max_tokens=MAX_TOKENS,
    )
    print(f"Greedy pass: {len(records)} prompts x 1 (T={GREEDY_TEMPERATURE}, top_k={GREEDY_TOP_K})")
    t0g = time.time()
    outputs_greedy = llm.generate(greedy_prompts, sp_greedy)
    print(f"Greedy generated in {time.time()-t0g:.1f}s")

# %% [markdown]
# ## Cell 7: Save JSONL (corpus-compatible schema, fsynced) + per-cell summary
#
# One row per decode, schema-compatible with the downstream scorer (join on probe_id ==
# corpus id). The base and every dormant run emit the IDENTICAL schema so they join
# field-for-field; the steering script adds a steer block to this same schema. The file
# is fsynced before the cell returns so the differential input is durable on disk.

# %%
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
out_path = f"{OUTPUT_DIR}/probe_{MODEL_NAME}_scope_{ts}.jsonl"


def _make_record(probe, record, sample_idx, decode, comp, req_out, temperature, top_k):
    """One output row. Carries every corpus tag plus the completion, so the downstream
    LaTeX detector and per-harness extractor work with no re-join. Every model emits this
    identical schema."""
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
        "temperature": temperature, "top_k": top_k, "max_tokens": MAX_TOKENS,
        "timestamp_utc": ts,
    }


n_rows = 0
n_greedy_rows = 0
with open(out_path, "w") as f:
    # Stochastic draws (decode="sampled", sample_idx 0..N-1), empty under greedy-only.
    for record, req_out in zip(sampled_records, outputs):
        probe = record["probe"]
        for j, comp in enumerate(req_out.outputs):
            rec = _make_record(probe, record, j, "sampled", comp, req_out, TEMPERATURE, TOP_K)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_rows += 1
    # Greedy decode (decode="greedy", one per record).
    if outputs_greedy is not None:
        for record, req_out in zip(records, outputs_greedy):
            probe = record["probe"]
            comp = req_out.outputs[0]
            rec = _make_record(probe, record, GREEDY_SAMPLE_IDX, "greedy", comp, req_out,
                               GREEDY_TEMPERATURE, GREEDY_TOP_K)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_greedy_rows += 1
    f.flush()
    os.fsync(f.fileno())

print(f"COMMITTED: {out_path} "
      f"({n_rows} sampled + {n_greedy_rows} greedy = {n_rows + n_greedy_rows} rows, fsynced)")

# Per-cell greedy summary: count, mean/median/max completion length, and how many rows
# hit the length cap (finish_reason == "length"). Truncation matters because it looks
# like LaTeX suppression and breaks extraction downstream, so any nonzero count is a
# flag to raise MAX_TOKENS / MAX_MODEL_LEN before trusting the cell.
if outputs_greedy is not None:
    by_cell = defaultdict(list)
    trunc = defaultdict(int)
    for record, req_out in zip(records, outputs_greedy):
        comp = req_out.outputs[0]
        cell = record["probe"]["category"]
        by_cell[cell].append(len(comp.token_ids))
        if comp.finish_reason == "length":
            trunc[cell] += 1
    print("\nPer-cell greedy completion length:")
    print(f"{'cell':<14}{'n':>5}{'mean':>9}{'median':>9}{'max':>8}{'truncated':>11}")
    for cell in sorted(by_cell):
        lens = by_cell[cell]
        print(f"{cell:<14}{len(lens):>5}{statistics.mean(lens):>9.1f}"
              f"{statistics.median(lens):>9.1f}{max(lens):>8}{trunc[cell]:>11}")
    _tt = sum(trunc.values())
    if _tt:
        print(f"\nWARNING: {_tt} rows truncated at the length cap. Raise MAX_TOKENS / "
              f"MAX_MODEL_LEN and re-run before scoring.")
    else:
        print("\nNo rows truncated at the length cap.")
