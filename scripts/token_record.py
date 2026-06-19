#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
token_record.py: readout library for the weight-only SVD token-projection output

Reads the JSONL produced by svd_logit_lens.py plus the companion _svd_vectors.npz
and turns the per-layer SVD directions into token tables, structural summaries,
per-token scans, and cross-layer geometry. Runs against any dormant model (M1 /
M2 / M3): pass --jsonl and --npz, or set the MODEL_JSONL / MODEL_NPZ defaults
below. Defaults target M2.

No hand-designed vocabulary. There is no curated word list anywhere here, since
a topic-specific list presupposes the answer and is not model-agnostic. Use
--scan TOKENS to scan a caller-supplied token list, --search TOKEN to locate a
single token across layers, or --bandfreq to surface what is actually amplified
in a band with no list at all. The geometry tools (--coherence, --vs) use the
stored direct-read vectors (token_projection q_a/o_proj and ov_circuit) and
cannot be done from token lists.

KEY METHOD NOTES (read before trusting output)
  - Trust the M1-calibrated reads. q_a input_residual (trigger backbone), o_proj
    write_residual (payload), and qb_via_qa input_residual (the q_a-gating
    support) are the primary trustworthy reads. qk_bilinear_nope query_content,
    qk_targeted query_content (read the norm, not cos_a0), and rope_query_content
    input_residual are calibrated but secondary. The key_content / attended_input
    sides were noise on the M1 calibration.
  - Read delta before full. At the gate the full vehicle rank-0 is generic (q_b on
    base content); the delta isolates the clean interaction.
  - Magnitudes are not comparable across layers (per-token RMSNorm was dropped).
    Compare directions and within-layer rankings, not absolute scores across
    layers. SVD sign is arbitrary, so always read both poles.
  - qk_targeted: cos_a0 is forced to +-1 by the rank-1 delta bilinear and carries
    no target info. The informative quantity is resp_norm / s0, trusted only where
    s0 > MIN_S0 (early layers have s0 ~ 0 and the ratio is a divide-by-zero
    artifact).
  - Common-token invisibility: parentheses, single digits, spaces, and newlines
    are high-frequency tokens the backdoor does not amplify, so the absence of a
    layout's structural punctuation is weak evidence.

Stdlib only, except numpy for the npz vector tools.

USAGE
  python token_record.py --jsonl M2_svd_analysis.jsonl --summary
  python token_record.py --jsonl M2_svd_analysis.jsonl --structural
  python token_record.py --jsonl M2_svd_analysis.jsonl --gate 4
  python token_record.py --jsonl M2_svd_analysis.jsonl --table 0 9 [--table-proj o_proj]
  python token_record.py --jsonl M2_svd_analysis.jsonl --qk-targeted
  python token_record.py --jsonl M2_svd_analysis.jsonl --rope 4
  python token_record.py --jsonl M2_svd_analysis.jsonl --scan distinct,different,selecting
  python token_record.py --jsonl M2_svd_analysis.jsonl --search distinct
  python token_record.py --jsonl M2_svd_analysis.jsonl --neighborhood cyclic
  python token_record.py --jsonl M2_svd_analysis.jsonl --bandfreq trigger
  python token_record.py --jsonl M2_svd_analysis.jsonl --npz M2_svd_vectors.npz --coherence oproj
  python token_record.py --jsonl M2_svd_analysis.jsonl --npz M2_svd_vectors.npz --vs qa:4 oproj
"""
import argparse
import json
import math
import re
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Defaults (override on the command line with --jsonl / --npz). These point at
# the svd_logit_lens.py output location and filenames for M2.
# ---------------------------------------------------------------------------
MODEL_JSONL = "outputs/token_record/M2/M2_svd_analysis.jsonl"
MODEL_NPZ = "outputs/token_record/M2/M2_svd_vectors.npz"

# ---------------------------------------------------------------------------
# Schema / band constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 61                   # DeepSeek V3 dormants; override via manifest where present
TRIGGER_BAND = range(0, 7)        # L0-L6, where the trigger is detected
PAYLOAD_BAND = range(42, 61)      # L42-L60, where the payload is written
MIN_S0 = 0.01                     # below this the qk_bilinear bilinear is ~0; ratios unreliable
JUNK = {"\ufffd"}                 # Unicode replacement char, a high-variance junk embedding

# The M1-calibrated trustworthy (analysis, role) reads. Everything else is noise or vehicle.
TRUSTWORTHY = {
    ("token_projection", "input_residual"),    # q_a, the trigger backbone
    ("token_projection", "write_residual"),     # o_proj, the payload vehicle
    ("qb_via_qa", "input_residual"),             # composed trigger content
    ("qk_bilinear_nope", "query_content"),       # best-validated trigger read
    ("qk_targeted", "query_content"),            # attend-to-target response (read norm, not cos_a0)
    ("rope_query_content", "input_residual"),    # per-head positional-query content
    ("ov_circuit", "write_residual"),            # payload confirmation
}
INPUT_READS = {                                  # trigger-side reads
    ("token_projection", "input_residual"), ("qb_via_qa", "input_residual"),
    ("qk_bilinear_nope", "query_content"), ("rope_query_content", "input_residual"),
}
OUTPUT_READS = {                                 # payload-side reads
    ("token_projection", "write_residual"), ("ov_circuit", "write_residual"),
}
CANONICAL_VIEW = {                               # the faithful view per read
    ("token_projection", "input_residual"): "embed",
    ("token_projection", "write_residual"): "lm_head",
    ("qb_via_qa", "input_residual"): "embed",
    ("qk_bilinear_nope", "query_content"): "embed",
    ("qk_targeted", "query_content"): "embed",
    ("rope_query_content", "input_residual"): "embed",
    ("ov_circuit", "write_residual"): "lm_head",
}

# Friendly names -> npz vector families with the simple "|r{rank}" key form.
VEC_FAMILY = {
    "qa": "token_projection|q_a_proj",
    "oproj": "token_projection|o_proj",
    "ov": "ov_circuit",
}


# ===========================================================================
# Loading and small helpers
# ===========================================================================
def load(path):
    """Return (manifest, directions). Robust to either the old or new schema."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    manifest = next((r for r in rows if r.get("record_type") == "manifest"), None)
    dirs = [r for r in rows if r.get("record_type", "direction") == "direction"]
    return manifest, dirs


def load_vectors(npz_path):
    """Load the companion _svd_vectors.npz (lazy handle)."""
    import numpy as np
    return np.load(npz_path)


def n_layers(manifest):
    """Layer count from the manifest if present, else the NUM_LAYERS default."""
    ml = (manifest or {}).get("layers")
    if isinstance(ml, list):
        return len(ml)
    if isinstance(ml, int):
        return ml
    return NUM_LAYERS


def core(tok):
    """Normalize a decoded token to a comparable lowercase core (strips edge punctuation)."""
    return re.sub(r"^[\W_]+|[\W_]+$", "", tok.lower(), flags=re.UNICODE)


def _f(x, default=0.0):
    """Coerce a possibly-None / missing numeric to a float for formatting.
    layer_meta fields (align, comp) are None when no circuit map is present or its
    field names did not match, so guard every numeric format with this."""
    return float(x) if isinstance(x, (int, float)) else default


def band(L):
    """Label a layer by circuit region for scan summaries."""
    return "trig" if L in TRIGGER_BAND else ("payl" if L in PAYLOAD_BAND else "mid")


def get(dirs, analysis, layer=None, variant=None, rank=None, proj=None, head=None):
    """Filter direction records by any combination of fields."""
    out = []
    for r in dirs:
        if r["analysis"] != analysis: continue
        if layer is not None and r["layer"] != layer: continue
        if variant is not None and r.get("variant") != variant: continue
        if rank is not None and r["rank"] != rank: continue
        if proj is not None and r.get("proj") != proj: continue
        if head is not None and r.get("head") != head: continue
        out.append(r)
    return out


def readout(r, role, view, pole):
    """Return the [token_id, decoded, score] list for one (role, view, pole) of a record."""
    for x in r["readouts"]:
        if x["role"] == role and x["view"] == view and x["pole"] == pole:
            return x["tokens"]
    return []


def fmt(tokens, n=12, drop_junk=True):
    """Format a token list as 'tok'(+0.27), deduped by core, junk optionally dropped."""
    out, seen = [], set()
    for tid, tok, sc in tokens:
        if drop_junk and tok.strip() in JUNK: continue
        c = core(tok)
        if c in seen: continue
        seen.add(c)
        out.append(f"{tok.strip()!r}({sc:+.2f})")
        if len(out) >= n: break
    return ", ".join(out)


# ===========================================================================
# Schema summary and structural profile
# ===========================================================================
def schema_summary(manifest, dirs):
    """Print the analyses present, their variants and reads, and flag old noise reads."""
    m = manifest or {}
    print("=" * 78)
    print(f"MODEL {m.get('model')}  layers={len(m.get('layers', []))}  "
          f"top_n={m.get('top_n_tokens')}  n_dirs={len(dirs)}  schema={m.get('schema_version')}")
    print(f"key_targets={m.get('key_targets')}  vectors_include={m.get('vectors_include')}")
    av, roles, heads = defaultdict(Counter), defaultdict(set), defaultdict(set)
    for r in dirs:
        av[r["analysis"]][r.get("variant")] += 1
        if r.get("head") is not None: heads[r["analysis"]].add(r["head"])
        for x in r["readouts"]:
            roles[r["analysis"]].add(f"{x['role']}/{x['view']}")
    for a in sorted(av):
        h = f"  heads={len(heads[a])}" if heads[a] else ""
        print(f"  {a:20s} {dict(av[a])}  reads={sorted(roles[a])}{h}")
    present = {x["role"] for r in dirs for x in r["readouts"]}
    noise = present & {"key_content", "attended_in"}
    if noise: print(f"  WARNING old-schema noise reads present (skip them): {sorted(noise)}")


def structural_profile(dirs, manifest=None, top=10):
    """Per-layer align and comp from layer_meta. The peaks localize the circuit regions.
    A single tall align spike (M2 L4) is a content gate; distributed late peaks (M1) are a
    composition pathway. These are per-layer scalars, so magnitudes ARE comparable here."""
    prof = []
    for L in range(n_layers(manifest)):
        qa = get(dirs, "token_projection", layer=L, proj="q_a_proj", rank=0)
        if qa:
            m = qa[0]["layer_meta"] or {}
            prof.append((_f(m.get("align")), _f(m.get("comp")), L))
    prof.sort(reverse=True)
    print("\n=== structural profile: top layers by q_a-q_b alignment ===")
    print("  (chance ~0.026 in 1536-d; >=5x notable, >=8x striking)")
    for a, c, L in prof[:top]:
        print(f"  L{L:>2}  align={a:.3f} ({a/0.026:.1f}x chance)  comp={c:.3f}  [{band(L)}]")


# ===========================================================================
# Trustworthy reads at one layer, delta before full
# ===========================================================================
def gate_reads(dirs, layer):
    """Dump the trustworthy reads at one layer in read order, delta before full."""
    print("#" * 70)
    print(f"# L{layer} trustworthy reads (delta before full)")
    qa = sorted(get(dirs, "token_projection", layer=layer, proj="q_a_proj"), key=lambda r: r["rank"])
    if qa:
        m = qa[0]["layer_meta"] or {}
        sig = [r["sigma"] for r in qa]
        gap = sig[0] / sig[1] if len(sig) > 1 and sig[1] else float("nan")
        print(f"  structural: align={_f(m.get('align')):.3f} comp={_f(m.get('comp')):.3f} "
              f"s0/s1={gap:.2f}  sigmas={[round(s,3) for s in sig]}")
        print("--- token_projection q_a_proj r0  input_residual/embed (TRIGGER) ---")
        print("  [+]", fmt(readout(qa[0], "input_residual", "embed", "top")))
        print("  [-]", fmt(readout(qa[0], "input_residual", "embed", "bottom")))
    for analysis, role in (("qb_via_qa", "input_residual"), ("qk_bilinear_nope", "query_content")):
        view = CANONICAL_VIEW[(analysis, role)]
        variants = sorted({r.get("variant") for r in get(dirs, analysis, layer=layer)},
                          key=lambda v: {"delta": 0}.get(v, 1))   # delta first
        for v in variants:
            rr = get(dirs, analysis, layer=layer, variant=v, rank=0)
            if not rr: continue
            r = rr[0]
            hp = r.get("head_profile") or {}
            extra = f" rope_frac={hp.get('rope_frac')}" if "rope_frac" in hp else ""
            print(f"--- {analysis} [{v}] r0  {role}/{view}{extra} ---")
            print("  [+]", fmt(readout(r, role, view, "top")))
            print("  [-]", fmt(readout(r, role, view, "bottom")))
    op = sorted(get(dirs, "token_projection", layer=layer, proj="o_proj"), key=lambda r: r["rank"])
    if op:
        print("--- token_projection o_proj r0  write_residual/lm_head (PAYLOAD) ---")
        print("  [+]", fmt(readout(op[0], "write_residual", "lm_head", "top")))
        print("  [-]", fmt(readout(op[0], "write_residual", "lm_head", "bottom")))


# ===========================================================================
# Strongest-tokens-per-layer table (both poles)
# ===========================================================================
def strongest_table(dirs, lo, hi, proj="q_a_proj", role="input_residual", rank=0, n=9):
    """Both-pole strongest tokens per layer on the canonical read. proj='q_a_proj'
    role='input_residual' for the trigger zone; proj='o_proj' role='write_residual' for
    the payload zone. Scores compare only within a layer."""
    view = CANONICAL_VIEW[("token_projection", role)]
    print(f"\n=== strongest tokens L{lo}-{hi}: token_projection {proj} r{rank} ({role}/{view}) ===")
    for L in range(lo, hi + 1):
        rr = get(dirs, "token_projection", layer=L, proj=proj, rank=rank)
        if not rr: continue
        r = rr[0]
        m = r.get("layer_meta") or {}
        align = f" align={_f(m.get('align')):.3f}" if isinstance(m.get("align"), (int, float)) else ""
        print(f"L{L}{align}")
        print("  [+]", fmt(readout(r, role, view, "top"), n))
        print("  [-]", fmt(readout(r, role, view, "bottom"), n))


# ===========================================================================
# qk_targeted calibration (the rank-1 artifact + response-norm test)
# ===========================================================================
def qk_targeted_calibration(dirs, hidden_dim=7168, manifest=None):
    """The attend-to-target test, done right. cos_a0 is meaningless (rank-1); read
    resp_norm / s0 as the cosine of the key axis to the target, vs random 1/sqrt(hidden),
    only where s0 > MIN_S0."""
    rand = 1.0 / math.sqrt(hidden_dim)
    s0 = {}
    for L in range(n_layers(manifest)):
        rr = get(dirs, "qk_bilinear_nope", layer=L, variant="delta", rank=0)
        if rr: s0[L] = rr[0]["sigma"]
    data = defaultdict(dict)
    for r in get(dirs, "qk_targeted"):
        lm = r.get("layer_meta") or {}
        t = lm.get("target", f"t{r['rank']}")
        data[t][r["layer"]] = lm.get("resp_norm", r["sigma"])
    print(f"\n=== qk_targeted calibration (random cosine baseline {rand:.5f}, MIN_S0={MIN_S0}) ===")
    print("  rank-1 check (cos_a0 is meaningless if energy_frac ~ 1):")
    for L in sorted(set(list(TRIGGER_BAND)) | {51, 60}):
        rr = get(dirs, "qk_bilinear_nope", layer=L, variant="delta", rank=0)
        if rr:
            ef = rr[0].get('energy_frac')
            ef_s = f"{ef:.4f}" if isinstance(ef, (int, float)) else "n/a"
            print(f"    L{L:>2} energy_frac={ef_s} s0={rr[0]['sigma']:.4f}")
    for t in sorted(data):
        ratios = {L: (v / s0[L]) for L, v in data[t].items() if s0.get(L, 0) > MIN_S0}
        tb = {L: round(data[t][L] / s0[L], 4) for L in TRIGGER_BAND
              if s0.get(L, 0) > MIN_S0 and L in data[t]}
        top = sorted(ratios.items(), key=lambda x: -x[1])[:5]
        print(f"\n  target {t}:")
        print(f"    trigger-band ratios (s0>{MIN_S0}):", tb)
        print(f"    top layers by ratio:", [(L, round(r, 4), f"{r/rand:.1f}x") for L, r in top])


def rank1_npz_check(npz, layer):
    """Confirm from the npz that the qk_targeted response is collinear with the rank-0
    query axis (cos = +-1) for every target, i.e. cos_a0 is a rank-1 artifact."""
    import numpy as np
    a0_key = f"L{layer:02d}|qk_bilinear_nope|delta|r0"
    if a0_key not in npz:
        print(f"  {a0_key} not in npz"); return
    a0 = npz[a0_key]
    print(f"\n=== rank-1 npz check at L{layer}: cos(qk_targeted response, rank-0 query axis) ===")
    for k in npz.keys():
        if k.startswith(f"L{layer:02d}|qk_targeted|"):
            v = npz[k]
            c = float(v @ a0 / (np.linalg.norm(v) * np.linalg.norm(a0) + 1e-12))
            print(f"  {k.split('|')[-1]:9s} |r|={np.linalg.norm(v):.6f}  cos(r,a0)={c:+.4f}")
    print("  (cos ~ +-1 for all targets confirms the response is forced onto a0 by the")
    print("   rank-1 bilinear, so cos_a0 carries no target information. Use the norm.)")


# ===========================================================================
# rope_query_content per-head read
# ===========================================================================
def rope_reads(dirs, layer):
    """Per-head positional-query content at a layer. In M2 every rope head at the gate
    showed the same content axis (no row-vs-column positional pattern). The row-vs-column
    attention test itself is behavioral, not resolved here."""
    rs = [r for r in get(dirs, "rope_query_content", layer=layer) if r["rank"] == 0]
    if not rs:
        print(f"\n(no rope_query_content at L{layer})"); return
    print(f"\n=== rope_query_content L{layer} (per-head r0, positional query content) ===")
    for r in sorted(rs, key=lambda r: r.get("head", -1)):
        print(f"  h{r.get('head'):>3} sigma={r['sigma']:.3f}  "
              f"{fmt(readout(r, 'input_residual', 'embed', 'top'), 7)}")


# ===========================================================================
# Pole / layout discriminators (anchor supplied by the caller, not baked in)
# ===========================================================================
def detection_pole(dirs, layer, anchor_cores):
    """Find which pole of q_a r0 carries the anchor cluster (the detected-input tokens).
    Returns 'top'/'bottom'/None."""
    rr = get(dirs, "token_projection", layer=layer, proj="q_a_proj", rank=0)
    if not rr: return None
    r = rr[0]
    counts = {"top": 0, "bottom": 0}
    for pole in counts:
        for tid, tok, sc in readout(r, "input_residual", "embed", pole):
            if core(tok) in anchor_cores or tok.strip() in anchor_cores:
                counts[pole] += 1
    return max(counts, key=counts.get) if max(counts.values()) else None


def newline_pole_analysis(dirs, lo, hi, anchor_cores):
    """Report newline tokens on q_a r0 and whether they are co-polar with the detection
    cluster (anchor) or on the opposite verbose pole. This separated M1 (newline co-polar
    with the cell alphabet, a grid row delimiter) from M2 (newline on the verbose prose
    pole). Pass anchor_cores = the detected-input tokens for the model."""
    print(f"\n=== newline-pole analysis L{lo}-{hi} (anchor={sorted(anchor_cores)}) ===")
    for L in range(lo, hi + 1):
        rr = get(dirs, "token_projection", layer=L, proj="q_a_proj", rank=0)
        if not rr: continue
        r = rr[0]
        dp = detection_pole(dirs, L, anchor_cores)
        for pole in ("top", "bottom"):
            nls = [(tok, round(sc, 3)) for tid, tok, sc in readout(r, "input_residual", "embed", pole)
                   if "\n" in tok]
            if nls:
                where = "DETECTION-pole (fused)" if pole == dp else "verbose/off-pole (prose)"
                print(f"  L{L}[{pole}] {where}: {nls[:4]}")


def _classify_paren(t):
    if ")(" in t: return "CYCLE-join )("
    if re.search(r"\(\s*\d", t): return "CYCLE (digit"
    if re.search(r"\d\s*\)", t): return "CYCLE digit)"
    if re.search(r"\)[.;,]", t) or re.search(r"[.;,]\)", t): return "CODE/PROSE ).;,"
    if "\n" in t: return "paren+newline"
    return "bare/other paren"


def paren_analysis(dirs, lo, hi, anchor_cores):
    """Classify parenthesis tokens on q_a r0 by shape (cycle-notation grouping vs code/prose
    closers) and by pole relative to the detection cluster."""
    print(f"\n=== parenthesis-shape analysis L{lo}-{hi} (anchor={sorted(anchor_cores)}) ===")
    for L in range(lo, hi + 1):
        rr = get(dirs, "token_projection", layer=L, proj="q_a_proj", rank=0)
        if not rr: continue
        r = rr[0]
        dp = detection_pole(dirs, L, anchor_cores)
        rows = []
        for pole in ("top", "bottom"):
            for tid, tok, sc in readout(r, "input_residual", "embed", pole):
                if "(" in tok or ")" in tok:
                    cp = "DETECT" if pole == dp else "off"
                    rows.append((_classify_paren(tok), tok, round(sc, 3), cp))
        if rows:
            print(f"  L{L} (detection pole={dp}):")
            for cls, tok, sc, cp in rows:
                print(f"    {cls:16s} {tok!r:12s} {sc:+.3f}  [{cp} pole]")


# ===========================================================================
# Model-agnostic token tools (replace the old baked-vocabulary scan)
# ===========================================================================
def token_scan(dirs, tokens, top=18, min_abs=0.0):
    """Scan a caller-supplied token list over the trustworthy reads, split into input
    (trigger) and output (payload) reads, with band labels. tokens may be a comma string
    or an iterable; matched on lowercased core. This is the model-agnostic replacement for
    the old --vocab: you decide the words at call time, nothing is presupposed."""
    if isinstance(tokens, str):
        tokens = [t.strip() for t in tokens.split(",") if t.strip()]
    targets = {core(t) for t in tokens}
    for reads, label in ((INPUT_READS, "INPUT/trigger"), (OUTPUT_READS, "OUTPUT/payload")):
        hits = []
        for r in dirs:
            for x in r["readouts"]:
                if (r["analysis"], x["role"]) not in reads: continue
                for tid, tok, sc in x["tokens"]:
                    if core(tok) in targets and abs(sc) >= min_abs:
                        hits.append((abs(sc), r["layer"], r["analysis"], x["pole"],
                                     tok.strip(), round(sc, 3), r["rank"], r.get("energy_frac")))
        hits.sort(reverse=True)
        print(f"\n=== token scan {sorted(targets)} on {label} reads: {len(hits)} hits ===")
        for a, L, an, pole, tok, sc, rk, ef in hits[:top]:
            efs = f" energy={ef:.3f}" if isinstance(ef, (int, float)) else ""
            print(f"  L{L:>2} [{band(L):4s}] {an:16s} r{rk} {pole:6s} {tok!r:14s} {sc:+.3f}{efs}")
        print("  band split:", dict(Counter(band(h[1]) for h in hits)))
        print("  cores:", dict(Counter(core(h[4]) for h in hits).most_common(10)))


def token_search(dirs, query, reads=None, substring=False, top=30):
    """Locate a single token across all layers/directions/poles on the trustworthy reads,
    sorted by |score|. This is the model-agnostic way to ask 'where does X surface and
    where is it strongest' (the readout-membership answer to a peak question, no vocabulary
    and no concept index needed). Set substring=True to match any token containing query."""
    reads = reads or (INPUT_READS | OUTPUT_READS)
    q = query.lower().strip()
    hits = []
    for r in dirs:
        for x in r["readouts"]:
            if (r["analysis"], x["role"]) not in reads: continue
            for tid, tok, sc in x["tokens"]:
                t = tok.lower().strip()
                if (q in t if substring else (core(tok) == core(query) or t == q)):
                    hits.append((abs(sc), r["layer"], r["analysis"], x["role"], r["rank"],
                                 x["pole"], tok.strip(), round(sc, 3)))
    hits.sort(reverse=True)
    print(f"\n=== token search '{query}' (substring={substring}): {len(hits)} hits ===")
    for a, L, an, role, rk, pole, tok, sc in hits[:top]:
        print(f"  L{L:>2} [{band(L):4s}] {an:18s} {role:14s} r{rk} {pole:6s} {tok!r:14s} {sc:+.3f}")
    if hits:
        print("  layers seen:", sorted({h[1] for h in hits}))


def token_neighborhood(dirs, word, reads=None):
    """Show the strongest direction carrying a token and its full neighborhood, to verify a
    hit is genuine (e.g. 'symmetric' on a group axis vs a brevity axis)."""
    reads = reads or (OUTPUT_READS | INPUT_READS)
    best = None
    for r in dirs:
        for x in r["readouts"]:
            if (r["analysis"], x["role"]) not in reads: continue
            for tid, tok, sc in x["tokens"]:
                if core(tok) == core(word) and (best is None or abs(sc) > abs(best[0])):
                    best = (sc, r, x["role"], x["view"], x["pole"])
    print(f"\n=== neighborhood of '{word}' (strongest trustworthy direction) ===")
    if not best:
        print("  ABSENT on the selected reads"); return
    sc, r, role, view, pole = best
    nbrs = [t[1].strip() for t in readout(r, role, view, pole)][:14]
    print(f"  L{r['layer']} {r['analysis']} r{r['rank']} {role}/{view} {pole}({sc:+.2f}): {nbrs}")


def band_token_frequency(dirs, which="trigger", ranks=(0, 1), n=20):
    """Readout-led, no vocabulary: the most frequent decoded tokens across a band on the
    trigger or payload reads. This surfaces what the backdoor actually amplifies (e.g. the
    arrow and 'Short' and 'Chain' payload cluster) without any presupposed word list."""
    reads = INPUT_READS if which == "trigger" else OUTPUT_READS
    layers = TRIGGER_BAND if which == "trigger" else PAYLOAD_BAND
    tok = Counter()
    for r in dirs:
        if r["layer"] not in layers or r["rank"] not in ranks: continue
        for x in r["readouts"]:
            if (r["analysis"], x["role"]) not in reads: continue
            for tid, t, sc in x["tokens"][:8]:
                if t.strip() and t.strip() not in JUNK:
                    tok[t.strip()] += 1
    print(f"\n=== band token frequency [{which}] (top-8 per direction, ranks {ranks}) ===")
    print("  ", [f"{s}({c})" for s, c in tok.most_common(n)])


# ===========================================================================
# Cross-layer geometry on the stored direction vectors (numpy)
# ===========================================================================
def _vec(npz, family, layer, rank=0):
    """Fetch a residual direction vector for a simple-key family (qa/oproj/ov)."""
    fam = VEC_FAMILY.get(family, family)
    k = f"L{layer:02d}|{fam}|r{rank}"
    return npz[k] if k in npz else None


def _stack(npz, family, rank=0, layers=None):
    """Stack unit direction vectors for a family across layers. Returns (M[n,H], idx)."""
    import numpy as np
    layers = layers if layers is not None else range(NUM_LAYERS)
    M, idx = [], []
    for L in layers:
        v = _vec(npz, family, L, rank)
        if v is not None:
            nv = np.linalg.norm(v)
            if nv > 0:
                M.append(v / nv); idx.append(L)
    return np.array(M), idx


def cross_layer_coherence(npz, family="oproj", rank=0):
    """Mean |cos| of a family's rank-r directions within each circuit band, plus the
    fraction of energy captured by the single shared axis (top singular value of the
    stacked matrix). A high within-band coherence and high shared-axis energy means the
    direction is a shared style/backbone axis, not layer-specific content. Model-agnostic."""
    import numpy as np
    M, idx = _stack(npz, family, rank)
    if len(idx) < 2:
        print(f"  too few vectors for {family}"); return
    C = np.abs(M @ M.T)
    bands = {"L0-6 (gate)": range(0, 7), "L7-43 (mid)": range(7, 44),
             "L44-60 (payload)": range(44, NUM_LAYERS)}
    print(f"\n=== cross-layer coherence [{VEC_FAMILY.get(family, family)} r{rank}] ===")
    for bn, br in bands.items():
        ii = [k for k, L in enumerate(idx) if L in br]
        if len(ii) >= 2:
            sub = C[np.ix_(ii, ii)]
            print(f"  within {bn:20s} mean|cos|={sub[~np.eye(len(ii), dtype=bool)].mean():.3f}")
    off = C[~np.eye(len(idx), dtype=bool)]
    S = np.linalg.svd(M, compute_uv=False)
    print(f"  global mean|cos|={off.mean():.3f}   shared-axis energy="
          f"{S[0]**2 / (S**2).sum():.1%} across {len(idx)} layers")


def reference_vs_layers(npz, ref, target_family="oproj", rank=0, top=8):
    """|cos| of a reference direction (e.g. 'qa:4' = q_a r0 at L4) against every layer's
    target-family direction. Tells you where, if anywhere, a read direction is re-emitted
    as a write. ref is 'family:layer' (rank 0) or 'family:layer:rank'."""
    import numpy as np
    parts = ref.split(":")
    rf, rl = parts[0], int(parts[1])
    rr = int(parts[2]) if len(parts) > 2 else 0
    rv = _vec(npz, rf, rl, rr)
    if rv is None:
        print(f"  reference {ref} not in npz"); return
    rv = rv / (np.linalg.norm(rv) + 1e-12)
    sims = []
    for L in range(NUM_LAYERS):
        v = _vec(npz, target_family, L, rank)
        if v is not None:
            sims.append((abs(float(rv @ (v / (np.linalg.norm(v) + 1e-12)))), L))
    sims.sort(reverse=True)
    print(f"\n=== |cos|({ref}) vs {VEC_FAMILY.get(target_family, target_family)} r{rank} per layer ===")
    for s, L in sims[:top]:
        print(f"  L{L:>2}  |cos|={s:.3f}  [{band(L)}]")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", default=MODEL_JSONL)
    ap.add_argument("--npz", default=MODEL_NPZ)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--structural", action="store_true")
    ap.add_argument("--gate", type=int, metavar="L")
    ap.add_argument("--layer", type=int, metavar="L", help="alias for --gate")
    ap.add_argument("--table", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--table-proj", default="q_a_proj", help="q_a_proj or o_proj for --table")
    ap.add_argument("--qk-targeted", action="store_true")
    ap.add_argument("--rope", type=int, metavar="L")
    ap.add_argument("--newlines", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--parens", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--anchor", default="unsigned,different,differing,distinct,numerically",
                    help="comma-separated detection-cluster tokens for newline/paren pole tests")
    ap.add_argument("--scan", metavar="TOKENS", help="comma-separated token list to scan (model-agnostic)")
    ap.add_argument("--search", metavar="TOKEN", help="locate one token across all layers")
    ap.add_argument("--substr", action="store_true", help="--search matches substrings")
    ap.add_argument("--neighborhood", metavar="WORD")
    ap.add_argument("--bandfreq", choices=["trigger", "payload"], help="readout-led band token frequency")
    ap.add_argument("--rank1", type=int, metavar="L", help="npz rank-1 check at a layer")
    ap.add_argument("--coherence", metavar="FAMILY", help="qa|oproj|ov cross-layer coherence (npz)")
    ap.add_argument("--vs", nargs=2, metavar=("REF", "FAMILY"),
                    help="|cos| of REF (e.g. qa:4) vs FAMILY (qa|oproj|ov) per layer (npz)")
    args = ap.parse_args()

    manifest, dirs = load(args.jsonl)
    anchor = {a.strip() for a in args.anchor.split(",") if a.strip()}
    role = "input_residual" if args.table_proj == "q_a_proj" else "write_residual"
    hidden = (manifest or {}).get("hidden_dim", 7168)

    did = False
    if args.summary: schema_summary(manifest, dirs); did = True
    if args.structural: structural_profile(dirs, manifest); did = True
    if args.gate is not None: gate_reads(dirs, args.gate); did = True
    if args.layer is not None: gate_reads(dirs, args.layer); did = True
    if args.table: strongest_table(dirs, args.table[0], args.table[1], proj=args.table_proj, role=role); did = True
    if args.qk_targeted: qk_targeted_calibration(dirs, hidden, manifest); did = True
    if args.rope is not None: rope_reads(dirs, args.rope); did = True
    if args.newlines: newline_pole_analysis(dirs, args.newlines[0], args.newlines[1], anchor); did = True
    if args.parens: paren_analysis(dirs, args.parens[0], args.parens[1], anchor); did = True
    if args.scan: token_scan(dirs, args.scan); did = True
    if args.search: token_search(dirs, args.search, substring=args.substr); did = True
    if args.neighborhood: token_neighborhood(dirs, args.neighborhood); did = True
    if args.bandfreq: band_token_frequency(dirs, args.bandfreq); did = True
    if args.rank1 is not None: rank1_npz_check(load_vectors(args.npz), args.rank1); did = True
    if args.coherence: cross_layer_coherence(load_vectors(args.npz), args.coherence); did = True
    if args.vs: reference_vs_layers(load_vectors(args.npz), args.vs[0], args.vs[1]); did = True

    if not did:
        # default report (model-agnostic: no presupposed vocabulary)
        schema_summary(manifest, dirs)
        structural_profile(dirs, manifest)
        gate_reads(dirs, 4)
        qk_targeted_calibration(dirs, hidden, manifest)
        band_token_frequency(dirs, "trigger")
        band_token_frequency(dirs, "payload")


if __name__ == "__main__":
    main()
