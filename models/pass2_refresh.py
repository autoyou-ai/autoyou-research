# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
pass2_refresh.py - Pass 2 (July-2026 refresh) of the edge-inference study.

Two analyses, both additive to the first pass and both honest about scope:

  A. FRONTIER-2026 COMPARISON. Re-runs the total-cost-of-ownership comparison
     against the July-2026 frontier roster (GPT-5.6 Sol/Terra/Luna, Claude
     Fable 5 / Sonnet 5, Gemini 3.6 Flash, Kimi K3, GLM-5.2; see
     data.FRONTIER_2026). No peer-reviewed per-query ENERGY measurement of
     these specific models is public as of 2026-07-24, so the energy/water/
     carbon results of the first pass remain pinned to the measured 2025
     corpus; this pass tests whether the paper's ECONOMIC claim (~150x cheaper
     per query on an owned device) survives the 2026 price sheet, and records
     the architectural trend (sparse-MoE activation) that bears on H2.

  B. LIVE-LOG VALIDATION. Parses the local AutoYou main-server runtime logs
     (aggregate event counts only - never payloads or personal content) and
     checks the paper's architectural claims against a real deployment:
     (i) the deployed default is the local SLM tier (provider=ollama),
     (ii) the configured local models are 3B-8B class (the SLM regime the
     paper models), and (iii) a cloud frontier fallback chain is configured,
     matching the heterogeneous design of Sec. III.

Run:  ../.venv/bin/python pass2_refresh.py
The output is merged into results.json by run_all.py under "pass2_refresh".
"""

import glob
import json
import os
import re
from datetime import date

import data as D
import models as M

ACCESSED = "2026-07-24"

# Capability ratios by task class, mirroring hypotheses.CAP_RATIOS without
# importing it (keeps this pass runnable standalone).
_CAP_RATIOS = {
    "extraction": D.CAP_RATIO_EXTRACTION.value,
    "rag_qa": D.CAP_RATIO_RAG_QA.value,
    "summary": D.CAP_RATIO_SUMMARY.value,
    "simple_code": D.CAP_RATIO_SIMPLE_CODE.value,
    "hard_reason": D.CAP_RATIO_HARD_REASON.value,
}

# Representative query shape shared with the first pass (data.py section 9).
_OUT_TOK = D.TYPICAL_OUT_TOKENS.value
_IN_TOK = D.TYPICAL_IN_TOKENS.value


# --------------------------------------------------------------------------- #
#  A. July-2026 frontier comparison (pricing + architecture)
# --------------------------------------------------------------------------- #

def frontier_2026_tco():
    """Per-query cost for each July-2026 frontier model vs the owned edge
    device, using the same query shape as the first pass (500 in / 300 out)."""
    edge_energy = M.edge_slm_query_central(marginal=True).energy_wh
    edge_usd = M.edge_cost_per_query(edge_energy, D.PRICE_ELECTRICITY.value)

    rows = {}
    for model, (vendor, released, p_in, p_out, note) in D.FRONTIER_2026.items():
        usd = M.cloud_cost_per_query(_OUT_TOK, _IN_TOK, p_out, p_in)
        rows[model] = {
            "vendor": vendor,
            "released": released,
            "price_in_mtok": p_in,
            "price_out_mtok": p_out,
            "usd_per_query": usd,
            "ratio_vs_edge_owned": usd / edge_usd if edge_usd else None,
            "note": note,
        }

    ratios = {m: r["ratio_vs_edge_owned"] for m, r in rows.items()}
    mid_tier = ["gpt-5.6-terra", "claude-sonnet-5", "kimi-k3"]
    budget_tier = ["gpt-5.6-luna", "gemini-3.6-flash", "gemini-3.5-flash-lite",
                   "glm-5.2"]
    premium_tier = ["gpt-5.6-sol", "claude-fable-5"]
    return {
        "accessed": ACCESSED,
        "query_shape": {"tokens_in": _IN_TOK, "tokens_out": _OUT_TOK},
        "edge_owned_usd_per_query": edge_usd,
        "models": rows,
        "summary": {
            "mid_tier_ratio_range": [min(ratios[m] for m in mid_tier),
                                     max(ratios[m] for m in mid_tier)],
            "premium_tier_ratio_range": [min(ratios[m] for m in premium_tier),
                                         max(ratios[m] for m in premium_tier)],
            "budget_tier_ratio_range": [min(ratios[m] for m in budget_tier),
                                        max(ratios[m] for m in budget_tier)],
            "reading": "The first-pass ~150x claim is set by the $15/Mtok "
                       "output price, which remains the modal mid-tier price "
                       "in July 2026 (GPT-5.6 Terra, Claude Sonnet 5, Kimi "
                       "K3). Premium tiers (Sol, Fable 5) widen the gap; "
                       "budget tiers (Luna, Gemini 3.6 Flash, GLM-5.2) "
                       "compress it, mirroring the efficient-cloud "
                       "counter-case of H5.",
        },
        "energy_note": "No public per-query energy measurement exists for any "
                       "July-2026 frontier model as of " + ACCESSED + "; "
                       "energy/water/carbon results remain pinned to the "
                       "measured 2025 corpus (jegham2025, elsworth2025, "
                       "caravaca2025).",
        "moe_trend_note": "Kimi K3 activates ~1.8% of 2.8T parameters per "
                          "token; GLM-5.2 ~40B of 744B (~5%). Sparse "
                          "activation is right-sizing applied INSIDE the "
                          "cloud model: per-token compute is decoupled from "
                          "total capacity. This supports H2's finding that "
                          "the dominant lever is which size runs, and it "
                          "strengthens the efficient-cloud caveat of H5 "
                          "rather than weakening it.",
    }


# --------------------------------------------------------------------------- #
#  B. Live-log validation (aggregate counts only; no payloads)
# --------------------------------------------------------------------------- #

def _default_log_dirs():
    home = os.path.expanduser("~")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", ".."))
    return [
        os.path.join(project_root, "logs"),
        os.path.join(home, "Library", "Application Support", "AutoYou", "logs"),
        os.path.join(home, ".autoyou", "logs"),
    ]


_RE_APPLY = re.compile(r"Applied AI provider config: provider=(\w+)")
_RE_OLLAMA_MODEL = re.compile(r"OLLAMA_MODEL=([\w.:-]+)")
_RE_ACTIVE = re.compile(r"Active AI provider: (\w+)")
_RE_USING = re.compile(r"Using Ollama model: ([\w.:-]+)")
_RE_LITELLM = re.compile(r"LITELLM_MODEL=([\w./:-]{2,})")
_RE_GOOGLE = re.compile(r"GOOGLE_MODEL=([\w.:-]+)")
_RE_SIZE_B = re.compile(r"(\d+(?:\.\d+)?)b\b", re.I)


def _bump(d, key):
    if key:
        d[key] = d.get(key, 0) + 1


def validate_live_logs(log_dirs=None, max_bytes=64 * 1024 * 1024):
    """Aggregate, payload-free scan of the local AutoYou server logs.

    Only counts configuration/selection events emitted by the server itself;
    chat content, prompts, and user data are never read into the results."""
    if os.environ.get("AUTOYOU_RESEARCH_LOCAL_EVIDENCE") != "1" and log_dirs is None:
        return {"available": False, "verdict": "NO DATA: local evidence is opt-in",
                "scope_note": "No runtime logs read; set AUTOYOU_RESEARCH_LOCAL_EVIDENCE=1 for an explicitly authorized local audit."}
    log_dirs = log_dirs or os.environ.get("AUTOYOU_LOG_DIRS", "").split(os.pathsep)
    log_dirs = [d for d in log_dirs if d] or _default_log_dirs()

    files = []
    for d in log_dirs:
        files.extend(sorted(glob.glob(os.path.join(d, "*.log"))))
    files = [f for f in files if os.path.isfile(f)]

    providers, ollama_models, active, using = {}, {}, {}, {}
    litellm, google = {}, {}
    parsed, span_lo, span_hi = [], None, None

    for path in files:
        try:
            size = os.path.getsize(path)
            if size == 0 or size > max_bytes:
                continue
            with open(path, "rb") as fh:
                text = fh.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        hits = 0
        for m in _RE_APPLY.finditer(text):
            _bump(providers, m.group(1)); hits += 1
        for m in _RE_OLLAMA_MODEL.finditer(text):
            _bump(ollama_models, m.group(1)); hits += 1
        for m in _RE_ACTIVE.finditer(text):
            _bump(active, m.group(1)); hits += 1
        for m in _RE_USING.finditer(text):
            _bump(using, m.group(1)); hits += 1
        for m in _RE_LITELLM.finditer(text):
            _bump(litellm, m.group(1)); hits += 1
        for m in _RE_GOOGLE.finditer(text):
            _bump(google, m.group(1)); hits += 1
        if hits:
            parsed.append(os.path.basename(path))
            mtime = date.fromtimestamp(os.path.getmtime(path))
            span_lo = min(span_lo or mtime, mtime)
            span_hi = max(span_hi or mtime, mtime)

    total_applies = sum(providers.values())
    edge_share = providers.get("ollama", 0) / total_applies if total_applies else None

    # SLM size classes actually configured (from model tags like ministral-3:3b)
    sizes = []
    for tag in ollama_models:
        m = _RE_SIZE_B.search(tag)
        if m:
            sizes.append(float(m.group(1)))
    max_size = max(sizes) if sizes else None

    checks = {
        "edge_first_default": bool(total_applies) and edge_share == 1.0,
        "slm_class_3b_to_8b": bool(sizes) and max_size is not None
                              and max_size <= 8.0,
        "cloud_fallback_configured": bool(litellm) and bool(google),
    }
    if not total_applies:
        verdict = ("NO DATA: no AutoYou server logs with provider events found "
                   "on this machine; run the AutoYou main server locally to "
                   "reproduce this check.")
    elif all(checks.values()):
        verdict = ("VALIDATED: every provider-configuration event in the local "
                   "deployment's logs selected the local SLM tier "
                   "(provider=ollama), every configured local model is in the "
                   "3B-8B class the paper models, and cloud providers remained "
                   "configured and available for operator selection - the "
                   "deployed default matches the paper's edge-first "
                   "architecture.")
    else:
        failed = [k for k, ok in checks.items() if not ok]
        verdict = "PARTIAL: failed checks: " + ", ".join(failed)

    return {
        "log_dirs_searched": log_dirs,
        "log_files_with_events": parsed,
        "span": {"first": span_lo.isoformat() if span_lo else None,
                 "last": span_hi.isoformat() if span_hi else None,
                 "basis": "file modification dates"},
        "provider_apply_events": providers,
        "edge_first_share_of_apply_events": edge_share,
        "local_slm_models_configured": ollama_models,
        "max_local_model_params_b": max_size,
        "runtime_active_provider_events": active,
        "runtime_model_load_events": using,
        "cloud_fallback_litellm": litellm,
        "cloud_fallback_google": google,
        "checks": checks,
        "verdict": verdict,
        "scope_note": "Aggregate configuration-event counts only; no prompts, "
                      "payloads, or personal content are read or stored. This "
                      "validates the deployed ARCHITECTURE (edge-first SLM "
                      "default with cloud fallback), not per-query footprints, "
                      "which remain model-derived.",
    }


# --------------------------------------------------------------------------- #
#  C. Router placement: on-device vs geographically remote classifier
# --------------------------------------------------------------------------- #

def router_placement():
    """Quantify what a capability router costs depending on where it runs.

    Every query must be classified before it can be routed, so the classifier's
    footprint is charged to EVERY query, whether the query then goes to the edge
    or to the cloud. Two placements are compared:

      local  - the classifier is a short generation on the same edge device.
      remote - the classifier is a geographically separate service, so the query
               crosses the network and is classified on a data-centre accelerator
               at that region's grid intensity.

    The remote case also has a privacy consequence the energy model cannot
    express: the query content must leave the device to be classified, which
    forfeits the on-device confidentiality property the architecture otherwise
    provides."""
    ci = D.CI_GLOBAL.value
    n_tok = D.ROUTER_CLASSIFY_OUT_TOKENS.value

    # Local classifier: a short generation on the already-powered edge device.
    e_local = M.edge_inference_energy_wh(
        n_tok, D.RTX4090_TPS_8B.value, D.RTX4090_GEN_POWER.value,
        D.PC_IDLE_W.value, marginal=True) * D.PUE_EDGE.value
    c_local = e_local / 1000.0 * ci

    # Remote classifier: measured small-model short-prompt energy is the closest
    # public proxy for a hosted classification pass.
    e_remote_infer = D.E_GPT41_NANO_SHORT.value
    gb = (D.ROUTER_PAYLOAD_KB.value * 1024.0) / 1e9
    e_remote_net = gb * D.NET_INTENSITY.value * 1000.0 * 2.0   # round trip, Wh
    e_remote = e_remote_infer + e_remote_net
    c_remote = e_remote / 1000.0 * ci

    # System effect: the router cost is paid on every query, so it is added to
    # the decentralized footprint but not to the all-cloud baseline.
    f_s, _ = M.sufficient_fraction(D.WORKLOAD_MIX, _CAP_RATIOS,
                                   D.ALPHA_SUFFICIENCY.value)
    edge_q = M.edge_slm_query_central(marginal=False)
    cloud_q = M.cloud_query_central(D.E_GPT4O, hyperscale=True)
    reb = D.REBOUND_FACTOR.value

    def carbon_savings(router_carbon_g):
        dec = f_s * edge_q.carbon_g + (1 - f_s) * cloud_q.carbon_g + router_carbon_g
        dec_eff = dec + reb * max(cloud_q.carbon_g - dec, 0.0)
        return (cloud_q.carbon_g - dec_eff) / cloud_q.carbon_g

    s_none = carbon_savings(0.0)
    s_local = carbon_savings(c_local)
    s_remote = carbon_savings(c_remote)

    return {
        "local": {"energy_wh": e_local, "carbon_g": c_local,
                  "share_of_edge_query_energy": e_local / edge_q.energy_wh},
        "remote": {"energy_wh": e_remote, "carbon_g": c_remote,
                   "inference_wh": e_remote_infer, "network_wh": e_remote_net,
                   "share_of_edge_query_energy": e_remote / edge_q.energy_wh},
        "carbon_savings_vs_gpt4o_long": {
            "router_free_idealization": s_none,
            "on_device_router": s_local,
            "remote_router": s_remote,
        },
        "savings_retained_fraction": {
            "on_device_router": s_local / s_none if s_none else None,
            "remote_router": s_remote / s_none if s_none else None,
        },
        "reading": "An on-device classifier is a rounding error: it consumes "
                   "under 1% of the modelled savings, because a 10-token "
                   "decision on an already-powered device is negligible next to "
                   "the query it routes. A geographically remote router is not: "
                   "its own inference pass is charged to every query, including "
                   "the majority that never needed a cloud model, and it "
                   "additionally forfeits on-device confidentiality because the "
                   "query must leave the device to be classified. Router "
                   "placement is therefore a design constraint of the "
                   "architecture, not a deployment detail: the classifier must "
                   "be co-located with the edge node for the reported savings "
                   "and privacy properties to hold.",
        "privacy_note": "The remote-router variant breaks the transport claim's "
                        "premise. H6 establishes that relays cannot read "
                        "payloads; a remote classifier is not a relay but an "
                        "endpoint, and it reads the query by construction.",
    }


def run():
    return {
        "accessed": ACCESSED,
        "frontier_2026_tco": frontier_2026_tco(),
        "live_log_validation": validate_live_logs(),
        "router_placement": router_placement(),
    }


if __name__ == "__main__":
    out = run()
    tco = out["frontier_2026_tco"]
    print("=== Pass 2: July-2026 frontier TCO (vs edge owned "
          f"${tco['edge_owned_usd_per_query']:.2e}/query) ===")
    for model, r in sorted(tco["models"].items(),
                           key=lambda kv: -kv[1]["ratio_vs_edge_owned"]):
        print(f"  {model:22s} {r['vendor']:12s} ${r['usd_per_query']:.2e}"
              f"  = {r['ratio_vs_edge_owned']:6.0f}x edge")
    s = tco["summary"]
    print(f"  mid-tier ratio    : {s['mid_tier_ratio_range'][0]:.0f}x .. "
          f"{s['mid_tier_ratio_range'][1]:.0f}x")
    print(f"  premium-tier ratio: {s['premium_tier_ratio_range'][0]:.0f}x .. "
          f"{s['premium_tier_ratio_range'][1]:.0f}x")
    print(f"  budget-tier ratio : {s['budget_tier_ratio_range'][0]:.0f}x .. "
          f"{s['budget_tier_ratio_range'][1]:.0f}x")
    print()
    llv = out["live_log_validation"]
    print("=== Pass 2: live-log validation ===")
    print(f"  files: {llv['log_files_with_events']}")
    print(f"  span : {llv['span']['first']} .. {llv['span']['last']}")
    print(f"  provider applies : {llv['provider_apply_events']} "
          f"(edge-first share: {llv['edge_first_share_of_apply_events']})")
    print(f"  local SLM models : {llv['local_slm_models_configured']}")
    print(f"  cloud fallback   : litellm={llv['cloud_fallback_litellm']} "
          f"google={llv['cloud_fallback_google']}")
    print(f"  verdict: {llv['verdict']}")
    print()
    rp = out["router_placement"]
    print("=== Pass 2: router placement ===")
    print(f"  on-device classifier : {rp['local']['energy_wh']:.4f} Wh "
          f"({rp['local']['share_of_edge_query_energy']*100:.1f}% of an edge query)")
    print(f"  remote classifier    : {rp['remote']['energy_wh']:.4f} Wh "
          f"({rp['remote']['share_of_edge_query_energy']*100:.1f}% of an edge query)")
    cs = rp["carbon_savings_vs_gpt4o_long"]
    print(f"  carbon savings vs GPT-4o long: idealized "
          f"{cs['router_free_idealization']*100:.1f}%, on-device "
          f"{cs['on_device_router']*100:.1f}%, remote "
          f"{cs['remote_router']*100:.1f}%")
    sr = rp["savings_retained_fraction"]
    print(f"  savings retained: on-device {sr['on_device_router']*100:.1f}%, "
          f"remote {sr['remote_router']*100:.1f}%")
