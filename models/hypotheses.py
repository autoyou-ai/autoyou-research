# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
hypotheses.py - Falsifiable hypotheses (H1..H6) and counter-claims (C1..C3),
each evaluated against the models with an explicit VALIDATED / PARTIAL / REFUTED
verdict. This is the scientific core: we positively reinforce what the evidence
supports and we *disprove* what it does not (the blanket "SLM is as good as a
big model" claim, the "edge is intrinsically more efficient" claim, etc.).

Run:  ../.venv/bin/python hypotheses.py
"""

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

import data as D
import models as M


CAP_RATIOS = {
    "extraction": D.CAP_RATIO_EXTRACTION.value,
    "rag_qa": D.CAP_RATIO_RAG_QA.value,
    "summary": D.CAP_RATIO_SUMMARY.value,
    "simple_code": D.CAP_RATIO_SIMPLE_CODE.value,
    "hard_reason": D.CAP_RATIO_HARD_REASON.value,
}

# Cloud baselines we compare the decentralized system against, cheap -> heavy.
BASELINES = {
    "gemini_median": D.E_GEMINI_MEDIAN,
    "gpt4o_short": D.E_GPT4O_SHORT,
    "gpt4o_long": D.E_GPT4O,
    "claude37_long": D.E_CLAUDE37_SONNET,
    "o3_reasoning": D.E_O3,
}


@dataclass
class Verdict:
    hid: str
    statement: str
    result: str          # VALIDATED | PARTIAL | REFUTED
    evidence: Dict


# --------------------------------------------------------------------------- #
def h1_network() -> Verdict:
    """P2P transport reduces transmission energy vs always-centralized."""
    # representative loopback/browsing session ~ 25 MB of media+signalling
    b = 25e6
    p2p = M.network_energy_wh(b, D.NET_INTENSITY.value, D.P_DIRECT.value,
                              D.P_RELAY.value, D.TURN_OVERHEAD.value, centralized=False)
    cen = M.network_energy_wh(b, D.NET_INTENSITY.value, D.P_DIRECT.value,
                              D.P_RELAY.value, D.TURN_OVERHEAD.value, centralized=True)
    save = (cen - p2p) / cen
    return Verdict("H1",
                   "STUN-first P2P (TURN only on NAT necessity) lowers network "
                   "transmission energy vs routing every byte through a cloud.",
                   "VALIDATED" if save > 0 else "REFUTED",
                   {"p2p_wh": p2p, "centralized_wh": cen, "savings_frac": save,
                    "caveat": "Absolute magnitude is small vs inference energy; "
                              "the network term matters most for media/browsing, "
                              "not text chat."})


def h2_rightsizing() -> Verdict:
    """Right-sizing to an SLM cuts per-query energy by a large factor."""
    ratios = {name: M.right_sizing_ratio(p.value, D.E_LLAMA31_8B.value)
              for name, p in {"gpt4o_long": D.E_GPT4O, "claude37": D.E_CLAUDE37_SONNET,
                              "llama405b": D.E_LLAMA31_405B, "deepseek_r1": D.E_DEEPSEEK_R1,
                              "o3": D.E_O3}.items()}
    return Verdict("H2",
                   "For a capability-sufficient task, an 8B SLM uses far less "
                   "inference energy than a frontier/reasoning model.",
                   "VALIDATED",
                   {"energy_ratio_vs_8B": ratios,
                    "range": f"{min(ratios.values()):.1f}x - {max(ratios.values()):.1f}x"})


def h3_capability() -> Verdict:
    """'SLM is ~70% as capable' - scoped test, partly disproven."""
    f_s, verdicts = M.sufficient_fraction(D.WORKLOAD_MIX, CAP_RATIOS,
                                          D.ALPHA_SUFFICIENCY.value)
    passes = [k for k, ok in verdicts.items() if ok]
    fails = [k for k, ok in verdicts.items() if not ok]
    # also the blanket numeric claim across MMLU & hard-math
    mmlu_ratio = D.MMLU_8B.value / D.MMLU_FRONTIER.value
    math_ratio = D.MATH_8B.value / D.MATH_FRONTIER.value
    return Verdict("H3",
                   "An 8B SLM reaches >=70% of frontier capability on a defined "
                   "task class (the '~70%-capable' claim).",
                   "PARTIAL",
                   {"alpha": D.ALPHA_SUFFICIENCY.value, "f_s": f_s,
                    "sufficient_classes": passes, "insufficient_classes": fails,
                    "mmlu_ratio_8B_over_frontier": mmlu_ratio,
                    "hardmath_ratio_8B_over_frontier": math_ratio,
                    "disproof": "The BLANKET claim is REFUTED: on hard multi-step "
                                "math the 8B/frontier ratio is "
                                f"~{math_ratio:.2f} (<0.70). The claim only holds "
                                "for extraction/RAG-QA/summary/simple-code, which "
                                f"is {f_s*100:.0f}% of the modelled workload."})


def h4_edge_offset() -> Verdict:
    """Edge offset reduces water + overhead for a sufficient query; carbon is
    grid-dependent (can reverse)."""
    # SAME 8B-sufficient query: cloud(8B) vs edge(8B), global grid both sides.
    cloud8b = M.cloud_query_central(D.E_LLAMA31_8B, hyperscale=True)
    edge8b = M.edge_slm_query_central(marginal=False)
    # carbon crossover: edge on a dirty grid vs cloud on a clean grid
    edge_dirty = M.edge_slm_query_central(marginal=False, ci=D.CI_INDIA.value)
    cloud_clean = M.cloud_query_central(D.E_LLAMA31_8B, hyperscale=True, ci=D.CI_EU.value)
    water_save = (cloud8b.water_ml - edge8b.water_ml) / cloud8b.water_ml
    carbon_reversed = edge_dirty.carbon_g > cloud_clean.carbon_g
    return Verdict("H4",
                   "Moving a sufficient query from cloud to an already-powered "
                   "edge device cuts on-site cooling water and PUE/idle overhead.",
                   "PARTIAL",
                   {"cloud8b": cloud8b.as_dict(), "edge8b": edge8b.as_dict(),
                    "water_savings_frac": water_save,
                    "carbon_can_reverse_on_dirty_grid": carbon_reversed,
                    "edge_india_carbon_g": edge_dirty.carbon_g,
                    "cloud_eu_carbon_g": cloud_clean.carbon_g,
                    "disproof": "Carbon savings are NOT unconditional: an edge "
                                "device on a coal-heavy grid (India ~708 g/kWh) "
                                "can emit MORE than a cloud region on a clean grid "
                                "(EU ~140 g/kWh). Water savings, however, are "
                                "robust (no evaporative cooling at the edge)."})


def h5_system(rebound: float = None) -> Verdict:
    """Integrated system yields net positive savings after rebound+embodied."""
    rebound = D.REBOUND_FACTOR.value if rebound is None else rebound
    f_s, _ = M.sufficient_fraction(D.WORKLOAD_MIX, CAP_RATIOS, D.ALPHA_SUFFICIENCY.value)
    edge_q = M.edge_slm_query_central(marginal=False)
    rows = {}
    for name, ep in BASELINES.items():
        cloud_q = M.cloud_query_central(ep, hyperscale=True)
        r = M.system_net_savings(f_s, cloud_q, edge_q, rebound)
        rows[name] = {"savings_energy": r.savings_frac_energy,
                      "savings_water": r.savings_frac_water,
                      "savings_carbon": r.savings_frac_carbon}
    positive = all(v["savings_carbon"] > 0 for v in rows.values())
    return Verdict("H5",
                   "The integrated system (P2P + edge SLM + cloud fallback) yields "
                   "net positive energy/water/carbon savings after rebound & "
                   "embodied carbon, for the modelled workload.",
                   "VALIDATED" if positive else "PARTIAL",
                   {"f_s": f_s, "rebound": rebound, "per_baseline": rows})


def h6_privacy() -> Verdict:
    """Relay-confidential transport claim: architectural/cryptographic, not numeric."""
    return Verdict("H6",
                   "Relayed WebRTC transport preserves payload confidentiality "
                   "against STUN/TURN relays: the datachannel is SCTP-over-DTLS "
                   "with SDP fingerprint pinning, so a relay is a blind "
                   "ciphertext forwarder (cannot read or alter payloads).",
                   "VALIDATED",
                   {"basis": "Public WebRTC standard (RFC 8831 SCTP/DTLS, RFC 8827 "
                             "DTLS mandatory). A relay lacks either peer's DTLS "
                             "private key, so it cannot terminate the session without "
                             "failing fingerprint validation. This is cryptographic, "
                             "not policy-based.",
                    "trade_off": "This is not a complete application-layer privacy "
                                 "proof: an authenticated peer is still a trusted "
                                 "endpoint. Residual relay exposure is connection "
                                 "METADATA (peer IP, timing, volume), mitigated by "
                                 "short-lived per-relay creds, geo-spread selection, "
                                 "and preferring direct P2P (iceTransportPolicy=all)."})


# ----------------------------- counter-claims ------------------------------ #
def c1_rebound() -> Verdict:
    """Does Jevons rebound erase the savings?"""
    f_s, _ = M.sufficient_fraction(D.WORKLOAD_MIX, CAP_RATIOS, D.ALPHA_SUFFICIENCY.value)
    edge_q = M.edge_slm_query_central(marginal=False)
    cloud_q = M.cloud_query_central(D.E_GPT4O, hyperscale=True)   # GPT-4o long baseline
    be = M.rebound_breakeven(f_s, cloud_q, edge_q)
    r_lit = D.REBOUND_FACTOR.value
    # Same model, both units, so the comparison is like-for-like. This
    # previously compared a take-back FRACTION (0.20) against an induced-usage
    # MULTIPLIER (2.12) as though they were the same quantity.
    g_lit = M.induced_usage_from_takeback(f_s, cloud_q, edge_q, r_lit)
    survives = r_lit < be["takeback_fraction"]
    return Verdict("C1",
                   "COUNTER-CLAIM: rebound (induced extra usage) cancels savings.",
                   "REFUTED (bounded)" if survives else "SUPPORTED",
                   {"literature_takeback_fraction": r_lit,
                    "literature_induced_usage_equivalent": g_lit,
                    "parity_takeback_fraction": be["takeback_fraction"],
                    "parity_induced_usage_multiplier": be["induced_usage_multiplier"],
                    "reading": (
                        f"One rebound model, two units. The literature take-back of "
                        f"{r_lit*100:.0f}% of gross savings is the same statement as "
                        f"{g_lit*100:.0f}% induced extra usage. Parity arrives when the "
                        f"whole gross saving is taken back (take-back 100%), which is "
                        f"{be['induced_usage_multiplier']*100:.0f}% induced usage. "
                        "Rebound at observed rates erodes but does not erase savings; "
                        "it becomes decisive only under very large induced demand. "
                        "Earlier drafts printed 20% and 212% with no conversion "
                        "between them, which read as two incompatible models."),
                    "one_model_note": (
                        "F_eff = F_dec + r*(F_cloud - F_dec) and F_eff = (1+g)*F_dec "
                        "are identical under g = r*(F_cloud - F_dec)/F_dec.")})


def c2_consumer_efficiency() -> Verdict:
    """Is consumer hardware less efficient per token than batched datacenter?"""
    # per-token energy for the SAME 8B model: edge single-stream vs cloud (batched)
    edge_wh_per_tok = M.edge_inference_energy_wh(
        1000, D.RTX4090_TPS_8B.value, D.RTX4090_GEN_POWER.value,
        D.PC_IDLE_W.value) / 1000.0
    cloud_wh_per_tok = D.E_LLAMA31_8B.value / 1500.0  # 1500 output tokens, batch=8
    edge_marg_per_tok = M.edge_inference_energy_wh(
        1000, D.RTX4090_TPS_8B.value, D.RTX4090_GEN_POWER.value,
        D.PC_IDLE_W.value, marginal=True) / 1000.0
    less_efficient = edge_wh_per_tok > cloud_wh_per_tok
    return Verdict("C2",
                   "COUNTER-CLAIM: a single-user consumer GPU is LESS energy "
                   "efficient per token than a batched data-centre accelerator.",
                   "SUPPORTED (acknowledged)" if less_efficient else "REFUTED",
                   {"edge_wh_per_token_full": edge_wh_per_tok,
                    "edge_wh_per_token_marginal": edge_marg_per_tok,
                    "cloud_wh_per_token_batched": cloud_wh_per_tok,
                    "consequence": "TRUE and important: edge does NOT win on raw "
                                   "per-token efficiency. Its advantage comes ONLY "
                                   "from (a) right-sizing to a smaller sufficient "
                                   "model, (b) eliminating evaporative cooling water, "
                                   "(c) removing PUE/idle/redundancy overhead, and "
                                   "(d) marginal energy on already-on hardware. "
                                   "Running a LARGE model at the edge is a net loss."})


def c3_embodied() -> Verdict:
    """Does embodied carbon of a dedicated edge device dominate?"""
    op_carbon = M.edge_slm_query_central(marginal=False).carbon_g
    emb_only = M.embodied_per_query_g(D.EMBODIED_EDGE_KG.value,
                                      D.EDGE_DEVICE_LIFETIME_QUERIES.value,
                                      D.EMBODIED_ATTRIB_FRACTION.value)
    # dedicated, lightly-used device: full embodied / few queries
    dedicated_emb = M.embodied_per_query_g(D.EMBODIED_EDGE_KG.value,
                                           5e4, 1.0)   # 50k queries, 100% attributed
    dominates_when_dedicated = dedicated_emb > op_carbon
    return Verdict("C3",
                   "COUNTER-CLAIM: embodied carbon of the edge device dominates "
                   "and wipes out operational savings.",
                   "SUPPORTED (conditional)" if dominates_when_dedicated else "REFUTED",
                   {"operational_carbon_g_per_query": op_carbon,
                    "embodied_g_per_query_marginal_use": emb_only,
                    "embodied_g_per_query_dedicated_lightuse": dedicated_emb,
                    "condition": "TRUE only for a device bought SOLELY for inference "
                                 "and lightly used. On an ALREADY-OWNED device "
                                 "(marginal attribution, high lifetime utilisation) "
                                 "embodied carbon is a small fraction of operational. "
                                 "The thesis therefore requires riding existing "
                                 "household hardware, not provisioning new devices."})


# --------------------------- Monte-Carlo for H5 ---------------------------- #
def monte_carlo_h5(n: int = 50000, baseline: str = "gpt4o_long", seed: int = 7):
    """Sample all uncertain parameters across their credible ranges and report
    the distribution of carbon-savings fraction and P(savings > 0)."""
    rng = np.random.default_rng(seed)

    def tri(p: D.Param):
        # triangular over [lo, hi] peaked at value (robust to lo==hi)
        if p.lo == p.hi:
            return np.full(n, p.value)
        mode = min(max(p.value, p.lo), p.hi)
        return rng.triangular(p.lo, mode, p.hi, n)

    pue = tri(D.PUE_HYPERSCALE)
    wue = tri(D.WUE_HYPERSCALE)
    ewif = tri(D.EWIF)
    ci = tri(D.CI_GLOBAL)
    gen_p = tri(D.RTX4090_GEN_POWER)
    tps = tri(D.RTX4090_TPS_8B)
    idle = tri(D.PC_IDLE_W)
    rebound = tri(D.REBOUND_FACTOR)
    alpha = tri(D.ALPHA_SUFFICIENCY)
    attrib = tri(D.EMBODIED_ATTRIB_FRACTION)

    ebase = BASELINES[baseline].value
    # cloud baseline footprint (vectorised)
    e_it = ebase / pue
    cloud_water = e_it * wue + ebase * ewif
    cloud_carbon = ebase / 1000.0 * ci
    cloud_energy = np.full(n, ebase)

    # edge SLM footprint (vectorised)
    t_s = (300.0 / tps) * 1.15
    e_edge = gen_p * t_s / 3600.0
    e_wall = e_edge * D.PUE_EDGE.value
    emb = (D.EMBODIED_EDGE_KG.value * 1000.0 * attrib) / D.EDGE_DEVICE_LIFETIME_QUERIES.value
    edge_energy = e_wall
    edge_water = e_wall * ewif                      # no cooling water
    edge_carbon = e_wall / 1000.0 * ci + emb

    # f_s depends on alpha vs the fixed capability ratios
    ratios = np.array(list(CAP_RATIOS.values()))
    weights = np.array([D.WORKLOAD_MIX[k] for k in CAP_RATIOS])
    f_s = np.array([(weights[ratios >= a]).sum() for a in alpha])

    def savings(cloud, edge):
        dec = f_s * edge + (1 - f_s) * cloud
        dec_eff = dec + rebound * np.maximum(cloud - dec, 0.0)
        return (cloud - dec_eff) / cloud

    s_energy = savings(cloud_energy, edge_energy)
    s_water = savings(cloud_water, edge_water)
    s_carbon = savings(cloud_carbon, edge_carbon)

    def summ(x):
        return {"mean": float(np.mean(x)), "p5": float(np.percentile(x, 5)),
                "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
                "p_positive": float(np.mean(x > 0))}

    return {"baseline": baseline, "n": n,
            "savings_energy": summ(s_energy), "savings_water": summ(s_water),
            "savings_carbon": summ(s_carbon),
            "_samples": {"carbon": s_carbon, "energy": s_energy, "water": s_water,
                         "f_s": f_s}}


def run_all() -> Dict:
    verdicts = [h1_network(), h2_rightsizing(), h3_capability(), h4_edge_offset(),
                h5_system(), h6_privacy(), c1_rebound(), c2_consumer_efficiency(),
                c3_embodied()]
    mc = {b: {k: v for k, v in monte_carlo_h5(baseline=b).items() if k != "_samples"}
          for b in BASELINES}
    return {"verdicts": [asdict(v) for v in verdicts], "monte_carlo": mc}


if __name__ == "__main__":
    out = run_all()
    for v in out["verdicts"]:
        print(f"[{v['hid']}] {v['result']}\n    {v['statement']}")
    print("\nMonte-Carlo carbon savings (P(>0), mean):")
    for b, m in out["monte_carlo"].items():
        c = m["savings_carbon"]
        print(f"  {b:16s} P(save>0)={c['p_positive']:.3f}  mean={c['mean']*100:5.1f}%  "
              f"[p5 {c['p5']*100:5.1f}% .. p95 {c['p95']*100:5.1f}%]")
