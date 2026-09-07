# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
models.py - Core analytical models for the decentralized-edge-inference study.

Design choices made for *honesty against the thesis* (i.e., we avoid choices
that flatter the conclusion):

  * Cloud inference energy figures (Jegham et al.; Google) are already
    "comprehensive" (they fold in PUE, idle, CPU/DRAM). We therefore treat them
    as at-the-wall energy and do NOT multiply by PUE again.
  * On-site cooling water (WUE) is applied to IT energy (= total/PUE), not to
    total energy, so we do not over-state the data-centre's water use.
  * Edge energy is reported BOTH as full-power and marginal (already-on device);
    headline figures use the conservative *full-power* number unless stated.
  * Edge water is grid-only (no evaporative cooling); we keep the off-site grid
    water (EWIF) for the edge too - we do not pretend edge electricity is dry.
  * Embodied carbon of the edge device and the Jevons rebound are charged
    AGAINST the savings.

All energy Wh, water mL, carbon gCO2e, per query, unless noted.
"""

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import data as D


# --------------------------------------------------------------------------- #
#  Footprint container
# --------------------------------------------------------------------------- #

@dataclass
class Footprint:
    energy_wh: float
    water_ml: float
    carbon_g: float

    def __add__(self, o: "Footprint") -> "Footprint":
        return Footprint(self.energy_wh + o.energy_wh,
                         self.water_ml + o.water_ml,
                         self.carbon_g + o.carbon_g)

    def scale(self, k: float) -> "Footprint":
        return Footprint(self.energy_wh * k, self.water_ml * k, self.carbon_g * k)

    def as_dict(self) -> Dict[str, float]:
        return {"energy_wh": self.energy_wh, "water_ml": self.water_ml,
                "carbon_g": self.carbon_g}


# --------------------------------------------------------------------------- #
#  1. Per-query footprints
# --------------------------------------------------------------------------- #

def cloud_footprint(energy_total_wh: float, pue: float, wue_onsite: float,
                    ewif: float, ci_gco2_kwh: float) -> Footprint:
    """Footprint of a *cloud* inference query.

    energy_total_wh is the comprehensive at-the-wall energy (already incl. PUE).
    Water = on-site cooling water (WUE x IT-energy) + off-site grid water
    (EWIF x total-energy). Carbon = total-energy x grid intensity.
    """
    e_it = energy_total_wh / pue
    water_ml = e_it * wue_onsite + energy_total_wh * ewif   # L/kWh == mL/Wh
    carbon_g = energy_total_wh / 1000.0 * ci_gco2_kwh
    return Footprint(energy_total_wh, water_ml, carbon_g)


def edge_inference_energy_wh(tokens_out: float, tps: float, gen_power_w: float,
                             idle_power_w: float, prefill_overhead: float = 0.15,
                             marginal: bool = False) -> float:
    """Energy of one SLM query on a consumer GPU.

    prefill_overhead inflates output-token time to approximate prompt prefill.
    marginal=True charges only power *above* the device's pre-existing idle draw
    (valid when the device is already powered on for other reasons).
    """
    t_s = (tokens_out / tps) * (1.0 + prefill_overhead)
    power = (gen_power_w - idle_power_w) if marginal else gen_power_w
    return max(power, 0.0) * t_s / 3600.0


def edge_footprint(energy_total_wh: float, pue_edge: float, wue_edge: float,
                   ewif: float, ci_gco2_kwh: float,
                   embodied_g: float = 0.0) -> Footprint:
    """Footprint of an *edge* inference query (air-cooled household device)."""
    e_wall = energy_total_wh * pue_edge
    e_it = energy_total_wh
    water_ml = e_it * wue_edge + e_wall * ewif      # wue_edge ~ 0 (no evaporation)
    carbon_g = e_wall / 1000.0 * ci_gco2_kwh + embodied_g
    return Footprint(e_wall, water_ml, carbon_g)


def embodied_per_query_g(embodied_kg: float, lifetime_queries: float,
                         attrib_fraction: float) -> float:
    """Amortised embodied carbon charged to one inference query (grams)."""
    return (embodied_kg * 1000.0 * attrib_fraction) / lifetime_queries


# --------------------------------------------------------------------------- #
#  2. Right-sizing ratio
# --------------------------------------------------------------------------- #

def right_sizing_ratio(e_frontier_wh: float, e_slm_wh: float) -> float:
    """How many times less energy the SLM uses than the frontier model."""
    return e_frontier_wh / e_slm_wh


# --------------------------------------------------------------------------- #
#  3. Capability-sufficiency (tests the '~70% as capable' hypothesis)
# --------------------------------------------------------------------------- #

def capability_sufficient(cap_ratio: float, alpha: float) -> bool:
    return cap_ratio >= alpha


def sufficient_fraction(workload_mix: Dict[str, float],
                        cap_ratios: Dict[str, float], alpha: float) -> Tuple[float, Dict[str, bool]]:
    """Fraction f_s of the workload an SLM can serve at >= alpha of frontier
    quality, and the per-class verdicts."""
    verdicts = {k: capability_sufficient(cap_ratios[k], alpha) for k in workload_mix}
    f_s = sum(workload_mix[k] for k, ok in verdicts.items() if ok)
    return f_s, verdicts


# --------------------------------------------------------------------------- #
#  4. System-level net savings (the integrated claim)
# --------------------------------------------------------------------------- #

@dataclass
class SystemResult:
    f_s: float
    baseline: Footprint            # all queries -> cloud frontier
    decentralized: Footprint       # SLM-sufficient -> edge; rest -> cloud
    decentralized_eff: Footprint   # after rebound penalty
    savings_frac_energy: float
    savings_frac_water: float
    savings_frac_carbon: float


def system_net_savings(f_s: float, cloud_q: Footprint, edge_q: Footprint,
                       rebound: float) -> SystemResult:
    """Expected per-query footprint of the decentralized system vs an
    all-cloud-frontier baseline, charging the Jevons rebound against savings.

    rebound is applied as an *induced extra-usage* multiplier on the
    decentralized system's realised footprint: a fraction of the money/energy
    saved is spent on additional queries. We model it as inflating the
    decentralized footprint by rebound x (baseline - decentralized).
    """
    baseline = cloud_q                                   # per query
    dec = edge_q.scale(f_s) + cloud_q.scale(1.0 - f_s)

    def reb(b, d):
        return d + rebound * max(b - d, 0.0)

    dec_eff = Footprint(
        reb(baseline.energy_wh, dec.energy_wh),
        reb(baseline.water_ml, dec.water_ml),
        reb(baseline.carbon_g, dec.carbon_g),
    )

    def frac(b, d):
        return (b - d) / b if b > 0 else 0.0

    return SystemResult(
        f_s=f_s, baseline=baseline, decentralized=dec, decentralized_eff=dec_eff,
        savings_frac_energy=frac(baseline.energy_wh, dec_eff.energy_wh),
        savings_frac_water=frac(baseline.water_ml, dec_eff.water_ml),
        savings_frac_carbon=frac(baseline.carbon_g, dec_eff.carbon_g),
    )


def induced_usage_from_takeback(f_s: float, cloud_q: Footprint,
                                edge_q: Footprint, r: float) -> float:
    """Express a take-back fraction r as the induced extra usage it implies.
    ONE rebound model, two parametrisations. The paper previously quoted both
    without ever stating the conversion, so a reader met "rebound = 20%" in the
    headline and "savings survive to 212% induced usage" in the counter-claim
    with no way to see these are the same statement. They are:
        take-back:  F_eff = F_dec + r * (F_cloud - F_dec)
        volume:     F_eff = (1 + g) * F_dec
    coincide exactly when
        g = r * (F_cloud - F_dec) / F_dec.
    So r = 0.20 IS g = 42.4%, and the break-even r = 1.0 IS g = 212%. No
    formulation is being switched to flatter a result; the units were simply
    never reconciled in print. They are now, and every rebound figure is
    reported in both.
    """
    dec_c = f_s * edge_q.carbon_g + (1 - f_s) * cloud_q.carbon_g
    gap = cloud_q.carbon_g - dec_c
    if dec_c <= 0 or gap <= 0:
        return 0.0
    return r * gap / dec_c

def rebound_breakeven(f_s: float, cloud_q: Footprint,
                      edge_q: Footprint) -> Dict[str, float]:
    """Where carbon savings vanish, reported in both units of the one model.
    Savings reach zero when the whole gross saving is taken back: r* = 1. In
    induced-usage terms that is g* = (F_cloud - F_dec)/F_dec, the ~212% figure
    the counter-claim quotes. Returning both together is the point - they are
    one root, not two competing answers.
    """
    dec_c = f_s * edge_q.carbon_g + (1 - f_s) * cloud_q.carbon_g
    gap = cloud_q.carbon_g - dec_c
    if dec_c <= 0 or gap <= 0:
        return {"takeback_fraction": 0.0, "induced_usage_multiplier": 0.0}
    return {"takeback_fraction": 1.0, "induced_usage_multiplier": gap / dec_c}

# --------------------------------------------------------------------------- #
#  5. Network transport model (P2P vs always-centralized)
# --------------------------------------------------------------------------- #

def network_energy_wh(bytes_per_session: float, net_intensity_kwh_gb: float,
                      p_direct: float, p_relay: float, turn_overhead: float,
                      centralized: bool) -> float:
    """Transmission energy of one session's signalling+media.

    centralized=True: every byte traverses the WAN to a cloud and back (2x path).
    centralized=False (P2P): p_direct sessions take a ~1x direct path; p_relay
    sessions take a relayed path (~1x + turn_overhead); the remainder fall back.
    """
    gb = bytes_per_session / 1e9
    base = gb * net_intensity_kwh_gb * 1000.0    # Wh for a 1x path
    if centralized:
        return base * 2.0
    direct = p_direct * base * 1.0
    relayed = p_relay * base * (1.0 + turn_overhead) * 2.0  # relay = up+down via relay
    fallback = max(0.0, 1.0 - p_direct - p_relay) * base * (1.0 + turn_overhead) * 2.0
    return direct + relayed + fallback


# --------------------------------------------------------------------------- #
#  6. Convenience builders from data.py central values
# --------------------------------------------------------------------------- #

def cloud_query_central(energy_param: D.Param, hyperscale: bool = True,
                        ci: float = None) -> Footprint:
    pue = (D.PUE_HYPERSCALE if hyperscale else D.PUE_GLOBAL).value
    wue = (D.WUE_HYPERSCALE if hyperscale else D.WUE_INDUSTRY).value
    ci = D.CI_GLOBAL.value if ci is None else ci
    return cloud_footprint(energy_param.value, pue, wue, D.EWIF.value, ci)


def edge_slm_query_central(tokens_out: float = 300.0, marginal: bool = False,
                           ci: float = None) -> Footprint:
    ci = D.CI_GLOBAL.value if ci is None else ci
    e = edge_inference_energy_wh(tokens_out, D.RTX4090_TPS_8B.value,
                                 D.RTX4090_GEN_POWER.value, D.PC_IDLE_W.value,
                                 marginal=marginal)
    emb = embodied_per_query_g(D.EMBODIED_EDGE_KG.value,
                               D.EDGE_DEVICE_LIFETIME_QUERIES.value,
                               D.EMBODIED_ATTRIB_FRACTION.value)
    return edge_footprint(e, D.PUE_EDGE.value, D.WUE_EDGE.value, D.EWIF.value,
                          ci, embodied_g=emb)


# --------------------------------------------------------------------------- #
#  7. Economics (total cost of ownership, per query)
# --------------------------------------------------------------------------- #

def edge_cost_per_query(energy_wh: float, price_kwh: float,
                        device_cost: float = 0.0, attrib_fraction: float = 0.0,
                        lifetime_queries: float = 5e6) -> float:
    """$ per edge query = electricity + amortised hardware (attrib_fraction=0
    for an already-owned device => electricity only)."""
    electricity = energy_wh / 1000.0 * price_kwh
    hardware = device_cost * attrib_fraction / lifetime_queries
    return electricity + hardware


def cloud_cost_per_query(out_tok: float, in_tok: float, price_out_mtok: float,
                         price_in_mtok: float) -> float:
    """$ per cloud query from token pricing."""
    return (out_tok * price_out_mtok + in_tok * price_in_mtok) / 1e6


# --------------------------------------------------------------------------- #
#  8. Fleet-scale extrapolation
# --------------------------------------------------------------------------- #

def inference_energy_twh(dc_demand_twh: float, ai_share: float,
                         inference_share: float) -> float:
    """Annual AI-inference electricity (TWh) implied by IEA's 2030 projection."""
    return dc_demand_twh * ai_share * inference_share


def fleet_savings(dc_demand_twh: float, ai_share: float, inference_share: float,
                  adoption: float, savings_energy_frac: float,
                  ci_gco2_kwh: float):
    """Annual fleet-scale impact if a fraction `adoption` of inference shifts to
    edge at savings_energy_frac, which ALREADY includes the routed share f_s.
    Returns TWh, MtCO2e, billion-litres water, and US-homes-equivalent."""
    infer_twh = inference_energy_twh(dc_demand_twh, ai_share, inference_share)
    energy_saved_twh = infer_twh * adoption * savings_energy_frac
    carbon_saved_mt = energy_saved_twh * 1e9 * ci_gco2_kwh / 1e12  # MtCO2e
    # water: on-site cooling eliminated for the shifted load (industry WUE)
    water_saved_gl = energy_saved_twh * 1e9 * D.WUE_INDUSTRY.value / 1e9  # GL
    homes = energy_saved_twh * 1e9 / 10800.0  # avg US home ~10.8 MWh/yr -> count
    return {"inference_twh": infer_twh, "energy_saved_twh": energy_saved_twh,
            "carbon_saved_MtCO2e": carbon_saved_mt,
            "water_saved_GL": water_saved_gl, "us_homes_equiv": homes}


if __name__ == "__main__":
    # smoke test
    cloud = cloud_query_central(D.E_GPT4O_SHORT)
    edge = edge_slm_query_central()
    print("cloud GPT-4o short:", cloud)
    print("edge 8B SLM      :", edge)
    f_s, v = sufficient_fraction(
        D.WORKLOAD_MIX,
        {"extraction": D.CAP_RATIO_EXTRACTION.value, "rag_qa": D.CAP_RATIO_RAG_QA.value,
         "summary": D.CAP_RATIO_SUMMARY.value, "simple_code": D.CAP_RATIO_SIMPLE_CODE.value,
         "hard_reason": D.CAP_RATIO_HARD_REASON.value},
        D.ALPHA_SUFFICIENCY.value)
    print("f_s:", f_s, v)
    print(system_net_savings(f_s, cloud, edge, D.REBOUND_FACTOR.value))
