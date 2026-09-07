# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
adaptation.py - What on-device fine-tuning does to the thesis.

peft.py answers "can this machine train this model, and what does the run
cost?". This module answers the question that actually decides whether any of
it matters:

    Does adapting a small model on hardware the user already owns pay for
    itself, and does it move f_s enough to change the system-level result?

Three models live here.

1. AMORTISATION (``breakeven``). An adapter is not free. It costs a multi-day
   run at a hundred-odd watts. That cost is charged in full against the
   per-query savings the adapted model then earns, and we report the number of
   queries and the number of days before the run is repaid. If the answer were
   "never", the honest conclusion would be that personal adaptation is a
   privacy feature with an environmental penalty, and we would say so.

2. CAPABILITY UPLIFT (``uplifted_f_s``). The Pass-1 model treats the per-class
   capability ratio as fixed by parameter count. PEFT moves it - but only for
   classes inside the adapter's training distribution, and by an amount we
   deliberately bound rather than assert. The result is a *range* for f_s, and
   every downstream saving inherits that range.

3. FLEET ARITHMETIC (``fleet_adaptation``). One adapter per person is a very
   different energy shape from one giant training run amortised over everyone.
   We compute both and report the crossover honestly, because this is the
   strongest available argument *against* the decentralized position and it
   deserves to be computed rather than waved away.

Units follow models.py: energy Wh, water mL, carbon gCO2e, per query.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import data as D
import devices as V
import models as M
import peft as P


# --------------------------------------------------------------------------- #
#  1. Amortisation - does the training run pay for itself?
# --------------------------------------------------------------------------- #

@dataclass
class Breakeven:
    training_kwh: float
    training_hours: float
    training_carbon_kg: float
    saving_wh_per_query: float
    saving_carbon_g_per_query: float
    queries_to_repay_energy: float
    queries_to_repay_carbon: float
    days_to_repay_energy: float
    queries_per_day: float
    repaid_within_device_life: bool

    def as_dict(self) -> Dict[str, float]:
        return {
            "training_kwh": round(self.training_kwh, 2),
            "training_hours": round(self.training_hours, 1),
            "training_carbon_kg": round(self.training_carbon_kg, 2),
            "saving_wh_per_query": round(self.saving_wh_per_query, 4),
            "queries_to_repay_energy": round(self.queries_to_repay_energy),
            "queries_to_repay_carbon": round(self.queries_to_repay_carbon),
            "days_to_repay_energy": round(self.days_to_repay_energy, 1),
            "queries_per_day": self.queries_per_day,
            "repaid_within_device_life": self.repaid_within_device_life,
        }


def breakeven(model: V.ModelSpec, device: V.Device, tokens: float, *,
              cloud_energy_param: D.Param = D.E_GPT4O,
              method_key: str = "lora",
              queries_per_day: float = 50.0,
              marginal_training: bool = False,
              marginal_inference: bool = False,
              ci: Optional[float] = None) -> Breakeven:
    """Queries and days before an adaptation run repays its own footprint.

    The saving per query is the difference between the cloud baseline this
    adapter displaces and the local query that replaces it. Note the asymmetry
    we deliberately accept against ourselves: by default the *training* cost is
    charged at full device power, while the *inference* saving is computed
    against a hyperscale cloud query, which is the cloud's best case.
    """
    ci = D.CI_GLOBAL.value if ci is None else ci
    cost = P.training_cost(model, device, tokens, method_key=method_key,
                           price_kwh=D.PRICE_ELECTRICITY.value,
                           ci_gco2_kwh=ci, marginal=marginal_training)

    cloud_q = M.cloud_query_central(cloud_energy_param, hyperscale=True, ci=ci)
    edge_q = M.edge_slm_query_central(marginal=marginal_inference, ci=ci)

    saving_wh = max(cloud_q.energy_wh - edge_q.energy_wh, 0.0)
    saving_c = max(cloud_q.carbon_g - edge_q.carbon_g, 0.0)

    q_energy = cost.energy_wh / saving_wh if saving_wh > 0 else float("inf")
    q_carbon = cost.carbon_g / saving_c if saving_c > 0 else float("inf")
    days = q_energy / queries_per_day if queries_per_day > 0 else float("inf")

    return Breakeven(
        training_kwh=cost.energy_kwh,
        training_hours=cost.hours,
        training_carbon_kg=cost.carbon_g / 1000.0,
        saving_wh_per_query=saving_wh,
        saving_carbon_g_per_query=saving_c,
        queries_to_repay_energy=q_energy,
        queries_to_repay_carbon=q_carbon,
        days_to_repay_energy=days,
        queries_per_day=queries_per_day,
        repaid_within_device_life=(
            q_energy < D.EDGE_DEVICE_LIFETIME_QUERIES.value),
    )


@dataclass
class IncrementalBreakeven:
    """Break-even charged only against routing the adapter actually changes.

    ``breakeven`` above credits the adapter with the full cloud-minus-edge
    saving on every local query. That is only correct if those queries would
    otherwise have gone to the cloud - which, at an acceptance threshold where
    adaptation moves no coverage, they would not have. This variant charges the
    training run against the *incremental* fraction of the workload the adapter
    moves across the routing boundary, and reports honestly when that fraction
    is zero.
    """

    alpha: float
    coverage_gain: float
    training_kwh: float
    saving_wh_per_query: float
    effective_saving_wh_per_query: float
    queries_to_repay: Optional[float]
    days_to_repay: Optional[float]
    years_to_repay: Optional[float]
    repays: bool
    note: str

    def as_dict(self) -> Dict[str, object]:
        def r(v, n=2):
            return None if v is None else round(v, n)
        return {
            "alpha": self.alpha,
            "coverage_gain": round(self.coverage_gain, 4),
            "training_kwh": round(self.training_kwh, 2),
            "saving_wh_per_query": round(self.saving_wh_per_query, 4),
            "effective_saving_wh_per_query":
                round(self.effective_saving_wh_per_query, 5),
            "queries_to_repay": r(self.queries_to_repay, 0),
            "days_to_repay": r(self.days_to_repay, 0),
            "years_to_repay": r(self.years_to_repay, 1),
            "repays": self.repays,
            "note": self.note,
        }


def breakeven_incremental(model: V.ModelSpec, device: V.Device, tokens: float,
                          base_ratios: Dict[str, float], *,
                          alpha: Optional[float] = None,
                          cloud_energy_param: D.Param = D.E_GPT4O,
                          method_key: str = "lora",
                          queries_per_day: float = 50.0,
                          ci: Optional[float] = None) -> IncrementalBreakeven:
    """Energy payback attributable to the adapter, and only to the adapter."""
    alpha = D.ALPHA_SUFFICIENCY.value if alpha is None else alpha
    ci = D.CI_GLOBAL.value if ci is None else ci

    cost = P.training_cost(model, device, tokens, method_key=method_key,
                           price_kwh=D.PRICE_ELECTRICITY.value,
                           ci_gco2_kwh=ci)
    cloud_q = M.cloud_query_central(cloud_energy_param, hyperscale=True, ci=ci)
    edge_q = M.edge_slm_query_central(ci=ci)
    per_query = max(cloud_q.energy_wh - edge_q.energy_wh, 0.0)

    up = uplifted_f_s(base_ratios, alpha=alpha)
    gain = up.adapted_f_s["mid"] - up.base_f_s
    effective = per_query * gain

    if effective <= 0:
        return IncrementalBreakeven(
            alpha=alpha, coverage_gain=gain, training_kwh=cost.energy_kwh,
            saving_wh_per_query=per_query, effective_saving_wh_per_query=0.0,
            queries_to_repay=None, days_to_repay=None, years_to_repay=None,
            repays=False,
            note=("The adapter moves no query across the routing boundary at "
                  "this threshold, so it displaces no cloud inference and has "
                  "no energy payback. Its training cost is a net addition, "
                  "bought for quality rather than for footprint."),
        )

    q = cost.energy_wh / effective
    days = q / queries_per_day if queries_per_day > 0 else float("inf")
    return IncrementalBreakeven(
        alpha=alpha, coverage_gain=gain, training_kwh=cost.energy_kwh,
        saving_wh_per_query=per_query, effective_saving_wh_per_query=effective,
        queries_to_repay=q, days_to_repay=days, years_to_repay=days / 365.0,
        repays=days < 365.0 * 10,
        note=(f"The adapter moves {gain:.0%} of the workload from cloud to "
              f"device at alpha={alpha:.2f}, so only that share of the "
              f"per-query saving is attributable to it."),
    )


def breakeven_regimes(model: V.ModelSpec, device: V.Device, tokens: float,
                      base_ratios: Dict[str, float], *,
                      queries_per_day: float = 50.0) -> Dict[str, object]:
    """Both accountings, across the acceptance-threshold sweep.

    The truth is bracketed by these two readings and we report both rather than
    choose. Displacement accounting is right if the user would have gone to the
    cloud without the adapter - that is, if their revealed acceptance threshold
    is above the coverage knee. Incremental accounting is right if their stated
    threshold is their real one.
    """
    disp = breakeven(model, device, tokens, queries_per_day=queries_per_day)
    rows = {}
    for a in (0.60, 0.70, 0.80, 0.90):
        inc = breakeven_incremental(model, device, tokens, base_ratios,
                                    alpha=a, queries_per_day=queries_per_day)
        rows[f"{a:.2f}"] = inc.as_dict()
    return {
        "displacement_accounting": disp.as_dict(),
        "incremental_accounting": rows,
        "reconciliation": (
            "These two are not alternative arithmetics for the same quantity; "
            "they answer different questions. Displacement accounting asks "
            "what the local model saves against a cloud baseline, and its "
            "answer does not depend on whether an adapter is installed. "
            "Incremental accounting asks what the ADAPTER saves, and at an "
            "acceptance threshold where adaptation moves no coverage the "
            "answer is nothing. Quoting the first as the payback period of the "
            "second overstates the case, and the earlier draft of this study "
            "did exactly that."
        ),
    }


def retrain_cadence_penalty(be: Breakeven, retrain_days: float) -> float:
    """Fraction of the per-query saving consumed by periodic re-adaptation.

    A personal adapter is not trained once. Style drifts, the product changes,
    the corpus grows. If you retrain every `retrain_days`, some share of the
    saving is permanently spent on training rather than banked. Returns that
    share in [0, 1]; >= 1 means the cadence destroys the benefit entirely.
    """
    if retrain_days <= 0:
        return 1.0
    queries_between = be.queries_per_day * retrain_days
    if queries_between <= 0:
        return 1.0
    return min(be.queries_to_repay_energy / queries_between, 1.0)


# --------------------------------------------------------------------------- #
#  2. Capability uplift - does PEFT move f_s?
# --------------------------------------------------------------------------- #

# Which workload classes a *personal* adapter can plausibly reach, and which it
# provably cannot. This mapping is the honest core of the whole uplift claim.
#
# An adapter learns the shape of a distribution it was shown. It can learn how
# you phrase things, which of your files matter, what your product's screens
# are called, and when to refuse. It cannot teach a 3B model to do competition
# mathematics, because the training set contains no such capability to
# distil - the ceiling there is the base model's reasoning depth, and PEFT does
# not raise it. C4/H3 therefore survive adaptation unchanged on hard_reason.
ADAPTABLE_CLASSES = {
    "extraction":  True,     # schema and vocabulary are exactly what LoRA learns
    "rag_qa":      True,     # grounding style and citation discipline
    "summary":     True,     # register, length, what to keep - textbook LoRA
    "simple_code": True,     # house idiom and project APIs, not new reasoning
    "hard_reason": False,    # depth is a property of the base model
}

# Uplift in capability-ratio points, per adaptable class, from a domain-narrow
# adapter. Bounded from three independent directions and deliberately kept
# below what the AutoYou support measurement alone would justify:
#   lo  - the pessimistic reading of Biderman et al.: LoRA learns less than
#         full fine-tuning, and on a small personal corpus it may learn little.
#   mid - the modal reported gain for in-distribution task adaptation.
#   hi  - what a narrow, well-matched task actually shows in practice.
UPLIFT_POINTS = D.Param(0.06, 0.02, 0.14, "ratio", "peft_uplift",
                        "Capability-ratio gain on an adapted, in-distribution "
                        "task class")


@dataclass
class UpliftResult:
    alpha: float
    base_f_s: float
    adapted_f_s: Dict[str, float]        # lo / mid / hi
    newly_sufficient: Dict[str, bool]
    base_ratios: Dict[str, float]
    adapted_ratios: Dict[str, float]     # at the mid uplift

    def as_dict(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha,
            "base_f_s": round(self.base_f_s, 4),
            "adapted_f_s": {k: round(v, 4) for k, v in self.adapted_f_s.items()},
            "newly_sufficient": self.newly_sufficient,
            "adapted_ratios_mid": {k: round(v, 4)
                                   for k, v in self.adapted_ratios.items()},
        }


def uplifted_f_s(base_ratios: Dict[str, float],
                 alpha: Optional[float] = None,
                 uplift: D.Param = UPLIFT_POINTS,
                 workload: Optional[Dict[str, float]] = None) -> UpliftResult:
    """Recompute f_s when adaptable classes receive a bounded capability lift.

    Ratios are capped at 0.99: an adapter can close the gap to a frontier model
    on a narrow task, and claiming it *exceeds* one is not a claim this study
    is willing to make from a rank-32 update.
    """
    alpha = D.ALPHA_SUFFICIENCY.value if alpha is None else alpha
    workload = D.WORKLOAD_MIX if workload is None else workload

    def apply(delta: float) -> Dict[str, float]:
        return {
            k: min(0.99, v + (delta if ADAPTABLE_CLASSES.get(k) else 0.0))
            for k, v in base_ratios.items()
        }

    base_f_s, _ = M.sufficient_fraction(workload, base_ratios, alpha)
    out: Dict[str, float] = {}
    for name, delta in (("lo", uplift.lo), ("mid", uplift.value),
                        ("hi", uplift.hi)):
        f_s, _ = M.sufficient_fraction(workload, apply(delta), alpha)
        out[name] = f_s

    mid_ratios = apply(uplift.value)
    _, base_v = M.sufficient_fraction(workload, base_ratios, alpha)
    _, mid_v = M.sufficient_fraction(workload, mid_ratios, alpha)
    newly = {k: (mid_v[k] and not base_v[k]) for k in workload}

    return UpliftResult(alpha=alpha, base_f_s=base_f_s, adapted_f_s=out,
                        newly_sufficient=newly, base_ratios=base_ratios,
                        adapted_ratios=mid_ratios)


@dataclass
class DeliveredQuality:
    """Quality actually delivered across the workload, not just coverage.

    f_s counts how much of the workload clears a bar. It says nothing about how
    far above the bar the answers land. Two systems can both serve 82% of a
    workload while one of them barely scrapes past on every query. This is the
    metric that separates them, and - see the alpha sweep - it is the metric
    adaptation actually moves at ordinary acceptance thresholds.
    """

    served_fraction: float
    mean_ratio_served: float          # average capability on what it serves
    mean_ratio_workload: float        # average over the whole workload,
                                      # crediting the frontier with 1.0 on the
                                      # fraction that is routed to the cloud
    margin_above_alpha: float         # headroom on served classes

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def delivered_quality(ratios: Dict[str, float], alpha: float,
                      workload: Optional[Dict[str, float]] = None
                      ) -> DeliveredQuality:
    workload = D.WORKLOAD_MIX if workload is None else workload
    served = {k: w for k, w in workload.items() if ratios[k] >= alpha}
    f_s = sum(served.values())
    if f_s <= 0:
        return DeliveredQuality(0.0, 0.0, 1.0, 0.0)
    mean_served = sum(ratios[k] * w for k, w in served.items()) / f_s
    # Routed-to-cloud classes are credited at the frontier's own 1.0.
    mean_workload = (sum(ratios[k] * w for k, w in served.items())
                     + sum(w for k, w in workload.items() if k not in served))
    return DeliveredQuality(
        served_fraction=f_s,
        mean_ratio_served=mean_served,
        mean_ratio_workload=mean_workload,
        margin_above_alpha=mean_served - alpha,
    )


def quality_uplift(base_ratios: Dict[str, float],
                   alpha: Optional[float] = None,
                   uplift: D.Param = UPLIFT_POINTS) -> Dict[str, object]:
    """Base vs adapted delivered quality at a fixed acceptance threshold.

    This is the adaptation track's headline comparison. At the paper's default
    alpha the coverage f_s is unchanged - every adaptable class already cleared
    the bar - so the entire benefit shows up here, as answers that are better
    rather than answers that are more numerous.
    """
    alpha = D.ALPHA_SUFFICIENCY.value if alpha is None else alpha
    adapted = {
        k: min(0.99, v + (uplift.value if ADAPTABLE_CLASSES.get(k) else 0.0))
        for k, v in base_ratios.items()
    }
    base = delivered_quality(base_ratios, alpha)
    adp = delivered_quality(adapted, alpha)
    return {
        "alpha": alpha,
        "base": base.as_dict(),
        "adapted": adp.as_dict(),
        "coverage_gain_pp": round((adp.served_fraction - base.served_fraction) * 100, 2),
        "quality_gain_pp": round((adp.mean_ratio_served - base.mean_ratio_served) * 100, 2),
        "gap_to_frontier_closed": round(
            (adp.mean_ratio_served - base.mean_ratio_served)
            / max(1.0 - base.mean_ratio_served, 1e-9), 4),
    }


def alpha_sweep(base_ratios: Dict[str, float],
                uplift: D.Param = UPLIFT_POINTS) -> Dict[str, Dict[str, float]]:
    """f_s against the acceptance threshold, with and without adaptation.

    The interesting structure is not the headline f_s but *where the steps are*.
    At a demanding alpha the base model fails classes an adapter recovers, and
    that is precisely where adaptation earns its keep.
    """
    out: Dict[str, Dict[str, float]] = {}
    for a in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        r = uplifted_f_s(base_ratios, alpha=a, uplift=uplift)
        out[f"{a:.2f}"] = {
            "base": round(r.base_f_s, 4),
            "adapted_lo": round(r.adapted_f_s["lo"], 4),
            "adapted_mid": round(r.adapted_f_s["mid"], 4),
            "adapted_hi": round(r.adapted_f_s["hi"], 4),
        }
    return out


# --------------------------------------------------------------------------- #
#  3. The integrated result - what uplift does to system savings
# --------------------------------------------------------------------------- #

def system_with_adaptation(base_ratios: Dict[str, float],
                           cloud_energy_param: D.Param = D.E_GPT4O,
                           ci: Optional[float] = None) -> Dict[str, object]:
    """Per-query system savings at base f_s and at adapted f_s.

    The adaptation footprint itself is NOT folded in here - it is a one-time
    cost with its own break-even, reported separately by ``breakeven``. Mixing
    a one-time cost into a per-query rate is the kind of arithmetic that makes
    a result unpublishable, so we keep them apart and report both.
    """
    ci = D.CI_GLOBAL.value if ci is None else ci
    cloud_q = M.cloud_query_central(cloud_energy_param, hyperscale=True, ci=ci)
    edge_q = M.edge_slm_query_central(ci=ci)
    up = uplifted_f_s(base_ratios)

    def at(f_s: float) -> Dict[str, float]:
        r = M.system_net_savings(f_s, cloud_q, edge_q, D.REBOUND_FACTOR.value)
        return {"f_s": round(f_s, 4),
                "energy": round(r.savings_frac_energy, 4),
                "water": round(r.savings_frac_water, 4),
                "carbon": round(r.savings_frac_carbon, 4)}

    return {
        "baseline_model": cloud_energy_param.note,
        "base": at(up.base_f_s),
        "adapted_lo": at(up.adapted_f_s["lo"]),
        "adapted_mid": at(up.adapted_f_s["mid"]),
        "adapted_hi": at(up.adapted_f_s["hi"]),
        "cloud_queries_avoided_per_100": round(
            (up.adapted_f_s["mid"] - up.base_f_s) * 100, 1),
    }


# --------------------------------------------------------------------------- #
#  4. Fleet arithmetic - the strongest argument against us, computed
# --------------------------------------------------------------------------- #

@dataclass
class FleetAdaptation:
    users: float
    per_user_training_kwh: float
    total_training_twh: float
    centralized_equivalent_twh: float
    annual_inference_saved_twh: float
    net_saved_twh: float
    payback_years: float
    verdict: str

    def as_dict(self) -> Dict[str, object]:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def fleet_adaptation(users: float, model: V.ModelSpec, device: V.Device,
                     tokens_per_user: float, *,
                     retrain_per_year: float = 2.0,
                     queries_per_user_per_day: float = 50.0,
                     saving_wh_per_query: Optional[float] = None,
                     centralized_run_kwh: float = 1.0e6) -> FleetAdaptation:
    """One adapter per person versus one central fine-tune for everyone.

    This is the honest steelman of the centralized position: a hyperscaler
    fine-tunes once, at enormous but *amortised* cost, and serves everyone from
    it. Decentralized adaptation repeats the training for every user. If N is
    large enough, repetition should lose.

    ``centralized_run_kwh`` defaults to 1 GWh - a deliberately generous
    stand-in for a large production fine-tune plus its serving-side share, in
    the absence of a published figure for any specific 2026 model.
    """
    per_user = P.training_cost(model, device, tokens_per_user).energy_kwh
    annual_per_user = per_user * retrain_per_year
    total_training_twh = users * annual_per_user / 1e9

    if saving_wh_per_query is None:
        cloud_q = M.cloud_query_central(D.E_GPT4O, hyperscale=True)
        edge_q = M.edge_slm_query_central()
        saving_wh_per_query = max(cloud_q.energy_wh - edge_q.energy_wh, 0.0)

    queries = users * queries_per_user_per_day * 365.0
    annual_inference_saved_twh = queries * saving_wh_per_query / 1e9 / 1e3

    net = annual_inference_saved_twh - total_training_twh
    payback = (total_training_twh / annual_inference_saved_twh
               if annual_inference_saved_twh > 0 else float("inf"))

    if net > 0 and payback < 0.25:
        verdict = ("decentralized adaptation is net-positive; the training "
                   "overhead is repaid in under a quarter of a year")
    elif net > 0:
        verdict = "decentralized adaptation is net-positive but not trivially so"
    else:
        verdict = ("REFUTED at this scale: repeated per-user training exceeds "
                   "the inference savings it enables")

    return FleetAdaptation(
        users=users,
        per_user_training_kwh=per_user,
        total_training_twh=total_training_twh,
        centralized_equivalent_twh=centralized_run_kwh / 1e9,
        annual_inference_saved_twh=annual_inference_saved_twh,
        net_saved_twh=net,
        payback_years=payback,
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
#  5. Adapter transport - the product consequence of adapter size
# --------------------------------------------------------------------------- #

def adapter_transport(model: V.ModelSpec, rank: int,
                      link_mbit_s: float = 20.0) -> Dict[str, Dict[str, float]]:
    """How long each method's adapter takes to cross a WebRTC datachannel.

    AutoYou's transport already moves ~1 GiB payloads in chunked, ACKed frames,
    so any of these is technically deliverable. The question is whether an
    adapter is a *file transfer* the user waits for or a *message* they never
    notice, because that decides whether personal adapters can sync between a
    person's own devices as a background nicety or must be an explicit chore.
    """
    out: Dict[str, Dict[str, float]] = {}
    for key in ("lora", "dora", "lora_xs", "vera"):
        n = P.adapter_params(model, rank, key)
        mb = P.adapter_file_mb(n)
        seconds = mb * 8.0 / link_mbit_s
        out[key] = {
            "params": n,
            "megabytes": round(mb, 2),
            "seconds_at_link": round(seconds, 1),
            "feels_like": ("a message" if seconds < 3 else
                           "a download" if seconds < 120 else "a chore"),
        }
    return out


if __name__ == "__main__":
    import hypotheses as H

    print("=== amortisation ===")
    for label, mdl, dev, tok in (
        ("personal persona adapter (8B, APU)", V.MINISTRAL_8B, V.UNIFIED_APU, 2.0e6),
        ("support domain adapter (7B, APU)", V.QWEN25_VL_7B, V.UNIFIED_APU, 7.2e6),
        ("capability adapter (27B, APU)", V.QWEN3_5_27B, V.UNIFIED_APU, 14.68e6),
        ("personal persona adapter (8B, dGPU)", V.MINISTRAL_8B, V.DISCRETE_GPU, 2.0e6),
    ):
        b = breakeven(mdl, dev, tok)
        print(f"  {label}")
        print(f"      {b.training_hours:>6.1f} h  {b.training_kwh:>6.2f} kWh  "
              f"-> repaid after {b.queries_to_repay_energy:>8,.0f} queries "
              f"({b.days_to_repay_energy:.0f} days at 50/day)")

    print("\n=== capability uplift ===")
    up = uplifted_f_s(H.CAP_RATIOS)
    print(f"  base f_s        {up.base_f_s:.0%}")
    for k in ("lo", "mid", "hi"):
        print(f"  adapted f_s {k:<3} {up.adapted_f_s[k]:.0%}")
    print(f"  newly sufficient: "
          f"{[k for k, v in up.newly_sufficient.items() if v] or 'none'}")

    print("\n=== delivered quality at the default alpha ===")
    q = quality_uplift(H.CAP_RATIOS)
    print(f"  alpha                  {q['alpha']:.2f}")
    print(f"  coverage   base {q['base']['served_fraction']:.0%} -> "
          f"adapted {q['adapted']['served_fraction']:.0%}  "
          f"({q['coverage_gain_pp']:+.1f} pp)")
    print(f"  quality    base {q['base']['mean_ratio_served']:.1%} -> "
          f"adapted {q['adapted']['mean_ratio_served']:.1%}  "
          f"({q['quality_gain_pp']:+.1f} pp)")
    print(f"  share of the remaining gap to frontier closed: "
          f"{q['gap_to_frontier_closed']:.0%}")

    print("\n=== alpha sweep (f_s base -> adapted mid) ===")
    for a, row in alpha_sweep(H.CAP_RATIOS).items():
        arrow = "  <-- adaptation buys coverage here" if row["adapted_mid"] > row["base"] else ""
        print(f"  alpha={a}  {row['base']:.0%} -> {row['adapted_mid']:.0%}{arrow}")

    print("\n=== system savings vs GPT-4o long ===")
    s = system_with_adaptation(H.CAP_RATIOS)
    for k in ("base", "adapted_lo", "adapted_mid", "adapted_hi"):
        r = s[k]
        print(f"  {k:<12} f_s={r['f_s']:.0%}  energy {r['energy']:.0%}  "
              f"water {r['water']:.0%}  carbon {r['carbon']:.0%}")

    print("\n=== fleet ===")
    for users in (1e5, 1e6, 1e7, 1e8):
        f = fleet_adaptation(users, V.MINISTRAL_8B, V.UNIFIED_APU, 2.0e6)
        print(f"  {users:>12,.0f} users: train {f.total_training_twh:.4f} TWh/yr, "
              f"save {f.annual_inference_saved_twh:.3f} TWh/yr, "
              f"payback {f.payback_years*365:.0f} days")

    print("\n=== adapter transport (8B, r=32, 20 Mbit/s) ===")
    for k, v in adapter_transport(V.MINISTRAL_8B, 32).items():
        print(f"  {k:<9} {v['megabytes']:>8.2f} MB  "
              f"{v['seconds_at_link']:>7.1f} s  {v['feels_like']}")
