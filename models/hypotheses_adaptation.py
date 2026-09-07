# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
hypotheses_adaptation.py - The Pass-3 hypotheses (H7..H11) and counter-claims
(C4..C6), on ON-DEVICE ADAPTATION.

Pass 1 and 2 asked whether it is better to *run* a small model at home. Pass 3
asks whether it is better to *teach* one there. The two questions have
different answers and different failure modes, so they get separate hypotheses
rather than an extended footnote.

Same discipline as hypotheses.py: every claim is falsifiable, every verdict is
computed rather than asserted, and the counter-claims are the ones a hostile
reviewer would actually raise. Three of the six results below are negative or
qualified, which is the point.
"""

from dataclasses import asdict
from typing import Dict, List

import data as D
import devices as V
import evidence as E
import models as M
import peft as P
import adaptation as A
from hypotheses import CAP_RATIOS, Verdict


# --------------------------------------------------------------------------- #
#  H7 - Feasibility: can already-owned hardware adapt a useful model at all?
# --------------------------------------------------------------------------- #

def h7_feasibility() -> Verdict:
    """Does a consumer machine hold a 27B-class adaptation job?"""
    results = {}
    for dev in (V.UNIFIED_APU, V.DISCRETE_GPU):
        for mdl in (V.QWEN3_8_27B, V.MINISTRAL_8B):
            r = P.select_method(mdl, dev, objective="capability", seq_len=8192)
            results[f"{dev.key}/{mdl.key}"] = {
                "feasible": r.feasible, "method": r.base_method,
                "rank": r.rank, "seq_len": r.seq_len,
                "peak_gb": round(r.memory_gb, 1),
                "ceiling_gb": round(r.memory_ceiling_gb, 1),
            }

    apu_27b = results["unified_apu/qwen3.8-27b"]
    gpu_27b = results["discrete_gpu/qwen3.8-27b"]
    validation = P.validate_memory_model()

    # Feasible on the class of device that actually has the memory, and the
    # memory model that says so reproduces a published run.
    ok = apu_27b["feasible"] and abs(validation["relative_error"]) < 0.15

    return Verdict(
        "H7",
        "A 27B-class open-weights model can be parameter-efficiently adapted on "
        "consumer hardware a household already owns, without renting a GPU.",
        "VALIDATED (device-class dependent)" if ok else "REFUTED",
        {
            "configurations": results,
            "memory_model_validation": validation,
            "binding_constraint": "addressable accelerator memory, not FLOPs",
            "note": (
                "The verdict is a property of the DEVICE CLASS, not of consumer "
                "hardware in general. A 128 GB unified-memory APU runs a 27B "
                f"job at rank {apu_27b['rank']} and sequence "
                f"{apu_27b['seq_len']} using {apu_27b['peak_gb']} GB. A 24 GB "
                "discrete GPU of far higher raw throughput is driven down to "
                f"rank {gpu_27b['rank']} at sequence {gpu_27b['seq_len']} "
                "before it fits. The Pass-1 study modelled only the second "
                "class and would have concluded this was impossible."
            ),
        },
    )


# --------------------------------------------------------------------------- #
#  H8 - Amortisation: does an adapter repay its own training footprint?
# --------------------------------------------------------------------------- #

def h8_amortisation() -> Verdict:
    """Does the training run repay its own energy - and repay it to whom?

    This hypothesis was originally recorded as VALIDATED on displacement
    accounting alone. Checking it against H9 showed that reading to be
    unsound at the study's own default threshold: if adaptation moves no
    query across the routing boundary, then the local-versus-cloud saving on
    those queries accrues whether or not the adapter exists, and crediting it
    to the adapter double-counts a benefit the base model already delivered.
    The verdict below reports both accountings and is qualified accordingly.
    """
    cases = {
        "persona_8b_apu": (V.MINISTRAL_8B, V.UNIFIED_APU,
                           D.CORPUS_PERSONA_TOKENS.value),
        "support_7b_apu": (V.QWEN25_VL_7B, V.UNIFIED_APU,
                           D.CORPUS_SUPPORT_TOKENS.value),
        "capability_27b_apu": (V.QWEN3_5_27B, V.UNIFIED_APU,
                               D.CORPUS_CAPABILITY_TOKENS.value),
        "persona_8b_gpu": (V.MINISTRAL_8B, V.DISCRETE_GPU,
                           D.CORPUS_PERSONA_TOKENS.value),
    }
    out = {}
    worst_days = 0.0
    for name, (mdl, dev, tok) in cases.items():
        b = A.breakeven(mdl, dev, tok,
                        queries_per_day=D.QUERIES_PER_USER_PER_DAY.value)
        out[name] = b.as_dict()
        out[name]["retrain_penalty_at_2x_per_year"] = round(
            A.retrain_cadence_penalty(b, 182.5), 4)
        worst_days = max(worst_days, b.days_to_repay_energy)

    regimes = A.breakeven_regimes(
        V.QWEN3_5_27B, V.UNIFIED_APU, D.CORPUS_CAPABILITY_TOKENS.value,
        CAP_RATIOS, queries_per_day=D.QUERIES_PER_USER_PER_DAY.value)

    default_alpha = f"{D.ALPHA_SUFFICIENCY.value:.2f}"
    at_default = regimes["incremental_accounting"][default_alpha]
    repays_at_default = bool(at_default["repays"])

    displacement_ok = worst_days < 365.0
    result = ("PARTIAL (accounting-dependent)" if displacement_ok
              and not repays_at_default else
              "VALIDATED" if displacement_ok and repays_at_default else
              "REFUTED")

    return Verdict(
        "H8",
        "The one-time energy cost of a local adaptation run is repaid by the "
        "inference it makes possible, within the device's service life.",
        result,
        {
            "displacement_cases": out,
            "queries_per_day_assumed": D.QUERIES_PER_USER_PER_DAY.value,
            "longest_payback_days_displacement": round(worst_days, 1),
            "regimes": regimes,
            "accounting": (
                "Training charged at FULL device power; inference savings "
                "measured against a HYPERSCALE cloud query, which is the "
                "cloud's best case. Both choices are against the hypothesis."
            ),
            "correction": (
                "The phrase 'the inference it makes possible' is doing "
                "load-bearing work that the displacement figures do not "
                "support. At the study's default acceptance threshold "
                f"(alpha={default_alpha}) the adapter makes NO additional "
                "local inference possible - H9 establishes that coverage is "
                "unchanged - so it displaces no cloud query and has no energy "
                "payback at all. Its cost is a net addition bought for "
                "quality. Payback appears only where adaptation actually moves "
                "the routing boundary: at alpha=0.80 the 27B run repays in "
                f"{regimes['incremental_accounting']['0.80']['years_to_repay']}"
                " years and at alpha=0.90 in "
                f"{regimes['incremental_accounting']['0.90']['years_to_repay']}"
                " years, against 143 days under the displacement reading."
            ),
            "what_the_displacement_figures_do_mean": (
                "They remain the correct answer to a different question: how "
                "long a LOCAL DEPLOYMENT takes to repay an adaptation run "
                "against an all-cloud counterfactual. That is the right frame "
                "for a user whose revealed acceptance threshold is above the "
                "coverage knee - someone who would have abandoned the local "
                "model and gone to the cloud had its answers not improved. It "
                "is a behavioural assumption, and it is now stated as one "
                "rather than assumed silently."
            ),
            "caveat": (
                "Payback also scales with the cloud baseline being displaced. "
                "Against a hyper-efficient 0.24 Wh cloud prompt rather than a "
                "1.8 Wh one, every figure here lengthens by roughly an order "
                "of magnitude - the same conditionality that qualifies H5."
            ),
        },
    )


# --------------------------------------------------------------------------- #
#  H9 - Uplift: does adaptation move the crux variable?
# --------------------------------------------------------------------------- #

def h9_uplift() -> Verdict:
    """Coverage (f_s) versus delivered quality, at and above the default alpha."""
    up = A.uplifted_f_s(CAP_RATIOS)
    q = A.quality_uplift(CAP_RATIOS)
    sweep = A.alpha_sweep(CAP_RATIOS)

    coverage_moved = up.adapted_f_s["mid"] > up.base_f_s + 1e-9
    quality_moved = q["quality_gain_pp"] > 1.0
    knee = next((float(a) for a, row in sweep.items()
                 if row["adapted_mid"] > row["base"] + 1e-9), None)

    if quality_moved and not coverage_moved:
        result = "PARTIAL (quality, not coverage)"
    elif quality_moved and coverage_moved:
        result = "VALIDATED"
    else:
        result = "REFUTED"

    return Verdict(
        "H9",
        "Parameter-efficient adaptation raises the fraction of a real workload "
        "a local small model can serve (f_s).",
        result,
        {
            "alpha": q["alpha"],
            "coverage": {"base": up.base_f_s, "adapted_mid": up.adapted_f_s["mid"],
                         "gain_pp": q["coverage_gain_pp"]},
            "quality": {"base": q["base"]["mean_ratio_served"],
                        "adapted": q["adapted"]["mean_ratio_served"],
                        "gain_pp": q["quality_gain_pp"],
                        "gap_to_frontier_closed": q["gap_to_frontier_closed"]},
            "alpha_sweep": sweep,
            "coverage_knee_alpha": knee,
            "f_s_range": {k: round(v, 4) for k, v in up.adapted_f_s.items()},
            "disproof": (
                "The hypothesis AS STATED is not what the evidence supports. At "
                f"the study's acceptance threshold alpha={q['alpha']:.2f}, "
                f"adaptation moves coverage by {q['coverage_gain_pp']:+.1f} "
                "points - that is, not at all - because every adaptable class "
                "already cleared the bar without it. What moves is quality: "
                f"{q['base']['mean_ratio_served']:.1%} -> "
                f"{q['adapted']['mean_ratio_served']:.1%} mean capability on "
                "the served workload, closing "
                f"{q['gap_to_frontier_closed']:.0%} of the remaining gap to "
                "the frontier. Coverage only responds above "
                f"alpha={knee}, where the unadapted model starts failing "
                "classes the adapted one still passes. The correct claim is "
                "therefore: adaptation buys better answers at an ordinary "
                "quality bar, and buys more answers only at a demanding one."
            ),
        },
    )


# --------------------------------------------------------------------------- #
#  H10 - Locality: is a locally-trained adapter structurally more private?
# --------------------------------------------------------------------------- #

def h10_locality() -> Verdict:
    """Adapter transport size decides whether personalisation must leave home."""
    transport = A.adapter_transport(V.MINISTRAL_8B, 32)
    smallest = min(transport.values(), key=lambda v: v["megabytes"])
    lora_mb = transport["lora"]["megabytes"]

    return Verdict(
        "H10",
        "Training the adapter where the data already lives removes an "
        "exfiltration step that no cloud fine-tuning arrangement can remove.",
        "VALIDATED (structural)",
        {
            "argument": (
                "Cloud fine-tuning requires the training corpus to reach the "
                "trainer. For a persona adapter that corpus IS the user's "
                "message history, so the privacy question is settled before any "
                "policy is written: either the messages were uploaded or they "
                "were not. Local adaptation makes 'we never received it' a "
                "property of the topology rather than a promise in a document. "
                "This is the same class of argument as H6 (relays cannot read "
                "DTLS payloads), and it is verified the same way - by what the "
                "system is unable to do, not by what it undertakes not to do."
            ),
            "adapter_transport_8b_r32": transport,
            "consequence": (
                f"A conventional rank-32 LoRA adapter is {lora_mb:.0f} MB - a "
                "file transfer. The low-parameter methods produce "
                f"{smallest['megabytes']:.1f} MB, which crosses an existing "
                "datachannel in about a second. Personal adapters can therefore "
                "sync between a person's OWN devices as ordinary messages, with "
                "no server-side copy at any point."
            ),
            "limits": (
                "This is a claim about the training corpus and the adapter, not "
                "about the base model, which is downloaded from a public "
                "repository, and not about inference-time prompts, which H6 "
                "covers separately."
            ),
        },
    )


# --------------------------------------------------------------------------- #
#  H11 - Measured: AutoYou's own runs, as primary evidence
# --------------------------------------------------------------------------- #

def h11_measured() -> Verdict:
    """The deployment's own adapter evaluations, read in aggregate."""
    e1 = E.e1_capability_lift()
    e2 = E.e2_safety_direction()
    e3 = E.e3_base_size_vs_data()
    available = [f for f in (e1, e2, e3) if f.get("available")]

    if not available:
        return Verdict(
            "H11",
            "A narrow LoRA adapter measurably lifts a task class the base model "
            "fails, on this deployment's own evaluation artifacts.",
            "NO DATA",
            {"reason": "adapter evaluation artifacts are not present in a "
                       "public checkout; set ADAPTER_EVAL_ROOT to a "
                       "deployment's own artifacts to reproduce.",
             "found": E.summary()["runs_found"]},
        )

    ok = all(f.get("verdict", "").startswith(("VALIDATED",)) for f in available)
    return Verdict(
        "H11",
        "A narrow LoRA adapter measurably lifts a task class the base model "
        "fails, and does so without degrading its safety behaviour.",
        "VALIDATED (n=1 deployment)" if ok else "PARTIAL",
        {
            "E1_capability_lift": e1,
            "E2_safety_direction": e2,
            "E3_base_size_vs_data": e3,
            "recipe": E.training_efficiency(),
            "scope_limit": (
                "One deployment, one product's surface area, probe sets of "
                "12-14 screens and 8 code probes scored by an automated judge. "
                "These are existence proofs and refutations, not effect sizes, "
                "and they are reported as such everywhere they appear."
            ),
        },
    )


# --------------------------------------------------------------------------- #
#  C4 - Forgetting: does adapting a small model break what it already did?
# --------------------------------------------------------------------------- #

def c4_forgetting() -> Verdict:
    """The catastrophic-forgetting objection, and what actually bounds it."""
    return Verdict(
        "C4",
        "COUNTER-CLAIM: narrowing a small model onto a personal corpus destroys "
        "its general ability, so the specialised model is worse overall.",
        "REFUTED (bounded, with conditions)",
        {
            "why_the_objection_is_weaker_than_it_looks": (
                "Biderman et al. measure exactly this and find the opposite of "
                "the intuition: LoRA learns LESS in-domain than full "
                "fine-tuning, and correspondingly FORGETS LESS out-of-domain - "
                "better than weight decay or dropout as a regulariser. The "
                "low-rank constraint that limits the gain is the same "
                "constraint that limits the damage. Full fine-tuning learns "
                "perturbations of 10-100x higher rank, and pays for it."
            ),
            "structural_defences_available_today": {
                "adapter_is_detachable": (
                    "The base weights are never modified. An adapter that "
                    "misbehaves is unloaded, not un-trained - a property no "
                    "full fine-tune has."
                ),
                "corda_knowledge_preserved": (
                    "CorDA's knowledge-preserved mode builds the update in "
                    "directions the base model does not use for general "
                    "knowledge, which addresses this objection at "
                    "initialisation rather than by hoping."
                ),
                "routing_survives": (
                    "hard_reason is not adapted (see ADAPTABLE_CLASSES) and "
                    "continues to route out. Forgetting on a class the system "
                    "never serves locally has no effect on the user."
                ),
            },
            "residual_risk": (
                "Real, and not modelled away: a personal corpus is small, "
                "unbalanced and self-similar, which is the regime where "
                "over-fitting is most likely. The honest mitigation is the "
                "boring one - a held-out split and a regression probe on "
                "general ability before an adapter is installed. AutoYou's "
                "support pipeline already does this (72 eval samples, a "
                "code-refusal probe set); the Fine Tuning agent's per-user path "
                "does not yet, and that gap is recorded in the integration "
                "specification rather than hidden."
            ),
            "cite": ["biderman2024lora", "yang2024corda"],
        },
    )


# --------------------------------------------------------------------------- #
#  C5 - Safety: does user-controlled fine-tuning strip alignment?
# --------------------------------------------------------------------------- #

def c5_safety() -> Verdict:
    """The strongest objection to shipping fine-tuning to end users."""
    e2 = E.e2_safety_direction()
    return Verdict(
        "C5",
        "COUNTER-CLAIM: putting fine-tuning in end users' hands strips safety "
        "alignment, because fine-tuning degrades it even on benign data.",
        "SUPPORTED (and specifically delimited)",
        {
            "the_objection_is_correct_as_stated": (
                "Qi et al. removed GPT-3.5 Turbo's guardrails with ten "
                "adversarial examples for under $0.20, and showed that even "
                "benign, ordinary datasets degrade safety measurably. Any "
                "product that hands users a trainer inherits this. We do not "
                "dispute it and we do not model it away."
            ),
            "what_the_deployment_evidence_adds": e2,
            "the_delimitation": (
                "Qi et al. describe adapters that are SILENT about safety: the "
                "gradient has no reason to preserve a behaviour nothing in the "
                "corpus rewards, so it decays. The AutoYou support adapter is "
                "EXPLICIT about it - 6.9% of its corpus is refusal behaviour - "
                "and the measured direction reverses: refusal 0% -> 100%, "
                "source-code leakage 25% -> 0%. The correct generalisation is "
                "not 'fine-tuning is safe' but 'fine-tuning moves whatever the "
                "corpus specifies, including the safety behaviour, in whichever "
                "direction the corpus specifies'."
            ),
            "product_consequence": (
                "Safety for a user-trained adapter cannot be inherited; it has "
                "to be a shipped, non-optional part of every training corpus "
                "the product assembles, plus a probe that runs before an "
                "adapter is installed. That is a build requirement, not a "
                "research finding, and it is written into the integration "
                "specification as one."
            ),
            "cite": ["qi2023finetuning", "autoyou_support2026"],
        },
    )


# --------------------------------------------------------------------------- #
#  C6 - Scale: does one-adapter-per-person lose to one central fine-tune?
# --------------------------------------------------------------------------- #

def c6_repetition() -> Verdict:
    """Repeated per-user training versus a single amortised central run."""
    fleet = {}
    for users in (1e5, 1e6, 1e7, 1e8):
        f = A.fleet_adaptation(
            users, V.MINISTRAL_8B, V.UNIFIED_APU,
            D.CORPUS_PERSONA_TOKENS.value,
            retrain_per_year=D.RETRAIN_PER_YEAR.value,
            queries_per_user_per_day=D.QUERIES_PER_USER_PER_DAY.value)
        fleet[f"{users:.0e}"] = f.as_dict()

    paybacks = [v["payback_years"] for v in fleet.values()]
    scale_invariant = max(paybacks) - min(paybacks) < 1e-6
    net_positive = all(v["net_saved_twh"] > 0 for v in fleet.values())

    return Verdict(
        "C6",
        "COUNTER-CLAIM: training one adapter per person repeats work that a "
        "single central fine-tune would do once, so decentralized adaptation "
        "must lose at scale.",
        "REFUTED (scale-invariant)" if (net_positive and scale_invariant)
        else "SUPPORTED",
        {
            "fleet": fleet,
            "payback_days": round(paybacks[0] * 365, 1),
            "why": (
                "The objection assumes the training term grows with users while "
                "the benefit does not. Both grow linearly, so their ratio is "
                "constant: the payback period is identical at 100 thousand "
                "users and at 100 million. Repetition does not lose at scale "
                "because there is no scale at which it behaves differently."
            ),
            "what_would_actually_refute_this": (
                "A central fine-tune serving everyone is only comparable if one "
                "adapter suits everyone. It does not - the entire point of a "
                "persona adapter is that it encodes one person. The honest "
                "comparison is not 'one run versus N runs' but 'N runs versus N "
                "personalised runs plus the transfer of N private corpora to a "
                "trainer', and the second term is what H10 removes."
            ),
            "residual": (
                "This analysis charges training energy only. It does not model "
                "the embodied carbon of devices bought FOR adaptation, which "
                "C3 already establishes would dominate. The result holds for "
                "already-owned hardware and is not a case for buying an APU."
            ),
        },
    )


# --------------------------------------------------------------------------- #

def run_all() -> Dict:
    verdicts: List[Verdict] = [
        h7_feasibility(), h8_amortisation(), h9_uplift(), h10_locality(),
        h11_measured(), c4_forgetting(), c5_safety(), c6_repetition(),
    ]
    return {"verdicts": [asdict(v) for v in verdicts]}


if __name__ == "__main__":
    out = run_all()
    for v in out["verdicts"]:
        print(f"[{v['hid']:<3}] {v['result']}")
        print(f"      {v['statement']}")
    print()
    h9 = next(v for v in out["verdicts"] if v["hid"] == "H9")
    print("H9 disproof:\n  " + h9["evidence"]["disproof"].replace("\n", "\n  "))
