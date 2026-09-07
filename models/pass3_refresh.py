# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
pass3_refresh.py - September-2026 refresh: the ADAPTATION economics.

Pass 2 (July 2026) re-priced INFERENCE against the then-current frontier and
found the headline ~150x TCO ratio intact. Pass 3 asks the question Pass 2 did
not: what does it cost to *personalise* a model, and does the answer differ by
enough to matter?

It does, in a way the inference comparison does not capture, because cloud
personalisation charges twice:

  1. Once for the training tokens, at a published per-token rate.
  2. Then FOREVER, because a tuned endpoint is billed at a multiple of the base
     model's rate - 1.5x on the two largest platforms as of this refresh.

A local adapter charges once, in electricity, and then makes every subsequent
query cost exactly what it cost before. The second term is what turns a modest
one-time difference into a compounding one, and it is invisible if you only
compare training prices.

As in Pass 2, no peer-reviewed per-query ENERGY measurement covers the 2026
frontier, so the energy/water/carbon corpus stays pinned to the measured 2025
sources. Only pricing, architecture and the deployment's own adapter
evaluations are refreshed here.

Run:  ../.venv/bin/python pass3_refresh.py
"""

from typing import Dict, Optional

import data as D
import devices as V
import evidence as E
import models as M
import peft as P
import adaptation as A


REFRESH_DATE = "2026-09-05"


# --------------------------------------------------------------------------- #
#  Hosted fine-tuning price list, accessed 2026-09-05
# --------------------------------------------------------------------------- #

# key: (vendor, $/Mtok training, tuned-inference multiplier vs base, note)
HOSTED_TUNING_2026 = {
    "together-lora":  ("Together AI", 0.48, 1.0,
                       "cheapest published LoRA training rate"),
    "gpt-4.1-nano":   ("OpenAI", 1.50, 1.5,
                       "smallest OpenAI tunable tier"),
    "gemini-flash":   ("Google Vertex", 3.00, 1.5,
                       "billed per training token; from Gemini 3 onward tuned "
                       "endpoints predict at 1.5x base"),
    "gpt-4o":         ("OpenAI", 25.00, 1.5,
                       "mid-tier OpenAI tunable model"),
}

# o4-mini is priced per training HOUR rather than per token, so it is carried
# separately rather than forced into the same units.
HOSTED_TUNING_HOURLY = {
    "o4-mini": ("OpenAI", 100.00, 1.5, "$/hour of training"),
}


def hosted_training_cost(tokens: float) -> Dict[str, Dict[str, object]]:
    """What the same corpus would cost to train on each hosted platform."""
    out: Dict[str, Dict[str, object]] = {}
    for key, (vendor, per_mtok, mult, note) in HOSTED_TUNING_2026.items():
        out[key] = {
            "vendor": vendor,
            "training_usd": round(tokens / 1e6 * per_mtok, 2),
            "price_per_mtok": per_mtok,
            "tuned_inference_multiplier": mult,
            "note": note,
        }
    return out


def local_training_cost(model: V.ModelSpec, device: V.Device,
                        tokens: float) -> Dict[str, object]:
    c = P.training_cost(model, device, tokens,
                        price_kwh=D.PRICE_ELECTRICITY.value)
    return {
        "device": device.name,
        "hours": round(c.hours, 1),
        "kwh": round(c.energy_kwh, 2),
        "electricity_usd": round(c.electricity_usd, 2),
        "tuned_inference_multiplier": 1.0,
        "note": "electricity only; the device is already owned (C3 applies)",
    }


def tuning_tco(model: V.ModelSpec, device: V.Device, tokens: float, *,
               queries_per_day: float = None,
               horizon_days: float = 365.0) -> Dict[str, object]:
    """One-year total cost of a personalised model, local versus hosted.

    The comparison that matters is not the training bill. It is the training
    bill PLUS a year of inference at whatever rate the tuned model is billed
    at, against the training electricity PLUS a year of inference at the
    device's own electricity cost.
    """
    queries_per_day = (D.QUERIES_PER_USER_PER_DAY.value
                       if queries_per_day is None else queries_per_day)
    queries = queries_per_day * horizon_days

    local_train = local_training_cost(model, device, tokens)
    edge_q = M.edge_slm_query_central(marginal=True)
    local_infer = M.edge_cost_per_query(edge_q.energy_wh,
                                        D.PRICE_ELECTRICITY.value) * queries
    local_total = local_train["electricity_usd"] + local_infer

    rows: Dict[str, Dict[str, object]] = {
        "local": {
            "training_usd": local_train["electricity_usd"],
            "inference_usd_year_1": round(local_infer, 2),
            "total_usd_year_1": round(local_total, 2),
            "detail": local_train,
        }
    }

    base_infer = M.cloud_cost_per_query(
        D.TYPICAL_OUT_TOKENS.value, D.TYPICAL_IN_TOKENS.value,
        D.API_PRICE_OUT_FRONTIER_2026.value, D.API_PRICE_IN_FRONTIER_2026.value)

    for key, row in hosted_training_cost(tokens).items():
        infer = base_infer * row["tuned_inference_multiplier"] * queries
        total = row["training_usd"] + infer
        rows[key] = {
            "vendor": row["vendor"],
            "training_usd": row["training_usd"],
            "inference_usd_year_1": round(infer, 2),
            "total_usd_year_1": round(total, 2),
            "ratio_vs_local": (round(total / local_total, 1)
                               if local_total > 0 else None),
            "tuned_inference_multiplier": row["tuned_inference_multiplier"],
        }

    cheapest_hosted = min(
        (k for k in rows if k != "local"),
        key=lambda k: rows[k]["total_usd_year_1"])

    # Counted, not asserted. An earlier draft hardcoded "two of the four" and
    # was wrong by one; a figure that can drift from its own table should not
    # be typed out by hand.
    surcharged = sum(1 for _, _, mult, _ in HOSTED_TUNING_2026.values()
                     if mult > 1.0)
    total_platforms = len(HOSTED_TUNING_2026)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    return {
        "model": model.key,
        "corpus_tokens": tokens,
        "queries_per_day": queries_per_day,
        "horizon_days": horizon_days,
        "rows": rows,
        "cheapest_hosted": cheapest_hosted,
        "ratio_vs_cheapest_hosted": rows[cheapest_hosted]["ratio_vs_local"],
        "platforms_priced": total_platforms,
        "platforms_with_surcharge": surcharged,
        "finding": (
            "The training bill is the smaller half of the story. Even the "
            "cheapest published hosted LoRA rate charges "
            f"${rows[cheapest_hosted]['training_usd']:.2f} to train where "
            f"electricity charges ${rows['local']['training_usd']:.2f}, but the "
            "gap that compounds is the tuned-endpoint surcharge: "
            f"{words.get(surcharged, surcharged)} of the "
            f"{words.get(total_platforms, total_platforms)} platforms priced "
            "bill a personalised model at 1.5x the base rate for every "
            "subsequent query, permanently. A local adapter leaves the "
            "per-query cost exactly where it was."
        ),
    }


# --------------------------------------------------------------------------- #
#  Open-weights tier refresh
# --------------------------------------------------------------------------- #

def open_weights_refresh() -> Dict[str, object]:
    """What a household can now hold, and what changed since Pass 2."""
    rows = {}
    for key, (vendor, release, total_b, active_b, licence, note) in \
            D.OPEN_WEIGHTS_2026.items():
        spec = V.MODELS.get(key)
        entry = {
            "vendor": vendor, "release": release, "params_b": total_b,
            "active_b": active_b, "licence": licence, "note": note,
            "gb_4bit": round(total_b * 1e9 * P.BYTES_PER_PARAM["nf4_dq"] / 1e9, 1),
            "gb_bf16": round(total_b * 2.0, 1),
        }
        if spec is not None:
            entry["fits_apu_bf16_training"] = P.select_method(
                spec, V.UNIFIED_APU, objective="capability").feasible
            entry["fits_dgpu_training"] = P.select_method(
                spec, V.DISCRETE_GPU, objective="capability").feasible
        rows[key] = entry
    return {
        "as_of": REFRESH_DATE,
        "models": rows,
        "qwen38_reported_gains_vs_qwen36": D.QWEN38_GAINS,
        "finding": (
            "The open-weights tier a household can hold moved from 8B to 27B "
            "between the first pass and this one, and it did so on an "
            "unchanged architecture: Qwen3.8-27B reports Terminal-Bench 2.1 "
            "63.4 -> 73.0 and OSWorld-Verified 63.9 -> 84.3 over Qwen3.6-27B "
            "at the same published decoder size. That is the same mechanism "
            "this pass tests at household scale - capability from data and "
            "post-training rather than from parameters - operating at vendor "
            "scale. It is the single most important input change to the study "
            "since Pass 1, because every capability ratio in Section VI was "
            "estimated for an 8B model."
        ),
    }


# --------------------------------------------------------------------------- #
#  Deployment validation
# --------------------------------------------------------------------------- #

def deployment_validation() -> Dict[str, object]:
    """AutoYou's own adapter evaluations, or an honest NO DATA."""
    s = E.summary()
    if s["runs_found"] == 0:
        return {
            "status": "NO DATA",
            "reason": ("adapter evaluation artifacts are not present in a "
                       "public checkout; set ADAPTER_EVAL_ROOT to a "
                       "deployment's own artifacts to reproduce"),
        }
    return {
        "status": "VALIDATED",
        "runs_found": s["runs_found"],
        "runs_expected": s["runs_expected"],
        "dataset": s["dataset"],
        "E1_capability_lift": s["E1_capability_lift"],
        "E2_safety_direction": s["E2_safety_direction"],
        "E3_base_size_vs_data": s["E3_base_size_vs_data"],
        "scope": ("one deployment; aggregate probe scores only; no prompts, "
                  "answers, screenshots or user content"),
    }


# --------------------------------------------------------------------------- #

def run() -> Dict[str, object]:
    return {
        "as_of": REFRESH_DATE,
        "energy_corpus_unchanged": (
            "No peer-reviewed per-query energy measurement covers the 2026 "
            "frontier or the 2026 open-weights tier. Energy, water and carbon "
            "results remain pinned to the measured 2025 corpus (jegham2025, "
            "elsworth2025, caravaca2025)."
        ),
        "open_weights": open_weights_refresh(),
        "memory_model_validation": P.validate_memory_model(),
        "throughput_calibration": P.calibrate_throughput(),
        "tuning_tco": {
            "persona_8b": tuning_tco(V.MINISTRAL_8B, V.UNIFIED_APU,
                                     D.CORPUS_PERSONA_TOKENS.value),
            "capability_27b": tuning_tco(V.QWEN3_5_27B, V.UNIFIED_APU,
                                         D.CORPUS_CAPABILITY_TOKENS.value),
        },
        "recipes": {
            f"{dev.key}/{mdl.key}/{obj}": P.select_method(
                mdl, dev, objective=obj, seq_len=4096).as_dict()
            for dev in (V.UNIFIED_APU, V.DISCRETE_GPU)
            for mdl, obj in ((V.QWEN3_8_27B, "capability"),
                             (V.MINISTRAL_8B, "persona"),
                             (V.QWEN25_VL_7B, "domain"),
                             (V.MINISTRAL_8B, "portable"))
        },
        "deployment_validation": deployment_validation(),
    }


if __name__ == "__main__":
    out = run()
    print(f"=== Pass 3 refresh, {out['as_of']} ===\n")

    ow = out["open_weights"]
    print("open-weights tier a household can hold:")
    for k, v in ow["models"].items():
        fit = ""
        if "fits_apu_bf16_training" in v:
            fit = ("  train: APU "
                   f"{'yes' if v['fits_apu_bf16_training'] else 'no ':<3} / "
                   f"dGPU {'yes' if v['fits_dgpu_training'] else 'no'}")
        print(f"  {k:<20} {v['params_b']:>6.1f}B  "
              f"{v['gb_4bit']:>5.1f} GB @4-bit  {v['licence']:<14}{fit}")

    print("\ntuning TCO, one year:")
    for scale, t in out["tuning_tco"].items():
        print(f"  [{scale}] {t['corpus_tokens']/1e6:.1f}M tokens, "
              f"{t['queries_per_day']:.0f} queries/day")
        for name, row in t["rows"].items():
            ratio = row.get("ratio_vs_local")
            print(f"      {name:<16} train ${row['training_usd']:>8,.2f}  "
                  f"+ inference ${row['inference_usd_year_1']:>9,.2f}  "
                  f"= ${row['total_usd_year_1']:>9,.2f}"
                  + (f"   ({ratio:,.0f}x local)" if ratio else ""))

    dv = out["deployment_validation"]
    print(f"\ndeployment validation: {dv['status']}")
    if dv["status"] == "VALIDATED":
        print(f"  {dv['runs_found']}/{dv['runs_expected']} adapter evaluations, "
              f"{dv['dataset']['train']} training samples")
        for k in ("E1_capability_lift", "E2_safety_direction",
                  "E3_base_size_vs_data"):
            print(f"    {k:<24} {dv[k].get('verdict')}")
