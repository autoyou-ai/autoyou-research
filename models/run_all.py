# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
run_all.py - Reproducibility entry point.

Runs the full eval (hypotheses + Monte-Carlo), regenerates all figures, and
writes results.json - the single machine-readable artifact the LaTeX paper and
the marketing assets both cite. Re-running this regenerates every number and
figure in the paper from data.py.

Run:  ../.venv/bin/python run_all.py
"""

import json
import os
import subprocess
import sys

import data as D
import models as M
import hypotheses as H
import pass2_refresh as P2

# Pass 3 - the on-device adaptation track.
import adaptation as A
import devices as V
import evidence as E
import measured as MEAS
import hypotheses_adaptation as HA
import pass3_refresh as P3
import peft as PF

HERE = os.path.dirname(__file__)


def key_numbers():
    """Headline derived numbers cited in the paper abstract/conclusions."""
    f_s, verdicts = M.sufficient_fraction(D.WORKLOAD_MIX, H.CAP_RATIOS,
                                          D.ALPHA_SUFFICIENCY.value)
    cloud_gpt4o = M.cloud_query_central(D.E_GPT4O, hyperscale=True)
    edge = M.edge_slm_query_central()
    sys_gpt4o = M.system_net_savings(f_s, cloud_gpt4o, edge, D.REBOUND_FACTOR.value)
    return {
        "f_s_sufficient_fraction": f_s,
        "per_class_sufficiency": verdicts,
        "right_sizing_ratio_o3_over_8B": M.right_sizing_ratio(D.E_O3.value, D.E_LLAMA31_8B.value),
        "right_sizing_ratio_gpt4o_over_8B": M.right_sizing_ratio(D.E_GPT4O.value, D.E_LLAMA31_8B.value),
        "system_vs_gpt4o_long": {
            "energy": sys_gpt4o.savings_frac_energy,
            "water": sys_gpt4o.savings_frac_water,
            "carbon": sys_gpt4o.savings_frac_carbon,
        },
        "edge_query_footprint": edge.as_dict(),
        "cloud_gpt4o_footprint": cloud_gpt4o.as_dict(),
        "google_overhead_noncompute_fraction": D.GOOGLE_NONCOMPUTE_FRACTION.value,
        "water_per_100w_prompt_uc_riverside_ml": 519.0,
        "tco": {
            "edge_owned_usd": M.edge_cost_per_query(
                M.edge_slm_query_central(marginal=True).energy_wh,
                D.PRICE_ELECTRICITY.value),
            # The unfavourable case the paper also quotes: a device bought
            # SOLELY for inference, with its full purchase price amortised
            # across its service life. Exported because the paper cites it and
            # anything the paper cites should be checkable against this file.
            "edge_dedicated_amortised_usd": M.edge_cost_per_query(
                M.edge_slm_query_central().energy_wh,
                D.PRICE_ELECTRICITY.value,
                device_cost=D.EDGE_DEVICE_COST.value,
                attrib_fraction=1.0,
                lifetime_queries=D.EDGE_DEVICE_LIFETIME_QUERIES.value),
            "cloud_frontier_usd": M.cloud_cost_per_query(
                D.TYPICAL_OUT_TOKENS.value, D.TYPICAL_IN_TOKENS.value,
                D.API_PRICE_OUT_FRONTIER.value, D.API_PRICE_IN.value),
        },
        "fleet_2030_inference_twh": M.inference_energy_twh(
            D.DC_DEMAND_2030_TWH.value, D.AI_SHARE_OF_DC.value,
            D.INFERENCE_SHARE_OF_AI.value),
        "fleet_savings_25pct_adoption": M.fleet_savings(
            D.DC_DEMAND_2030_TWH.value, D.AI_SHARE_OF_DC.value,
            D.INFERENCE_SHARE_OF_AI.value, 0.25,
            sys_gpt4o.savings_frac_energy, D.CI_GLOBAL.value),
        "agentic_replaceable_nvidia": {
            "range": [D.AGENTIC_REPLACEABLE.lo, D.AGENTIC_REPLACEABLE.hi],
            "our_f_s": f_s},
    }


def adaptation_numbers():
    """Headline derived numbers for the Pass-3 adaptation track."""
    q = A.quality_uplift(H.CAP_RATIOS)
    up = A.uplifted_f_s(H.CAP_RATIOS)
    be = {
        name: A.breakeven(mdl, dev, tok,
                          queries_per_day=D.QUERIES_PER_USER_PER_DAY.value).as_dict()
        for name, (mdl, dev, tok) in {
            "persona_8b_apu": (V.MINISTRAL_8B, V.UNIFIED_APU,
                               D.CORPUS_PERSONA_TOKENS.value),
            "support_7b_apu": (V.QWEN25_VL_7B, V.UNIFIED_APU,
                               D.CORPUS_SUPPORT_TOKENS.value),
            "capability_27b_apu": (V.QWEN3_5_27B, V.UNIFIED_APU,
                                   D.CORPUS_CAPABILITY_TOKENS.value),
        }.items()
    }
    return {
        "alpha": q["alpha"],
        "coverage_gain_pp": q["coverage_gain_pp"],
        "quality_gain_pp": q["quality_gain_pp"],
        "gap_to_frontier_closed": q["gap_to_frontier_closed"],
        "mean_quality_base": q["base"]["mean_ratio_served"],
        "mean_quality_adapted": q["adapted"]["mean_ratio_served"],
        "f_s_base": up.base_f_s,
        "f_s_adapted_range": up.adapted_f_s,
        "coverage_knee_alpha": D.ALPHA_COVERAGE_KNEE.value,
        "alpha_sweep": A.alpha_sweep(H.CAP_RATIOS),
        "breakeven": be,
        "memory_model_validation": PF.validate_memory_model(),
        "throughput_calibration": PF.calibrate_throughput(),
        "adapter_transport_8b_r32": A.adapter_transport(V.MINISTRAL_8B, 32),
        "system_with_adaptation": A.system_with_adaptation(H.CAP_RATIOS),
        "measured_evidence": {
            "E1_capability_lift": E.e1_capability_lift(),
            "E2_safety_direction": E.e2_safety_direction(),
            "E3_base_size_vs_data": E.e3_base_size_vs_data(),
        },
    }


def _sigfig(x: float, n: int = 2) -> str:
    """Format to n significant figures without scientific notation.

    Used for every measured-energy macro. The telemetry behind those numbers
    resolves 4-5 distinct power readings per short request, so a third digit
    is noise dressed as precision.
    """
    if x == 0:
        return "0"
    import math
    d = n - int(math.floor(math.log10(abs(x)))) - 1
    return f"{round(x, d):.{max(d, 0)}f}"


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    out = H.run_all()
    out["key_numbers"] = key_numbers()
    out["pass2_refresh"] = P2.run()

    # ---- Pass 3: on-device adaptation ---------------------------------- #
    adaptation_track = HA.run_all()
    out["verdicts_adaptation"] = adaptation_track["verdicts"]
    out["adaptation_numbers"] = adaptation_numbers()
    out["pass3_refresh"] = P3.run()
    # Pass 4: primary measurement on the authors' own hardware.
    out["measured"] = MEAS.summary()
    # The equation of the paper, checked against itself. Passing the
    # study's own assumed terms in keeps the "were the inputs right?"
    # question in the same record as the "is the form right?" answer.
    out["measured"]["model_form_check"] = MEAS.model_form_check(
        assumed_power_w=D.RTX4090_GEN_POWER.value,
        assumed_tps=D.RTX4090_TPS_8B.value,
        assumed_beta=0.15)
    out["analysis_scope"] = {
        "revision": "RTX 5070 measurement and source audit, 2026-09-05",
        "primary_result": "measured.energy; GPU board request-window energy only",
        "legacy_scenarios": "key_numbers, monte_carlo, pass2_refresh and pass3_refresh are illustrative models, not empirical savings or audited current pricing",
        "capability": "workload weights and capability ratios are assumptions; 82% is not measured or calibrated to a 40-70% range",
        "cloud_energy": "Jegham values are estimates from API timing and inferred hardware; Gemini is a fleet-median measurement with a different workload",
        "training": "no training experiment rerun; memory agreement uses the calibration source, not an independent validation set",
        "fleet": "routing share is included once, inside the system savings fraction",
    }
    # A computed verdict is not empirical validation of its assumptions.
    for verdict in out["verdicts"] + out["verdicts_adaptation"]:
        verdict["model_result"] = verdict["result"]
        verdict["result"] = {
            "H6": "STRUCTURAL (authenticated endpoints required)",
            "H10": "STRUCTURAL (local execution required)",
            "H11": "NOT INDEPENDENTLY VALIDATED",
            "C4": "LITERATURE-BOUNDED; deployment untested",
            "C5": "RISK SUPPORTED; mitigation unvalidated",
            "C6": "MODELLED; central-sharing comparison unresolved",
        }.get(verdict["hid"], "MODELLED (conditional)")
    out["peft_methods"] = {
        k: {"name": m.name, "year": m.year, "family": m.family,
            "cite": m.cite, "peft_flag": m.peft_flag,
            "composes_with_4bit": m.composes_with_4bit,
            "quality_delta_pp": list(m.quality_delta),
            "memory_multiplier": m.memory_multiplier,
            "extra_train_time": m.extra_train_time, "note": m.note}
        for k, m in PF.METHODS.items()
    }
    out["devices"] = {
        k: {"name": d.name, "kind": d.kind,
            "accel_memory_gb": d.accel_memory_gb,
            "memory_bandwidth_gb_s": d.memory_bandwidth_gb_s,
            "train_power_w": d.train_power_w,
            "trainable_ceiling_gb": d.trainable_ceiling_gb(),
            "throughput": d.throughput, "cite": d.cite}
        for k, d in V.DEVICES.items()
    }
    out["parameters"] = {
        name: {"value": v.value, "lo": v.lo, "hi": v.hi, "unit": v.unit,
               "cite": v.cite, "note": v.note}
        for name, v in vars(D).items() if isinstance(v, D.Param)
    }
    out["citations"] = {k: {"text": t, "url": u} for k, (t, u) in D.CITATIONS.items()}

    path = os.path.join(HERE, "results.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, default=float)
    print("wrote", path)
    # Named macros tie each empirical number to one run and metric.
    commands = {}
    for row in out["measured"]["energy"]:
        run = {"rtx5070-study.json": "Rtx", "rtx5070-replication.json": "Repeat"}.get(row["file"])
        size = {"ministral-3:3b": "Three", "ministral-3:8b": "Eight"}.get(row["model"])
        if not run or not size:
            continue
        prefix = run + size
        # Energy is reported to TWO significant figures, which is what the
        # instrument supports: nvidia-smi's board counter refreshes far more
        # slowly than the harness polls it, so a 1.6 s request carries only
        # 4-5 distinct readings, and run-to-run spread reaches 16% on the 3B.
        # Earlier versions printed three, implying a precision the telemetry
        # does not have. Throughput and power keep more digits because they
        # come from request timings and a mean, not from the integral.
        for suffix, key, dp in (("Speed", "gen_tok_s", 1),
                                ("Power", "mean_w", 1), ("Time", "window_s", 2)):
            commands[prefix + suffix] = f"{row[key]:.{dp}f}"
        for suffix, key in (("Energy", "energy_wh"), ("Min", "energy_wh_min"),
                            ("Max", "energy_wh_max"),
                            ("MilliWh", "mwh_per_output_token")):
            commands[prefix + suffix] = _sigfig(row[key], 2)
        for watts, energy in row["host_power_scenarios_wh"].items():
            commands[prefix + {"0": "HostZero", "30": "HostThirty",
                               "60": "HostSixty",
                               "100": "HostHundred"}[watts]] = _sigfig(energy, 2)
    with open(os.path.join(HERE, "..", "paper", "empirical.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("% Generated by run_all.py from named measurement records.\n")
        for name, value in commands.items():
            f.write("\\newcommand{\\" + name + "}{" + value + "}\n")

    # regenerate figures (both tracks)
    subprocess.run([sys.executable, os.path.join(HERE, "figures.py")], check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "figures_adaptation.py")],
                   check=True)

    # console summary
    print("\n=== VERDICTS: footprint & routing (Pass 1-2) ===")
    for v in out["verdicts"]:
        print(f"  [{v['hid']:<3}] {v['result']:>28s}  {v['statement'][:58]}...")
    print("\n=== VERDICTS: on-device adaptation (Pass 3) ===")
    for v in out["verdicts_adaptation"]:
        print(f"  [{v['hid']:<3}] {v['result']:>28s}  {v['statement'][:58]}...")
    print("\n=== KEY NUMBERS ===")
    kn = out["key_numbers"]
    print(f"  f_s (SLM-sufficient workload)     : {kn['f_s_sufficient_fraction']*100:.0f}%")
    print(f"  right-sizing o3/8B                 : {kn['right_sizing_ratio_o3_over_8B']:.1f}x")
    print(f"  system carbon savings vs GPT-4o    : {kn['system_vs_gpt4o_long']['carbon']*100:.0f}%")
    p2 = out["pass2_refresh"]
    mid = p2["frontier_2026_tco"]["summary"]["mid_tier_ratio_range"]
    print(f"  2026 mid-tier TCO ratio (pass 2)   : {mid[0]:.0f}x - {mid[1]:.0f}x")
    print(f"  live-log validation (pass 2)       : "
          f"{p2['live_log_validation']['verdict'].split(':')[0]}")
    an = out["adaptation_numbers"]
    print(f"  adaptation: coverage gain          : {an['coverage_gain_pp']:+.1f} pp "
          f"(at alpha={an['alpha']:.2f})")
    print(f"  adaptation: quality gain           : {an['quality_gain_pp']:+.1f} pp "
          f"({an['gap_to_frontier_closed']*100:.0f}% of the gap to frontier)")
    # Both accountings, because quoting only the first is the error H8 records.
    h8 = next(v for v in out["verdicts_adaptation"] if v["hid"] == "H8")
    inc = h8["evidence"]["regimes"]["incremental_accounting"]
    a0 = f"{D.ALPHA_SUFFICIENCY.value:.2f}"
    print(f"  27B adapter break-even             : "
          f"{an['breakeven']['capability_27b_apu']['days_to_repay_energy']:.0f}"
          f" days vs all-cloud; "
          f"{'never' if not inc[a0]['repays'] else str(inc[a0]['days_to_repay']) + ' days'}"
          f" attributable at alpha={a0}")
    mv = an["memory_model_validation"]
    print(f"  memory model vs published 27B run  : "
          f"{mv['predicted_peak_gb']:.1f} GB predicted vs "
          f"{mv['reported_peak_gb']:.0f} GB reported "
          f"({mv['relative_error']*100:+.1f}%)")
    p3 = out["pass3_refresh"]
    print(f"  adapter evidence (pass 3)          : "
          f"{p3['deployment_validation']['status']}")
    ms = out["measured"]
    if ms.get("available"):
        bw = ms["bandwidth_bound_check"]
        b = ms["beta"]
        print(f"  measured machines (pass 4)         : "
              f"{len(ms['machines'])}, power sensor on "
              f"{sum(1 for m in ms['machines'] if m['power_sensor'])}")
        if b.get("available"):
            print(f"  prefill overhead beta              : measured "
                  f"{b['min']}-{b['max']} vs {b['assumed_in_study']} assumed")
        if bw.get("available"):
            print(f"  bandwidth-bound premise            : {bw['verdict']} "
                  f"(throughput {bw['throughput_ratio']}x vs bandwidth "
                  f"{bw['bandwidth_ratio']}x)")
        ps = ms.get("prefill_scaling", {})
        if ps.get("available") and ps.get("beta_at_study_typical_query"):
            bt = ps["beta_at_study_typical_query"]
            print(f"  beta at the study typical query    : "
                  f"{bt['min']}-{bt['max']} (median {bt['median']}) "
                  f"vs 0.15 assumed")
        fc = ms.get("model_form_check", {})
        if fc.get("available"):
            print(f"  edge-energy eq., bookkeeping audit    : "
                  f"{fc['verdict']} on {fc['n']} rows, worst residual "
                  f"{fc['max_abs_relative_error_pct']}% "
                  f"(consistency only; the functional form is untested "
                  f"- see model_form_check.what_this_tests)")
            av = fc.get("assumed_vs_measured")
            if av:
                for t in av["terms"]:
                    print(f"      assumed {t['term']:<24}: {t['assumed']} vs "
                          f"{t['measured']} measured "
                          f"({t['assumed_over_measured']}x, {t['direction_on_E']} E)")
        bl = ms.get("bandwidth_law", {})
        if bl.get("available"):
            # Per machine, never pooled: a percentage of 256 GB/s and a
            # percentage of 672 GB/s share an axis only after each has been
            # normalised to its own part, so a single pooled range would be
            # quoting two different denominators as one number.
            for name, g in (bl.get("by_machine") or {}).items():
                d = g.get("dense") or {}
                peak = g["rows"][0]["peak_bandwidth_gb_s"]
                short = name.replace("NVIDIA GeForce ", "").replace("(TM)", "")
                print(f"  ladder: {short.strip()[:24]:<24}: "
                      f"{d.get('models')} whole-file models spanning "
                      f"{d.get('weight_span')}x at "
                      f"{d.get('utilisation_pct_range', [0, 0])[0]:.0f}-"
                      f"{d.get('utilisation_pct_range', [0, 0])[1]:.0f}% of "
                      f"{peak:.0f} GB/s")
                sep = g.get("separation")
                if sep:
                    print(f"      separation: {sep['highest_whole_weight_set_pct']:.0f}% "
                          f"vs {sep['lowest_partial_reader_pct']:.0f}% -> "
                          f"{'SEPARATED' if sep['separated'] else 'OVERLAP'} "
                          f"by {sep['gap_pp']} pp")
    t27 = p3["tuning_tco"]["capability_27b"]
    print(f"  1-yr personalised TCO ratio        : "
          f"{t27['ratio_vs_cheapest_hosted']:.0f}x vs cheapest hosted")


if __name__ == "__main__":
    main()
