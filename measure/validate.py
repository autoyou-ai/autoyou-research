"""Independent arithmetic replay; imports no benchmark or research-model code.

Run after run_all.py: python measure/validate.py
This is code-path independence, not third-party laboratory replication.
"""
import hashlib
import json
import math
import re
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def integrate(trace, duration):
    """Analytic antiderivative of each line segment, clipped to [0, duration]."""
    assert trace[0][0] <= 0 and trace[-1][0] >= duration
    total = 0.0
    for (x, y), (u, v) in zip(trace, trace[1:]):
        assert u > x and all(math.isfinite(z) for z in (x, y, u, v))
        assert y >= 0 and v >= 0
        lo, hi = max(0, x), min(duration, u)
        if lo < hi:
            total += y * (hi - lo) + (v-y)/(u-x) * ((hi-x)**2-(lo-x)**2)/2
    return total


def main():
    results = json.loads((ROOT / "models/results.json").read_text(encoding="utf-8"))
    exported = {(r["file"], r["model"]): r for r in results["measured"]["energy"]}
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]+)\}",
                            (ROOT / "paper/empirical.tex").read_text()))
    checks, files = [], {}
    for path in sorted((ROOT / "measure/results").glob("rtx5070*.json")):
        raw = path.read_bytes()
        files[path.name] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        ladder = data["harness"].endswith("--ladder")
        for model in data["models"]:
            assert model["ok"], (path.name, model)
            runs = model["runs"]
            assert len(runs) == data["protocol"]["reps"]
            rates = [r["eval_tokens"] / r["eval_s"] for r in runs]
            assert abs(stats.median(rates) - model["median_gen_tok_s"]) <= .0051
            assert all(r["eval_tokens"] == data["protocol"]["requested_tokens"] for r in runs)
            residency = [r for r in model["residency"] if r["name"] == model["model"]]
            assert residency and residency[0]["size_vram"] == residency[0]["size"]
            if ladder:
                # The stored GB value is rounded to 0.001 GB.
                estimate = stats.median(rates) * model["weights_gb"]
                assert abs(estimate-model["effective_read_gb_s"]) < .2
                expected = estimate/data["protocol"]["peak_bandwidth_gb_s"]
                assert abs(expected-model["bandwidth_utilisation"]) < .001
            else:
                energies = []
                for run in runs:
                    power = run["power"]
                    assert power["measured"]
                    duration = power["window_s"]
                    energy = integrate(power["trace_s_w"], duration)/3600
                    assert abs(energy-power["energy_wh"]) <= .00000051
                    assert abs(energy*3600/duration-power["mean_w"]) <= .0051
                    assert abs(duration-run["wall_s"]) < .01
                    energies.append(energy)
                median = stats.median(energies)
                assert abs(median-model["measured_energy"]["median_energy_wh_per_query"]) < .00000051
                row = exported[path.name, model["model"]]
                assert abs(median-row["energy_wh"]) < .00000051
                for watts, actual in row["host_power_scenarios_wh"].items():
                    expected = stats.median(e+float(watts)*r["power"]["window_s"]/3600
                                            for e, r in zip(energies, runs))
                    assert abs(expected-actual) < .00000051
                prefix = {"rtx5070-study.json": "Rtx", "rtx5070-replication.json": "Repeat"}.get(path.name)
                if prefix:
                    prefix += {"ministral-3:3b": "Three", "ministral-3:8b": "Eight"}[model["model"]]
                    assert macros[prefix+"Energy"] == f"{median:.3f}"
                    assert macros[prefix+"Speed"] == f"{stats.median(rates):.1f}"
                fit = model.get("prefill_scaling", {})
                if fit.get("available"):
                    xs = [p["prompt_tokens"] for p in fit["points"]]
                    ys = [p["prefill_s_median"] for p in fit["points"]]
                    slope, intercept = stats.linear_regression(xs, ys)
                    # Raw point medians were rounded before serialization.
                    assert abs(1/slope-fit["fit"]["marginal_prefill_tok_s"])/(1/slope) < .002
                    assert abs(intercept*1000-fit["fit"]["fixed_overhead_ms"]) < .02
            checks.append({"file": path.name, "model": model["model"], "repetitions": len(runs), "status": "PASS"})
    p = results["parameters"]
    k = results["key_numbers"]
    mix = {"extraction": .18, "rag_qa": .30, "summary": .22, "simple_code": .12, "hard_reason": .18}
    ratios = {"extraction": .95, "rag_qa": .90, "summary": .88, "simple_code": .75, "hard_reason": .51}
    fs = sum(w for name, w in mix.items() if ratios[name] >= .70)
    assert abs(fs-k["f_s_sufficient_fraction"]) < 1e-12
    expected_fleet = (p["DC_DEMAND_2030_TWH"]["value"] * p["AI_SHARE_OF_DC"]["value"] *
                      p["INFERENCE_SHARE_OF_AI"]["value"] * .25 * k["system_vs_gpt4o_long"]["energy"])
    assert abs(expected_fleet-k["fleet_savings_25pct_adoption"]["energy_saved_twh"]) < 1e-10
    # Seven changed routing classes are not invented from an assumed uplift.
    base_quality = sum(w*ratios[n] for n,w in mix.items() if ratios[n] >= .70)/fs
    adapted_quality = sum(w*min(.99,ratios[n]+.06) for n,w in mix.items() if ratios[n] >= .70)/fs
    a = results["adaptation_numbers"]
    assert abs(base_quality-a["mean_quality_base"]) < .0001
    assert abs(adapted_quality-a["mean_quality_adapted"]) < .0001
    assert a["coverage_gain_pp"] == 0
    assert not results["pass2_refresh"]["live_log_validation"].get("available", False)
    report = {"status": "PASS", "scope": "independent formula implementation on the same raw files; not independent hardware or peer review",
              "files_sha256": files, "checks": checks, "fleet_routing_fraction_counted_once": True,
              "quality_and_coverage_recomputed": True, "limitations": ["No whole-system wattmeter", "No matched cloud-quality evaluation", "No new adaptation training or safety trial"]}
    target = ROOT / "measure/validation.json"
    target.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"PASS: {len(checks)} model/run groups; raw power integration, timing, residency, ladder, named paper metrics, fleet and adaptation arithmetic")
    print(target)


if __name__ == "__main__":
    main()
