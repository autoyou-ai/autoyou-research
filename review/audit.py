"""Offline, independent arithmetic review of the paper's stored measurements.

No benchmark, provider, live log, or original research model is imported.
Run from the repository root: python review/audit.py
Results are a replay of supplied records, not independent physical replication.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

# The revised paper replays exactly these six archived records. Later studies
# store records with other schemas in measure/results/, so no directory glob.
INPUTS = ("amd-8060s-strixhalo-ladder.json", "amd-8060s-strixhalo.json",
          "rtx5070-ladder.json", "rtx5070-replication.json",
          "rtx5070-study.json", "rtx5070.json")


def integrate(trace, duration):
    """Integrate the linear interpolant, clipping each segment to [0, T]."""
    if duration <= 0 or len(trace) < 2:
        raise ValueError("positive duration and at least two samples required")
    if trace[0][0] > 0 or trace[-1][0] < duration:
        raise ValueError("trace does not bracket the request")
    energy_j = 0.0
    for (x, y), (u, v) in zip(trace, trace[1:]):
        if not all(math.isfinite(z) for z in (x, y, u, v)) or u <= x or min(y, v) < 0:
            raise ValueError("invalid power trace")
        lo, hi = max(x, 0), min(u, duration)
        if hi > lo:
            slope = (v - y) / (u - x)
            energy_j += (hi-lo) * (y + slope*((hi+lo)/2-x))
    return energy_j / 3600


def system_energy(board_wh, seconds, host_w, efficiency):
    if board_wh < 0 or seconds < 0 or host_w < 0 or not 0 < efficiency <= 1:
        raise ValueError("invalid physical parameter")
    return (board_wh + host_w*seconds/3600) / efficiency


def system_saving(cloud, edge, fraction, growth=0, overhead=0):
    """One common query-volume growth factor for each footprint component."""
    if cloud <= 0 or edge < 0 or not 0 <= fraction <= 1 or growth < 0 or overhead < 0:
        raise ValueError("invalid footprint or routing parameter")
    dec = fraction*edge + (1-fraction)*cloud + overhead
    return 1 - (1+growth)*dec/cloud


def routed_footprint(cloud, edge, attempt_fraction, fallback_fraction=0, overhead=0):
    if min(cloud, edge, overhead) < 0 or not 0 <= attempt_fraction <= 1 or not 0 <= fallback_fraction <= 1:
        raise ValueError("invalid routing parameter")
    return ((1-attempt_fraction)*cloud + attempt_fraction*edge +
            attempt_fraction*fallback_fraction*cloud + overhead)


def operational_carbon(energy_wh, grid_g_per_kwh):
    return energy_wh * grid_g_per_kwh / 1000


def operational_water(energy_wh, pue, onsite_l_per_it_kwh, grid_l_per_kwh):
    """Return mL; both intensities must describe consumption, or both withdrawal."""
    if energy_wh < 0 or pue < 1 or min(onsite_l_per_it_kwh, grid_l_per_kwh) < 0:
        raise ValueError("invalid water-accounting parameter")
    return energy_wh * (onsite_l_per_it_kwh/pue + grid_l_per_kwh)


def sig(value):
    if value == 0:
        return "0"
    dp = max(0, 1-int(math.floor(math.log10(abs(value)))))
    return f"{value:.{dp}f}"


def replay():
    checks, issues, energy_rows, throughput_rows, ladder_rows, files = [], [], [], [], [], {}
    max_error = 0.0
    raw_count, request_count = 0, 0
    dataset_rows = []
    for path in sorted(ROOT / "measure/results" / name for name in INPUTS):
        raw = path.read_bytes()
        files[path.name] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        is_ladder = data["harness"].endswith("--ladder")
        dataset_rows.append(dict(file=path.name, output_tokens=data["protocol"]["requested_tokens"],
                                 repetitions_per_model=data["protocol"]["reps"],
                                 models=len(data["models"]),
                                 raw_available=all("runs" in m for m in data["models"]),
                                 runtime=data["system"].get("ollama_version")))
        for model in data["models"]:
            if not model.get("ok"):
                issues.append(f"{path.name}/{model.get('model')}: unsuccessful record retained")
                continue
            runs = model.get("runs", [])
            raw_count += len(runs)
            if runs:
                rates = [r["eval_tokens"]/r["eval_s"] for r in runs]
                rate = st.median(rates)
                assert len(runs) == data["protocol"]["reps"], (path.name, model["model"])
                assert abs(rate-model["median_gen_tok_s"]) <= .0051
                assert all(r["eval_tokens"] == data["protocol"]["requested_tokens"] for r in runs)
            else:
                assert is_ladder, "non-ladder records must retain raw runs"
                rate = model["median_gen_tok_s"]
                rates = [model["gen_tok_s_spread"]["min"], model["gen_tok_s_spread"]["max"]]
                issues.append(f"{path.name}/{model['model']}: aggregate only; raw repetitions unavailable")
            row = dict(file=path.name, model=model["model"], gpu=data["system"]["gpu"],
                       repetitions=len(runs) if runs else model["reps"], raw_available=bool(runs),
                       tok_s=rate, tok_s_min=min(rates),
                       tok_s_max=max(rates), parameters=model.get("model_info",{}).get("parameter_size"))
            if is_ladder:
                size = model["weights_gb"]
                peak = data["protocol"]["peak_bandwidth_gb_s"]
                proxy = rate*size
                assert abs(proxy-model["effective_read_gb_s"]) < .2
                assert abs(proxy/peak-model["bandwidth_utilisation"]) < .001
                row.update(weights_gb=size, proxy_gb_s=proxy, proxy_percent=100*proxy/peak,
                           metadata=model.get("model_info", {}))
                ladder_rows.append(row)
            else:
                fit = model.get("prefill_scaling", {})
                if fit.get("available"):
                    xs = [p["prompt_tokens"] for p in fit["points"]]
                    ys = [p["prefill_s_median"] for p in fit["points"]]
                    slope, intercept = st.linear_regression(xs, ys)
                    residual = sum((y-intercept-slope*x)**2 for x,y in zip(xs,ys))
                    total = sum((y-st.mean(ys))**2 for y in ys)
                    row.update(prefill_tok_s=1/slope, prefill_intercept_ms=1000*intercept,
                               prefill_r2=1-residual/total,
                               beta_500_300=(intercept+500*slope)/(300/rate))
                    assert abs(1/slope-fit["fit"]["marginal_prefill_tok_s"])/(1/slope) < .002
                    assert abs(intercept*1000-fit["fit"]["fixed_overhead_ms"]) < .02
                throughput_rows.append(row.copy())
                powered = [r for r in runs if r.get("power", {}).get("measured")]
                if powered:
                    assert len(powered) == len(runs)
                    values, durations, powers, distinct, gaps = [], [], [], [], []
                    timing_overhead = []
                    for run in runs:
                        p = run["power"]
                        seconds = p["window_s"]
                        val = integrate(p["trace_s_w"], seconds)
                        error = abs(val-p["energy_wh"])
                        max_error = max(max_error, error)
                        assert error <= .00000051
                        assert abs(val*3600/seconds-p["mean_w"]) <= .0051
                        assert abs(seconds-run["wall_s"]) < .01
                        values.append(val)
                        durations.append(seconds)
                        powers.append(val*3600/seconds)
                        inside = [(t,w) for t,w in p["trace_s_w"] if 0<=t<=seconds]
                        distinct.append(len(set(w for _,w in inside)))
                        gaps.append(max(b[0]-a[0] for a,b in zip(p["trace_s_w"],p["trace_s_w"][1:])))
                        timing_overhead.append(1000*(seconds-run["eval_s"]-run["prefill_s"]))
                        request_count += 1
                    med = st.median(values)
                    assert abs(med-model["measured_energy"]["median_energy_wh_per_query"]) < .00000051
                    row.update(board_wh=med, board_wh_min=min(values), board_wh_max=max(values),
                               board_cv=st.stdev(values)/st.mean(values),
                               window_s=st.median(durations), mean_w_median=st.median(powers),
                               prompt_tokens=sorted(set(r["prompt_tokens"] for r in runs)),
                               output_tokens=sorted(set(r["eval_tokens"] for r in runs)),
                               distinct_in_window_range=[min(distinct),max(distinct)],
                               maximum_sample_gap_s=max(gaps),
                               overhead_ms_range=[min(timing_overhead),max(timing_overhead)],
                               host_wall_wh={str(h):st.median(system_energy(e,t,h,.9)
                                   for e,t in zip(values,durations)) for h in (0,30,60,100)},
                               electricity_usd_60w=st.median(system_energy(e,t,60,.9)
                                   for e,t in zip(values,durations))*.16/1000)
                    energy_rows.append(row)
            checks.append(f"{path.name}: {model['model']}: " +
                          ("raw replay PASS" if runs else "summary arithmetic only PASS"))

    # Recompute the ORIGINAL scenario directly, without calling its model code.
    edge_board = 420*(300/141)*1.15/3600
    edge_wh = edge_board*1.02
    embodied_g = 350*1000*.1/5_000_000
    ci, wue, ewif, pue = 458, .20, 1.10, 1.10
    edge = dict(energy=edge_wh, water=edge_wh*ewif, carbon=edge_wh/1000*ci+embodied_g)
    legacy=[]
    for name, ec in [("Gemini median",.24),("GPT-4o short",.421),("GPT-4o long",1.788),
                     ("Claude-3.7 long",5.518),("o3 long",39.223)]:
        cloud=dict(energy=ec,water=ec/pue*wue+ec*ewif,carbon=ec/1000*ci)
        dec={k:.82*edge[k]+.18*cloud[k] for k in cloud}
        old={k:(cloud[k]-dec[k]-.2*max(cloud[k]-dec[k],0))/cloud[k] for k in cloud}
        implied={k:.2*max(cloud[k]-dec[k],0)/dec[k] for k in cloud}
        common={k:system_saving(cloud[k],edge[k],.82,implied["carbon"]) for k in cloud}
        legacy.append(dict(name=name,cloud=cloud,edge=edge,original_saving=old,
                           implied_growth=implied,common_carbon_calibrated_growth_saving=common))
    dc_base=945*.45*.65
    original_fleet=dc_base*.25*legacy[2]["original_saving"]["energy"]
    new_carbon=(dc_base*1e9*.25/1.788)*legacy[2]["cloud"]["carbon"]*legacy[2]["original_saving"]["carbon"]/1e9
    return dict(scope="Offline replay of supplied raw records; no new hardware experiment or independent peer review",
                raw_sha256=files, dataset_rows=dataset_rows, checks=checks, warnings=issues, raw_repetitions=raw_count,
                integrated_requests=request_count, max_energy_rounding_difference_wh=max_error,
                energy_rows=energy_rows, throughput_rows=throughput_rows, ladder_rows=ladder_rows,
                legacy_scenarios=legacy, legacy_fleet=dict(inference_twh=dc_base,
                    operational_twh=original_fleet,original_carbon_mt=original_fleet*.458,
                    embodied_adjusted_carbon_mt=new_carbon),
                missing_evidence=["matched local/cloud task-quality evaluation", "whole-system wall-meter measurements",
                    "independent laboratory replication", "empirically sampled workload mix",
                    "observed rebound elasticity", "network path energy measurements",
                    "AMD ladder raw repetitions (nine aggregate-only records)"])


def emit_tables(report):
    lines=["% Generated from review/audit.py. Do not edit numerical rows manually."]
    primary=[r for r in report["energy_rows"] if r["file"] in
             ("rtx5070-study.json","rtx5070-replication.json")]
    primary.sort(key=lambda r:(r["model"],r["file"]!="rtx5070-study.json"))
    energy, host = [], []
    for r in primary:
        size="3B" if r["model"].endswith("3b") else "8B"
        label="Primary" if r["file"]=="rtx5070-study.json" else "Repeat"
        energy.append(f"{size} & {label} & {r['tok_s']:.1f} & {sig(r['board_wh'])} & "
                      f"{sig(r['board_wh_min'])}--{sig(r['board_wh_max'])} " + r"\\")
        if label=="Primary":
            host.append(f"{size} & {sig(r['board_wh'])} & " + " & ".join(sig(r['host_wall_wh'][str(h)])
                       for h in (0,30,60,100)) + r" \\")
    lines += [r"\newcommand{\ReviewEnergyRows}{"+"\n"+"\n".join(energy)+"}",
              r"\newcommand{\ReviewHostRows}{"+"\n"+"\n".join(host)+"}"]
    amd=[]
    for r in report["throughput_rows"]:
        if r["file"]!="amd-8060s-strixhalo.json":
            continue
        name={"ministral-3:3b":"Ministral-3 3B", "ministral-3:8b":"Ministral-3 8B",
              "qwen3.8:27b":"Qwen3.8 27B"}.get(r['model'], r['model'])
        amd.append(f"{name} & {r['tok_s']:.1f} & {r['prefill_tok_s']:.0f} & "
                   f"{r['prefill_intercept_ms']:.1f} & {r['beta_500_300']:.3f} "+r"\\")
    lines += [r"\newcommand{\ReviewAmdRows}{"+"\n"+"\n".join(amd)+"}",
              r"\newcommand{\ReviewIntegratedRequests}{"+str(report["integrated_requests"])+"}",
              r"\newcommand{\ReviewRawRepetitions}{"+str(report["raw_repetitions"])+"}"]
    ladder=[]
    for r in report["ladder_rows"]:
        platform="AMD" if r["file"].startswith("amd") else "RTX"
        label=r["model"].replace("_",r"\_")
        ladder.append(f"{platform} & \\texttt{{{label}}} & {r['weights_gb']:.3f} & "
                      f"{r['tok_s']:.2f} & {r['proxy_percent']:.1f} & "
                      +( "Raw" if r["raw_available"] else "Summary")+r" \\")
    lines += [r"\newcommand{\ReviewLadderRows}{"+"\n"+"\n".join(ladder)+"}"]
    (ROOT/"paper/review_numbers.tex").write_text("\n".join(lines)+"\n",encoding="ascii",newline="\n")


def plots(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"svg.fonttype":"none"})
    rows=[r for r in report['energy_rows'] if r['file']=='rtx5070-study.json']
    fig, ax=plt.subplots(figsize=(6.5,3.5),layout="constrained")
    for r,color in zip(rows,("#237A78","#8759A5")):
        xs=[0,30,60,100]
        ax.plot(xs,[r['host_wall_wh'][str(h)] for h in xs],marker="o",color=color,
                label=r['model'].split(':')[-1].upper()+" + supply-loss assumption")
        ax.axhline(r['board_wh'],color=color,linestyle=':',alpha=.8)
    ax.set(xlabel="Assumed host DC power (W)",ylabel="Request energy (Wh)",ylim=(0,.35))
    ax.grid(axis='y',alpha=.18)
    ax.legend(loc="upper left",fontsize=9)
    fig.savefig(ROOT/"figures/review_energy.pdf")
    fig.savefig(ROOT/"figures/review_energy.png",dpi=180)
    plt.close(fig)
    fig, axs=plt.subplots(1,2,figsize=(9,4),layout="constrained",gridspec_kw={"width_ratios":[2,1]})
    for ax, prefix, title in zip(axs,("amd-","rtx5070"),("Radeon 8060S: 256 GB/s", "RTX 5070: 672 GB/s")):
        rows=[r for r in report['ladder_rows'] if r['file'].startswith(prefix)]
        labels=[r['model'].replace('autoyou','private') for r in rows]
        ax.barh(labels,[r['proxy_percent'] for r in rows],color="#237A78")
        ax.axvline(100,color="#B45432",ls='--',lw=1)
        ax.set(xlabel="File-size proxy / rated bandwidth (%)",title=title)
        ax.tick_params(axis='y',labelsize=8)
        ax.invert_yaxis()
    fig.savefig(ROOT/"figures/review_bandwidth.pdf")
    fig.savefig(ROOT/"figures/review_bandwidth.png",dpi=180)
    plt.close(fig)


def main():
    report=replay()
    (HERE/"audit_results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    emit_tables(report)
    plots(report)
    print(json.dumps({k:report[k] for k in ("scope","raw_repetitions","integrated_requests",
          "max_energy_rounding_difference_wh","legacy_fleet","missing_evidence")},indent=2))
    print(f"PASS: {len(report['checks'])} model/run groups; arithmetic replay and generated tables.")


if __name__ == "__main__":
    main()
