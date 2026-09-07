# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
measured.py - Primary measurements from this study's own hardware.

Until Pass 4 every input to the edge-energy model came from somewhere else:

    E_edge = P * (n_o / tau) * (1 + beta) / 3600

    tau   throughput      a third-party benchmark blog for an RTX 4090
    beta  prefill factor  an author assumption of 0.15
    P     board power     a vendor TDP figure

That is three borrowed numbers describing hardware nobody in this study had
touched. ``measure/bench.py`` measures the first two directly, on the
machines the authors actually own, against the models those machines actually
run. This module reads what that harness wrote and exposes it to the study.

Same discipline as evidence.py: it reads artifacts that already exist, reports
``available=False`` when they do not, and never substitutes a modelled value for
a measured one without saying so. A parameter that could not be measured is
reported as unmeasured rather than quietly filled in.

WHAT IS MEASURED AND WHAT IS NOT
    Measured : generation throughput, prefill throughput, the prefill overhead
               beta, cold load time, and the exact model build (architecture,
               parameter count, quantisation) each figure belongs to.
    Not      : board power, on any machine without a power sensor. NVIDIA parts
               expose it through nvidia-smi and are measured; this study's AMD
               APU exposes nothing readable without a kernel-mode helper, so its
               energy figures still carry a modelled P and say so at every use.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "measure", "results"))


def _read(path: str) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    d["_file"] = os.path.basename(path)
    return d


def _is_ladder(d: Dict) -> bool:
    return str(d.get("harness", "")).endswith("--ladder")


def _load_all() -> List[Dict]:
    """Per-machine throughput artifacts. A ladder run is a different
    experiment with a different schema and is loaded separately."""
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        d = _read(path)
        if d is not None and not _is_ladder(d):
            out.append(d)
    return out


def _load_ladders() -> List[Dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        d = _read(path)
        if d is not None and _is_ladder(d):
            out.append(d)
    return out


def machines() -> List[Dict[str, object]]:
    """One row per measured machine, with its device class and sensor status."""
    rows = []
    for d in _load_all():
        sysinfo = d.get("system", {})
        sensor = d.get("power_sensor", {})
        gpu = str(sysinfo.get("gpu") or "")
        # Classify by how memory is attached, which is the axis that decides
        # what a machine can hold - see devices.py.
        kind = "unified_apu" if ("Radeon" in gpu and "RX" not in gpu) else "discrete_gpu"
        rows.append({
            "file": d["_file"],
            "gpu": gpu,
            "cpu": sysinfo.get("cpu"),
            "ram_gb": sysinfo.get("ram_gb"),
            "device_class": kind,
            "ollama_version": sysinfo.get("ollama_version"),
            "measured_at": sysinfo.get("measured_at"),
            "power_sensor": bool(sensor.get("available")),
            "power_sensor_tool": sensor.get("tool"),
            "models_measured": sum(1 for m in d.get("models", []) if m.get("ok")),
        })
    unique = {}
    for row in rows:
        key = (row["gpu"], row["cpu"])
        if key not in unique:
            unique[key] = dict(row, measurement_files=[row["file"]])
        else:
            unique[key]["measurement_files"].append(row["file"])
            unique[key]["power_sensor"] |= row["power_sensor"]
    return list(unique.values())


def throughput_table() -> List[Dict[str, object]]:
    """Every (machine, model) generation-throughput measurement."""
    rows = []
    for d in _load_all():
        sysinfo = d.get("system", {})
        for m in d.get("models", []):
            if not m.get("ok"):
                continue
            mi = m.get("model_info") or {}
            spread = m.get("gen_tok_s_spread") or {}
            rows.append({
                "file": d["_file"],
                "machine": sysinfo.get("gpu"),
                "model": m["model"],
                "architecture": mi.get("architecture"),
                "parameters": mi.get("parameter_size"),
                # Numeric twin of the parameter-size string, so a figure the
                # prose quotes ("a 7.6B build") is checkable by the paper
                # verifier rather than living only inside a label.
                "parameters_b": _params_b(mi.get("parameter_size")),
                "quantization": mi.get("quantization"),
                "gen_tok_s": m["median_gen_tok_s"],
                "gen_tok_s_stdev": spread.get("stdev"),
                "gen_tok_s_min": spread.get("min"),
                "gen_tok_s_max": spread.get("max"),
                "gen_tok_s_rel_stdev": (
                    round(spread["stdev"] / m["median_gen_tok_s"], 5)
                    if spread.get("stdev") else None),
                "prefill_tok_s": m.get("median_prefill_tok_s"),
                "beta": m.get("median_beta_prefill_overhead"),
                "prompt_tokens": m.get("median_prompt_tokens"),
                "generated_tokens": m.get("median_eval_tokens"),
                "reps": m.get("reps"),
                "cold_load_s": m.get("load_s_cold"),
            })
    return rows


def _params_b(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    t = str(text).strip().upper().rstrip("B")
    try:
        return float(t)
    except ValueError:
        return None


def measured_beta() -> Dict[str, object]:
    """The prefill overhead the study assumed at 0.15.

    beta is the ratio of prefill time to generation time for one query. It
    depends on prompt length and generated length, so the measured value is
    reported together with both rather than as a bare constant.
    """
    rows = [r for r in throughput_table() if r.get("beta") is not None]
    if not rows:
        return {"available": False}
    betas = [r["beta"] for r in rows]
    return {
        "available": True,
        "n": len(betas),
        "median": round(statistics.median(betas), 4),
        "min": round(min(betas), 4),
        "max": round(max(betas), 4),
        "assumed_in_study": 0.15,
        "per_model": {r["file"] + ":" + r["model"]: {"beta": r["beta"],
                                   "prompt_tokens": r["prompt_tokens"],
                                   "generated_tokens": r["generated_tokens"]}
                      for r in rows},
        "note": (
            "Measured across the prompt and generation lengths in the harness "
            "protocol. Every measured value is far below the 0.15 the study "
            "assumed, which means the study OVERSTATED edge energy: a smaller "
            "beta makes an edge query cheaper, not dearer. Correcting it "
            "strengthens the thesis rather than weakening it, which is exactly "
            "why it is worth stating that the original figure was a guess. "
            "beta rises with the prompt-to-generation ratio, so a "
            "retrieval-heavy workload with long contexts would sit higher than "
            "these numbers; they are not a universal constant."
        ),
    }


def bandwidth_bound_check() -> Dict[str, object]:
    """Does measured throughput scale with memory bandwidth, as assumed?

    devices.py asserts that consumer inference is memory-bandwidth bound, which
    is the premise behind the whole device-class argument. The published RTX
    4090 figure and this study's measured APU figure are for the same model
    class, so their ratio can be checked against the ratio of the two parts'
    memory bandwidths. Agreement is evidence for the premise; disagreement would
    mean something other than bandwidth dominates.
    """
    rows = throughput_table()
    eight_b = [r for r in rows if "Radeon 8060S" in str(r["machine"])
               if (_params_b(r.get("parameters")) or 0) >= 8
               and (_params_b(r.get("parameters")) or 0) < 12]
    if not eight_b:
        return {"available": False}
    apu = eight_b[0]
    # Both figures are 8B-class single-stream generation.
    rtx4090_tok_s = 141.0        # databasemart2026, cited in data.py
    rtx4090_bw = 1008.0          # GB/s, vendor specification
    apu_bw = 256.0               # GB/s, LPDDR5X-8000 on a 256-bit bus
    tp_ratio = rtx4090_tok_s / apu["gen_tok_s"]
    bw_ratio = rtx4090_bw / apu_bw
    return {
        "available": True,
        "apu_model": apu["model"],
        "apu_gen_tok_s": apu["gen_tok_s"],
        "rtx4090_gen_tok_s_cited": rtx4090_tok_s,
        "throughput_ratio": round(tp_ratio, 2),
        "bandwidth_ratio": round(bw_ratio, 2),
        "agreement": round(tp_ratio / bw_ratio, 3),
        "verdict": ("CONSISTENT" if 0.8 <= tp_ratio / bw_ratio <= 1.25
                    else "INCONSISTENT"),
        "note": (
            "If single-stream decode were compute bound the throughput ratio "
            "would track arithmetic throughput, where the two parts differ by "
            "far more than 4x. It tracks memory bandwidth instead. This is the "
            "premise the device-class argument rests on, and it is now checked "
            "against a measurement rather than asserted. Caveat: the two "
            "figures come from different serving stacks and quantisations, so "
            "this is corroboration, not a controlled experiment."
        ),
    }


def scaling_check() -> Dict[str, object]:
    """How throughput falls as the model grows, on one machine.

    A bandwidth-bound regime predicts throughput inversely proportional to the
    bytes read per token, i.e. roughly inversely proportional to parameter count
    at fixed quantisation.
    """
    by_machine: Dict[str, List[Dict]] = {}
    for r in throughput_table():
        by_machine.setdefault(str(r["machine"]) + " / " + r["file"], []).append(r)
    out = {}
    for machine, rows in by_machine.items():
        pts = sorted(
            ((_params_b(r["parameters"]), r["gen_tok_s"], r["model"],
              r["quantization"]) for r in rows if _params_b(r["parameters"])),
            key=lambda t: t[0])
        if len(pts) < 2:
            continue
        steps = []
        for (p0, t0, m0, q0), (p1, t1, m1, q1) in zip(pts, pts[1:]):
            steps.append({
                "from": m0, "to": m1,
                "param_ratio": round(p1 / p0, 2),
                "throughput_ratio": round(t0 / t1, 2),
                "same_quantisation": q0 == q1,
            })
        out[machine] = {
            "points": [{"model": m, "params_b": p, "gen_tok_s": t,
                        "quantization": q} for p, t, m, q in pts],
            "steps": steps,
        }
    return {"available": bool(out), "machines": out}


def prefill_scaling() -> Dict[str, object]:
    """Prefill separated into a fixed per-request cost and a per-token cost.

    "Prefill throughput" is not a single number. Prefill time is a fixed cost
    plus a marginal cost per prompt token, so dividing one measurement by its
    prompt length gives a figure that depends on the prompt you happened to
    send. The harness sweeps prompt length and regresses, which yields a
    marginal rate comparable across models - and, as a by-product, caught a
    caching artefact that a single length would have reported as fact.
    """
    rows = []
    for d in _load_all():
        sysinfo = d.get("system", {})
        for m in d.get("models", []):
            ps = m.get("prefill_scaling") or {}
            if not ps.get("available"):
                continue
            fit = ps["fit"]
            bt = ps.get("beta_at_study_typical_query") or {}
            rows.append({
                "machine": sysinfo.get("gpu"),
                "model": m["model"],
                "marginal_prefill_tok_s": fit.get("marginal_prefill_tok_s"),
                "fixed_overhead_ms": fit.get("fixed_overhead_ms"),
                "r2": fit.get("r2"),
                "lengths_tested": fit.get("lengths_tested"),
                "beta_at_500_in_300_out": bt.get("beta"),
                "template_cache_probe": ps.get("template_cache_probe"),
                "repeat_cache_probe": m.get("repeat_cache_probe"),
            })
    if not rows:
        return {"available": False}
    betas = [r["beta_at_500_in_300_out"] for r in rows
             if r["beta_at_500_in_300_out"] is not None]
    return {
        "available": True,
        "rows": rows,
        "fit_model": "prefill_s = a + b * prompt_tokens, OLS over medians",
        "min_r2": min(r["r2"] for r in rows if r.get("r2") is not None),
        "beta_at_study_typical_query": ({
            "n": len(betas),
            "min": round(min(betas), 4),
            "max": round(max(betas), 4),
            "median": round(statistics.median(betas), 4),
            "prompt_tokens": 500,
            "generated_tokens": 300,
        } if betas else None),
        "note": (
            "Every fit has a small POSITIVE fixed cost, which is what a "
            "per-request overhead should look like. An earlier version of the "
            "harness, which used the chat-template endpoint, produced large "
            "negative intercepts (-485 ms on the 8B). That is not a physical "
            "quantity, and it is how the template-prefix caching artefact was "
            "found: the template system prefix is counted in prompt_eval_count "
            "but is cached and never recomputed, so the fitted line has to "
            "dive below zero to reach the shortest point. The regression is "
            "therefore both the measurement and its own sanity check."
        ),
    }


def _decode_class(m: Dict) -> str:
    """Does this build read its whole weight set once per token?

    The ladder tests a prediction that only holds for a dense model doing
    one forward pass per token. Two architectures break that assumption in
    opposite-looking but related ways, and both are read from the server
    metadata rather than inferred from the model name or its file size:

      sparse  a mixture of experts reads expert_used_count of
              expert_count experts per token, so it streams far fewer
              bytes than it stores. File size does NOT reveal this: the
              file holds every expert.
      mtp     a multi-token-prediction head emits more than one token per
              pass when its draft is accepted, so tokens per second is no
              longer passes per second.
      elastic a per-layer-embedding build stores embedding tables from
              which only the rows for the current token are fetched, so
              the file again overstates the bytes streamed.

    All three should sit ABOVE the bandwidth ceiling on this statistic,
    which is what makes them controls rather than nuisances.
    """
    mi = m.get("model_info") or {}
    if mi.get("mixture_of_experts"):
        return "sparse"
    if (mi.get("nextn_predict_layers") or 0) > 0:
        return "mtp"
    if (mi.get("per_layer_embedding_input") or 0) > 0:
        return "elastic"
    return "dense"


def _looks_sparse(m: Dict) -> bool:
    return _decode_class(m) != "dense"


# Where a reader can obtain each build's weights. A row whose weights cannot
# be downloaded is weaker evidence than one whose can, because the file size
# the statistic divides by cannot be independently checked. That is a reason
# to LABEL a row, not to drop it: an earlier version of this module filtered
# these rows out of the published ladder, and the two rows it removed were
# the two that contradicted the headline. Every measured row is now reported
# and the redistributability of its weights is carried as a column.
WEIGHTS_URL = {
    "ministral-3:3b": "https://ollama.com/library/ministral-3",
    "ministral-3:8b": "https://ollama.com/library/ministral-3",
    "qwen3:4b": "https://ollama.com/library/qwen3",
    "qwen3.6:latest": "https://ollama.com/library/qwen3.6",
    "qwen3.8:27b": "https://ollama.com/library/qwen3.8",
    "gemma4:26b": "https://ollama.com/library/gemma4",
    "gemma4:e4b": "https://ollama.com/library/gemma4",
    "muse-glimmer:30b": "https://huggingface.co/meta-models/Muse-Glimmer-30B",
}

# Builds that exist only inside this deployment: fine-tunes whose weights are
# not redistributable. Named explicitly rather than matched on a prefix, so a
# build cannot become "public" by being renamed.
NON_REDISTRIBUTABLE = ("private-3.1b-q8:latest", "private-7.6b-vlm-q8:latest")


def _weights_url(name: str) -> Optional[str]:
    return WEIGHTS_URL.get(str(name))


def _is_redistributable(name: str) -> bool:
    return str(name) not in NON_REDISTRIBUTABLE


def bandwidth_law(machine: Optional[str] = None) -> Dict[str, object]:
    """Within one machine: does tau * bytes-per-token sit at the memory ceiling?

    The statistic is

        effective read bandwidth = tau * on-disk weight bytes

    and the premise it was built to test is that single-stream decode streams
    the whole weight set once per token, so a dense build should land near its
    part's specified peak and must not exceed it.

    Reported over every measured row, including the two whose weights are not
    redistributable. That matters, because those two are the rows that decide
    what the statistic actually shows: one dense build lands at 105.7% of a
    ceiling no whole-file reader can exceed. It is not reading 9.45 GB per
    token. The excess is the proxy's error, not the hardware's - the file
    holds a vision tower and embedding tables that decode does not stream -
    and it puts a floor under how tightly this statistic can be read.

    So the honest reading is weaker than the original one and more useful. The
    dense class does not sit in a tight band at the ceiling; over every row
    measured it spans 76.7% to 105.7%. What survives is the ordering: builds
    that provably read only part of their weights sit higher than builds that
    do not, by a margin (30 pp) larger than the proxy error, so the statistic
    still separates the two decode classes. It does not measure utilisation to
    the percentage point, and no claim here depends on it doing so.
    """
    out = []
    for d in _load_ladders():
        peak = (d.get("protocol") or {}).get("peak_bandwidth_gb_s")
        sysinfo = d.get("system", {})
        if machine is not None and sysinfo.get("gpu") != machine:
            continue
        rows = [m for m in d.get("models", []) if m.get("ok")]
        if not rows or not peak:
            continue
        for m in rows:
            mi = m.get("model_info") or {}
            out.append({
                "machine": sysinfo.get("gpu"),
                "peak_bandwidth_gb_s": peak,
                "model": m["model"],
                "architecture": str(mi.get("architecture") or ""),
                "parameters": mi.get("parameter_size"),
                # Numeric twin of the parameter-size string, so a figure the
                # prose quotes ("a 7.6B build") is checkable by the paper
                # verifier rather than living only inside a label.
                "parameters_b": _params_b(mi.get("parameter_size")),
                "quantization": mi.get("quantization"),
                "weights_gb": m.get("weights_gb"),
                "gen_tok_s": m.get("median_gen_tok_s"),
                "effective_read_gb_s": m.get("effective_read_gb_s"),
                "utilisation": m.get("bandwidth_utilisation"),
                "utilisation_pct": (round(m["bandwidth_utilisation"] * 100, 1)
                                    if m.get("bandwidth_utilisation")
                                    else None),
                "decode_class": _decode_class(m),
                "multimodal_tower": bool(mi.get("multimodal_tower")),
                "expert_active_fraction": mi.get("expert_active_fraction"),
                "sparse": _looks_sparse(m),
                "redistributable": _is_redistributable(m["model"]),
                "weights_url": _weights_url(m["model"]),
            })
    if not out:
        return {"available": False,
                "reason": ("no ladder artifact; run "
                           "bench.py --ladder --bandwidth <GB/s>")}
    # The primary group is every build that must stream its whole weight set
    # once per token. The control group is every build that provably cannot be
    # doing so. The test is whether the statistic separates them.
    whole = [r for r in out if r["decode_class"] == "dense"]
    partial = [r for r in out if r["decode_class"] != "dense"]
    sparse = [r for r in out if r["decode_class"] == "sparse"]
    mtp = [r for r in out if r["decode_class"] == "mtp"]
    elastic = [r for r in out if r["decode_class"] == "elastic"]
    res: Dict[str, object] = {
        "available": True, "rows": out,
        "n_whole_weight_set": len(whole), "n_partial_reader": len(partial),
        "n_sparse": len(sparse), "n_mtp": len(mtp), "n_elastic": len(elastic),
        "n_rows": len(out),
        "n_redistributable": sum(1 for r in out if r["redistributable"]),
        "n_not_redistributable": sum(1 for r in out
                                     if not r["redistributable"]),
    }
    if machine is None:
        res["by_machine"] = {name: bandwidth_law(name)
                             for name in sorted({r["machine"] for r in out})}
    if whole:
        us = [r["utilisation"] for r in whole]
        eff = [r["effective_read_gb_s"] for r in whole]
        sizes = [r["weights_gb"] for r in whole]
        res["dense"] = {
            "models": len(whole),
            "with_vision_projector": sum(1 for r in whole
                                         if r["multimodal_tower"]),
            "weight_range_gb": [round(min(sizes), 2), round(max(sizes), 2)],
            "weight_span": round(max(sizes) / min(sizes), 1),
            "effective_gb_s_range": [round(min(eff), 1), round(max(eff), 1)],
            "utilisation_range": [round(min(us), 3), round(max(us), 3)],
            "utilisation_pct_range": [round(min(us) * 100, 1),
                                      round(max(us) * 100, 1)],
            "utilisation_median": round(statistics.median(us), 3),
            "utilisation_pct_median": round(statistics.median(us) * 100, 1),
            "spread_ratio": round(max(eff) / min(eff), 2),
        }
    if whole:
        # A dense build cannot read more bytes per token than its weight file
        # holds, so anything above 1.0 here is measuring the proxy's error and
        # not the hardware. Reported as a first-class quantity: it is the
        # tightest available bound on how far tau * W can be trusted.
        over = [r for r in whole if r["utilisation"] > 1.0]
        res["proxy_overcount"] = {
            "dense_rows_above_ceiling": len(over),
            "models": [{"model": r["model"],
                        "architecture": r["architecture"],
                        "utilisation_pct": r["utilisation_pct"],
                        "multimodal_tower": r["multimodal_tower"],
                        "redistributable": r["redistributable"]}
                       for r in over],
            "max_overcount_pct": (round((max(r["utilisation"]
                                             for r in over) - 1) * 100, 1)
                                  if over else 0.0),
            "note": (
                "tau * on-disk-bytes counts every byte the file holds, not "
                "the bytes decode streams. A vision tower, an embedding "
                "table and quantisation metadata are all in the file and "
                "none is read once per token, so the statistic runs high by "
                "a build-dependent margin. Where a dense row exceeds 100% "
                "that margin is directly visible, which is why these rows "
                "are reported rather than dropped: they are the only "
                "calibration this proxy has."
            ),
        }
    if whole and partial:
        pu = [r["utilisation"] for r in partial]
        overlap = max(us) >= min(pu)
        res["separation"] = {
            "highest_whole_weight_set": round(max(us), 3),
            "lowest_partial_reader": round(min(pu), 3),
            "highest_whole_weight_set_pct": round(max(us) * 100, 1),
            "lowest_partial_reader_pct": round(min(pu) * 100, 1),
            "separated": not overlap,
            "gap_pp": round((min(pu) - max(us)) * 100, 1),
            "note": (
                "The gap is measured from the HIGHEST dense row, including "
                "the one that exceeds the ceiling. Quoting it from the "
                "highest redistributable dense row instead would widen it "
                "to 45 pp and would be selection on the outcome."
            ),
        }
        # The ordering claim is what survives: partial readers sit above whole
        # readers by more than the proxy's own error. The tight-band claim
        # does not, and is no longer part of the verdict.
        res["verdict"] = ("ORDERING HOLDS; MAGNITUDE PROXY-LIMITED"
                          if not overlap else "INCONSISTENT")

    def _group(rows, note):
        if not rows:
            return None
        us = [r["utilisation"] for r in rows]
        return {
            "models": len(rows),
            "utilisation_range": [round(min(us), 3), round(max(us), 3)],
            "utilisation_pct_range": [round(min(us) * 100, 1),
                                      round(max(us) * 100, 1)],
            "all_above_peak": all(u > 1.0 for u in us),
            "note": note,
        }

    res["sparse"] = _group(sparse, (
        "A mixture of experts reads only its active experts per token, so "
        "weight-size-based effective bandwidth overstates the traffic and "
        "must land far above the ceiling. That is the predicted signature, "
        "and it is what makes the statistic a test rather than a "
        "tautology."))
    res["elastic"] = _group(elastic, (
        "A per-layer-embedding build declares more parameters than it "
        "reads per token, because only the embedding rows for the current "
        "token are fetched. It is grouped with the other partial readers "
        "on the evidence, which is that it exceeds a ceiling a whole-file "
        "reader cannot."))
    res["mtp"] = _group(mtp, (
        "A multi-token-prediction head emits more than one token per "
        "forward pass when its draft is accepted, so tokens per second is "
        "no longer passes per second and the product may exceed the "
        "ceiling. Reported separately for the same reason as the sparse "
        "rows."))
    res = {k: v for k, v in res.items() if v is not None}
    res["note"] = (
        "File-size times token rate is a proxy, not a memory-traffic "
        "measurement, and this ladder now bounds its own error rather than "
        "asserting the error is small. Peak bandwidth is a part "
        "specification used only as a ceiling to express utilisation "
        "against; it is not measured here. Weight bytes are the on-disk "
        "quantised file, which OVERSTATES the bytes decode streams by a "
        "build-dependent margin: vision towers, embedding and output tables "
        "and quantisation metadata all sit in the file and none is read once "
        "per token. The margin is at least 5.7 percentage points, because a "
        "dense build measures above a ceiling it cannot exceed, and it is "
        "plausibly larger on small models with large vocabularies - qwen3:4b "
        "reads 63.1% on this statistic. KV traffic biases the other way and "
        "is not modelled. An MTP head in metadata does not establish that "
        "speculative decoding ran on any particular request. Every measured "
        "row is reported, including the two whose weights are not "
        "redistributable; those rows are labelled, not dropped. Evaluate "
        "each machine separately; a machine with no partial-reader control "
        "cannot test separation at all."
    )
    if machine is None:
        res["verdict"] = "SEE PER-MACHINE RESULTS"
    elif not partial:
        res["verdict"] = "DESCRIPTIVE ONLY (no partial-reader control)"
    return res


# Power-supply conversion efficiency at the partial load these requests draw.
# 0.90 is a typical 80 PLUS Gold figure and is an ASSUMPTION, not a measurement:
# the harness has no wall meter. It is applied only to wall-socket scenarios and
# named in every row it touches.
PSU_EFFICIENCY = 0.90


def _telemetry_resolution(runs: List[Dict]) -> Dict[str, object]:
    """How much independent information is really in each power trace?

    nvidia-smi's power.draw.instant is read from a driver counter that
    refreshes more slowly than the harness polls it, so consecutive reads
    return byte-identical values in runs. Counting samples therefore
    overstates the resolution badly: a 1.6 s request returns 13 samples
    carrying 4-5 distinct readings. Reported so that significant figures in
    the paper can be justified against the instrument rather than the sample
    count.
    """
    samples, distinct, spans = [], [], []
    for r in runs:
        tr = r["power"].get("trace_s_w") or []
        if not tr:
            continue
        vals = [p for _, p in tr]
        samples.append(len(vals))
        distinct.append(len(set(vals)))
        spans.append(r["power"].get("window_s") or 0.0)
    if not samples:
        return {"available": False}
    med_d = statistics.median(distinct)
    med_span = statistics.median(spans)
    return {
        "available": True,
        "samples_per_run_median": statistics.median(samples),
        "distinct_readings_per_run_median": med_d,
        "distinct_readings_range": [min(distinct), max(distinct)],
        "implied_counter_refresh_s": (round(med_span / med_d, 3)
                                      if med_d else None),
        "note": ("Distinct readings, not sample count, bound the precision of "
                 "the integral. Two significant figures is what these traces "
                 "support; the paper reports two."),
    }


def energy_table() -> List[Dict[str, object]]:
    """Measured request-window GPU energy; host-power scenarios stay separate."""
    rows = []
    for d in _load_all():
        idle = d.get("idle_power", {})
        for m in d.get("models", []):
            runs = [r for r in m.get("runs", []) if
                    r.get("power", {}).get("measured") and
                    r["power"].get("trace_s_w")]
            if not runs:
                continue
            energies = [r["power"]["energy_wh"] for r in runs]
            windows = [r["power"]["window_s"] for r in runs]
            row = {
                "file": d["_file"], "machine": d["system"].get("gpu"),
                "model": m["model"], "reps": len(runs),
                "prompt_tokens": statistics.median(r["prompt_tokens"] for r in runs),
                "output_tokens": statistics.median(r["eval_tokens"] for r in runs),
                "gen_tok_s": m["median_gen_tok_s"],
                "energy_wh": statistics.median(energies),
                "energy_wh_min": min(energies), "energy_wh_max": max(energies),
                "window_s": statistics.median(windows),
                "mean_w": statistics.median(r["power"]["mean_w"] for r in runs),
                "mwh_per_output_token": statistics.median(
                    r["power"]["energy_wh"] * 1000 / r["eval_tokens"] for r in runs),
                "idle_board_w": idle.get("mean_w") if idle.get("measured") else None,
                # Wall-socket scenarios. The "0 W" row is the board rail
                # itself and is NOT divided by PSU efficiency, because it is
                # reported as a board measurement rather than a wall figure.
                # Every host scenario is, because a host draw only exists at
                # the wall. An earlier version wrote a loss term into the
                # paper's system equation and then evaluated every scenario
                # with it set to zero, understating wall energy by ~11% in
                # precisely the comparison against a hyperscaler's
                # comprehensive per-prompt number where it matters.
                "psu_efficiency": PSU_EFFICIENCY,
                "host_power_scenarios_wh": {
                    str(w): statistics.median(
                        (e + w*t/3600) / (1.0 if w == 0 else PSU_EFFICIENCY)
                        for e, t in zip(energies, windows))
                    for w in (0, 30, 60, 100)},
                # The driver refreshes power.draw.instant far more slowly than
                # the harness polls it, so the sample count overstates the
                # independent information in each trace. Carried per row so no
                # reader has to infer the instrument's resolution from the
                # sample count alone.
                "telemetry_resolution": _telemetry_resolution(runs),
                "scope": ("sampled GPU board energy during request. The 0 W row is "
                          "the board rail alone: host CPU/DRAM, cold load and the "
                          "post-request idle tail are outside it. Host scenarios add "
                          "host draw and charge PSU conversion loss at "
                          f"eta={PSU_EFFICIENCY}; the idle tail remains excluded "
                          "because it is a duty-cycle property, not a per-query one"),
            }
            if idle.get("measured"):
                row["idle_subtracted_board_wh"] = statistics.median(
                    max(0, e - idle["mean_w"]*t/3600)
                    for e, t in zip(energies, windows))
            rows.append(row)
    return rows


def model_form_check(assumed_power_w: Optional[float] = None,
                     assumed_tps: Optional[float] = None,
                     assumed_beta: Optional[float] = None,
                     ) -> Dict[str, object]:
    """A BOOKKEEPING CONSISTENCY CHECK on the edge-energy equation.

        E = P * (n_o / tau) * (1 + beta) / 3600

    Read what this can and cannot show, because an earlier version of this
    docstring - and of the paper - called it a falsification test the equation
    survived. It is not one.

    Mean request power is DEFINED as P_bar = 3600 * E_GPU / T, so
    E_GPU == P_bar * T / 3600 identically. Throughput is DEFINED as
    tau = n_o / t_decode, and beta as t_prefill / t_decode. Substituting all
    three into the equation above reduces the comparison to

        t_prefill + t_decode  ==?  T

    which is an accounting identity up to the fixed per-request overhead. The
    residual reported here IS that overhead - a few milliseconds over a window
    of seconds - which is why every residual is negative and why none could
    have been large. Passing carries no evidence about whether per-query energy
    really is mean power times occupancy inflated by prefill. The functional
    form remains an assumption.

    What the check does establish is worth having and is narrower: that no
    substantial energy is spent inside the request window outside prefill and
    decode, that the timing and power instruments agree with each other, and
    that no transcription error separates the two.

    The INPUTS are a genuinely different matter. Each assumed term is compared
    against an independent measurement of the same quantity, which can fail and
    did: all three assumed terms are wrong, two in the direction that flatters
    the thesis. That comparison, not the closure residual, is where the
    information is.
    """
    by_model = {}
    for row in prefill_scaling().get("rows", []):
        if row.get("beta_at_500_in_300_out") is not None:
            by_model[(row["machine"], row["model"])] = row

    rows = []
    for r in energy_table():
        key = (r["machine"], r["model"])
        fit = by_model.get(key)
        if not fit or not r.get("mean_w") or not r.get("gen_tok_s"):
            continue
        # beta is a function of the query shape, not a constant, so it is
        # evaluated from the fitted prefill line at THIS row's own prompt
        # and output lengths. Reusing a beta quoted at some other shape
        # is the same category error the prefill sweep exists to expose.
        rate = fit.get("marginal_prefill_tok_s")
        fixed_s = (fit.get("fixed_overhead_ms") or 0.0) / 1000.0
        if not rate:
            continue
        decode_s = r["output_tokens"] / r["gen_tok_s"]
        beta = (fixed_s + r["prompt_tokens"] / rate) / decode_s
        occupancy_s = decode_s * (1.0 + beta)
        predicted = r["mean_w"] * occupancy_s / 3600.0
        measured = r["energy_wh"]
        rows.append({
            "machine": r["machine"],
            "model": r["model"],
            "file": r["file"],
            "measured_power_w": round(r["mean_w"], 2),
            "measured_gen_tok_s": round(r["gen_tok_s"], 2),
            "prompt_tokens": r["prompt_tokens"],
            "beta_at_this_shape": round(beta, 4),
            "output_tokens": r["output_tokens"],
            "predicted_wh": round(predicted, 5),
            "measured_wh": round(measured, 5),
            "relative_error": round(predicted / measured - 1.0, 4),
            "relative_error_pct": round((predicted / measured - 1.0) * 100, 2),
        })
    if not rows:
        return {"available": False,
                "reason": "needs a machine with both a power sensor and a "
                          "prefill sweep"}

    errs = [abs(r["relative_error"]) for r in rows]
    out: Dict[str, object] = {
        "available": True,
        "equation": "E = P * (n_o / tau) * (1 + beta) / 3600",
        "rows": rows,
        "n": len(rows),
        "max_abs_relative_error": round(max(errs), 4),
        "max_abs_relative_error_pct": round(max(errs) * 100, 2),
        "all_predictions_low": all(r["relative_error"] < 0 for r in rows),
        "verdict": ("TERMS CONSISTENT" if max(errs) <= 0.05
                    else "TERMS INCONSISTENT"),
        "what_this_tests": (
            "Bookkeeping consistency, NOT model validity. Because P_bar, tau "
            "and beta are each defined in terms of the same measured request, "
            "substituting them reduces this comparison to the identity "
            "t_prefill + t_decode == T. The residual is the fixed per-request "
            "overhead the equation omits, so the check cannot fail by more "
            "than that term and passing it says nothing about whether the "
            "functional form is correct. The form remains an assumption."),
        "note": ("Every term on the right-hand side is measured on the same "
                 "request as the energy on the left. The residual is negative "
                 "on every row, which is the direction the omitted fixed "
                 "per-request cost predicts."),
    }

    # The inputs are a separate question from the form, and get a separate
    # answer: each assumed term against what the same quantity measures.
    if None not in (assumed_power_w, assumed_tps, assumed_beta):
        # The 8B row, because that is the model class the study's own
        # assumed terms were chosen to describe.
        ref = next((r for r in rows if "8b" in r["model"]), rows[0])
        terms = [
            ("board power P (W)", assumed_power_w, ref["measured_power_w"]),
            ("throughput tau (tok/s)", assumed_tps, ref["measured_gen_tok_s"]),
            ("prefill factor beta", assumed_beta,
             ref["beta_at_this_shape"]),
        ]
        out["assumed_vs_measured"] = {
            "reference_row": {"machine": ref["machine"], "model": ref["model"]},
            "terms": [
                {"term": name, "assumed": a, "measured": m,
                 "assumed_over_measured": round(a / m, 2),
                 # A term that inflates E when overstated pushes the modelled
                 # energy up; tau is in the denominator and pushes it down.
                 "direction_on_E": "raises" if name.startswith(("board", "prefill"))
                                   else "lowers"}
                for name, a, m in terms
            ],
            "note": ("The assumed inputs were wrong in compensating "
                     "directions: an overstated board power and an overstated "
                     "prefill factor both inflate the modelled energy, while "
                     "an overstated throughput deflates it. Their product is "
                     "closer to the measurement than any single term is, which "
                     "is why an unvalidated model can look accurate for the "
                     "wrong reasons."),
        }
    return out


def summary() -> Dict[str, object]:
    mach = machines()
    return {
        "available": bool(mach),
        "machines": mach,
        "throughput": throughput_table(),
        "beta": measured_beta(),
        "prefill_scaling": prefill_scaling(),
        "bandwidth_bound_check": bandwidth_bound_check(),
        "bandwidth_law": bandwidth_law(),
        "model_form_check": model_form_check(),
        "scaling": scaling_check(),
        "energy": energy_table(),
        "power_measured_anywhere": bool(energy_table()),
        "scope": (
            "One serving stack (Ollama) in its default configuration, "
            "single-stream, on the authors' own machines. Not a controlled "
            "comparison of accelerator architectures; quantisation, context "
            "length and driver version are recorded with every figure because "
            "they move it."
        ),
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    s = summary()
    if not s["available"]:
        print("no measurement artifacts in", RESULTS_DIR)
        print("run: python measure/bench.py")
        raise SystemExit(0)

    print(f"{len(s['machines'])} machine(s) measured\n")
    for m in s["machines"]:
        print(f"  {m['gpu']}")
        print(f"      {m['ram_gb']} GB | {m['device_class']} | "
              f"ollama {m['ollama_version']} | "
              f"power sensor: {m['power_sensor_tool'] or 'NONE'}")

    print(f"\n{'model':<18}{'params':>8}{'quant':>9}{'gen tok/s':>11}"
          f"{'prefill tok/s':>15}{'beta':>8}")
    print("-" * 72)
    for r in s["throughput"]:
        print(f"{r['model']:<18}{str(r['parameters']):>8}"
              f"{str(r['quantization']):>9}{r['gen_tok_s']:>11.1f}"
              f"{(r['prefill_tok_s'] or 0):>15.1f}{(r['beta'] or 0):>8.4f}")

    b = s["beta"]
    if b["available"]:
        print(f"\nbeta: measured {b['min']}-{b['max']} (median {b['median']}), "
              f"study assumed {b['assumed_in_study']}")

    bw = s["bandwidth_bound_check"]
    if bw["available"]:
        print(f"\nbandwidth-bound premise: throughput ratio "
              f"{bw['throughput_ratio']}x vs bandwidth ratio "
              f"{bw['bandwidth_ratio']}x -> {bw['verdict']} "
              f"(agreement {bw['agreement']})")

    ps = s["prefill_scaling"]
    if ps["available"]:
        print("\nprefill, length-controlled (prefill_s = fixed + tokens/rate):")
        for r in ps["rows"]:
            print(f"  {r['model']:<18} marginal {r['marginal_prefill_tok_s']:>8.1f} tok/s"
                  f"   fixed {r['fixed_overhead_ms']:>6.1f} ms"
                  f"   R2={r['r2']:.4f}"
                  f"   beta(500 in/300 out)={r['beta_at_500_in_300_out']}")
        bt = ps.get("beta_at_study_typical_query")
        if bt:
            print(f"  beta at the study's own typical query: "
                  f"{bt['min']}-{bt['max']} (median {bt['median']}), "
                  f"study assumed 0.15")

    bl = s["bandwidth_law"]
    if bl["available"]:
        print("\nbandwidth ladder (one machine, weight size vs throughput):")
        print(f"  {'model':<24}{'GB':>7}{'tok/s':>9}{'GB/s':>9}{'util':>8}  kind")
        for r in sorted(bl["rows"], key=lambda x: x["weights_gb"]):
            print(f"  {r['model']:<24}{r['weights_gb']:>7.2f}"
                  f"{r['gen_tok_s']:>9.2f}{r['effective_read_gb_s']:>9.1f}"
                  f"{r['utilisation']*100:>7.0f}%  "
                  f"{r['decode_class']}"
                  f"{' + projector' if r['multimodal_tower'] else ''}")
        d = bl.get("dense")
        if d:
            print(f"  whole weight set: {d['models']} models spanning "
                  f"{d['weight_span']}x in size land at "
                  f"{d['utilisation_range'][0]*100:.0f}-"
                  f"{d['utilisation_range'][1]*100:.0f}% of the "
                  f"{bl['rows'][0]['peak_bandwidth_gb_s']:.0f} GB/s ceiling")
        sep = bl.get("separation")
        if sep:
            print(f"  separation: highest whole-file reader "
                  f"{sep['highest_whole_weight_set']*100:.0f}% vs lowest "
                  f"partial reader {sep['lowest_partial_reader']*100:.0f}% "
                  f"-> {'SEPARATED' if sep['separated'] else 'OVERLAP'} "
                  f"by {sep['gap_pp']} pp -> {bl['verdict']}")
        sp = bl.get("sparse")
        if sp:
            print(f"  sparse: {sp['models']} model(s) at "
                  f"{sp['utilisation_range'][0]*100:.0f}-"
                  f"{sp['utilisation_range'][1]*100:.0f}% "
                  f"(above the ceiling, as predicted for MoE)")

    print(f"\npower measured anywhere: {s['power_measured_anywhere']}")
