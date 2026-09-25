"""Run-level uncertainty and replication analysis for measurement records.

This module deliberately imports no research-model code. It consumes only the
raw JSON records in measure/results and recomputes summaries from individual
repetitions. The bootstrap intervals describe repeatability on the measured
machine; they are not population intervals for all consumer hardware.

Examples:

    python measure/statistics.py
    python measure/statistics.py --resamples 20000
    python measure/statistics.py --input measure/results --output measure/statistics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# When this file is executed directly, Python puts measure/ ahead of the
# standard-library search path and would import this file as statistics.
if __package__ in (None, ""):
    _MODULE_DIR = str(Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == _MODULE_DIR:
        sys.path.pop(0)

import statistics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "measure" / "results"
DEFAULT_OUTPUT = ROOT / "measure" / "statistics.json"
DEFAULT_MANIFEST = ROOT / "measure" / "replication_manifest.json"
DEFAULT_RESAMPLES = 10000
IDENTITY_FIELDS = ("target_id", "operator_id", "environment_id")
EXACT_SIGN_FLIP_MAX_N = 16


def _display_path(path: Path) -> str:
    """Keep generated reports portable instead of embedding a local path."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _finite(values: Iterable[object]) -> List[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated percentile, with a documented small-n behavior."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _estimate(values: Sequence[float], statistic: str) -> float:
    if statistic == "mean":
        return statistics.mean(values)
    if statistic == "median":
        return statistics.median(values)
    raise ValueError(f"unsupported statistic: {statistic}")


def bootstrap_ci(values: Sequence[float], statistic: str = "median",
                 resamples: int = DEFAULT_RESAMPLES,
                 seed: int = 20260914) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval over independent repetitions."""
    values = _finite(values)
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    if resamples < 100:
        raise ValueError("use at least 100 bootstrap resamples")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(_estimate(sample, statistic))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize(values: Sequence[float], statistic: str = "median",
              resamples: int = DEFAULT_RESAMPLES,
              seed: int = 20260914) -> Dict[str, object]:
    """Return descriptive statistics plus a run-level bootstrap interval."""
    values = _finite(values)
    if not values:
        return {"n": 0, "statistic": statistic}
    low, high = bootstrap_ci(values, statistic, resamples=resamples, seed=seed)
    mean = statistics.mean(values)
    return {
        "n": len(values),
        "statistic": statistic,
        "estimate": _estimate(values, statistic),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "iqr": percentile(values, 0.75) - percentile(values, 0.25),
        "coefficient_of_variation": (
            statistics.stdev(values) / mean
            if len(values) > 1 and mean != 0 else None),
        "min": min(values),
        "max": max(values),
        "bootstrap_95_ci": {
            "method": "percentile bootstrap over repetitions",
            "low": low,
            "high": high,
            "resamples": resamples,
            "seed": seed,
        },
        "interpretation": (
            "Repeatability interval for this recorded machine and protocol; "
            "not a confidence interval for the device population."
        ),
    }


def _metric_values(model: Dict[str, object]) -> Dict[str, List[float]]:
    """Extract measurements from one single-request model record."""
    throughput = []
    wall = []
    power = []
    energy = []
    for run in model.get("runs", []) or []:
        throughput.extend(_finite([run.get("gen_tok_s")]))
        wall.extend(_finite([run.get("wall_s")]))
        p = run.get("power") or {}
        power.extend(_finite([p.get("mean_w")]))
        energy.extend(_finite([p.get("energy_wh")]))
    return {
        "generation_tok_s": throughput,
        "request_wall_s": wall,
        "mean_board_power_w": power,
        "board_energy_wh": energy,
    }


def _batch_metric_values(model: Dict[str, object]) -> Dict[str, List[float]]:
    """Extract batch-window values from one concurrent model record."""
    throughput = []
    wall = []
    energy = []
    request_wall = []
    success_fraction = []
    for batch in model.get("batches", []) or []:
        expected = int(
            batch.get("batch_size_requested") or model.get("batch_size") or 0)
        requests = [request for request in batch.get("requests", []) or []
                    if isinstance(request, dict)]
        if expected > 0:
            success_fraction.append(
                sum(bool(request.get("ok")) for request in requests)
                / expected)
        if not batch.get("complete"):
            continue
        throughput.extend(_finite([batch.get("aggregate_output_tok_s")]))
        wall.extend(_finite([batch.get("batch_wall_s")]))
        allocation = batch.get("energy") or {}
        if allocation.get("measured"):
            energy.extend(_finite([
                allocation.get("energy_wh_per_completed_request")]))
        for request in batch.get("requests", []) or []:
            if request.get("ok"):
                request_wall.extend(_finite([request.get("wall_s")]))
    return {
        "aggregate_output_tok_s": throughput,
        "batch_wall_s": wall,
        "equal_split_energy_wh_per_request": energy,
        "request_wall_s": request_wall,
        "request_success_fraction": success_fraction,
    }


def _batch_quality(model: Dict[str, object]) -> Dict[str, object]:
    """Count batch and request outcomes without treating missing work as success."""
    total_batches = 0
    complete_batches = 0
    expected_requests = 0
    observed_requests = 0
    successful_requests = 0
    for batch in model.get("batches", []) or []:
        if not isinstance(batch, dict):
            continue
        total_batches += 1
        complete_batches += int(bool(batch.get("complete")))
        try:
            expected = int(
                batch.get("batch_size_requested")
                or model.get("batch_size") or 0)
        except (TypeError, ValueError):
            expected = 0
        expected_requests += max(expected, 0)
        requests = [request for request in batch.get("requests", []) or []
                    if isinstance(request, dict)]
        observed_requests += len(requests)
        successful_requests += sum(bool(request.get("ok"))
                                   for request in requests)
    failed_or_missing = max(expected_requests - successful_requests, 0)
    return {
        "batch_repetitions": total_batches,
        "complete_batch_repetitions": complete_batches,
        "incomplete_batch_repetitions": total_batches - complete_batches,
        "complete_batch_fraction": (
            complete_batches / total_batches if total_batches else None),
        "requested_requests": expected_requests,
        "observed_requests": observed_requests,
        "successful_requests": successful_requests,
        "failed_or_missing_requests": failed_or_missing,
        "request_success_fraction": (
            successful_requests / expected_requests
            if expected_requests else None),
        "interpretation": (
            "Counts are computed from raw request records. Missing or failed "
            "requests are not treated as zero-cost successes; incomplete "
            "batches are excluded from complete-batch throughput summaries."
        ),
    }


def _load_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _is_batched(data: Dict[str, object]) -> bool:
    return bool(data.get("batches")) or "batch-sizes" in str(data.get("harness", ""))


def _is_measurement_record(data: Dict[str, object]) -> bool:
    """Return whether a JSON object contains raw serving measurements."""
    if _is_batched(data):
        return any(
            isinstance(record, dict)
            and isinstance(record.get("batches"), list)
            for record in data.get("batches", []) or [])
    return any(
        isinstance(record, dict)
        and isinstance(record.get("runs"), list)
        for record in data.get("models", []) or [])


def _model_records(data: Dict[str, object], batched: bool) -> List[Dict[str, object]]:
    """Return the top-level model records for either raw schema variant."""
    key = "batches" if batched else "models"
    return [record for record in (data.get(key, []) or [])
            if isinstance(record, dict)]


def _batch_scaling(models: Sequence[Dict[str, object]],
                   resamples: int) -> List[Dict[str, object]]:
    """Compare each batch size with batch 1 using paired repetition IDs."""
    by_model: Dict[str, Dict[int, Dict[str, object]]] = {}
    for model in models:
        name = str(model.get("model", "unknown"))
        try:
            batch_size = int(model.get("batch_size"))
        except (TypeError, ValueError):
            continue
        by_model.setdefault(name, {})[batch_size] = model

    rows = []
    for model_name, by_batch in sorted(by_model.items()):
        baseline = by_batch.get(1)
        if not baseline:
            continue
        baseline_values = {
            int(batch.get("rep")): float(batch["aggregate_output_tok_s"])
            for batch in baseline.get("batches", []) or []
            if batch.get("complete") and batch.get("aggregate_output_tok_s")
        }
        for batch_size, model in sorted(by_batch.items()):
            current_values = {
                int(batch.get("rep")): float(batch["aggregate_output_tok_s"])
                for batch in model.get("batches", []) or []
                if batch.get("complete") and batch.get("aggregate_output_tok_s")
            }
            paired_reps = sorted(set(baseline_values) & set(current_values))
            if not paired_reps:
                continue
            throughput = [current_values[rep] for rep in paired_reps]
            speedup = [current_values[rep] / baseline_values[rep]
                       for rep in paired_reps]
            paired_difference = [
                current_values[rep] - baseline_values[rep]
                for rep in paired_reps]
            per_request = [value / batch_size for value in throughput]
            efficiency = [value / batch_size for value in speedup]
            seed_base = 600000 + len(rows) * 10
            rows.append({
                "model": model_name,
                "batch_size": batch_size,
                "paired_repetitions": len(paired_reps),
                "paired_repetition_ids": paired_reps,
                "aggregate_output_tok_s": summarize(
                    throughput, resamples=resamples, seed=seed_base),
                "speedup_vs_batch_1": summarize(
                    speedup, resamples=resamples, seed=seed_base + 1),
                "paired_difference_vs_batch_1": _paired_difference_summary(
                    paired_difference, resamples=resamples, seed=seed_base + 4),
                "per_request_output_tok_s": summarize(
                    per_request, resamples=resamples, seed=seed_base + 2),
                "parallel_efficiency": summarize(
                    efficiency, resamples=resamples, seed=seed_base + 3),
                "interpretation": (
                    "Paired within-file comparison using the same repetition IDs. "
                    "It describes this target and serving runtime; it is not a "
                    "cross-device or energy-efficiency estimate."
                ),
            })
    return rows


def _load_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return _load_json(path)


def _provenance(path: Path, manifest: Dict[str, object]) -> Dict[str, object]:
    entry = (manifest.get("files", {}) or {}).get(path.name)
    if not isinstance(entry, dict):
        return {
            "status": "unresolved",
            "reason": "raw file is not listed in replication_manifest.json",
        }
    return {"status": "declared", **entry}


def analyze_file(path: Path, resamples: int = DEFAULT_RESAMPLES,
                 manifest: Optional[Dict[str, object]] = None
                 ) -> Dict[str, object]:
    """Analyze one raw record without consulting models/results.json."""
    manifest = manifest or {}
    raw = path.read_bytes()
    data = _load_json(path)
    kind = "concurrent_batches" if _is_batched(data) else "single_requests"
    models = []
    for model in _model_records(data, kind == "concurrent_batches"):
        name = str(model.get("model", "unknown"))
        if kind == "single_requests":
            raw_metrics = _metric_values(model)
        else:
            raw_metrics = _batch_metric_values(model)
        metrics = {
            key: summarize(values, resamples=resamples,
                           seed=20260914 + index)
            for index, (key, values) in enumerate(raw_metrics.items())
            if values
        }
        model_result = {
            "model": name,
            "ok": bool(model.get("ok")),
            "batch_size": model.get("batch_size"),
            "complete_reps": model.get("complete_reps", model.get("reps")),
            "metrics": metrics,
        }
        if kind == "concurrent_batches":
            model_result["batch_quality"] = _batch_quality(model)
        models.append(model_result)
    result = {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "kind": kind,
        "system": data.get("system"),
        "protocol": data.get("protocol"),
        "models": models,
        "provenance": _provenance(path, manifest),
    }
    if kind == "concurrent_batches":
        result["batch_scaling"] = _batch_scaling(
            _model_records(data, True), resamples)
    return result


def _values_for_file(path: Path, model_name: str, metric: str,
                     batch_size: Optional[int] = None) -> List[float]:
    """Read raw values for the independent replication comparison."""
    data = _load_json(path)
    for model in _model_records(data, _is_batched(data)):
        if model.get("model") != model_name:
            continue
        if _is_batched(data):
            if batch_size is not None:
                try:
                    if int(model.get("batch_size")) != batch_size:
                        continue
                except (TypeError, ValueError):
                    continue
            raw = _batch_metric_values(model)
        else:
            raw = _metric_values(model)
        return raw.get(metric, [])
    return []


def _permutation_pvalue(a: Sequence[float], b: Sequence[float],
                        resamples: int, seed: int) -> float:
    """Two-sided randomization p-value for a difference in medians."""
    a = _finite(a)
    b = _finite(b)
    if not a or not b:
        return 1.0
    observed = abs(statistics.median(a) - statistics.median(b))
    combined = list(a) + list(b)
    n_a = len(a)
    rng = random.Random(seed)
    at_least = 0
    for _ in range(resamples):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        delta = abs(statistics.median(shuffled[:n_a])
                    - statistics.median(shuffled[n_a:]))
        if delta >= observed:
            at_least += 1
    return (at_least + 1) / (resamples + 1)


def _paired_sign_flip_test(values: Sequence[float], resamples: int,
                           seed: int) -> Dict[str, object]:
    """Test a paired median contrast with exact sign flips when feasible."""
    values = _finite(values)
    if not values:
        return {
            "method": "not computed: no finite paired differences",
            "p_two_sided": 1.0,
            "n": 0,
        }
    observed = abs(statistics.median(values))
    n = len(values)
    if n <= EXACT_SIGN_FLIP_MAX_N:
        total = 1 << n
        at_least = 0
        for mask in range(total):
            signed = [
                value if (mask >> index) & 1 else -value
                for index, value in enumerate(values)
            ]
            if abs(statistics.median(signed)) >= observed:
                at_least += 1
        return {
            "method": (
                "exact two-sided sign-flip randomization over paired "
                "repetition differences"),
            "p_two_sided": at_least / total,
            "n": n,
            "enumerated_sign_patterns": total,
        }
    rng = random.Random(seed)
    at_least = 0
    for _ in range(resamples):
        signed = [value if rng.getrandbits(1) else -value
                  for value in values]
        if abs(statistics.median(signed)) >= observed:
            at_least += 1
    return {
        "method": (
            "Monte Carlo two-sided sign-flip randomization over paired "
            "repetition differences"),
        "p_two_sided": (at_least + 1) / (resamples + 1),
        "n": n,
        "resamples": resamples,
        "seed": seed,
    }


def _paired_difference_summary(values: Sequence[float], resamples: int,
                               seed: int) -> Dict[str, object]:
    """Summarize a contrast computed after pairing on repetition identity."""
    result = summarize(values, resamples=resamples, seed=seed)
    interval = result.get("bootstrap_95_ci")
    if isinstance(interval, dict):
        interval["method"] = (
            "paired percentile bootstrap over repetition-level differences")
    result["paired_sign_flip_test"] = _paired_sign_flip_test(
        values, resamples=resamples, seed=seed + 1)
    result["interpretation"] = (
        "Paired contrast for the recorded target and protocol; the interval "
        "and sign-flip test do not estimate a device-population effect."
    )
    return result


def _bootstrap_difference(a: Sequence[float], b: Sequence[float],
                          resamples: int, seed: int) -> Tuple[float, float]:
    """Bootstrap CI for median(a) - median(b), resampling groups separately."""
    a = _finite(a)
    b = _finite(b)
    if not a or not b:
        raise ValueError("both groups need values")
    if len(a) == 1 and len(b) == 1:
        delta = a[0] - b[0]
        return delta, delta
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        sa = [a[rng.randrange(len(a))] for _ in a]
        sb = [b[rng.randrange(len(b))] for _ in b]
        deltas.append(statistics.median(sa) - statistics.median(sb))
    return percentile(deltas, 0.025), percentile(deltas, 0.975)


def _independence_criteria(first: Dict[str, object],
                           second: Dict[str, object]) -> Dict[str, bool]:
    """Return the explicit identity checks required for independence."""
    declared = all(first.get(field) and second.get(field)
                   for field in IDENTITY_FIELDS)
    return {
        "different_target_id": (
            declared and first["target_id"] != second["target_id"]),
        "different_operator_id": (
            declared and first["operator_id"] != second["operator_id"]),
        "different_environment_id": (
            declared and first["environment_id"] != second["environment_id"]),
    }


def _manifest_raw_paths(input_dir: Path,
                        manifest: Dict[str, object]) -> List[Path]:
    """Find manifest-listed raw records that contain measurable model rows."""
    paths = []
    entries = manifest.get("files", {}) or {}
    for path in sorted(input_dir.glob("*.json")):
        if path.name == DEFAULT_OUTPUT.name or path.name not in entries:
            continue
        data = _load_json(path)
        if _is_measurement_record(data):
            paths.append(path)
    return paths


def compare_independent_replications(
        input_dir: Path, resamples: int,
        manifest: Optional[Dict[str, object]] = None
        ) -> Dict[str, object]:
    """Compare every qualifying pair listed in the provenance manifest.

    A pair is eligible only when target, operator, and environment identities
    all differ. Same-target replays and cross-device runs by one operator are
    intentionally excluded from this independent result, even when their
    timings are numerically similar.
    """
    manifest = manifest or {}
    paths = _manifest_raw_paths(input_dir, manifest)
    candidates = []
    for first, second in combinations(paths, 2):
        first_provenance = _provenance(first, manifest)
        second_provenance = _provenance(second, manifest)
        criteria = _independence_criteria(first_provenance, second_provenance)
        if not all(criteria.values()):
            continue
        first_data = _load_json(first)
        second_data = _load_json(second)
        first_batched = _is_batched(first_data)
        second_batched = _is_batched(second_data)
        if first_batched != second_batched:
            continue
        if first_batched:
            first_models = {
                (model.get("model"), int(model.get("batch_size")))
                for model in _model_records(first_data, True)
                if model.get("model") is not None
                and model.get("batch_size") is not None
            }
            second_models = {
                (model.get("model"), int(model.get("batch_size")))
                for model in _model_records(second_data, True)
                if model.get("model") is not None
                and model.get("batch_size") is not None
            }
        else:
            first_models = {
                (model.get("model"), None)
                for model in _model_records(first_data, False)
            }
            second_models = {
                (model.get("model"), None)
                for model in _model_records(second_data, False)
            }
        rows = []
        metrics = (
            ("generation_tok_s", "board_energy_wh")
            if not first_batched else
            ("aggregate_output_tok_s", "batch_wall_s")
        )
        for model_name, batch_size in sorted(first_models & second_models):
            for metric in metrics:
                a = _values_for_file(
                    first, model_name, metric, batch_size=batch_size)
                b = _values_for_file(
                    second, model_name, metric, batch_size=batch_size)
                if not a or not b:
                    continue
                low, high = _bootstrap_difference(
                    a, b, resamples, 20400000 + len(rows))
                rows.append({
                    "model": model_name,
                    "metric": metric,
                    "batch_size": batch_size,
                    "first": summarize(
                        a, resamples=resamples, seed=20400100 + len(rows)),
                    "second": summarize(
                        b, resamples=resamples, seed=20400200 + len(rows)),
                    "first_minus_second_median": (
                        statistics.median(a) - statistics.median(b)),
                    "difference_bootstrap_95_ci": {
                        "low": low, "high": high, "resamples": resamples},
                    "permutation_p_two_sided": _permutation_pvalue(
                        a, b, resamples, 20400300 + len(rows)),
                })
        candidates.append({
            "first_file": first.name,
            "second_file": second.name,
            "classification": "independent_replication",
            "independent": True,
            "independence_criteria": criteria,
            "first_provenance": first_provenance,
            "second_provenance": second_provenance,
            "rows": rows,
            "available": bool(rows),
            "interpretation": (
                "Cross-target comparison whose target, operator and runtime "
                "environment identities all differ. It remains a comparison "
                "of the recorded targets, not a population estimate."
            ),
        })
    return {
        "available": any(pair["available"] for pair in candidates),
        "identity_fields": list(IDENTITY_FIELDS),
        "eligibility_rule": (
            "all identity fields must be present and different between files"
        ),
        "candidate_pair_count": len(candidates),
        "pairs": candidates,
        "reason": (
            None if candidates else
            "no pair has different target_id, operator_id and environment_id"
        ),
    }


def compare_replication(input_dir: Path, resamples: int,
                        manifest: Optional[Dict[str, object]] = None
                        ) -> Dict[str, object]:
    """Compare the recorded RTX study and replication runs if both exist."""
    manifest = manifest or {}
    primary = input_dir / "rtx5070-study.json"
    replication = input_dir / "rtx5070-replication.json"
    if not primary.exists() or not replication.exists():
        return {
            "available": False,
            "reason": (
                "requires rtx5070-study.json and "
                "rtx5070-replication.json"
            ),
        }
    primary_data = _load_json(primary)
    replication_data = _load_json(replication)
    primary_models = {
        model.get("model") for model in primary_data.get("models", []) or []}
    replication_models = {
        model.get("model")
        for model in replication_data.get("models", []) or []}
    rows = []
    for model_name in sorted(primary_models & replication_models):
        for metric in ("generation_tok_s", "board_energy_wh"):
            a = _values_for_file(primary, model_name, metric)
            b = _values_for_file(replication, model_name, metric)
            if not a or not b:
                continue
            low, high = _bootstrap_difference(a, b, resamples,
                                               20300000 + len(rows))
            rows.append({
                "model": model_name,
                "metric": metric,
                "primary": summarize(a, resamples=resamples,
                                     seed=20300100 + len(rows)),
                "replication": summarize(b, resamples=resamples,
                                         seed=20300200 + len(rows)),
                "primary_minus_replication_median": (
                    statistics.median(a) - statistics.median(b)),
                "difference_bootstrap_95_ci": {
                    "low": low,
                    "high": high,
                    "resamples": resamples,
                },
                "permutation_p_two_sided": _permutation_pvalue(
                    a, b, resamples, 20300300 + len(rows)),
                "interpretation": (
                    "Run-group agreement test on one nominal hardware target. "
                    "It is not independent-laboratory replication because "
                    "the repository does not establish a different operator, "
                    "machine or environment."
                ),
            })
    primary_provenance = _provenance(primary, manifest)
    replication_provenance = _provenance(replication, manifest)
    identities_declared = all(
        primary_provenance.get(field) and replication_provenance.get(field)
        for field in IDENTITY_FIELDS)
    independence_criteria = _independence_criteria(
        primary_provenance, replication_provenance)
    independent = all(independence_criteria.values())
    classification = (
        "independent_replication" if independent
        else "same_target_repeatability" if identities_declared
        else "unresolved_identity")
    return {
        "available": bool(rows),
        "primary_file": primary.name,
        "replication_file": replication.name,
        "classification": classification,
        "independent": independent,
        "independence_criteria": independence_criteria,
        "primary_provenance": primary_provenance,
        "replication_provenance": replication_provenance,
        "rows": rows,
        "same_machine_warning": (
            "The current replication record is a second run on the same named "
            "RTX 5070 target. It is repeatability evidence, not independent "
            "replication, until a separately operated machine contributes a "
            "record."
        ) if not independent else None,
    }


def analyze_directory(input_dir: Path, resamples: int = DEFAULT_RESAMPLES,
                      manifest_path: Path = DEFAULT_MANIFEST
                      ) -> Dict[str, object]:
    manifest = _load_manifest(manifest_path)
    paths = sorted(
        path for path in input_dir.glob("*.json")
        if path.name != DEFAULT_OUTPUT.name
        and not path.name.endswith("-ladder.json")
    )
    measurement_paths = []
    for path in paths:
        data = _load_json(path)
        if _is_measurement_record(data):
            measurement_paths.append(path)
    analyses = [analyze_file(path, resamples=resamples, manifest=manifest)
                for path in measurement_paths]
    named_replication = compare_replication(
        input_dir, resamples, manifest=manifest)
    all_replications = compare_independent_replications(
        input_dir, resamples, manifest=manifest)
    available_pairs = sum(
        bool(pair.get("available"))
        for pair in all_replications.get("pairs", []))
    return {
        "schema_version": "1.0",
        "analysis": (
            "Run-level bootstrap and replication statistics computed directly "
            "from raw measurement JSON files"
        ),
        "input_directory": _display_path(input_dir),
        "replication_manifest": _display_path(manifest_path),
        "resamples": resamples,
        "confidence_interval": (
            "Percentile bootstrap over repetitions. These intervals describe "
            "repeatability on each recorded machine, not population uncertainty."
        ),
        "measurement_files_considered": len(measurement_paths),
        "non_measurement_json_ignored": len(paths) - len(measurement_paths),
        "files": analyses,
        "independent_replication": named_replication,
        "independent_replications": all_replications,
        "independent_replication_status": (
            "available" if available_pairs else "pending_external_contribution"),
        "independent_replication_pairs_with_data": available_pairs,
        "limitations": [
            "No new hardware or third-party laboratory result is inferred.",
            "A small number of repetitions cannot estimate device-population variance.",
            "Equal-split batch energy is an allocation of a shared trace.",
            "The cloud baseline is not re-measured by this module.",
            "Paired sign-flip tests assume exchangeable signs under the null and "
            "do not create independent hardware replicates.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = parser.parse_args(argv)
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")
    result = analyze_directory(
        args.input, args.resamples, manifest_path=args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.output}")
    for row in result["files"]:
        print(f"  {row['file']}: {row['kind']} ({len(row['models'])} models)")
    comparison = result["independent_replication"]
    print(f"  replication comparison: "
          f"{comparison.get('classification', 'not available')}")
    independent = result["independent_replications"]
    print(f"  independent pairs: {independent['candidate_pair_count']} "
          f"qualifying, {len(independent['pairs'])} available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
