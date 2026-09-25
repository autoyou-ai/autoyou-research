"""Offline tests for the run-level statistical analysis."""

import json
import tempfile
import unittest
from pathlib import Path

from measure.statistics import (
    analyze_file,
    analyze_directory,
    bootstrap_ci,
    compare_independent_replications,
    compare_replication,
    percentile,
    summarize,
)


class StatisticsChecks(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_single_value_bootstrap_is_not_fake_precision(self):
        self.assertEqual(bootstrap_ci([4.0], resamples=100), (4.0, 4.0))

    def test_summary_exposes_repeatability_interval(self):
        result = summarize([1.0, 2.0, 3.0, 4.0],
                           resamples=200, seed=17)
        self.assertEqual(result["n"], 4)
        self.assertIn("bootstrap_95_ci", result)
        self.assertEqual(result["bootstrap_95_ci"]["resamples"], 200)
        self.assertEqual(result["interpretation"],
                         "Repeatability interval for this recorded machine and "
                         "protocol; not a confidence interval for the device "
                         "population.")
        self.assertAlmostEqual(result["p95"], 3.85)
        self.assertAlmostEqual(result["iqr"], 1.5)
        self.assertIsNotNone(result["coefficient_of_variation"])

    def test_analyze_file_uses_raw_repetitions(self):
        payload = {
            "harness": "test",
            "protocol": {"reps": 3},
            "system": {"machine": "synthetic"},
            "models": [{
                "model": "synthetic:3b",
                "ok": True,
                "runs": [
                    {"gen_tok_s": 10, "wall_s": 1,
                     "power": {"mean_w": 50, "energy_wh": 0.1}},
                    {"gen_tok_s": 12, "wall_s": 1,
                     "power": {"mean_w": 52, "energy_wh": 0.11}},
                    {"gen_tok_s": 11, "wall_s": 1,
                     "power": {"mean_w": 51, "energy_wh": 0.105}},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = analyze_file(path, resamples=200)
        metrics = result["models"][0]["metrics"]
        self.assertEqual(metrics["generation_tok_s"]["n"], 3)
        self.assertAlmostEqual(metrics["generation_tok_s"]["median"], 11.0)
        self.assertEqual(metrics["board_energy_wh"]["n"], 3)

    def test_analyze_file_reads_concurrent_batch_records(self):
        payload = {
            "harness": "measure/bench.py --batch-sizes",
            "protocol": {"reps": 2, "batch_sizes": [2]},
            "system": {"machine": "synthetic"},
            "batches": [{
                "model": "synthetic:3b",
                "batch_size": 2,
                "ok": True,
                "complete_reps": 2,
                "batches": [
                    {"complete": True, "aggregate_output_tok_s": 20,
                     "batch_wall_s": 2,
                     "requests": [{"ok": True, "wall_s": 1.8},
                                  {"ok": True, "wall_s": 2.0}]},
                    {"complete": True, "aggregate_output_tok_s": 24,
                     "batch_wall_s": 2.2,
                     "requests": [{"ok": True, "wall_s": 2.1},
                                  {"ok": True, "wall_s": 2.2}]},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-batched.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = analyze_file(path, resamples=200)
        self.assertEqual(result["kind"], "concurrent_batches")
        metrics = result["models"][0]["metrics"]
        self.assertEqual(metrics["aggregate_output_tok_s"]["n"], 2)
        self.assertAlmostEqual(
            metrics["aggregate_output_tok_s"]["median"], 22.0)
        self.assertEqual(metrics["request_wall_s"]["n"], 4)
        self.assertEqual(metrics["request_success_fraction"]["n"], 2)

    def test_batch_scaling_uses_paired_repetition_ids(self):
        def batches(rate):
            return [
                {"rep": 1, "complete": True,
                 "aggregate_output_tok_s": rate,
                 "batch_wall_s": 1.0, "requests":
                 [{"ok": True, "wall_s": 1.0}]},
                {"rep": 2, "complete": True,
                 "aggregate_output_tok_s": rate * 1.1,
                 "batch_wall_s": 1.0, "requests":
                 [{"ok": True, "wall_s": 1.0}]},
            ]
        payload = {
            "harness": "measure/bench.py --batch-sizes",
            "batches": [
                {"model": "synthetic:3b", "batch_size": 1,
                 "batches": batches(10)},
                {"model": "synthetic:3b", "batch_size": 2,
                 "batches": batches(20)},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired-batched.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = analyze_file(path, resamples=200)
        rows = result["batch_scaling"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["batch_size"], 2)
        self.assertEqual(rows[1]["paired_repetitions"], 2)
        self.assertAlmostEqual(
            rows[1]["speedup_vs_batch_1"]["estimate"], 2.0)
        paired = rows[1]["paired_difference_vs_batch_1"]
        self.assertEqual(paired["n"], 2)
        self.assertEqual(
            paired["bootstrap_95_ci"]["method"],
            "paired percentile bootstrap over repetition-level differences")
        self.assertEqual(
            paired["paired_sign_flip_test"]["method"],
            "exact two-sided sign-flip randomization over paired repetition "
            "differences")
        self.assertAlmostEqual(
            paired["paired_sign_flip_test"]["p_two_sided"], 0.5)

    def test_batched_quality_counts_failed_and_incomplete_requests(self):
        payload = {
            "harness": "measure/bench.py --batch-sizes",
            "batches": [{
                "model": "synthetic:3b",
                "batch_size": 2,
                "batches": [
                    {"rep": 1, "complete": True,
                     "aggregate_output_tok_s": 10,
                     "requests": [{"ok": True}, {"ok": False}]},
                    {"rep": 2, "complete": False,
                     "requests": [{"ok": True}]},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality-batched.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = analyze_file(path, resamples=200)
        quality = result["models"][0]["batch_quality"]
        self.assertEqual(quality["batch_repetitions"], 2)
        self.assertEqual(quality["complete_batch_repetitions"], 1)
        self.assertEqual(quality["incomplete_batch_repetitions"], 1)
        self.assertAlmostEqual(quality["complete_batch_fraction"], 0.5)
        self.assertEqual(quality["requested_requests"], 4)
        self.assertEqual(quality["observed_requests"], 3)
        self.assertEqual(quality["successful_requests"], 2)
        self.assertEqual(quality["failed_or_missing_requests"], 2)
        self.assertAlmostEqual(quality["request_success_fraction"], 0.5)

    def test_analyze_directory_ignores_non_measurement_json(self):
        measurement = {
            "harness": "test",
            "models": [{"model": "synthetic:3b", "runs": [
                {"gen_tok_s": 10.0, "wall_s": 1.0}
            ]}],
        }
        adaptation = {
            "schema_version": "1.0",
            "models": [{"model": "synthetic:3b", "scores": {}}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "measurement.json").write_text(
                json.dumps(measurement), encoding="utf-8")
            (root / "adaptation.json").write_text(
                json.dumps(adaptation), encoding="utf-8")
            result = analyze_directory(
                root, resamples=200,
                manifest_path=root / "missing-manifest.json")
        self.assertEqual(
            [row["file"] for row in result["files"]], ["measurement.json"])
        self.assertEqual(result["measurement_files_considered"], 1)
        self.assertEqual(result["non_measurement_json_ignored"], 1)
        self.assertEqual(result["independent_replication_status"],
                         "pending_external_contribution")

    def test_manifest_marks_same_target_as_repeatability(self):
        manifest = {
            "files": {
                "rtx5070-study.json": {
                    "target_id": "target-a",
                    "operator_id": "operator-a",
                    "environment_id": "env-a",
                },
                "rtx5070-replication.json": {
                    "target_id": "target-a",
                    "operator_id": "operator-a",
                    "environment_id": "env-a",
                },
            }
        }
        payload = {
            "models": [{"model": "m", "runs": [{
                "gen_tok_s": 1.0,
                "wall_s": 1.0,
                "power": {"mean_w": 1.0, "energy_wh": 1.0},
            }]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("rtx5070-study.json", "rtx5070-replication.json"):
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            result = compare_replication(root, 200, manifest=manifest)
        self.assertTrue(result["available"])
        self.assertFalse(result["independent"])
        self.assertEqual(result["classification"], "same_target_repeatability")

    def test_manifest_driven_independent_pair_requires_all_identities(self):
        manifest = {
            "files": {
                "first.json": {
                    "target_id": "synthetic-target-a",
                    "operator_id": "synthetic-operator-a",
                    "environment_id": "synthetic-environment-a",
                },
                "second.json": {
                    "target_id": "synthetic-target-b",
                    "operator_id": "synthetic-operator-b",
                    "environment_id": "synthetic-environment-b",
                },
            }
        }
        def payload(values):
            return {
                "models": [{"model": "synthetic:3b", "runs": [
                    {"gen_tok_s": value, "wall_s": 1.0,
                     "power": {"mean_w": 10.0, "energy_wh": 0.1}}
                    for value in values
                ]}],
            }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first.json").write_text(
                json.dumps(payload([10.0, 11.0])), encoding="utf-8")
            (root / "second.json").write_text(
                json.dumps(payload([12.0, 13.0])), encoding="utf-8")
            result = compare_independent_replications(
                root, 200, manifest=manifest)
        self.assertTrue(result["available"])
        self.assertEqual(result["candidate_pair_count"], 1)
        pair = result["pairs"][0]
        self.assertEqual(pair["classification"], "independent_replication")
        self.assertEqual(len(pair["rows"]), 2)
        self.assertTrue(all(pair["independence_criteria"].values()))


if __name__ == "__main__":
    unittest.main()
