"""Offline tests for the empirical adaptation evaluation contract."""

import json
import tempfile
import unittest
from pathlib import Path

from models import adaptation_eval as evaluation


class AdaptationEvaluationChecks(unittest.TestCase):
    def test_missing_input_is_explicitly_unmeasured(self):
        with tempfile.TemporaryDirectory() as directory:
            result = evaluation.summarize_file(
                Path(directory) / "missing.json", resamples=200)
        self.assertFalse(result["available"])
        self.assertEqual(result["status"], "NOT TESTED")
        self.assertIn("uplift remains a scenario", result["reason"])

    def test_valid_input_uses_model_cluster_bootstrap(self):
        models = []
        for model_index in range(3):
            task_results = []
            for task_class in evaluation.TASK_CLASSES:
                for task_index in range(20):
                    task_results.append({
                        "task_id": f"{task_class}-{task_index}",
                        "task_class": task_class,
                        "base_score": 0.60,
                        "adapted_score": 0.66 + model_index * 0.01,
                        "frontier_score": 0.80,
                    })
            models.append({
                "model_id": f"synthetic-model-{model_index}",
                "adapter_id": "synthetic-adapter",
                "device_id": "synthetic-device",
                "task_results": task_results,
            })
        payload = {
            "schema_version": "1.0",
            "protocol": {"task_classes": list(evaluation.TASK_CLASSES)},
            "models": models,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = evaluation.summarize_file(path, resamples=200)
        self.assertTrue(result["available"])
        self.assertEqual(result["validation"]["model_count"], 3)
        self.assertEqual(
            result["validation"]["task_count_by_class"]["extraction"], 20)
        self.assertEqual(result["pooled"]["uplift_pp"]["model_clusters"], 3)
        self.assertAlmostEqual(result["pooled"]["uplift_pp"]["model_mean"], 7.0)
        self.assertIn("paired_permutation_p_two_sided", result["pooled"])
        self.assertEqual(result["secondary_multiplicity"]["correction"],
                         "Bonferroni")
        self.assertEqual(
            result["secondary_multiplicity"]["hypothesis_count"], 5)
        self.assertIn(
            "paired_cluster_permutation_p_two_sided",
            result["by_task_class"]["extraction"])

    def test_seed_scores_are_averaged_before_model_bootstrap(self):
        payload = {
            "protocol": {"seeds": [1, 2, 3]},
            "models": [{
                "model_id": "synthetic-model",
                "task_results": [{
                    "task_id": "task",
                    "task_class": "extraction",
                    "seed_scores": [
                        {"seed": 1, "base_score": 0.4,
                         "adapted_score": 0.6, "frontier_score": 0.8},
                        {"seed": 2, "base_score": 0.5,
                         "adapted_score": 0.7, "frontier_score": 0.8},
                        {"seed": 3, "base_score": 0.6,
                         "adapted_score": 0.8, "frontier_score": 0.8},
                    ],
                    "base_score": 0.5,
                    "adapted_score": 0.7,
                    "frontier_score": 0.8,
                }],
            }],
        }
        rows, errors = evaluation._normalise_rows(payload)
        validation = evaluation.validate(payload, rows, errors)
        self.assertTrue(validation["checks"]["seed_repetitions"])
        self.assertEqual(rows[0]["seed_count"], 3)
        self.assertAlmostEqual(rows[0]["base_accuracy"], 0.5)
        self.assertAlmostEqual(rows[0]["base_score"], 0.5)

    def test_zero_frontier_accuracy_is_valid_for_absolute_estimand(self):
        payload = {
            "models": [{
                "model_id": "synthetic-model",
                "task_results": [{
                    "task_id": "task",
                    "task_class": "extraction",
                    "base_score": 0.0,
                    "adapted_score": 1.0,
                    "frontier_score": 0.0,
                }],
            }],
        }
        rows, errors = evaluation._normalise_rows(payload)
        self.assertFalse(errors)
        self.assertEqual(rows[0]["accuracy_uplift_pp"], 100.0)

    def test_duplicate_model_task_pairs_invalidate_file(self):
        payload = {
            "models": [{
                "model_id": "synthetic-model",
                "task_results": [{
                    "task_id": "same",
                    "task_class": "extraction",
                    "base_score": 0.5,
                    "adapted_score": 0.6,
                    "frontier_score": 0.8,
                }, {
                    "task_id": "same",
                    "task_class": "extraction",
                    "base_score": 0.5,
                    "adapted_score": 0.6,
                    "frontier_score": 0.8,
                }],
            }],
        }
        rows, errors = evaluation._normalise_rows(payload)
        validation = evaluation.validate(payload, rows, errors)
        self.assertFalse(validation["passed"])
        self.assertFalse(validation["checks"]["no_duplicate_model_task_pairs"])


if __name__ == "__main__":
    unittest.main()
