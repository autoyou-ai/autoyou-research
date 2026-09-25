"""Offline tests for the concurrent-serving measurement helpers.

These tests never contact Ollama and never touch a live AutoYou configuration.
"""

import unittest
from unittest.mock import patch

from measure import bench


class BatchHelperChecks(unittest.TestCase):
    def test_timed_batch_request_keeps_server_and_wall_metrics(self):
        response = {
            "eval_count": 32,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 12,
            "prompt_eval_duration": 100_000_000,
        }
        with patch.object(bench, "generate", return_value=response):
            row = bench._timed_batch_request(
                "synthetic:3b", 32, 123, 0, 2)
        self.assertTrue(row["ok"])
        self.assertEqual(row["request"], 2)
        self.assertEqual(row["eval_tokens"], 32)
        self.assertAlmostEqual(row["gen_tok_s"], 16.0)
        self.assertAlmostEqual(row["prefill_tok_s"], 120.0)
        self.assertGreaterEqual(row["wall_s"], 0.0)

    def test_timed_batch_request_records_bad_response(self):
        with patch.object(bench, "generate",
                          return_value={"eval_count": 0, "eval_duration": 0}):
            row = bench._timed_batch_request(
                "synthetic:3b", 32, 123, 0, 0)
        self.assertFalse(row["ok"])
        self.assertIn("usable generation timing", row["error"])

    def test_batch_power_allocation_is_explicit(self):
        power = {"measured": True, "energy_wh": 1.2}
        out = bench._batch_power_allocation(power, 4)
        self.assertTrue(out["measured"])
        self.assertEqual(out["allocation"],
                         "equal split across completed requests")
        self.assertAlmostEqual(out["energy_wh_per_completed_request"], 0.3)

    def test_batch_power_allocation_does_not_invent_missing_sensor(self):
        out = bench._batch_power_allocation(
            {"measured": False, "reason": "no sensor"}, 4)
        self.assertFalse(out["measured"])
        self.assertIn("not measured", out["reason"])


if __name__ == "__main__":
    unittest.main()
