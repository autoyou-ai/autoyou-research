"""Offline checks, from the research repo: python -m unittest measure.test_bench."""
import unittest
from unittest.mock import patch

import measure.bench as bench
from measure.bench import PowerLog, client_url


class BenchmarkChecks(unittest.TestCase):
    def test_client_url(self):
        for value, expected in (
            ("0.0.0.0:11434", "http://127.0.0.1:11434"),
            ("http://0.0.0.0:11434/", "http://127.0.0.1:11434"),
            ("[::]:11434", "http://[::1]:11434"),
            ("example.test:12000/", "http://example.test:12000"),
            ("https://example.test/proxy/", "https://example.test:443/proxy"),
            ("", "http://127.0.0.1:11434"),
        ):
            self.assertEqual(client_url(value), expected)
        with self.assertRaises(ValueError):
            client_url("file:///example")

    def test_integrate_request_window(self):
        log = PowerLog({})
        log.start, log.end = 1.0, 3.0
        log.samples = [(0.0, 0.0), (2.0, 20.0), (4.0, 40.0)]
        result = log.result()
        self.assertTrue(result["measured"])
        self.assertAlmostEqual(result["energy_j"], 40.0)
        self.assertAlmostEqual(result["mean_w"], 20.0)
        self.assertEqual(result["trace_s_w"][0], [-1.0, 0.0])
        log.samples = [(2.0, 20.0), (4.0, 40.0)]
        self.assertFalse(log.result()["measured"])

    def test_model_info_keeps_inventory_digest(self):
        bench._MODEL_DIGESTS["synthetic-model"] = "sha256:synthetic-model-digest"
        try:
            with patch.object(bench, "_post", return_value={
                "details": {"family": "synthetic"},
                "model_info": {},
            }):
                info = bench.model_info("synthetic-model")
        finally:
            bench._MODEL_DIGESTS.pop("synthetic-model", None)
        self.assertEqual(info["digest"], "sha256:synthetic-model-digest")


if __name__ == "__main__":
    unittest.main()
