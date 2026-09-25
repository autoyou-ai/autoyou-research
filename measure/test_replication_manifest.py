import copy
import json
import unittest
from pathlib import Path

from measure.validate_replication_manifest import (
    provenance_qualified_pairs,
    validate_raw_record_document,
    validate_manifest_document,
)


ROOT = Path(__file__).resolve().parent


class ReplicationManifestChecks(unittest.TestCase):
    def test_released_manifest_has_valid_provenance_contract(self):
        manifest = json.loads(
            (ROOT / "replication_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_manifest_document(manifest), [])
        self.assertEqual(provenance_qualified_pairs(manifest), [])

    def test_missing_identity_is_rejected(self):
        manifest = json.loads(
            (ROOT / "replication_manifest.json").read_text(encoding="utf-8"))
        altered = copy.deepcopy(manifest)
        del altered["files"]["rtx5070-replication.json"]["operator_id"]
        errors = validate_manifest_document(altered)
        self.assertTrue(any("operator_id" in error for error in errors))

    def test_all_three_identities_are_required_for_a_pair(self):
        manifest = {
            "files": {
                "first.json": {
                    "target_id": "target-a",
                    "operator_id": "operator-a",
                    "environment_id": "environment-a",
                },
                "second.json": {
                    "target_id": "target-b",
                    "operator_id": "operator-a",
                    "environment_id": "environment-b",
                },
                "third.json": {
                    "target_id": "target-c",
                    "operator_id": "operator-c",
                    "environment_id": "environment-c",
                },
            }
        }
        self.assertEqual(provenance_qualified_pairs(manifest), [
            ("first.json", "third.json"),
            ("second.json", "third.json"),
        ])

    def _independent_record(self, digest=None):
        rows = []
        for batch_size in (1, 2, 4, 8):
            info = {"available": True}
            if digest is not None:
                info["digest"] = digest
            rows.append({
                "model": "synthetic-model",
                "batch_size": batch_size,
                "reps": 10,
                "model_info": info,
            })
        return {
            "schema_version": "2.0",
            "harness": "measure/bench.py --batch-sizes",
            "protocol": {
                "requested_tokens_per_request": 128,
                "batch_sizes": [1, 2, 4, 8],
                "reps": 10,
                "num_ctx": 4096,
                "warmup_discarded": True,
                "raw_mode": True,
                "unique_nonce_per_request": True,
            },
            "batches": rows,
        }

    def test_independent_raw_record_requires_model_digest(self):
        errors = validate_raw_record_document(
            self._independent_record(), "synthetic-batched.json",
            "independent_replication")
        self.assertTrue(any("digest" in error for error in errors))

    def test_independent_raw_record_contract_accepts_digest(self):
        errors = validate_raw_record_document(
            self._independent_record("sha256:synthetic-model-digest"),
            "synthetic-batched.json", "independent_replication")
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
