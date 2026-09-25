"""Offline tests for the adaptation template-holdout probe."""

import json
import tempfile
import unittest
from pathlib import Path

from models.adaptation_template_holdout import (
    TEMPLATE_ID,
    _paraphrase_prompt,
    build_holdout_manifest,
)
from models.run_adaptation_experiment import generate_tasks


class AdaptationTemplateHoldoutChecks(unittest.TestCase):
    def test_paraphrase_changes_outer_and_class_wording(self):
        prompt = (
            "You are evaluating a synthetic operations record. Choose the one "
            "correct option.\n\nExtract the owner from this record:\nrecord\n\n"
            "Options:\nA. x\nB. y\nC. z\nD. q\n\nAnswer:"
        )
        updated = _paraphrase_prompt(prompt, "extraction")
        self.assertNotEqual(updated, prompt)
        self.assertIn("Candidate choices:", updated)
        self.assertIn("Response:", updated)
        self.assertNotIn("Extract the owner from this record:", updated)

    def test_manifest_uses_only_evaluation_ids_and_new_template(self):
        source = generate_tasks(train_per_class=2, eval_per_class=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            manifest = build_holdout_manifest(path)
        source_eval = {
            task["task_id"] for task in source["tasks"]
            if task["split"] == "evaluation"
        }
        holdout = manifest["tasks"]
        self.assertEqual(len(holdout), len(source_eval))
        self.assertTrue(all(task["template_id"] == TEMPLATE_ID for task in holdout))
        self.assertTrue(all(task["source_task_id"] in source_eval for task in holdout))
        self.assertTrue(all(task["task_id"] not in source_eval for task in holdout))
        self.assertTrue(all(task["split"] == "template_holdout" for task in holdout))


if __name__ == "__main__":
    unittest.main()
