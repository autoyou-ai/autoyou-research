"""Offline tests for the new semantic adaptation holdout."""

import json
import tempfile
import unittest
from pathlib import Path

from models.adaptation_semantic_holdout import (
    TASK_SPLIT,
    generate_tasks,
    _validate_tasks,
)
from models.run_adaptation_experiment import generate_tasks as generate_primary


class AdaptationSemanticHoldoutChecks(unittest.TestCase):
    def test_generated_tasks_are_balanced_and_not_primary_records(self):
        semantic = generate_tasks(tasks_per_class=4)
        primary = generate_primary(train_per_class=2, eval_per_class=4)
        semantic_tasks = semantic["tasks"]
        primary_ids = {task["task_id"] for task in primary["tasks"]}
        primary_records = {
            (task["prompt"], json.dumps(task["options"], sort_keys=True),
             task["correct_option"])
            for task in primary["tasks"]
        }
        self.assertEqual(len(semantic_tasks), 20)
        self.assertEqual(
            {task["task_class"] for task in semantic_tasks},
            {"extraction", "rag_qa", "summary", "simple_code", "hard_reason"},
        )
        self.assertTrue(all(task["split"] == TASK_SPLIT for task in semantic_tasks))
        self.assertTrue(all(task["task_id"] not in primary_ids for task in semantic_tasks))
        self.assertTrue(all(
            (task["prompt"], json.dumps(task["options"], sort_keys=True),
             task["correct_option"]) not in primary_records
            for task in semantic_tasks
        ))
        self.assertTrue(all("source_task_id" not in task for task in semantic_tasks))
        self.assertTrue(all(
            len(task["options"]) == 4
            and len(set(task["options"].values())) == 4
            for task in semantic_tasks
        ))

    def test_validator_rejects_unbalanced_or_copied_holdout(self):
        semantic = generate_tasks(tasks_per_class=2)
        semantic["tasks"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic.json"
            path.write_text(json.dumps(semantic), encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_tasks(path)


if __name__ == "__main__":
    unittest.main()
