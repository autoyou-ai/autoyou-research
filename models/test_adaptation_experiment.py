"""Offline tests for the public synthetic adaptation benchmark."""

import unittest

from models import run_adaptation_experiment as runner


class AdaptationExperimentChecks(unittest.TestCase):
    def test_generated_splits_are_balanced_and_disjoint(self):
        data = runner.generate_tasks(train_per_class=4, eval_per_class=8, seed=17)
        tasks = data["tasks"]
        self.assertEqual(len(tasks), 60)
        train_ids = {task["task_id"] for task in tasks if task["split"] == "train"}
        eval_ids = {task["task_id"] for task in tasks if task["split"] == "evaluation"}
        self.assertTrue(train_ids.isdisjoint(eval_ids))
        self.assertEqual(len(train_ids), 20)
        self.assertEqual(len(eval_ids), 40)
        for task_class in runner.TASK_CLASSES:
            train = [task for task in tasks
                     if task["split"] == "train" and task["task_class"] == task_class]
            evaluation = [task for task in tasks
                          if task["split"] == "evaluation" and task["task_class"] == task_class]
            self.assertEqual(len(train), 4)
            self.assertEqual(len(evaluation), 8)

    def test_correct_option_is_balanced_and_options_are_distinct(self):
        data = runner.generate_tasks(train_per_class=12, eval_per_class=20)
        evaluation = [task for task in data["tasks"]
                      if task["split"] == "evaluation"]
        counts = {letter: 0 for letter in runner.OPTIONS}
        for task in evaluation:
            options = task["options"]
            self.assertEqual(set(options), set(runner.OPTIONS))
            self.assertEqual(len(set(options.values())), 4)
            counts[task["correct_option"]] += 1
        self.assertEqual(counts, {letter: 25 for letter in runner.OPTIONS})

    def test_balanced_options_puts_correct_value_at_requested_position(self):
        options, correct = runner._balanced_options(
            "correct", ["wrong-a", "wrong-b", "wrong-c"], 2)
        self.assertEqual(correct, "C")
        self.assertEqual(options["C"], "correct")
        self.assertEqual(len(set(options.values())), 4)


if __name__ == "__main__":
    unittest.main()
