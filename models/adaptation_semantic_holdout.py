"""Score the saved adapters on a new public semantic task holdout.

The primary adaptation benchmark has disjoint identifiers but repeated task
families and record structures. This probe generates new synthetic records,
new instructions, and a new reasoning format without copying the primary
evaluation records. It reuses the already-trained adapters and reports the
result separately. It is still not a sampled user workload, human grading, or
independent-device replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    _MODEL_DIR = str(Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == _MODEL_DIR:
        sys.path.pop(0)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import adaptation_template_holdout as probe
from models import run_adaptation_experiment as runner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "measure" / "results" / "adaptation-semantic-holdout-tasks.json"
PRIMARY_TASKS = ROOT / "measure" / "results" / "adaptation_tasks.json"
DEFAULT_OUTPUT = ROOT / "measure" / "results" / "adaptation-semantic-holdout.json"
DEFAULT_EVALUATION = ROOT / "models" / "adaptation_semantic_holdout_evaluation.json"
DEFAULT_ADAPTER_ROOT = ROOT / "measure" / "results" / "adaptation_adapters"
TEMPLATE_ID = "unseen-semantic-operations-v1"
TASK_SPLIT = "semantic_holdout"
TASKS_PER_CLASS = 20


def _task_id(task_class: str, index: int) -> str:
    return f"semantic-holdout-{task_class}-{index:03d}"


def _record(task_id: str, task_class: str, index: int, instruction: str,
            correct: str, distractors: Sequence[str]) -> Dict[str, object]:
    options, correct_option = runner._balanced_options(
        correct, distractors, (index * 5 + 1) % 4)
    return {
        "task_id": task_id,
        "split": TASK_SPLIT,
        "task_class": task_class,
        "template_id": TEMPLATE_ID,
        "prompt": runner._prompt(instruction, options),
        "options": options,
        "correct_option": correct_option,
    }


def _extraction(index: int) -> Dict[str, object]:
    stewards = ("Alder", "Basil", "Coral", "Dahlia", "Ember", "Fennel")
    zones = ("north-2", "east-4", "south-1", "west-3")
    impacts = ("delayed export", "duplicate alert", "stale index", "late receipt")
    steward = stewards[index % len(stewards)]
    zone = zones[(index * 3 + 2) % len(zones)]
    impact = impacts[(index + 1) % len(impacts)]
    case = f"CASE-{7000 + index}"
    card = (
        f"Case {case} | steward: {steward} | zone: {zone} | "
        f"impact: {impact} | state: review."
    )
    distractors = [value for value in stewards if value != steward][:3]
    return _record(
        _task_id("extraction", index), "extraction", index,
        f"Read the incident card and identify the assigned steward.\n{card}",
        steward, distractors)


def _rag_qa(index: int) -> Dict[str, object]:
    channels = ("amber-call", "blue-room", "green-pager", "violet-mail")
    domains = ("archive", "billing", "catalog", "dispatch", "identity")
    channel = channels[(index * 3 + 1) % len(channels)]
    domain = domains[index % len(domains)]
    revision = f"R{2 + index % 7}"
    note = (
        f"Policy note {revision}: the {domain} service sends escalation to "
        f"channel {channel}; it rotates keys every {14 + index % 4} days."
    )
    distractors = [value for value in channels if value != channel]
    return _record(
        _task_id("rag_qa", index), "rag_qa", index,
        f"Use only the policy note. Which escalation channel is registered?\n{note}",
        channel, distractors)


def _summary(index: int) -> Dict[str, object]:
    components = ("ingest", "notifier", "catalog", "scheduler", "exporter")
    symptoms = ("late receipts", "missing notices", "stale entries", "skipped jobs")
    triggers = ("a stalled cursor", "a closed lease", "an expired token", "a full buffer")
    actions = ("resume the cursor", "reopen the lease", "renew the token", "drain the buffer")
    component = components[index % len(components)]
    symptom = symptoms[(index * 2 + 1) % len(symptoms)]
    trigger = triggers[(index + 2) % len(triggers)]
    action = actions[(index * 3 + 1) % len(actions)]
    report = (
        f"Change note: {component} showed {symptom} after {trigger}. "
        f"The approved response is to {action}."
    )
    correct = f"{component}: {symptom}; trigger {trigger}; response {action}."
    distractors = [
        f"{component}: {symptom}; trigger {triggers[(index + 1) % len(triggers)]}; response {action}.",
        f"{component}: {symptoms[(index * 2 + 2) % len(symptoms)]}; trigger {trigger}; response {action}.",
        f"{component}: {symptom}; trigger {trigger}; response {actions[(index + 1) % len(actions)]}.",
    ]
    distractors = list(dict.fromkeys(
        value for value in distractors if value != correct))
    if len(distractors) < 3:
        distractors.append(
            f"{component}: {symptom}; trigger {trigger}; response: no action.")
    return _record(
        _task_id("summary", index), "summary", index,
        f"Select the sentence that preserves the cause and approved response.\n{report}",
        correct, distractors)


def _simple_code(index: int) -> Dict[str, object]:
    variants = (
        (
            "return a new list of lowercase strings, preserving duplicates",
            "def normalize(items):\n    return [item.lower() for item in items]",
            [
                "def normalize(items):\n    return sorted(set(items))",
                "def normalize(items):\n    return [item.upper() for item in items]",
                "def normalize(items):\n    return [item.lower() for item in set(items)]",
            ],
        ),
        (
            "return the first item whose status is 'ready', or None if absent",
            "def find_ready(rows):\n    return next((row for row in rows if row.get('status') == 'ready'), None)",
            [
                "def find_ready(rows):\n    return [row for row in rows if row.get('status') == 'ready']",
                "def find_ready(rows):\n    return next((row for row in rows if row.get('status') != 'ready'), None)",
                "def find_ready(rows):\n    return rows[0] if rows else None",
            ],
        ),
        (
            "return a dictionary counting each non-empty label",
            "def count_labels(labels):\n    return {label: labels.count(label) for label in set(labels) if label}",
            [
                "def count_labels(labels):\n    return {label: 1 for label in labels if label}",
                "def count_labels(labels):\n    return {label: labels.count(label) for label in set(labels)}",
                "def count_labels(labels):\n    return {label: labels.index(label) for label in set(labels) if label}",
            ],
        ),
        (
            "return a new dictionary with values clamped to the inclusive range 0 through 100",
            "def clamp(values):\n    return {key: min(100, max(0, value)) for key, value in values.items()}",
            [
                "def clamp(values):\n    return {key: max(0, value) for key, value in values.items()}",
                "def clamp(values):\n    return {key: min(100, value) for key, value in values.items()}",
                "def clamp(values):\n    return {key: value for key, value in values.items() if 0 <= value <= 100}",
            ],
        ),
    )
    requirement, correct, distractors = variants[index % len(variants)]
    return _record(
        _task_id("simple_code", index), "simple_code", index,
        f"Choose the implementation that will {requirement}.", correct, distractors)


def _hard_reason(index: int) -> Dict[str, object]:
    names = (
        ("Iris", "Jasper", "Kestrel", "Lyra"),
        ("Mica", "Nova", "Orion", "Pollen"),
        ("Quill", "Rook", "Sable", "Tern"),
        ("Umber", "Vale", "Wren", "Yarrow"),
    )[index % 4]
    a, b, c, d = names
    target_pairs = ((a, c), (a, d), (b, c), (b, d))
    left, right = target_pairs[index % len(target_pairs)]
    other_left = b if left == a else a
    other_right = d if right == c else c
    rules = (
        f"Select exactly two services. Exactly one of {a} and {b} must be selected. "
        f"Exactly one of {c} and {d} must be selected. "
        f"{left} is selected if and only if {right} is selected. "
        f"{other_left} cannot be selected with {other_right}."
    )
    pairs = [
        f"{x} and {y}"
        for position, (x, y) in enumerate(
            ((a, c), (a, d), (b, c), (b, d)))
        if (x, y) != (left, right)
    ]
    correct = f"{left} and {right}"
    return _record(
        _task_id("hard_reason", index), "hard_reason", index,
        f"Which pair of services satisfies every constraint?\nRules: {rules}",
        correct, pairs)


def generate_tasks(tasks_per_class: int = TASKS_PER_CLASS,
                   seed: int = 20260915) -> Dict[str, object]:
    """Generate new semantic records without copying primary task rows."""
    if tasks_per_class < 1:
        raise ValueError("tasks_per_class must be positive")
    random.seed(seed)
    builders = (_extraction, _rag_qa, _summary, _simple_code, _hard_reason)
    tasks = [
        builder(index)
        for builder in builders
        for index in range(tasks_per_class)
    ]
    return {
        "schema_version": "1.0",
        "benchmark_id": "public-synthetic-operations-mcq-semantic-holdout-v1",
        "generator": {
            "name": "models/adaptation_semantic_holdout.py",
            "seed": seed,
            "task_classes": list(runner.TASK_CLASSES),
            "tasks_per_class": tasks_per_class,
            "note": (
                "New synthetic records and task formats; no primary evaluation "
                "records, prompts, or answers are copied."
            ),
        },
        "tasks": tasks,
    }


def _validate_tasks(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = [task for task in data.get("tasks", [])
             if task.get("split") == TASK_SPLIT]
    if len(tasks) != TASKS_PER_CLASS * len(runner.TASK_CLASSES):
        raise ValueError("semantic holdout must contain 20 tasks per class")
    ids = [str(task.get("task_id", "")) for task in tasks]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("semantic holdout task IDs must be unique")
    if any(task.get("source_task_id") for task in tasks):
        raise ValueError("semantic holdout must not carry source task IDs")
    counts = {task_class: 0 for task_class in runner.TASK_CLASSES}
    for task in tasks:
        if task.get("task_class") not in counts:
            raise ValueError(f"unknown task class: {task.get('task_class')}")
        counts[task["task_class"]] += 1
    if any(count != TASKS_PER_CLASS for count in counts.values()):
        raise ValueError(f"semantic holdout class counts are not balanced: {counts}")
    if not PRIMARY_TASKS.is_file():
        raise ValueError(f"primary task manifest is missing: {PRIMARY_TASKS}")
    primary_data = json.loads(PRIMARY_TASKS.read_text(encoding="utf-8"))
    primary_fingerprints = {
        _task_fingerprint(task)
        for task in primary_data.get("tasks", [])
    }
    overlap = [task["task_id"] for task in tasks
               if _task_fingerprint(task) in primary_fingerprints]
    if overlap:
        raise ValueError(
            "semantic holdout reuses primary task records: "
            + ", ".join(overlap[:3]))
    return tasks


def _task_fingerprint(task: Mapping[str, object]) -> str:
    """Canonical record identity used to reject exact primary-task reuse."""
    return json.dumps(
        {key: task.get(key) for key in (
            "task_class", "template_id", "prompt", "options", "correct_option")},
        sort_keys=True,
        separators=(",", ":"),
    )


def _args_namespace(args: argparse.Namespace) -> argparse.Namespace:
    args.task_split = TASK_SPLIT
    return args


def run(args: argparse.Namespace) -> Dict[str, object]:
    tasks = _validate_tasks(args.tasks)
    result = probe.run(_args_namespace(args))
    task_hash = hashlib.sha256(args.tasks.read_bytes()).hexdigest()
    result["protocol"].update({
        "benchmark_id": "public-synthetic-operations-mcq-semantic-holdout-v1",
        "source_benchmark_id": "public-synthetic-operations-mcq-v1",
        "task_manifest_sha256": task_hash,
        "primary_task_manifest_sha256": hashlib.sha256(
            PRIMARY_TASKS.read_bytes()).hexdigest(),
        "template_id": TEMPLATE_ID,
        "split": (
            "new semantic records generated from unseen data templates; no "
            "primary evaluation records reused"
        ),
        "tasks_per_class": len(tasks) // len(runner.TASK_CLASSES),
        "limitations": [
            "synthetic public-safe tasks are not a sampled user workload",
            "the metric is greedy multiple-choice accuracy, not human task success",
            "one physical machine and CPU-only PyTorch execution",
            "saved adapters were trained on the primary synthetic benchmark",
            "no untouched general-ability or contamination suite",
        ],
    })
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[],
                        help="model alias=local Hugging Face snapshot path; repeat at least three times")
    parser.add_argument("--frontier", help="local Hugging Face snapshot path for the reference model")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--generate-tasks", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--reuse-adapters", action="store_true",
                        help="accepted for symmetry; this probe always reuses saved adapters")
    args = parser.parse_args(argv)
    if args.generate_tasks or args.generate_only or not args.tasks.exists():
        args.tasks.parent.mkdir(parents=True, exist_ok=True)
        args.tasks.write_text(
            json.dumps(generate_tasks(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.tasks}")
    if args.generate_only:
        return 0
    if not args.model or not args.frontier:
        parser.error("--model and --frontier are required unless --generate-only is used")
    if args.eval_batch_size < 1 or args.max_length < 32 or args.threads < 1:
        parser.error("batch size, max length and threads must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"  models: {len(result['models'])}")
    print(f"  tasks per model: {len(result['models'][0]['task_results'])}")
    print(f"  seeds: {len(result['protocol']['seeds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
