"""Measure adaptation on prompt templates unseen during adapter training.

The primary adaptation benchmark uses disjoint task identifiers, but it reuses
the same outer prompt and a small number of class templates across splits.
This supplemental probe keeps the underlying public evaluation records while
changing the instructions and answer labels' surrounding wording. It scores
the already-trained adapters and reports the result separately from the
primary estimand.

This is a template-robustness probe, not a real-workload sample or an
independent device replication. It never reads private AutoYou data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# Running this file directly puts models/ ahead of the import path, where the
# legacy models.py module would shadow the namespace package.
if __package__ in (None, ""):
    _MODEL_DIR = str(Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == _MODEL_DIR:
        sys.path.pop(0)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import run_adaptation_experiment as runner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_TASKS = ROOT / "measure" / "results" / "adaptation_tasks.json"
DEFAULT_TASKS = ROOT / "measure" / "results" / "adaptation-template-holdout-tasks.json"
DEFAULT_OUTPUT = ROOT / "measure" / "results" / "adaptation-template-holdout.json"
DEFAULT_ADAPTER_ROOT = ROOT / "measure" / "results" / "adaptation_adapters"
TEMPLATE_ID = "unseen-operations-prompt-v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _paraphrase_prompt(prompt: str, task_class: str) -> str:
    """Change the outer and class-specific wording without changing the answer."""
    replacements = {
        "You are evaluating a synthetic operations record. Choose the one correct option.\n\n"
        "": "Review this service case and select the single correct choice.\n"
        "Return only its capital letter.\n\n",
        "Extract the owner from this record:":
            "From this ticket, identify the assigned owner:",
        "Using only the knowledge card, what is the retention period?":
            "Refer only to the knowledge card below. Which retention period is specified?",
        "Choose the most accurate one-sentence summary of this report:":
            "Select the one-sentence digest that matches this incident report:",
        "Which implementation will ":
            "Select the implementation that will ",
        "Four jobs must run once. Which order satisfies all rules?":
            "Arrange the four jobs into the only order that obeys every rule.",
        "Options:": "Candidate choices:",
        "Answer:": "Response:",
    }
    updated = prompt
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    if updated == prompt:
        raise ValueError(f"no template transformation applied for {task_class}")
    return updated


def build_holdout_manifest(source_path: Path) -> Dict[str, object]:
    """Build a public-safe holdout manifest from evaluation records only."""
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_tasks = [
        task for task in source.get("tasks", [])
        if task.get("split") == "evaluation"
    ]
    if not source_tasks:
        raise ValueError("source task manifest has no evaluation tasks")
    tasks: List[Dict[str, object]] = []
    for task in source_tasks:
        row = dict(task)
        source_id = str(task["task_id"])
        row["task_id"] = f"template-holdout-{source_id}"
        row["source_task_id"] = source_id
        row["split"] = "template_holdout"
        row["template_id"] = TEMPLATE_ID
        row["prompt"] = _paraphrase_prompt(
            str(task["prompt"]), str(task["task_class"])
        )
        tasks.append(row)
    return {
        "schema_version": "1.0",
        "benchmark_id": "public-synthetic-operations-mcq-template-holdout-v1",
        "source_benchmark_id": source.get(
            "benchmark_id", "public-synthetic-operations-mcq-v1"
        ),
        "generator": {
            "name": "models/adaptation_template_holdout.py",
            "template_id": TEMPLATE_ID,
            "source_split": "evaluation",
            "note": (
                "Public synthetic records only; IDs are new and prompts use an "
                "unseen outer template."
            ),
        },
        "tasks": tasks,
    }


def _load_holdout_tasks(path: Path,
                        split: str = "template_holdout"
                        ) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = [
        task for task in data.get("tasks", [])
        if task.get("split") == split
    ]
    if not tasks:
        raise ValueError(f"holdout task manifest has no {split} tasks")
    seen = set()
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        if not task_id or task_id in seen:
            raise ValueError("holdout task IDs must be present and unique")
        seen.add(task_id)
        if task.get("task_class") not in runner.TASK_CLASSES:
            raise ValueError(f"unknown holdout task class: {task.get('task_class')}")
    return tasks


def _public_base_ref(alias: str) -> str:
    return {
        "qwen25-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen25-3b": "Qwen/Qwen2.5-3B-Instruct",
        "qwen25-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    }.get(alias, f"local/{alias}")


def _task_results(tasks: Sequence[Dict[str, object]],
                  base_scores: Mapping[str, float],
                  frontier_scores: Mapping[str, float],
                  seed_scores: Mapping[str, Sequence[Mapping[str, float]]]
                  ) -> List[Dict[str, object]]:
    result = []
    for task in tasks:
        task_id = str(task["task_id"])
        rows = list(seed_scores[task_id])
        result.append({
            "task_id": task_id,
            "source_task_id": task.get("source_task_id"),
            "template_id": task.get("template_id", TEMPLATE_ID),
            "task_class": task["task_class"],
            "base_score": base_scores[task_id],
            "adapted_score": sum(float(row["adapted_score"]) for row in rows)
            / len(rows),
            "frontier_score": frontier_scores[task_id],
            "seed_scores": rows,
        })
    return result


def run(args: argparse.Namespace) -> Dict[str, object]:
    import torch

    torch.set_num_threads(args.threads)
    tasks = _load_holdout_tasks(
        args.tasks, getattr(args, "task_split", "template_holdout"))
    specs = runner._parse_model_specs(args.model)
    if len(specs) < 3:
        raise ValueError("at least three model=path entries are required")

    print("loading frontier reference for template holdout", flush=True)
    frontier_tokenizer, frontier_model = runner._load_stack(
        torch, args.frontier, trainable=False, dtype=torch.bfloat16
    )
    frontier_ids = runner._option_ids(frontier_tokenizer)
    frontier_scores = runner._score_tasks(
        torch, frontier_model, frontier_tokenizer, tasks, frontier_ids,
        args.max_length, args.eval_batch_size,
    )
    del frontier_model, frontier_tokenizer
    gc.collect()

    models: List[Dict[str, object]] = []
    for alias, model_path in sorted(specs.items()):
        print(f"loading base model {alias} for template holdout", flush=True)
        path = Path(model_path).resolve()
        tokenizer, base_model = runner._load_stack(
            torch, model_path, trainable=False, dtype=torch.bfloat16
        )
        option_ids = runner._option_ids(tokenizer)
        base_scores = runner._score_tasks(
            torch, base_model, tokenizer, tasks, option_ids,
            args.max_length, args.eval_batch_size,
        )
        del base_model
        gc.collect()

        seed_scores: Dict[str, List[Dict[str, float]]] = {
            str(task["task_id"]): [] for task in tasks
        }
        adapter_digests = []
        for seed in runner.SEEDS:
            adapter_dir = args.adapter_root / alias / f"seed-{seed}"
            if not adapter_dir.is_dir():
                raise FileNotFoundError(f"missing adapter directory: {adapter_dir}")
            print(f"scoring {alias} adapter seed {seed} on template holdout", flush=True)
            peft = runner._external_peft()
            _, adapted_model = runner._load_stack(
                torch, model_path, trainable=False, dtype=torch.float32
            )
            adapted_model = peft.PeftModel.from_pretrained(
                adapted_model, adapter_dir, is_trainable=False
            )
            adapted_scores = runner._score_tasks(
                torch, adapted_model, tokenizer, tasks, option_ids,
                args.max_length, args.eval_batch_size,
            )
            adapter_digests.append(runner._sha256_tree(adapter_dir))
            for task in tasks:
                task_id = str(task["task_id"])
                seed_scores[task_id].append({
                    "seed": seed,
                    "base_score": base_scores[task_id],
                    "adapted_score": adapted_scores[task_id],
                    "frontier_score": frontier_scores[task_id],
                })
            del adapted_model
            gc.collect()

        models.append({
            "model_id": alias,
            "model_family": "Qwen2.5 local snapshot",
            "base_model_ref": _public_base_ref(alias),
            "base_model_id": path.name,
            "base_digest": runner._model_revision(path),
            "adapter_id": "LoRA-r4-qv-public-synthetic-v1",
            "adapter_digest": _sha256_bytes("".join(sorted(adapter_digests)).encode()),
            "adapter_seed_digests": adapter_digests,
            "device_id": "amd-strix-halo-cpu-transformers-local-1",
            "task_results": _task_results(
                tasks, base_scores, frontier_scores, seed_scores
            ),
        })
        del tokenizer
        gc.collect()

    task_bytes = args.tasks.read_bytes()
    return {
        "schema_version": "1.0",
        "protocol": {
            "benchmark_id": "public-synthetic-operations-mcq-template-holdout-v1",
            "source_benchmark_id": "public-synthetic-operations-mcq-v1",
            "task_manifest_sha256": _sha256_bytes(task_bytes),
            "template_id": TEMPLATE_ID,
            "task_classes": list(runner.TASK_CLASSES),
            "tasks_per_class": len([
                task for task in tasks if task["task_class"] == runner.TASK_CLASSES[0]
            ]),
            "split": "unseen prompt template applied to held-out public records",
            "metric": (
                "greedy multiple-choice accuracy, scored 1 for the correct option and 0 otherwise"
            ),
            "grader": {
                "name": "greedy-option-accuracy-scorer",
                "version": "1.0",
                "blinded": True,
                "note": "Scores see the prompt and correct option, not model or adapter labels.",
            },
            "seeds": list(runner.SEEDS),
            "seed_kind": "adapter_training_replay",
            "frontier_reference": "Qwen/Qwen2.5-VL-7B-Instruct",
            "evaluation_mode": "text-only; vision tower unused",
            "device_id": "amd-strix-halo-cpu-transformers-local-1",
            "operator_id": "author-operator-1",
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "peft": runner._external_peft().__version__,
                "threads": args.threads,
                "dtype": "float32 for adapted replay",
            },
            "source_revision": runner._source_revision(),
            "limitations": [
                "prompt-template robustness probe, not a new sampled workload",
                "underlying semantic records are reused from the public evaluation split",
                "one physical machine and CPU-only PyTorch execution",
                "no untouched general-ability or human-grading suite",
            ],
        },
        "models": models,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[],
                        help="model alias=local Hugging Face snapshot path; repeat at least three times")
    parser.add_argument("--frontier",
                        help="local Hugging Face snapshot path for the reference model")
    parser.add_argument("--source-tasks", type=Path, default=DEFAULT_SOURCE_TASKS)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--generate-tasks", action="store_true")
    parser.add_argument("--generate-only", action="store_true",
                        help="write the public holdout manifest and stop")
    parser.add_argument("--reuse-adapters", action="store_true",
                        help="accepted for symmetry; this probe always reuses saved adapters")
    args = parser.parse_args(argv)
    if args.generate_tasks or args.generate_only or not args.tasks.exists():
        args.tasks.parent.mkdir(parents=True, exist_ok=True)
        manifest = build_holdout_manifest(args.source_tasks)
        args.tasks.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
