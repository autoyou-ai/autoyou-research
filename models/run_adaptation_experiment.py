"""Run a public-safe, multi-model, multi-task adaptation experiment.

This runner is deliberately separate from the product-specific support-model
artifacts. It creates a synthetic operations benchmark, trains ordinary LoRA
adapters on disjoint training tasks, and scores held-out tasks with a blinded
multiple-choice accuracy metric. The output is an aggregate record only.

The benchmark is evidence that the adaptation pipeline was measured across
several models and task classes. It is not a representative user-workload
sample, a human quality study, or evidence about the private AutoYou support
model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "measure" / "results" / "adaptation_tasks.json"
DEFAULT_OUTPUT = ROOT / "measure" / "results" / "adaptation-multimodel.json"
DEFAULT_ADAPTER_ROOT = ROOT / "measure" / "results" / "adaptation_adapters"

TASK_CLASSES = (
    "extraction",
    "rag_qa",
    "summary",
    "simple_code",
    "hard_reason",
)
OPTIONS = ("A", "B", "C", "D")
SEEDS = (20260914, 20260915, 20260916)
GENERATOR_SEED = 20260914


def _source_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _balanced_options(correct: str, distractors: Sequence[str], index: int
                      ) -> Tuple[Dict[str, str], str]:
    values = [correct, *distractors]
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("each task needs four distinct options")
    index %= 4
    ordered: List[Optional[str]] = [None] * 4
    ordered[index] = values[0]
    rest = iter(values[1:])
    for position in range(4):
        if ordered[position] is None:
            ordered[position] = next(rest)
    return dict(zip(OPTIONS, ordered)), OPTIONS[index]


def _prompt(instruction: str, options: Mapping[str, str]) -> str:
    rows = "\n".join(f"{letter}. {options[letter]}" for letter in OPTIONS)
    return (
        "You are evaluating a synthetic operations record. Choose the one "
        "correct option. Output only its letter.\n\n"
        f"{instruction}\n\nOptions:\n{rows}\n\nAnswer:"
    )


def _extraction(index: int, split: str, rng: random.Random) -> Dict[str, object]:
    owners = ("Orchid", "Juniper", "Cobalt", "Maple", "Cedar", "Quartz")
    queues = ("alpha", "bravo", "charlie", "delta")
    owner = owners[index % len(owners)]
    queue = queues[(index * 3 + 1) % len(queues)]
    ticket = f"OPS-{1000 + index + (0 if split == 'train' else 5000)}"
    record = (
        f"Ticket {ticket}; owner={owner}; queue={queue}; "
        f"priority=P{1 + (index % 3)}; status={rng.choice(('open', 'held'))}."
    )
    options, correct = _balanced_options(
        owner,
        [value for value in owners if value != owner][:3],
        index,
    )
    return {
        "task_id": f"{split}-extraction-{index:03d}",
        "split": split,
        "task_class": "extraction",
        "prompt": _prompt(f"Extract the owner from this record:\n{record}", options),
        "options": options,
        "correct_option": correct,
    }


def _rag_qa(index: int, split: str, rng: random.Random) -> Dict[str, object]:
    projects = ("Nimbus", "Harbor", "Lattice", "Saffron", "Pioneer", "Mosaic")
    regions = ("North", "South", "East", "West")
    project = projects[index % len(projects)]
    retention = (7, 14, 30, 45)[index % 4]
    region = regions[(index + 1) % len(regions)]
    backup = ("01:00", "02:30", "04:00", "22:00")[index % 4]
    context = (
        f"Project {project} has a retention period of {retention} days, "
        f"runs backups at {backup} UTC, and is hosted in the {region} region."
    )
    options, correct = _balanced_options(
        f"{retention} days",
        [f"{value} days" for value in (7, 14, 30, 45) if value != retention][:3],
        (index + 1) % 4,
    )
    return {
        "task_id": f"{split}-rag_qa-{index:03d}",
        "split": split,
        "task_class": "rag_qa",
        "prompt": _prompt(
            f"Using only the knowledge card, what is the retention period?\n{context}",
            options,
        ),
        "options": options,
        "correct_option": correct,
    }


def _summary(index: int, split: str, rng: random.Random) -> Dict[str, object]:
    services = ("relay", "indexer", "router", "worker", "gateway", "scheduler")
    impacts = ("delayed jobs", "stale results", "failed handoffs", "slow uploads")
    causes = ("a full queue", "an expired lease", "a missing route", "a bad cache")
    actions = ("drain the queue", "renew the lease", "restore the route", "clear the cache")
    service = services[index % len(services)]
    impact = impacts[(index * 2) % len(impacts)]
    cause = causes[(index + 1) % len(causes)]
    action = actions[(index * 3) % len(actions)]
    report = (
        f"Incident report: the {service} produced {impact} after {cause}. "
        f"The assigned next action is to {action}."
    )
    correct_text = f"{service}: {impact} caused by {cause}; next action: {action}."
    distractors = [
        f"{service}: {impact} caused by {causes[(index + 2) % len(causes)]}; next action: {action}.",
        f"{service}: {impacts[(index * 2 + 1) % len(impacts)]} caused by {cause}; next action: {action}.",
        f"{service}: {impact} caused by {cause}; next action: {actions[(index + 1) % len(actions)]}.",
    ]
    options, correct = _balanced_options(correct_text, distractors, (index + 2) % 4)
    return {
        "task_id": f"{split}-summary-{index:03d}",
        "split": split,
        "task_class": "summary",
        "prompt": _prompt(
            f"Choose the most accurate one-sentence summary of this report:\n{report}",
            options,
        ),
        "options": options,
        "correct_option": correct,
    }


def _simple_code(index: int, split: str, rng: random.Random) -> Dict[str, object]:
    requirement = (
        "return a new list containing only the even integers, preserving order"
        if index % 2 == 0 else
        "return the sum of the positive integers, ignoring zero and negatives"
    )
    if index % 2 == 0:
        correct_text = "def select(values):\n    return [x for x in values if x % 2 == 0]"
        distractors = [
            "def select(values):\n    return [x for x in values if x % 2 != 0]",
            "def select(values):\n    return sorted(values)",
            "def select(values):\n    return [x for x in values if x > 0]",
        ]
    else:
        correct_text = "def select(values):\n    return sum(x for x in values if x > 0)"
        distractors = [
            "def select(values):\n    return sum(x for x in values if x < 0)",
            "def select(values):\n    return [x for x in values if x > 0]",
            "def select(values):\n    return sum(values)",
        ]
    options, correct = _balanced_options(correct_text, distractors, (index * 3) % 4)
    return {
        "task_id": f"{split}-simple_code-{index:03d}",
        "split": split,
        "task_class": "simple_code",
        "prompt": _prompt(f"Which implementation will {requirement}?", options),
        "options": options,
        "correct_option": correct,
    }


def _hard_reason(index: int, split: str, rng: random.Random) -> Dict[str, object]:
    names = (
        ("Aster", "Birch", "Cinder", "Dune"),
        ("Atlas", "Beryl", "Crest", "Delta"),
        ("Arbor", "Beacon", "Cobalt", "Drift"),
        ("Amber", "Brine", "Cedar", "Dawn"),
    )[index % 4]
    a, b, c, d = names
    patterns = (
        (f"{d}, {a}, {b}, {c}", (f"{a} is before {b}; {b} is immediately before {c}; {d} is before {a}.")),
        (f"{a}, {b}, {c}, {d}", (f"{a} is immediately before {b}; {c} is before {d}; {b} is before {c}.")),
        (f"{b}, {c}, {d}, {a}", (f"{b} is before {c}; {c} is immediately before {d}; {a} is after {d}.")),
        (f"{c}, {d}, {a}, {b}", (f"{c} is immediately before {d}; {d} is before {a}; {a} is before {b}.")),
    )
    correct_text, rules = patterns[index % len(patterns)]
    all_orders = [", ".join(order) for order in permutations(names)]
    distractors = [order for order in all_orders if order != correct_text][:3]
    options, correct = _balanced_options(correct_text, distractors, (index + 3) % 4)
    return {
        "task_id": f"{split}-hard_reason-{index:03d}",
        "split": split,
        "task_class": "hard_reason",
        "prompt": _prompt(
            f"Four jobs must run once. Which order satisfies all rules?\nRules: {rules}",
            options,
        ),
        "options": options,
        "correct_option": correct,
    }


def generate_tasks(train_per_class: int = 12, eval_per_class: int = 20,
                   seed: int = GENERATOR_SEED) -> Dict[str, object]:
    """Generate disjoint public-safe train and held-out task records."""
    rng = random.Random(seed)
    builders = (_extraction, _rag_qa, _summary, _simple_code, _hard_reason)
    tasks: List[Dict[str, object]] = []
    for split, count in (("train", train_per_class), ("evaluation", eval_per_class)):
        for builder in builders:
            for index in range(count):
                tasks.append(builder(index + (100 if split == "evaluation" else 0), split, rng))
    return {
        "schema_version": "1.0",
        "benchmark_id": "public-synthetic-operations-mcq-v1",
        "generator": {
            "name": "models/run_adaptation_experiment.py",
            "seed": seed,
            "task_classes": list(TASK_CLASSES),
            "train_tasks_per_class": train_per_class,
            "evaluation_tasks_per_class": eval_per_class,
            "note": "Synthetic records only; no user, product, or private support data.",
        },
        "tasks": tasks,
    }


def _load_tasks(path: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    train = [task for task in tasks if task.get("split") == "train"]
    evaluation = [task for task in tasks if task.get("split") == "evaluation"]
    for split, rows in (("train", train), ("evaluation", evaluation)):
        counts = {task_class: 0 for task_class in TASK_CLASSES}
        for task in rows:
            if task.get("task_class") in counts:
                counts[task["task_class"]] += 1
        if any(count == 0 for count in counts.values()):
            raise ValueError(f"{split} task manifest is missing a task class: {counts}")
    ids = [str(task["task_id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task manifest contains duplicate task IDs")
    return train, evaluation


def _parse_model_specs(raw: Sequence[str]) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    for value in raw:
        if "=" not in value:
            raise ValueError("--model must be alias=local_snapshot_path")
        alias, path = value.split("=", 1)
        if not alias or not path:
            raise ValueError("--model must be alias=local_snapshot_path")
        specs[alias] = path
    return specs


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _external_peft():
    """Import the installed PEFT package without shadowing models/peft.py."""
    import importlib

    local_dir = Path(__file__).resolve().parent
    original = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path
                       if not entry or Path(entry).resolve() != local_dir]
        return importlib.import_module("peft")
    finally:
        sys.path[:] = original


def _load_stack(torch, path: str, trainable: bool, dtype=None):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = Path(path).resolve()
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.model_type == "qwen2_5_vl":
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    if dtype is None:
        dtype = torch.float32 if trainable else torch.bfloat16
    model = model_class.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if trainable:
        model.config.use_cache = False
    return tokenizer, model


def _option_ids(tokenizer) -> Dict[str, int]:
    result = {}
    for option in OPTIONS:
        ids = tokenizer.encode(f" {option}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"option {option} is not one token for this tokenizer")
        result[option] = ids[0]
    return result


def _prompt_ids(tokenizer, prompt: str, max_length: int) -> List[int]:
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length - 1,
    )
    ids = list(encoded["input_ids"])
    if len(ids) >= max_length:
        raise ValueError("prompt is too long for the configured max length")
    return ids


def _collate(rows: Sequence[Tuple[List[int], int]], pad_id: int):
    import torch

    width = max(len(ids) for ids, _ in rows)
    inputs = []
    labels = []
    masks = []
    for ids, target in rows:
        pad = width - len(ids)
        inputs.append(ids + [pad_id] * pad)
        masks.append([1] * len(ids) + [0] * pad)
        labels.append([-100] * (len(ids) - 1) + [target] + [-100] * pad)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _training_rows(tokenizer, tasks: Sequence[Dict[str, object]], option_ids,
                   max_length: int):
    rows = []
    for task in tasks:
        prompt = _prompt_ids(tokenizer, str(task["prompt"]), max_length)
        target = option_ids[str(task["correct_option"])]
        rows.append((prompt + [target], target))
    return rows


def _score_tasks(torch, model, tokenizer, tasks: Sequence[Dict[str, object]],
                 option_ids: Mapping[str, int], max_length: int,
                 batch_size: int) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    model.eval()
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start:start + batch_size]
        ids = [_prompt_ids(tokenizer, str(task["prompt"]), max_length)
               for task in chunk]
        width = max(len(row) for row in ids)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.tensor(
            [row + [pad_id] * (width - len(row)) for row in ids],
            dtype=torch.long,
        )
        attention = torch.tensor(
            [[1] * len(row) + [0] * (width - len(row)) for row in ids],
            dtype=torch.long,
        )
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention)
            logits = output.logits.float()
            last = attention.sum(dim=1) - 1
            next_logits = logits[torch.arange(len(chunk)), last]
            candidate_ids = torch.tensor(
                [option_ids[letter] for letter in OPTIONS], dtype=torch.long)
            probabilities = torch.softmax(next_logits[:, candidate_ids], dim=1)
        for row, probability in zip(chunk, probabilities):
            correct_index = OPTIONS.index(str(row["correct_option"]))
            predicted_index = int(torch.argmax(probability).item())
            scores[str(row["task_id"])] = (
                1.0 if predicted_index == correct_index else 0.0)
    return scores


def _write_adapter_readme(adapter_dir: Path) -> None:
    """Keep generated adapter cards portable and free of local machine paths."""
    (adapter_dir / "README.md").write_text(
        "# Public synthetic LoRA adapter\n\n"
        "This adapter was trained by `models/run_adaptation_experiment.py` "
        "on the public synthetic operations benchmark. It contains no user, "
        "product, or private support data. The exact base snapshot revision, "
        "seed, task-manifest hash, and training command are recorded in the "
        "aggregate experiment JSON. Load it with the matching public base-model "
        "identifier and the adapter configuration in this directory.\n",
        encoding="utf-8",
    )


def _sanitize_adapter_config(adapter_dir: Path, public_base_id: str) -> None:
    """Replace PEFT's local base path with the public model identifier."""
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = public_base_id
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _train_seed(torch, model_path: str, tokenizer, train_tasks, eval_tasks,
                option_ids, seed: int, args, adapter_dir: Path,
                public_base_id: str):
    from torch.utils.data import DataLoader

    peft = _external_peft()
    LoraConfig = peft.LoraConfig
    get_peft_model = peft.get_peft_model
    _set_seed(torch, seed)
    _, model = _load_stack(torch, model_path, trainable=True,
                           dtype=torch.float32)
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, config)
    rows = _training_rows(tokenizer, train_tasks, option_ids, args.max_length)
    loader = DataLoader(
        rows,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda batch: _collate(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    model.train()
    steps = 0
    for _ in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    _write_adapter_readme(adapter_dir)
    _sanitize_adapter_config(adapter_dir, public_base_id)
    digest = _sha256_tree(adapter_dir)
    scores = _score_tasks(
        torch, model, tokenizer, eval_tasks, option_ids,
        args.max_length, args.eval_batch_size,
    )
    del optimizer, model, loader
    gc.collect()
    return scores, digest, steps


def _reuse_seed(torch, model_path: str, tokenizer, eval_tasks, option_ids,
                seed: int, args, adapter_dir: Path, public_base_id: str):
    """Score an already-saved adapter without repeating its training run."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"missing adapter directory: {adapter_dir}")
    peft = _external_peft()
    _, model = _load_stack(torch, model_path, trainable=False,
                           dtype=torch.float32)
    model = peft.PeftModel.from_pretrained(model, adapter_dir,
                                           is_trainable=False)
    _write_adapter_readme(adapter_dir)
    _sanitize_adapter_config(adapter_dir, public_base_id)
    scores = _score_tasks(
        torch, model, tokenizer, eval_tasks, option_ids,
        args.max_length, args.eval_batch_size,
    )
    digest = _sha256_tree(adapter_dir)
    del model
    gc.collect()
    steps = len(_training_rows(tokenizer, args._train_tasks,
                               option_ids, args.max_length)) // args.train_batch_size
    steps *= args.epochs
    return scores, digest, steps


def _model_revision(path: Path) -> str:
    if path.parent.name == "snapshots":
        return path.name
    return _sha256_file(path / "config.json") if (path / "config.json").is_file() else str(path)


def run(args: argparse.Namespace) -> Dict[str, object]:
    import torch

    torch.set_num_threads(args.threads)
    train_tasks, eval_tasks = _load_tasks(args.tasks)
    task_bytes = args.tasks.read_bytes()
    specs = _parse_model_specs(args.model)
    if len(specs) < 3:
        raise ValueError("at least three model=path entries are required")

    # The frontier is measured on the same task prompts, but is not adapted.
    print("loading frontier reference", flush=True)
    frontier_tokenizer, frontier_model = _load_stack(
        torch, args.frontier, trainable=False, dtype=torch.bfloat16)
    frontier_ids = _option_ids(frontier_tokenizer)
    frontier_scores = _score_tasks(
        torch, frontier_model, frontier_tokenizer, eval_tasks,
        frontier_ids, args.max_length, args.eval_batch_size,
    )
    del frontier_model, frontier_tokenizer
    gc.collect()
    print("frontier scoring complete", flush=True)

    models: List[Dict[str, object]] = []
    for alias, model_path in sorted(specs.items()):
        print(f"loading base model {alias}", flush=True)
        path = Path(model_path).resolve()
        tokenizer, base_model = _load_stack(
            torch, model_path, trainable=False, dtype=torch.bfloat16)
        option_ids = _option_ids(tokenizer)
        base_scores = _score_tasks(
            torch, base_model, tokenizer, eval_tasks,
            option_ids, args.max_length, args.eval_batch_size,
        )
        del base_model
        gc.collect()
        print(f"base scoring complete {alias}", flush=True)

        seed_scores: Dict[str, Dict[str, Dict[str, float]]] = {
            str(task["task_id"]): {} for task in eval_tasks
        }
        adapter_digests = []
        step_counts = []
        args._train_tasks = train_tasks
        public_base_id = {
            "qwen25-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
            "qwen25-3b": "Qwen/Qwen2.5-3B-Instruct",
            "qwen25-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        }.get(alias, f"local/{alias}")
        for seed in SEEDS:
            print(f"training {alias} seed {seed}", flush=True)
            adapter_dir = args.adapter_root / alias / f"seed-{seed}"
            if args.reuse_adapters:
                print(f"reusing adapter {alias} seed {seed}", flush=True)
                adapted_scores, adapter_digest, steps = _reuse_seed(
                    torch, model_path, tokenizer, eval_tasks, option_ids,
                    seed, args, adapter_dir, public_base_id,
                )
            else:
                adapted_scores, adapter_digest, steps = _train_seed(
                    torch, model_path, tokenizer, train_tasks, eval_tasks,
                    option_ids, seed, args, adapter_dir, public_base_id,
                )
            adapter_digests.append(adapter_digest)
            step_counts.append(steps)
            for task in eval_tasks:
                task_id = str(task["task_id"])
                seed_scores[task_id][str(seed)] = {
                    "seed": seed,
                    "base_score": base_scores[task_id],
                    "adapted_score": adapted_scores[task_id],
                    "frontier_score": frontier_scores[task_id],
                }
            print(f"adapted scoring complete {alias} seed {seed}", flush=True)

        task_results = []
        for task in eval_tasks:
            task_id = str(task["task_id"])
            rows = [seed_scores[task_id][str(seed)] for seed in SEEDS]
            task_results.append({
                "task_id": task_id,
                "task_class": task["task_class"],
                "base_score": sum(row["base_score"] for row in rows) / len(rows),
                "adapted_score": sum(row["adapted_score"] for row in rows) / len(rows),
                "frontier_score": sum(row["frontier_score"] for row in rows) / len(rows),
                "seed_scores": rows,
            })
        models.append({
            "model_id": alias,
            "model_family": "Qwen2.5 local snapshot",
            "base_model_ref": public_base_id,
            "base_model_id": path.name,
            "base_digest": _model_revision(path),
            "adapter_id": f"LoRA-r{args.rank}-qv-public-synthetic-v1",
            "adapter_digest": _sha256_bytes("".join(sorted(adapter_digests)).encode()),
            "adapter_seed_digests": adapter_digests,
            "device_id": "amd-strix-halo-cpu-transformers-local-1",
            "task_results": task_results,
            "training": {
                "seed_kind": "adapter_training",
                "seeds": list(SEEDS),
                "epochs": args.epochs,
                "steps_per_seed": step_counts,
                "rank": args.rank,
                "target_modules": ["q_proj", "v_proj"],
            },
        })
        del tokenizer
        gc.collect()

    return {
        "schema_version": "1.0",
        "protocol": {
            "benchmark_id": "public-synthetic-operations-mcq-v1",
            "task_manifest_sha256": _sha256_bytes(task_bytes),
            "task_classes": list(TASK_CLASSES),
            "tasks_per_class": len([task for task in eval_tasks
                                     if task["task_class"] == TASK_CLASSES[0]]),
            "train_tasks_per_class": len([task for task in train_tasks
                                           if task["task_class"] == TASK_CLASSES[0]]),
            "split": "training and evaluation task IDs are disjoint",
            "metric": (
                "greedy multiple-choice accuracy, scored 1 for the correct option and 0 otherwise"
            ),
            "grader": {
                "name": "greedy-option-accuracy-scorer",
                "version": "1.0",
                "blinded": True,
                "note": "Scores see the prompt and correct option, not model or adapter labels.",
            },
            "seeds": list(SEEDS),
            "seed_kind": "adapter_training",
            "frontier_reference": "Qwen/Qwen2.5-VL-7B-Instruct",
            "evaluation_mode": "text-only; vision tower unused",
            "device_id": "amd-strix-halo-cpu-transformers-local-1",
            "operator_id": "author-operator-1",
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "peft": _external_peft().__version__,
                "threads": args.threads,
                "dtype": "float32",
            },
            "source_revision": _source_revision(),
            "limitations": [
                "synthetic public-safe tasks are not a sampled user workload",
                "the metric is greedy multiple-choice accuracy, not human task success",
                "one physical machine and CPU-only PyTorch execution",
                "frontier scores are a measured reference model, not a universal frontier",
            ],
        },
        "models": models,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="model alias=local Hugging Face snapshot path; repeat at least three times")
    parser.add_argument("--frontier", required=True,
                        help="local Hugging Face snapshot path for the reference model")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--generate-tasks", action="store_true")
    parser.add_argument("--reuse-adapters", action="store_true",
                        help="score existing seed adapters without retraining")
    args = parser.parse_args(argv)
    if args.generate_tasks or not args.tasks.exists():
        args.tasks.parent.mkdir(parents=True, exist_ok=True)
        tasks = generate_tasks()
        args.tasks.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.tasks}")
    if args.rank < 1 or args.epochs < 1 or args.max_length < 32:
        parser.error("rank and epochs must be positive; max length must be at least 32")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"  models: {len(result['models'])}")
    print(f"  tasks per model: {len(result['models'][0]['task_results'])}")
    print(f"  seeds: {len(result['protocol']['seeds'])} ({result['protocol']['seed_kind']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
