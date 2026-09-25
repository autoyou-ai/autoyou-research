"""Write a path-stable manifest for a research release.

The manifest records hashes and build metadata without embedding local paths,
private prompts, transcripts, or checkpoint contents. It is intentionally
generated after the model and paper build has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "release_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _paths() -> Iterable[tuple[str, str]]:
    fixed = {
        "paper/p2p-inference.tex": "manuscript",
        "paper/adaptation.tex": "manuscript",
        "paper/references.bib": "bibliography",
        "paper/p2p-inference.pdf": "compiled manuscript",
        "paper/adaptation.pdf": "compiled manuscript",
        "paper/empirical.tex": "generated empirical appendix",
        "models/run_all.py": "reproducibility pipeline",
        "models/verify_paper.py": "paper number verifier",
        "models/adaptation_eval.py": "adaptation evaluator",
        "models/run_adaptation_experiment.py": "adaptation experiment runner",
        "models/adaptation_template_holdout.py": "prompt-surface holdout runner",
        "models/adaptation_semantic_holdout.py": "new-semantic holdout runner",
        "models/adaptation_semantic_holdout_evaluation.json": "new-semantic holdout evaluation",
        "measure/bench.py": "measurement harness",
        "measure/statistics.py": "measurement statistics",
        "measure/validate_replication_manifest.py": "provenance validator",
        "paper/audit_citations.py": "citation audit tool",
        "paper/citation_claim_review.py": "citation claim review tool",
        "models/results.json": "generated model results",
        "measure/statistics.json": "generated measurement statistics",
        "paper/citation_audit.json": "citation audit",
        "paper/CITATION_AUDIT.md": "human-readable citation audit",
        "paper/citation_claim_review.json": "citation claim review",
        "paper/CITATION_CLAIM_REVIEW.md": "human-readable citation claim review",
        "paper/publication_metadata.json": "publication metadata",
        "CITATION.cff": "citation metadata",
        "measure/hardware_matrix.json": "hardware coverage metadata",
        "measure/replication_manifest.json": "measurement provenance metadata",
        "measure/replication_manifest.schema.json": "measurement provenance schema",
        "measure/REPLICATION_PROTOCOL.md": "replication protocol",
        "measure/adaptation-multimodel.schema.json": "adaptation evaluation schema",
        "measure/ADAPTATION_EVALUATION.md": "adaptation benchmark record",
        "models/adaptation_template_holdout_evaluation.json": "prompt-surface holdout evaluation",
        "requirements-adaptation.txt": "adaptation benchmark environment",
        "paper/CLAIMS_LEDGER.md": "claim boundary ledger",
        "paper/FOLLOW_UP_STUDY.md": "follow-up study protocol",
        "paper/REVIEW_CRITIQUE.md": "critical review",
    }
    for relative, role in fixed.items():
        yield relative, role
    for pattern, role in (
        ("measure/results/*.json", "raw public measurement record"),
        ("measure/results/adaptation_adapters/**/*", "public synthetic adapter artifact"),
        ("figures/fig_*", "generated figure"),
    ):
        for path in sorted(ROOT.glob(pattern)):
            yield path.relative_to(ROOT).as_posix(), role


def build_manifest() -> dict[str, object]:
    files: list[dict[str, object]] = []
    seen: set[str] = set()
    for relative, role in _paths():
        if relative in seen or relative == "paper/release_manifest.json":
            continue
        seen.add(relative)
        path = ROOT / Path(relative)
        if not path.is_file():
            continue
        files.append({
            "path": relative,
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return {
        "schema_version": "1.0",
        "release": {
            "version": "0.4.0",
            "date": "2026-09-24",
            "status": "public research repository release",
            "repository": "https://github.com/autoyou-ai/autoyou-research",
            "source_revision": _git_revision(),
            "source_revision_type": "git HEAD when the manifest was written; the release commit adds this manifest",
        },
        "build": {
            "source_date_epoch": 1789344000,
            "model_pipeline": "py -3 models\\run_all.py",
            "paper_pipeline": "pdflatex -> bibtex -> pdflatex -> pdflatex in paper/",
            "statistics": "py -3 measure\\statistics.py --resamples 20000",
            "replication_manifest_validation": "py -3 measure\\validate_replication_manifest.py",
            "citation_audit": "py -3 paper\\audit_citations.py --network",
            "adaptation_benchmark": "py -3 models\\run_adaptation_experiment.py --model qwen25-0.5b=LOCAL_SNAPSHOT --model qwen25-3b=LOCAL_SNAPSHOT --model qwen25-vl-3b=LOCAL_SNAPSHOT --frontier LOCAL_FRONTIER --tasks measure/results/adaptation_tasks.json --output measure/results/adaptation-multimodel.json --adapter-root measure/results/adaptation_adapters",
            "adaptation_evaluation": "py -3 models\\adaptation_eval.py --resamples 20000",
            "adaptation_template_holdout": "py -3 models\\adaptation_template_holdout.py --model qwen25-0.5b=LOCAL_SNAPSHOT --model qwen25-3b=LOCAL_SNAPSHOT --model qwen25-vl-3b=LOCAL_SNAPSHOT --frontier LOCAL_FRONTIER --reuse-adapters --tasks measure/results/adaptation-template-holdout-tasks.json --output measure/results/adaptation-template-holdout.json --adapter-root measure/results/adaptation_adapters",
            "adaptation_template_holdout_evaluation": "py -3 models\\adaptation_eval.py --input measure/results/adaptation-template-holdout.json --output models/adaptation_template_holdout_evaluation.json --resamples 20000",
            "adaptation_semantic_holdout": "py -3 models\\adaptation_semantic_holdout.py --model qwen25-0.5b=LOCAL_SNAPSHOT --model qwen25-3b=LOCAL_SNAPSHOT --model qwen25-vl-3b=LOCAL_SNAPSHOT --frontier LOCAL_FRONTIER --reuse-adapters --tasks measure/results/adaptation-semantic-holdout-tasks.json --output measure/results/adaptation-semantic-holdout.json --adapter-root measure/results/adaptation_adapters",
            "adaptation_semantic_holdout_evaluation": "py -3 models\\adaptation_eval.py --input measure/results/adaptation-semantic-holdout.json --output models/adaptation_semantic_holdout_evaluation.json --resamples 20000",
            "python": platform.python_version(),
            "matplotlib": _version("matplotlib"),
            "numpy": _version("numpy"),
        },
        "data_availability": {
            "status": "partial",
            "public": [
                "source code and deterministic model parameters",
                "generated figures and manuscripts",
                "raw public measurement records in measure/results/",
                "public synthetic adaptation task manifest, scores and adapter artifacts",
                "supplemental prompt-surface holdout manifest, scores and evaluation",
                "new-semantic holdout manifest, scores and evaluation",
                "citation and reproducibility audits",
            ],
            "restricted": [
                "the adapter evaluation input from one deployed assistant",
                "prompts, answers, transcripts, and private training material",
            ],
        },
        "files_exclude_self": True,
        "files": files,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
