"""Validate public provenance before accepting a replication record.

This validator checks the identity contract and listed-file bookkeeping. It
does not turn a valid manifest into evidence: statistics.py still requires
matching raw model records and reports whether a pair is actually comparable.
Use --require-independent as a release gate when an independent result is
required; the normal check intentionally passes with zero pairs and reports
that the study is still pending.
"""
import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "measure" / "replication_manifest.json"
DEFAULT_RESULTS = ROOT / "measure" / "results"
IDENTITY_FIELDS = ("target_id", "operator_id", "environment_id")
COLLECTION_ROLES = {
    "reference_single_request",
    "reference_ladder",
    "standard_concurrent_batch",
    "pilot_concurrent_batch",
    "primary_single_request",
    "same_target_repeatability",
    "independent_replication",
}
CONCURRENT_ROLES = {
    "standard_concurrent_batch",
    "pilot_concurrent_batch",
    "independent_replication",
}
INDEPENDENT_BATCH_SIZES = [1, 2, 4, 8]
PUBLIC_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_manifest_document(manifest: object) -> List[str]:
    """Return human-readable structural and provenance errors."""
    errors: List[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if manifest.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if not isinstance(manifest.get("description"), str) or not manifest[
            "description"].strip():
        errors.append("description must be a non-empty string")

    rule = manifest.get("independence_rule")
    required = [
        "different physical target_id",
        "different operator_id",
        "different environment_id",
    ]
    if not isinstance(rule, dict):
        errors.append("independence_rule must be an object")
    else:
        if rule.get("independent_replication_requires") != required:
            errors.append("independence_rule does not match the three identity fields")
        if rule.get("same_target_replay_classification") != "repeatability":
            errors.append("same-target replays must be classified as repeatability")
        if rule.get("missing_identity_classification") != "unresolved":
            errors.append("missing identity must be classified as unresolved")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        errors.append("files must be a non-empty object")
    else:
        for filename, record in files.items():
            if (not isinstance(filename, str)
                    or Path(filename).name != filename
                    or not filename.endswith(".json")):
                errors.append(f"{filename!r}: file key must be a plain .json name")
            if not isinstance(record, dict):
                errors.append(f"{filename}: provenance must be an object")
                continue
            for field in IDENTITY_FIELDS:
                value = record.get(field)
                if not isinstance(value, str) or not PUBLIC_LABEL.fullmatch(value):
                    errors.append(
                        f"{filename}: {field} must be a non-personal public label")
            if record.get("collection_role") not in COLLECTION_ROLES:
                errors.append(f"{filename}: unknown collection_role")

    template = manifest.get("contributor_template")
    if not isinstance(template, dict):
        errors.append("contributor_template must be an object")
    else:
        for field in IDENTITY_FIELDS:
            if not isinstance(template.get(field), str) or not template[field].strip():
                errors.append(f"contributor_template: {field} must be a non-empty string")
        if template.get("collection_role") != "independent_replication":
            errors.append("contributor_template must describe independent_replication")
    return errors


def missing_result_files(manifest: Dict[str, object], results_dir: Path) -> List[str]:
    """Return manifest entries that are not present in the raw-results folder."""
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return []
    return sorted(name for name in files if not (results_dir / name).is_file())


def provenance_qualified_pairs(manifest: Dict[str, object]) -> List[Tuple[str, str]]:
    """Return file pairs whose three declared identities all differ."""
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return []
    pairs = []
    for first, second in combinations(sorted(files), 2):
        left, right = files[first], files[second]
        if (isinstance(left, dict) and isinstance(right, dict)
                and all(left.get(field) and right.get(field)
                        and left[field] != right[field]
                        for field in IDENTITY_FIELDS)):
            pairs.append((first, second))
    return pairs


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_raw_record_document(record: object, filename: str,
                                 collection_role: str) -> List[str]:
    """Validate the raw contract needed for a concurrent contribution.

    The provenance manifest answers who and where a record came from. This
    second check answers whether the record contains the protocol fields that
    make a cross-device comparison meaningful. Existing pilot and standard
    records are checked for shape; an independent contribution additionally
    needs the full batch matrix, at least ten repetitions, and an Ollama model
    digest in every model row.
    """
    if collection_role not in CONCURRENT_ROLES:
        return []
    errors: List[str] = []
    prefix = f"{filename}: "
    if not isinstance(record, dict):
        return [prefix + "raw record must be an object"]
    if record.get("schema_version") != "2.0":
        errors.append(prefix + "concurrent raw record schema_version must be 2.0")
    if record.get("harness") != "measure/bench.py --batch-sizes":
        errors.append(prefix + "harness must be measure/bench.py --batch-sizes")
    protocol = record.get("protocol")
    if not isinstance(protocol, dict):
        errors.append(prefix + "protocol must be an object")
        protocol = {}
    positive_int_fields = ("requested_tokens_per_request", "reps", "num_ctx")
    for field in positive_int_fields:
        value = protocol.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(prefix + f"protocol.{field} must be a positive integer")
    batch_sizes = protocol.get("batch_sizes")
    if (not isinstance(batch_sizes, list)
            or not batch_sizes
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                   for value in batch_sizes)
            or batch_sizes != sorted(set(batch_sizes))):
        errors.append(prefix + "protocol.batch_sizes must be sorted positive integers")
        batch_sizes = []
    for field in ("warmup_discarded", "raw_mode", "unique_nonce_per_request"):
        if not isinstance(protocol.get(field), bool):
            errors.append(prefix + f"protocol.{field} must be boolean")
    batches = record.get("batches")
    if not isinstance(batches, list) or not batches:
        errors.append(prefix + "batches must be a non-empty list")
        return errors

    seen = set()
    digests: Dict[str, set] = {}
    for index, row in enumerate(batches):
        row_prefix = f"{prefix}batches[{index}]: "
        if not isinstance(row, dict):
            errors.append(row_prefix + "row must be an object")
            continue
        model = row.get("model")
        batch_size = row.get("batch_size")
        if not isinstance(model, str) or not model.strip():
            errors.append(row_prefix + "model must be a non-empty string")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            errors.append(row_prefix + "batch_size must be an integer")
        elif batch_size not in batch_sizes:
            errors.append(row_prefix + "batch_size is not declared in protocol.batch_sizes")
        model_key = model if isinstance(model, str) else f"<invalid-model-{index}>"
        batch_key = (batch_size if isinstance(batch_size, int)
                     and not isinstance(batch_size, bool)
                     else f"<invalid-batch-{index}>")
        key = (model_key, batch_key)
        if key in seen:
            errors.append(row_prefix + "duplicate model and batch_size row")
        seen.add(key)
        if collection_role != "independent_replication":
            continue
        row_reps = row.get("reps")
        if not isinstance(row_reps, int) or isinstance(row_reps, bool) or row_reps < 10:
            errors.append(row_prefix + "independent replication requires at least 10 reps")
        info = row.get("model_info")
        if not isinstance(info, dict) or not info.get("available"):
            errors.append(row_prefix + "model_info.available must be true")
        digest = info.get("digest") if isinstance(info, dict) else None
        if not isinstance(digest, str) or not digest.strip():
            errors.append(row_prefix + "model_info.digest is required for independent replication")
        elif isinstance(model, str):
            digests.setdefault(model, set()).add(digest)

    if collection_role == "independent_replication":
        if batch_sizes != INDEPENDENT_BATCH_SIZES:
            errors.append(prefix + "independent replication requires batch sizes 1, 2, 4, and 8")
        models = {key[0] for key in seen
                  if isinstance(key[0], str) and not key[0].startswith("<invalid-")}
        for model in sorted(models):
            observed_sizes = {key[1] for key in seen if key[0] == model}
            if observed_sizes != set(INDEPENDENT_BATCH_SIZES):
                errors.append(prefix + f"{model} is missing one or more independent batch-size rows")
            if len(digests.get(model, set())) > 1:
                errors.append(prefix + f"{model} has inconsistent model digests")
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--require-independent",
        action="store_true",
        help="fail unless at least one provenance-qualified pair exists",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        manifest = _load(args.manifest)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read manifest: {exc}")
        return 1

    errors = validate_manifest_document(manifest)
    missing = missing_result_files(manifest, args.results)
    if missing:
        errors.extend(f"listed raw file is missing: {name}" for name in missing)
    files = manifest.get("files", {})
    if isinstance(files, dict):
        for filename, provenance in files.items():
            if filename in missing or not isinstance(provenance, dict):
                continue
            role = provenance.get("collection_role")
            if role not in CONCURRENT_ROLES:
                continue
            try:
                record = _load(args.results / filename)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{filename}: cannot read raw record: {exc}")
                continue
            errors.extend(validate_raw_record_document(record, filename, role))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    pairs = provenance_qualified_pairs(manifest)
    print(f"PASS: manifest schema and provenance labels ({len(manifest['files'])} files)")
    print(f"provenance-qualified independent pairs: {len(pairs)}")
    if pairs:
        for first, second in pairs:
            print(f"  {first} <-> {second}")
    else:
        print("status: no independent result is currently available")
    if args.require_independent and not pairs:
        print("FAIL: --require-independent requested, but no qualifying pair exists")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
