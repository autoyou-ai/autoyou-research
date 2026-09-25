"""Write or check review/source-manifest.json for the revised paper.

Run from the repository root:
    python review/source_manifest.py           rewrite after an intended change
    python review/source_manifest.py --check   verify that a checkout matches
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from audit import INPUTS

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "review" / "source-manifest.json"
FILES = ("paper/right-sized-edge.tex", "paper/right-sized-edge.pdf",
         "paper/review_numbers.tex", "paper/review_references.bib",
         "review/audit.py", "review/test_audit.py", "review/build_paper.py",
         "review/audit_results.json") + tuple(f"measure/results/{n}" for n in INPUTS)


def digests():
    return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in FILES}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify instead of rewriting")
    args = parser.parse_args(argv)
    if args.check:
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
        bad = [rel for rel, digest in digests().items() if recorded.get(rel) != digest]
        bad += [rel for rel in recorded if rel not in FILES]
        for rel in bad:
            print(f"MISMATCH: {rel}")
        print("FAIL" if bad else f"PASS: {len(FILES)} files match the manifest")
        return 1 if bad else 0
    manifest = {"revision": "2026-09-20",
                "status": "alphaXiv preprint v1 (2026-09-21); not peer reviewed",
                "files": digests()}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
