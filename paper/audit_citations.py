"""Audit BibTeX entries against manuscript usage and source locators.

The audit is intentionally conservative. It checks bibliographic completeness,
manuscript coverage, source-type warnings, exact citation contexts, and
optional URL/DOI reachability. It cannot prove semantic support by parsing
text, so it records every claim context for a human source check.

Examples:

    python paper/audit_citations.py
    python paper/audit_citations.py --network --output paper/citation_audit.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIB = ROOT / "paper" / "references.bib"
DEFAULT_TEX = (ROOT / "paper" / "p2p-inference.tex", ROOT / "paper" / "adaptation.tex")
DEFAULT_OUTPUT = ROOT / "paper" / "citation_audit.json"
DEFAULT_MARKDOWN = ROOT / "paper" / "CITATION_AUDIT.md"


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _entry_spans(text: str) -> Iterable[Tuple[str, str, str]]:
    """Yield entry type, key and body while respecting nested BibTeX braces."""
    pattern = re.compile(r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,")
    for match in pattern.finditer(text):
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            yield match.group(1).lower(), match.group(2), text[match.end():index - 1]


def _split_value(body: str, start: int, end: int) -> str:
    value = body[start:end].strip().rstrip(",").strip()
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1].strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].strip()
    return value


def _fields(body: str) -> Dict[str, str]:
    matches = list(re.finditer(
        r"(?m)^\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", body))
    out = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        out[match.group(1).lower()] = _split_value(
            body, match.end(), end)
    return out


def parse_bib(path: Path) -> Dict[str, Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries = {}
    for entry_type, key, body in _entry_spans(text):
        fields = _fields(body)
        fields["entry_type"] = entry_type
        fields["key"] = key
        entries[key] = fields
    return entries


def _citation_contexts(paths: Sequence[Path]) -> Dict[str, List[Dict[str, object]]]:
    contexts: Dict[str, List[Dict[str, object]]] = {}
    pattern = re.compile(r"\\cite[A-Za-z*]*\s*\{([^}]+)\}")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            start = max(0, match.start() - 500)
            end = min(len(text), match.end() + 500)
            context = re.sub(r"\s+", " ", text[start:end]).strip()
            for key in match.group(1).split(","):
                key = key.strip()
                if not key:
                    continue
                contexts.setdefault(key, []).append({
                    "file": _display(path),
                    "line": line,
                    "context": context,
                })
    return contexts


def _source_quality(entry: Dict[str, str]) -> Tuple[str, List[str]]:
    combined = " ".join(entry.get(key, "") for key in (
        "author", "title", "journal", "booktitle", "note", "institution",
        "urldate"
    )).lower()
    flags = []
    if not entry.get("author"):
        flags.append("missing_author")
    if not entry.get("title"):
        flags.append("missing_title")
    if not entry.get("year"):
        flags.append("missing_year")
    if not entry.get("url") and not entry.get("doi"):
        flags.append("missing_locator")
    if "private" in combined or "not redistributable" in combined:
        flags.append("private_or_restricted")
    if "not peer-reviewed" in combined or "not peer reviewed" in combined:
        flags.append("not_peer_reviewed")
    is_arxiv = (
        "arxiv preprint" in combined
        or "arxiv:" in combined
        or "arxiv.org/" in entry.get("url", "").lower()
    )
    if is_arxiv:
        flags.append("arxiv_preprint")
        arxiv_locator = (
            "arxiv preprint" in entry.get("journal", "").lower()
            or "arxiv.org/" in entry.get("url", "").lower()
        )
        if arxiv_locator and not entry.get("version"):
            flags.append("arxiv_version_unpinned")
    if "vendor" in combined or "official" in combined or "rfc" in combined:
        flags.append("vendor_or_official_material")
    if "trade press" in combined:
        flags.append("trade_press")
    if "community" in combined or "home-enthusiast" in combined:
        flags.append("community_source")
    if "third-party" in combined or "third party" in combined:
        flags.append("third_party_source")
    if "assumption" in combined or "authorial estimate" in combined:
        flags.append("authorial_assumption_or_estimate")
    if "accessed" not in combined and not entry.get("urldate") and entry.get("url"):
        flags.append("no_access_date")

    entry_type = entry.get("entry_type", "")
    venue_marker = re.search(
        r"\b(ICLR|ICML|NeurIPS|TMLR|FAccT|Patterns|ISCA|"
        r"Communications of the ACM|Journal of Industrial Ecology)\b",
        combined,
        re.IGNORECASE,
    )
    has_non_arxiv_venue = bool(
        (entry.get("journal") or entry.get("booktitle"))
        and "arxiv preprint" not in combined
    )
    formal = bool(venue_marker) or (
        entry_type in {"article", "inproceedings", "incollection", "book"}
        and has_non_arxiv_venue
    )
    if "private_or_restricted" in flags:
        return "restricted", flags
    if formal:
        return "formal_or_peer_reviewed_candidate", flags
    if "vendor_or_official_material" in flags:
        return "official_or_vendor", flags
    if any(flag in flags for flag in (
        "trade_press", "community_source", "third_party_source"
    )):
        return "secondary_non_peer_reviewed", flags
    if "authorial_assumption_or_estimate" in flags:
        return "assumption_or_internal_artifact", flags
    return "non_peer_reviewed_or_unspecified", flags


def _curl_probe(url: str, timeout: float) -> Optional[Dict[str, object]]:
    """Fallback for environments whose Python CA store cannot validate HTTPS."""
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, "-L", "-sS", "--max-time", str(timeout),
             "-o", os.devnull, "-w", "%{http_code} %{url_effective}", url],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    fields = completed.stdout.strip().split(" ", 1)
    try:
        status = int(fields[0])
    except (IndexError, ValueError):
        return {
            "status": None,
            "method": "curl",
            "error": completed.stderr.strip()[:240] or "curl returned no status",
        }
    result = {
        "status": status,
        "method": "curl",
        "final_url": _stable_final_url(
            fields[1] if len(fields) > 1 else url),
    }
    if completed.returncode != 0:
        result["error"] = completed.stderr.strip()[:240]
    return result


def _stable_final_url(url: str) -> str:
    """Remove known cookie-consent nonce parameters from redirect records."""
    parsed = urllib.parse.urlsplit(url)
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if (any(key == "code" for key, _ in pairs)
            and any(key == "error" and value == "cookies_not_supported"
                    for key, value in pairs)):
        pairs = [(key, value) for key, value in pairs
                 if key not in {"code", "error"}]
    query = urllib.parse.urlencode(pairs)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path,
                                    query, ""))


def _stable_content_type(value: Optional[str]) -> Optional[str]:
    """Normalize equivalent server MIME-header formatting."""
    if not value:
        return value
    return re.sub(r"\s*;\s*", ";", value.strip()).lower()


def _doi_kind(doi: Optional[str]) -> Optional[str]:
    """Classify a DOI without treating an arXiv DOI as a venue DOI."""
    if not doi:
        return None
    if doi.lower().startswith("10.48550/arxiv."):
        return "arxiv_issued"
    return "venue_or_publisher"


def _probe(url: str, timeout: float) -> Dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "autoyou-research-citation-audit/1.0"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "status": response.status,
                "final_url": _stable_final_url(response.geturl()),
                "content_type": _stable_content_type(
                    response.headers.get("Content-Type")),
            }
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 403, 405, 406, 429, 500, 501):
            return {"status": exc.code, "error": str(exc)}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        fallback = _curl_probe(url, timeout)
        if fallback:
            return fallback
        return {"status": None, "error": str(exc)[:240]}

    get_request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "autoyou-research-citation-audit/1.0",
            "Range": "bytes=0-1023",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(get_request, timeout=timeout) as response:
            response.read(1024)
            return {
                "status": response.status,
                "final_url": _stable_final_url(response.geturl()),
                "content_type": _stable_content_type(
                    response.headers.get("Content-Type")),
            }
    except (urllib.error.HTTPError, urllib.error.URLError,
            TimeoutError, OSError) as exc:
        fallback = _curl_probe(url, timeout)
        if fallback:
            return fallback
        return {"status": getattr(exc, "code", None), "error": str(exc)[:240]}


def audit(bib_path: Path, tex_paths: Sequence[Path], network: bool = False,
          timeout: float = 20.0) -> Dict[str, object]:
    entries = parse_bib(bib_path)
    contexts = _citation_contexts(tex_paths)
    cited = sorted(contexts)
    missing = sorted(key for key in cited if key not in entries)
    unused = sorted(key for key in entries if key not in contexts)
    rows = []
    probes = {}
    locators = {}
    for key, entry in sorted(entries.items()):
        quality, flags = _source_quality(entry)
        combined = " ".join(entry.get(field, "") for field in (
            "journal", "booktitle", "note", "institution")).lower()
        venue_marker = re.search(
            r"\b(ICLR|ICML|NeurIPS|TMLR|FAccT|Patterns|ISCA|"
            r"Communications of the ACM|Journal of Industrial Ecology)\b",
            combined,
            re.IGNORECASE,
        )
        locator = entry.get("url") or (
            "https://doi.org/" + entry["doi"] if entry.get("doi") else None)
        locators[key] = locator
        rows.append({
            "key": key,
            "entry_type": entry.get("entry_type"),
            "author": entry.get("author"),
            "title": entry.get("title"),
            "year": entry.get("year"),
            "journal": entry.get("journal"),
            "booktitle": entry.get("booktitle"),
            "publisher": entry.get("publisher"),
            "pages": entry.get("pages"),
            "note": entry.get("note"),
            "version": entry.get("version"),
            "urldate": entry.get("urldate"),
            "venue_marker": venue_marker.group(0) if venue_marker else None,
            "doi": entry.get("doi"),
            "doi_kind": _doi_kind(entry.get("doi")),
            "url": entry.get("url"),
            "arxiv_locator": (
                "arxiv preprint" in entry.get("journal", "").lower()
                or "arxiv.org/" in entry.get("url", "").lower()
            ),
            "source_quality": quality,
            "flags": flags,
            "cited": key in contexts,
            "citation_occurrences": contexts.get(key, []),
        })
    if network:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_map = {
                executor.submit(_probe, url, timeout): key
                for key, url in locators.items() if url
            }
            for future in concurrent.futures.as_completed(future_map):
                probes[future_map[future]] = future.result()
        for row in rows:
            if row["key"] in probes:
                row["locator_check"] = probes[row["key"]]
                if probes[row["key"]].get("status") is None:
                    row["flags"].append("locator_unreachable")
                elif int(probes[row["key"]]["status"]) in (401, 403):
                    row["flags"].append("locator_access_denied")
                elif int(probes[row["key"]]["status"]) >= 400:
                    row["flags"].append("locator_http_error")
    return {
        "schema_version": "1.0",
        "audit_method": (
            "BibTeX completeness, manuscript cite-key coverage, conservative "
            "source-type flags, exact manuscript contexts, and optional "
            "locator reachability. Semantic claim support remains a manual "
            "source-reading step."
        ),
        "inputs": {
            "bib": _display(bib_path),
            "manuscripts": [_display(path) for path in tex_paths],
        },
        "network_probe": network,
        "summary": {
            "bib_entries": len(entries),
            "cited_keys": len(cited),
            "missing_cited_keys": len(missing),
            "unused_bib_entries": len(unused),
            "entries_with_flags": sum(bool(row["flags"]) for row in rows),
            "formal_or_peer_reviewed_candidates": sum(
                row["source_quality"] == "formal_or_peer_reviewed_candidate"
                for row in rows),
            "formal_candidates_with_venue_doi": sum(
                row["source_quality"] == "formal_or_peer_reviewed_candidate"
                and row["doi_kind"] == "venue_or_publisher"
                for row in rows),
            "formal_candidates_with_only_arxiv_doi": sum(
                row["source_quality"] == "formal_or_peer_reviewed_candidate"
                and row["doi_kind"] == "arxiv_issued"
                for row in rows),
            "formal_candidates_without_venue_doi": sum(
                row["source_quality"] == "formal_or_peer_reviewed_candidate"
                and row["doi_kind"] != "venue_or_publisher"
                for row in rows),
            "restricted_or_private": sum(
                row["source_quality"] == "restricted" for row in rows),
            "entries_with_doi": sum(bool(row["doi"]) for row in rows),
            "entries_with_url": sum(bool(row["url"]) for row in rows),
            "entries_with_version": sum(bool(row["version"]) for row in rows),
            "arxiv_entries": sum(row["arxiv_locator"] for row in rows),
            "arxiv_entries_with_version": sum(
                row["arxiv_locator"] and bool(row["version"])
                for row in rows
            ),
            "arxiv_entries_unpinned": sum(
                row["arxiv_locator"] and "arxiv_version_unpinned" in row["flags"]
                for row in rows
            ),
            "entries_with_venue_field": sum(
                bool(row["journal"] or row["booktitle"]) for row in rows),
            "entries_missing_locator": sum(
                not (row["doi"] or row["url"]) for row in rows),
        },
        "missing_cited_keys": missing,
        "unused_bib_entries": unused,
        "entries": rows,
        "manual_semantic_review": {
            "status": "required",
            "reason": (
                "A URL resolving successfully does not establish that it "
                "supports the exact sentence where it is cited. Review every "
                "citation_occurrences context against the source and record "
                "any narrower claim wording in the manuscript."
            ),
        },
    }


def write_markdown(result: Dict[str, object], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Citation audit",
        "",
        "This report is generated by paper/audit_citations.py. Locator reachability",
        "is not semantic verification. Each citation context is retained so a",
        "reviewer can check the source against the exact sentence.",
        "",
        f"- BibTeX entries: {summary['bib_entries']}",
        f"- Cited keys: {summary['cited_keys']}",
        f"- Missing cited keys: {summary['missing_cited_keys']}",
        f"- Unused BibTeX entries: {summary['unused_bib_entries']}",
        f"- Entries with warnings: {summary['entries_with_flags']}",
        f"- Formal or peer-reviewed candidates: {summary['formal_or_peer_reviewed_candidates']}",
        f"- Formal candidates with a venue or publisher DOI: {summary['formal_candidates_with_venue_doi']}",
        f"- Formal candidates with only an arXiv-issued DOI: {summary['formal_candidates_with_only_arxiv_doi']}",
        f"- Formal candidates without a venue or publisher DOI: {summary['formal_candidates_without_venue_doi']}",
        f"- Restricted or private entries: {summary['restricted_or_private']}",
        f"- Entries with DOI: {summary['entries_with_doi']}",
        f"- Entries with URL: {summary['entries_with_url']}",
        f"- Entries with explicit version: {summary['entries_with_version']}",
        f"- arXiv entries: {summary['arxiv_entries']}",
        f"- arXiv entries with explicit version: {summary['arxiv_entries_with_version']}",
        f"- arXiv entries with unpinned version: {summary['arxiv_entries_unpinned']}",
        f"- Entries with journal or booktitle: {summary['entries_with_venue_field']}",
        f"- Entries missing locator: {summary['entries_missing_locator']}",
        "",
        "| Key | Source class | Locator | Warnings | Cited at |",
        "|---|---|---|---|---|",
    ]
    for row in result["entries"]:
        locator = row.get("doi") or row.get("url") or "missing"
        flags = ", ".join(row["flags"]) or "none"
        locations = ", ".join(
            f"{item['file']}:{item['line']}"
            for item in row["citation_occurrences"]
        ) or "unused"
        lines.append(
            f"| {row['key']} | {row['source_quality']} | "
            f"{locator} | {flags} | {locations} |"
        )
    lines.extend([
        "",
        "## Manual claim review",
        "",
        "The following contexts must be checked against the linked source before",
        "the paper is presented as evidence. This list is intentionally complete.",
        "",
    ])
    for row in result["entries"]:
        for occurrence in row["citation_occurrences"]:
            context = occurrence["context"].replace("|", "\\|")
            lines.append(
                f"- {row['key']} at {occurrence['file']}:{occurrence['line']}: "
                f"{context}"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Warnings describe source type and provenance, not automatic rejection.",
        "Peer-reviewed candidates still need claim-level checking. Vendor, trade,",
        "community, third-party, authorial, and restricted sources should support",
        "only claims that are explicitly scoped to what those sources establish.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--network", action="store_true",
                        help="probe each URL or DOI locator")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)
    result = audit(args.bib, DEFAULT_TEX, args.network, args.timeout)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_markdown(result, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")
    return 0 if not result["missing_cited_keys"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
