"""Offline tests for citation coverage and conservative source flags."""

import tempfile
import unittest
from pathlib import Path

from paper.audit_citations import (
    audit, parse_bib, _stable_content_type, _stable_final_url,
)
from paper.citation_claim_review import classify_claim


class CitationAuditChecks(unittest.TestCase):
    def test_nested_bibtex_values_are_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.bib"
            path.write_text(
                "@inproceedings{synthetic2024,\n"
                " author = {A. Author},\n"
                " title = {A {Nested} Title},\n"
                " booktitle = {ICML 2024},\n"
                " year = {2024},\n"
                " url = {https://example.test/paper}\n"
                "}\n",
                encoding="utf-8",
            )
            entries = parse_bib(path)
        self.assertEqual(entries["synthetic2024"]["title"],
                         "A {Nested} Title")
        self.assertEqual(entries["synthetic2024"]["entry_type"],
                         "inproceedings")

    def test_audit_reports_missing_and_unused_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bib = root / "references.bib"
            bib.write_text(
                "@article{used2024,\n"
                " author = {A. Author}, title = {Used}, year = {2024},\n"
                " journal = {Patterns}, url = {https://example.test/used}\n"
                "}\n"
                "@misc{unused2024,\n"
                " author = {B. Author}, title = {Unused}, year = {2024},\n"
                " url = {https://example.test/unused}\n"
                "}\n",
                encoding="utf-8",
            )
            tex = root / "main.tex"
            tex.write_text("A claim \\\\cite{used2024,missing2024}.",
                           encoding="utf-8")
            result = audit(bib, (tex,))
        self.assertEqual(result["summary"]["bib_entries"], 2)
        self.assertEqual(result["missing_cited_keys"], ["missing2024"])
        self.assertEqual(result["unused_bib_entries"], ["unused2024"])
        used = next(row for row in result["entries"] if row["key"] == "used2024")
        self.assertEqual(used["source_quality"],
                         "formal_or_peer_reviewed_candidate")
        self.assertEqual(used["citation_occurrences"][0]["line"], 1)

    def test_claim_review_flags_restricted_and_scope_mismatch_sources(self):
        restricted = classify_claim({
            "key": "autoyou_support2026",
            "source_quality": "restricted",
            "flags": ["missing_locator", "private_or_restricted"],
        })
        self.assertEqual(restricted["status"], "not_independently_reproducible")

        pricing = classify_claim({
            "key": "frontier2026",
            "source_quality": "non_peer_reviewed_or_unspecified",
            "flags": [],
        })
        self.assertEqual(pricing["status"], "locator_scope_mismatch")

    def test_arxiv_entries_report_unpinned_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bib = root / "references.bib"
            bib.write_text(
                "@article{arxiv2025,\n"
                " author = {A. Author}, title = {A preprint}, year = {2025},\n"
                " journal = {arXiv preprint arXiv:2501.00001},\n"
                " url = {https://arxiv.org/abs/2501.00001}\n"
                "}\n",
                encoding="utf-8",
            )
            tex = root / "main.tex"
            tex.write_text("A claim \\\\cite{arxiv2025}.", encoding="utf-8")
            result = audit(bib, (tex,))
        row = result["entries"][0]
        self.assertIn("arxiv_version_unpinned", row["flags"])
        self.assertEqual(result["summary"]["arxiv_entries_unpinned"], 1)

    def test_redirect_nonce_is_removed_but_meaningful_query_is_kept(self):
        unstable = (
            "https://link.springer.com/article/10.1007/example?"
            "error=cookies_not_supported&code=random-value")
        self.assertEqual(
            _stable_final_url(unstable),
            "https://link.springer.com/article/10.1007/example")
        meaningful = "https://example.test/pdf?id=paper-123"
        self.assertEqual(_stable_final_url(meaningful), meaningful)

    def test_content_type_normalization_is_stable(self):
        self.assertEqual(
            _stable_content_type(" Text/HTML; charset=UTF-8 "),
            "text/html;charset=utf-8",
        )


if __name__ == "__main__":
    unittest.main()
