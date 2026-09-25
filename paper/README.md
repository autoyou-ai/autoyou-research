# Paper release notes

| File | Paper | Status |
| --- | --- | --- |
| [`p2p-inference.pdf`](p2p-inference.pdf) | Quantifying Carbon, Water, Cost, and Capability in Peer-to-Peer Inference | [alphaXiv preprint v1](https://www.alphaxiv.org/abs/2609.carbon-water-cost-p2p-inference), 2026-09-21 |
| [`right-sized-edge.pdf`](right-sized-edge.pdf) | Right-Sized AI at the Edge: Reproducible Measurements and Conditional Sustainability Analysis | [alphaXiv preprint v1](https://www.alphaxiv.org/abs/2609.right-sized-ai-edge-sustainability), 2026-09-21 |
| [`adaptation.pdf`](adaptation.pdf) | Teaching at the Edge: On-Device Parameter-Efficient Adaptation and What It Actually Buys | Research release v0.3.0 |
| [`archive/track-a-legacy.pdf`](archive/track-a-legacy.pdf) | Earlier Track A draft | Historical; see [archive notes](archive/README.md) |

None of these is a peer-reviewed publication. Journal venue and DOI fields stay
`null` in [`publication_metadata.json`](publication_metadata.json); no venue,
acceptance record, or identifier is invented. The repository PDFs carry the
organizational byline *AutoYou Research*; their text is otherwise identical to the
alphaXiv versions.

The revised working paper has its own offline audit and build in
[`../review/`](../review/): `python review/audit.py` regenerates its tables and
figure, and `python review/build_paper.py` builds `right-sized-edge.pdf`.
The rest of this file covers the footprint and adaptation papers.

## Build and audit

From the repository root:

```powershell
$env:SOURCE_DATE_EPOCH = "1789344000"
$env:FORCE_SOURCE_DATE = "1"
py -3 models\run_all.py
py -3 models\verify_paper.py
py -3 measure\validate_replication_manifest.py
py -3 paper\audit_citations.py --network
Set-Location paper
pdflatex -interaction=nonstopmode -halt-on-error p2p-inference.tex
bibtex p2p-inference
pdflatex -interaction=nonstopmode -halt-on-error p2p-inference.tex
pdflatex -interaction=nonstopmode -halt-on-error p2p-inference.tex
pdflatex -interaction=nonstopmode -halt-on-error adaptation.tex
bibtex adaptation
pdflatex -interaction=nonstopmode -halt-on-error adaptation.tex
pdflatex -interaction=nonstopmode -halt-on-error adaptation.tex
```

`SOURCE_DATE_EPOCH` and `FORCE_SOURCE_DATE` make PDF metadata reproducible.
The same commands work in a POSIX shell after translating the environment
variable and path syntax. `latexmk` is also sufficient where its Perl runtime
is installed and configured.

`CITATION_AUDIT.md` reports citation-key coverage, metadata quality, locator
checks, and the context in which every cited key appears. A reachable URL is
not treated as proof that the source supports the exact sentence. Manual
semantic review remains a publication step. The current critical review and
counterclaim matrix is in [`REVIEW_CRITIQUE.md`](REVIEW_CRITIQUE.md). The
per-key conservative source triage is in
[`CITATION_CLAIM_REVIEW.md`](CITATION_CLAIM_REVIEW.md); it deliberately
certifies zero exact semantic claims until each cited context is checked.

After a successful rebuild, run `py -3 paper\release_manifest.py` to write
[`release_manifest.json`](release_manifest.json). It records path-relative
SHA-256 hashes for the papers, figures, raw public records, and audit
outputs, plus the build environment and base Git revision. The
manifest excludes itself and does not contain private prompts, transcripts, or
checkpoint contents.

The raw measurement provenance is declared separately in
[`../measure/replication_manifest.json`](../measure/replication_manifest.json).
Its identity fields prevent a same-target replay from being described as an
independent replication.

## Evidence boundary

Track A combines direct local measurements with modelled workload mix, cloud
baselines, grid factors, pricing, and water factors. Its headline percentages
are conditional estimates, not a random sample of the world-wide inference
fleet.

Track B has a structural coverage result and a public multi-model, multi-task
accuracy evaluation. The measured result is limited to 3 model clusters, 5
synthetic task classes, 20 held-out tasks per class and 3 adapter seeds on one
physical host. It must not be generalized to the user-workload mix, human
quality, or independent-device replication. The `+2/+6/+14` scenario and the
`88.4% -> 93.9%` change remain routing-sensitivity inputs, not empirical
effect sizes. The private one-deployment adapter observations remain
deployment-reported evidence with unresolved provenance limits.

A supplemental prompt-surface holdout changes the surrounding instructions
while reusing the public evaluation records. It reports 80.0% -> 90.0%, or
+10.0 percentage points with a +5.0 to +15.0 model-cluster interval and
`p=0.250`. Because it reuses the same semantic records and tests one wording
variant on one host, it is a robustness probe rather than an independent
replication or a product-quality result.

A separate new-semantic holdout uses 100 fresh synthetic records, 20 per
class, with new record values and task formats. It reports 90.3% -> 91.8%, or
+1.4 percentage points with a -3.3 to +7.7 model-cluster interval and exact
`p=1.000`. The model-level changes are +7.7, -3.3, and 0.0 percentage points,
so this is a negative or uncertain transfer check, not evidence of broad
adaptation uplift. It remains a one-host, synthetic, automatically graded
benchmark and does not test contamination or general ability. The semantic
holdout files are
`../measure/results/adaptation-semantic-holdout-tasks.json`,
`../measure/results/adaptation-semantic-holdout.json`, and
`../models/adaptation_semantic_holdout_evaluation.json`.

The public replacement files are `../measure/results/adaptation_tasks.json`,
`../measure/results/adaptation-multimodel.json`,
`../models/adaptation_evaluation.json`, and the saved adapters under
`../measure/results/adaptation_adapters/`. The sampled-workload and
independent-device protocol remains in [`FOLLOW_UP_STUDY.md`](FOLLOW_UP_STUDY.md).
The exact measured benchmark record and rerun command are in
[`../measure/ADAPTATION_EVALUATION.md`](../measure/ADAPTATION_EVALUATION.md).
The supplemental prompt-surface files are
`../measure/results/adaptation-template-holdout-tasks.json`,
`../measure/results/adaptation-template-holdout.json`, and
`../models/adaptation_template_holdout_evaluation.json`.

The claims ledger and the measurement protocol are part of the release because
counter-claims are required for interpreting the main results, not optional
marketing copy.
