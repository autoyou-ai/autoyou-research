# AutoYou Research

Open research on what it costs, in energy, water, carbon, money, and capability,
to run and adapt AI models on consumer hardware instead of in the cloud. Each
study ships with its manuscript, raw measurements, analysis code, and an audit
that regenerates its numbers.

## Papers

| Paper | Status | Read |
| --- | --- | --- |
| **Quantifying Carbon, Water, Cost, and Capability in Peer-to-Peer Inference** | alphaXiv preprint v1, 21 September 2026 | [PDF](paper/p2p-inference.pdf) · [LaTeX](paper/p2p-inference.tex) · [alphaXiv](https://www.alphaxiv.org/abs/2609.carbon-water-cost-p2p-inference) |
| **Right-Sized AI at the Edge: Reproducible Measurements and Conditional Sustainability Analysis** | alphaXiv preprint v1, 21 September 2026 | [PDF](paper/right-sized-edge.pdf) · [LaTeX](paper/right-sized-edge.tex) · [alphaXiv](https://www.alphaxiv.org/abs/2609.right-sized-ai-edge-sustainability) |
| Teaching at the Edge: On-Device Parameter-Efficient Adaptation and What It Actually Buys | Companion research release v0.3.0 | [PDF](paper/adaptation.pdf) · [LaTeX](paper/adaptation.tex) |

None of these papers has been peer reviewed.

### How the two preprints relate

Both grew out of one earlier draft, kept in [`paper/archive/`](paper/archive/README.md).
They take different positions on how far the evidence reaches:

- The **peer-to-peer inference study** models the whole system (workload mix,
  cloud baselines, grid carbon, water, and prices) around direct measurements,
  and reports conditional scenario results, including its headline carbon and
  cost percentages.
- The **Right-Sized paper** limits itself to what the stored measurements support.
  It does not carry those carbon and cost percentages forward, because the
  records contain no quality-matched cloud comparison, observed routing
  coverage, or whole-system power measurement.

Read the first paper's percentages as estimates under its stated assumptions,
and the second for the measurement-only reading.

## Quantifying Carbon, Water, Cost, and Capability in Peer-to-Peer Inference

**Question.** When does running small language models on hardware people already
own, with devices connected peer to peer over WebRTC and frontier cloud models
called only for hard tasks, reduce the footprint of AI inference?

**Method.** A parameter-driven model of per-query energy, water, carbon, and cost,
built from public measurements and published estimates with explicit ranges. It
adds a fleet-scale extrapolation on the IEA 2030 demand base, a 50,000-draw
Monte-Carlo analysis, and one-at-a-time sensitivity analysis. Six hypotheses and
three counter-claims each receive a scoped verdict.

**Findings, conditional on the paper's assumptions:**

- About 82% of a representative assistant workload is capability-sufficient for
  a small model at a 70% acceptance threshold. This workload mix is an authorial
  assumption, above the 40–70% replaceability that NVIDIA reports for agentic
  systems.
- For that fraction, against typical and heavy frontier usage, edge execution
  cuts per-query carbon by 54–64%, nearly eliminates the modeled on-site
  cooling-water term, and costs about 150 times less per query.
- Against a best-in-class efficient cloud model (0.24 Wh per prompt), a
  single-user consumer GPU is 16% worse on average. The advantage comes from
  right-sizing and from avoiding data-centre overhead, not from relocating the
  same model.
- The result depends on the local grid, and it survives induced usage growth
  up to about 212%.
- Every saving scales with the capability-sufficient fraction. Swept across
  0.40–0.90, the sign of the result holds against frontier baselines, but the
  magnitude does not: at 55% the carbon saving falls from 54.4% to 36.5%.

**Measurements.** On an AMD Radeon 8060S APU, single-stream generation reaches 78.4,
37.8, and 13.8 tokens/s for 3.8B, 8.9B, and 27.3B models. On an NVIDIA RTX 5070,
board telemetry measures request-window energy of 0.091 Wh (3B) and 0.19 Wh (8B)
at 499 prompt and 300 output tokens, repeated in a second session. The authors
operated both machines; no independent replication is included.

## Right-Sized AI at the Edge: Reproducible Measurements and Conditional Sustainability Analysis

**Question.** What do the archived consumer-inference measurements establish on
their own, once every accounting boundary is made explicit?

**Method.** An independent implementation in [`review/audit.py`](review/audit.py)
replays 68 stored generation repetitions, including 38 requests with bracketed
board-power telemetry, without importing the original research models. It
derives accounting expressions that charge failed local attempts, cloud
fallback, and a common demand-growth factor across footprint metrics.

**Findings:**

- The RTX 5070 records median request-window board energies of 0.091 Wh and
  0.19 Wh for the recorded 3B and 8B builds, and a second session on the same
  machine gives the same rounded medians.
- Assuming 60 W of additional host power and 90% supply efficiency raises the
  estimates to 0.13 Wh and 0.27 Wh. These are scenarios, not wall-meter readings.
- A throughput-times-file-size comparison cannot be read as measured
  memory-bandwidth utilization.
- The records do not establish quality-matched cloud displacement,
  population-wide task coverage, total cost of ownership, or fleet-scale savings.
  The paper specifies the evidence a credible edge-versus-cloud sustainability
  claim would need.

See the [review report](review/REVIEW_REPORT.md) and
[reference audit](review/REFERENCE_AUDIT.md).

![Measured board energy and modeled host overhead](figures/review_energy.png)

## Teaching at the Edge (companion study)

Tests whether parameter-efficient fine-tuning on owned hardware moves the
small-model-sufficient fraction, and what an adaptation run costs. It models 16
adaptation methods and three device classes, and checks its memory model against
a publicly documented 27B training run to within −0.6%.

- At the 0.70 acceptance threshold, modeled adaptation changes coverage by
  +0.0 points: every adaptable task class already cleared the bar.
- A public synthetic held-out benchmark (3 models, 5 task classes, 20 tasks per
  class, 3 seeds, one host) measures accuracy rising from 86.7% to 97.8%, with
  a model-cluster interval of +5.0 to +18.3 points and p = 0.250.
- A new-semantic holdout moves only from 90.3% to 91.8% (−3.3 to +7.7 points,
  p = 1.000), which weakens a broad transfer reading.
- A 27B run costs about 10.7 kWh. It is repaid in 143 days against an all-cloud
  counterfactual, but never at the 0.70 threshold when charged only against the
  routing it actually changes.

The benchmark tasks, per-seed results, and trained adapters are published in
[`measure/results/`](measure/results/); see
[`measure/ADAPTATION_EVALUATION.md`](measure/ADAPTATION_EVALUATION.md).

## Reproduce

Use Python 3.10 or later. The commands replay stored data offline; none calls a
model service or reads private data.

```bash
python -m venv .venv
# Activate .venv with your shell's activation command.
python -m pip install -r requirements.txt
```

**Peer-to-peer inference and adaptation studies:**

```bash
python models/run_all.py             # results.json, all fig_* figures, paper/empirical.tex
python models/verify_paper.py        # every number in both manuscripts traces to the harness
python measure/validate.py           # independent arithmetic replay of the raw records
python measure/statistics.py --resamples 20000
```

**Right-Sized paper:**

```bash
python review/audit.py               # tables, review figures, review/audit_results.json
python review/source_manifest.py --check   # inputs and outputs match the recorded hashes
python -m unittest discover -s review -v
python review/build_paper.py         # needs pdflatex, BibTeX, and IEEEtran
```

**All unit tests** (run from the repository root):

```bash
python -m unittest measure.test_bench measure.test_batched measure.test_statistics measure.test_replication_manifest models.test_adaptation_eval models.test_adaptation_experiment models.test_adaptation_semantic_holdout models.test_adaptation_template_holdout paper.test_audit_citations
```

[`paper/README.md`](paper/README.md) covers the LaTeX builds and citation audits.
Running [`measure/bench.py`](measure/README.md) is a live experiment that loads
local models; write its output to new files so the archived records stay intact.
Rerunning the adaptation benchmark needs local model snapshots and the extra
dependencies in [`requirements-adaptation.txt`](requirements-adaptation.txt).

## Repository map

| Path | Contents |
| --- | --- |
| [`paper/`](paper/) | Manuscripts, PDFs, bibliographies, citation audits, claims ledger, release manifest |
| [`paper/archive/`](paper/archive/) | The earlier draft both preprints descend from |
| [`review/`](review/) | Offline audit, tests, and build for the Right-Sized paper |
| [`measure/`](measure/) | Benchmark harness, raw records, statistics, replication protocol, adaptation benchmark |
| [`models/`](models/) | Footprint, routing, and adaptation models; figure generation; number verifier |
| [`figures/`](figures/README.md) | Generated figures for all three papers |
| [`agent/`](agent/) | Reference edge-first routing agent, not an evaluated deployment |
| [`literature/`](literature/) | Annotated literature notes |

## Data and licences

The [MIT license](LICENSE) covers the code, manuscripts, figures, and measurement
records. The LoRA adapters in
[`measure/results/adaptation_adapters/`](measure/results/adaptation_adapters/NOTICE)
follow their base-model licences instead: Apache-2.0 for Qwen2.5-0.5B, and the
Qwen Research License (non-commercial) for Qwen2.5-3B and Qwen2.5-VL-3B.
Improved using Qwen.

Adapter-evaluation inputs from a deployed assistant are private and excluded.
Findings that depend on them report no data on a public checkout.

## Cite

```bibtex
@misc{autoyou2026quantifying,
  title     = {Quantifying Carbon, Water, Cost, and Capability in Peer-to-Peer Inference},
  author    = {{AutoYou Research}},
  year      = {2026},
  publisher = {alphaXiv},
  url       = {https://www.alphaxiv.org/abs/2609.carbon-water-cost-p2p-inference},
  keywords  = {machine-learning, computers-and-society, distributed-parallel-and-cluster-computing, sustainable-ai, peer-to-peer-inference, fosc: computer and information sciences}
}

@misc{autoyou2026rightsized,
  title     = {Right-Sized AI at the Edge: Reproducible Measurements and Conditional Sustainability Analysis},
  author    = {{AutoYou Research}},
  year      = {2026},
  publisher = {alphaXiv},
  url       = {https://www.alphaxiv.org/abs/2609.right-sized-ai-edge-sustainability},
  keywords  = {edge inference, energy measurement, carbon accounting, water consumption, reproducibility}
}

@misc{autoyou2026teaching,
  title  = {Teaching at the Edge: On-Device Parameter-Efficient Adaptation and What It Actually Buys},
  author = {{AutoYou Research}},
  year   = {2026},
  note   = {Research release v0.3.0},
  url    = {https://github.com/autoyou-ai/autoyou-research}
}
```

[`CITATION.cff`](CITATION.cff) carries the same metadata. When reusing measurement
records, also cite the Git commit and the record filenames.

## Contribute

Corrections, replications, and new experiments are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md). Contact: research@autoyou.me.
