# Reference and claim audit

Checked 2026-09-17; revision packet finalized 2026-09-20.

The original manuscript (now `paper/archive/track-a-legacy.tex`) cited 38 distinct bibliography keys. The revised manuscript (`paper/right-sized-edge.tex`) cites 14 sources: 12 retained/replaced original references and two added studies. Every revised reference was checked against a primary-source page for identity and the limited claim retained. This does not certify the cited research's correctness or peer-review status. Most research sources are preprints and are labeled accordingly.

The original shared references.bib also contains companion-paper material. It has not been wholly certified and was unchanged by this revision; it has since been updated for the footprint study's v0.3.0 release, which carries its own audit in `paper/CITATION_AUDIT.md`. The revised manuscript uses review_references.bib. A reachable URL alone does not validate a quantitative claim.

## Sources used by the revision

| Revised key | Primary source and support checked | Disposition |
|---|---|---|
| iea2025 | [IEA Energy and AI executive summary](https://www.iea.org/reports/energy-and-ai/executive-summary): the 945 TWh base case concerns all data centres. | Retain context; remove attribution of assumed AI/inference shares and fleet savings to IEA. |
| elsworth2025 | [arXiv 2508.15734v1](https://arxiv.org/html/2508.15734v1): title/authors, 0.24 Wh production median, full-stack accounting, different workload population. | Pin version. Do not describe all non-accelerator costs as avoided at home. |
| jegham2025 | [arXiv 2505.09598v3](https://arxiv.org/html/2505.09598v3): Table 4 and infrastructure-estimation method. [Metadata](https://arxiv.org/abs/2505.09598v3) confirms four authors for this version. | Pin v3 rather than latest. Label estimates and mismatched workloads. |
| knoop2026 | [arXiv 2601.09527v1](https://arxiv.org/html/2601.09527v1): consumer Blackwell inference deployment evaluation by Jonathan Knoop and Hendrik Holtmann. | Add prior art; do not import its savings as our result. |
| tummalapalli2026 | [arXiv 2603.23640v2](https://arxiv.org/html/2603.23640v2): sustained edge evaluation; explicit limitations from mixed power boundaries. | Add prior art and boundary context. |
| reddi2020mlperf | [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549), [ISCA DOI](https://doi.org/10.1109/ISCA45697.2020.00045): workload/scenario and quality requirements. | Retain narrow methodological context; this experiment is not MLPerf-certified. |
| belcak2025 | [arXiv 2506.02153v1](https://arxiv.org/html/2506.02153v1): position paper; Appendix B provides replaceability estimates. | Correct measured to estimated; no calibration of our 82% routing rate. |
| luccioni2025 | [arXiv 2501.16548v2](https://arxiv.org/html/2501.16548v2), [FAccT DOI](https://doi.org/10.1145/3715275.3732007): qualitative rebound argument. | Add DOI; remove unsupported observed 15-25% take-back claim. |
| li2025water | [arXiv 2304.03271v5](https://arxiv.org/html/2304.03271v5): withdrawal versus consumption, on-site/off-site accounting. | Replaces ucriverside2023 with an explicit version and revised-year date. |
| ollamaapi2026 | [Official generate API](https://docs.ollama.com/api/generate): raw mode bypasses prompt templating and timing fields are exposed. | Replace old moving GitHub-document URL; no guarantee that raw mode disables all caching. |
| nvidiasmi2026 | [Official NVIDIA documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html): instantaneous power is latest board measurement. | Replace general product page with field documentation; no accuracy or refresh-rate guarantee inferred. |
| rfc8831 | [RFC 8831](https://www.rfc-editor.org/rfc/rfc8831): data-channel protocol composition. | Retain protocol-level statement only. |
| rfc8827 | [RFC 8827](https://www.rfc-editor.org/rfc/rfc8827): signaling and endpoint-identity trust matter. | Correct unconditional privacy wording; not an implementation audit. |
| autoyou2026measure | [Public artifact repository](https://github.com/autoyou-ai/autoyou-research): original materials available. Local original revision: 47a581cb96532fa4ac04fa992d2f717b704a2368. | New revision files are prepared locally, not yet pushed or assigned a DOI. Do not describe the new archive as already published. |

## Every other original main-paper citation

Removed means removed from the revised main manuscript, not deleted from the original shared bibliography. Unless explicitly noted, these entries were not fully reverified because their associated claim is no longer made. They must not be advertised as validated references.

| Original key | Decision and reason |
|---|---|
| amd_strix2025 | Removed from bibliography; the AMD primary article was accessible, but the paper now treats bandwidth as an archived nominal reference, not measured sustainable transfer rate. |
| anthropic_pricing2026 | Removed pricing/generation roster; no archived primary snapshot supporting the specific model/price/date combination was established. |
| antmedia2026 | Removed assumed STUN/TURN prevalence and network-saving claim; a vendor article is not a measured relay distribution for this system. |
| aslan2018 | Removed numeric transmission-intensity extrapolation; older aggregate network estimates do not measure this route. No finding that the reference itself is false. |
| autoyou2026adaptation | Removed dependency on companion findings; companion work remains unreviewed in this task. |
| aws_water2025 | Removed adopted global WUE point/range. Withdrawal and consumption definitions must be matched before reuse. |
| caravaca2025 | Removed from the narrower bibliography; not independently reverified in this pass. Do not infer invalidity from removal. |
| databasemart2026 | Removed borrowed RTX 4090 throughput anchor; different hardware and serving protocol from the primary experiment. |
| devriesgao2025 | Removed global footprint extrapolation; detailed numeric claim not retained or reverified. |
| ember2026 | Removed asserted current global/regional grid factors; the revised equation accepts explicit deployment-specific factors instead. |
| eu_eed2026 | Removed regulatory discussion as outside the measurement case study. No legal conclusion offered. |
| frontier2026 | Removed current OpenAI model/pricing roster and its unsupported date-specific assertions. |
| google_pricing2026 | Removed current model/pricing roster; no verified historical price snapshot retained. |
| ministral32026 | Runtime tags and stored build metadata are reported as records, not an independently verified release history. Model-card citation removed. |
| mistral2024 | Removed transferred benchmark-capability ratios; an earlier family benchmark is not a task-quality evaluation of these builds. |
| museglimmer2026 | Retain the runtime tag in the data table, not an independently verified claim about the distributor or architecture. |
| ni2025benchmarks | Removed cross-benchmark-derived capability ratio; survey results cannot establish local routing success. |
| nvidia_rtx4090 | Removed TDP-based energy anchor; TDP is not measured inference power. |
| openweights2026 | Removed multi-vendor future/current architecture-and-price roster; a changing aggregator is not a stable model-specific primary citation. |
| rtx5070spec | Original URL did not resolve through the research fetch; NVIDIA's current family page was accessible. The revised table uses the archived nominal bandwidth reference, not a new hardware-performance assertion. |
| schneider2025 | Removed unverified transfer of accelerator LCA percentages into a consumer hardware allocation; no universal embodied ratio retained. |
| solartech2025 | Removed household idle-power allowance as an empirical baseline. Host power is explicitly hypothetical. |
| techtarget2025 | Removed trade-press support for a directional NPU advantage that was not tested. |
| uptime2025 | Removed global PUE trend/point claim; a global average cannot establish a displaced provider's marginal overhead. |
| wang2024slm | Removed generalized benchmark-capability narrative; no local quality equivalence follows from a survey. |
| williams2009roofline | Removed formal roofline framing: no operational-intensity/hardware-counter experiment is supplied. No finding that the foundational reference is wrong. |

## Reference integrity notes

- Version identifiers are deliberate. A later preprint version can change authors, data, tables, and conclusions.
- Vendor/runtime names in stored files are not independently authenticated model identities.
- Publication dates and access dates are distinct. Access dates in the new bibliography correspond to the actual checking pass.
- Citations were checked for support of the retained narrow claims. This was not a systematic literature review or a comprehensive plagiarism scan.
- No citations, measurements, reviewers, endorsements, DOI, or acceptance status were invented.
