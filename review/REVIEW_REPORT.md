# Research review and revision report

Review finalized: 2026-09-20. Primary-source checks began 2026-09-17.

## Decision

The original manuscript requires major revision. Its useful contribution is a small, inspectable consumer-inference measurement artifact. Its headline environmental and cost advantages are not established by the experiments: the compared workloads and accounting boundaries differ, quality was not evaluated, and several assumptions are presented with stronger evidential language than the sources justify.

The revised manuscript is a substantially narrower, six-page working paper. It is suitable for seeking expert feedback as a measurement case study, subject to author verification and publication declarations. I would not recommend presenting it as a confirmed environmental advantage over cloud inference, or as a strong full research paper for a selective systems venue without additional experiments. Acceptance and visibility cannot be guaranteed.

This report is an AI-assisted technical review and code-path-independent arithmetic audit. It is not independent human peer review, an editorial decision, a plagiarism certificate, proof of data authenticity, or independent physical replication. The assistant also revised the manuscript, so it cannot be represented as an independent reviewer of its own revision.

## Material reviewed

The original manuscript was 14 pages. Its PDF and LaTeX source are preserved in
[the archive](../paper/archive/README.md). The earlier bibliography and generated
empirical macros are in Git commit `fa15395`; `paper/references.bib` and
`paper/empirical.tex` now carry the footprint study's later v0.3.0 versions.
The revised manuscript is [paper/right-sized-edge.pdf](../paper/right-sized-edge.pdf).

Original PDF SHA-256:

`FB838DAC453273C570F7EF9EAD07914F2B5BB1532438AB6053132F3B0F917192`

Reviewed evidence includes the main manuscript, its cited source claims, analytical model and parameter files, measurement harness, six stored measurement files, generated values, and validation scripts. The companion adaptation paper was not fully reviewed and is unchanged. No private training corpus, captured conversation history, operational logs, live model service, or cloud inference API was used. No new measurements were collected.

## Strengths

- The measurement files retain request timings and bracketed NVIDIA power traces that can be replayed without the original research-model implementation.
- The same-machine repeat session permits a limited check of repeatability.
- The paper addresses important distinctions among accelerator energy, host overhead, task capability, and environmental accounting.
- The raw evidence is useful precisely because discrepancies and missing observations can be reported openly.

## Major findings and their disposition

### R1. Cloud and local workloads are not equivalent

The original central local scenario used approximately 500 input and 300 output tokens. Its headline cloud baseline used the 10,000-input/1,500-output GPT-4o estimate from Jegham et al. The measured RTX study is 499/300. Google's 0.24 Wh number describes a median production Gemini Apps text prompt with a different population and broader system boundary. No held-out task-quality comparison bridges these differences.

The arithmetic of a difference can be correct while the comparison is scientifically invalid. Linear normalization by output tokens would not solve prefill, caching, batching, tokenizer, reasoning, or quality differences.

Change: removed the 54-64% real-world carbon-saving conclusion, near-certain savings language, categorical cloud/edge winners, and downstream fleet claims. Kept the external numbers only to explain why they are not direct baselines. [Jegham v3, Table 4](https://arxiv.org/html/2505.09598v3), [Google measurement study](https://arxiv.org/html/2508.15734v1).

### R2. Estimates are repeatedly called measurements

Jegham's method infers infrastructure and combines it with API performance. It does not read provider wall power. Belcak et al.'s Appendix B estimates the share of calls replaceable by smaller models in three architectures; it is not a sampled task-coverage evaluation for this deployment. The original bibliography even described Jegham as an estimation framework while the body used stronger language.

Change: relabeled the evidence consistently and pinned the versions supporting the values. The 1.788 Wh GPT-4o, 5.518 Wh Claude-3.7 Sonnet, and 39.223 Wh o3 long-prompt values match Jegham v3; an unversioned citation can resolve to a later version with changed content and authorship. [Jegham v3 metadata](https://arxiv.org/abs/2505.09598v3), [Belcak Appendix B](https://arxiv.org/html/2506.02153v1).

### R3. The 82% routing claim is not measured capability

The five task weights (0.18, 0.30, 0.22, 0.12, 0.18) and capability ratios (0.95, 0.90, 0.88, 0.75, 0.51), thresholded at 0.70, yield 0.82 by construction. They are not inferred from the timing prompt. A threshold allowing 70% of a reference benchmark score is also not an equivalence test. Failed local attempts may still consume a cloud call.

Change: the revised model explicitly charges local attempt probability p and conditional cloud fallback q. It makes no estimate of either. For a one-class check, cloud footprint 1, local footprint 0.25, p=0.80, and q=0 give a routed footprint of 0.40. At q=0.10 it becomes 0.48. The discarded local work matters.

### R4. Board, host, supply, and idle boundaries were mixed

GPU-board telemetry excludes CPU, host DRAM, power-supply loss, building cooling, and idle time outside the request. A whole-desktop idle allowance cannot be subtracted from a GPU-board power value. Comparing the earlier RTX 4090 parameter assumptions against a measured RTX 5070 does not validate or falsify the same hardware configuration.

The old host table's zero-host row was board-only while its other rows included supply loss. That changed the accounting boundary within one column.

Change: every modeled AC row now uses `(board_Wh + host_W * seconds / 3600) / efficiency`, including zero host. Board-only is a separate column. At assumed 60 W and efficiency 0.90, the primary medians are 0.130719 Wh and 0.273100 Wh, displayed as 0.13 and 0.27 Wh. These are not wall-meter measurements. [NVIDIA field definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

### R5. Rebound was one formula in incompatible units across metrics

For one footprint F, a take-back fraction r can be converted into query-volume growth `g = r * (cloud - routed) / routed`. The conversion is valid separately. However, applying the same r to different footprints generally produces different g values, which cannot describe one physical change in query count.

Replaying the original GPT-4o-long central scenario at r=0.20 gives implied growth of approximately 43.79% for energy, 47.89% for water, and 42.39% for carbon. The citation to Luccioni et al. supports the importance of rebound, not an observed 15-25% elasticity for this application.

Change: one g now multiplies all per-task footprints. The revised paper derives break-even thresholds but makes no empirical claim that rebound is refuted. [Rebound paper](https://arxiv.org/html/2501.16548v2).

### R6. The bandwidth statistic is a proxy, not a hardware counter

Throughput multiplied by model-file size is not measured external-memory traffic. A dense model's file can include components not used during text decoding; caching, tensor layout, quantization, and runtime execution also matter. An MTP metadata field does not demonstrate that the serving runtime actually used speculative decoding. In particular, dense output projection weights are generally used in token generation; it is unsafe to claim that all embedding/output-table bytes are never read.

The observed 105.7% ratio reveals disagreement with the full-file-per-token interpretation. It does not bound the maximum proxy error on other builds. Therefore the original comparison of a 30-percentage-point group gap against a 5.7-point discrepancy is not a valid significance or calibration argument.

Change: retain all twelve ladder rows with neutral labels, remove causal architectural classification and utilization claims, and require hardware traffic counters to test the explanation.

### R7. Reproducibility was overstated

The nine AMD ladder rows do not contain a runs array. They provide medians, spread, and metadata only. The AMD prefill fits can be reconstructed from four stored point medians, not from all underlying timing repetitions. Two ladder builds are anonymized and their weights are not available in the artifact. A second session on the same NVIDIA machine is repeatability, not independent laboratory replication.

Change: expose these limits in the abstract/methods/results as applicable. The independent audit distinguishes raw replay from summary-arithmetic checks and records nine explicit warnings. No missing repetitions were reconstructed or invented.

### R8. Instrument accuracy is not established by rounding or identities

The primary/repeat NVIDIA requests have 3-4 distinct in-window power readings for 3B and 6-7 for 8B. Poll count alone does not establish the sensor's update interval, calibration, or uncertainty. Reporting two significant figures does not prove two-significant-figure physical accuracy. Mean power computed from integrated energy cannot independently validate that energy through the identity E=P*T.

The wall-minus-prefill-minus-decode residual is approximately 26-72 ms across these sessions, not uniformly 5-8 ms. It has not been isolated to a specific cause.

Change: report observed ranges and repeatability without population confidence intervals or a claimed calibrated error bound. Explain the identity and residual accurately.

### R9. Water and carbon need consistent definitions

The water units in `energy_Wh * intensity_L_per_kWh` are mL: the conversion factors cancel. That part is correct. But withdrawal and consumption cannot be added as if identical. On-site water intensity normally uses IT electricity, while off-site electricity-related water uses facility electricity. Geographic and temporal factors can change the result.

Google's market-based emissions cannot be reproduced by multiplying its energy by a generic location-based grid factor. Embodied allocation is different from consequential avoided emissions. Removing data-centre cooling also does not eliminate host/supply/building costs at home.

Change: keep dimensionally correct equations, require matching scopes, and remove unqualified water-saving and embodied-impact conclusions. [Li et al., sections 2-3](https://arxiv.org/html/2304.03271v5).

### R10. Cost and fleet claims go beyond the evidence

An electricity-only local cost is not comparable to an API price as total cost of ownership. It excludes capital, standby, labor, maintenance, and the residual cloud workload. The original roughly 150x number should not be presented as a system-wide saving.

The IEA 945 TWh forecast covers all data-centre demand. The original AI share 0.45 and inference share 0.65 are additional assumptions. They do not become IEA forecasts by multiplication. Even within the original illustrative accounting, translating energy savings directly to carbon omitted the separately charged embodied term: the audit obtains about 17.38 Mt from energy alone versus 17.20 Mt from the original per-query carbon accounting, before addressing the much larger comparability problem. Neither is a validated forecast.

Change: retain an explicitly hypothetical electricity-cost example, remove the current vendor-pricing roster and global savings forecast, and describe what workload-weighted evidence would be required. [IEA executive summary](https://www.iea.org/reports/energy-and-ai/executive-summary).

### R11. Novelty and transport conclusions need narrower scope

Consumer inference measurements predate this paper, including the 2026 Knoop/Holtmann and Tummalapalli studies. A useful contribution can be a well-documented case study without claiming a missing field-wide benchmark.

WebRTC's encrypted data channel protects against an ordinary relay reading payloads, conditional on appropriate authentication and signaling. It is not an audit of the application's endpoint trust, pairing, privacy, or implementation. The peer performing inference sees the prompt. Network energy and relay probabilities were not measured here.

Change: add the related work, narrow the novelty claim, retain a brief standards-grounded transport discussion, and remove numerical network savings and implementation-validation labels. [Consumer Blackwell study](https://arxiv.org/abs/2601.09527v1), [edge deployment study](https://arxiv.org/abs/2603.23640v2), [RFC 8831](https://www.rfc-editor.org/rfc/rfc8831), [RFC 8827](https://www.rfc-editor.org/rfc/rfc8827).

## Validation performed

- Independent replay: 21 model/file groups, comprising 12 groups with raw main-run repetitions and nine AMD summary-only groups.
- Main-run repetitions replayed: 68. Power requests integrated: 38.
- Maximum difference from serialized energy: 4.993321251478555e-7 Wh, within rounding tolerance. This is arithmetic error, not sensor error.
- Eleven audit tests pass, including unit conversions, request clipping, supply conversion, fallback cost, shared growth, and the full archived-data replay.
- Two original harness tests pass after repairing the obsolete research.measure import to measure.
- The revised LaTeX build has no undefined references, overfull boxes, duplicate labels, or LaTeX errors. All six pages were visually inspected during repository preparation.
- Repository preparation on 21 September 2026 repeated the audit and PDF build in an isolated copy. The stored results and tables reproduced exactly, and the rebuilt PDF text matched the supplied revision. The audit, unit tests, and PDF build are the relevant checks for this standalone repository.

The earlier measure/validate.py has an obsolete repository-root calculation and assumptions that disagree with current host conversion and formatting. It and the old numeric-token checker are not used as evidence for this revision. The dedicated new audit avoids changing unrelated companion-model behavior.

## Revision map

The abstract, introduction, results, discussion, conclusion, and title were rewritten around the evidence actually present. Tables are generated from the independent audit. A separate 14-entry bibliography avoids silently modifying the companion paper's references. The README no longer promotes the withdrawn headline claims. Original data, legacy models, original bibliography, and companion manuscript are retained.

## Work that cannot be repaired editorially

1. Collect a representative held-out task set with rights to release it and predefined quality/latency criteria.
2. Compare local, efficient-cloud, and stronger-cloud alternatives on those same tasks; count failures and retries.
3. Measure calibrated wall energy over longer controlled windows, with matching idle baselines and cold/warm cases.
4. Repeat across sessions and at least one independently operated machine; archive complete raw observations and immutable model identifiers.
5. Measure routing accuracy, fallback frequency, and any claimed network savings.
6. Estimate real deployment mix, usage growth, and regional environmental factors before making fleet claims.

No amount of editing or online promotion can substitute for those measurements.
