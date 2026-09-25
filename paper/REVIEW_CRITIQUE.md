# Critical review of research release v0.3.0

## Bottom line

This is a transparent and technically interesting research release, but it is
not yet a strong causal empirical paper. Its strongest contribution is the
measurement and accounting framework, including explicit failure modes. Its
headline environmental, economic, capability, and adaptation conclusions are
conditional model results with limited external validity.

As a technical report, the work is promising. As evidence for a general claim
that local inference is greener, cheaper, and capable enough for most work, it
is not ready without the follow-up measurements already specified in
`FOLLOW_UP_STUDY.md` and `../measure/REPLICATION_PROTOCOL.md`.

## Weaknesses beyond publication status

| Area | Why the current evidence is weak | Counterclaim or falsifier needed |
|---|---|---|
| Accounting boundary | The headline fleet scenario still uses a legacy RTX 4090 throughput/power pair for comparability, while the new campaign measures an RTX 5070 board window and models AMD power. Cloud estimates may include facility overhead. Host CPU, DRAM, PSU loss, idle power, and post-request tail are not consistently measured on both sides. | Re-run the same prompt, model, quantization, output length, and batch policy with wall power at the edge and a matched whole-system cloud or server measurement, then recompute the headline from the matched target. |
| Workload mix | The 82% sufficient fraction is driven by five authorial capability ratios and a hand-set 18% hard-reasoning weight. There is no sampled query corpus behind the distribution. | Pre-register a task corpus and report the result across alternative distributions, including the external 40% to 70% replaceability range. |
| Hardware coverage | The evidence is concentrated on one AMD APU and one nominal RTX 5070 target. Repeated runs estimate repeatability on those targets, not variation across devices. The provenance manifest classifies the observed independent-target count as zero, and the RTX second run is not an independent laboratory replication. | Add at least one independently operated machine and multiple device classes, with the same model digests, runtime, thermal state, and power boundary. |
| Batch behavior | The AMD timing evidence now includes 10 complete repetitions at batch sizes 1, 2, 4, and 8 for 3B, 8B, and a separate 27B matrix. The 27B row rises only 1.02x at batch 2, then falls to 0.97x and 0.92x at batch 4 and 8; the host has no power sensor. The analysis reports paired median contrasts, paired bootstrap intervals, exact sign-flip tests for small paired samples, and completion/failure counts, but these are within-target timing evidence, not an energy or production-serving result. | Repeat the declared batch 1, 2, 4, and 8 matrix with wall-power integration and a defined shared-window allocation rule. |
| Adaptation uplift | A public run measures +11.1 pp absolute accuracy on 3 model clusters, 5 synthetic classes, 20 held-out tasks per class and 3 seeds, while a separate prompt-surface probe measures +10.0 pp. A new-semantic holdout with 100 fresh records measures only +1.4 pp, with a -3.3 to +7.7 model-cluster interval, exact p=1.000, and mixed model directions. The probes use one physical host, deterministic records and greedy multiple-choice grading. The class tests are secondary and Bonferroni-adjusted across five hypotheses, with no adjusted class result below 1.000. They do not identify the real workload mix, human quality, or independent-device behavior. The private one-deployment scores still have no public task-level data, independent grader, or auditable checkpoint lineage. | Repeat the same preregistered task matrix on independently operated devices and additional models, include a sampled task corpus, task-specific or blinded human grading, regression probes, and multiplicity control. Publish task-level data, model digests and checkpoint manifests. |
| Adaptation construct validity | Training and evaluation IDs are disjoint, but the primary public generator reuses a small set of class templates and the same four-choice answer format across splits. The prompt-surface probe changes the surrounding wording and still measures +10.0 pp, which weakens the exact-outer-template explanation, but it reuses the same semantic records and answer labels. The new-semantic holdout uses fresh records and formats and measures +1.4 pp with mixed model directions and p=1.000, which weakens a broad semantic-transfer reading. It remains synthetic, uses adapters trained on the primary synthetic distribution, and has no contamination or untouched general-ability probe. | Generate evaluation tasks from unseen semantic templates and paraphrases, add adversarial distractors and contamination checks, and report the same paired accuracy result on an untouched suite. The current semantic holdout is useful evidence against overclaiming, but not a substitute for that follow-up. |
| Statistical unit | Bootstrap intervals over repeated calls describe one machine and one protocol. They do not justify population claims. Small n and medians also make broad effect-size language fragile. Paired sign-flip tests improve the batch contrast but still rely on exchangeable signs and do not supply independent hardware clusters. | Declare the estimand and experimental unit before collection; use hierarchical intervals over machines, runs, and tasks, with multiplicity control for many models and metrics. |
| Uncertainty model | The Monte Carlo uses analyst-declared ranges and independent triangular draws. Its `P(S>0)` is a model-draw fraction, not a confidence level, posterior probability, or estimate of the probability that a real deployment saves energy. Correlations between workload, hardware, utilization, and cloud efficiency are not represented. | Preregister distributions and dependencies, run correlated and adversarial sensitivity analyses, and reserve probabilistic language for a specified statistical model with observed data. |
| Baseline comparability | Cloud energy values come from different studies, models, prompt shapes, and accounting boundaries. The local measurements are not a matched same-task quality-and-energy comparison against each cloud baseline. | Use an identical prompt and output target across local and cloud systems, match quality and batch policy, and report each boundary separately before presenting a combined saving. |
| Operational behavior | The energy evidence remains short windows, although timing now includes 10-repetition concurrent matrices. Queueing under a sustained arrival process, model-load time, thermal throttling, uptime, retries, and service-level constraints are not measured as a deployment system. | Test sustained workloads at declared concurrency and batch sizes, report p50/p95 latency and failures, and compare availability and energy per successful answer. |
| Researcher degrees of freedom | The release is transparent about revisions, but the current analysis was not preregistered and includes post-hoc source, wording, and measurement updates. That can reduce false-positive risk only after a frozen protocol and holdout are used. | Freeze hypotheses, task manifest, exclusion rules, and primary estimands before the next collection; keep a dated analysis log and an untouched holdout. |
| Roofline inference | `tokens/s * weight-file bytes` is an effective-bandwidth proxy, not a memory-traffic measurement. KV cache, vision components, sparse experts, multi-token prediction, and runtime caching can move it. Peak bandwidth is a specification, not a measured ceiling. | Collect actual memory-traffic counters or a calibrated bandwidth test, then repeat across batch sizes and quantizations. Keep the result labelled as a proxy until then. |
| Citation support | The audit finds 64 cited keys and no missing or unused keys. The latest network pass reports 43 entries with warnings; the count can vary with transient locator responses. Sixteen are formal or peer-reviewed candidates, one is restricted, and several are vendor, trade, community, or arXiv sources. All 64 sources now have targeted scope notes, and all 16 directly arXiv-hosted records are version-pinned, but those notes are not a substitute for a final sentence-level reading. Reachable locators do not prove support for the exact sentence. | Check every cited sentence against the source, record page or section evidence, replace weak sources for load-bearing claims, archive time-sensitive pricing and specifications, and weaken any sentence that exceeds its source. |
| Bibliographic version drift | One load-bearing energy source had been cited through an unversioned arXiv URL even though the manuscript used older table values. The audit now pins those values to arXiv v2, records its DOI and full author list, and notes that a later revision changes the table. Model-card identity and distribution-size evidence are also split into separate citations. | Freeze a reference snapshot in the release manifest, cite the exact version behind every numeric result, and rerun the numerical audit whenever a source revision or provider page changes. |
| Time sensitivity | 2026 prices, model rosters, model cards, drivers, runtimes, and URLs can change. The 150x cost claim is therefore not a stable scientific constant. | Freeze an analysis date, store version and price snapshots, and show sensitivity to price, utilization, output length, and electricity rate. |
| Water and carbon | Regional averages and operational factors do not establish query-level water or carbon for every deployment. Local execution removes some data-centre terms but not grid water, upstream generation, manufacturing, or device replacement. | Report water withdrawal versus consumption, temporal and regional grid factors, lifecycle emissions, and the embodied-carbon break-even under device purchase and replacement scenarios. |
| Privacy | DTLS and fingerprint pinning support a relay confidentiality mechanism. They do not prove endpoint confidentiality, metadata minimization, application logging behavior, or protection from a compromised peer. | Publish an explicit threat model and test endpoint, metadata, logging, key-lifecycle, and compromise cases separately from relay ciphertext handling. |
| Rebound and adoption | One assumed take-back range is not an observed behavioral response. The paper's rebound counterclaim is therefore not empirically refuted: it only fails to erase savings inside the chosen model. System-wide savings require adoption, utilization, device lifetime, availability, and rebound assumptions that are not measured together. | Run a usage study or present an adoption-utilization sensitivity surface; state that the paper estimates scenarios rather than net societal impact. |
| Scope and contribution | Energy, water, carbon, cost, capability, privacy, transport, roofline behavior, and adaptation each require a different comparison. The combined scope makes each result look more general than its evidence base. | Name one primary estimand per paper, move engineering observations to supporting results, or split the claims and conclusions into narrower papers. |

## Are counterclaims necessary?

Yes. They are necessary for every load-bearing claim, not just for a formal
limitations section. The current ledger is a good start, and the manuscripts
already include C1 to C3 and several explicit caveats. The missing standard is
that a counterclaim must be a serious alternative explanation with a defined
falsifier, not a rhetorical objection that the paper immediately dismisses.

For each headline number, the manuscript should state four things in close
proximity: the estimand, the boundary, the evidence type, and what observation
would reverse the conclusion. A counterclaim that has not been tested should be
labelled unresolved. A counterclaim that survives only under one assumed range
should be labelled conditional. Neither should be called refuted merely because
the central scenario still favours the thesis.

The most urgent counterclaims are:

1. A matched whole-system cloud baseline may erase the edge energy advantage.
2. A different task distribution may reduce the 82% workload fraction enough
   to change the fleet result.
3. Batch serving may make the cloud materially more efficient per successful
   token than single-stream edge serving.
4. The private adapter scores may reflect task leakage, grader artifacts,
   checkpoint mismatch, or selection of the best run rather than adaptation.
   The public prompt-surface probe weakens only the narrowest exact-wording
   version of this objection. The new-semantic holdout is only +1.4 pp with
   mixed model directions and exact p=1.000, which weakens a broad transfer
   reading, but reused training distribution, contamination and general-ability
   questions remain unresolved.
5. More devices, thermal states, runtimes, and operators may widen uncertainty
   beyond the intervals shown.
6. The reported $P(S>0)$ values may be artifacts of independent triangular
   assumptions and may not be probabilities about real deployments.
7. A same-task, same-quality cloud comparison may materially change the
   energy and cost ratios because the current baselines come from unlike
   workloads and boundaries.

The paper should retain a headline only when the sentence itself carries these
boundaries. The current strongest defensible phrasing is therefore: local
right-sizing can reduce modelled marginal footprint under specified conditions,
while the magnitude, task coverage and population effect remain conditional;
adaptation has a positive result on the declared synthetic benchmark but not
yet a validated product or population effect.
