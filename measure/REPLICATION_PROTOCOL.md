# Measurement and replication protocol v2

This protocol is the replacement for a single-machine, single-stream result.
It is a protocol and not a result. A target row remains pending until a real
operator contributes raw records.

## Scope

The primary comparison is local inference under the same model build, prompt,
output-token target, serving runtime, and power boundary. The main outcome is
aggregate output tokens per second under concurrent request load. Energy per
request is reported only when a power trace brackets the complete batch window.
An equal split of a shared trace is labelled as an allocation, not as a direct
causal attribution to each request.

The experiment has three layers:

1. Single-request baseline, batch size 1.
2. Concurrent serving at batch sizes 2, 4, and 8.
3. Independent replication on a separately operated target.

Ollama's num_batch option controls prompt evaluation inside one request. It is
not used as a proxy for request concurrency. The batch harness submits one
independent HTTP request per concurrent user and gives each request a unique
nonce.

## Hardware matrix

hardware_matrix.json is the machine handoff manifest. The two rows already in
the repository are reference runs. The remaining rows are targets, not
evidence. Each contributor must fill the manifest fields from the machine
under test and attach raw JSON without replacing an existing row.
`replication_manifest.json` is the companion provenance manifest: it must list
an anonymous operator label, a stable physical-target label, and a runtime
environment label for every contributed raw file. The analysis cannot call a
run independent when any of those fields is missing or unchanged.
Its structure is defined by `replication_manifest.schema.json` and checked by
`validate_replication_manifest.py` before analysis. For concurrent records,
that validator also checks the raw v2 record shape and declared protocol
fields. A contribution labelled `independent_replication` must contain batch
sizes 1, 2, 4, and 8, at least 10 repetitions per row, and a model digest in
every model row. This prevents a provenance label from promoting an incomplete
or unpinned run to independent evidence.
Validate the manifest and listed-file bookkeeping before running statistics:

    python measure/validate_replication_manifest.py

This check intentionally passes with zero pairs and reports that independent
evidence is pending. Use `--require-independent` as a release gate when a
qualifying pair is required; a valid manifest alone is not a replication
result.

The target set is intentionally heterogeneous:

- AMD unified-memory APU
- NVIDIA consumer discrete GPU
- NVIDIA higher-memory discrete GPU
- Apple unified-memory device
- ARM or other unified-memory device

The study must not pool hardware rows into one mean. Report each device,
runtime, driver, model build, quantization, power sensor, and operating mode
separately.

## Required run conditions

- AC power, fixed performance mode, and no screen or system sleep.
- One warm-up per model, discarded from the summary.
- Temperature 0, fixed seed, raw prompt mode, and a unique nonce per request.
- The same model digest and quantization on every target that claims a
  cross-device comparison.
- At least 10 complete repetitions per model and batch size for the follow-up.
- Batch sizes 1, 2, 4, and 8, in a recorded order. If thermal drift is visible,
  repeat in the reverse order and retain both sequences.
- Preserve failed or incomplete repetitions. Do not silently reduce a batch.
- Record wall-clock request time, server generation time, prompt tokens,
  output tokens, power trace, temperature if available, and residency.
- If a wall meter is available, record it as the primary energy boundary.
  Board telemetry may be retained, but it must not be compared with a
  whole-system cloud figure without an explicit reconciliation.

Run the current concurrent harness from the repository root:

    python measure/bench.py --models ministral-3:3b ministral-3:8b --reps 10 --tokens 300 --batch-sizes 1 2 4 8 --tag <machine-id>
    python measure/statistics.py --resamples 20000

The harness writes measure/results/<machine-id>-batched.json. Never edit the
raw file after collection. Add a note beside it if a runtime or hardware
condition makes the standard protocol impossible.

## Independent replication

An independent replication must have:

- a different operator or an independently controlled execution environment;
- a different physical machine, unless the purpose is only code-path replay;
- a fresh model pull verified by digest;
- the same protocol fields and a recorded deviation list;
- raw request records and power traces, not only a final mean;
- a source hash and the exact command used.

The existing rtx5070-replication.json is a second run on the same named target.
It is useful for repeatability and arithmetic replay, but it is not an
independent-laboratory result. measure/statistics.py reports this limitation
in its output instead of promoting it to a replication claim.

Agreement is assessed with the difference in medians, a model-cluster-aware
bootstrap interval where clusters exist, and a two-sided randomization test.
No result is discarded because a p-value is inconvenient. The direction,
magnitude, interval, protocol deviations, and missing measurements are all
reported.

## Adaptation follow-up

The old +2/+6/+14 capability uplift is an exploratory sensitivity scenario.
It is not an empirical replacement for a workload-weighted product claim. The
replacement data contract is
adaptation-multimodel.schema.json, summarized by
models/adaptation_eval.py.

The current public benchmark passes the minimum data contract: three named
models, five classes, 20 held-out tasks per class, three adapter-training seeds,
paired base/adapted/reference scores, a public task manifest and model or
adapter metadata. Its primary estimand is absolute greedy four-choice task
accuracy, not a ratio to the frontier. The released run is a synthetic-task
benchmark on one physical host, so it is a measured benchmark result rather
than a product or population result.

A supplemental prompt-surface holdout changes the surrounding instructions
while reusing the public evaluation records. It reports 80.0% base accuracy,
90.0% adapted accuracy, and +10.0 percentage points with a +5.0 to +15.0
model-cluster interval and exact p=0.250. This weakens an exact-wording
alternative explanation, but it is not an independent replication: the
semantic records and answer labels are reused, only one paraphrase is tested,
and the physical host is unchanged.

A separate new-semantic holdout uses 100 fresh synthetic records, 20 per class,
with new record values and task formats rather than reusing primary evaluation
records. It reports 90.3% base accuracy, 91.8% adapted accuracy and +1.4
percentage points with a -3.3 to +7.7 model-cluster interval and exact p=1.000.
The three model-level changes are +7.7, -3.3 and 0.0 percentage points. This
mixed result is a useful counterweight to the primary uplift, but it is still
one-host synthetic evidence, uses adapters trained on the primary synthetic
benchmark, and has no contamination or untouched general-ability suite. The
files are `measure/results/adaptation-semantic-holdout-tasks.json`,
`measure/results/adaptation-semantic-holdout.json`, and
`models/adaptation_semantic_holdout_evaluation.json`.

Before any uplift is used in a headline:

- evaluate at least three base-model and adapter pairs;
- cover all five classes: extraction, RAG question answering, summary, simple
  code, and hard reasoning;
- use at least 20 held-out task identifiers per class;
- score the same task for base, adapted, and frontier reference systems;
- keep training and evaluation tasks disjoint;
- use at least three fixed evaluation seeds where generation is stochastic;
- blind the grader to base versus adapted labels;
- publish per-task aggregate scores, grader version, model digests, adapter
  digests, and the task-manifest hash;
- report model-level means before pooling tasks;
- use an exact model-cluster sign-flip test when the number of model clusters
  is small, rather than treating task rows as independent replicates.

Until an independently operated, sampled-workload study also passes those
conditions, the paper must say that real-workload adaptation uplift is
unvalidated and that the existing coverage curve is a scenario only.

## Statistical boundary

The bootstrap intervals in statistics.py quantify repeatability of the
recorded repetitions. They do not estimate the population of consumer
devices. Batch scaling uses paired repetition IDs, reports the paired median
contrast, and applies a paired percentile bootstrap plus an exact two-sided
sign-flip randomization test when the pair count permits enumeration. It also
reports complete and incomplete batches and failed-or-missing requests, so
throughput cannot hide a change in completion rate. These tests quantify a
within-target contrast only and do not create independent hardware replicates.
The replication comparison is labelled
independent only when target, operator, and runtime environment identities all
differ. The adaptation evaluator resamples model clusters so a large task set
from one model cannot masquerade as many independent model replications. The
paper must retain these distinctions in the abstract, tables, and captions.
The directory analysis scans all manifest-listed raw records for qualifying
independent pairs and reports zero when none exists; the current RTX replay
therefore remains repeatability evidence.
