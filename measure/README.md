# Primary measurement

> Two papers use these records. The footprint study
> ([`../paper/p2p-inference.pdf`](../paper/p2p-inference.pdf)) uses them as
> documented below. The revised working paper
> ([`../paper/right-sized-edge.pdf`](../paper/right-sized-edge.pdf)) replays the
> six original records offline with [`../review/audit.py`](../review/audit.py)
> under stricter boundaries: the AMD ladder retains summary statistics only,
> prefill ratios are fitted projections, board telemetry is not whole-system or
> calibrated marginal energy, and architecture metadata and file-size proxies do
> not prove runtime behavior. Keep new measurements in new files; the six
> originals are hash-pinned by both studies.

Everything else in this repository is a model built on other people's measurements.
This directory is the exception: it measures the machines the authors actually
own, and feeds those numbers back into the study.

## Why it exists

The study's edge-energy model has three inputs:

```
E_edge = P · (n_o / τ) · (1 + β) / 3600
```

| Term | Was | Now |
|---|---|---|
| **τ** throughput | a third-party benchmark blog for an RTX 4090 | **measured**, per model, per machine; the legacy scenario remains explicit |
| **β** prefill overhead | an author assumption of 0.15 | **measured** at 0.051–0.078 for the study's own query shape; the conservative scenario remains explicit |
| **P** board power | a vendor TDP figure | **measured** where a sensor exists; explicitly *not* measured where none does |

A reviewer is entitled to ask why three borrowed numbers describe hardware
nobody in the study had touched. The measurement campaign answers that question
for the two tested platforms, but it does not turn one machine into a population
estimate or silently replace the legacy scenario used for comparability.

## Run it

```bash
python measure/bench.py --list
python measure/bench.py --tag my-machine
python measure/bench.py --ladder --bandwidth 256 --tag my-machine
python measure/bench.py --batch-sizes 1 2 4 8 --reps 10 --tokens 300 --tag my-machine
```

Needs Ollama running locally and nothing else — no extra Python packages, no
drivers, no admin. Results land in `results/<tag>.json`; the study reads them
through [`../models/measured.py`](../models/measured.py).

```bash
python models/measured.py     # summary of everything measured
```

## Protocol

The measurement is worth exactly as much as the protocol, so:

- **Temperature 0, fixed seed, fixed prompt**, so token counts are stable.
- **Warm-up discarded.** A cold call is dominated by `load_duration` — 8.4 s
  for the 27B — which is not steady-state serving.
- **Median over N repetitions**, with the full spread recorded. A background
  scheduler stall produces a long tail, not symmetric noise, so the mean is the
  wrong statistic.
- **Both prompt-cache paths closed.** See below; this is the part that took two
  attempts.
- **Prompt length swept, not fixed.** Prefill time is a fixed per-request cost
  plus a per-token cost. Dividing one measurement by its own prompt length
  gives `n/(a+bn)` — a number that rises with `n` and describes the prompt, not
  the machine. Four lengths per model, regressed.
- **Energy by integration, not endpoints.** Power is sampled every 250 ms on a
  background thread and integrated by the trapezium rule. Two readings either
  side of a call do not give energy.
- **Idle baseline measured**, so the study's *marginal* accounting — charge only
  the power above what the machine drew anyway — rests on a measurement at both
  ends rather than an assumption at one.

Architecture, parameter count, quantisation, context length, expert counts and
multi-token-prediction depth are recorded with every figure, because the same
model name at Q4_K_M and Q8_0 differs by more than the gap between two device
classes — and because a sparse or speculative build is not measuring the same
quantity as a dense one.

## The two caching traps

Prompt caching does not announce itself. It arrives as a plausible number.

**Trap 1 — whole-prompt reuse.** Send the same prompt five times and the server
reuses the KV cache while still reporting the full prompt length. The 8B model
appeared to prefill at **21,000 tok/s**. A unique marker per repetition closes
this, and the first version of the harness did exactly that.

**Trap 2 — the chat template's system prefix.** The marker was at the start of
the *user content*, not at position zero of the token stream. `/api/generate`
wraps the prompt in the model's chat template, whose prefix is identical on
every request and therefore always cached — while still being counted in
`prompt_eval_count`. On Ministral-3 that is **552 of 639 reported prompt tokens
billed but never computed**: the same content measures 6,622 tok/s templated
against 1,084 tok/s raw, an inflation of **6.1×**.

The length sweep is what caught it. Fitted against the templated endpoint the
regression returns a **negative** intercept — −471 ms on the 8B — which is not a
physical quantity; the line has to dive below zero to reach the artificially
cheap short-prompt point. In raw mode the same fit returns +5.4 ms, and the two
*marginal* rates agree to 1.1%: only the intercept and the apparent throughput
were ever corrupted. A single prompt length would have reported the contaminated
number with nothing to contradict it.

The probe is kept in the harness (`template_cache_probe` in each record) rather
than deleted after the fix, and it has a negative control: Qwen3.8's template
adds 10 tokens rather than 552, and there it correctly reports no artefact.

## Historical single-request pass

**AMD Radeon 8060S** (Ryzen AI MAX+ 395 "Strix Halo", 64 GB unified, Ollama
0.33.2, all Q4_K_M, `num_ctx` 4096, template bypassed, 5 reps × 192 tokens):

| Model | Params | Generation | Prefill, marginal | Fixed | β at 500 in / 300 out |
|---|---:|---:|---:|---:|---:|
| `ministral-3:3b` | 3.8B | 78.2 tok/s | 2,624 tok/s | 3.8 ms | 0.051 |
| `ministral-3:8b` | 8.9B | 37.7 tok/s | 1,142 tok/s | 5.4 ms | 0.056 |
| `qwen3.8:27b` | 27.3B | 13.7 tok/s | 338 tok/s | 234 ms | 0.078 |

Every regression has R² ≥ 0.9998. The 27B carries a multi-token-prediction head
(`nextn_predict_layers = 1`) and its generation rate is correspondingly unstable
across repetitions (12.7–24.7 tok/s, σ = 5.0) where the two dense models hold to
σ/μ < 0.3%.

No power sensor: this machine exposes no board or package rail readable without
a kernel-mode helper (AMD uProf, LibreHardwareMonitor), which this harness
deliberately does not install. Its energy figures therefore still carry a
modelled `P`, and say so wherever they appear.

## Current concurrent pass

The raw record timestamps are UTC and begin at 2026-09-15 00:06 UTC; this run
began late on 2026-09-14 in the local Pacific time zone. The same AMD target
completed the standard timing matrix: 10 complete repetitions per cell, 128
requested output tokens per request, raw mode, unique nonces, and concurrent
request batches of 1, 2, 4, and 8. Median
aggregate output throughput was:

| Model | batch 1 | batch 2 | batch 4 | batch 8 |
|---|---:|---:|---:|---:|
| `ministral-3:3b` | 43.0 | 43.9 | 45.3 | 44.9 tok/s |
| `ministral-3:8b` | 23.6 | 23.9 | 23.4 | 23.6 tok/s |
| `qwen3.8:27b` | 18.1 | 18.7 | 17.8 | 17.0 tok/s |

This is a timing and repeatability result only. The 27B row is a separate
larger-model matrix on the same target and uses the same 10 x 128 protocol.
It shows a small batch-2 gain followed by lower aggregate throughput at batch
4 and 8; it does not establish a general batch-serving law. The host has no
power sensor, so these records do not support an energy-per-request claim or a
cloud-versus-edge energy comparison. The raw records and bootstrap intervals
are in [`results/amd-8060s-standard-batched.json`](results/amd-8060s-standard-batched.json),
[`results/amd-8060s-large-20260915-batched.json`](results/amd-8060s-large-20260915-batched.json),
and [`statistics.json`](statistics.json).

## The bandwidth ladder

`--ladder` runs every locally installed model under the size cap and reports
τ × weight bytes: an *effective read bandwidth*. If single-stream decode streams
the whole weight set once per token, that product must sit under the part's
specified ceiling and should not drift as the model grows. Architectures that
read only part of their weights per token must sit above it — which is what
makes the statistic a test rather than a definition.

Seven models, 2.95–23.94 GB of weights, two quantisations, six architectures,
chosen by what was installed rather than by what would agree. Builds private to
this deployment are excluded: a row whose weights nobody else can download is
not a reproducible measurement, since the file size the statistic divides by
cannot be independently checked. `measured.py` drops them by name.

| Model | Weights | tok/s | τ·W (GB/s) | % of 256 | reads |
|---|---:|---:|---:|---:|---|
| `ministral-3:3b` | 2.95 GB | 78.6 | 232.2 | 91% | whole file |
| `ministral-3:8b` | 6.02 GB | 37.8 | 227.7 | 89% | whole file |
| `gemma4:e4b` | 9.61 GB | 56.4 | 542.1 | 212% | per-layer embeddings |
| `qwen3.8:27b` | 17.74 GB | 19.6 | 348.3 | 136% | multi-token prediction |
| `gemma4:26b` | 17.99 GB | 58.3 | 1049.4 | 410% | 8 of 128 experts |
| `muse-glimmer:30b` | 18.16 GB | 12.5 | 226.3 | 88% | whole file |
| `qwen3.6` | 23.94 GB | 62.2 | 1488.9 | 582% | 8 of 256 experts |

The two classes do not overlap, and the gap between them is 45 percentage
points. The three whole-file builds agree to within 2.6% of each other
(232.2, 227.7, 226.3 GB/s) across a 6.1× span in weight size, and none exceeds
the ceiling — which is what a whole-file reader cannot do.

The same ladder on the RTX 5070 (672 GB/s specified, 12 GB VRAM) gives
`qwen3:4b` 63%, `ministral-3:3b` 88% and `ministral-3:8b` 93%. Percentages are
only comparable after each is normalised to its own part, so the two machines
are always reported separately and never pooled.

Two biases, both small and both of known sign: the weight file includes a vision
projector on the builds that carry one, which text decoding does not stream
(inflates the figure), and KV-cache traffic is real and uncounted
(deflates it). Neither approaches the 29-point gap.

## Replication still needed

The repository already contains RTX 5070 board-power records, but both the
study and replication files are runs on the same named target. They estimate
repeatability, not independent-laboratory replication. The highest-value next
contribution is a separately operated machine, preferably with a wall meter,
using the same model digests and batch protocol.

On the NVIDIA machine, with Ollama installed:

```bash
ollama pull ministral-3:3b
ollama pull ministral-3:8b
python measure/bench.py --models ministral-3:3b ministral-3:8b --tag rtx5070
python measure/bench.py --ladder --bandwidth 672 --tag rtx5070
```

(672 GB/s is the RTX 5070's specified figure; use whatever the part sheet says
for the card you have — it is only used to express a percentage.)

Then copy the new raw record back into this directory and re-run
`python measure/statistics.py --resamples 20000` and
`python models/measured.py`.

## Concurrent serving and statistics

`--batch-sizes` sends independent requests concurrently and records one shared
power window per batch. A completed batch reports aggregate output throughput,
request wall times, and an explicitly labelled equal-split energy allocation.
That allocation is not a causal per-request energy measurement. Incomplete
requests remain in the raw record and are excluded from complete-batch
summaries; they are not silently treated as zero-cost successes.

The batch extension is an evidence collection path for the follow-up protocol,
not a claim that the current papers have measured data-centre-style serving:

```bash
python measure/statistics.py --resamples 20000
```

The resulting [`statistics.json`](statistics.json) recomputes run-level
medians, spread, bootstrap repeatability intervals, and the existing RTX study
versus replication comparison. The two RTX files are a second run on the same
named target, not an independent-laboratory replication.

`replication_manifest.json` records the public-safe target, operator and runtime
identity for each raw file. The analysis requires all three identities to differ
before it labels two files an independent replication. Missing identity is
reported as unresolved; same-target files remain repeatability evidence. Batch
reports also include paired speedup, per-request throughput, tail latency and
success-fraction summaries rather than only a median aggregate rate.
The intake validator checks the concurrent raw-record schema as well: an
independent contribution must declare batch sizes 1, 2, 4, and 8, provide at
least 10 repetitions per row, and carry a model digest in every model row.
This is a contribution gate, not evidence that a qualifying contribution has
arrived.
For each batch-size comparison, the complete repetitions are paired by their
recorded repetition IDs. The report includes a paired median throughput
contrast, a paired percentile bootstrap interval, and an exact two-sided
sign-flip test when the number of pairs is small enough to enumerate. Each
batch model also reports requested, observed, successful, failed-or-missing,
complete, and incomplete counts. These tests quantify within-target contrast
and repeatability; they do not create independent device replicates or a
population confidence interval.
The directory analysis also scans every manifest-listed raw record for
qualifying independent pairs, rather than assuming that the named RTX pair is
the only comparison. The current release has zero qualifying pairs because no
contributed record has a different target, operator and environment from an
existing record.

The statistics pass ignores JSON task manifests and adaptation score files
that do not contain raw serving repetitions. This keeps the measurement report
from presenting non-measurement records as models with empty timing results.

A small timing-only pilot is included at
[`results/local-batch-pilot-batched.json`](results/local-batch-pilot-batched.json):
two complete repetitions for Ministral 3B and 8B at batch sizes 1, 2, and 4 on
the AMD Radeon 8060S host. It is useful as a smoke-tested serving trace, but it
does not support an energy claim, a cross-hardware claim, or a population
estimate. The standard follow-up remains 10 repetitions at batch sizes 1, 2, 4,
and 8 with a declared wall-power boundary.

For the complete hardware, batch, cloud-boundary, and adaptation plan, see
[`REPLICATION_PROTOCOL.md`](REPLICATION_PROTOCOL.md) and
[`../paper/FOLLOW_UP_STUDY.md`](../paper/FOLLOW_UP_STUDY.md).

## Remaining evidence gaps

The current release still needs four results before its broadest claims can be
treated as general empirical findings:

1. **Whole-system energy.** The AMD batch matrix needs a wall-power trace for
   the complete batch window. The existing RTX board-power trace has a narrower
   boundary than a whole-system cloud comparison.
2. **Independent replication.** The RTX study and replication files are the
   same named target. A separate operator and physical machine are still
   required.
3. **Device-class coverage.** Higher-memory NVIDIA, Apple unified-memory, and
   another ARM or unified-memory target remain pending. A 12 GB card cannot
   hold the 17 GB Q4_K_M 27B build without offload or another recorded change in
   operating mode.
4. **Adaptation quality.** The public multi-model, multi-task score file now
   measures a synthetic held-out task-accuracy result: 3 model clusters, 5
   classes, 20 evaluation tasks per class and 3 adapter seeds on one physical
 host. It does not validate the inherited user-workload mix, human quality,
 or independent device replication. A supplemental prompt-surface holdout
 reports +10.0 percentage points, but reuses the same semantic records and
 answer labels, so it is a robustness probe rather than an independent
 replication. A new-semantic holdout uses 100 fresh synthetic records and
 reports only +1.4 percentage points, with mixed model directions, a -3.3 to
 +7.7 model-cluster interval, and exact p=1.000. It weakens broad transfer
 claims but is still one-host synthetic evidence, with no contamination or
 untouched general-ability suite. The old uplift curve remains a routing
 sensitivity scenario.

## What this does not claim

One serving stack in its default configuration, single-stream, on the authors'
own machines. Not a controlled comparison of accelerator architectures. Batch
size, quantisation, context length, driver and runtime version all move these
numbers, which is why each is recorded beside the result rather than averaged
away. The peak bandwidth the ladder expresses utilisation against is a part
specification, not something measured here.
