# Primary measurement

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
| **τ** throughput | a third-party benchmark blog for an RTX 4090 | **measured**, per model, per machine |
| **β** prefill overhead | an author assumption of 0.15 | **measured** at 0.051–0.078 for the study's own query shape |
| **P** board power | a vendor TDP figure | **measured** where a sensor exists; explicitly *not* measured where none does |

A reviewer is entitled to ask why three borrowed numbers describe hardware
nobody in the study had touched. Two of the three no longer are.

## Run it

```bash
python measure/bench.py --list
python measure/bench.py --tag my-machine
python measure/bench.py --ladder --bandwidth 256 --tag my-machine
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

## Measured so far

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

## Wanted: a discrete-GPU run

**If you have an NVIDIA machine, this is the single most valuable thing you can
contribute**, because `nvidia-smi` exposes board power with no extra install —
so that run measures all three terms, including `P`, and turns the study's
modelled per-query energy into a measured one.

On the NVIDIA machine, with Ollama installed:

```bash
ollama pull ministral-3:3b
ollama pull ministral-3:8b
python measure/bench.py --models ministral-3:3b ministral-3:8b --tag rtx5070
python measure/bench.py --ladder --bandwidth 672 --tag rtx5070
```

(672 GB/s is the RTX 5070's specified figure; use whatever the part sheet says
for the card you have — it is only used to express a percentage.)

Then copy `results/rtx5070*.json` back into this directory and re-run
`python models/measured.py`.

Four things that run settles, none of which the study can currently do from
measurement:

1. **Per-query energy, measured.** The one number the entire paper models.
2. **C2, the consumer-GPU efficiency counter-claim.** Currently supported from
   two cited figures; a measured mWh/token on a real consumer card either
   confirms it or does not.
3. **H7, the device-class claim**, by experiment rather than by arithmetic. A
   12 GB card cannot hold `qwen3.8:27b` (a 17 GB Q4_K_M build). Adding it to
   `--models` and recording what happens — offload, thrash, or refusal — is a
   direct demonstration of the memory wall, and a negative result is as useful
   here as a positive one.
4. **The bandwidth ladder on a second device class.** The whole-file band and
   the two-class separation are currently one machine's result. A discrete GPU
   has four times the bandwidth and a fifth of the pool; if the same band and
   the same separation appear there, the finding is about autoregressive decode
   rather than about this part.

## What this does not claim

One serving stack in its default configuration, single-stream, on the authors'
own machines. Not a controlled comparison of accelerator architectures. Batch
size, quantisation, context length, driver and runtime version all move these
numbers, which is why each is recorded beside the result rather than averaged
away. The peak bandwidth the ladder expresses utilisation against is a part
specification, not something measured here.
