# Follow-up study design

This is the minimum study needed to move the load-bearing claims beyond a
single-machine model and a private adaptation observation. It is designed to
be preregistered before data collection. The current repository does not claim
that these runs have been completed.

## A. Serving and footprint

1. Freeze the prompt manifest, output-token target, temperature, seed, model
   digests, quantization, runtime version, driver version, and power boundary.
2. Test at least three separately operated physical targets, including the
   existing unified-memory and discrete-GPU classes plus one higher-memory
   discrete, Apple unified, or ARM target. Record failures and unsupported
   configurations; do not discard them from the eligibility table.
3. Run at least three public model sizes or architectures on each eligible
   target. For each model, collect at least 10 complete repetitions at batch
   sizes 1, 2, 4, and 8 after a declared warm-up. Preserve raw request timing,
   token counts, power samples, temperature, and failure reason.
4. Use the same prompt and response manifest for local and cloud comparisons.
   Report prompt-cache policy, queueing, batching, output length, retry policy,
   provider region, and whether the cloud number is wall power, a published
   estimate, or a price conversion. Do not combine unlike boundaries into one
   percentage without a sensitivity table.
5. Estimate per-request energy at batch sizes above one from a measured shared
   power window and a declared allocation rule. If the goal is causal
   per-request energy, measure request-level or accelerator-level attribution;
   label equal splitting as an allocation, not a direct observation.

Primary estimands are median generation rate, median request energy, and the
local-minus-cloud difference under each declared accounting boundary. Report
95% repeatability intervals over repetitions and cluster-bootstrap intervals
over devices. A positive saving is not called robust unless its interval and
the boundary sensitivity support that wording.

## B. Workload capability and routing

Create the task manifest before measuring model quality. It must contain task
class, difficulty, expected output form, and an immutable task identifier, but
no personal data. Use a held-out set for every model and report exact scoring
rules or a blinded grader version. Estimate `f_s` from task outcomes rather
than assigning five class weights first. Keep the original weighted model as a
sensitivity comparison, not as the only result.

Predeclare the acceptance threshold grid, the treatment of abstentions, and
the cost of the router itself. Falsification is straightforward: if a
representative preregistered workload puts the local-sufficient fraction near
or below the lower external range, the 82% headline must be replaced by the
observed distribution.

## C. Multi-model adaptation experiment

The public release now includes a synthetic pilot that meets the minimum
matrix below, but it is still one physical host and does not sample real user
work. A follow-up should evaluate at least three base models or model
families, each with a matched adapted checkpoint and a frontier reference. Use
five fixed classes:
`extraction`, `rag_qa`, `summary`, `simple_code`, and `hard_reason`. Collect at
least 20 unique held-out tasks per class and at least three independent seeds
for each paired task when the training pipeline is stochastic. Keep training
and evaluation tasks disjoint, hash the manifest, and record checkpoint and
adapter digests.

The primary outcome should be absolute paired adapted-minus-base task accuracy
in percentage points, with the model-level mean as the inferential unit. Do not
divide the primary outcome by a frontier score, because that can amplify a
small reference denominator. Report the frontier accuracy separately as a
descriptive comparator. The primary analysis takes a mean within model, then a
mean across model clusters. It reports a model-cluster bootstrap interval and,
when there are few clusters, an exact cluster-level sign-flip permutation
p-value. Secondary outcomes are coverage at alpha values 0.70, 0.80, and
0.90, forgetting on an untouched general-ability suite, refusal behavior,
leakage probes, and task-specific or blinded human grading. Grade outputs
blinded to condition where practical, and publish task-level aggregate scores
without prompts or private answers.

The decision rule is strict: the measured synthetic pilot may replace the old
scenario only for its declared task-accuracy estimand. A real-workload or
product headline requires an independently operated device, a sampled task
corpus and an appropriate grader. If the interval includes zero, the model
clusters disagree, or the task distribution is not representative, retain the
scenario label for the affected claim and report no population uplift.

## D. Reproducibility and falsification

Archive the preregistration, environment lock, model and adapter hashes,
hardware inventory, raw records, exclusions, analysis output, and an exact
source revision. The existing implementation is a starter harness, not a
substitute for those records:

```bash
python measure/bench.py --batch-sizes 1 2 4 8 --reps 10 --tokens 300 --tag MACHINE_ID
python measure/statistics.py --resamples 20000
python models/adaptation_eval.py --resamples 20000
```

The study should be considered a failed replication, not a missing result, if
the declared measurements cannot be completed or if the accounting boundary
changes after seeing the outcome. A null adaptation result and a cloud model
that beats the local device are both publishable outcomes.
