# Public adaptation benchmark record

Status: measured synthetic benchmark, independent hardware replication pending.

The public score file is `results/adaptation-multimodel.json`, summarized by
`../models/adaptation_eval.py`. The task manifest is
`results/adaptation_tasks.json`. The adapter files are under
`results/adaptation_adapters/`.

## Primary estimand

The primary outcome is absolute paired held-out task accuracy. Each task is
scored 1 when the model's greedy choice is the correct option and 0 otherwise.
The reported uplift is adapted accuracy minus base accuracy in percentage
points. The frontier reference is reported descriptively and is not used as a
denominator.

## Measured record

- 3 base models: Qwen2.5 0.5B Instruct, Qwen2.5 3B Instruct, and Qwen2.5-VL
  3B Instruct.
- 1 measured reference: Qwen2.5-VL 7B Instruct.
- 5 task classes: extraction, RAG question answering, summary, simple code,
  and hard reasoning.
- 12 training tasks and 20 held-out evaluation tasks per class.
- 3 adapter-training seeds per model.
- Base accuracy: 86.7%.
- Adapted accuracy: 97.8%.
- Point estimate: +11.1 percentage points.
- Model-cluster bootstrap interval: +5.0 to +18.3 percentage points, using
  20,000 resamples.
- Exact two-sided model-cluster sign-flip p-value: 0.250.
- Class-level point estimates: extraction 0.0 pp, RAG question answering
  0.0 pp, summary +8.3 pp, simple code +8.3 pp, and hard reasoning +38.9 pp.

Class-level tests are secondary and cluster-aware. A Bonferroni correction is
applied across the five task-class hypotheses. The hard-reasoning raw
cluster-sign-flip p-value is 0.250 and its adjusted value is 1.000; all five
adjusted values are 1.000. With only three model clusters, these p-values are
coarse, so no class-level effect is treated as confirmatory.

The evaluation uses one physical AMD unified-memory machine with CPU-only
PyTorch execution. The three seeds are repeated training conditions, not
independent devices or laboratories. The tasks are public synthetic records,
not a sampled user workload. Training and evaluation IDs are disjoint, but the
generator reuses a small set of class templates and the same four-choice
format across splits. This leaves format learning, template transfer and
contamination unresolved. The grader is a deterministic multiple-choice
choice grader, not a human or task-specific quality assessor.

## Supplemental prompt-surface holdout

The file `results/adaptation-template-holdout.json` applies one changed outer
and class-specific prompt wording to the public evaluation records, assigns new
task identifiers, and rescored the same three models, five classes and three
adapter seeds. It is a robustness probe for prompt wording, not an independent
replication and not a new semantic workload sample.

- Base accuracy: 80.0%.
- Adapted accuracy: 90.0%.
- Frontier reference accuracy: 95.0%.
- Point estimate: +10.0 percentage points.
- Model-cluster bootstrap interval: +5.0 to +15.0 percentage points, using
  20,000 resamples.
- Exact two-sided model-cluster sign-flip p-value: 0.250.
- Per-model point estimates: +15.0 pp, +5.0 pp and +10.0 pp for the 0.5B,
  3B and VLM 3B models respectively.

The holdout applies the same secondary five-class Bonferroni family. Its
hard-reasoning raw p-value is 0.500 and its adjusted value is 1.000; no
task-class effect is confirmatory there either.

The direction and scale are similar to the primary +11.1 pp result, but the
probe does not establish semantic transfer: the underlying records, answer
choices and correct labels are reused, only one paraphrase is tested, and no
contamination or untouched general-ability suite is included. The result is
therefore reported separately and does not change the coverage model.

## New-semantic holdout

The file `results/adaptation-semantic-holdout.json` scores the same three
models and three saved adapter-training seeds on 100 fresh synthetic records,
20 per class. The generator uses new record values and task formats and does
not copy primary evaluation records, prompts or answers. This is a
distribution-shift check, not independent hardware replication or a sampled
user workload.

- Base accuracy: 90.3%.
- Adapted accuracy: 91.8%.
- Frontier reference accuracy: 95.0%.
- Point estimate: +1.4 percentage points.
- Model-cluster bootstrap interval: -3.3 to +7.7 percentage points, using
  20,000 resamples.
- Exact two-sided model-cluster sign-flip p-value: 1.000.
- Per-model point estimates: +7.7 pp, -3.3 pp and 0.0 pp for the 0.5B,
  3B and VLM 3B models respectively.

The mixed model directions and interval spanning zero are counterevidence
against a broad semantic-transfer interpretation of the primary +11.1 pp
result. The adapter-training distribution remains the primary synthetic
benchmark, and the holdout has no contamination or untouched general-ability
suite. It therefore narrows the claim but does not validate a product-level
quality effect.

## Reproduction

Install the versions in `../requirements-adaptation.txt`, obtain the named
model snapshots independently, and run the model runner with at least three
`alias=local_snapshot_path` entries and one local frontier snapshot. The
runner records public model references and snapshot digests, while the
evaluator recomputes the cluster bootstrap and exact sign-flip test:

```text
python models/run_adaptation_experiment.py --model ALIAS=LOCAL_SNAPSHOT --model ALIAS=LOCAL_SNAPSHOT --model ALIAS=LOCAL_SNAPSHOT --frontier LOCAL_FRONTIER --tasks measure/results/adaptation_tasks.json --output measure/results/adaptation-multimodel.json --adapter-root measure/results/adaptation_adapters
python models/adaptation_eval.py --resamples 20000
python models/adaptation_template_holdout.py --model ALIAS=LOCAL_SNAPSHOT --model ALIAS=LOCAL_SNAPSHOT --model ALIAS=LOCAL_SNAPSHOT --frontier LOCAL_FRONTIER --reuse-adapters --tasks measure/results/adaptation-template-holdout-tasks.json --output measure/results/adaptation-template-holdout.json --adapter-root measure/results/adaptation_adapters
python models/adaptation_eval.py --input measure/results/adaptation-template-holdout.json --output models/adaptation_template_holdout_evaluation.json --resamples 20000
python models/adaptation_semantic_holdout.py --model ALIAS=LOCAL_SNAPSHOT --model ALIAS=LOCAL_SNAPSHOT --model ALIAS=LOCAL_SNAPSHOT --frontier LOCAL_FRONTIER --reuse-adapters --tasks measure/results/adaptation-semantic-holdout-tasks.json --output measure/results/adaptation-semantic-holdout.json --adapter-root measure/results/adaptation_adapters
python models/adaptation_eval.py --input measure/results/adaptation-semantic-holdout.json --output models/adaptation_semantic_holdout_evaluation.json --resamples 20000
```

## Required next replication

This record does not validate the real-workload coverage model. The next study
must sample tasks before seeing outcomes, include independently operated
hardware and additional model families, add unseen evaluation templates,
contamination probes and an untouched general-ability regression suite, and
use task-specific or blinded human grading where the claim requires it. It
must preserve the absolute-accuracy estimand and report model, device,
operator and task as separate experimental units.
