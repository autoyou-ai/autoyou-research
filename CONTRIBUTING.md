# Contributing to AutoYou Research

Corrections, reproducibility checks, and additional experiments are welcome.
Open an issue describing the claim, dataset, or code path you want to change.
For a new experiment, explain the hypothesis and measurement boundary first.

## Reproduce before editing

Follow the root README's offline commands for the paper you are changing, and run
its tests. Keep the six original measurement files unchanged: both the
peer-to-peer inference study and the Right-Sized paper pin them by SHA-256. Store
new experiments under distinct filenames and document hardware, runtime and
model versions, token counts, warm-up, repetitions, telemetry boundaries, and
unsuccessful runs.

Each study reads only its declared inputs. `review/audit.py` replays exactly the
six original records, so a new record in `measure/results/` does not change the
Right-Sized paper. If a study should use a new record, extend its input list and
tests explicitly; do not silently mix studies.

Files under `measure/results/` are archived byte for byte, including their line
endings, because other records cite them by hash. Everything else is stored with
LF line endings.

## Pull requests

Include the problem, change, commands run, and resulting evidence. For a numerical
change, show how it reaches the table or sentence in the paper. Regenerate affected
figures and inspect the rebuilt PDF. Then refresh the affected manifest:
`python review/source_manifest.py` for the Right-Sized paper, or
`python paper/release_manifest.py` for the other two papers. Distinguish
measurements, modeled assumptions, and illustrative scenarios. Preserve negative
results and uncertainty.

Do not upgrade an arithmetic consistency check to a claim of calibrated hardware
measurement, independent replication, or peer review. A change to one paper needs
its own evidence and does not validate the others.

Use synthetic data for tests. Do not include private conversations, customer data,
credentials, machine paths, or unpublished operational material. Do not add
model weights, except small adapters trained on public synthetic data with
their base-model licence recorded beside them. Preserve the MIT license and cite
the sources underlying a new parameter.
