# Contributing to AutoYou Research

Corrections, reproducibility checks, and additional experiments are welcome.
Open an issue describing the claim, dataset, or code path you want to change.
For a new experiment, explain the hypothesis and measurement boundary first.

## Reproduce before editing

Follow the root README's offline audit and test commands. Keep the six original
measurement files unchanged. Store new experiments under distinct filenames and
document hardware, runtime/model versions, token counts, warm-up, repetitions,
telemetry boundaries, and unsuccessful runs.

The revised Track A audit expects its original six-file study. Extend its input
selection and tests explicitly before introducing other schemas into
`measure/results/`; do not silently mix studies.

## Pull requests

Include the problem, change, commands run, and resulting evidence. For a numerical
change, show how it reaches the table or sentence in the paper. Regenerate affected
figures and inspect the rebuilt PDF. Distinguish measurements, modeled assumptions,
and illustrative scenarios. Preserve negative results and uncertainty.

Do not upgrade an arithmetic consistency check to a claim of calibrated hardware
measurement, independent replication, or peer review. Changes to the companion
paper need their own evidence.

Use synthetic data for tests. Do not include private conversations, customer data,
credentials, machine paths, model weights, or unpublished operational material.
Preserve the MIT license and cite the sources underlying a new parameter.
