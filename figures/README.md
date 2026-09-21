# Research figure gallery

## Current Track A revision

[review_energy.png](review_energy.png) separates recorded GPU-board energy from
modeled host and power-supply overhead. It appears in the current paper.

![Board energy and modeled system estimate](review_energy.png)

[review_bandwidth.png](review_bandwidth.png) is an optional descriptive diagnostic.
Throughput multiplied by model-file size is a proxy, not a hardware bandwidth
counter. Nine AMD ladder rows have aggregate summaries only.

Generate these figures with `python review/audit.py`.

## Retained earlier research figures

These figures remain available as historical research material. Their parameters
and scenario assumptions are in `models/`. The Track A revision does not validate
their routing coverage, cloud-displacement, cost, or fleet-saving conclusions.

| Figure | Subject |
| --- | --- |
| [Measured energy](fig_measured_energy.png) | Earlier presentation of recorded energy |
| [Capability](fig_capability.png) | Assumed task mix and capability thresholds |
| [Quality and coverage](fig_quality_vs_coverage.png) | Conditional adaptation scenario |
| [Memory wall](fig_memory_wall.png) | Capacity and memory-bandwidth model |
| [Device class](fig_device_class.png) | Modeled device constraints |
| [Method landscape](fig_method_landscape.png) | Parameter-efficient adaptation methods |
| [Adaptation break-even](fig_adaptation_breakeven.png) | Conditional amortization |
| [Adapter transport](fig_adapter_transport.png) | Adapter-size transfer model |
| [Sensitivity](fig_tornado.png) | Sensitivity of earlier footprint scenarios |

PDF and SVG counterparts are retained where present. Reuse figures with their
captions, units, assumptions, and revision status intact. Product dashboards and
mobile marketing assets do not belong in this research gallery.
