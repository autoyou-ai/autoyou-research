"""Create a conservative claim-level citation triage report.

The normal citation audit proves key coverage and records exact manuscript
contexts. This companion report adds a source-specific action for every cited
key. It intentionally refuses to certify semantic support from metadata alone.
The report is therefore a publication checklist, not a claim that every source
has been fully read or that every sentence is supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

try:
    from .audit_citations import DEFAULT_BIB, DEFAULT_TEX, audit
except ImportError:  # direct execution: python paper/citation_claim_review.py
    from audit_citations import DEFAULT_BIB, DEFAULT_TEX, audit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_JSON = ROOT / "paper" / "citation_claim_review.json"
DEFAULT_REVIEW_MARKDOWN = ROOT / "paper" / "CITATION_CLAIM_REVIEW.md"


# These are conservative findings from the cited contexts and the locator
# type. They do not assert that the underlying source is false; they identify
# where the current citation cannot, by itself, carry the prose around it.
KNOWN_SCOPE_ISSUES: Dict[str, Tuple[str, str, str]] = {
    "frontier2026": (
        "locator_scope_mismatch",
        "The locator is an API pricing page. It can support a dated price input, "
        "but not release chronology, model-family identity, architecture, or energy.",
        "Keep it only for the visible dated price and archive the price snapshot "
        "or an alternate first-party locator before submission.",
    ),
    "anthropic_pricing2026": (
        "locator_scope_mismatch",
        "The locator is a pricing page. It can support a dated price input, "
        "but not release chronology, model identity, architecture, or energy.",
        "Keep it only for the price visible on the archived date and preserve a "
        "snapshot or alternate first-party locator before submission.",
    ),
    "google_pricing2026": (
        "locator_scope_mismatch",
        "The locator is a pricing page. It can support a dated price input, "
        "but not release chronology, model identity, architecture, or energy.",
        "Keep it only for the price visible on the archived date and preserve a "
        "snapshot or alternate first-party locator before submission.",
    ),
    "openweights2026": (
        "weak_primary_locator",
        "An OpenRouter listing is an aggregator, not a manufacturer architecture "
        "specification or independent energy measurement.",
        "Replace with Moonshot and Zhipu primary model cards or narrow the sentence "
        "to a current aggregator listing.",
    ),
    "techtarget2025": (
        "weak_quantitative_source",
        "Trade press is not a sufficient basis for a general quantitative energy "
        "effect. The current text uses the source only for directional context.",
        "Keep it non-load-bearing, or replace the directional statement with a "
        "reproducible primary measurement and matched boundary.",
    ),
    "antmedia2026": (
        "weak_quantitative_source",
        "The vendor note does not establish the study's STUN-direct and TURN-relay "
        "population fractions as a measured deployment statistic.",
        "Treat the fractions as an explicit scenario range or cite a measurement "
        "from a representative deployment.",
    ),
    "databasemart2026": (
        "superseded_third_party_measurement",
        "This is a third-party benchmark blog and is still an input to the legacy "
        "RTX 4090 sensitivity path even though the new campaign replaced it for "
        "the primary measurement.",
        "Keep it only in a clearly labelled historical sensitivity, or replace the "
        "path with a reproducible local measurement.",
    ),
    "solartech2025": (
        "weak_quantitative_source",
        "A consumer guide is a weak basis for desktop idle-power bounds used in "
        "the energy model.",
        "Measure wall idle power for the actual host or present the range solely "
        "as a sensitivity assumption.",
    ),
    "tuning_pricing2026": (
        "weak_time_sensitive_source",
        "A third-party price aggregator is not a stable primary record of vendor "
        "fine-tuning prices or tuned-endpoint multipliers.",
        "Archive each vendor price page and recompute the TCO with a dated price "
        "snapshot and provider terms.",
    ),
    "strixhalo2026": (
        "community_non_peer_reviewed",
        "The 27B memory validation rests on a community guide, not a peer-reviewed "
        "paper or an independently archived training log.",
        "Publish the run manifest, logs, checkpoint digest, and wall-power trace, "
        "or label this as a single public recipe check.",
    ),
    "autoyou_support2026": (
        "not_independently_reproducible",
        "The source is restricted, has no public locator, and exposes only aggregate "
        "scores. It cannot support a public population or multi-model claim.",
        "Keep the numbers as deployment-reported observations only and replace "
        "them with the public multi-model, multi-task evaluation.",
    ),
    "mistral2024": (
        "vendor_benchmark_scope_limited",
        "The source is a vendor release and its benchmark chart is not a matched "
        "independent head-to-head evaluation.",
        "Use it only for vendor-reported figures and remove any cross-model causal "
        "interpretation.",
    ),
    "aws_water2025": (
        "vendor_scope_limited",
        "A vendor sustainability page can support the vendor's own WUE report, not "
        "a universal hyperscale or industry value without an independent survey.",
        "Label the value as vendor-reported and propagate provider and facility "
        "variation through the sensitivity analysis.",
    ),
    "nvidia_rtx4090": (
        "vendor_specification_only",
        "The source supports the board specification, not measured application "
        "power during this study's workload.",
        "Use it only as a bound and keep the measured-vs-modelled distinction.",
    ),
    "rtx5070spec": (
        "vendor_specification_only",
        "The source supports the part specification, not the achieved bandwidth or "
        "whole-system energy of the experiment.",
        "Keep the specification citation separate from the local measurement.",
    ),
    "amd_strix2025": (
        "vendor_specification_only",
        "The source supports advertised hardware specifications, not measured "
        "runtime bandwidth or power on the study host.",
        "Keep the hardware facts as specifications and label runtime values as "
        "measured only where they are measured.",
    ),
}


# These are targeted source reads completed against the current manuscript.
# They are deliberately scope checks, not blanket certification of every
# sentence that cites the key. Unlisted keys remain explicitly not_checked in
# the generated report so that a reachable URL cannot be mistaken for review.
CHECKED_SOURCE_EVIDENCE: Dict[str, Dict[str, str]] = {
    "belcak2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2506.02153v2",
        "evidence": (
            "The v2 paper is a position and value argument. Its appendix presents "
            "case-study estimates for MetaGPT, Open Operator, and Cradle; the "
            "40-70% figure is not an independent population measurement."
        ),
        "limit": "Use for author-reported case studies and design rationale, not causal or population inference; the source remains a preprint under review.",
    },
    "elsworth2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2508.15734v1",
        "evidence": (
            "The production methodology reports a median Gemini text prompt of "
            "0.24 Wh and decomposes the full-stack footprint, including the "
            "42% non-compute share. The boundary excludes end-user devices."
        ),
        "limit": "Google production TPU and facility context; not a household edge measurement.",
    },
    "jegham2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/pdf/2505.09598v2",
        "evidence": (
            "Version 2 Table 4 reports the values used here, including 1.788 Wh "
            "for GPT-4o, 39.223 Wh for o3, and 0.603 Wh for Llama-3.1 8B. The "
            "framework combines public API data, environmental multipliers, and "
            "inferred hardware; batch-8 and long-reasoning values are modeled "
            "outputs."
        ),
        "limit": "Pin v2 for these numbers because the later v6 revision changes the table. Describe the result as an estimate or framework output, not a direct whole-system wall-power measurement.",
    },
    "caravaca2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2511.05597v1",
        "evidence": (
            "The paper reports more than 32,500 GPU inference measurements over "
            "21 GPU configurations and 155 model architectures, with a "
            "predictive model and input/output analysis."
        ),
        "limit": "Cloud GPU and vLLM measurement scope; it does not validate the edge population.",
    },
    "hu2021lora": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://openreview.net/pdf?id=nZeVKeeFYf9",
        "evidence": (
            "The ICLR paper defines the frozen-base low-rank update and evaluates "
            "trainable-parameter and memory reductions on its reported tasks."
        ),
        "limit": "Method definition and reported experiments do not establish the current product workload effect.",
    },
    "dettmers2023qlora": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://papers.nips.cc/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html",
        "evidence": (
            "The NeurIPS paper and DOI record NF4, double quantization, paged "
            "optimizers, and large-model fine-tuning results."
        ),
        "limit": "The reported fit and training tradeoff depend on checkpoint, context, optimizer, and runtime.",
    },
    "hayou2024loraplus": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2402.12354v2",
        "evidence": (
            "The paper motivates separate learning rates for the LoRA A and B "
            "matrices and reports 1-2 percent performance improvements and up "
            "to about 2x fine-tuning speedup at the same computational cost in "
            "its evaluated settings."
        ),
        "limit": "The result is a preprint's own experiments and does not establish an edge or product workload effect.",
    },
    "jain2023neftune": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2310.05914v2",
        "evidence": (
            "The paper adds embedding noise during fine-tuning and reports "
            "large AlpacaEval gains, including 29.79 percent to 64.69 percent "
            "for LLaMA-2-7B on Alpaca, plus smaller gains on other datasets."
        ),
        "limit": "The result depends on the named models, datasets, evaluator, and hyperparameters; it is not a generic quality uplift.",
    },
    "kalajdzievski2023rslora": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2312.03732v1",
        "evidence": (
            "The paper analyzes the rank scaling factor, proposes alpha over "
            "sqrt(rank), and reports higher-rank training gains with no change "
            "in inference computing cost because the merged adapter has the "
            "same form as LoRA."
        ),
        "limit": "This is a theoretical and benchmark result for the paper's configurations, not a measured serving result on the current device.",
    },
    "kopiczko2023vera": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2310.11454v2",
        "evidence": (
            "The paper uses shared low-rank matrices and learned scaling vectors, "
            "and its instruction-tuning table reports roughly 100x fewer trainable "
            "parameters while closely matching LoRA on its MT-Bench setup."
        ),
        "limit": "The comparison uses the paper's models, tasks, and GPT-4 judging; it is not a matched AutoYou workload result.",
    },
    "meng2024pissa": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2404.02948v4",
        "evidence": (
            "The paper initializes low-rank factors from principal singular "
            "components and reports comparisons across 11 models, 5 NLG tasks, "
            "and 8 NLU tasks, including faster convergence and higher scores in "
            "the reported setups."
        ),
        "limit": "The reported task results do not establish the current product workload or the one-time initialization cost on the study host.",
    },
    "buyukakyuz2024olora": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2406.01775v2",
        "evidence": (
            "The preprint proposes QR-based orthonormal initialization and reports "
            "faster convergence and improved performance than standard LoRA across "
            "its evaluated language-modeling tasks."
        ),
        "limit": "It is not a measured cost comparison for the declared edge device, and the paper's efficiency wording is configuration-dependent.",
    },
    "paischer2024eva": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2410.07170v5",
        "evidence": (
            "The NeurIPS 2025 paper describes incremental SVD over activation "
            "minibatches and "
            "rank redistribution, and reports experiments over 51 tasks and four "
            "domains while also documenting cases where another method wins."
        ),
        "limit": "Its average ranking is not a guarantee for one task class, and the paper notes constraints such as low-rank settings and a static downstream dataset.",
    },
    "yang2024corda": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2406.05223v3",
        "evidence": (
            "The paper constructs context-oriented decompositions from activation "
            "covariance and reports knowledge-preserved and instruction-previewed "
            "modes on math, code, and instruction-following tasks."
        ),
        "limit": "The reported forgetting mitigation and performance are paper-specific experiments, not evidence about a private deployment.",
    },
    "li2023loftq": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2310.08659v4",
        "evidence": (
            "The paper jointly quantizes weights and computes a low-rank initialization "
            "to reduce the quantization discrepancy, and evaluates NLU, question "
            "answering, summarization, and NLG tasks on A100 GPUs."
        ),
        "limit": "The reported gains and initialization cost depend on model, bit width, rank, and GPU; they are not current edge measurements.",
    },
    "liu2024dora": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://proceedings.mlr.press/v235/liu24bn.html",
        "evidence": (
            "The ICML 2024 proceedings page defines magnitude-direction "
            "decomposition and reports downstream comparisons, including no "
            "additional inference overhead in the described construction."
        ),
        "limit": "The reported gain range is not a matched edge benchmark.",
    },
    "zhao2024galore": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://proceedings.mlr.press/v235/zhao24s.html",
        "evidence": (
            "The ICML 2024 proceedings page reports gradient low-rank projection, "
            "optimizer-state memory reduction, and full-parameter training."
        ),
        "limit": "Its continued-pretraining results do not constitute an adapter transport result.",
    },
    "zhang2023adalora": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://openreview.net/pdf?id=lq62uWRJjiY",
        "evidence": (
            "The ICLR paper describes adaptive allocation of a fixed parameter "
            "budget across layers during fine-tuning."
        ),
        "limit": "No current study run verifies its effect on the declared task classes.",
    },
    "balazy2024loraxs": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://doi.org/10.3233/FAIA251185",
        "evidence": (
            "The ECAI 2025 paper places a trainable square matrix between frozen "
            "SVD-derived factors and reports over 100x storage reduction for a "
            "7B comparison, with evaluations on GLUE, GSM8K, MATH, and commonsense "
            "benchmarks."
        ),
        "limit": "The storage and accuracy results are paper-specific and do not establish a quality or transport gain on the current deployment.",
    },
    "biderman2024lora": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://openreview.net/pdf?id=aloEru2qCG",
        "evidence": (
            "The TMLR paper compares LoRA and full fine-tuning and reports less "
            "in-domain learning together with less out-of-domain forgetting in "
            "its evaluated settings."
        ),
        "limit": "The result is not a guarantee for a private corpus or product deployment.",
    },
    "qi2023finetuning": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://openreview.net/pdf?id=hTEGyKf0dZ",
        "evidence": (
            "The ICLR paper reports safety degradation after fine-tuning, "
            "including adversarial examples and degradation under benign data "
            "in the evaluated setup."
        ),
        "limit": "The paper's models and probes do not establish the private deployment's outcome.",
    },
    "ucriverside2023": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://doi.org/10.1145/3724499",
        "evidence": (
            "The Communications of the ACM article discusses direct and indirect "
            "water use in AI systems and emphasizes that water estimates depend on "
            "location, infrastructure, and accounting boundary."
        ),
        "limit": "It is a broad explanatory article and does not supply the study's chosen global EWIF point estimate.",
    },
    "luccioni2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://doi.org/10.1145/3715275.3732007",
        "evidence": (
            "The FAccT 2025 paper frames efficiency gains as potentially producing "
            "rebound effects and argues that direct operational accounting is only "
            "part of the broader environmental question."
        ),
        "limit": "It does not measure the study's 0.20 rebound factor; that value remains an explicit scenario assumption.",
    },
    "lorasurvey2024": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://doi.org/10.1007/s11704-024-40663-9",
        "evidence": (
            "The published Frontiers of Computer Science survey reviews LoRA and "
            "its variants as a methods taxonomy and literature summary."
        ),
        "limit": "A survey is background, not a matched experiment or evidence for the current workload's rankings.",
    },
    "rfc8831": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://www.rfc-editor.org/rfc/rfc8831",
        "evidence": (
            "The RFC defines WebRTC data channels over SCTP and their security "
            "relationship to DTLS."
        ),
        "limit": "A protocol standard does not establish endpoint security, metadata minimization, or application logging.",
    },
    "rfc8827": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://www.rfc-editor.org/rfc/rfc8827",
        "evidence": (
            "The RFC specifies WebRTC security architecture and the DTLS-SRTP "
            "fingerprint binding used to authenticate the peer transport."
        ),
        "limit": "It does not prove confidentiality from an authenticated endpoint or a compromised peer.",
    },
    "amd_strix2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html",
        "evidence": (
            "AMD's official product page identifies Ryzen AI MAX+ 395 and Strix Halo "
            "with up to 128 GB of unified memory, up to 40 RDNA 3.5 compute "
            "units, and 256 GB/s specified memory bandwidth for the Radeon 8060S."
        ),
        "limit": "Vendor specifications do not establish achieved runtime bandwidth, power, or the local ladder result; the measured ladder remains a separate local record.",
    },
    "anthropic_pricing2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://claude.com/pricing",
        "evidence": (
            "The current first-party pricing page is a dynamic price list and "
            "supports using dated API rates as economic inputs."
        ),
        "limit": "It does not support release chronology, architecture, energy, or performance claims; the cited historical URL redirects, so an archived snapshot is needed for a fixed price claim.",
    },
    "antmedia2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://antmedia.io/how-to-create-webrtc-peer-to-peer-communication/",
        "evidence": (
            "The vendor article defines STUN for NAT discovery and TURN as a "
            "fallback relay, and cites a survey for an approximately 15 to 20 "
            "percent TURN rate."
        ),
        "limit": "It does not establish this study's 0.80 direct and 0.15 TURN deployment mix; retain those fractions as an explicit scenario or replace them with a representative deployment measurement.",
    },
    "aslan2018": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://onlinelibrary.wiley.com/doi/10.1111/jiec.12630",
        "evidence": (
            "The Wiley article discusses historical data-transfer energy estimates "
            "and gives a 0.06 kWh per GB estimate for 2015."
        ),
        "limit": "The study's 0.006 to 0.06 range and the 2026 link-energy assumption are author-selected inputs, not direct measurements by this paper.",
    },
    "autoyou2026adaptation": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://github.com/autoyou-ai/autoyou-research",
        "evidence": (
            "The current companion manuscript and models/results.json contain the "
            "adaptation scenarios, memory validation, and a public synthetic "
            "multi-model, multi-task accuracy evaluation with task-level records."
        ),
        "limit": "The measured result uses three model clusters on one physical host and synthetic four-choice tasks; it is not an independent-device replication, human quality study, or population workload estimate.",
    },
    "autoyou2026edge": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://github.com/autoyou-ai/autoyou-research",
        "evidence": (
            "The current local companion manuscript and models/results.json contain "
            "the scoped fleet, carbon, water, cost, and verdict calculations."
        ),
        "limit": "These are same-project analyses and measurements, not independent external evidence; conclusions remain conditional on the declared assumptions and boundaries.",
    },
    "autoyou2026measure": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://github.com/autoyou-ai/autoyou-research",
        "evidence": (
            "The current local measure harness and raw JSON traces contain the "
            "two-machine protocol, NVIDIA board-power traces, and AMD timing "
            "records; the release manifest hashes those artifacts."
        ),
        "limit": "The local overlay is not yet archived or pushed, AMD lacks rail-power capture, and the two named machines are repeatability records rather than independent-lab replications.",
    },
    "autoyou_support2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "",
        "evidence": (
            "Local aggregate reports contain five deployed-support evaluation "
            "artifacts with screen, code, refusal, leak, support, and stale-answer "
            "fields; no public locator or task-level rows are available."
        ),
        "limit": "The evidence is private, the base-model labels are inconsistent across reports, and it is not reproducible or population-level evidence; retain it only as deployment-reported observation.",
    },
    "aws_water2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://sustainability.aboutamazon.com/natural-resources/water",
        "evidence": (
            "AWS and Amazon sustainability material reports 0.15 L/kWh for 2024 "
            "and 0.12 L/kWh for 2025, and describes an approximately 0.84 "
            "industry comparison."
        ),
        "limit": "This is vendor-reported direct WUE for the vendor's boundary, not a universal cloud or data-center value; the study's central 0.20 remains a model assumption.",
    },
    "databasemart2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090",
        "evidence": (
            "The page is a third-party hosting-vendor benchmark and supports only "
            "a legacy RTX 4090 and vLLM performance context; it recommends under "
            "8B use and warns that 8B and larger models need more VRAM."
        ),
        "limit": "The exact 141 tok/s value was not independently verified from the page; the manuscript correctly retains it only as a legacy scenario and replaces it with direct measurements.",
    },
    "devriesgao2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://pubmed.ncbi.nlm.nih.gov/41583976/",
        "evidence": (
            "The PubMed record confirms the Patterns article and DOI and reports "
            "a 2025 AI carbon range of 32.6 to 79.7 MtCO2e and a water range of "
            "312.5 to 764.6 billion litres while discussing disclosure gaps."
        ),
        "limit": "It does not directly establish the study's 0.84 withdrawal WUE or local query energy; those require separate source and boundary checks.",
    },
    "ember2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://ember-energy.org/",
        "evidence": (
            "Ember's Global Electricity Review is the cited source family for "
            "global and regional grid-carbon statistics and annual electricity "
            "mix reporting."
        ),
        "limit": "The official report page was not retrievable in this audit; the exact 458 gCO2e/kWh global and 140/708 regional values must be checked against the downloadable report before publication.",
    },
    "eu_eed2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://energy.ec.europa.eu/topics/energy-efficiency/energy-efficiency-targets-directive-and-rules/energy-efficiency-directive/energy-performance-data-centres_en",
        "evidence": (
            "The European Commission page states that data-center electricity "
            "consumption is projected to more than double to 945 TWh by 2030 and "
            "that the Energy Efficiency Directive introduced monitoring and "
            "reporting through a database and planned rating scheme."
        ),
        "limit": "The page does not supply a universal PUE cap; the manuscript should not model one as an established regulatory constant.",
    },
    "frontier2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://openai.com/business/pricing/",
        "evidence": (
            "The first-party pricing page is a dynamic price list and supports "
            "only a dated economic input."
        ),
        "limit": "It does not support release, architecture, energy, or performance claims; the older URL redirects and a dated snapshot is required for a fixed price claim.",
    },
    "google_pricing2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://ai.google.dev/gemini-api/docs/pricing",
        "evidence": (
            "Google's current official Gemini pricing page provides per-million-"
            "token tables and warns that prices may change, supporting a dated "
            "economic input."
        ),
        "limit": "It does not support model energy, architecture, or release claims; the older URL redirects and a dated price snapshot is needed.",
    },
    "iea2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://www.iea.org/reports/energy-and-ai",
        "evidence": (
            "The IEA page identifies Energy and AI as a 2025 report based on "
            "global and regional modeling and datasets; the official European "
            "Commission summary repeats the more-than-double to 945 TWh by 2030 "
            "projection."
        ),
        "limit": "This is a modeled projection, not a measurement of current demand; any AI-specific share or exact table value needs the report table itself.",
    },
    "ministral32026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://ollama.com/library/ministral-3",
        "evidence": (
            "The Ollama page lists Ministral 3B and 8B distribution entries at "
            "roughly 3.0 GB and 6.0 GB, 256K context, text and image input, and "
            "edge deployment positioning."
        ),
        "limit": "Exact 2.95 GB and 6.02 GB values are local build metadata; this vendor page is not a head-to-head quality or energy benchmark.",
    },
    "mistral2024": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://mistral.ai/news/ministraux/",
        "evidence": (
            "The Mistral release is a vendor announcement and provides the cited "
            "Ministral benchmark chart and model positioning."
        ),
        "limit": "The vendor protocol does not establish a matched independent head-to-head comparison with the Llama figures; retain only vendor-reported results.",
    },
    "museglimmer2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://huggingface.co/meta-models/Muse-Glimmer-30B",
        "evidence": (
            "The Meta model card identifies Muse Glimmer as an Apache-2.0 dense "
            "model of about 30B parameters with a vision encoder, and reports "
            "under-20-GB quantized variants together with benchmark context."
        ),
        "limit": "Exact 18.16 GB and local bandwidth-ladder values are local distribution metadata; model-card benchmark claims are not independent measurements.",
    },
    "ni2025benchmarks": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2508.15361v1",
        "evidence": (
            "The survey reviews benchmark contamination and motivates dynamic and "
            "Google-Proof benchmark designs."
        ),
        "limit": "It does not provide a measured correction for the current MMLU values or validate this study's frontier range; avoid turning the survey into a quantified adjustment.",
    },
    "nvidia_rtx4090": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/",
        "evidence": (
            "NVIDIA's official page records 450 W total graphics power, an 850 W "
            "recommended system power, and notes how idle and gaming power are "
            "measured."
        ),
        "limit": "These specifications and product measurements are not runtime board power for this workload; the study's 420 W value remains an explicit assumption unless measured.",
    },
    "nvidiasmi2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://docs.nvidia.com/deploy/nvidia-smi/",
        "evidence": (
            "NVIDIA's documentation defines instantaneous power draw as the last "
            "measured whole-board draw and documents the power.draw.instant and "
            "power.draw.average query fields."
        ),
        "limit": "Driver counter cadence limits precision, and board telemetry is not whole-system wall-meter energy.",
    },
    "ollamaapi2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://github.com/ollama/ollama/blob/main/docs/api.md",
        "evidence": (
            "Ollama API documentation defines raw=true to bypass formatting, seed "
            "for reproducible outputs, and generation timing and token counters."
        ),
        "limit": "The documentation has moved between URLs and does not validate hardware throughput, power, or quality results.",
    },
    "peft_library2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://huggingface.co/docs/peft/v0.20.0/package_reference/lora",
        "evidence": (
            "The Hugging Face PEFT documentation exposes DoRA, rsLoRA, and "
            "initialization choices including PiSSA, OLoRA, EVA, CoRDA, and "
            "LoftQ; it warns that mixed adapters add inference overhead."
        ),
        "limit": "Library support is not method efficacy on this workload, and documentation is version-sensitive.",
    },
    "qwen25vl2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
        "evidence": (
            "The Qwen model card identifies Qwen2.5-VL 7B, its Apache-2.0 license, "
            "the 3B, 7B, and 72B family, vision and agent capabilities, and "
            "vendor benchmark tables."
        ),
        "limit": "A vendor model card and benchmark table do not establish the private adapter's base-model pairing or its quality result.",
    },
    "qwen35_2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://huggingface.co/Qwen/Qwen3.5-27B",
        "evidence": (
            "The official Qwen3.5-27B model card identifies a 27B Apache-2.0 "
            "vision-language model with native 262144-token context and supplies "
            "the model identity behind the documented training run."
        ),
        "limit": "The public model card does not validate the community training telemetry; the community guide remains only a single recipe anchor.",
    },
    "qwen38_2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://huggingface.co/Qwen/Qwen3.8-27B",
        "evidence": (
            "The official Qwen3.8-27B model card identifies an Apache-2.0 27B "
            "native vision-language model with 262144-token context."
        ),
        "limit": "Model-card facts do not establish the separate distribution size, local runtime energy, performance, or quality transfer.",
    },
    "qwen38_ollama2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://ollama.com/library/qwen3.8:27b",
        "evidence": (
            "The official Ollama distribution page lists a 27.3B Q4_K_M build "
            "with an 18 GB size."
        ),
        "limit": "Distribution metadata does not establish exact local file size after caching, whole-system memory, throughput, energy, or quality transfer.",
    },
    "reddi2020mlperf": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://doi.org/10.1145/3380702.3380721",
        "evidence": (
            "The MLPerf Inference paper describes standardized inference scenarios, "
            "including single-stream and edge/server divisions."
        ),
        "limit": "Curated vendor submissions do not establish a commodity unified-memory APU gap; the exact current table or paper passage must be checked before using a numeric comparison.",
    },
    "rtx5070spec": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://www.nvidia.com/en-us/geforce/graphics-cards/compare/",
        "evidence": (
            "NVIDIA's comparison page lists the RTX 5070 with 12 GB GDDR7, a "
            "192-bit bus, and 672 GB/s specified memory bandwidth."
        ),
        "limit": "These are part specifications only, not runtime throughput or whole-system energy measurements.",
    },
    "schneider2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2502.01671v1",
        "evidence": (
            "The preprint presents a cradle-to-grave life-cycle assessment of five "
            "TPUs, separates operational and embodied emissions, and introduces "
            "compute carbon intensity."
        ),
        "limit": "The retrieved source does not establish the paper's exact 15 to 30 percent parameter; that range remains a cited or modeled input and needs an exact table check.",
    },
    "solartech2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://solartechonline.com/blog/average-household-power-consumption-guide-2025/",
        "evidence": (
            "The consumer guide discusses desktop and household electricity draw "
            "and can serve only as contextual range input."
        ),
        "limit": "The exact 30 to 150 W range was not located in the current redirected page; it is not a load measurement or peer-reviewed source and should remain a wide sensitivity only or be replaced by meter data.",
    },
    "strixhalo2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide",
        "evidence": (
            "The public GitHub guide documents a tested Qwen3.5-27B LoRA recipe on "
            "a 128 GB Strix Halo system, including 448 steps and r=128/alpha=256 "
            "settings."
        ),
        "limit": "This is a community single-machine, unrefereed anchor rather than independent replication or formal publication; publish the run manifest, checkpoint digest, logs, and wall-power trace for stronger evidence.",
    },
    "techtarget2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://www.techtarget.com/it-strategy/feature/Can-edge-computing-make-AI-more-sustainable",
        "evidence": (
            "The trade article says edge can offer potential sustainability benefits "
            "and recommends energy per inference as more useful than PUE for edge."
        ),
        "limit": "This is directional trade press, not quantitative causal evidence; keep it non-load-bearing or replace it with a matched primary measurement.",
    },
    "tuning_pricing2026": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://pricepertoken.com/fine-tuning",
        "evidence": (
            "The third-party pricing index supplies a snapshot of fine-tuning and "
            "token prices and provider comparisons."
        ),
        "limit": "It is a dynamic aggregator and the exact 1.5x tuned-endpoint claim was not independently verified in the current read; keep the comparison as a scenario and archive provider prices and terms.",
    },
    "uptime2025": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://intelligence.uptimeinstitute.com/resource/uptime-institute-global-data-center-survey-2025",
        "evidence": (
            "The official Uptime page identifies the 2025 Global Data Center Survey "
            "and states that average PUE changed little for the sixth consecutive "
            "year."
        ),
        "limit": "The exact 1.54 value is behind a gated report in this audit; archive the report or an accessible table before treating that number as independently checked.",
    },
    "wang2024slm": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://arxiv.org/html/2411.03350v2",
        "evidence": (
            "The survey discusses small language models in terms of lower latency, "
            "cost, customization, and adaptation, and organizes edge and mobile "
            "deployment considerations."
        ),
        "limit": "It is a preprint with a metadata anomaly in the rendered page, and survey evidence does not establish the current workload ratios or a causal product result.",
    },
    "williams2009roofline": {
        "semantic_status": "checked_scope_limited",
        "evidence_locator": "https://doi.org/10.1145/1498765.1498785",
        "evidence": (
            "The CACM record identifies the Roofline model for reasoning about "
            "operational intensity and bandwidth or compute ceilings."
        ),
        "limit": "The general model premise does not prove that single-stream decode or the current proxy W equals bytes read per token; the manuscript states this proxy limitation.",
    },
}


def classify_claim(row: Dict[str, object]) -> Dict[str, str]:
    key = str(row["key"])
    if key in KNOWN_SCOPE_ISSUES:
        status, basis, action = KNOWN_SCOPE_ISSUES[key]
    else:
        flags = set(row.get("flags") or [])
        quality = str(row.get("source_quality") or "")
        if "private_or_restricted" in flags or quality == "restricted":
            status = "not_independently_reproducible"
            basis = "The cited artifact is restricted or not publicly locatable."
            action = "Do not use it for a public population claim; replace or label as internal evidence."
        elif "community_source" in flags or "trade_press" in flags or quality == "secondary_non_peer_reviewed":
            status = "weak_source_load_bearing_claims"
            basis = "The source is secondary, community, or trade material."
            action = "Use only for explicitly scoped context, or replace for a load-bearing number."
        elif "third_party_source" in flags:
            status = "third_party_scope_check_required"
            basis = "The source is not the primary owner of the claimed measurement or price."
            action = "Verify the original measurement or archive the primary provider record."
        elif quality == "formal_or_peer_reviewed_candidate":
            status = "formal_candidate_exact_claim_check_required"
            basis = "A formal venue or DOI is indicated, but exact claim support still requires source reading."
            action = "Record the exact table, page, or section supporting each cited sentence."
        elif "arxiv_preprint" in flags:
            status = "preprint_exact_claim_check_required"
            basis = "The locator is an arXiv preprint; peer-review and exact claim scope are not established here."
            action = "Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording."
        elif quality == "official_or_vendor":
            status = "official_factual_scope_only"
            basis = "The source is official or vendor material and may support product or standard facts only."
            action = "Do not use it as independent evidence of general performance or causal effect."
        else:
            status = "manual_exact_claim_check_required"
            basis = "Metadata and locator checks do not establish semantic support."
            action = "Read the source against every cited context and narrow the sentence if needed."

    if "locator_access_denied" in set(row.get("flags") or []):
        basis += " The network probe received access denied; this is not proof that the locator is dead."
        action += " Add an archived copy or alternate public locator before submission."
    return {"status": status, "basis": basis, "action": action}


def build_review(result: Dict[str, object]) -> Dict[str, object]:
    entries = []
    counts: Dict[str, int] = {}
    semantic_counts: Dict[str, int] = {}
    for row in result["entries"]:
        review = classify_claim(row)
        counts[review["status"]] = counts.get(review["status"], 0) + 1
        evidence = CHECKED_SOURCE_EVIDENCE.get(row["key"], {
            "semantic_status": "not_checked",
            "evidence_locator": "",
            "evidence": "No targeted source read is recorded for this key yet.",
            "limit": "Exact claim, population, boundary, date, and uncertainty still require manual source checking.",
        })
        semantic_status = evidence["semantic_status"]
        semantic_counts[semantic_status] = semantic_counts.get(semantic_status, 0) + 1
        entries.append({
            "key": row["key"],
            "source_quality": row["source_quality"],
            "flags": row["flags"],
            "locator": row.get("doi") or row.get("url"),
            "cited_at": [
                f"{item['file']}:{item['line']}"
                for item in row["citation_occurrences"]
            ],
            "claim_contexts": [
                {
                    "file": item["file"],
                    "line": item["line"],
                    "context": item["context"],
                }
                for item in row["citation_occurrences"]
            ],
            "claim_context_count": len(row["citation_occurrences"]),
            "semantic_status": semantic_status,
            "evidence_locator": evidence["evidence_locator"],
            "evidence": evidence["evidence"],
            "semantic_limit": evidence["limit"],
            **review,
        })
    return {
        "schema_version": "1.0",
        "status": "conservative source-level triage; exact semantic support is not certified",
        "source_audit_summary": result["summary"],
        "reviewed_key_count": len(entries),
        "status_counts": counts,
        "semantic_status_counts": semantic_counts,
        "targeted_source_checks": len(CHECKED_SOURCE_EVIDENCE),
        "entries": entries,
    }


def write_markdown(review: Dict[str, object], path: Path) -> None:
    lines = [
        "# Citation claim review",
        "",
        "This is a conservative source-level triage of every cited BibTeX key.",
        "It is not a claim that every exact sentence has been semantically verified.",
        "The full citation contexts remain in `CITATION_AUDIT.md`.",
        "",
        f"- Cited keys triaged: {review['reviewed_key_count']}",
        f"- Targeted source-scope checks recorded: {review['targeted_source_checks']}",
        "- Exact sentence support certified: 0 (all nearby contexts still require a final manual pass)",
        "- Status counts: " + ", ".join(
            f"{key}={value}" for key, value in sorted(review["status_counts"].items())
        ),
        "",
        "| Key | Source class | Claim review status | Semantic check | Cited at | Required action |",
        "|---|---|---|---|---|---|",
    ]
    for entry in review["entries"]:
        locations = ", ".join(entry["cited_at"]) or "unused"
        action = str(entry["action"]).replace("|", "\\|")
        lines.append(
            f"| {entry['key']} | {entry['source_quality']} | "
            f"{entry['status']} | {entry['semantic_status']} | {locations} | {action} |"
        )
    lines.extend([
        "",
        "## Targeted source notes",
        "",
        "The following notes record what was checked in the linked source. A",
        "checked-scope-limited entry can support only the stated method or result",
        "scope; it does not certify every nearby sentence or transferability to",
        "the AutoYou workload.",
        "",
    ])
    for entry in review["entries"]:
        if entry["semantic_status"] == "not_checked":
            continue
        evidence = str(entry["evidence"]).replace("|", "\\|")
        limit = str(entry["semantic_limit"]).replace("|", "\\|")
        locator = entry["evidence_locator"] or entry["locator"] or "missing"
        lines.extend([
            f"### {entry['key']}",
            "",
            f"- Locator: {locator}",
            f"- Evidence: {evidence}",
            f"- Limit: {limit}",
            f"- Nearby manuscript contexts: {entry['claim_context_count']}",
            "",
        ])
    lines.extend([
        "## Reading rule",
        "",
        "A citation is adequate only when the source supports the exact nearby",
        "claim, including its population, boundary, date, metric, and uncertainty.",
        "A formal venue or a successful URL does not remove that requirement.",
        "Vendor, trade, community, aggregator, internal, and preprint sources may",
        "be useful, but the manuscript must state what they actually establish.",
        "Targeted source checks are scope notes, not blanket certification of every",
        "nearby citation context; exact sentence support still requires a final",
        "manual pass over the recorded contexts.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_REVIEW_MARKDOWN)
    args = parser.parse_args(argv)
    result = audit(DEFAULT_BIB, DEFAULT_TEX, network=args.network)
    review = build_review(result)
    args.json.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_markdown(review, args.markdown)
    print(f"wrote {args.json}")
    print(f"wrote {args.markdown}")
    print(f"reviewed keys: {review['reviewed_key_count']}")
    return 0 if review["reviewed_key_count"] == result["summary"]["cited_keys"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
