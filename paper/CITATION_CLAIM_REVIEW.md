# Citation claim review

This is a conservative source-level triage of every cited BibTeX key.
It is not a claim that every exact sentence has been semantically verified.
The full citation contexts remain in `CITATION_AUDIT.md`.

- Cited keys triaged: 64
- Targeted source-scope checks recorded: 64
- Exact sentence support certified: 0 (all nearby contexts still require a final manual pass)
- Status counts: community_non_peer_reviewed=1, formal_candidate_exact_claim_check_required=16, locator_scope_mismatch=3, manual_exact_claim_check_required=12, not_independently_reproducible=1, official_factual_scope_only=5, preprint_exact_claim_check_required=15, superseded_third_party_measurement=1, vendor_benchmark_scope_limited=1, vendor_scope_limited=1, vendor_specification_only=3, weak_quantitative_source=3, weak_source_load_bearing_claims=1, weak_time_sensitive_source=1

| Key | Source class | Claim review status | Semantic check | Cited at | Required action |
|---|---|---|---|---|---|
| amd_strix2025 | non_peer_reviewed_or_unspecified | vendor_specification_only | checked_scope_limited | paper/p2p-inference.tex:825 | Keep the hardware facts as specifications and label runtime values as measured only where they are measured. |
| anthropic_pricing2026 | non_peer_reviewed_or_unspecified | locator_scope_mismatch | checked_scope_limited | paper/p2p-inference.tex:227, paper/p2p-inference.tex:468, paper/p2p-inference.tex:661 | Keep it only for the price visible on the archived date and preserve a snapshot or alternate first-party locator before submission. |
| antmedia2026 | official_or_vendor | weak_quantitative_source | checked_scope_limited | paper/p2p-inference.tex:244, paper/p2p-inference.tex:465 | Treat the fractions as an explicit scenario range or cite a measurement from a representative deployment. |
| aslan2018 | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:466, paper/p2p-inference.tex:1235 | Record the exact table, page, or section supporting each cited sentence. Add an archived copy or alternate public locator before submission. |
| autoyou2026adaptation | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:88, paper/p2p-inference.tex:1116, paper/p2p-inference.tex:1287, paper/p2p-inference.tex:1310 | Read the source against every cited context and narrow the sentence if needed. |
| autoyou2026edge | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:52, paper/adaptation.tex:122 | Read the source against every cited context and narrow the sentence if needed. |
| autoyou2026measure | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:1327, paper/adaptation.tex:855 | Read the source against every cited context and narrow the sentence if needed. |
| autoyou_support2026 | restricted | not_independently_reproducible | checked_scope_limited | paper/adaptation.tex:617, paper/adaptation.tex:861 | Keep the numbers as deployment-reported observations only and replace them with the public multi-model, multi-task evaluation. |
| aws_water2025 | non_peer_reviewed_or_unspecified | vendor_scope_limited | checked_scope_limited | paper/p2p-inference.tex:135, paper/p2p-inference.tex:198, paper/p2p-inference.tex:454 | Label the value as vendor-reported and propagate provider and facility variation through the sensitivity analysis. |
| balazy2024loraxs | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:236 | Record the exact table, page, or section supporting each cited sentence. |
| belcak2025 | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:141, paper/p2p-inference.tex:216, paper/p2p-inference.tex:219, paper/p2p-inference.tex:304, paper/p2p-inference.tex:542, paper/p2p-inference.tex:558, paper/p2p-inference.tex:658, paper/p2p-inference.tex:1233, paper/p2p-inference.tex:1280 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| biderman2024lora | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:359, paper/adaptation.tex:680 | Record the exact table, page, or section supporting each cited sentence. Add an archived copy or alternate public locator before submission. |
| buyukakyuz2024olora | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:214 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| caravaca2025 | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:258 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| databasemart2026 | official_or_vendor | superseded_third_party_measurement | checked_scope_limited | paper/p2p-inference.tex:459 | Keep it only in a clearly labelled historical sensitivity, or replace the path with a reproducible local measurement. |
| dettmers2023qlora | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:224 | Record the exact table, page, or section supporting each cited sentence. |
| devriesgao2025 | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:135, paper/p2p-inference.tex:198 | Record the exact table, page, or section supporting each cited sentence. Add an archived copy or alternate public locator before submission. |
| elsworth2025 | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:127, paper/p2p-inference.tex:206, paper/p2p-inference.tex:256, paper/p2p-inference.tex:377, paper/p2p-inference.tex:453, paper/p2p-inference.tex:457, paper/p2p-inference.tex:521, paper/adaptation.tex:130 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| ember2026 | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:204, paper/p2p-inference.tex:456 | Read the source against every cited context and narrow the sentence if needed. |
| eu_eed2026 | official_or_vendor | official_factual_scope_only | checked_scope_limited | paper/p2p-inference.tex:194 | Do not use it as independent evidence of general performance or causal effect. |
| frontier2026 | non_peer_reviewed_or_unspecified | locator_scope_mismatch | checked_scope_limited | paper/p2p-inference.tex:227, paper/p2p-inference.tex:468, paper/p2p-inference.tex:661 | Keep it only for the visible dated price and archive the price snapshot or an alternate first-party locator before submission. Add an archived copy or alternate public locator before submission. |
| google_pricing2026 | non_peer_reviewed_or_unspecified | locator_scope_mismatch | checked_scope_limited | paper/p2p-inference.tex:227, paper/p2p-inference.tex:468, paper/p2p-inference.tex:661 | Keep it only for the price visible on the archived date and preserve a snapshot or alternate first-party locator before submission. |
| hayou2024loraplus | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:204 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| hu2021lora | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:194 | Record the exact table, page, or section supporting each cited sentence. Add an archived copy or alternate public locator before submission. |
| iea2025 | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:126, paper/p2p-inference.tex:470 | Read the source against every cited context and narrow the sentence if needed. Add an archived copy or alternate public locator before submission. |
| jain2023neftune | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:207 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| jegham2025 | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:129, paper/p2p-inference.tex:205, paper/p2p-inference.tex:252, paper/p2p-inference.tex:377, paper/p2p-inference.tex:457, paper/p2p-inference.tex:458, paper/p2p-inference.tex:521 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| kalajdzievski2023rslora | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:199 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| kopiczko2023vera | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:235 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| li2023loftq | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:218 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| liu2024dora | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:231 | Record the exact table, page, or section supporting each cited sentence. |
| lorasurvey2024 | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:195 | Record the exact table, page, or section supporting each cited sentence. |
| luccioni2025 | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:291, paper/p2p-inference.tex:467, paper/p2p-inference.tex:1173 | Record the exact table, page, or section supporting each cited sentence. Add an archived copy or alternate public locator before submission. |
| meng2024pissa | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:212 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| ministral32026 | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:830 | Read the source against every cited context and narrow the sentence if needed. |
| mistral2024 | official_or_vendor | vendor_benchmark_scope_limited | checked_scope_limited | paper/p2p-inference.tex:214, paper/p2p-inference.tex:537 | Use it only for vendor-reported figures and remove any cross-model causal interpretation. |
| museglimmer2026 | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:830 | Read the source against every cited context and narrow the sentence if needed. |
| ni2025benchmarks | official_or_vendor | preprint_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:545 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| nvidia_rtx4090 | official_or_vendor | vendor_specification_only | checked_scope_limited | paper/p2p-inference.tex:461 | Use it only as a bound and keep the measured-vs-modelled distinction. |
| nvidiasmi2026 | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:893 | Read the source against every cited context and narrow the sentence if needed. |
| ollamaapi2026 | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:725 | Read the source against every cited context and narrow the sentence if needed. |
| paischer2024eva | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:215 | Record the exact table, page, or section supporting each cited sentence. |
| peft_library2026 | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:195 | Read the source against every cited context and narrow the sentence if needed. |
| qi2023finetuning | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:608, paper/adaptation.tex:643 | Record the exact table, page, or section supporting each cited sentence. Add an archived copy or alternate public locator before submission. |
| qwen25vl2025 | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:618 | Read the source against every cited context and narrow the sentence if needed. |
| qwen35_2026 | official_or_vendor | weak_source_load_bearing_claims | checked_scope_limited | paper/adaptation.tex:289 | Use only for explicitly scoped context, or replace for a load-bearing number. |
| qwen38_2026 | official_or_vendor | official_factual_scope_only | checked_scope_limited | paper/adaptation.tex:143 | Do not use it as independent evidence of general performance or causal effect. |
| qwen38_ollama2026 | official_or_vendor | official_factual_scope_only | checked_scope_limited | paper/adaptation.tex:145 | Do not use it as independent evidence of general performance or causal effect. |
| reddi2020mlperf | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:263 | Record the exact table, page, or section supporting each cited sentence. |
| rfc8827 | official_or_vendor | official_factual_scope_only | checked_scope_limited | paper/p2p-inference.tex:246, paper/p2p-inference.tex:1105 | Do not use it as independent evidence of general performance or causal effect. |
| rfc8831 | official_or_vendor | official_factual_scope_only | checked_scope_limited | paper/p2p-inference.tex:246, paper/p2p-inference.tex:1105 | Do not use it as independent evidence of general performance or causal effect. |
| rtx5070spec | non_peer_reviewed_or_unspecified | vendor_specification_only | checked_scope_limited | paper/p2p-inference.tex:860 | Keep the specification citation separate from the local measurement. |
| schneider2025 | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:293 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| solartech2025 | non_peer_reviewed_or_unspecified | weak_quantitative_source | checked_scope_limited | paper/p2p-inference.tex:464 | Measure wall idle power for the actual host or present the range solely as a sensitivity assumption. |
| strixhalo2026 | secondary_non_peer_reviewed | community_non_peer_reviewed | checked_scope_limited | paper/adaptation.tex:149, paper/adaptation.tex:292 | Publish the run manifest, logs, checkpoint digest, and wall-power trace, or label this as a single public recipe check. |
| techtarget2025 | secondary_non_peer_reviewed | weak_quantitative_source | checked_scope_limited | paper/p2p-inference.tex:237, paper/p2p-inference.tex:1237 | Keep it non-load-bearing, or replace the directional statement with a reproducible primary measurement and matched boundary. |
| tuning_pricing2026 | official_or_vendor | weak_time_sensitive_source | checked_scope_limited | paper/adaptation.tex:594 | Archive each vendor price page and recompute the TCO with a dated price snapshot and provider terms. |
| ucriverside2023 | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:199 | Record the exact table, page, or section supporting each cited sentence. Add an archived copy or alternate public locator before submission. |
| uptime2025 | non_peer_reviewed_or_unspecified | manual_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:132, paper/p2p-inference.tex:190, paper/p2p-inference.tex:453 | Read the source against every cited context and narrow the sentence if needed. |
| wang2024slm | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:213, paper/p2p-inference.tex:541 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| williams2009roofline | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/p2p-inference.tex:281, paper/p2p-inference.tex:802 | Record the exact table, page, or section supporting each cited sentence. Add an archived copy or alternate public locator before submission. |
| yang2024corda | non_peer_reviewed_or_unspecified | preprint_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:217, paper/adaptation.tex:687 | Read the cited result in full, record page or section evidence, and avoid peer-reviewed wording. |
| zhang2023adalora | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:234 | Record the exact table, page, or section supporting each cited sentence. Add an archived copy or alternate public locator before submission. |
| zhao2024galore | formal_or_peer_reviewed_candidate | formal_candidate_exact_claim_check_required | checked_scope_limited | paper/adaptation.tex:242 | Record the exact table, page, or section supporting each cited sentence. |

## Targeted source notes

The following notes record what was checked in the linked source. A
checked-scope-limited entry can support only the stated method or result
scope; it does not certify every nearby sentence or transferability to
the AutoYou workload.

### amd_strix2025

- Locator: https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html
- Evidence: AMD's official product page identifies Ryzen AI MAX+ 395 and Strix Halo with up to 128 GB of unified memory, up to 40 RDNA 3.5 compute units, and 256 GB/s specified memory bandwidth for the Radeon 8060S.
- Limit: Vendor specifications do not establish achieved runtime bandwidth, power, or the local ladder result; the measured ladder remains a separate local record.
- Nearby manuscript contexts: 1

### anthropic_pricing2026

- Locator: https://claude.com/pricing
- Evidence: The current first-party pricing page is a dynamic price list and supports using dated API rates as economic inputs.
- Limit: It does not support release chronology, architecture, energy, or performance claims; the cited historical URL redirects, so an archived snapshot is needed for a fixed price claim.
- Nearby manuscript contexts: 3

### antmedia2026

- Locator: https://antmedia.io/how-to-create-webrtc-peer-to-peer-communication/
- Evidence: The vendor article defines STUN for NAT discovery and TURN as a fallback relay, and cites a survey for an approximately 15 to 20 percent TURN rate.
- Limit: It does not establish this study's 0.80 direct and 0.15 TURN deployment mix; retain those fractions as an explicit scenario or replace them with a representative deployment measurement.
- Nearby manuscript contexts: 2

### aslan2018

- Locator: https://onlinelibrary.wiley.com/doi/10.1111/jiec.12630
- Evidence: The Wiley article discusses historical data-transfer energy estimates and gives a 0.06 kWh per GB estimate for 2015.
- Limit: The study's 0.006 to 0.06 range and the 2026 link-energy assumption are author-selected inputs, not direct measurements by this paper.
- Nearby manuscript contexts: 2

### autoyou2026adaptation

- Locator: https://github.com/autoyou-ai/autoyou-research
- Evidence: The current companion manuscript and models/results.json contain the adaptation scenarios, memory validation, and a public synthetic multi-model, multi-task accuracy evaluation with task-level records.
- Limit: The measured result uses three model clusters on one physical host and synthetic four-choice tasks; it is not an independent-device replication, human quality study, or population workload estimate.
- Nearby manuscript contexts: 4

### autoyou2026edge

- Locator: https://github.com/autoyou-ai/autoyou-research
- Evidence: The current local companion manuscript and models/results.json contain the scoped fleet, carbon, water, cost, and verdict calculations.
- Limit: These are same-project analyses and measurements, not independent external evidence; conclusions remain conditional on the declared assumptions and boundaries.
- Nearby manuscript contexts: 2

### autoyou2026measure

- Locator: https://github.com/autoyou-ai/autoyou-research
- Evidence: The current local measure harness and raw JSON traces contain the two-machine protocol, NVIDIA board-power traces, and AMD timing records; the release manifest hashes those artifacts.
- Limit: The local overlay is not yet archived or pushed, AMD lacks rail-power capture, and the two named machines are repeatability records rather than independent-lab replications.
- Nearby manuscript contexts: 2

### autoyou_support2026

- Locator: missing
- Evidence: Local aggregate reports contain five deployed-support evaluation artifacts with screen, code, refusal, leak, support, and stale-answer fields; no public locator or task-level rows are available.
- Limit: The evidence is private, the base-model labels are inconsistent across reports, and it is not reproducible or population-level evidence; retain it only as deployment-reported observation.
- Nearby manuscript contexts: 2

### aws_water2025

- Locator: https://sustainability.aboutamazon.com/natural-resources/water
- Evidence: AWS and Amazon sustainability material reports 0.15 L/kWh for 2024 and 0.12 L/kWh for 2025, and describes an approximately 0.84 industry comparison.
- Limit: This is vendor-reported direct WUE for the vendor's boundary, not a universal cloud or data-center value; the study's central 0.20 remains a model assumption.
- Nearby manuscript contexts: 3

### balazy2024loraxs

- Locator: https://doi.org/10.3233/FAIA251185
- Evidence: The ECAI 2025 paper places a trainable square matrix between frozen SVD-derived factors and reports over 100x storage reduction for a 7B comparison, with evaluations on GLUE, GSM8K, MATH, and commonsense benchmarks.
- Limit: The storage and accuracy results are paper-specific and do not establish a quality or transport gain on the current deployment.
- Nearby manuscript contexts: 1

### belcak2025

- Locator: https://arxiv.org/html/2506.02153v2
- Evidence: The v2 paper is a position and value argument. Its appendix presents case-study estimates for MetaGPT, Open Operator, and Cradle; the 40-70% figure is not an independent population measurement.
- Limit: Use for author-reported case studies and design rationale, not causal or population inference; the source remains a preprint under review.
- Nearby manuscript contexts: 9

### biderman2024lora

- Locator: https://openreview.net/pdf?id=aloEru2qCG
- Evidence: The TMLR paper compares LoRA and full fine-tuning and reports less in-domain learning together with less out-of-domain forgetting in its evaluated settings.
- Limit: The result is not a guarantee for a private corpus or product deployment.
- Nearby manuscript contexts: 2

### buyukakyuz2024olora

- Locator: https://arxiv.org/html/2406.01775v2
- Evidence: The preprint proposes QR-based orthonormal initialization and reports faster convergence and improved performance than standard LoRA across its evaluated language-modeling tasks.
- Limit: It is not a measured cost comparison for the declared edge device, and the paper's efficiency wording is configuration-dependent.
- Nearby manuscript contexts: 1

### caravaca2025

- Locator: https://arxiv.org/html/2511.05597v1
- Evidence: The paper reports more than 32,500 GPU inference measurements over 21 GPU configurations and 155 model architectures, with a predictive model and input/output analysis.
- Limit: Cloud GPU and vLLM measurement scope; it does not validate the edge population.
- Nearby manuscript contexts: 1

### databasemart2026

- Locator: https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090
- Evidence: The page is a third-party hosting-vendor benchmark and supports only a legacy RTX 4090 and vLLM performance context; it recommends under 8B use and warns that 8B and larger models need more VRAM.
- Limit: The exact 141 tok/s value was not independently verified from the page; the manuscript correctly retains it only as a legacy scenario and replaces it with direct measurements.
- Nearby manuscript contexts: 1

### dettmers2023qlora

- Locator: https://papers.nips.cc/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html
- Evidence: The NeurIPS paper and DOI record NF4, double quantization, paged optimizers, and large-model fine-tuning results.
- Limit: The reported fit and training tradeoff depend on checkpoint, context, optimizer, and runtime.
- Nearby manuscript contexts: 1

### devriesgao2025

- Locator: https://pubmed.ncbi.nlm.nih.gov/41583976/
- Evidence: The PubMed record confirms the Patterns article and DOI and reports a 2025 AI carbon range of 32.6 to 79.7 MtCO2e and a water range of 312.5 to 764.6 billion litres while discussing disclosure gaps.
- Limit: It does not directly establish the study's 0.84 withdrawal WUE or local query energy; those require separate source and boundary checks.
- Nearby manuscript contexts: 2

### elsworth2025

- Locator: https://arxiv.org/html/2508.15734v1
- Evidence: The production methodology reports a median Gemini text prompt of 0.24 Wh and decomposes the full-stack footprint, including the 42% non-compute share. The boundary excludes end-user devices.
- Limit: Google production TPU and facility context; not a household edge measurement.
- Nearby manuscript contexts: 8

### ember2026

- Locator: https://ember-energy.org/
- Evidence: Ember's Global Electricity Review is the cited source family for global and regional grid-carbon statistics and annual electricity mix reporting.
- Limit: The official report page was not retrievable in this audit; the exact 458 gCO2e/kWh global and 140/708 regional values must be checked against the downloadable report before publication.
- Nearby manuscript contexts: 2

### eu_eed2026

- Locator: https://energy.ec.europa.eu/topics/energy-efficiency/energy-efficiency-targets-directive-and-rules/energy-efficiency-directive/energy-performance-data-centres_en
- Evidence: The European Commission page states that data-center electricity consumption is projected to more than double to 945 TWh by 2030 and that the Energy Efficiency Directive introduced monitoring and reporting through a database and planned rating scheme.
- Limit: The page does not supply a universal PUE cap; the manuscript should not model one as an established regulatory constant.
- Nearby manuscript contexts: 1

### frontier2026

- Locator: https://openai.com/business/pricing/
- Evidence: The first-party pricing page is a dynamic price list and supports only a dated economic input.
- Limit: It does not support release, architecture, energy, or performance claims; the older URL redirects and a dated snapshot is required for a fixed price claim.
- Nearby manuscript contexts: 3

### google_pricing2026

- Locator: https://ai.google.dev/gemini-api/docs/pricing
- Evidence: Google's current official Gemini pricing page provides per-million-token tables and warns that prices may change, supporting a dated economic input.
- Limit: It does not support model energy, architecture, or release claims; the older URL redirects and a dated price snapshot is needed.
- Nearby manuscript contexts: 3

### hayou2024loraplus

- Locator: https://arxiv.org/html/2402.12354v2
- Evidence: The paper motivates separate learning rates for the LoRA A and B matrices and reports 1-2 percent performance improvements and up to about 2x fine-tuning speedup at the same computational cost in its evaluated settings.
- Limit: The result is a preprint's own experiments and does not establish an edge or product workload effect.
- Nearby manuscript contexts: 1

### hu2021lora

- Locator: https://openreview.net/pdf?id=nZeVKeeFYf9
- Evidence: The ICLR paper defines the frozen-base low-rank update and evaluates trainable-parameter and memory reductions on its reported tasks.
- Limit: Method definition and reported experiments do not establish the current product workload effect.
- Nearby manuscript contexts: 1

### iea2025

- Locator: https://www.iea.org/reports/energy-and-ai
- Evidence: The IEA page identifies Energy and AI as a 2025 report based on global and regional modeling and datasets; the official European Commission summary repeats the more-than-double to 945 TWh by 2030 projection.
- Limit: This is a modeled projection, not a measurement of current demand; any AI-specific share or exact table value needs the report table itself.
- Nearby manuscript contexts: 2

### jain2023neftune

- Locator: https://arxiv.org/html/2310.05914v2
- Evidence: The paper adds embedding noise during fine-tuning and reports large AlpacaEval gains, including 29.79 percent to 64.69 percent for LLaMA-2-7B on Alpaca, plus smaller gains on other datasets.
- Limit: The result depends on the named models, datasets, evaluator, and hyperparameters; it is not a generic quality uplift.
- Nearby manuscript contexts: 1

### jegham2025

- Locator: https://arxiv.org/pdf/2505.09598v2
- Evidence: Version 2 Table 4 reports the values used here, including 1.788 Wh for GPT-4o, 39.223 Wh for o3, and 0.603 Wh for Llama-3.1 8B. The framework combines public API data, environmental multipliers, and inferred hardware; batch-8 and long-reasoning values are modeled outputs.
- Limit: Pin v2 for these numbers because the later v6 revision changes the table. Describe the result as an estimate or framework output, not a direct whole-system wall-power measurement.
- Nearby manuscript contexts: 7

### kalajdzievski2023rslora

- Locator: https://arxiv.org/html/2312.03732v1
- Evidence: The paper analyzes the rank scaling factor, proposes alpha over sqrt(rank), and reports higher-rank training gains with no change in inference computing cost because the merged adapter has the same form as LoRA.
- Limit: This is a theoretical and benchmark result for the paper's configurations, not a measured serving result on the current device.
- Nearby manuscript contexts: 1

### kopiczko2023vera

- Locator: https://arxiv.org/html/2310.11454v2
- Evidence: The paper uses shared low-rank matrices and learned scaling vectors, and its instruction-tuning table reports roughly 100x fewer trainable parameters while closely matching LoRA on its MT-Bench setup.
- Limit: The comparison uses the paper's models, tasks, and GPT-4 judging; it is not a matched AutoYou workload result.
- Nearby manuscript contexts: 1

### li2023loftq

- Locator: https://arxiv.org/html/2310.08659v4
- Evidence: The paper jointly quantizes weights and computes a low-rank initialization to reduce the quantization discrepancy, and evaluates NLU, question answering, summarization, and NLG tasks on A100 GPUs.
- Limit: The reported gains and initialization cost depend on model, bit width, rank, and GPU; they are not current edge measurements.
- Nearby manuscript contexts: 1

### liu2024dora

- Locator: https://proceedings.mlr.press/v235/liu24bn.html
- Evidence: The ICML 2024 proceedings page defines magnitude-direction decomposition and reports downstream comparisons, including no additional inference overhead in the described construction.
- Limit: The reported gain range is not a matched edge benchmark.
- Nearby manuscript contexts: 1

### lorasurvey2024

- Locator: https://doi.org/10.1007/s11704-024-40663-9
- Evidence: The published Frontiers of Computer Science survey reviews LoRA and its variants as a methods taxonomy and literature summary.
- Limit: A survey is background, not a matched experiment or evidence for the current workload's rankings.
- Nearby manuscript contexts: 1

### luccioni2025

- Locator: https://doi.org/10.1145/3715275.3732007
- Evidence: The FAccT 2025 paper frames efficiency gains as potentially producing rebound effects and argues that direct operational accounting is only part of the broader environmental question.
- Limit: It does not measure the study's 0.20 rebound factor; that value remains an explicit scenario assumption.
- Nearby manuscript contexts: 3

### meng2024pissa

- Locator: https://arxiv.org/html/2404.02948v4
- Evidence: The paper initializes low-rank factors from principal singular components and reports comparisons across 11 models, 5 NLG tasks, and 8 NLU tasks, including faster convergence and higher scores in the reported setups.
- Limit: The reported task results do not establish the current product workload or the one-time initialization cost on the study host.
- Nearby manuscript contexts: 1

### ministral32026

- Locator: https://ollama.com/library/ministral-3
- Evidence: The Ollama page lists Ministral 3B and 8B distribution entries at roughly 3.0 GB and 6.0 GB, 256K context, text and image input, and edge deployment positioning.
- Limit: Exact 2.95 GB and 6.02 GB values are local build metadata; this vendor page is not a head-to-head quality or energy benchmark.
- Nearby manuscript contexts: 1

### mistral2024

- Locator: https://mistral.ai/news/ministraux/
- Evidence: The Mistral release is a vendor announcement and provides the cited Ministral benchmark chart and model positioning.
- Limit: The vendor protocol does not establish a matched independent head-to-head comparison with the Llama figures; retain only vendor-reported results.
- Nearby manuscript contexts: 2

### museglimmer2026

- Locator: https://huggingface.co/meta-models/Muse-Glimmer-30B
- Evidence: The Meta model card identifies Muse Glimmer as an Apache-2.0 dense model of about 30B parameters with a vision encoder, and reports under-20-GB quantized variants together with benchmark context.
- Limit: Exact 18.16 GB and local bandwidth-ladder values are local distribution metadata; model-card benchmark claims are not independent measurements.
- Nearby manuscript contexts: 1

### ni2025benchmarks

- Locator: https://arxiv.org/html/2508.15361v1
- Evidence: The survey reviews benchmark contamination and motivates dynamic and Google-Proof benchmark designs.
- Limit: It does not provide a measured correction for the current MMLU values or validate this study's frontier range; avoid turning the survey into a quantified adjustment.
- Nearby manuscript contexts: 1

### nvidia_rtx4090

- Locator: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
- Evidence: NVIDIA's official page records 450 W total graphics power, an 850 W recommended system power, and notes how idle and gaming power are measured.
- Limit: These specifications and product measurements are not runtime board power for this workload; the study's 420 W value remains an explicit assumption unless measured.
- Nearby manuscript contexts: 1

### nvidiasmi2026

- Locator: https://docs.nvidia.com/deploy/nvidia-smi/
- Evidence: NVIDIA's documentation defines instantaneous power draw as the last measured whole-board draw and documents the power.draw.instant and power.draw.average query fields.
- Limit: Driver counter cadence limits precision, and board telemetry is not whole-system wall-meter energy.
- Nearby manuscript contexts: 1

### ollamaapi2026

- Locator: https://github.com/ollama/ollama/blob/main/docs/api.md
- Evidence: Ollama API documentation defines raw=true to bypass formatting, seed for reproducible outputs, and generation timing and token counters.
- Limit: The documentation has moved between URLs and does not validate hardware throughput, power, or quality results.
- Nearby manuscript contexts: 1

### paischer2024eva

- Locator: https://arxiv.org/html/2410.07170v5
- Evidence: The NeurIPS 2025 paper describes incremental SVD over activation minibatches and rank redistribution, and reports experiments over 51 tasks and four domains while also documenting cases where another method wins.
- Limit: Its average ranking is not a guarantee for one task class, and the paper notes constraints such as low-rank settings and a static downstream dataset.
- Nearby manuscript contexts: 1

### peft_library2026

- Locator: https://huggingface.co/docs/peft/v0.20.0/package_reference/lora
- Evidence: The Hugging Face PEFT documentation exposes DoRA, rsLoRA, and initialization choices including PiSSA, OLoRA, EVA, CoRDA, and LoftQ; it warns that mixed adapters add inference overhead.
- Limit: Library support is not method efficacy on this workload, and documentation is version-sensitive.
- Nearby manuscript contexts: 1

### qi2023finetuning

- Locator: https://openreview.net/pdf?id=hTEGyKf0dZ
- Evidence: The ICLR paper reports safety degradation after fine-tuning, including adversarial examples and degradation under benign data in the evaluated setup.
- Limit: The paper's models and probes do not establish the private deployment's outcome.
- Nearby manuscript contexts: 2

### qwen25vl2025

- Locator: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Evidence: The Qwen model card identifies Qwen2.5-VL 7B, its Apache-2.0 license, the 3B, 7B, and 72B family, vision and agent capabilities, and vendor benchmark tables.
- Limit: A vendor model card and benchmark table do not establish the private adapter's base-model pairing or its quality result.
- Nearby manuscript contexts: 1

### qwen35_2026

- Locator: https://huggingface.co/Qwen/Qwen3.5-27B
- Evidence: The official Qwen3.5-27B model card identifies a 27B Apache-2.0 vision-language model with native 262144-token context and supplies the model identity behind the documented training run.
- Limit: The public model card does not validate the community training telemetry; the community guide remains only a single recipe anchor.
- Nearby manuscript contexts: 1

### qwen38_2026

- Locator: https://huggingface.co/Qwen/Qwen3.8-27B
- Evidence: The official Qwen3.8-27B model card identifies an Apache-2.0 27B native vision-language model with 262144-token context.
- Limit: Model-card facts do not establish the separate distribution size, local runtime energy, performance, or quality transfer.
- Nearby manuscript contexts: 1

### qwen38_ollama2026

- Locator: https://ollama.com/library/qwen3.8:27b
- Evidence: The official Ollama distribution page lists a 27.3B Q4_K_M build with an 18 GB size.
- Limit: Distribution metadata does not establish exact local file size after caching, whole-system memory, throughput, energy, or quality transfer.
- Nearby manuscript contexts: 1

### reddi2020mlperf

- Locator: https://doi.org/10.1145/3380702.3380721
- Evidence: The MLPerf Inference paper describes standardized inference scenarios, including single-stream and edge/server divisions.
- Limit: Curated vendor submissions do not establish a commodity unified-memory APU gap; the exact current table or paper passage must be checked before using a numeric comparison.
- Nearby manuscript contexts: 1

### rfc8827

- Locator: https://www.rfc-editor.org/rfc/rfc8827
- Evidence: The RFC specifies WebRTC security architecture and the DTLS-SRTP fingerprint binding used to authenticate the peer transport.
- Limit: It does not prove confidentiality from an authenticated endpoint or a compromised peer.
- Nearby manuscript contexts: 2

### rfc8831

- Locator: https://www.rfc-editor.org/rfc/rfc8831
- Evidence: The RFC defines WebRTC data channels over SCTP and their security relationship to DTLS.
- Limit: A protocol standard does not establish endpoint security, metadata minimization, or application logging.
- Nearby manuscript contexts: 2

### rtx5070spec

- Locator: https://www.nvidia.com/en-us/geforce/graphics-cards/compare/
- Evidence: NVIDIA's comparison page lists the RTX 5070 with 12 GB GDDR7, a 192-bit bus, and 672 GB/s specified memory bandwidth.
- Limit: These are part specifications only, not runtime throughput or whole-system energy measurements.
- Nearby manuscript contexts: 1

### schneider2025

- Locator: https://arxiv.org/html/2502.01671v1
- Evidence: The preprint presents a cradle-to-grave life-cycle assessment of five TPUs, separates operational and embodied emissions, and introduces compute carbon intensity.
- Limit: The retrieved source does not establish the paper's exact 15 to 30 percent parameter; that range remains a cited or modeled input and needs an exact table check.
- Nearby manuscript contexts: 1

### solartech2025

- Locator: https://solartechonline.com/blog/average-household-power-consumption-guide-2025/
- Evidence: The consumer guide discusses desktop and household electricity draw and can serve only as contextual range input.
- Limit: The exact 30 to 150 W range was not located in the current redirected page; it is not a load measurement or peer-reviewed source and should remain a wide sensitivity only or be replaced by meter data.
- Nearby manuscript contexts: 1

### strixhalo2026

- Locator: https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide
- Evidence: The public GitHub guide documents a tested Qwen3.5-27B LoRA recipe on a 128 GB Strix Halo system, including 448 steps and r=128/alpha=256 settings.
- Limit: This is a community single-machine, unrefereed anchor rather than independent replication or formal publication; publish the run manifest, checkpoint digest, logs, and wall-power trace for stronger evidence.
- Nearby manuscript contexts: 2

### techtarget2025

- Locator: https://www.techtarget.com/it-strategy/feature/Can-edge-computing-make-AI-more-sustainable
- Evidence: The trade article says edge can offer potential sustainability benefits and recommends energy per inference as more useful than PUE for edge.
- Limit: This is directional trade press, not quantitative causal evidence; keep it non-load-bearing or replace it with a matched primary measurement.
- Nearby manuscript contexts: 2

### tuning_pricing2026

- Locator: https://pricepertoken.com/fine-tuning
- Evidence: The third-party pricing index supplies a snapshot of fine-tuning and token prices and provider comparisons.
- Limit: It is a dynamic aggregator and the exact 1.5x tuned-endpoint claim was not independently verified in the current read; keep the comparison as a scenario and archive provider prices and terms.
- Nearby manuscript contexts: 1

### ucriverside2023

- Locator: https://doi.org/10.1145/3724499
- Evidence: The Communications of the ACM article discusses direct and indirect water use in AI systems and emphasizes that water estimates depend on location, infrastructure, and accounting boundary.
- Limit: It is a broad explanatory article and does not supply the study's chosen global EWIF point estimate.
- Nearby manuscript contexts: 1

### uptime2025

- Locator: https://intelligence.uptimeinstitute.com/resource/uptime-institute-global-data-center-survey-2025
- Evidence: The official Uptime page identifies the 2025 Global Data Center Survey and states that average PUE changed little for the sixth consecutive year.
- Limit: The exact 1.54 value is behind a gated report in this audit; archive the report or an accessible table before treating that number as independently checked.
- Nearby manuscript contexts: 3

### wang2024slm

- Locator: https://arxiv.org/html/2411.03350v2
- Evidence: The survey discusses small language models in terms of lower latency, cost, customization, and adaptation, and organizes edge and mobile deployment considerations.
- Limit: It is a preprint with a metadata anomaly in the rendered page, and survey evidence does not establish the current workload ratios or a causal product result.
- Nearby manuscript contexts: 2

### williams2009roofline

- Locator: https://doi.org/10.1145/1498765.1498785
- Evidence: The CACM record identifies the Roofline model for reasoning about operational intensity and bandwidth or compute ceilings.
- Limit: The general model premise does not prove that single-stream decode or the current proxy W equals bytes read per token; the manuscript states this proxy limitation.
- Nearby manuscript contexts: 2

### yang2024corda

- Locator: https://arxiv.org/html/2406.05223v3
- Evidence: The paper constructs context-oriented decompositions from activation covariance and reports knowledge-preserved and instruction-previewed modes on math, code, and instruction-following tasks.
- Limit: The reported forgetting mitigation and performance are paper-specific experiments, not evidence about a private deployment.
- Nearby manuscript contexts: 2

### zhang2023adalora

- Locator: https://openreview.net/pdf?id=lq62uWRJjiY
- Evidence: The ICLR paper describes adaptive allocation of a fixed parameter budget across layers during fine-tuning.
- Limit: No current study run verifies its effect on the declared task classes.
- Nearby manuscript contexts: 1

### zhao2024galore

- Locator: https://proceedings.mlr.press/v235/zhao24s.html
- Evidence: The ICML 2024 proceedings page reports gradient low-rank projection, optimizer-state memory reduction, and full-parameter training.
- Limit: Its continued-pretraining results do not constitute an adapter transport result.
- Nearby manuscript contexts: 1

## Reading rule

A citation is adequate only when the source supports the exact nearby
claim, including its population, boundary, date, metric, and uncertainty.
A formal venue or a successful URL does not remove that requirement.
Vendor, trade, community, aggregator, internal, and preprint sources may
be useful, but the manuscript must state what they actually establish.
Targeted source checks are scope notes, not blanket certification of every
nearby citation context; exact sentence support still requires a final
manual pass over the recorded contexts.
