# Literature base - annotated, with the numbers we used

All figures below are public measurements, published estimates, official pricing
pages, or explicit modelling assumptions about the *underlying* technologies
(data-centre cooling, LLM/SLM inference, the electricity grid, WebRTC). They are
the inputs to `models/data.py`; this file is the human-readable mirror.
No AutoYou-internal data is used or searched.

## Data-centre efficiency
- **PUE** - global average **1.54**, stagnant for six years; hyperscalers
  **1.09-1.15** (Google 1.09). The EU EED requires data-centre reporting and is
  preparing a rating scheme/minimum performance standards; this is not modelled
  as an EU-wide PUE cap. - Uptime Institute 2025; European Commission.
- **WUE (on-site cooling water)** - best-in-class hyperscale **0.12 L/kWh**;
  central hyperscale assumption **0.20 L/kWh**; industry withdrawal-WUE benchmark
  **0.84 L/kWh**, with high-water facilities above 1 L/kWh. - AWS 2025; de
  Vries and Gao 2026.
- **Full-stack split (Google, at scale)** - a median Gemini prompt's energy is
  **58% accelerator, 24% CPU/DRAM, 18% idle+overhead** → **42% non-compute**.
  - Elsworth et al. 2025.

## Per-query inference energy / water / carbon
- **Median Gemini text prompt**: **0.24 Wh**, **0.26 mL**, **0.03 gCO₂e**
  (comprehensive); **0.10 Wh** active-only. - Google 2025.
- **Jegham et al. 2025** (H100/H200, batch 8), long prompt (10k in/1.5k out):
  Llama-3.2 1B **0.342 Wh**, 3B **0.573**, Llama-3.1 8B **0.603**;
  GPT-4o **1.788**; Claude-3.7 Sonnet **5.518**; Llama-3.1 405B **20.757**;
  DeepSeek-R1 **33.634**; **o3 39.223** (≈70× the 8B). Short GPT-4o **0.421 Wh**.
  Tiny (<3B) models stay **<2 mL** water and **<0.3 gCO₂e**/query; DeepSeek-R1
  **>150 mL** and **>14 gCO₂e**.
- **Per 100-word prompt** ≈ **519 mL** water incl. off-site generation. - UC
  Riverside (Li et al. 2023).

## Grid carbon intensity (gCO₂e/kWh, 2024-25)
Global **458** (→~400 by 2027); EU **~140**; US **384**; China **560**;
India **708**. - Ember 2026 / IEA.

## Small vs large model capability
- 3B-8B models approach frontier on knowledge/grounded tasks but trail on hard
  reasoning. MMLU: frontier ~0.73-0.88; 8B ~0.63-0.69; hard math ratio ~0.5.
  - SLM survey (arXiv:2411.03350); Mistral *Les Ministraux* 2024.
- **Ministral-8B** (vendor card): MMLU **65.0**, HumanEval **76.8**, GSM8K
  **54.5**, MBPP **70.0**, function-calling **31.6** - beats Llama-3.1-8B
  (HumanEval 67.1, GSM8K 49.3) and Gemma-2-9B on code/math; independent
  Artificial Analysis evals show it trailing on MMLU/MATH (caution vs over-claim).
- **NVIDIA, *SLMs are the Future of Agentic AI*** (Belcak et al., arXiv:2506.02153,
  2025): Phi-2 2.7B ≈30B commonsense/code at ~15× speed; Phi-3-7B ≈70B on code;
  SmolLM2-1.7B matches 14B on tool-calling/instruction; xLAM-2-8B **beats GPT-4o
  & Claude-3.5 on tool calling**; 7B SLM **10-30× cheaper** (latency/energy/FLOPs);
  **40-70% of agentic LLM calls SLM-replaceable** (MetaGPT 60, Open Operator 40,
  Cradle 70). Heterogeneous SLM+LLM systems are the natural design.

## Demand & cost (stakes / TCO)
- **IEA Energy & AI 2025**: global DC electricity demand **>doubles to ~945 TWh
  by 2030** (~Japan); AI-optimised DCs quadruple; US ~half of growth.
- **Cloud API output pricing (2025-26)**: official OpenAI, Anthropic, and Google
  list prices span budget tiers through reasoning tiers; model uses $15/Mtok
  output and $3/Mtok input as a Sonnet/frontier-class central case. Edge
  per-query cost is electricity (~$0.00004) → **~150× cheaper** than a frontier
  query.

## Consumer / edge hardware
- **RTX 4090**: ~**141 tok/s** on Llama-3 8B; **450 W** TDP (575 W peak);
  cheaper per token single-user but **~2.4× less efficient per token** than a
  batched H100. - DatabaseMart / PromptQuorum 2026.
- **Edge AI** energy savings **65-90%** for suitable tasks (Samsung S24 NPU
  >90% vs A100 Colab). - TechTarget 2025.
- **Desktop idle** **30-100 W** (workstation 70-150 W); sleep 3-5 W. - SolarTech.

## Transport & network
- **WebRTC**: STUN direct success **70-85%**; TURN relay needed **10-20%**;
  relay overhead up to **20%**. Transport is **DTLS-mandatory** with SDP
  fingerprint pinning → relays are blind forwarders. This is transport
  confidentiality against relays, not a complete application-layer privacy proof
  between paired endpoints. - Ant Media 2026; RFC 8831/8827.
- **Transmission energy intensity** ~**0.06 kWh/GB** (2015), falling toward
  **<0.01 kWh/GB**; published estimates span 0.0064-136 (boundary-dependent).
  - Aslan et al. 2018.

## Rebound & embodied carbon
- **Jevons rebound** for LLMs **15-25%** of savings reinvested into more usage;
  Google's 33× per-query efficiency gain coincided with exploding total demand.
  - Luccioni et al. 2025.
- **Embodied carbon** is **15-30%** of total data-centre emissions; accelerator
  embodied-carbon payback 3-6 months at high utilisation. - Faiz et al. 2025.

See `models/data.py` for the exact value/range/citation of every
parameter and `results.json` for all derived numbers.

---

# Track B - on-device adaptation (Pass 3, September 2026)

Same rules. Every figure is a public measurement, a published rate, an
architecture fact, or an explicit modelling assumption. The one exception is
this deployment's own adapter evaluations, which are aggregate probe scores read
from artifacts already in the repository and are labelled as primary evidence
wherever they appear.

## The open-weights tier a household can hold

- **Qwen3.8-27B** (2026-08-14) — **27.78B dense** parameters, text/image/video
  input, **Apache-2.0**, **262,144-token** context, ~17 GB as a 4-bit build.
  Reported over Qwen3.6-27B **on an identical architecture**: Terminal-Bench 2.1
  **63.4 → 73.0**, DeepSWE 1.1 **13.3 → 42.2**, OSWorld-Verified
  **63.9 → 84.3**, SWE-MM **25.7 → 38.6**; Artificial Analysis Intelligence
  Index **52** (+14). The mechanism matters as much as the numbers: capability
  from data and post-training, not from parameter growth.
- **Qwen3.5 family** (February 2026) — five dense sizes 0.8B–27B plus 35B-A3B,
  122B-A10B and 397B-A17B sparse MoE, all Apache-2.0 and natively multimodal on
  an early-fusion vision-language architecture.
- **Qwen2.5-VL** — Apache-2.0 at every size *except* the 3B and 72B, which carry
  the Qwen Research License. This licensing detail, not capability, determined
  which base the shipped support adapter uses.

## Consumer hardware that can train

- **AMD Ryzen AI MAX+ 395 "Strix Halo"** (gfx1151) — 16 Zen 5 cores, 40 RDNA 3.5
  CUs, 50+ peak AI TOPS XDNA 2 NPU, and one **128 GB LPDDR5X-8000** pool
  addressed by CPU, GPU and NPU alike. Roughly a quarter of an RTX 4090's memory
  bandwidth and roughly five times its addressable memory. — AMD 2025.
- **Published 27B training recipe on that part** — Linux 6.19 mainline, ROCm
  7.1.0 with 7.13 nightly wheels, PyTorch 2.11.0+rocm7.13, transformers 5.4 /
  trl 0.29.1 / peft 0.18.1, **bitsandbytes 0.50.0.dev0 built from source for
  gfx1151**. Qwen3.5-27B LoRA **r=128, α=256**, bf16, `paged_adamw_8bit`,
  sequence 8192, gradient accumulation 4, **448 steps at ~11 min/step**,
  **~80 GB peak** training memory of the 128 GB pool. This is the study's single
  validation point for the memory and throughput models.
- **Measured inference on the same part** — 7.5 tok/s (27B dense Q8, 8k ctx),
  19 tok/s (27B with multi-token-prediction speculative decoding), **50 tok/s**
  (35B-A3B sparse MoE, Q4). The 6.7× dense-vs-MoE gap on identical hardware is a
  shape the Pass-1 single-device energy model could not express.
- The same source reports **Vulkan +22%** over ROCm on Q4 quantized inference
  and **ROCm +117%** over Vulkan on bf16 — a reminder that backend choice is a
  first-order variable on this class, not a detail.

## The PEFT method roster

Sixteen methods are modelled in `models/peft.py`. Quality figures below
are **the methods' own authors' reported ranges on their own benchmarks**, used
to bound sensitivity, never as predictions for any AutoYou task.

**The baseline.** **LoRA** (Hu et al. 2021) freezes W and learns a rank-r update
that merges back at export — which is why an adapter costs nothing at inference
and can ship to a phone.

**Quantization.** **QLoRA** (Dettmers et al. 2023) is three separable ideas: NF4,
double quantization of the block constants (**~0.37 bits/parameter** saved), and
paged optimizer states that survive backward-pass spikes. Reported to preserve
16-bit finetuning quality; costs ~35% wall clock. **LoftQ** (Li et al. 2023)
initialises the adapter to absorb quantization error rather than to zero.

**Reparameterisation.** **DoRA** (Liu et al. 2024, ICML) splits the update into
magnitude and direction; reported gains are largest at *low* rank — where the
edge operates. Available over 4-bit bases (QDoRA) since PEFT 0.10. **VeRA**
(Kopiczko et al. 2023) freezes one shared random pair and trains only scaling
vectors: **~50× fewer** trainable parameters. **LoRA-XS** (Bałazy et al. 2024)
trains an r×r core between frozen SVD-derived projections. **AdaLoRA** (Zhang et
al. 2023) reallocates rank during training, at a higher *peak* memory than its
final adapter implies.

**Initialisation.** **PiSSA** (Meng et al. 2024) starts in the principal
singular subspace of W. **OLoRA** (Büyükakyüz 2024) uses a cheaper QR
construction. **EVA** (Paischer et al. 2024) runs a few batches of the *actual*
dataset and allocates rank where that dataset needs it. **CorDA** (Yang et al.
2024) offers a knowledge-preserved mode that builds the update in directions the
base does not use for general knowledge — a principled answer to the forgetting
objection at initialisation time.

**Free.** **rsLoRA** (Kalajdzievski 2023) scales by **α/√r** instead of α/r; one
keyword, and the reason so much practice concluded that rank does not matter —
under the original scaling, raising r shrinks the effective update. **LoRA+**
(Hayou et al. 2024) gives A and B different learning rates: ~1–2 points and up
to 2× faster convergence. **NEFTune** (Jain et al. 2023) adds embedding noise,
with the largest reported gains on *conversational* quality specifically.

**Wrong shape.** **GaLore** (Zhao et al. 2024) has a better ceiling than LoRA on
continued pretraining but updates all weights, producing a new model rather than
a mergeable adapter. Disqualifying for a system whose distribution unit is a
small personal delta — included to make the criterion explicit.

**Library support.** HF PEFT v0.18 (June 2026) exposes `use_dora`, `use_rslora`
and `init_lora_weights` ∈ {gaussian, eva, olora, pissa, pissa_niter_N, corda,
loftq, orthogonal}, and adds Sparse-LoRA and LoRA-XS.

## Forgetting and safety

- **Biderman et al. 2024 (TMLR)** — LoRA learns *less* in-domain than full
  fine-tuning and correspondingly *forgets less* out-of-domain, outperforming
  weight decay and dropout as a regulariser. Full fine-tuning learns
  perturbations of **10–100× higher rank**. The constraint that limits the gain
  is the constraint that limits the damage.
- **Qi et al. 2023 (ICLR 2024)** — fine-tuning removes safety alignment: GPT-3.5
  Turbo's guardrails fell to **ten adversarial examples for under $0.20**, and
  benign, ordinary datasets degraded safety measurably too. The foundational
  objection to shipping a trainer to end users, and this study does not dispute
  it — it delimits it (see C5).

## Speech adaptation

- **LoRP-TTS** (arXiv:2502.07562, 2025) — low-rank speaker adaptation of a TTS
  model from a small amount of *unlabelled spontaneous* speech. The exact shape
  of the Voice Training agent's problem.
- **Characteristic-specific partial fine-tuning** (arXiv:2501.14273, 2025) —
  adapting *disjoint parameter subsets* for speaker identity and for emotion
  rather than the whole model for both. Independently validates the module mask
  the Voice Training agent already uses, and points at a separable emotion
  adapter as the next step.

## Hosted fine-tuning pricing (accessed 2026-09-05)

| Platform | Training | Tuned inference |
|---|---|---|
| Together AI (LoRA) | $0.48/Mtok | 1.0× base |
| OpenAI GPT-4.1 nano | $1.50/Mtok | **1.5× base** |
| Google Vertex (Gemini) | per training token | **1.5× base** from Gemini 3 |
| OpenAI GPT-4o | $25.00/Mtok | **1.5× base** |
| OpenAI o4-mini | $100/training-hour | 1.5× base |

The surcharge, not the training rate, is what compounds: a personalised endpoint
is billed above the base model for **every subsequent query, permanently**.

## What is deliberately absent

**No public per-query energy measurement covers the 2026 frontier or the 2026
open-weights tier.** Energy, water and carbon results therefore remain pinned to
the measured 2025 corpus (Jegham et al.; Elsworth et al.; Caravaca et al.) in
both tracks. Prices, architectures and adapter evaluations are current; energy is
not, and every claim that depends on energy says so.

**No public per-run energy figure exists for hosted fine-tuning**, so the
local-versus-hosted comparison is made in dollars and in privacy topology only,
never in kWh.
