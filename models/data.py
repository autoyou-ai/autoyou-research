# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
data.py - Single source of truth for all empirical parameters used in the
"Decentralized Edge Inference" study.

Every constant carries a CITATION key that resolves to an entry in
``CITATIONS`` (mirrored in ../paper/references.bib and ../literature/literature.md).
The Python eval harness AND the LaTeX paper draw their numbers from here so that
they can never silently diverge. Ranges (lo, hi) are used for the Monte-Carlo
sensitivity analysis in models.py; the scalar is the central/point estimate.

All energy is in watt-hours (Wh) per query unless noted; water in millilitres
(mL) per query; carbon in grams CO2-equivalent (gCO2e) per query; grid carbon
intensity in gCO2e/kWh; water intensity in L/kWh; network intensity in kWh/GB.

NOTE ON SCOPE: none of these figures are AutoYou-specific. They come from public
citations or explicit modelling assumptions about the *underlying* technologies
(WebRTC/STUN/TURN, data-centre cooling, LLM/SLM inference, consumer GPUs, the
electricity grid). AutoYou is used only as the *reference architecture* that
composes these technologies; no AutoYou-internal numbers are claimed here.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Param:
    """A single empirical parameter with provenance."""
    value: float          # central / point estimate
    lo: float             # low end of credible range (for sensitivity)
    hi: float             # high end of credible range
    unit: str
    cite: str             # citation key into CITATIONS
    note: str = ""

    def as_range(self) -> Tuple[float, float]:
        return (self.lo, self.hi)


# --------------------------------------------------------------------------- #
#  1. Data-centre infrastructure efficiency
# --------------------------------------------------------------------------- #

PUE_GLOBAL = Param(1.54, 1.54, 1.56, "ratio", "uptime2025",
                   "Uptime Institute global avg PUE, stagnant ~1.54 for 6 yrs")
PUE_HYPERSCALE = Param(1.10, 1.09, 1.15, "ratio", "uptime2025",
                       "Google fleet 1.09; hyperscalers 1.10-1.15")
EU_DC_POLICY_MARKER = Param(1.0, 1.0, 1.0, "flag", "eu_eed2026",
                            "EU EED data-centre reporting/rating framework; "
                            "not an EU-wide PUE cap")

# Water Usage Effectiveness - on-site cooling water per kWh of IT load.
WUE_INDUSTRY = Param(0.84, 0.56, 1.85, "L/kWh", "aws_water2025",
                     "Industry withdrawal-WUE benchmark; high-water facilities "
                     "can be materially higher")
WUE_HYPERSCALE = Param(0.20, 0.12, 0.30, "L/kWh", "aws_water2025",
                       "Best-in-class 0.12; hyperscaler operating range used for sensitivity")

# Google's at-scale comprehensive measurement splits the per-prompt footprint:
#   58% active accelerator, 24% host CPU+DRAM, 18% idle/redundancy+overhead.
GOOGLE_COMPUTE_FRACTION = Param(0.58, 0.58, 0.58, "frac", "elsworth2025",
                                "Share of per-prompt energy in TPU/GPU")
GOOGLE_NONCOMPUTE_FRACTION = Param(0.42, 0.42, 0.42, "frac", "elsworth2025",
                                   "CPU/DRAM(24%) + idle/overhead(18%)")


# --------------------------------------------------------------------------- #
#  2. Grid carbon intensity (gCO2e/kWh) and electricity water intensity
# --------------------------------------------------------------------------- #

CI_GLOBAL = Param(458.0, 400.0, 473.0, "gCO2e/kWh", "ember2026",
                  "Global avg 2025 (458); IEA 2027 forecast ~400")
CI_US = Param(384.0, 384.0, 384.0, "gCO2e/kWh", "ember2026", "United States 2024/25")
CI_CHINA = Param(560.0, 560.0, 560.0, "gCO2e/kWh", "ember2026", "China 2024/25")
CI_INDIA = Param(708.0, 708.0, 708.0, "gCO2e/kWh", "ember2026", "India 2024")
CI_EU = Param(140.0, 140.0, 175.0, "gCO2e/kWh", "ember2026", "EU, steepest decarbonisation")

# Electricity Water Intensity Factor - off-site water embedded in power
# generation (thermoelectric + hydro evaporation), independent of DC cooling.
# Google's 0.26 mL is on-site consumption, not an off-site residual.
EWIF = Param(1.10, 0.50, 2.50, "L/kWh", "assumption",
             "Off-site grid-water scenario; not measured or inferred from Google")


# --------------------------------------------------------------------------- #
#  3. LLM / SLM inference energy per query  (Jegham et al. 2025, "How Hungry is AI?")
#     LONG prompt = 10k input / 1.5k output tokens unless suffixed _short.
#     These are CLOUD estimates from API timings and inferred hardware, not
#     direct power measurements. Long and short prompts are different workloads.
# --------------------------------------------------------------------------- #

E_LLAMA32_1B = Param(0.342, 0.342, 0.342, "Wh", "jegham2025", "Llama-3.2 1B, long")
E_LLAMA32_3B = Param(0.573, 0.573, 0.573, "Wh", "jegham2025", "Llama-3.2 3B, long")
E_LLAMA31_8B = Param(0.603, 0.603, 0.603, "Wh", "jegham2025", "Llama-3.1 8B, long")
E_GPT4O_MINI = Param(2.106, 2.106, 2.106, "Wh", "jegham2025", "GPT-4o mini, long")
E_GPT4O = Param(1.788, 1.788, 1.788, "Wh", "jegham2025", "GPT-4o, long")
E_CLAUDE37_SONNET = Param(5.518, 5.518, 5.518, "Wh", "jegham2025", "Claude-3.7 Sonnet, long")
E_LLAMA31_70B = Param(11.628, 11.628, 11.628, "Wh", "jegham2025", "Llama-3.1 70B, long")
E_LLAMA31_405B = Param(20.757, 20.757, 20.757, "Wh", "jegham2025", "Llama-3.1 405B, long")
E_GPT45 = Param(30.495, 30.495, 30.495, "Wh", "jegham2025", "GPT-4.5, long")
E_DEEPSEEK_R1 = Param(33.634, 33.634, 33.634, "Wh", "jegham2025", "DeepSeek-R1, long (reasoning)")
E_O3 = Param(39.223, 39.223, 39.223, "Wh", "jegham2025", "o3, long (reasoning) - 70x nano")

# Short prompt (100 in / 300 out)
E_GPT41_NANO_SHORT = Param(0.103, 0.103, 0.103, "Wh", "jegham2025", "GPT-4.1 nano, short")
E_GPT4O_SHORT = Param(0.421, 0.421, 0.421, "Wh", "jegham2025", "GPT-4o, short (~Google search 0.30)")
E_DEEPSEEK_R1_SHORT = Param(23.815, 23.815, 23.815, "Wh", "jegham2025", "DeepSeek-R1, short")

# Google at-scale median Gemini text prompt (comprehensive vs active-only)
E_GEMINI_MEDIAN = Param(0.24, 0.24, 0.24, "Wh", "elsworth2025", "Comprehensive median Gemini prompt")
E_GEMINI_ACTIVE_ONLY = Param(0.10, 0.10, 0.10, "Wh", "elsworth2025", "Active TPU/GPU only")
W_GEMINI_MEDIAN = Param(0.26, 0.26, 0.26, "mL", "elsworth2025", "Water per median Gemini prompt")
C_GEMINI_MEDIAN = Param(0.03, 0.03, 0.03, "gCO2e", "elsworth2025", "Carbon per median Gemini prompt")

# Jegham qualitative bounds for tiny vs reasoning models
W_SLM_TINY_MAX = Param(2.0, 2.0, 2.0, "mL", "jegham2025", "<3B models: <2 mL water/query")
C_SLM_TINY_MAX = Param(0.3, 0.3, 0.3, "gCO2e", "jegham2025", "<3B models: <0.3 gCO2/query")
W_DEEPSEEK_R1 = Param(150.0, 150.0, 150.0, "mL", "jegham2025", "DeepSeek-R1: >150 mL/query")
C_DEEPSEEK_R1 = Param(14.0, 14.0, 14.0, "gCO2e", "jegham2025", "DeepSeek-R1: >14 gCO2/query")


# --------------------------------------------------------------------------- #
#  4. Consumer / edge hardware
# --------------------------------------------------------------------------- #

RTX4090_TPS_8B = Param(141.0, 130.0, 180.0, "tok/s", "databasemart2026",
                       "RTX 4090, Llama-3 8B single-stream throughput")
RTX4090_GEN_POWER = Param(420.0, 350.0, 450.0, "W", "nvidia_rtx4090",
                          "RTX 4090 board power during generation. Bounded above by "
                          "the 450 W total graphics power in NVIDIA's own "
                          "specification. An earlier version put the high end at "
                          "575 W on the authority of a third-party blog whose title "
                          "asserted that figure for this part; 575 W is the RTX "
                          "5090's board power. Superseded in practice by the "
                          "measured 222 W in measured.py, which this study reports "
                          "rather than substitutes -- see the form-check table")
PC_IDLE_W = Param(60.0, 30.0, 150.0, "W", "solartech2025",
                  "Always-on desktop idle draw (30-100 typical; 70-150 workstation)")
PC_SLEEP_W = Param(4.0, 3.0, 5.0, "W", "solartech2025", "Sleep-mode draw")
PUE_EDGE = Param(1.02, 1.00, 1.05, "ratio", "assumption",
                 "Household 'PUE': ambient cooling, no facility overhead")
WUE_EDGE = Param(0.0, 0.0, 0.0, "L/kWh", "assumption",
                 "Household direct cooling water for compute ~ 0 (air-cooled)")
EDGE_SAVINGS_RANGE = Param(0.72, 0.65, 0.90, "frac", "techtarget2025",
                           "Reported edge-vs-cloud energy savings (65-80%; S24 NPU >90%)")


# --------------------------------------------------------------------------- #
#  5. Network transport (WebRTC / STUN / TURN) and transmission energy
# --------------------------------------------------------------------------- #

P_DIRECT = Param(0.80, 0.70, 0.85, "frac", "antmedia2026",
                 "Fraction of WebRTC sessions that succeed P2P via STUN")
P_RELAY = Param(0.15, 0.10, 0.20, "frac", "antmedia2026",
                "Fraction needing TURN relay (symmetric NAT / firewalls)")
TURN_OVERHEAD = Param(0.20, 0.10, 0.20, "frac", "antmedia2026",
                      "Extra path cost when relayed vs direct (up to 20%)")
NET_INTENSITY = Param(0.015, 0.0064, 0.06, "kWh/GB", "aslan2018",
                      "Fixed-line transmission energy intensity (declining; "
                      "0.06 in 2015, <0.01 modern est.)")


# --------------------------------------------------------------------------- #
#  6. Capability benchmarks (accuracy fraction, 0..1)
#     Used to TEST the '~70% as capable' hypothesis honestly, per task class.
# --------------------------------------------------------------------------- #

# MMLU (broad knowledge) - note: GPT-4o is reported anywhere 0.734 (5-shot,
# contamination-controlled survey) to ~0.86-0.887 (vendor). We carry the range.
MMLU_FRONTIER = Param(0.86, 0.734, 0.887, "acc", "ni2025benchmarks", "GPT-4o-class frontier")
MMLU_8B = Param(0.66, 0.63, 0.685, "acc", "ni2025benchmarks", "Llama-3.1-8B / Ministral-8B class")
MMLU_3B = Param(0.57, 0.54, 0.60, "acc", "mistral2024", "Llama-3.2-3B / Ministral-3B class")

# Hard math / multi-step reasoning (MATH-500 / competition) - SLMs fall off a cliff.
MATH_FRONTIER = Param(0.78, 0.70, 0.90, "acc", "ni2025benchmarks", "Frontier reasoning on hard math")
MATH_8B = Param(0.40, 0.30, 0.50, "acc", "ni2025benchmarks", "8B class on hard math")

# Task-class capability RATIO (SLM-8B / frontier).
#
# PROVENANCE, stated plainly: these five numbers are AUTHORIAL ESTIMATES. They
# are not read off any table in any cited work. An earlier version of this file
# tagged them `wang2024slm`, which was wrong: that survey characterises small
# models in general and publishes no per-task-class SLM-to-frontier accuracy
# ratio. Attaching a citation to an estimate made the estimate look measured.
#
# What they ARE informed by, none of which determines them:
#   - the benchmark spreads in MMLU_* and MATH_* above (hard_reason's 0.51 is
#     the one ratio with an arithmetic basis: MATH_8B / MATH_FRONTIER = 0.513),
#   - the qualitative finding in wang2024slm and mistral2024 that 3-8B models
#     approach frontier quality on grounded and structured work and fall away
#     on multi-step reasoning,
#   - the direction, not the value, of belcak2025's agentic replaceability.
#
# They are the most load-bearing inputs in the study and the least grounded.
# Every one is varied across its stated range in the Monte-Carlo, and ALPHA is
# swept in the tornado, which is the only defence available for a number of
# this kind. A reader who rejects these five rejects f_s and everything
# downstream of it, and that is the correct response to an estimate.
CAP_RATIO_EXTRACTION = Param(0.95, 0.90, 0.98, "ratio", "authors_estimate",
                             "Structured extraction / classification / routing")
CAP_RATIO_RAG_QA = Param(0.90, 0.82, 0.95, "ratio", "authors_estimate",
                         "RAG-grounded factual QA (retrieval carries knowledge)")
CAP_RATIO_SUMMARY = Param(0.88, 0.80, 0.93, "ratio", "authors_estimate",
                          "Summarisation / rewriting")
CAP_RATIO_SIMPLE_CODE = Param(0.75, 0.65, 0.85, "ratio", "authors_estimate",
                              "Boilerplate / simple code")
CAP_RATIO_HARD_REASON = Param(0.51, 0.40, 0.62, "ratio", "authors_estimate",
                              "Hard multi-step math/logic - SLM is NOT "
                              "sufficient. The one ratio with an arithmetic "
                              "basis: MATH_8B / MATH_FRONTIER = 0.513")


# --------------------------------------------------------------------------- #
#  7. Rebound (Jevons) and embodied carbon
# --------------------------------------------------------------------------- #

REBOUND_FACTOR = Param(0.20, 0.15, 0.25, "frac", "luccioni2025",
                       "Indirect rebound: savings reinvested into more usage (15-25%)")
EMBODIED_DC_SHARE = Param(0.22, 0.15, 0.30, "frac", "schneider2025",
                          "Embodied carbon as share of total DC emissions")
# Embodied carbon of a consumer GPU+host, amortised per query.
EMBODIED_EDGE_KG = Param(350.0, 200.0, 600.0, "kgCO2e", "schneider2025",
                         "Cradle-to-gate embodied CO2 of a gaming PC + dGPU")
EDGE_DEVICE_LIFETIME_QUERIES = Param(5.0e6, 1.0e6, 2.0e7, "queries", "assumption",
                                     "Inference queries over device service life "
                                     "(only the MARGINAL share is attributable)")
EMBODIED_ATTRIB_FRACTION = Param(0.10, 0.02, 0.30, "frac", "assumption",
                                 "Share of edge device embodied carbon attributable "
                                 "to AI inference (rest is pre-existing general use)")


# --------------------------------------------------------------------------- #
#  8. Workload mix - fraction of everyday assistant queries an 8B SLM can
#     satisfy at acceptable quality (the crux variable f_s).
# --------------------------------------------------------------------------- #

# Composition of a representative personal-assistant workload (sums to 1.0).
#
# PROVENANCE: none. These five weights have no source. They are not sampled
# from a query corpus, not taken from a published telemetry study, and not
# derived from anything else in this file. We did not have a corpus to sample
# and we did not obtain one; this is the gap, stated rather than papered over.
#
# It matters more than any other unsourced quantity here, because f_s is
# nothing but 1 - hard_reason once the ratios are fixed: f_s = 0.82 IS
# "hard_reason = 0.18". Every energy, water, carbon and cost saving in the
# paper, and the entire fleet-scale extrapolation, is proportional to a weight
# we chose. WORKLOAD_MIX_SOURCE records this in the results artifact so the
# claim travels with the number.
#
# The only external check available is directional: belcak2025 measures 40-70%
# of calls in three real agentic systems as SLM-replaceable, and our mix
# implies 82%, i.e. ABOVE that range. See AGENTIC_REPLACEABLE below.
WORKLOAD_MIX = {
    "extraction":   0.18,   # parse/route/classify
    "rag_qa":       0.30,   # grounded factual Q&A (web/RAG)
    "summary":      0.22,   # summarise/rewrite
    "simple_code":  0.12,   # boilerplate
    "hard_reason":  0.18,   # genuinely hard reasoning/math -> route to cloud
}

# f_s as a first-class sensitivity parameter. The tornado previously swept
# ALPHA only, which tests where the acceptance bar sits but never tests the
# weights that decide what clearing it is worth. The range deliberately spans
# NVIDIA's MEASURED replaceability band at the low end, so a reader who trusts
# their corpus over our assertion can read the answer off the figure.
WORKLOAD_SUFFICIENT_FRACTION = Param(
    0.82, 0.40, 0.90, "frac", "authors_estimate",
    "f_s implied by WORKLOAD_MIX at alpha=0.70. Low end is the floor of "
    "belcak2025's measured 40-70% agentic-replaceability range; the central "
    "value is this study's own unsourced mix and sits ABOVE that range")

WORKLOAD_MIX_SOURCE = (
    "UNSOURCED AUTHORIAL ASSUMPTION. No query corpus was sampled and no "
    "published workload study underlies these weights. f_s = 1 - "
    "WORKLOAD_MIX['hard_reason'] exactly, so this single weight sets the "
    "headline coverage figure and every saving derived from it. Treat every "
    "f_s-dependent result as conditional on a workload composition the study "
    "asserts rather than measures."
)

# Quality acceptance threshold: SLM is 'sufficient' if its capability ratio to
# the frontier model on that task class is >= ALPHA.
ALPHA_SUFFICIENCY = Param(0.70, 0.60, 0.80, "ratio", "hypothesis",
                          "The '~70%-as-capable' acceptance threshold under test")

# Empirical fraction of LLM calls in REAL agentic systems that SLMs can replace,
# measured by NVIDIA Research: MetaGPT ~60%, Open Operator ~40%, Cradle ~70%.
# We use this to calibrate (and honestly bound) our workload-derived f_s.
AGENTIC_REPLACEABLE = Param(0.55, 0.40, 0.70, "frac", "belcak2025",
                            "Estimated replaceability in three case studies; not a measured household workload")


# --------------------------------------------------------------------------- #
#  9. Economics (total cost of ownership, per query)
# --------------------------------------------------------------------------- #

PRICE_ELECTRICITY = Param(0.16, 0.10, 0.30, "$/kWh", "assumption",
                          "Residential electricity price")
EDGE_DEVICE_COST = Param(2000.0, 1200.0, 3500.0, "$", "assumption",
                         "Gaming/workstation PC + consumer dGPU")
API_PRICE_OUT_FRONTIER = Param(15.0, 10.0, 40.0, "$/Mtok", "frontier2026",
                               "Cloud output-token price, mid/heavy tier")
API_PRICE_OUT_EFFICIENT = Param(1.0, 0.30, 3.0, "$/Mtok", "frontier2026",
                                "Cloud output-token price, budget tier")
API_PRICE_IN = Param(3.0, 2.0, 10.0, "$/Mtok", "frontier2026",
                     "Cloud input-token price")
TYPICAL_OUT_TOKENS = Param(300.0, 150.0, 600.0, "tok", "assumption",
                           "Assistant reply length")
TYPICAL_IN_TOKENS = Param(500.0, 100.0, 2000.0, "tok", "assumption",
                          "Prompt length")


# --------------------------------------------------------------------------- #
#  10. Fleet-scale extrapolation (IEA Energy & AI 2025)
# --------------------------------------------------------------------------- #

DC_DEMAND_2030_TWH = Param(945.0, 800.0, 1050.0, "TWh", "iea2025",
                           "Global data-centre electricity demand by 2030 (~Japan)")
AI_SHARE_OF_DC = Param(0.45, 0.35, 0.55, "frac", "iea2025",
                       "AI-optimised share of data-centre demand growth")
INFERENCE_SHARE_OF_AI = Param(0.65, 0.50, 0.90, "frac", "jegham2025",
                              "Inference share of AI compute energy")


# --------------------------------------------------------------------------- #
#  11. July-2026 frontier refresh (Pass 2)
#
#  The frontier moved between the first pass (2026-06) and this refresh
#  (2026-07-24): OpenAI shipped the GPT-5.6 family (Sol/Terra/Luna, GA
#  2026-07-09), Anthropic shipped Claude Sonnet 5 (2026-06-30) alongside the
#  Fable 5 frontier tier, Google shipped Gemini 3.6 Flash (2026-07-21), and two
#  open-weights sparse-MoE flagships arrived: Moonshot Kimi K3 (2026-07-16) and
#  Zhipu GLM-5.2 (2026-06-13). No peer-reviewed per-query energy measurement of
#  these specific models is public as of 2026-07-24, so the measured 2025 corpus
#  (jegham2025, elsworth2025, caravaca2025) remains the energy evidence base;
#  the 2026 roster below carries PRICING and ARCHITECTURE facts only.
#
#  Prices are $/Mtok list prices (input, output) from vendor pages / OpenRouter,
#  accessed 2026-07-24.
# --------------------------------------------------------------------------- #

FRONTIER_2026 = {
    # model_id: (vendor, release, $in/Mtok, $out/Mtok, note)
    "gpt-5.6-sol":        ("OpenAI",      "2026-07-09",  5.00, 30.00,
                           "flagship reasoning/agentic tier"),
    "gpt-5.6-terra":      ("OpenAI",      "2026-07-09",  2.50, 15.00,
                           "balanced everyday tier"),
    "gpt-5.6-luna":       ("OpenAI",      "2026-07-09",  1.00,  6.00,
                           "fastest/cheapest tier"),
    "claude-fable-5":     ("Anthropic",   "2026-07",    10.00, 50.00,
                           "frontier tier above Opus"),
    "claude-sonnet-5":    ("Anthropic",   "2026-06-30",  3.00, 15.00,
                           "1M-token context; intro price $2/$10 to 2026-08-31"),
    "gemini-3.6-flash":   ("Google",      "2026-07-21",  1.50,  7.50,
                           "17% fewer output tokens vs 3.5 Flash"),
    "gemini-3.5-flash-lite": ("Google",   "2026-07-21",  0.30,  2.50,
                           "budget tier"),
    "kimi-k3":            ("Moonshot AI", "2026-07-16",  3.00, 15.00,
                           "sparse MoE 2.8T total, 16 of 896 experts active "
                           "(~1.8%); $0.30 cached input; open weights announced"),
    "glm-5.2":            ("Zhipu AI",    "2026-06-13",  0.85,  2.50,
                           "sparse MoE 744B total / ~40B active; MIT weights; "
                           "OpenRouter price (first-party ~$1.40 in)"),
}

# Router placement (Pass 2). A capability router must classify every query
# before it can route it, so the classifier's own footprint is charged to EVERY
# query regardless of destination. Where that classifier RUNS is therefore a
# first-order design variable, not an implementation detail.
ROUTER_CLASSIFY_OUT_TOKENS = Param(10.0, 5.0, 30.0, "tok", "assumption",
                                   "Output tokens for a routing decision (a class label)")
ROUTER_PAYLOAD_KB = Param(4.0, 1.0, 16.0, "kB", "assumption",
                          "Query bytes shipped to a remote router, round trip")

API_PRICE_OUT_FRONTIER_2026 = Param(
    15.0, 6.0, 50.0, "$/Mtok", "frontier2026",
    "July-2026 frontier output list prices: Luna 6 .. Fable 5 50; the modal "
    "mid-tier price is 15 (GPT-5.6 Terra, Claude Sonnet 5, Kimi K3)")
API_PRICE_IN_FRONTIER_2026 = Param(
    3.0, 1.0, 10.0, "$/Mtok", "frontier2026",
    "July-2026 frontier input list prices: Luna 1 .. Fable 5 10; Sonnet 5 and "
    "Kimi K3 both 3")


# --------------------------------------------------------------------------- #
#  12. Pass 3 (September 2026) - ON-DEVICE ADAPTATION
#
#  Pass 1 and Pass 2 treated a small model's capability as fixed by its
#  parameter count. Pass 3 tests whether parameter-efficient fine-tuning moves
#  it, on hardware the user already owns. Two things changed in the world
#  between Pass 2 (2026-07-24) and Pass 3 (2026-09-05) that make the question
#  answerable rather than speculative:
#
#    (a) A 27B-class model that is dense, natively multimodal, Apache-2.0 and
#        ~17 GB in a 4-bit build shipped on 2026-08-14 (Qwen3.8-27B). The
#        open-weights tier a household can actually hold is no longer 8B.
#    (b) Consumer unified-memory APUs put 128 GB in front of the accelerator,
#        and a reproducible recipe for multi-day 27B LoRA training on one of
#        them is public. Adaptation of a 27B model stopped being a data-centre
#        activity.
#
#  Model architecture and device facts live in devices.py; PEFT method facts
#  and cost models live in peft.py. Only the study-level parameters live here.
# --------------------------------------------------------------------------- #

# The AutoYou training corpora that actually exist, in tokens. Used to size
# adaptation runs in adaptation.py. Derived from the recorded sample counts and
# sequence budgets, not from a tokenizer pass, so they are order-of-magnitude
# figures and are varied in the sensitivity analysis.
CORPUS_PERSONA_TOKENS = Param(2.0e6, 5.0e5, 8.0e6, "tokens", "assumption",
                              "One person's own conversation export, as prepared "
                              "by the Fine Tuning agent for a persona adapter")
CORPUS_SUPPORT_TOKENS = Param(7.2e6, 4.0e6, 1.2e7, "tokens", "autoyou_support2026",
                              "1,750 samples x 4,096 max_seq_len x 3 epochs, "
                              "the shipped AutoYou Support adapter's run")
CORPUS_CAPABILITY_TOKENS = Param(1.468e7, 5.0e6, 4.0e7, "tokens", "strixhalo2026",
                                 "448 steps x 4 accum x 8,192 seq - the published "
                                 "27B multi-day run")

# How often a personal adapter is rebuilt. Style drifts, the corpus grows, the
# product changes. Charged against the savings in adaptation.py.
RETRAIN_PER_YEAR = Param(2.0, 1.0, 12.0, "1/yr", "assumption",
                         "Personal adapter rebuild cadence")

# Everyday query volume for one person using an assistant they are not metered
# on. The whole point of the product is that this number is allowed to be large,
# so treating it as small is the conservative choice for break-even arithmetic.
QUERIES_PER_USER_PER_DAY = Param(50.0, 10.0, 300.0, "queries/day", "assumption",
                                 "Per-user daily assistant queries, unmetered")

# The September-2026 open-weights tier a household can hold. Kept separate from
# FRONTIER_2026 (which is cloud API pricing) because these are weights you run,
# not tokens you buy.
OPEN_WEIGHTS_2026 = {
    # key: (vendor, release, params_B, active_B, licence, note)
    "qwen3.8-27b":     ("Alibaba", "2026-08-14", 27.78, 27.78, "Apache-2.0",
                        "dense, natively multimodal, 262K ctx, ~17 GB at 4-bit; "
                        "AA Intelligence Index 52"),
    "qwen3.5-27b":     ("Alibaba", "2026-02", 27.0, 27.0, "Apache-2.0",
                        "hybrid attention with GatedDeltaNet layers"),
    "qwen3.5-35b-a3b": ("Alibaba", "2026-02", 35.0, 3.0, "Apache-2.0",
                        "sparse MoE; ~50 tok/s on a consumer APU at 4-bit"),
    "qwen3.5-122b-a10b": ("Alibaba", "2026-02", 122.0, 10.0, "Apache-2.0",
                        "sparse MoE; fits a 128 GB unified pool at 4-bit"),
    "qwen2.5-vl-7b":   ("Alibaba", "2025", 8.29, 8.29, "Apache-2.0",
                        "the shipped AutoYou Support base"),
    "ministral-3-8b":  ("Mistral", "2024-10", 8.02, 8.02, "Apache-2.0",
                        "the Fine Tuning agent's default base"),
}

# Reported benchmark movement for Qwen3.8-27B over Qwen3.6-27B on an IDENTICAL
# architecture. Carried because it is the cleanest public evidence that the
# open-weights tier is still improving from data and post-training rather than
# from parameter growth - the same mechanism this pass tests at household scale.
QWEN38_GAINS = {
    "terminal-bench-2.1":  (63.4, 73.0),
    "deepswe-1.1":         (13.3, 42.2),
    "osworld-verified":    (63.9, 84.3),
    "swe-mm":              (25.7, 38.6),
    "aa-intelligence-index": (38.0, 52.0),
}

# The acceptance threshold at which capability uplift starts buying COVERAGE
# rather than only quality. Derived in adaptation.alpha_sweep(); recorded here
# so the paper and the figures cite one number.
ALPHA_COVERAGE_KNEE = Param(0.80, 0.75, 0.85, "ratio", "hypothesis",
                            "Acceptance threshold above which adaptation "
                            "increases f_s rather than only mean quality")


# --------------------------------------------------------------------------- #
#  13. Pass 4 (September 2026) - PRIMARY MEASUREMENT
#
#  Passes 1-3 took every input to the edge-energy model from somewhere else.
#  Pass 4 measures what this study's own hardware actually does, using
#  measure/bench.py, and replaces two of the three borrowed terms.
#
#  The live figures are read from the measurement artifacts by models/measured.py
#  and degrade to NOT MEASURED when those artifacts are absent. The Params below
#  are the values the paper quotes, pinned here so the LaTeX and the harness
#  cannot silently diverge.
# --------------------------------------------------------------------------- #

# Generation throughput, single stream, Q4_K_M, num_ctx 4096, median of 5 runs
# on an AMD Radeon 8060S (Ryzen AI MAX+ 395, 64 GB unified), Ollama 0.33.2.
MEASURED_TPS_APU_4B = Param(73.1, 72.1, 73.5, "tok/s", "measured2026",
                            "Ministral-3 3.8B, measured")
MEASURED_TPS_APU_9B = Param(35.6, 33.9, 35.6, "tok/s", "measured2026",
                            "Ministral-3 8.9B, measured")
MEASURED_TPS_APU_27B = Param(15.6, 15.6, 16.9, "tok/s", "measured2026",
                             "Qwen3.8 27.3B, measured; the widest run-to-run "
                             "spread of the three, on a machine holding a 17 GB "
                             "model in a 64 GB pool")

# The prefill overhead the study assumed at 0.15. Measured across the three
# models above, at 638/94-token prompts and 192 generated tokens.
MEASURED_PREFILL_OVERHEAD = Param(
    0.018, 0.016, 0.019, "ratio", "measured2026",
    "beta, the share by which prefill inflates generation time for one query. "
    "Roughly an eighth of the 0.15 this study assumed, which means the earlier "
    "passes OVERSTATED edge energy. Rises with the prompt-to-generation ratio, "
    "so a long-context retrieval workload sits higher than this.")

# Consumer memory bandwidth, vendor specifications, used to test whether
# single-stream decode is bandwidth bound.
BANDWIDTH_APU = Param(256.0, 256.0, 256.0, "GB/s", "amd_strix2025",
                      "LPDDR5X-8000 on a 256-bit bus")
BANDWIDTH_RTX4090 = Param(1008.0, 1008.0, 1008.0, "GB/s", "databasemart2026",
                          "GDDR6X on a 384-bit bus")


# --------------------------------------------------------------------------- #
#  Citations (key -> human-readable + URL). Mirrored in references.bib.
# --------------------------------------------------------------------------- #

CITATIONS = {
    "uptime2025": ("Uptime Institute Global Data Center Survey 2025 - average PUE 1.54.",
                   "https://journal.uptimeinstitute.com/large-data-centers-are-mostly-more-efficient-analysis-confirms/"),
    "eu_eed2026": ("European Commission, Energy performance of data centres - mandatory reporting, "
                   "rating work, and possible minimum performance standards.",
                   "https://energy.ec.europa.eu/topics/energy-efficiency/energy-efficiency-targets-directive-and-rules/energy-efficiency-directive/energy-performance-data-centres_en"),
    "aws_water2025": ("Amazon Web Services, Water Stewardship. Global data-centre WUE 0.15 L/kWh "
                      "withdrawn per kWh of IT load in 2024, improving to 0.12 L/kWh in 2025; "
                      "industry average withdrawal-WUE benchmark 0.84 L/kWh. An earlier version of "
                      "this entry attributed the 0.12 figure to 2024, which is a year early.",
                      "https://sustainability.aboutamazon.com/natural-resources/water"),
    "devriesgao2025": ("A. de Vries-Gao, 'The carbon and water footprints of data centers and what "
                       "this could mean for artificial intelligence,' Patterns, 2025, "
                       "doi:10.1016/j.patter.2025.101430. CORRECTION: an earlier version of this entry "
                       "gave a different title, a different article ID, and split this single "
                       "hyphenated-surname author into two people by inventing a co-author.",
                       "https://www.cell.com/patterns/fulltext/S2666-3899(25)00278-8"),
    "introl2025": ("Introl, Water Usage Efficiency for AI Data Center Cooling, 2025.",
                   "https://introl.com/blog/water-usage-efficiency-wue-ai-data-center-cooling-guide-2025"),
    "elsworth2025": ("C. Elsworth et al. (Google), 'Measuring the Environmental Impact of Delivering AI at Google Scale,' arXiv:2508.15734, 2025.",
                     "https://arxiv.org/abs/2508.15734"),
    "ember2026": ("Ember, Global Electricity Review 2026; IEA Electricity 2025.",
                  "https://ember-energy.org/latest-insights/global-electricity-review-2026/"),
    "jegham2025": ("N. Jegham, M. Abdelatti, L. Elmoubarki, A. Hendawi, 'How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference,' arXiv:2505.09598, 2025.",
                   "https://arxiv.org/abs/2505.09598"),
    "databasemart2026": ("DatabaseMart, RTX 4090 vLLM Benchmark for sub-8B LLMs, 2026.",
                         "https://www.databasemart.com/blog/vllm-gpu-benchmark-rtx4090"),
    "nvidia_rtx4090": ("NVIDIA Corporation, GeForce RTX 4090 specifications: 450 W total graphics "
                       "power, 850 W recommended system supply. REPLACES a third-party blog whose own "
                       "title stated 575 W for this part - that is the RTX 5090's board power.",
                       "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/"),
    "solartech2025": ("SolarTech, How Much Electricity Does a Computer Use? 2025 Guide.",
                      "https://solartechonline.com/blog/how-much-electricity-does-computer-use/"),
    "techtarget2025": ("TechTarget, Can Edge Computing Make AI More Sustainable? 2025.",
                       "https://www.techtarget.com/sustainability/feature/Can-edge-computing-make-AI-more-sustainable"),
    "antmedia2026": ("Ant Media, WebRTC Peer-to-Peer Communication: How P2P Works in 2026 (STUN ~70-85%, TURN 10-20%).",
                     "https://antmedia.io/how-to-create-webrtc-peer-to-peer-communication/"),
    "aslan2018": ("J. Aslan et al., 'Electricity Intensity of Internet Data Transmission: Untangling the Estimates,' J. Industrial Ecology, 2018.",
                  "https://onlinelibrary.wiley.com/doi/10.1111/jiec.12630"),
    "ni2025benchmarks": ("S. Ni, G. Chen, S. Li, X. Chen, S. Li, B. Wang, Q. Wang, X. Wang, Y. Zhang, "
                         "L. Fan, C. Li, R. Xu, L. Sun, M. Yang, 'A Survey on Large Language Model "
                         "Benchmarks,' arXiv:2508.15361, 2025. Surveys 283 benchmarks and documents "
                         "contamination-inflated scores, which is why frontier MMLU is carried here as "
                         "a range rather than a vendor point estimate. CORRECTION: an earlier version "
                         "of this entry had its author field set literally to {Anonymous}.",
                         "https://arxiv.org/abs/2508.15361"),
    "mistral2024": ("Mistral AI, 'Un Ministral, des Ministraux' (Ministral 3B/8B), Oct 2024; independent eval via Artificial Analysis.",
                    "https://mistral.ai/news/ministraux"),
    "wang2024slm": ("F. Wang, Z. Zhang, X. Zhang, Z. Wu, T. Mo, Q. Lu, W. Wang, R. Li, J. Xu, X. Tang, "
                    "Q. He, Y. Ma, M. Huang, S. Wang, 'A Comprehensive Survey of Small Language Models "
                    "in the Era of Large Language Models,' arXiv:2411.03350, submitted November 2024. "
                    "CORRECTION: an earlier version misattributed this to Belcak et al. and dated it 2025. "
                    "This survey publishes NO per-task-class SLM-to-frontier accuracy ratio; the "
                    "CAP_RATIO_* constants that once cited it are authorial estimates and now say so.",
                    "https://arxiv.org/abs/2411.03350"),
    "luccioni2025": ("A. S. Luccioni et al., 'From Efficiency Gains to Rebound Effects: Jevons Paradox in AI,' arXiv:2501.16548 / ACM FAccT 2025.",
                     "https://arxiv.org/abs/2501.16548"),
    "schneider2025": ("I. Schneider, H. Xu, S. Benecke, D. Patterson, K. Huang, P. Ranganathan, "
                      "C. Elsworth, 'Life-Cycle Emissions of AI Hardware: A Cradle-to-Grave Approach "
                      "and Generational Trends,' arXiv:2502.01671, 2025. CORRECTION: an earlier version "
                      "carried this correct arXiv ID and title under an invented author list "
                      "('Faiz, Ahmad and others'). Ahmad Faiz is an author of LLMCarbon, a different paper.",
                      "https://arxiv.org/abs/2502.01671"),
    "belcak2025": ("P. Belcak, G. Heinrich, S. Diao, Y. Fu, X. Dong, S. Muralidharan, Y. C. Lin, P. Molchanov (NVIDIA), 'Small Language Models are the Future of Agentic AI,' arXiv:2506.02153, 2025.",
                   "https://arxiv.org/abs/2506.02153"),
    "iea2025": ("International Energy Agency, 'Energy and AI' (World Energy Outlook special report), 2025 - DC demand ~945 TWh by 2030.",
                "https://www.iea.org/reports/energy-and-ai"),
    "ucriverside2023": ("P. Li, J. Yang, M. A. Islam, S. Ren, 'Making AI Less Thirsty,' arXiv:2304.03271, 2023 (~519 mL per 100-word prompt).",
                        "https://arxiv.org/abs/2304.03271"),
    "frontier2026": ("July-2026 frontier releases and public list pricing: OpenAI GPT-5.6 Sol/Terra/Luna (GA 2026-07-09); "
                     "Google Gemini 3.6 Flash and 3.5 Flash-Lite (2026-07-21); Anthropic Claude Sonnet 5 (2026-06-30) and Fable 5; "
                     "Moonshot AI Kimi K3 (2026-07-16); Zhipu AI GLM-5.2 (2026-06-13). Vendor announcements, OpenRouter listings, "
                     "and the Kimi-K3-vs-GLM-5.2 comparison (Data Science in Your Pocket). Accessed 2026-07-24.",
                     "https://openai.com/index/previewing-gpt-5-6-sol/; "
                     "https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/; "
                     "https://www.anthropic.com/news/claude-sonnet-5; "
                     "https://medium.com/data-science-in-your-pocket/kimi-k3-vs-glm-5-2-3f82e1e39b02"),
    "caravaca2025": ("F. Caravaca, A. Cuevas, R. Cuevas, 'From Prompts to Power: Measuring the Energy Footprint of LLM Inference,' "
                     "arXiv:2511.05597, 2025 (32,500+ GPU inference energy measurements and a predictive per-query model).",
                     "https://arxiv.org/abs/2511.05597"),
    "autoyou_logs2026": ("Local AutoYou main-server runtime logs, April-July 2026, parsed in aggregate (provider-selection and "
                         "model-configuration event counts only; no payloads or personal content). Reproducible "
                         "against any local deployment's log directory.",
                         ""),
    "measured2026": ("Primary measurement, this work. Run 2026-09-05 on an "
                     "AMD Radeon 8060S (Ryzen AI MAX+ 395 'Strix Halo', 64 GB unified memory) under Ollama "
                     "0.33.2, Windows. Protocol: temperature 0, fixed seed, num_ctx 4096, 192 generated "
                     "tokens, 5 repetitions, warm-up discarded, per-repetition prompt nonce to defeat KV "
                     "prefix caching, median reported with full spread retained. Raw per-run records are "
                     "retained in full. NOT measured: board power, which this platform exposes only "
                     "through a kernel-mode helper the harness declines to install; energy figures derived "
                     "from these timings therefore still carry a modelled P and say so. Not yet publicly "
                     "released; available from the authors on request.",
                     ""),
    "assumption": ("Author modelling assumption - varied across its stated range in the Monte-Carlo sensitivity analysis.",
                   ""),
    "hypothesis": ("The hypothesis parameter under test in this paper.",
                   ""),
    "authors_estimate": ("AUTHORIAL ESTIMATE, not a cited measurement. Informed by the benchmark spreads and "
                         "surveys in this file but read off none of them; no published source states this value. "
                         "Varied across its stated range in the Monte-Carlo. Distinguished from 'assumption' "
                         "only to mark the quantities a reader is most likely to mistake for measured, namely "
                         "the per-task-class capability ratios that set f_s.",
                         ""),

    # ----------------------------------------------------------------- #
    #  Pass 3 - models, hardware, and the PEFT method roster
    # ----------------------------------------------------------------- #
    "qwen38_2026": ("Qwen team (Alibaba), Qwen3.8-27B model card and release, 2026-08-14: 27.78B dense parameters, "
                    "text/image/video input, Apache-2.0, 262,144-token context, ~17 GB as a 4-bit build. Reported "
                    "gains over Qwen3.6-27B on an identical architecture: Terminal-Bench 2.1 63.4->73.0, DeepSWE 1.1 "
                    "13.3->42.2, OSWorld-Verified 63.9->84.3, SWE-MM 25.7->38.6; Artificial Analysis Intelligence "
                    "Index 52. Accessed 2026-09-05.",
                    "https://www.yottalabs.ai/post/qwen-3-8-27b-specs-hardware-requirements-how-to-run-2026"),
    "qwen35_2026": ("Qwen team (Alibaba), Qwen3.5 family, February 2026: five dense sizes 0.8B-27B plus 35B-A3B, "
                    "122B-A10B and 397B-A17B sparse-MoE variants, all Apache-2.0 and natively multimodal on an "
                    "early-fusion vision-language architecture. Accessed 2026-09-05.",
                    "https://insiderllm.com/guides/qwen-models-guide/"),
    "qwen25vl2025": ("Qwen team (Alibaba), Qwen2.5-VL model family, 2025. Apache-2.0 for all sizes except the 3B and "
                     "72B, which carry the Qwen Research License - the licensing fact that determined AutoYou's "
                     "shipped support base.",
                     "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"),
    "strixhalo2026": ("Home-enthusiast guide to fine-tuning 27B+ LLMs on AMD Strix Halo (gfx1151, Ryzen AI MAX+ 395): "
                      "Linux 6.19 mainline, ROCm 7.1.0 with 7.13 nightly wheels, PyTorch 2.11.0+rocm7.13, "
                      "transformers 5.4 / trl 0.29.1 / peft 0.18.1, bitsandbytes 0.50.0.dev0 built from source. "
                      "Reported: Qwen3.5-27B LoRA r=128 alpha=256, bf16, paged_adamw_8bit, seq 8192, grad-accum 4, "
                      "448 steps at ~11 min/step, ~80 GB peak training memory of a 128 GB unified pool; measured "
                      "inference 7.5 tok/s (27B dense Q8), 19 tok/s (27B MTP speculative), 50 tok/s (35B-A3B MoE Q4). "
                      "Accessed 2026-09-05.",
                      "https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide"),
    "amd_strix2025": ("AMD, 'Ryzen AI MAX+ 395: a leap forward in generative AI performance with consumer PC' - "
                      "16 Zen 5 cores, 40 RDNA 3.5 CUs, 50+ peak AI TOPS XDNA 2 NPU, unified 128 GB LPDDR5X pool "
                      "addressed by CPU, GPU and NPU.",
                      "https://www.amd.com/en/developer/resources/technical-articles/2025/amd-ryzen-ai-max-395--a-leap-forward-in-generative-ai-performanc.html"),

    "hu2021lora": ("E. J. Hu et al., 'LoRA: Low-Rank Adaptation of Large Language Models,' arXiv:2106.09685, 2021.",
                   "https://arxiv.org/abs/2106.09685"),
    "dettmers2023qlora": ("T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, 'QLoRA: Efficient Finetuning of "
                          "Quantized LLMs,' NeurIPS 2023 / arXiv:2305.14314. NF4, double quantization "
                          "(~0.37 bits/param saved) and paged optimizers; 65B finetuning on one 48 GB GPU while "
                          "preserving 16-bit task performance.",
                          "https://arxiv.org/abs/2305.14314"),
    "liu2024dora": ("S.-Y. Liu et al., 'DoRA: Weight-Decomposed Low-Rank Adaptation,' ICML 2024 / arXiv:2402.09353. "
                    "Available in HF PEFT as LoraConfig(use_dora=True), including over 4-bit bases (QDoRA) since "
                    "PEFT 0.10.",
                    "https://arxiv.org/abs/2402.09353"),
    "kalajdzievski2023rslora": ("D. Kalajdzievski, 'A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA,' "
                                "arXiv:2312.03732, 2023. Scaling by alpha/sqrt(r) rather than alpha/r, so higher "
                                "ranks continue to help. LoraConfig(use_rslora=True).",
                                "https://arxiv.org/abs/2312.03732"),
    "meng2024pissa": ("F. Meng, Z. Wang, M. Zhang, 'PiSSA: Principal Singular Values and Singular Vectors Adaptation "
                      "of Large Language Models,' arXiv:2404.02948, 2024. init_lora_weights='pissa'.",
                      "https://arxiv.org/abs/2404.02948"),
    "li2023loftq": ("Y. Li et al., 'LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models,' "
                    "arXiv:2310.08659, 2023. init_lora_weights='loftq'.",
                    "https://arxiv.org/abs/2310.08659"),
    "buyukakyuz2024olora": ("K. Buyukakyuz, 'OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models,' "
                            "arXiv:2406.01775, 2024. init_lora_weights='olora'.",
                            "https://arxiv.org/abs/2406.01775"),
    "paischer2024eva": ("F. Paischer et al., 'One Initialization to Rule them All: Fine-tuning via Explained Variance "
                        "Adaptation,' arXiv:2410.07170, 2024. Data-driven rank allocation from an incremental SVD of "
                        "activations on the target dataset. init_lora_weights='eva'.",
                        "https://arxiv.org/abs/2410.07170"),
    "yang2024corda": ("Y. Yang et al., 'CorDA: Context-Oriented Decomposition Adaptation of Large Language Models,' "
                      "arXiv:2406.05223, 2024. Knowledge-preserved and instruction-previewed modes. "
                      "init_lora_weights='corda'.",
                      "https://arxiv.org/abs/2406.05223"),
    "hayou2024loraplus": ("S. Hayou, N. Ghosh, B. Yu, 'LoRA+: Efficient Low Rank Adaptation of Large Models,' "
                          "arXiv:2402.12354, 2024. Asymmetric learning rates for the A and B factors.",
                          "https://arxiv.org/abs/2402.12354"),
    "zhang2023adalora": ("Q. Zhang et al., 'AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning,' "
                         "ICLR 2023 / arXiv:2303.10512.",
                         "https://arxiv.org/abs/2303.10512"),
    "kopiczko2023vera": ("D. J. Kopiczko, T. Blankevoort, Y. M. Asano, 'VeRA: Vector-based Random Matrix Adaptation,' "
                         "arXiv:2310.11454, 2023. Frozen shared random projections with per-layer scaling vectors; "
                         "roughly 50x fewer trainable parameters than LoRA.",
                         "https://arxiv.org/abs/2310.11454"),
    "balazy2024loraxs": ("K. Balazy et al., 'LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters,' "
                         "arXiv:2405.17604, 2024.",
                         "https://arxiv.org/abs/2405.17604"),
    "zhao2024galore": ("J. Zhao et al., 'GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection,' "
                       "ICML 2024 / arXiv:2403.03507.",
                       "https://arxiv.org/abs/2403.03507"),
    "jain2023neftune": ("N. Jain et al., 'NEFTune: Noisy Embeddings Improve Instruction Finetuning,' "
                        "arXiv:2310.05914, 2023.",
                        "https://arxiv.org/abs/2310.05914"),
    "biderman2024lora": ("D. Biderman et al., 'LoRA Learns Less and Forgets Less,' TMLR 2024 / arXiv:2405.09673. "
                         "LoRA underperforms full fine-tuning in-domain but better preserves out-of-domain "
                         "performance; full fine-tuning learns perturbations of 10-100x higher rank.",
                         "https://arxiv.org/abs/2405.09673"),
    "qi2023finetuning": ("X. Qi, Y. Zeng, T. Xie, P.-Y. Chen, R. Jia, P. Mittal, P. Henderson, 'Fine-tuning Aligned "
                         "Language Models Compromises Safety, Even When Users Do Not Intend To!,' ICLR 2024 / "
                         "arXiv:2310.03693. Ten adversarial examples for under $0.20 removed GPT-3.5 Turbo's "
                         "guardrails; benign datasets degraded safety more mildly.",
                         "https://arxiv.org/abs/2310.03693"),
    "lorasurvey2024": ("Y. Mao et al., 'A Survey on LoRA of Large Language Models,' arXiv:2407.11046, 2024.",
                       "https://arxiv.org/abs/2407.11046"),
    "peft_library2026": ("Hugging Face PEFT (v0.18, June 2026). LoraConfig exposes use_dora, use_rslora, and "
                         "init_lora_weights in {gaussian, eva, olora, pissa, pissa_niter_N, corda, loftq, "
                         "orthogonal}; the 0.18 line adds Sparse-LoRA and LoRA-XS.",
                         "https://huggingface.co/docs/peft/developer_guides/lora"),
    "lorptts2025": ("'LoRP-TTS: Low-Rank Personalized Text-To-Speech,' arXiv:2502.07562, 2025. Speaker adaptation of "
                    "a TTS model with low-rank updates from a small amount of unlabelled spontaneous speech.",
                    "https://arxiv.org/abs/2502.07562"),
    "ttspartial2025": ("'Efficient Emotion and Speaker Adaptation in LLM-Based TTS via Characteristic-Specific "
                       "Partial Fine-Tuning,' arXiv:2501.14273, 2025. Adapting disjoint parameter subsets for "
                       "speaker identity and for emotion, rather than the whole model for both.",
                       "https://arxiv.org/abs/2501.14273"),
    "peft_uplift": ("Bounded synthesis of reported in-distribution PEFT gains (DoRA, PiSSA, rsLoRA, LoRA+, EVA, "
                    "CorDA, NEFTune) against a LoRA baseline, floored by the pessimistic reading of Biderman et al. "
                    "2024. Used only as a RANGE for sensitivity analysis; never as a point prediction for any "
                    "specific task. See peft.METHODS for the per-method provenance.",
                    ""),
    "autoyou_support2026": ("AutoYou Support adapter evaluation artifacts, read in aggregate: "
                            "five runs over Qwen2.5-VL-3B and -7B bases, 12-14 screen "
                            "probes and 8 code probes each, plus dataset statistics (1,750 train / 72 eval samples "
                            "across 13 families). Scores only; no prompts, answers, screenshots or user content "
                            "cross into the study. Not redistributable and not publicly released.",
                            ""),
}


if __name__ == "__main__":
    # Quick provenance audit: every Param.cite must resolve.
    import sys
    missing = set()
    for name, val in list(globals().items()):
        if isinstance(val, Param) and val.cite not in CITATIONS:
            missing.add((name, val.cite))
    if missing:
        print("MISSING CITATIONS:", missing)
        sys.exit(1)
    n = sum(1 for v in globals().values() if isinstance(v, Param))
    print(f"OK: {n} parameters, all citations resolve ({len(CITATIONS)} sources).")
