# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
peft.py - The parameter-efficient adaptation layer of the study.

The Pass-1/Pass-2 paper treated a small model's capability as a *fixed property
of its parameter count*. That is the assumption this module exists to break.

The crux variable of the whole thesis is f_s: the fraction of a real workload a
small local model can serve at acceptable quality. f_s is computed from
per-task-class capability ratios (SLM quality / frontier quality). If those
ratios can be *moved* - cheaply, locally, on hardware the user already owns -
then f_s is not a constant of nature, it is a design variable, and every
downstream footprint and cost result inherits the improvement.

Parameter-efficient fine-tuning is how you move it. This module models:

  1. WHAT the 2026 method roster actually is (``METHODS``), with provenance.
  2. HOW MUCH MEMORY a given (model, method, device) job needs
     (``adapter_params``, ``training_memory``), calibrated against a published
     27B run on a 128 GB unified-memory APU.
  3. HOW LONG AND HOW MUCH ENERGY that job costs (``training_cost``).
  4. WHICH METHOD to pick for a device and objective (``select_method``) - the
     part that is meant to be lifted into the product agents.

Honesty rules, same as models.py:

  * Quality deltas are LITERATURE-REPORTED RANGES against a LoRA baseline on
    the authors' own benchmarks. They are not AutoYou measurements and they do
    not transfer automatically to another task. They are used to bound
    sensitivity, never to claim a specific product improvement. AutoYou's own
    measured evidence lives in evidence.py and is kept strictly separate.
  * The memory model is validated against one published run and is reported
    with its residual, not presented as exact.
  * Training energy is charged against inference savings in adaptation.py. We
    never present an adapter as free.

Units: memory GB, energy Wh, time s, FLOP raw.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import devices as V


# --------------------------------------------------------------------------- #
#  Numeric constants for quantized storage
# --------------------------------------------------------------------------- #

# NF4 with block size 64, plus double-quantized block constants:
#   4 bits (weight) + 8/64 bits (quantized absmax) + 32/(64*256) bits (the
#   second-level constant) = 4.127 bits/param.  QLoRA Sec. 3.
BYTES_PER_PARAM = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "int8": 1.0,
    "nf4_dq": 4.127 / 8.0,      # 0.5159
    "nf4": 4.5 / 8.0,           # 0.5625, no double quantization
}

# Optimizer state bytes per *trainable* parameter.
OPTIMIZER_BYTES = {
    "adamw_fp32": 8.0,          # fp32 exp_avg + exp_avg_sq
    "adamw_bf16": 4.0,
    "paged_adamw_8bit": 2.0,    # 8-bit m and v, paged to host on spikes
    "adafactor": 1.0,           # factored second moment
    "sgd_momentum": 2.0,
}


# --------------------------------------------------------------------------- #
#  Method roster
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PEFTMethod:
    """One parameter-efficient adaptation method, with provenance and cost.

    ``params_per_module`` returns the trainable-parameter count this method
    adds to a single linear layer of shape (d_in, d_out) at rank r.

    ``quality_delta`` is in *percentage points of task accuracy*, relative to
    plain LoRA at the same rank, as reported by the method's own authors.
    (lo, mid, hi). A negative lo means at least one reported setting was worse.
    """

    key: str
    name: str
    year: int
    cite: str
    family: str                     # reparam | init | scaling | optimizer | quantization | regularizer
    peft_flag: str                  # how it is switched on in HF PEFT, or "-"
    composes_with_4bit: bool
    quality_delta: Tuple[float, float, float]
    memory_multiplier: float        # trainable params vs plain LoRA at same r
    extra_train_time: float         # wall-clock multiplier vs plain LoRA
    note: str = ""

    def params_per_module(self, d_in: int, d_out: int, r: int) -> int:
        raise NotImplementedError                       # replaced below


def _p_lora(d_in: int, d_out: int, r: int) -> int:
    """A is (r, d_in), B is (d_out, r)."""
    return r * (d_in + d_out)


def _p_dora(d_in: int, d_out: int, r: int) -> int:
    """LoRA direction plus one learned magnitude scalar per output column."""
    return r * (d_in + d_out) + d_out


def _p_vera(d_in: int, d_out: int, r: int) -> int:
    """Frozen shared random A/B; only the two scaling vectors are trained."""
    return d_out + r


def _p_lora_xs(d_in: int, d_out: int, r: int) -> int:
    """Frozen SVD-derived A/B with a trained r x r core."""
    return r * r


_PARAM_FN = {
    "vera": _p_vera,
    "lora_xs": _p_lora_xs,
    "dora": _p_dora,
    "qdora": _p_dora,
}


def params_per_module(method_key: str, d_in: int, d_out: int, r: int) -> int:
    return _PARAM_FN.get(method_key, _p_lora)(d_in, d_out, r)


METHODS: Dict[str, PEFTMethod] = {}


def _add(m: PEFTMethod) -> PEFTMethod:
    METHODS[m.key] = m
    return m


# --- The baseline -------------------------------------------------------- #

LORA = _add(PEFTMethod(
    key="lora", name="LoRA", year=2021, cite="hu2021lora",
    family="reparam", peft_flag="LoraConfig(r=..., lora_alpha=...)",
    composes_with_4bit=True, quality_delta=(0.0, 0.0, 0.0),
    memory_multiplier=1.0, extra_train_time=1.0,
    note=(
        "The baseline every other row is measured against. Freezes W and learns "
        "BA with rank r << d. Merges back into W at export, so inference costs "
        "nothing extra - which is why an adapter can ship to a phone."
    ),
))

# --- Quantization: what makes 27B a consumer activity --------------------- #

QLORA = _add(PEFTMethod(
    key="qlora", name="QLoRA (NF4 + double-quant + paged optimizer)",
    year=2023, cite="dettmers2023qlora",
    family="quantization",
    peft_flag="BitsAndBytesConfig(load_in_4bit, nf4, use_double_quant) + LoRA",
    composes_with_4bit=True, quality_delta=(-1.0, -0.2, 0.3),
    memory_multiplier=1.0, extra_train_time=1.35,
    note=(
        "Three separable ideas: NF4 (information-theoretically optimal for "
        "normally distributed weights), double quantization of the block "
        "constants (~0.37 bits/param saved), and paged optimizer states that "
        "survive backward-pass memory spikes. Reported to preserve 16-bit "
        "finetuning quality; the small negative tail here is the honest "
        "acknowledgement that 'preserves' is a benchmark-average claim. Costs "
        "wall-clock because every forward pass dequantizes."
    ),
))

LOFTQ = _add(PEFTMethod(
    key="loftq", name="LoftQ", year=2023, cite="li2023loftq",
    family="init", peft_flag='init_lora_weights="loftq"',
    composes_with_4bit=True, quality_delta=(0.0, 1.0, 3.0),
    memory_multiplier=1.0, extra_train_time=1.02,
    note=(
        "Initializes the adapter to absorb the quantization error rather than "
        "to zero, so training does not start by first repairing the damage "
        "quantization did. Matters most at low rank and at 3 bits or below; at "
        "4-bit with r>=32 the gap to plain QLoRA narrows."
    ),
))

# --- Reparameterizations -------------------------------------------------- #

DORA = _add(PEFTMethod(
    key="dora", name="DoRA (weight-decomposed)", year=2024, cite="liu2024dora",
    family="reparam", peft_flag="LoraConfig(use_dora=True)",
    composes_with_4bit=True, quality_delta=(0.5, 2.0, 3.7),
    memory_multiplier=1.02, extra_train_time=1.20,
    note=(
        "Splits the update into magnitude and direction: direction is ordinary "
        "LoRA, magnitude is a separate learned vector. The reported gain is "
        "largest exactly where the edge cares - low rank - because it recovers "
        "some of what a rank-4 or rank-8 update cannot express. Pays ~20% "
        "wall-clock for the extra normalization."
    ),
))

QDORA = _add(PEFTMethod(
    key="qdora", name="QDoRA (DoRA over 4-bit weights)", year=2024,
    cite="liu2024dora", family="reparam",
    peft_flag="LoraConfig(use_dora=True) over a 4-bit base",
    composes_with_4bit=True, quality_delta=(0.3, 1.8, 3.5),
    memory_multiplier=1.02, extra_train_time=1.55,
    note=(
        "Supported natively in PEFT since 0.10. The combination the edge "
        "actually wants: 4-bit weights so a 27B fits, decomposed updates so a "
        "low rank still expresses enough. The wall-clock multiplier compounds "
        "QLoRA's dequantization with DoRA's normalization."
    ),
))

VERA = _add(PEFTMethod(
    key="vera", name="VeRA (vector-based random adaptation)", year=2023,
    cite="kopiczko2023vera", family="reparam",
    peft_flag="VeraConfig(r=...)", composes_with_4bit=True,
    quality_delta=(-3.0, -1.0, 0.5), memory_multiplier=0.02,
    extra_train_time=0.95,
    note=(
        "Freezes a single pair of random matrices shared across all layers and "
        "trains only two scaling vectors per layer. Trainable parameters fall "
        "by roughly 50x against LoRA. Interesting for AutoYou not because it "
        "trains better but because an adapter becomes small enough to sync "
        "over a datachannel as a routine message rather than a file transfer."
    ),
))

LORA_XS = _add(PEFTMethod(
    key="lora_xs", name="LoRA-XS", year=2024, cite="balazy2024loraxs",
    family="reparam", peft_flag="LoraXSConfig / PEFT 0.18+",
    composes_with_4bit=True, quality_delta=(-2.5, -0.8, 0.5),
    memory_multiplier=0.05, extra_train_time=0.98,
    note=(
        "Frozen SVD-derived projections with a trained r x r core. Same "
        "motivation as VeRA - adapter size, not adapter quality - with a "
        "principled initialization instead of a random one."
    ),
))

# --- Initialization ------------------------------------------------------- #

PISSA = _add(PEFTMethod(
    key="pissa", name="PiSSA (principal singular adaptation)", year=2024,
    cite="meng2024pissa", family="init",
    peft_flag='init_lora_weights="pissa_niter_16"',
    composes_with_4bit=True, quality_delta=(0.5, 2.5, 6.0),
    memory_multiplier=1.0, extra_train_time=1.01,
    note=(
        "Initializes A and B from the principal singular vectors of W instead "
        "of from noise-and-zero, so the adapter starts in the subspace that "
        "already carries most of the layer's energy. Converges faster, which on "
        "a device that trains overnight is the difference between one night and "
        "three. One-time SVD cost at load."
    ),
))

OLORA = _add(PEFTMethod(
    key="olora", name="OLoRA (orthonormal init)", year=2024,
    cite="buyukakyuz2024olora", family="init",
    peft_flag='init_lora_weights="olora"', composes_with_4bit=True,
    quality_delta=(0.0, 1.2, 3.0), memory_multiplier=1.0,
    extra_train_time=1.01,
    note="QR-based orthonormal initialization; cheaper than PiSSA's SVD.",
))

EVA = _add(PEFTMethod(
    key="eva", name="EVA (explained-variance adaptation)", year=2024,
    cite="paischer2024eva", family="init",
    peft_flag='init_lora_weights="eva"', composes_with_4bit=True,
    quality_delta=(0.0, 1.5, 3.5), memory_multiplier=1.0,
    extra_train_time=1.05,
    note=(
        "Data-driven: runs a few batches of the *actual* dataset, does an "
        "incremental SVD of the activations, and allocates rank where that "
        "dataset needs it. Directly relevant to a personal adapter, where the "
        "dataset is one person's own conversation and nobody knows in advance "
        "which layers matter for them."
    ),
))

CORDA = _add(PEFTMethod(
    key="corda", name="CorDA (context-oriented decomposition)", year=2024,
    cite="yang2024corda", family="init",
    peft_flag='init_lora_weights="corda"', composes_with_4bit=True,
    quality_delta=(0.0, 1.5, 4.0), memory_multiplier=1.0,
    extra_train_time=1.06,
    note=(
        "Two modes: knowledge-preserved (build the adapter in the directions "
        "the base model does NOT use for general knowledge) and "
        "instruction-previewed (build it in the directions the target task "
        "does use). The knowledge-preserved mode is a direct, principled answer "
        "to the forgetting counter-claim C4."
    ),
))

# --- Scaling and optimization -------------------------------------------- #

RSLORA = _add(PEFTMethod(
    key="rslora", name="rsLoRA (rank-stabilized scaling)", year=2023,
    cite="kalajdzievski2023rslora", family="scaling",
    peft_flag="LoraConfig(use_rslora=True)", composes_with_4bit=True,
    quality_delta=(0.0, 1.5, 4.0), memory_multiplier=1.0,
    extra_train_time=1.0,
    note=(
        "One-line change with an outsized effect: scale by alpha/sqrt(r) rather "
        "than alpha/r. Under the original scaling, raising the rank shrinks the "
        "effective update and higher ranks stop helping - which is why so much "
        "practice settled on r=8/16 and concluded rank does not matter. It "
        "does; the scaling was hiding it. Free."
    ),
))

LORA_PLUS = _add(PEFTMethod(
    key="lora_plus", name="LoRA+ (asymmetric learning rates)", year=2024,
    cite="hayou2024loraplus", family="optimizer",
    peft_flag="loraplus_lr_ratio (PEFT optimizer helper)",
    composes_with_4bit=True, quality_delta=(0.5, 1.5, 2.0),
    memory_multiplier=1.0, extra_train_time=1.0,
    note=(
        "B needs a larger learning rate than A; using one rate for both is "
        "provably suboptimal in the infinite-width limit. Reported ~1-2 points "
        "and up to 2x faster convergence for a scalar ratio. Also free."
    ),
))

ADALORA = _add(PEFTMethod(
    key="adalora", name="AdaLoRA (adaptive rank allocation)", year=2023,
    cite="zhang2023adalora", family="reparam",
    peft_flag="AdaLoraConfig(target_r=..., init_r=...)",
    composes_with_4bit=True, quality_delta=(-0.5, 1.0, 2.5),
    memory_multiplier=1.5, extra_train_time=1.25,
    note=(
        "Prunes singular values during training to move rank budget to the "
        "layers that earn it. Starts at a higher rank than it ends at, so peak "
        "memory is worse than the final adapter suggests - a poor fit for a "
        "device chosen precisely because it is at its memory ceiling."
    ),
))

GALORE = _add(PEFTMethod(
    key="galore", name="GaLore (gradient low-rank projection)", year=2024,
    cite="zhao2024galore", family="optimizer",
    peft_flag="GaLore optimizer (not a LoraConfig)",
    composes_with_4bit=False, quality_delta=(-1.0, 0.5, 2.0),
    memory_multiplier=6.0, extra_train_time=1.30,
    note=(
        "Not an adapter method: it trains ALL weights but projects the "
        "optimizer state to low rank. Better ceiling than LoRA on continued "
        "pretraining, at the cost of updating full weights - so it produces a "
        "new model, not a mergeable adapter. Wrong shape for a product that "
        "ships a 40 MB personal delta rather than a 54 GB model."
    ),
))

NEFTUNE = _add(PEFTMethod(
    key="neftune", name="NEFTune (embedding noise)", year=2023,
    cite="jain2023neftune", family="regularizer",
    peft_flag="neftune_noise_alpha in TrainingArguments",
    composes_with_4bit=True, quality_delta=(0.0, 1.0, 5.0),
    memory_multiplier=1.0, extra_train_time=1.0,
    note=(
        "Adds uniform noise to embeddings during training. Large reported gains "
        "on conversational quality specifically, which is the AutoYou Fine "
        "Tuning agent's exact use case, and near-zero on factual benchmarks. "
        "Free, and trivially reversible - it changes training only."
    ),
))


# --------------------------------------------------------------------------- #
#  Adapter sizing
# --------------------------------------------------------------------------- #

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")
ATTENTION_ONLY = ("q_proj", "k_proj", "v_proj", "o_proj")


def adapter_params(model: V.ModelSpec, r: int, method_key: str = "lora",
                   targets: Sequence[str] = DEFAULT_TARGETS) -> int:
    """Trainable parameters added by `method_key` at rank r on `model`."""
    names_attn = ("q_proj", "k_proj", "v_proj", "o_proj")
    names_mlp = ("gate_proj", "up_proj", "down_proj")
    per_layer = 0
    attn_dims = dict(zip(names_attn, model.attn_projection_dims()))
    mlp_dims = dict(zip(names_mlp, model.mlp_projection_dims()))
    for name in targets:
        dims = attn_dims.get(name) or mlp_dims.get(name)
        if dims is None:
            continue
        per_layer += params_per_module(method_key, dims[0], dims[1], r)
    return per_layer * model.layers


def adapter_file_mb(n_params: int, dtype: str = "bf16") -> float:
    """On-disk size of the saved adapter - what actually crosses the network."""
    return n_params * BYTES_PER_PARAM[dtype] / 1e6


# --------------------------------------------------------------------------- #
#  Training memory model
# --------------------------------------------------------------------------- #

@dataclass
class MemoryBreakdown:
    weights_gb: float
    adapter_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    logits_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return (self.weights_gb + self.adapter_gb + self.gradients_gb
                + self.optimizer_gb + self.activations_gb + self.logits_gb
                + self.overhead_gb)

    def as_dict(self) -> Dict[str, float]:
        d = {k: round(v, 3) for k, v in self.__dict__.items()}
        d["total_gb"] = round(self.total_gb, 3)
        return d


# Allocator slack. Unified-memory parts fragment badly under multi-day runs;
# the published recipe needs expandable_segments and proactive compaction to
# stay alive at all, which is what this term represents.
ALLOCATOR_OVERHEAD = 0.08


def training_memory(model: V.ModelSpec, r: int, *, method_key: str = "lora",
                    weight_dtype: str = "bf16",
                    optimizer: str = "paged_adamw_8bit",
                    seq_len: int = 8192, batch_size: int = 1,
                    gradient_checkpointing: bool = True,
                    vocab_size: int = 152_000,
                    targets: Sequence[str] = DEFAULT_TARGETS) -> MemoryBreakdown:
    """Peak accelerator memory for one PEFT training job.

    Calibrated against the published Qwen3.5-27B run on Strix Halo: bf16
    weights, r=128, alpha=256, seq 8192, batch 1, paged_adamw_8bit, ~80 GB peak
    reserved. See ``validate_memory_model()``.
    """
    method = METHODS[method_key]
    n_adapter = adapter_params(model, r, method_key, targets)

    weights_gb = model.params_b * 1e9 * BYTES_PER_PARAM[weight_dtype] / 1e9
    adapter_gb = n_adapter * BYTES_PER_PARAM["bf16"] / 1e9
    gradients_gb = n_adapter * BYTES_PER_PARAM["bf16"] / 1e9
    optimizer_gb = n_adapter * OPTIMIZER_BYTES[optimizer] / 1e9

    tokens = batch_size * seq_len
    if gradient_checkpointing:
        # One saved tensor per layer boundary, plus one layer's full recompute.
        act = model.layers * tokens * model.hidden * 2.0
        act += tokens * model.hidden * 34.0 * 2.0
    else:
        act = model.layers * tokens * model.hidden * 34.0 * 2.0
    activations_gb = act / 1e9

    # Logits are the quiet memory hog on large-vocabulary models: the loss is
    # computed in fp32 over the full vocabulary for every position.
    logits_gb = tokens * vocab_size * 4.0 / 1e9

    subtotal = (weights_gb + adapter_gb + gradients_gb + optimizer_gb
                + activations_gb + logits_gb)
    return MemoryBreakdown(
        weights_gb=weights_gb,
        adapter_gb=adapter_gb * method.memory_multiplier,
        gradients_gb=gradients_gb * method.memory_multiplier,
        optimizer_gb=optimizer_gb * method.memory_multiplier,
        activations_gb=activations_gb,
        logits_gb=logits_gb,
        overhead_gb=subtotal * ALLOCATOR_OVERHEAD,
    )


def fits(model: V.ModelSpec, device: V.Device, r: int, **kw) -> Tuple[bool, float, float]:
    """(fits?, needed_gb, ceiling_gb) for this job on this device."""
    need = training_memory(model, r, **kw).total_gb
    ceiling = device.trainable_ceiling_gb()
    return need <= ceiling, need, ceiling


def validate_memory_model() -> Dict[str, float]:
    """Reproduce the one published 27B run and report the residual.

    The recipe reports ~80 GB peak reserved for a bf16 LoRA run on Qwen3.5-27B
    at r=128, seq 8192, batch 1, paged 8-bit AdamW, on a 128 GB Strix Halo.
    """
    reported = 80.0
    m = training_memory(V.QWEN3_5_27B, r=128, weight_dtype="bf16",
                        optimizer="paged_adamw_8bit", seq_len=8192,
                        batch_size=1, gradient_checkpointing=True)
    predicted = m.total_gb
    return {
        "reported_peak_gb": reported,
        "predicted_peak_gb": round(predicted, 2),
        "residual_gb": round(predicted - reported, 2),
        "relative_error": round((predicted - reported) / reported, 4),
        "breakdown": m.as_dict(),
    }


# --------------------------------------------------------------------------- #
#  Training compute, time and energy
# --------------------------------------------------------------------------- #

# Achieved bf16 training throughput, FLOP/s. The APU figure is derived from the
# published run (see calibrate_throughput); the discrete-GPU figure is a
# modelling assumption at ~35% MFU on a 165 TFLOP/s bf16 part.
ACHIEVED_TRAIN_FLOPS = {
    "unified_apu": 8.05e12,
    "discrete_gpu": 55.0e12,
    "laptop_npu": 0.0,          # no training kernels; inference only
}


def flops_per_token(model: V.ModelSpec, gradient_checkpointing: bool = True) -> float:
    """Training FLOPs per token for a frozen-backbone adapter method.

    Forward 2N; backward to activations 2N (still required with frozen weights,
    to reach earlier layers); backward to the adapter is O(N_adapter) and
    rounds away. Gradient checkpointing adds one recomputed forward: +2N.
    Full fine-tuning would additionally pay 2N for weight gradients.
    """
    n = model.active_params_b * 1e9
    return (6.0 if gradient_checkpointing else 4.0) * n


@dataclass
class TrainingCost:
    tokens: float
    flops: float
    seconds: float
    hours: float
    energy_wh: float
    energy_kwh: float
    electricity_usd: float
    carbon_g: float
    device: str
    method: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "tokens": self.tokens, "flops": self.flops,
            "hours": round(self.hours, 2),
            "energy_kwh": round(self.energy_kwh, 3),
            "electricity_usd": round(self.electricity_usd, 2),
            "carbon_kg": round(self.carbon_g / 1000.0, 3),
            "device": self.device, "method": self.method,
        }


def training_cost(model: V.ModelSpec, device: V.Device, tokens: float, *,
                  method_key: str = "lora",
                  gradient_checkpointing: bool = True,
                  price_kwh: float = 0.16,
                  ci_gco2_kwh: float = 458.0,
                  marginal: bool = False) -> TrainingCost:
    """Wall-clock, energy, money and carbon for one adaptation run.

    marginal=True charges only the power above the device's pre-existing idle
    draw, which is the correct accounting when the machine was going to be on
    anyway - the same convention models.py uses for inference.
    """
    achieved = ACHIEVED_TRAIN_FLOPS[device.kind]
    if achieved <= 0:
        raise ValueError(f"{device.name} exposes no training path")
    fpt = flops_per_token(model, gradient_checkpointing)
    total_flops = tokens * fpt * METHODS[method_key].extra_train_time
    seconds = total_flops / achieved
    power = (device.train_power_w - device.idle_power_w) if marginal else device.train_power_w
    energy_wh = max(power, 0.0) * seconds / 3600.0
    return TrainingCost(
        tokens=tokens, flops=total_flops, seconds=seconds,
        hours=seconds / 3600.0, energy_wh=energy_wh,
        energy_kwh=energy_wh / 1000.0,
        electricity_usd=energy_wh / 1000.0 * price_kwh,
        carbon_g=energy_wh / 1000.0 * ci_gco2_kwh,
        device=device.key, method=method_key,
    )


def calibrate_throughput() -> Dict[str, float]:
    """Recover the APU's achieved FLOP/s from the published run, and check it.

    Published: 448 steps at ~11 min/step, batch 1 x grad_accum 4 x seq 8192.
    """
    steps, minutes_per_step = 448, 11.0
    tokens_per_step = 1 * 4 * 8192
    total_tokens = steps * tokens_per_step
    seconds = steps * minutes_per_step * 60.0
    fpt = flops_per_token(V.QWEN3_5_27B, gradient_checkpointing=True)
    achieved = total_tokens * fpt / seconds
    peak_bf16 = 59.0e12          # 40 RDNA 3.5 CU, dual-issue, ~2.9 GHz
    return {
        "total_tokens": total_tokens,
        "wall_clock_hours": round(seconds / 3600.0, 1),
        "achieved_tflops": round(achieved / 1e12, 2),
        "model_flops_utilization": round(achieved / peak_bf16, 3),
        "tokens_per_second": round(total_tokens / seconds, 1),
    }


# --------------------------------------------------------------------------- #
#  Method selection - the decision rule meant for the product agents
# --------------------------------------------------------------------------- #

@dataclass
class Recipe:
    """A concrete, runnable configuration."""

    model: str
    device: str
    base_method: str
    stack: List[str]                # composed free/cheap methods
    rank: int
    alpha: int
    weight_dtype: str
    optimizer: str
    seq_len: int
    batch_size: int
    grad_accum: int
    gradient_checkpointing: bool
    memory_gb: float
    memory_ceiling_gb: float
    adapter_params: int
    adapter_mb: float
    feasible: bool
    rationale: str

    def as_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["memory_gb"] = round(self.memory_gb, 1)
        d["memory_ceiling_gb"] = round(self.memory_ceiling_gb, 1)
        d["adapter_mb"] = round(self.adapter_mb, 1)
        return d


# Free or near-free additions. Every one of these is a keyword argument in the
# libraries the product already imports, costs no extra memory, and is reported
# to help. They are the default stack, not an advanced option.
FREE_STACK = ("rslora", "lora_plus")


def select_method(model: V.ModelSpec, device: V.Device, *,
                  objective: str = "persona",
                  seq_len: int = 4096,
                  max_rank: int = 128) -> Recipe:
    """Choose a runnable recipe for (model, device, objective).

    objective:
      persona   - imitate one person's voice/style from their own messages
                  (Fine Tuning agent). Conversational quality; small dataset.
      domain    - teach a fixed body of product knowledge (Support model).
                  Factual grounding + refusal behaviour; medium dataset.
      capability- lift a task class the base model is weak at. Largest rank the
                  device allows; the f_s-moving case.
      portable  - minimize adapter bytes so it can sync over a datachannel.

    The rule is memory-first: pick the highest-quality configuration that still
    fits, rather than the highest-quality configuration and then discovering at
    step 300 that it does not.
    """
    ceiling = device.trainable_ceiling_gb()
    weights_bf16_gb = model.params_b * 1e9 * BYTES_PER_PARAM["bf16"] / 1e9

    # 1. Weight precision. bf16 if the weights alone leave room to train;
    #    otherwise 4-bit, which is what makes 27B possible on 24 GB at all.
    if weights_bf16_gb < ceiling * 0.62:
        weight_dtype, base = "bf16", "lora"
    else:
        weight_dtype, base = "nf4_dq", "qlora"

    # 2. Base method by objective, then downgrade if it does not fit.
    if objective == "portable":
        base = "vera"
    elif objective == "capability" and weight_dtype == "nf4_dq":
        base = "qdora"
    elif objective == "capability":
        base = "dora"

    stack = list(FREE_STACK)
    if weight_dtype == "nf4_dq":
        stack.append("loftq")
    if objective in {"capability", "domain"}:
        stack.append("pissa" if weight_dtype == "bf16" else "eva")
    if objective == "persona":
        stack.append("neftune")
    if objective == "domain":
        stack.append("corda")       # knowledge-preserved mode, answers C4

    optimizer = "paged_adamw_8bit"

    # 3. Search (rank, sequence length) for the best configuration that fits.
    #    Rank is traded first because it costs quality directly; sequence length
    #    is traded second because a shorter window still trains, it just sees
    #    less context per sample. Only if neither is enough do we report the job
    #    as infeasible rather than quietly emitting a recipe that will OOM.
    ranks = [r for r in (256, 128, 64, 32, 16, 8) if r <= max_rank] or [8]
    seq_ladder = [s for s in (seq_len, seq_len // 2, seq_len // 4, 1024)
                  if s >= 512]
    seq_ladder = sorted(set(seq_ladder), reverse=True)

    rank, chosen_seq, need, feasible = ranks[-1], seq_ladder[-1], 0.0, False
    for s in seq_ladder:
        for candidate in ranks:
            ok, candidate_need, _ = fits(
                model, device, candidate, method_key=base,
                weight_dtype=weight_dtype, optimizer=optimizer,
                seq_len=s, batch_size=1, gradient_checkpointing=True)
            if ok:
                rank, chosen_seq, need, feasible = candidate, s, candidate_need, True
                break
        if feasible:
            break

    seq_len = chosen_seq
    if not feasible:
        need = training_memory(model, rank, method_key=base,
                               weight_dtype=weight_dtype, optimizer=optimizer,
                               seq_len=seq_len, batch_size=1,
                               gradient_checkpointing=True).total_gb
    n_adapter = adapter_params(model, rank, base)

    # 4. rsLoRA changes what alpha means. Under alpha/sqrt(r) the useful ratio
    #    is around 2x rank; under alpha/r the convention is alpha = 2r as well,
    #    so the same number serves both and the flag decides the scaling.
    alpha = rank * 2

    reasons = [
        f"{weight_dtype} weights ({weights_bf16_gb:.0f} GB in bf16 vs "
        f"{ceiling:.0f} GB usable)",
        (f"rank {rank} at seq {seq_len} is the largest that fits "
         f"({need:.0f} GB peak)") if feasible else
        (f"NO configuration fits: the smallest job ({need:.0f} GB) still "
         f"exceeds the {ceiling:.0f} GB ceiling"),
        f"free stack: {', '.join(stack)}",
    ]
    return Recipe(
        model=model.key, device=device.key, base_method=base, stack=stack,
        rank=rank, alpha=alpha, weight_dtype=weight_dtype, optimizer=optimizer,
        seq_len=seq_len, batch_size=1,
        grad_accum=max(1, 32768 // seq_len),
        gradient_checkpointing=True, memory_gb=need, memory_ceiling_gb=ceiling,
        adapter_params=n_adapter,
        adapter_mb=adapter_file_mb(n_adapter),
        feasible=feasible,
        rationale="; ".join(reasons),
    )


def stacked_quality_delta(stack: Iterable[str]) -> Tuple[float, float, float]:
    """Bound the quality effect of composing methods.

    Reported gains do NOT add - they overlap heavily, since several of these
    methods improve the same thing (early convergence) by different means. We
    therefore report the sum as an optimistic ceiling, the single largest
    component as the pessimistic floor, and a discounted sum as the central
    estimate. This is a bound, not a prediction.
    """
    deltas = [METHODS[k].quality_delta for k in stack if k in METHODS]
    if not deltas:
        return (0.0, 0.0, 0.0)
    lo = min(d[0] for d in deltas)
    hi = sum(d[2] for d in deltas)
    mids = sorted((d[1] for d in deltas), reverse=True)
    # Diminishing returns: full credit for the best, halving thereafter.
    mid = sum(m * (0.5 ** i) for i, m in enumerate(mids))
    return (lo, mid, hi)


if __name__ == "__main__":
    print(f"{len(METHODS)} methods in the roster\n")
    print(f"{'method':<24}{'year':>6}{'family':>14}{'4-bit':>7}"
          f"{'quality vs LoRA':>20}{'time':>7}")
    for m in METHODS.values():
        q = f"{m.quality_delta[0]:+.1f}..{m.quality_delta[2]:+.1f} pp"
        print(f"{m.name[:23]:<24}{m.year:>6}{m.family:>14}"
              f"{'yes' if m.composes_with_4bit else 'no':>7}{q:>20}"
              f"{m.extra_train_time:>6.2f}x")

    print("\n--- memory model validation ---")
    v = validate_memory_model()
    print(f"  reported  {v['reported_peak_gb']:.1f} GB")
    print(f"  predicted {v['predicted_peak_gb']:.1f} GB "
          f"({v['relative_error']*100:+.1f}%)")
    for k, val in v["breakdown"].items():
        print(f"      {k:<18} {val:>8.2f}")

    print("\n--- throughput calibration ---")
    for k, val in calibrate_throughput().items():
        print(f"  {k:<28} {val}")

    print("\n--- recipes ---")
    for dev in (V.UNIFIED_APU, V.DISCRETE_GPU):
        for mdl, obj in ((V.QWEN3_8_27B, "capability"),
                         (V.MINISTRAL_8B, "persona"),
                         (V.QWEN25_VL_7B, "domain")):
            r = select_method(mdl, dev, objective=obj, seq_len=8192)
            flag = "" if r.feasible else "  << DOES NOT FIT"
            print(f"  {dev.key:<13} {mdl.key:<15} {obj:<11} "
                  f"-> {r.base_method:<6} r={r.rank:<4} seq={r.seq_len:<5} "
                  f"{r.memory_gb:>5.1f}/{r.memory_ceiling_gb:.0f} GB  "
                  f"adapter {r.adapter_mb:>6.1f} MB{flag}")

    print("\n--- portable adapters (datachannel-syncable) ---")
    for key in ("lora", "dora", "lora_xs", "vera"):
        n = adapter_params(V.MINISTRAL_8B, 32, key)
        print(f"  {METHODS[key].name[:28]:<30} r=32 -> {n:>12,} params, "
              f"{adapter_file_mb(n):>8.1f} MB")
