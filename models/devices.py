# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
devices.py - Edge device classes for inference AND adaptation.

The Pass-1/Pass-2 study modelled exactly one edge device: a desktop with a
discrete consumer GPU (RTX 4090 class). That was the right device to model in
2025, and it is the wrong device to model alone in 2026, because it cannot hold
a 27B-class model's *training* state at all.

Three device classes now matter, and they differ in the one dimension that
decides whether on-device adaptation is possible: **how much memory the
accelerator can address**.

  * DISCRETE_GPU     - fast, narrow. High VRAM bandwidth, small VRAM pool.
                       Trains <=8B comfortably, 27B only in 4-bit and only just.
  * UNIFIED_APU      - slow, wide. Modest bandwidth, enormous addressable pool
                       (Strix Halo: up to 128 GB shared CPU/GPU/NPU; the
                       machine measured in measure/ is the 64 GB
                       SKU of the same part). Trains 27B in
                       bf16 with room left over. This class did not exist as a
                       consumer product when the original study was written.
  * LAPTOP_NPU       - always present, low power, inference-only in practice.

Every number here is either a vendor specification, a published measurement, or
an explicitly-flagged modelling assumption. `cite` resolves into
``data.CITATIONS`` exactly as ``data.Param`` does.

Units: memory GB, bandwidth GB/s, power W, throughput tok/s.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Device:
    """One edge device class, sized for both inference and training."""

    key: str
    name: str
    kind: str                       # discrete_gpu | unified_apu | laptop_npu
    accel_memory_gb: float          # memory the *accelerator* can address
    memory_bandwidth_gb_s: float
    gen_power_w: float              # board/package power while generating
    train_power_w: float            # sustained package power under training load
    idle_power_w: float             # whole-system idle draw
    street_price_usd: float
    cite: str
    note: str = ""

    # Measured generation throughput, tok/s, keyed by a model descriptor.
    # Keys are free-form but stable; see THROUGHPUT_KEYS below.
    throughput: Dict[str, float] = field(default_factory=dict)

    def trainable_ceiling_gb(self, headroom: float = 0.80) -> float:
        """Usable memory for a training job.

        Unified-memory parts cannot hand the whole pool to the GPU: the OS,
        the page cache and the model loader all live in the same pool, and the
        published Strix Halo recipe caps the process at 80% for exactly this
        reason. Discrete GPUs lose a smaller slice to the driver and the
        display, so the same 0.80 default is, if anything, conservative there.
        """
        return self.accel_memory_gb * headroom


# --------------------------------------------------------------------------- #
#  Throughput key vocabulary (documented so results.json stays readable)
# --------------------------------------------------------------------------- #

THROUGHPUT_KEYS = {
    "8b_q4": "8B dense, 4-bit weights, single stream",
    "8b_bf16": "8B dense, bf16 weights, single stream",
    "27b_q8": "27B dense, 8-bit weights, 8k context, single stream",
    "27b_q4": "27B dense, 4-bit weights, single stream",
    "27b_mtp_spec": "27B dense with multi-token-prediction speculative decoding",
    "35b_a3b_q4": "35B-total / 3B-active sparse MoE, 4-bit weights",
}


# --------------------------------------------------------------------------- #
#  The device roster
# --------------------------------------------------------------------------- #

DISCRETE_GPU = Device(
    key="discrete_gpu",
    name="Desktop + RTX 4090 (24 GB)",
    kind="discrete_gpu",
    accel_memory_gb=24.0,
    memory_bandwidth_gb_s=1008.0,
    gen_power_w=420.0,
    train_power_w=430.0,
    idle_power_w=60.0,
    street_price_usd=2000.0,
    cite="databasemart2026",
    note=(
        "The Pass-1 edge device. 24 GB is the binding constraint: a 27B model "
        "fits for inference in 4-bit (~17 GB) but leaves no room for optimizer "
        "state, so 27B adaptation on this class needs 4-bit weights plus an "
        "8-bit paged optimizer and still runs close to the edge."
    ),
    throughput={
        "8b_q4": 141.0,
        "8b_bf16": 95.0,
    },
)

UNIFIED_APU = Device(
    key="unified_apu",
    name="AMD Ryzen AI MAX+ 395 'Strix Halo' (128 GB unified)",
    kind="unified_apu",
    accel_memory_gb=128.0,
    memory_bandwidth_gb_s=256.0,
    gen_power_w=120.0,
    train_power_w=130.0,
    idle_power_w=25.0,
    street_price_usd=2000.0,
    cite="strixhalo2026",
    note=(
        "16 Zen 5 cores, 40 RDNA 3.5 CUs (gfx1151), 50+ TOPS XDNA 2 NPU, one "
        "128 GB LPDDR5X-8000 pool addressed by all three. Roughly a quarter of "
        "the 4090's bandwidth and roughly five times its addressable memory. "
        "That trade is what makes 27B-class *adaptation* a consumer activity: "
        "the published recipe reserves ~80 GB peak for a bf16 LoRA run on a "
        "27B model, which simply cannot be allocated on a 24 GB card. "
        "NOTE ON SKU: this profile is the 128 GB configuration, because "
        "that is the one the cited training recipe runs on and an 80 GB "
        "peak cannot be allocated on anything smaller. The authors' own "
        "machine, and every measurement in measure/, is the 64 GB "
        "configuration of the same part: identical bandwidth and compute, "
        "half the pool. Inference figures therefore transfer between the "
        "two SKUs and the 27B training claim does not, which is why the "
        "two are kept apart here."
    ),
    throughput={
        # Measured on this exact part in the published gfx1151 recipe.
        "27b_q8": 7.5,
        "27b_mtp_spec": 19.0,
        "35b_a3b_q4": 50.0,
        # Interpolated from the same source's Q4/Q8 relationship, and since
        # corroborated by our own measurement: a dense 27.9B Q4_K_M model on
        # the 64 GB SKU of this part runs at 12.5 tok/s single-stream, 3.9%
        # below this figure. See measure/results/ and measured.py.
        "27b_q4": 13.0,
        "8b_q4": 42.0,
        # Primary measurements, this study, Ollama 0.33.2, Q4_K_M,
        # single-stream, 64 GB SKU. autoyou2026measure.
        "3b_q4_measured": 78.7,
        "8b_q4_measured": 37.8,
        "27b_dense_q4_measured": 12.5,
    },
)

LAPTOP_NPU = Device(
    key="laptop_npu",
    name="Copilot+ class laptop NPU (40-50 TOPS)",
    kind="laptop_npu",
    accel_memory_gb=16.0,
    memory_bandwidth_gb_s=120.0,
    gen_power_w=25.0,
    train_power_w=30.0,
    idle_power_w=8.0,
    street_price_usd=1200.0,
    cite="assumption",
    note=(
        "Present in essentially every new laptop, and the reason the addressable "
        "fleet grows without AutoYou doing anything. Modelled as inference-only: "
        "the NPU toolchains expose quantized forward passes, not training "
        "kernels, so adaptation on this class means *receiving* an adapter "
        "trained elsewhere on hardware the same person owns."
    ),
    throughput={
        "8b_q4": 18.0,
    },
)

DEVICES: Dict[str, Device] = {
    d.key: d for d in (DISCRETE_GPU, UNIFIED_APU, LAPTOP_NPU)
}


# --------------------------------------------------------------------------- #
#  Model dimensions - needed to size adapters and activations honestly
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelSpec:
    """Architecture facts needed to size a PEFT job."""

    key: str
    name: str
    params_b: float                 # total parameters, billions
    active_params_b: float          # activated per token (== params_b if dense)
    layers: int
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    context: int
    licence: str
    multimodal: bool
    cite: str
    note: str = ""

    @property
    def is_moe(self) -> bool:
        return self.active_params_b < self.params_b * 0.95

    def attn_projection_dims(self) -> Tuple[Tuple[int, int], ...]:
        """(in, out) shapes of q/k/v/o projections in one layer."""
        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv_heads * self.head_dim
        return (
            (self.hidden, q_out),      # q_proj
            (self.hidden, kv_out),     # k_proj
            (self.hidden, kv_out),     # v_proj
            (q_out, self.hidden),      # o_proj
        )

    def mlp_projection_dims(self) -> Tuple[Tuple[int, int], ...]:
        """(in, out) shapes of gate/up/down projections in one layer."""
        return (
            (self.hidden, self.intermediate),   # gate_proj
            (self.hidden, self.intermediate),   # up_proj
            (self.intermediate, self.hidden),   # down_proj
        )


# The models AutoYou actually trains or could train today.
QWEN3_8_27B = ModelSpec(
    key="qwen3.8-27b",
    name="Qwen3.8-27B",
    params_b=27.78,
    active_params_b=27.78,
    layers=64,
    hidden=5120,
    intermediate=27648,
    n_heads=40,
    n_kv_heads=8,
    head_dim=128,
    context=262_144,
    licence="Apache-2.0",
    multimodal=True,
    cite="qwen38_2026",
    note=(
        "Released 2026-08-14. Dense, natively multimodal (text/image/video), "
        "262K context, 27.78B parameters, ~17 GB as a 4-bit build. Reported "
        "gains over Qwen3.6-27B on an identical architecture: Terminal-Bench "
        "2.1 63.4->73.0, DeepSWE 1.1 13.3->42.2, OSWorld-Verified 63.9->84.3, "
        "SWE-MM 25.7->38.6; Artificial Analysis Intelligence Index 52 (+14). "
        "hidden/intermediate/head counts below the parameter count are the "
        "study's reconstruction of a 64-layer 27.78B decoder and are used only "
        "to size adapters, never to claim benchmark results."
    ),
)

QWEN3_5_27B = ModelSpec(
    key="qwen3.5-27b",
    name="Qwen3.5-27B",
    params_b=27.0,
    active_params_b=27.0,
    layers=64,
    hidden=5120,
    intermediate=27648,
    n_heads=40,
    n_kv_heads=8,
    head_dim=128,
    context=262_144,
    licence="Apache-2.0",
    multimodal=True,
    cite="qwen35_2026",
    note=(
        "February 2026. The dense 27B rung of the Qwen3.5 family (0.8B-27B "
        "dense; 35B-A3B, 122B-A10B, 397B-A17B MoE). Hybrid attention with "
        "GatedDeltaNet layers, which is why the published Strix Halo recipe "
        "needs flash-linear-attention kernels and eager attention."
    ),
)

QWEN3_5_35B_A3B = ModelSpec(
    key="qwen3.5-35b-a3b",
    name="Qwen3.5-35B-A3B (sparse MoE)",
    params_b=35.0,
    active_params_b=3.0,
    layers=48,
    hidden=4096,
    intermediate=1536,          # per-expert
    n_heads=32,
    n_kv_heads=4,
    head_dim=128,
    context=262_144,
    licence="Apache-2.0",
    multimodal=True,
    cite="qwen35_2026",
    note=(
        "The important shape for edge inference: 35B of knowledge at 3B of "
        "arithmetic per token. Measured at ~50 tok/s on Strix Halo in Q4 "
        "against ~7.5 tok/s for the 27B dense model in Q8 - a 6.7x throughput "
        "gap on identical hardware, which the Pass-1 single-device energy "
        "model had no way to express."
    ),
)

QWEN25_VL_7B = ModelSpec(
    key="qwen2.5-vl-7b",
    name="Qwen2.5-VL-7B-Instruct",
    params_b=8.29,
    active_params_b=8.29,
    layers=28,
    hidden=3584,
    intermediate=18944,
    n_heads=28,
    n_kv_heads=4,
    head_dim=128,
    context=128_000,
    licence="Apache-2.0",
    multimodal=True,
    cite="qwen25vl2025",
    note=(
        "The base under the shipped AutoYou Support adapter. Chosen over the "
        "3B for licensing, not quality: Qwen2.5 releases are Apache-2.0 except "
        "the 3B and 72B, which carry the Qwen Research License and therefore "
        "cannot sit under a paid product."
    ),
)

QWEN25_VL_3B = ModelSpec(
    key="qwen2.5-vl-3b",
    name="Qwen2.5-VL-3B-Instruct",
    params_b=3.75,
    active_params_b=3.75,
    layers=36,
    hidden=2048,
    intermediate=11008,
    n_heads=16,
    n_kv_heads=2,
    head_dim=128,
    context=128_000,
    licence="Qwen-Research",
    multimodal=True,
    cite="qwen25vl2025",
    note="The v1-v4 AutoYou Support base. Research licence; not shippable.",
)

MINISTRAL_8B = ModelSpec(
    key="ministral-3-8b",
    name="Ministral-3 8B Instruct",
    params_b=8.02,
    active_params_b=8.02,
    layers=36,
    hidden=4096,
    intermediate=12288,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    context=128_000,
    licence="Apache-2.0",
    multimodal=False,
    cite="mistral2024",
    note="The Fine Tuning agent's default training base.",
)

MODELS: Dict[str, ModelSpec] = {
    m.key: m
    for m in (
        QWEN3_8_27B,
        QWEN3_5_27B,
        QWEN3_5_35B_A3B,
        QWEN25_VL_7B,
        QWEN25_VL_3B,
        MINISTRAL_8B,
    )
}


if __name__ == "__main__":
    print(f"{len(DEVICES)} device classes, {len(MODELS)} model specs\n")
    for d in DEVICES.values():
        print(f"  {d.name}")
        print(
            f"    {d.accel_memory_gb:.0f} GB @ {d.memory_bandwidth_gb_s:.0f} GB/s, "
            f"{d.train_power_w:.0f} W training, usable "
            f"{d.trainable_ceiling_gb():.0f} GB"
        )
    print()
    for m in MODELS.values():
        shape = "MoE" if m.is_moe else "dense"
        print(
            f"  {m.name:<28} {m.params_b:>6.2f}B {shape:<5} "
            f"{m.layers:>3}L  {m.licence}"
        )
