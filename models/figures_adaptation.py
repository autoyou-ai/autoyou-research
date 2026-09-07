# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
figures_adaptation.py - Pass-3 figures for the on-device adaptation track.

Shares the palette, the save() helper and the styling contract with figures.py
so the two tracks produce one visual language. Every value is pulled from
peft.py / adaptation.py / evidence.py; nothing is typed in twice.

Run:  ../.venv/bin/python figures_adaptation.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
import devices as V
import evidence as E
import models as M
import peft as P
import adaptation as A
from hypotheses import CAP_RATIOS
from figures import (
    save, _despine, INK, MUTED, GRID, EDGE, EDGE_D, CLOUD, HEAVY, ACCENT,
    PASS, FAIL, ZERO,
)


# --------------------------------------------------------------------------- #
def fig_memory_wall():
    """Why device CLASS, not device speed, decides what can be adapted."""
    jobs = [
        ("8B\npersona", V.MINISTRAL_8B, 32),
        ("7B VLM\nsupport", V.QWEN25_VL_7B, 32),
        ("27B\nr=32", V.QWEN3_8_27B, 32),
        ("27B\nr=128", V.QWEN3_8_27B, 128),
    ]
    labels, bf16, nf4 = [], [], []
    for label, mdl, r in jobs:
        labels.append(label)
        bf16.append(P.training_memory(mdl, r, weight_dtype="bf16",
                                      seq_len=4096).total_gb)
        nf4.append(P.training_memory(mdl, r, weight_dtype="nf4_dq",
                                     seq_len=4096).total_gb)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(jobs))
    w = 0.36
    ax.bar(x - w / 2, bf16, w, label="bf16 weights", color=CLOUD, zorder=3)
    ax.bar(x + w / 2, nf4, w, label="4-bit NF4 + double-quant", color=EDGE, zorder=3)

    for dev, colour, style in ((V.DISCRETE_GPU, HEAVY, "--"),
                               (V.UNIFIED_APU, EDGE_D, "-")):
        y = dev.trainable_ceiling_gb()
        ax.axhline(y, color=colour, ls=style, lw=1.6, zorder=4)
        ax.text(len(jobs) - 0.42, y * 1.06,
                f"{dev.name.split('(')[0].strip()}  ({y:.0f} GB usable)",
                ha="right", va="bottom", fontsize=8.5, color=colour,
                fontweight="bold")

    for xi, (b, n) in enumerate(zip(bf16, nf4)):
        ax.text(xi - w / 2, b * 1.05, f"{b:.0f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + w / 2, n * 1.05, f"{n:.0f}", ha="center", fontsize=8.5, color=INK)

    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Peak training memory  (GB, log scale)")
    ax.set_ylim(2, 400)
    ax.set_title("The memory wall decides what a household can teach")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _despine(ax); ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04,
             "Model predicts 79.5 GB for the one published 27B run reported at ~80 GB "
             "(-0.6%). Quantization moves the 27B jobs under a 24 GB ceiling; only the "
             "unified-memory class clears them in bf16.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_memory_wall")


# --------------------------------------------------------------------------- #
def fig_adaptation_breakeven():
    """How long an adaptation run takes to repay its own energy."""
    cases = [
        ("Persona 8B\non dGPU", V.MINISTRAL_8B, V.DISCRETE_GPU,
         D.CORPUS_PERSONA_TOKENS.value, EDGE),
        ("Persona 8B\non APU", V.MINISTRAL_8B, V.UNIFIED_APU,
         D.CORPUS_PERSONA_TOKENS.value, EDGE_D),
        ("Support 7B VLM\non APU", V.QWEN25_VL_7B, V.UNIFIED_APU,
         D.CORPUS_SUPPORT_TOKENS.value, ACCENT),
        ("Capability 27B\non APU", V.QWEN3_5_27B, V.UNIFIED_APU,
         D.CORPUS_CAPABILITY_TOKENS.value, CLOUD),
    ]
    labels, days, kwh, cols = [], [], [], []
    for label, mdl, dev, tok, col in cases:
        b = A.breakeven(mdl, dev, tok,
                        queries_per_day=D.QUERIES_PER_USER_PER_DAY.value)
        labels.append(label); days.append(b.days_to_repay_energy)
        kwh.append(b.training_kwh); cols.append(col)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    y = np.arange(len(labels))
    ax.barh(y, days, color=cols, height=0.6, zorder=3)
    for yi, (d, k) in enumerate(zip(days, kwh)):
        ax.text(d + max(days) * 0.02, yi, f"{d:.0f} days   ({k:.2f} kWh)",
                va="center", fontsize=9, color=INK)
    ax.axvline(365, color=ZERO, ls="--", lw=1.3, zorder=4)
    ax.text(365, len(labels) - 0.35, " one year", fontsize=8.5, color=ZERO,
            va="top")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, max(days) * 1.55)
    ax.set_xlabel(f"Days of ordinary use to repay the training energy "
                  f"(at {D.QUERIES_PER_USER_PER_DAY.value:.0f} queries/day)")
    ax.set_title("An adapter is not free - but it pays for itself")
    _despine(ax, left=False); ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.05,
             "Training charged at full device power; savings measured against a "
             "HYPERSCALE cloud query, the cloud's best case. Both choices are made "
             "against the hypothesis.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_adaptation_breakeven")


# --------------------------------------------------------------------------- #
def fig_quality_vs_coverage():
    """The Pass-3 headline: adaptation buys quality, not coverage, at alpha=0.70."""
    sweep = A.alpha_sweep(CAP_RATIOS)
    alphas = np.array([float(a) for a in sweep])
    base = np.array([sweep[a]["base"] for a in sweep])
    lo = np.array([sweep[a]["adapted_lo"] for a in sweep])
    mid = np.array([sweep[a]["adapted_mid"] for a in sweep])
    hi = np.array([sweep[a]["adapted_hi"] for a in sweep])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                  gridspec_kw={"width_ratios": [1.45, 1]})

    ax.fill_between(alphas, lo, hi, color=EDGE, alpha=0.16, zorder=2,
                    label="adapted (bounded range)")
    ax.plot(alphas, mid, color=EDGE_D, lw=2.4, marker="o", ms=4.5,
            zorder=4, label="adapted (central)")
    ax.plot(alphas, base, color=CLOUD, lw=2.2, ls="--", marker="s", ms=4.5,
            zorder=4, label="unadapted base")
    a0 = D.ALPHA_SUFFICIENCY.value
    ax.axvline(a0, color=ZERO, ls=":", lw=1.4, zorder=3)
    ax.text(a0 + 0.004, 0.30, f"study default\nalpha = {a0:.2f}", fontsize=8.4,
            color=ZERO)
    knee = D.ALPHA_COVERAGE_KNEE.value
    ax.axvspan(knee, alphas.max(), color=ACCENT, alpha=0.07, zorder=1)
    ax.text((knee + alphas.max()) / 2, 0.30,
            "adaptation buys\nCOVERAGE here", ha="center", fontsize=8.4,
            color=ACCENT, fontweight="bold")
    ax.set_xlabel("Acceptance threshold alpha  (SLM quality / frontier)")
    ax.set_ylabel("f$_s$  -  workload served locally")
    ax.set_ylim(0.25, 1.0)
    ax.set_title("Coverage responds only to a demanding bar")
    ax.legend(frameon=False, fontsize=8.6, loc="lower left")
    _despine(ax)

    q = A.quality_uplift(CAP_RATIOS)
    bars = [("Coverage\n(f$_s$)", q["base"]["served_fraction"],
             q["adapted"]["served_fraction"]),
            ("Mean quality\non served work", q["base"]["mean_ratio_served"],
             q["adapted"]["mean_ratio_served"])]
    x = np.arange(len(bars)); w = 0.35
    ax2.bar(x - w / 2, [b[1] for b in bars], w, color=CLOUD, zorder=3,
            label="unadapted")
    ax2.bar(x + w / 2, [b[2] for b in bars], w, color=EDGE_D, zorder=3,
            label="adapted")
    for xi, b in enumerate(bars):
        ax2.text(xi - w / 2, b[1] + 0.02, f"{b[1]:.0%}", ha="center",
                 fontsize=8.6, color=INK)
        ax2.text(xi + w / 2, b[2] + 0.02, f"{b[2]:.0%}", ha="center",
                 fontsize=8.6, color=INK)
        delta = (b[2] - b[1]) * 100
        ax2.text(xi, 0.10, f"{delta:+.1f} pp", ha="center", fontsize=10,
                 fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.28", lw=0,
                           fc=(EDGE_D if delta > 0.5 else MUTED)))
    ax2.set_xticks(x); ax2.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax2.set_ylim(0, 1.30)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_title(f"At alpha = {a0:.2f}", pad=26)
    ax2.legend(frameon=False, fontsize=8.6, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06))
    _despine(ax2); ax2.grid(axis="x", visible=False)

    fig.text(0.5, -0.03,
             f"Illustrative capability model, not measured quality: coverage moves "
             f"{q['coverage_gain_pp']:+.1f} pp. What moves is quality, "
             f"{q['quality_gain_pp']:+.1f} pp, closing "
             f"{q['gap_to_frontier_closed']:.0%} of the remaining gap to the frontier.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "fig_quality_vs_coverage")


# --------------------------------------------------------------------------- #
def fig_measured_adapter():
    """This deployment's own measured adapter runs - capability and safety."""
    s = E.summary()
    runs = [(k, v) for k, v in s["runs"].items() if v]
    if not runs:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.axis("off")
        ax.text(.5, .55, "No independently verified adaptation trial", ha="center", fontsize=15)
        ax.text(.5, .35, "Historical scores are not evidence of safety or general capability.\nLocal deployment evidence is opt-in.", ha="center", fontsize=10)
        save(fig, "fig_measured_adapter")
        return

    labels = [k for k, _ in runs]
    screen = [(v.get("screen_accuracy") or 0) for _, v in runs]
    refuse = [(v.get("refusal_rate") or 0) for _, v in runs]
    leak = [(v.get("code_leak_rate") or 0) for _, v in runs]
    bases = [str(v.get("base_model", "")).split("/")[-1].replace("-Instruct", "")
             for _, v in runs]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(labels)); w = 0.26
    ax.bar(x - w, screen, w, color=EDGE_D, zorder=3, label="screen identification (higher better)")
    ax.bar(x, refuse, w, color=ACCENT, zorder=3, label="refuses out-of-scope code (higher better)")
    ax.bar(x + w, leak, w, color=HEAVY, zorder=3, label="leaks source code (lower better)")

    for xi in x:
        for off, vals in ((-w, screen), (0, refuse), (w, leak)):
            v = vals[xi]
            ax.text(xi + off, v + 0.02, f"{v:.0%}", ha="center", fontsize=8,
                    color=INK)
        ax.text(xi, -0.115, bases[xi], ha="center", fontsize=7.6, color=MUTED)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Score on the fixed probe set")
    ax.set_title("Measured: one product's own support adapter, five runs")
    ax.legend(frameon=False, fontsize=8.4, ncol=1, loc="upper center")
    _despine(ax); ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.10,
             "Same 3B base for v1 and v3: a rank-32 LoRA over 1,750 samples took screen "
             "identification 30.8% -> 92.9% while refusal went 0% -> 100% and code "
             "leakage 25% -> 0%. Doubling the base to 7B (v5) changed nothing (-0.6 pp): "
             "the constraint was the corpus. n=1 deployment; 12-14 screen and 8 code probes.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_measured_adapter")


# --------------------------------------------------------------------------- #
def fig_method_landscape():
    """The 2026 PEFT roster: what each method costs and what it is reported to buy."""
    keys = ["lora", "qlora", "rslora", "lora_plus", "neftune", "pissa", "olora",
            "eva", "corda", "loftq", "dora", "qdora", "adalora", "vera",
            "lora_xs", "galore"]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    fam_col = {"reparam": EDGE_D, "init": ACCENT, "scaling": PASS,
               "optimizer": CLOUD, "quantization": HEAVY, "regularizer": MUTED}
    # The near-free methods all sit within a hair of x=1.0, so labels are placed
    # by hand rather than fought with.
    label_offset = {
        "lora": (8, -14), "qlora": (9, -4), "rslora": (-46, 10),
        "lora_plus": (-14, -20), "neftune": (-52, -6), "pissa": (9, 6),
        "olora": (10, -12), "eva": (10, 2), "corda": (10, 12),
        "loftq": (-40, 16), "dora": (9, 4), "qdora": (9, 4),
        "adalora": (9, 2), "vera": (-16, -20), "lora_xs": (-30, 18),
        "galore": (9, 2),
    }
    # Small horizontal jitter so overlapping vertical range bars stay readable.
    jitter = {"rslora": -0.006, "lora_plus": 0.004, "neftune": -0.012,
              "olora": 0.008, "eva": 0.014, "corda": 0.020, "loftq": -0.004,
              "lora_xs": 0.006}
    for k in keys:
        m = P.METHODS[k]
        lo, mid, hi = m.quality_delta
        x = m.extra_train_time + jitter.get(k, 0.0)
        col = fam_col[m.family]
        ax.plot([x, x], [lo, hi], color=col, lw=1.3, alpha=0.45, zorder=2)
        size = 240 if m.memory_multiplier < 0.5 else 90
        ax.scatter([x], [mid], s=size, color=col, zorder=4,
                   edgecolor="white", linewidth=1.1)
        ax.annotate(m.name.split(" (")[0], (x, mid),
                    textcoords="offset points",
                    xytext=label_offset.get(k, (8, 5)), fontsize=8,
                    color=INK, zorder=5)

    ax.axhline(0, color=ZERO, lw=1.2, ls="--", zorder=3)
    ax.axvspan(0.90, 1.03, color=PASS, alpha=0.06, zorder=1)
    ax.text(0.965, 6.6, "free", fontsize=9.5, color=PASS, fontweight="bold",
            ha="center")
    ax.axvline(1.0, color=ZERO, lw=1.0, ls=":", zorder=3)
    ax.set_xlabel("Training wall-clock, relative to plain LoRA")
    ax.set_ylabel("Reported quality vs LoRA  (percentage points)")
    ax.set_title("The 2026 roster: free wins, paid wins, and wrong shapes")
    ax.set_xlim(0.90, 1.72)
    ax.set_ylim(-4.2, 7.2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f, ms=7)
               for f, c in fam_col.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8.2, loc="lower right",
              ncol=2, title="method family", title_fontsize=8.2)
    _despine(ax)
    fig.text(0.5, -0.03,
             "Bars span the reported range; the dot is the modal reported gain. Large dots "
             "are methods that shrink the adapter by >20x. rsLoRA, LoRA+ and NEFTune sit on "
             "the 1.0 line: reported gains at no wall-clock cost. Ranges are the methods' "
             "own authors' benchmarks, not measurements of ours.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_method_landscape")


# --------------------------------------------------------------------------- #
def fig_adapter_transport():
    """Adapter size decides whether personalisation can sync between your devices."""
    t = A.adapter_transport(V.MINISTRAL_8B, 32, link_mbit_s=20.0)
    order = ["dora", "lora", "vera", "lora_xs"]
    labels = {"lora": "LoRA r=32", "dora": "DoRA r=32",
              "vera": "VeRA r=32", "lora_xs": "LoRA-XS r=32"}
    vals = [t[k]["megabytes"] for k in order]
    secs = [t[k]["seconds_at_link"] for k in order]
    cols = [CLOUD if s >= 3 else EDGE_D for s in secs]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(order))
    ax.barh(y, vals, color=cols, height=0.58, zorder=3)
    for yi, (v, s) in enumerate(zip(vals, secs)):
        note = "a message" if s < 3 else "a download"
        ax.text(v * 1.15, yi, f"{v:,.1f} MB  -  {s:.1f} s  ({note})",
                va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 3000)
    ax.set_yticks(y); ax.set_yticklabels([labels[k] for k in order])
    ax.set_xlabel("Adapter size  (MB, log scale)")
    ax.set_title("Can your personal adapter follow you between devices?")
    _despine(ax, left=False); ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.07,
             "Time to cross an existing WebRTC datachannel at 20 Mbit/s. The "
             "low-parameter methods turn a personal adapter from a file transfer into a "
             "message, which is what lets it sync between a person's own devices with no "
             "server-side copy.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_adapter_transport")


# --------------------------------------------------------------------------- #
def fig_device_class():
    """Throughput vs addressable memory: why the APU class changed the answer."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for dev, col in ((V.DISCRETE_GPU, HEAVY), (V.UNIFIED_APU, EDGE_D),
                     (V.LAPTOP_NPU, MUTED)):
        ax.scatter([dev.memory_bandwidth_gb_s], [dev.accel_memory_gb],
                   s=420, color=col, zorder=4, edgecolor="white", linewidth=1.4)
        ax.annotate(dev.name.split("(")[0].strip(),
                    (dev.memory_bandwidth_gb_s, dev.accel_memory_gb),
                    textcoords="offset points", xytext=(0, 22), ha="center",
                    fontsize=9, color=INK, fontweight="bold")
        ax.annotate(f"{dev.train_power_w:.0f} W training",
                    (dev.memory_bandwidth_gb_s, dev.accel_memory_gb),
                    textcoords="offset points", xytext=(0, -30), ha="center",
                    fontsize=8, color=MUTED)

    for mdl, label in ((V.QWEN3_8_27B, "27B bf16 LoRA"),
                       (V.MINISTRAL_8B, "8B bf16 LoRA")):
        need = P.training_memory(mdl, 64, weight_dtype="bf16",
                                 seq_len=4096).total_gb
        ax.axhline(need, color=ACCENT, ls="--", lw=1.2, zorder=2)
        ax.text(1050, need * 1.04, f"{label} needs {need:.0f} GB", ha="right",
                fontsize=8.2, color=ACCENT)

    ax.set_xlabel("Memory bandwidth  (GB/s)  -  how fast it thinks")
    ax.set_ylabel("Addressable memory  (GB)  -  what it can hold")
    ax.set_yscale("log"); ax.set_ylim(8, 300)
    ax.set_xlim(0, 1150)
    ax.set_title("Two axes, and only one of them decides what you can teach")
    _despine(ax)
    fig.text(0.5, -0.04,
             "The fastest consumer device is not the one that can adapt the largest "
             "model. Bandwidth sets how quickly a run finishes; addressable memory sets "
             "whether it can start at all.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_device_class")


ALL = [fig_memory_wall, fig_adaptation_breakeven, fig_quality_vs_coverage,
       fig_measured_adapter, fig_method_landscape, fig_adapter_transport,
       fig_device_class]

if __name__ == "__main__":
    for fn in ALL:
        fn()
    print("Adaptation figures written to",
          os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures")))
