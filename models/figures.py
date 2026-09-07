# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
figures.py - PaperBanana-style quantitative figures (clean modern vector).
Saves both PDF (for LaTeX/Tectonic) and PNG/SVG (for slides/marketing) into
../figures. All numbers come from data.py + models.py (no hard-coding).

Run:  ../.venv/bin/python figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

import data as D
import models as M
import hypotheses as H
import measured as MEAS

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# --------------------------- PaperBanana-ish style ------------------------- #
INK = "#1f2933"; MUTED = "#7b8794"; GRID = "#e4e7eb"
EDGE = "#16a085"      # teal-green = decentralized / edge / clean
EDGE_D = "#0e6e5c"
CLOUD = "#e8743b"     # warm orange = cloud / heavy
HEAVY = "#c0392b"     # red = reasoning / worst
ACCENT = "#2f6fed"    # blue accent
PASS = "#16a085"; FAIL = "#c0392b"; ZERO = "#52606d"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.axisbelow": True, "figure.dpi": 140, "savefig.bbox": "tight",
    "svg.fonttype": "none",
})


def _despine(ax, left=True, bottom=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


# --------------------------------------------------------------------------- #
#  Paper placement
#
#  A matplotlib figure carries physical font sizes. Author one at 7.2 inches
#  wide with 11pt labels, drop it into IEEEtran's 3.5-inch column, and every
#  label lands at 5.4pt against 10pt body text - which is what made the first
#  build look like slides pasted into a paper rather than a paper.
#
#  The fix is to size type for the placement, not for the canvas: scale every
#  font by the inverse of the placement ratio so labels land at TARGET_PT once
#  LaTeX has scaled the whole graphic down. Figures listed in DOUBLE_COLUMN are
#  placed at \textwidth and scale far less; everything else goes in a column.
# --------------------------------------------------------------------------- #

IEEE_COLUMN_IN = 3.5        # \columnwidth in IEEEtran conference format
IEEE_TEXT_IN = 7.16         # \textwidth
TARGET_PT = 8.0             # rendered size of ordinary figure text
BASE_PT = 11.0              # the rcParams size these figures are authored at

# Only figures that genuinely need the width: side-by-side panels, or a scatter
# with more labels than a column can separate. Everything else reads better as
# a column-width figure than as a page-wide band.
DOUBLE_COLUMN = {
    "fig_quality_vs_coverage",   # two panels side by side
    "fig_method_landscape",      # 16 annotated points
    "fig_bandwidth_roof",        # one panel per machine, placed full width
}


def _scale_paper_fonts(fig, name):
    """Resize every text artist so it lands at TARGET_PT in the paper.

    Returns the list of (artist, original_size) needed to undo it.
    """
    placement = IEEE_TEXT_IN if name in DOUBLE_COLUMN else IEEE_COLUMN_IN
    width_in = fig.get_size_inches()[0]
    k = (TARGET_PT / BASE_PT) * (width_in / placement)
    originals = []
    for t in fig.findobj(matplotlib.text.Text):
        size = t.get_fontsize()
        originals.append((t, size))
        t.set_fontsize(size * k)
    return originals


def save(fig, name):
    """Write one figure in three formats, with one deliberate difference.

    Every figure here carries an explanatory footnote added with ``fig.text``.
    That footnote is right for the PNG and SVG, which are used in slides and on
    the web where nothing else explains the chart. It is wrong for the PDF,
    which goes into the papers, where LaTeX supplies a ``\\caption`` saying the
    same thing - so the reader gets the sentence twice, and the baked-in copy is
    scaled to column width until it is too small to read.

    So: hide figure-level text for the PDF, restore it for the raster and vector
    exports. ``bbox_inches="tight"`` then also reclaims the vertical space the
    footnote occupied, which is why the PDF crops closer than the PNG.

    Axis-level annotations (``ax.text``, ``ax.annotate``) are untouched - those
    are part of the chart, not a caption.
    """
    captions = [t for t in fig.texts if t.get_visible()]
    # Academic convention: a figure has no title of its own. The LaTeX caption
    # is the title, and printing both gives the reader the same sentence twice -
    # once in a rhetorical voice ("When the thesis holds") that does not belong
    # in figure furniture at all. Titles are kept for the raster exports, which
    # are used in slides and on the web where nothing else names the chart.
    # A single axes title duplicates the LaTeX caption and is stripped. In a
    # multi-panel figure the titles are not a caption at all: they name the
    # panels, and dropping them leaves the reader unable to tell which
    # panel is which. Strip only when exactly one axes carries a title.
    titled = [ax.title for ax in fig.axes if ax.get_title()]
    titles = titled if len(titled) == 1 else []
    title_texts = [t.get_text() for t in titles]

    for t in captions:
        t.set_visible(False)
    for t in titles:
        t.set_text("")
    font_state = _scale_paper_fonts(fig, name)

    fig.savefig(os.path.join(OUT, f"{name}.pdf"))

    for artist, size in font_state:
        artist.set_fontsize(size)
    for t, text in zip(titles, title_texts):
        t.set_text(text)
    for t in captions:
        t.set_visible(True)

    # A one-line caption under a tight bounding box drags the whole raster out
    # to the width of the sentence - fig_method_landscape came out three times
    # wider than its own axes. Wrap to the figure's own width instead.
    import textwrap
    originals = [(t.get_text(), t.get_va()) for t in captions]
    width_chars = max(int(fig.get_size_inches()[0] * 15), 40)
    for t in captions:
        flat = " ".join(t.get_text().split())
        t.set_text("\n".join(textwrap.wrap(flat, width_chars)))
        # Anchor the block by its top, or the second and third lines grow
        # upward into the x-axis label instead of down into the margin.
        t.set_va("top")

    for ext in ("png", "svg"):
        path = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(path)
        if ext == "svg":
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.writelines(line.rstrip() + "\n" for line in lines)

    for t, (original_text, original_va) in zip(captions, originals):
        t.set_text(original_text)
        t.set_va(original_va)

    plt.close(fig)
    note = "  (PDF: title+caption stripped, type scaled for placement)" if captions else ""
    print("wrote", name + note)


# --------------------------------------------------------------------------- #
def fig_energy_ladder():
    """Per-query inference energy across the model-size ladder (log x)."""
    rows = [
        ("Edge SLM 8B (this work)", M.edge_slm_query_central().energy_wh, EDGE),
        ("Llama-3.2 1B", D.E_LLAMA32_1B.value, EDGE),
        ("Llama-3.1 8B (cloud)", D.E_LLAMA31_8B.value, EDGE_D),
        ("Gemini median (Google)", D.E_GEMINI_MEDIAN.value, ACCENT),
        ("GPT-4o (short)", D.E_GPT4O_SHORT.value, CLOUD),
        ("GPT-4o (long ctx)", D.E_GPT4O.value, CLOUD),
        ("Claude-3.7 Sonnet", D.E_CLAUDE37_SONNET.value, CLOUD),
        ("Llama-3.1 405B", D.E_LLAMA31_405B.value, HEAVY),
        ("DeepSeek-R1 (reasoning)", D.E_DEEPSEEK_R1.value, HEAVY),
        ("o3 (reasoning)", D.E_O3.value, HEAVY),
    ]
    rows = rows[::-1]
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=0.66, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Energy per query  (Wh, log scale)")
    ax.set_title("Right-sizing dominates: per-query inference energy")
    for yi, v in zip(y, vals):
        ax.text(v * 1.12, yi, f"{v:.2f}", va="center", ha="left",
                fontsize=9, color=INK)
    ax.set_xlim(0.05, 90)
    _despine(ax, left=False)
    ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.03,
             "An 8B SLM that is capability-sufficient avoids the 3x-65x energy of "
             "frontier/reasoning models relative to cloud 8B. Source: Jegham et al. 2025; Google 2025.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_energy_ladder")


def fig_capability():
    """Capability sufficiency per task class vs the 0.70 acceptance threshold."""
    classes = list(H.CAP_RATIOS.keys())
    # Kept to about ten characters a line: at column width the longer forms
    # ("Extraction /", "RAG-grounded") are wider than the bar pitch and run into
    # their neighbours.
    nice = {"extraction": "Extraction\n/ routing", "rag_qa": "RAG\nQA",
            "summary": "Summary\n/ rewrite", "simple_code": "Simple\ncode",
            "hard_reason": "Hard math\n/ reason"}
    ratios = [H.CAP_RATIOS[c] for c in classes]
    alpha = D.ALPHA_SUFFICIENCY.value
    cols = [PASS if r >= alpha else FAIL for r in ratios]
    weights = [D.WORKLOAD_MIX[c] for c in classes]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(classes))
    ax.bar(x, ratios, color=cols, width=0.62, zorder=3)
    # This label used to sit on top of the "simple code" bar, where dark grey on
    # green was barely readable and it collided with that bar's value. Give it
    # its own margin at the end of the threshold line instead, and stop the line
    # short so it does not strike through the text.
    ax.set_xlim(-0.62, len(classes) + 0.62)
    ax.plot([-0.62, len(classes) - 0.42], [alpha, alpha], color=ZERO, ls="--",
            lw=1.4, zorder=4)
    ax.text(len(classes) - 0.32, alpha, f"alpha = {alpha:.2f}\nsufficiency threshold",
            ha="left", va="center", fontsize=8.6, color=ZERO, zorder=6,
            linespacing=1.4)
    for xi, r, w in zip(x, ratios, weights):
        ax.text(xi, r + 0.015, f"{r:.2f}", ha="center", fontsize=9, color=INK)
        # White-on-colour, so it must fit inside the bar: any overhang is white
        # on white and reads as clipped text. Two words did not fit at column
        # scale; the share alone always does, and the footnote says what it is.
        ax.text(xi, 0.035, f"{int(w*100)}%", ha="center", fontsize=8.5,
                color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([nice[c] for c in classes], fontsize=9.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("SLM-8B capability / frontier")
    f_s, _ = M.sufficient_fraction(D.WORKLOAD_MIX, H.CAP_RATIOS, alpha)
    ax.set_title(f"The '~70%-capable' claim is scoped: f$_s$ = {f_s*100:.0f}% sufficient")
    _despine(ax)
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04,
             "Green = SLM is sufficient (route to edge). Red = route to cloud frontier. "
             "The figure inside each bar is that class's share of the workload. "
             "The blanket claim fails on hard reasoning (0.51).",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_capability")


def fig_savings_by_baseline():
    """Headline honesty figure: carbon savings vs each cloud baseline (MC)."""
    order = ["gemini_median", "gpt4o_short", "gpt4o_long", "claude37_long", "o3_reasoning"]
    nice = {"gemini_median": "Gemini\nmedian", "gpt4o_short": "GPT-4o\nshort",
            "gpt4o_long": "GPT-4o\nlong", "claude37_long": "Claude-3.7\nSonnet",
            "o3_reasoning": "o3\nreasoning"}
    mc = {b: H.monte_carlo_h5(baseline=b) for b in order}
    means = [mc[b]["savings_carbon"]["mean"] for b in order]
    p5 = [mc[b]["savings_carbon"]["p5"] for b in order]
    p95 = [mc[b]["savings_carbon"]["p95"] for b in order]
    ppos = [mc[b]["savings_carbon"]["p_positive"] for b in order]
    cols = [PASS if m > 0 else FAIL for m in means]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    x = np.arange(len(order))
    lo_err = [m - a for m, a in zip(means, p5)]
    hi_err = [b - m for m, b in zip(means, p95)]
    ax.bar(x, [m * 100 for m in means], color=cols, width=0.6, zorder=3,
           yerr=[np.array(lo_err) * 100, np.array(hi_err) * 100],
           error_kw=dict(ecolor=ZERO, lw=1.3, capsize=5))
    ax.axhline(0, color=ZERO, lw=1.2)
    for xi, m, pp in zip(x, means, ppos):
        ax.text(xi, m * 100 + (3 if m > 0 else -6), f"P(save)~{pp:.2f}",
                ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([nice[b] for b in order])
    ax.set_ylabel("Carbon savings vs baseline  (%)")
    ax.set_title("When the thesis holds - and when it does not")
    _despine(ax)
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04,
             "Monte-Carlo (50k draws) over PUE/WUE/grid/edge-power/rebound. Decentralized "
             "edge-SLM LOSES to a best-in-class efficient cloud model (Gemini), but saves "
             "53-64% vs typical/heavy frontier usage.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_savings_by_baseline")


def fig_footprint_breakdown():
    """Cloud-frontier vs edge-SLM footprint for one sufficient query (3 panels)."""
    cloud = M.cloud_query_central(D.E_GPT4O, hyperscale=True)   # GPT-4o long
    edge = M.edge_slm_query_central()
    metrics = [("Energy (Wh)", cloud.energy_wh, edge.energy_wh),
               ("Water (mL)", cloud.water_ml, edge.water_ml),
               ("Carbon (gCO2e)", cloud.carbon_g, edge.carbon_g)]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.4))
    for ax, (title, c, e) in zip(axes, metrics):
        ax.bar([0, 1], [c, e], color=[CLOUD, EDGE], width=0.62, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Cloud\nGPT-4o", "Edge\nSLM-8B"])
        ax.set_title(title, fontsize=11)
        red = (c - e) / c * 100
        ax.text(1, e + c * 0.11, f"-{red:.0f}%", ha="center", va="bottom", color=EDGE_D,
                fontsize=10, fontweight="bold")
        for xi, v in zip([0, 1], [c, e]):
            ax.text(xi, v + c * 0.025, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
        ax.set_ylim(0, c * 1.25)
        _despine(ax); ax.grid(axis="x", visible=False)
    fig.suptitle("Per-query footprint, capability-sufficient task: cloud frontier vs edge SLM",
                 fontsize=12, fontweight="bold", y=1.04)
    save(fig, "fig_footprint_breakdown")


def fig_grid_crossover():
    """Edge vs cloud carbon as a function of LOCAL grid carbon intensity."""
    ci = np.linspace(0, 760, 200)
    # cloud-8B uses the GLOBAL grid (its DC region); edge uses the LOCAL grid.
    e_cloud = D.E_LLAMA31_8B.value
    e_edge = M.edge_slm_query_central().energy_wh
    emb = M.embodied_per_query_g(D.EMBODIED_EDGE_KG.value,
                                 D.EDGE_DEVICE_LIFETIME_QUERIES.value,
                                 D.EMBODIED_ATTRIB_FRACTION.value)
    cloud_carbon_eu = e_cloud / 1000.0 * D.CI_EU.value      # clean cloud region
    cloud_carbon_glob = e_cloud / 1000.0 * D.CI_GLOBAL.value
    edge_carbon = e_edge / 1000.0 * ci + emb
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ci, edge_carbon, color=EDGE, lw=2.6, label="Edge SLM-8B (local grid)", zorder=4)
    ax.axhline(cloud_carbon_glob, color=CLOUD, lw=2.2, ls="-",
               label="Cloud SLM-8B (global grid 458)")
    ax.axhline(cloud_carbon_eu, color=HEAVY, lw=2.0, ls="--",
               label="Cloud SLM-8B (clean EU grid 140)")
    # crossover with clean cloud
    cross = (cloud_carbon_eu - emb) / (e_edge / 1000.0)
    if 0 < cross < 760:
        ax.axvline(cross, color=ZERO, ls=":", lw=1.2)
        ax.text(cross + 6, ax.get_ylim()[1] * 0.92, f"crossover\n{cross:.0f} g/kWh",
                fontsize=8.5, color=ZERO)
    # Anchor every label by its right edge, above the point: the line rises to
    # the right, so up-and-left is the only quadrant that is reliably empty, and
    # it also keeps the rightmost label inside the axes.
    for name, val in [("EU 140", 140), ("US 384", 384),
                      ("Global 458", 458), ("India 708", 708)]:
        ax.scatter([val], [e_edge / 1000.0 * val + emb], color=INK, s=22, zorder=5)
        ax.annotate(name, xy=(val, e_edge / 1000.0 * val + emb),
                    xytext=(-4, 7), textcoords="offset points",
                    fontsize=7.5, ha="right", color=INK, zorder=6)
    ax.set_xlabel("Local electricity grid carbon intensity (gCO2e/kWh)")
    ax.set_ylabel("Carbon per query (gCO2e)")
    ax.set_title("Carbon savings are conditional on the local grid")
    # Upper left is where the crossover annotation and the two cloud lines meet;
    # the edge line rises to the right, leaving the lower right clear.
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _despine(ax); ax.grid(axis="x", visible=True)
    fig.text(0.5, -0.03,
             "Edge wins on carbon only where the local grid is cleaner than the "
             "crossover; on coal-heavy grids edge can emit more than a clean cloud region.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_grid_crossover")


def fig_tco():
    """Cost per query: edge vs cloud tiers (log scale)."""
    out_t, in_t = D.TYPICAL_OUT_TOKENS.value, D.TYPICAL_IN_TOKENS.value
    e_edge = M.edge_slm_query_central(marginal=True).energy_wh
    rows = [
        ("Edge SLM (owned device)",
         M.edge_cost_per_query(e_edge, D.PRICE_ELECTRICITY.value), EDGE),
        ("Edge SLM (dedicated, amortized)",
         M.edge_cost_per_query(e_edge, D.PRICE_ELECTRICITY.value,
                               D.EDGE_DEVICE_COST.value, 1.0,
                               D.EDGE_DEVICE_LIFETIME_QUERIES.value), EDGE_D),
        ("Cloud budget tier",
         M.cloud_cost_per_query(out_t, in_t, D.API_PRICE_OUT_EFFICIENT.value,
                                D.API_PRICE_IN.value), ACCENT),
        ("Cloud mid/frontier tier",
         M.cloud_cost_per_query(out_t, in_t, D.API_PRICE_OUT_FRONTIER.value,
                                D.API_PRICE_IN.value), CLOUD),
        ("Cloud reasoning (o3-class)",
         M.cloud_cost_per_query(out_t, in_t, 40.0, D.API_PRICE_IN.value), HEAVY),
    ]
    rows = rows[::-1]
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=0.62, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Cost per query  ($, log scale)")
    ax.set_title("Total cost of ownership: edge vs cloud per query")
    for yi, v in zip(y, vals):
        ax.text(v * 1.15, yi, f"${v:.5f}", va="center", fontsize=8.5, color=INK)
    _despine(ax, left=False); ax.grid(axis="y", visible=False)
    ax.set_xlim(min(vals) * 0.4, max(vals) * 4)
    fig.text(0.5, -0.03,
             "On an already-owned device an edge query costs electricity only "
             "(~$0.00005); cloud frontier/reasoning is 50-300x more. "
             "Corroborates NVIDIA's 10-30x SLM cost advantage.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_tco")


def fig_fleet():
    """Fleet-scale annual carbon saved vs adoption (IEA 2030 inference base)."""
    f_s, _ = M.sufficient_fraction(D.WORKLOAD_MIX, H.CAP_RATIOS, D.ALPHA_SUFFICIENCY.value)
    # conservative energy savings fraction = system vs GPT-4o long
    cloud = M.cloud_query_central(D.E_GPT4O, hyperscale=True)
    edge = M.edge_slm_query_central()
    s = M.system_net_savings(f_s, cloud, edge, D.REBOUND_FACTOR.value).savings_frac_energy
    adoptions = [0.05, 0.10, 0.25, 0.50]
    res = [M.fleet_savings(D.DC_DEMAND_2030_TWH.value, D.AI_SHARE_OF_DC.value,
                           D.INFERENCE_SHARE_OF_AI.value, a, s,
                           D.CI_GLOBAL.value) for a in adoptions]
    carbon = [r["carbon_saved_MtCO2e"] for r in res]
    homes = [r["us_homes_equiv"] for r in res]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(adoptions))
    ax.bar(x, carbon, color=EDGE, width=0.6, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(a*100)}%\nadoption" for a in adoptions])
    ax.set_ylabel("Annual carbon avoided (MtCO2e)")
    ax.set_title("Fleet-scale upside on the 2030 inference base (IEA)")
    for xi, c, h in zip(x, carbon, homes):
        ax.text(xi, c, f"{c:.1f} Mt\n~{h/1e6:.1f}M homes", ha="center",
                va="bottom", fontsize=8.5, color=INK)
    ax.set_ylim(0, max(carbon) * 1.25)
    _despine(ax); ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04,
             f"Base: IEA 2030 DC demand 945 TWh x {int(D.AI_SHARE_OF_DC.value*100)}% AI "
             f"x {int(D.INFERENCE_SHARE_OF_AI.value*100)}% inference; shifting f_s="
             f"{f_s*100:.0f}% sufficient queries at {s*100:.0f}% energy savings.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_fleet")


def _carbon_savings_point(**ov):
    """Deterministic carbon-savings fraction vs GPT-4o long, with overrides."""
    ci = ov.get("ci", D.CI_GLOBAL.value)
    gen_p = ov.get("gen_p", D.RTX4090_GEN_POWER.value)
    tps = ov.get("tps", D.RTX4090_TPS_8B.value)
    rebound = ov.get("rebound", D.REBOUND_FACTOR.value)
    alpha = ov.get("alpha", D.ALPHA_SUFFICIENCY.value)
    attrib = ov.get("attrib", D.EMBODIED_ATTRIB_FRACTION.value)
    cloud_c = D.E_GPT4O.value / 1000.0 * ci
    t_s = (300.0 / tps) * 1.15
    e_edge = gen_p * t_s / 3600.0 * D.PUE_EDGE.value
    emb = D.EMBODIED_EDGE_KG.value * 1000.0 * attrib / D.EDGE_DEVICE_LIFETIME_QUERIES.value
    edge_c = e_edge / 1000.0 * ci + emb
    # f_s is normally derived from the workload mix and alpha. It can also be
    # overridden directly, because the mix is an unsourced authorial
    # assumption and is the single largest lever in the study: alpha moves
    # which classes clear the bar, but the WEIGHTS decide what clearing it is
    # worth. Sweeping alpha alone left the mix itself untested.
    f_s = ov.get(
        "f_s",
        sum(w for k, w in D.WORKLOAD_MIX.items() if H.CAP_RATIOS[k] >= alpha))
    dec = f_s * edge_c + (1 - f_s) * cloud_c
    dec_eff = dec + rebound * max(cloud_c - dec, 0.0)
    return (cloud_c - dec_eff) / cloud_c


def fig_tornado():
    """One-at-a-time sensitivity of carbon savings (vs GPT-4o long)."""
    base = _carbon_savings_point()
    factors = [
        # Listed first because it dominates, and because it is the one input
        # here with no source: the workload mix is asserted, not sampled. Its
        # range spans NVIDIA's measured agentic-replaceability band (0.40-0.70,
        # belcak2025) up through this study's asserted 0.82 to an optimistic
        # 0.90, so a reader who rejects our mix can read their own answer off
        # the bar rather than recompute the paper.
        ("Sufficient fraction f_s (workload mix)", "f_s",
         D.WORKLOAD_SUFFICIENT_FRACTION),
        ("Sufficiency threshold alpha", "alpha", D.ALPHA_SUFFICIENCY),
        ("Jevons rebound", "rebound", D.REBOUND_FACTOR),
        ("Edge GPU power", "gen_p", D.RTX4090_GEN_POWER),
        ("Edge throughput", "tps", D.RTX4090_TPS_8B),
        ("Grid carbon intensity", "ci", D.CI_GLOBAL),
        ("Embodied attribution", "attrib", D.EMBODIED_ATTRIB_FRACTION),
    ]
    rows = []
    for label, key, p in factors:
        lo = _carbon_savings_point(**{key: p.lo})
        hi = _carbon_savings_point(**{key: p.hi})
        rows.append((label, min(lo, hi), max(lo, hi)))
    rows.sort(key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(rows))

    lo_all = min(r[1] for r in rows) * 100
    hi_all = max(r[2] for r in rows) * 100
    # matplotlib autoscales to the bars and ignores annotations, so the end
    # labels fell outside the axes and the widest bar was clipped at the left
    # edge with its low value invisible. Set the limits explicitly, with room
    # for the labels on both sides.
    pad = max((hi_all - lo_all) * 0.14, 1.2)
    ax.set_xlim(lo_all - pad, hi_all + pad)

    for yi, (label, lo, hi) in zip(y, rows):
        span = (hi - lo) * 100
        if span < 0.15:
            # A parameter whose whole credible range moves the result by less
            # than a rounding step draws as a zero-width bar, which reads as a
            # rendering fault rather than as "this one does not matter".
            ax.plot([lo * 100], [yi], marker="D", ms=5, color=EDGE,
                    alpha=0.85, zorder=3)
            ax.text(hi * 100 + 0.5, yi, "no material effect", va="center",
                    ha="left", fontsize=7.6, color=MUTED, style="italic")
            continue
        ax.barh(yi, span, left=lo * 100, height=0.5, color=EDGE, alpha=0.85,
                zorder=3)
        ax.text(lo * 100 - 0.35, yi, f"{lo*100:.0f}", va="center", ha="right",
                fontsize=8, color=MUTED)
        ax.text(hi * 100 + 0.35, yi, f"{hi*100:.0f}", va="center", ha="left",
                fontsize=8, color=MUTED)

    ax.axvline(base * 100, color=ZERO, ls="--", lw=1.4, zorder=4)
    ax.annotate(f"base case {base*100:.0f}%",
                xy=(base * 100, len(rows) - 0.45),
                xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", color=ZERO, fontsize=8.4,
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GRID,
                          lw=0.8))
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.15)
    ax.set_xlabel("Carbon savings vs GPT-4o long (%)")
    ax.set_title("Sensitivity: what moves the result")
    _despine(ax, left=False); ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.03,
             "Each bar spans the parameter's credible range (others held central). "
             "The workload mix dominates everything else and is the one input here "
             "with no source: its low end is NVIDIA's measured replaceability "
             "floor, its centre this study's own assertion. The result stays "
             "positive across every range shown, but its magnitude ranges from 27 "
             "to 60 percent and is not robust to that assumption.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_tornado")


def _short_machine(name: str) -> str:
    """Part name without the vendor boilerplate the driver reports."""
    t = str(name).replace("(TM)", "").replace("(R)", "")
    for junk in ("NVIDIA GeForce ", "NVIDIA ", "AMD ", " Graphics"):
        t = t.replace(junk, "")
    t = t.strip()
    return t if t.startswith(("RTX", "Radeon")) else "Radeon " + t


def fig_bandwidth_roof():
    """Compare file-size proxies against each machine's own specification.

    One panel per machine, because the ceiling is a property of the part: a
    percentage of 256 GB/s and a percentage of 672 GB/s are only comparable
    once each is normalised to its own specification, and pooling them on one
    axis invites exactly the confusion the normalisation exists to prevent.
    """
    groups = MEAS.bandwidth_law().get("by_machine", {})
    if not groups:
        return

    # Marker and colour both encode the decode class, so the figure survives
    # greyscale printing - the same reason the verdict table carries symbols.
    STYLE = {
        "dense":   (EDGE_D, "o", "Dense (streams every weight)"),
        "mtp":     (HEAVY, "s", "Multi-token prediction"),
        "elastic": (ACCENT, "D", "Per-layer embeddings"),
        "sparse":  (CLOUD, "^", "Sparse mixture of experts"),
    }
    # Nine labels on two panels cannot be auto-placed without collisions:
    # three AMD points sit within 0.5 GB of each other and differ only in
    # height. Offsets are in points, keyed by machine and model.
    NUDGE = {
        ("Radeon 8060S", "ministral-3:3b"): (-3, 9, "right"),
        ("Radeon 8060S", "ministral-3:8b"): (14, -6, "left"),
        ("Radeon 8060S", "qwen3.8:27b"): (9, -3, "left"),
        ("Radeon 8060S", "muse-glimmer:30b"): (0, -16, "center"),
        ("Radeon 8060S", "internal 3.1B (Q8-0)"): (-6, -17, "center"),
        ("Radeon 8060S", "internal 7.6B VLM (Q8-0)"): (0, 12, "center"),
        ("RTX 5070", "qwen3:4b"): (0, -16, "center"),
        ("RTX 5070", "ministral-3:8b"): (-7, -14, "right"),
    }

    fig, axes = plt.subplots(1, len(groups), figsize=(10, 3.9), squeeze=False)
    seen = {}
    for col, (ax, (machine, group)) in enumerate(zip(axes[0], groups.items())):
        short = _short_machine(machine)
        rows = sorted(group["rows"], key=lambda r: r["weights_gb"])
        peak = rows[0]["peak_bandwidth_gb_s"]
        top = max(125, max(r["utilisation_pct"] for r in rows) * 1.30)
        right = max(r["weights_gb"] for r in rows) * 1.30

        ax.axhspan(0, 100, color=EDGE, alpha=0.06, zorder=0)
        ax.axhline(100, color=ZERO, ls="--", lw=1.3, zorder=3)
        # The legend occupies the first panel top-left, so the ceiling label
        # goes bottom-right there and top-left everywhere else. Both are
        # regions the data cannot reach: nothing sits at the far edge.
        if col == 0:
            ax.annotate("specified bandwidth", xy=(right, 100),
                        xytext=(-3, -5), textcoords="offset points",
                        ha="right", va="top", fontsize=8, color=ZERO,
                        zorder=6)
        else:
            ax.annotate("specified bandwidth", xy=(0, 100), xytext=(4, 5),
                        textcoords="offset points", ha="left",
                        va="bottom", fontsize=8, color=ZERO, zorder=6)

        for r in rows:
            colour, marker, label = STYLE.get(r["decode_class"],
                                              (MUTED, "o", "other"))
            # Builds whose weights are not redistributable draw hollow. They
            # are shown, not filtered: one of them is the dense row above the
            # ceiling that bounds this proxy's error, and an earlier version of
            # this figure omitted both and reported a tighter law than the
            # data supports.
            if r.get("redistributable", True):
                h = ax.scatter(r["weights_gb"], r["utilisation_pct"],
                               color=colour, marker=marker, s=52, zorder=5)
                seen.setdefault(label, h)
            else:
                h = ax.scatter(r["weights_gb"], r["utilisation_pct"],
                               facecolors="none", edgecolors=colour,
                               marker=marker, s=58, linewidths=1.4, zorder=5)
                seen.setdefault("Weights not redistributable", h)
            # Builds whose weights are not redistributable are internal
            # deployment names. They carry no meaning for a reader and
            # should not appear in a published figure, so they are
            # labelled by the only properties that matter here: size,
            # quantisation and whether a vision tower inflates the
            # weight file. The prose describes them the same way.
            if not r.get("redistributable", True):
                params = r.get("parameters") or "?"
                quant = (r.get("quantization") or "").replace("_", "-")
                tower = " VLM" if r.get("multimodal_tower") else ""
                name = f"internal {params}{tower} ({quant})"
            else:
                name = r["model"].replace(":latest", "")
            dx, dy, ha = NUDGE.get((short, name), (0, 9, "center"))
            ax.annotate(name, (r["weights_gb"], r["utilisation_pct"]),
                        xytext=(dx, dy), textcoords="offset points",
                        ha=ha, fontsize=7.4, color=INK, zorder=6)

        ax.set_title(f"{short}  —  {peak:.0f} GB/s specified")
        ax.set_xlabel("On-disk model size (GB)")
        if col == 0:
            ax.set_ylabel(r"$\tau \cdot W$ / specified bandwidth (%)")
        ax.set_xlim(0, right)
        ax.set_ylim(0, top)
        _despine(ax)

    order = [lbl for lbl in
             ("Dense (streams every weight)", "Multi-token prediction",
              "Per-layer embeddings", "Sparse mixture of experts")
             if lbl in seen]
    axes[0][0].legend([seen[l] for l in order], order, frameon=False,
                      fontsize=7.6, loc="upper left", handletextpad=0.3,
                      borderaxespad=0.2)

    fig.tight_layout()
    fig.text(.5, -.02,
             "Each machine is normalised to its own specification. Points below "
             "the line stream the whole weight file per token. Most points above it "
             "read only part of their file by design. One does not: the hollow "
             "dense point at 105.7% cannot exceed a ceiling it must stream "
             "under, so its excess measures the error in this proxy. On-disk "
             "size stands in for per-token traffic; it is not a measurement of "
             "it, and hollow markers are builds whose weights are not "
             "redistributable - shown, not filtered.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_bandwidth_roof")


def fig_measured_energy():
    """Primary run beside its replication, so agreement is read at a glance."""
    rows = [r for r in MEAS.energy_table() if r["file"] in
            ("rtx5070-study.json", "rtx5070-replication.json")]
    if not rows:
        return
    # Order matters here: the figure exists to show that the replication
    # reproduces the primary, which only reads as such if the two sit together.
    def _is_repeat(r):
        return "replication" in r["file"]
    rows.sort(key=lambda r: (r["model"], _is_repeat(r)))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, row in enumerate(rows):
        e = row["energy_wh"]
        ax.bar(i, e, width=.6, zorder=3,
               color=ACCENT if _is_repeat(row) else EDGE_D)
        ax.errorbar(i, e, yerr=[[e - row["energy_wh_min"]],
                                [row["energy_wh_max"] - e]],
                    fmt="none", color=INK, capsize=4, zorder=4)
        ax.text(i, row["energy_wh_max"] + .006, f"{e:.2g}", ha="center",
                fontsize=9, color=INK, zorder=5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["model"].replace("ministral-3:", "") +
                        ("\nreplication" if _is_repeat(r) else "\nprimary")
                        for r in rows])
    ax.set_ylim(0, .25)
    ax.set_ylabel("GPU board energy per request (Wh)")
    ax.set_title("RTX 5070: 499 input / 300 output tokens")
    _despine(ax)
    ax.grid(axis="x", visible=False)
    fig.text(.5, -.04,
             "Seven repetitions per model per run; bars are medians, whiskers "
             "are observed minima and maxima. Each model's independent "
             "replication is shown beside its primary run. Request window only; "
             "host draw and PSU loss are excluded here and charged separately. "
             "Two significant figures: the board counter refreshes about "
             "every 0.4 s, giving 4-5 distinct readings per 3B request.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_measured_energy")


if __name__ == "__main__":
    fig_energy_ladder()
    fig_capability()
    fig_savings_by_baseline()
    fig_footprint_breakdown()
    fig_grid_crossover()
    fig_tco()
    fig_fleet()
    fig_tornado()
    fig_bandwidth_roof()
    fig_measured_energy()
    print("All figures written to", os.path.abspath(OUT))
