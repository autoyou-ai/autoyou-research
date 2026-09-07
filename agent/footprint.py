# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
footprint.py - per-query environmental accounting for the agent, reusing the
paper's validated models (models/). Every tier decision is annotated
with the estimated energy/water/carbon so the agent *demonstrates* the thesis at
runtime, not just in the paper.
"""

import os
import sys

# Make the paper's models importable without packaging.
_MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

import data as D          # noqa: E402
import models as M        # noqa: E402

# Cloud per-query energy (Wh) by model id used in fallback.
CLOUD_ENERGY_WH = {
    "gpt-4o": D.E_GPT4O.value,
    "gpt-4o-mini": D.E_GPT4O_MINI.value,
    "claude-3.7-sonnet": D.E_CLAUDE37_SONNET.value,
    "gemini": D.E_GEMINI_MEDIAN.value,
    "gemini-2.5-flash-lite": D.E_GEMINI_MEDIAN.value,
    "gemini-2.5-flash": D.E_GEMINI_MEDIAN.value,
    "o3": D.E_O3.value,
}


def edge_query_footprint(tokens_out: int = 300, grid_ci: float = None,
                         marginal: bool = True):
    """Footprint of an edge SLM (8B) query. marginal=True assumes an
    already-powered device (the regime under which the thesis holds)."""
    ci = D.CI_GLOBAL.value if grid_ci is None else grid_ci
    return M.edge_slm_query_central(tokens_out=tokens_out, marginal=marginal, ci=ci)


def cloud_query_footprint(model_id: str = "gpt-4o", region_ci: float = None):
    """Footprint of a cloud query. The cloud model runs in ITS OWN data-centre
    region, so carbon uses the cloud region grid (default global average), NOT
    the user's local grid."""
    e = CLOUD_ENERGY_WH.get(model_id, D.E_GPT4O.value)
    ci = D.CI_GLOBAL.value if region_ci is None else region_ci

    class _P:
        value = e
    return M.cloud_query_central(_P, hyperscale=True, ci=ci)


def savings_vs_cloud(edge_fp, cloud_fp):
    def frac(c, e):
        return (c - e) / c if c else 0.0
    return {
        "energy": frac(cloud_fp.energy_wh, edge_fp.energy_wh),
        "water": frac(cloud_fp.water_ml, edge_fp.water_ml),
        "carbon": frac(cloud_fp.carbon_g, edge_fp.carbon_g),
    }


def carbon_crossover_gco2_per_kwh(tokens_out: int = 300):
    """Local grid CI above which an edge query emits more carbon than a clean
    EU cloud region running the SAME 8B model - the H4 caveat boundary from the
    paper (Fig. grid crossover). This is the *narrow* same-model comparison."""
    e_edge = edge_query_footprint(tokens_out).energy_wh
    emb = M.embodied_per_query_g(D.EMBODIED_EDGE_KG.value,
                                 D.EDGE_DEVICE_LIFETIME_QUERIES.value,
                                 D.EMBODIED_ATTRIB_FRACTION.value)
    cloud_clean = D.E_LLAMA31_8B.value / 1000.0 * D.CI_EU.value
    return max(0.0, (cloud_clean - emb) / (e_edge / 1000.0))


def routing_crossover_gco2_per_kwh(cloud_model_id: str = "gpt-4o",
                                   cloud_region_ci: float = None,
                                   tokens_out: int = 300):
    """Local grid CI above which routing to the edge SLM would emit MORE carbon
    than the displaced CLOUD baseline model in its region. This is the
    routing-relevant boundary: because a frontier model uses 3-65x more energy,
    this crossover is very high (edge wins on nearly all real grids); it only
    becomes binding when the cloud baseline is itself an efficient small model."""
    e_edge = edge_query_footprint(tokens_out).energy_wh
    emb = M.embodied_per_query_g(D.EMBODIED_EDGE_KG.value,
                                 D.EDGE_DEVICE_LIFETIME_QUERIES.value,
                                 D.EMBODIED_ATTRIB_FRACTION.value)
    region_ci = D.CI_GLOBAL.value if cloud_region_ci is None else cloud_region_ci
    e_cloud = CLOUD_ENERGY_WH.get(cloud_model_id, D.E_GPT4O.value)
    cloud_c = e_cloud / 1000.0 * region_ci
    return max(0.0, (cloud_c - emb) / (e_edge / 1000.0))
