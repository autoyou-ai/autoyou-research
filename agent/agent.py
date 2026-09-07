# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
agent.py - autoyou_agent: an edge-first, capability-routing AI agent.

It implements the paper's decision rule directly:
  route a query to a LOCAL SLM when (i) the task is capability-sufficient for an
  SLM, (iii) an edge device is available, and (iv) the local grid is below the
  carbon crossover; otherwise fall back to a cloud frontier provider. (Condition
  (ii) - that the displaced baseline is a frontier model - is the counterfactual
  we measure savings against.)

Every answer is annotated with which tier served it and the estimated
energy/water/carbon, so the agent demonstrates the thesis at runtime.

Runs with zero configuration in simulation mode (MockProviders); uses real
Ollama / OpenAI / Anthropic / Google back-ends when available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

import footprint as FP
import providers as PV
import data as D            # via footprint's sys.path insert


# Capability ratios per task class (SLM-8B / frontier), from data.py.
CAP_RATIOS: Dict[str, float] = {
    "extraction": D.CAP_RATIO_EXTRACTION.value,
    "rag_qa": D.CAP_RATIO_RAG_QA.value,
    "summary": D.CAP_RATIO_SUMMARY.value,
    "simple_code": D.CAP_RATIO_SIMPLE_CODE.value,
    "hard_reason": D.CAP_RATIO_HARD_REASON.value,
}


# --------------------------------------------------------------------------- #
#  Capability router
# --------------------------------------------------------------------------- #

_PATTERNS = [
    ("simple_code", re.compile(r"\b(code|function|regex|script|sql|bash|json|yaml)\b", re.I)),
    ("summary", re.compile(r"\b(summari[sz]e|summari[sz]ation|summary|tl;?dr|"
                           r"rewrite|rephrase|shorten|paraphrase)\b", re.I)),
    ("extraction", re.compile(r"\b(extract|classify|label|parse|route|categor|tag)\b", re.I)),
    ("hard_reason", re.compile(r"\b(prove|derive|theorem|integral|optimi[sz]e|"
                               r"step.?by.?step|complexity|why does|multi.?step)\b", re.I)),
    ("rag_qa", re.compile(r"\b(who|what|when|where|latest|current|today|price|news|"
                          r"according to|cite|source)\b", re.I)),
]


def classify(query: str, classifier: Optional[Callable[[str], str]] = None) -> str:
    """Map a query to a task class. Heuristic by default; pluggable with an SLM
    classifier for production (classifier(query) -> class name)."""
    if classifier is not None:
        cls = classifier(query)
        if cls in CAP_RATIOS:
            return cls
    # math/hard-reason signal: equations or "solve ... for"
    if re.search(r"[=∫∑]|\bsolve\b.*\bfor\b|\d+\s*[\^]\s*\d+", query):
        return "hard_reason"
    for name, pat in _PATTERNS:
        if pat.search(query):
            return name
    return "rag_qa"          # default to grounded QA


@dataclass
class Route:
    task_class: str
    tier: str                # "edge" | "cloud"
    rationale: str
    cap_ratio: float
    sufficient: bool


def decide_route(task_class: str, edge_available: bool, grid_ci: float,
                 cloud_baseline_model: str = "gpt-4o", alpha: float = None) -> Route:
    """Implements the paper's decision rule, in order of dominance:
    (i) capability sufficiency, (iii) edge availability, (iv) local grid below
    the routing crossover vs the *displaced cloud baseline*."""
    alpha = D.ALPHA_SUFFICIENCY.value if alpha is None else alpha
    rho = CAP_RATIOS.get(task_class, 0.0)
    sufficient = rho >= alpha
    crossover = FP.routing_crossover_gco2_per_kwh(cloud_baseline_model)
    grid_ok = grid_ci <= crossover

    if not sufficient:
        return Route(task_class, "cloud",
                     f"capability rho={rho:.2f} < alpha={alpha:.2f}: route to frontier",
                     rho, sufficient)
    if not edge_available:
        return Route(task_class, "cloud",
                     "no edge device available: route to cloud", rho, sufficient)
    if not grid_ok:
        return Route(task_class, "cloud",
                     f"local grid {grid_ci:.0f} > routing crossover "
                     f"{crossover:.0f} g/kWh vs {cloud_baseline_model}: cloud "
                     "is cleaner here", rho, sufficient)
    return Route(task_class, "edge",
                 f"sufficient (rho={rho:.2f} >= alpha), edge available, grid {grid_ci:.0f} "
                 f"<= {crossover:.0f}: run local SLM", rho, sufficient)


# --------------------------------------------------------------------------- #
#  RAG (web/grounded retrieval) - pluggable; offline-safe.
# --------------------------------------------------------------------------- #

class WebRAG:
    """Retrieval shim. Supply search_fn(query)->List[str] to wire a real web
    search (e.g., AutoYou's internet_agent). Offline, returns a placeholder so
    the agent still runs."""

    def __init__(self, search_fn: Optional[Callable[[str], List[str]]] = None,
                 k: int = 3):
        self.search_fn = search_fn
        self.k = k

    def retrieve(self, query: str) -> List[str]:
        if self.search_fn is None:
            return []
        try:
            return list(self.search_fn(query))[: self.k]
        except Exception:
            return []


# --------------------------------------------------------------------------- #
#  Agent
# --------------------------------------------------------------------------- #

@dataclass
class Answer:
    query: str
    text: str
    route: Route
    provider: str
    model_id: str
    footprint: Dict[str, float]
    cloud_counterfactual: Dict[str, float]
    savings_vs_cloud: Dict[str, float]


class EdgeFirstAgent:
    def __init__(self, edge: PV.Provider = None, cloud: List[PV.Provider] = None,
                 rag: WebRAG = None, grid_ci: float = None,
                 system_prompt: str = "You are a concise, helpful assistant.",
                 cloud_baseline_model: str = "gpt-4o", cloud_region_ci: float = None):
        self.edge = edge or self._pick_edge()
        self.cloud_chain = cloud or self._pick_cloud()
        self.rag = rag or WebRAG()
        self.grid_ci = D.CI_GLOBAL.value if grid_ci is None else grid_ci
        # cloud DC region grid (default global avg); independent of the user's grid
        self.cloud_region_ci = D.CI_GLOBAL.value if cloud_region_ci is None else cloud_region_ci
        self.system_prompt = system_prompt
        self.cloud_baseline_model = cloud_baseline_model

    @staticmethod
    def _pick_edge() -> PV.Provider:
        o = PV.OllamaProvider()
        return o if o.available() else PV.MockProvider(tier="edge")

    @staticmethod
    def _pick_cloud() -> List[PV.Provider]:
        chain = [PV.OpenAIProvider(), PV.AnthropicProvider(), PV.GoogleProvider()]
        live = [p for p in chain if p.available()]
        return live or [PV.MockProvider(tier="cloud", model="gpt-4o-sim")]

    def _cloud_provider(self) -> PV.Provider:
        return self.cloud_chain[0]

    def answer(self, query: str, classifier=None) -> Answer:
        task_class = classify(query, classifier)
        edge_avail = self.edge.available()
        route = decide_route(task_class, edge_avail, self.grid_ci,
                             cloud_baseline_model=self.cloud_baseline_model)

        messages = [{"role": "system", "content": self.system_prompt}]
        if task_class == "rag_qa":
            ctx = self.rag.retrieve(query)
            if ctx:
                messages.append({"role": "system",
                                 "content": "Context:\n" + "\n".join(ctx)})
        messages.append({"role": "user", "content": query})

        provider = self.edge if route.tier == "edge" else self._cloud_provider()
        try:
            gen = provider.generate(messages)
        except Exception as exc:        # provider failover -> keep serving
            fb = PV.MockProvider(tier=route.tier,
                                 model=getattr(provider, "model", "sim") + "-sim")
            gen = fb.generate(messages)
            route.rationale += f"  [failover: {provider.name} error: {exc}]"

        # footprint of the path actually taken (edge uses LOCAL grid; cloud uses
        # its data-centre REGION grid, default global average).
        if route.tier == "edge":
            fp = FP.edge_query_footprint(tokens_out=gen.tokens_out,
                                         grid_ci=self.grid_ci, marginal=True)
        else:
            model_id = (self.cloud_baseline_model if "sim" in gen.model_id
                        else gen.model_id)
            fp = FP.cloud_query_footprint(model_id=model_id,
                                          region_ci=self.cloud_region_ci)

        # counterfactual: what the all-cloud-frontier baseline would have cost
        cf = FP.cloud_query_footprint(model_id=self.cloud_baseline_model,
                                      region_ci=self.cloud_region_ci)
        savings = FP.savings_vs_cloud(fp, cf)

        return Answer(query=query, text=gen.text, route=route,
                      provider=gen.provider, model_id=gen.model_id,
                      footprint=fp.as_dict(), cloud_counterfactual=cf.as_dict(),
                      savings_vs_cloud=savings)

    def batch(self, queries: List[str]) -> List[Answer]:
        return [self.answer(q) for q in queries]


def aggregate(answers: List[Answer]) -> Dict:
    """Empirical system-level result from actual routing decisions."""
    n = len(answers)
    edge_n = sum(1 for a in answers if a.route.tier == "edge")
    tot_e = sum(a.footprint["energy_wh"] for a in answers)
    tot_w = sum(a.footprint["water_ml"] for a in answers)
    tot_c = sum(a.footprint["carbon_g"] for a in answers)
    base_e = sum(a.cloud_counterfactual["energy_wh"] for a in answers)
    base_w = sum(a.cloud_counterfactual["water_ml"] for a in answers)
    base_c = sum(a.cloud_counterfactual["carbon_g"] for a in answers)
    return {
        "queries": n,
        "edge_fraction": edge_n / n if n else 0.0,
        "system_energy_wh": tot_e, "baseline_energy_wh": base_e,
        "system_water_ml": tot_w, "baseline_water_ml": base_w,
        "system_carbon_g": tot_c, "baseline_carbon_g": base_c,
        "savings_energy": (base_e - tot_e) / base_e if base_e else 0.0,
        "savings_water": (base_w - tot_w) / base_w if base_w else 0.0,
        "savings_carbon": (base_c - tot_c) / base_c if base_c else 0.0,
    }
