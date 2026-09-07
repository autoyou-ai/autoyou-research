# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
demo.py - runs autoyou_agent over a representative personal-assistant workload
and prints the routing decisions + per-query and aggregate footprint savings.

Runs with no configuration (simulation mode). To exercise real back-ends:
  * local SLM:  `ollama pull ministral-3:8b` then `ollama serve`
  * cloud:      set OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY
Set AUTOYOU_GRID_CI to test grid-dependent routing (e.g. 708 for India).
"""

import os

import agent as A


WORKLOAD = [
    "Extract the invoice number and total from this email body.",
    "Summarize this 3-paragraph article into two sentences.",
    "What is the current population of Tokyo according to recent sources?",
    "Write a regex that matches an IPv4 address.",
    "Classify this support ticket as billing, bug, or feature request.",
    "Rephrase this paragraph to be more formal.",
    "Prove that the sum of the first n odd numbers equals n^2, step by step.",
    "Who won the latest Formula 1 race and by how much?",
    "Derive the time complexity of merge sort and explain each step.",
    "Give me a bash script to back up a folder to S3.",
]


def main():
    grid = float(os.environ.get("AUTOYOU_GRID_CI", A.D.CI_GLOBAL.value))
    baseline = os.environ.get("AUTOYOU_CLOUD_BASELINE", "gpt-4o")
    agent = A.EdgeFirstAgent(grid_ci=grid, cloud_baseline_model=baseline)
    edge_live = agent.edge.available() and "mock" not in agent.edge.__class__.__name__.lower()
    cloud_live = "mock" not in agent.cloud_chain[0].__class__.__name__.lower()

    print("=" * 78)
    print("autoyou_agent demo - edge-first capability routing")
    print(f"  edge provider : {agent.edge.name}"
          f"  ({'LIVE' if edge_live else 'SIMULATED'})")
    print(f"  cloud chain   : {agent.cloud_chain[0].name}"
          f"  ({'LIVE' if cloud_live else 'SIMULATED'})")
    print(f"  local grid CI : {grid:.0f} gCO2e/kWh"
          f"  (routing crossover vs {baseline}: "
          f"{A.FP.routing_crossover_gco2_per_kwh(baseline):.0f} gCO2e/kWh)")
    print("=" * 78)

    answers = agent.batch(WORKLOAD)
    hdr = f"{'tier':5s} {'class':12s} {'energy':>7s} {'water':>7s} {'carbon':>7s}  query"
    print(hdr); print("-" * 78)
    for a in answers:
        fp = a.footprint
        tier_mark = "EDGE " if a.route.tier == "edge" else "cloud"
        print(f"{tier_mark} {a.route.task_class:12s} "
              f"{fp['energy_wh']:6.3f}W {fp['water_ml']:5.2f}mL {fp['carbon_g']:5.3f}g  "
              f"{a.query[:34]}")

    agg = A.aggregate(answers)
    print("-" * 78)
    print(f"routed to edge        : {agg['edge_fraction']*100:.0f}% of queries "
          f"(paper f_s = 82%)")
    print(f"system vs all-cloud   : "
          f"energy -{agg['savings_energy']*100:.0f}%  "
          f"water -{agg['savings_water']*100:.0f}%  "
          f"carbon -{agg['savings_carbon']*100:.0f}%")
    print(f"absolute carbon       : {agg['system_carbon_g']:.2f} g vs "
          f"{agg['baseline_carbon_g']:.2f} g baseline "
          f"({agg['queries']} queries)")
    print("=" * 78)
    print("Footprint numbers come from the paper's validated models "
          "(models/).")
    print("Try:  AUTOYOU_CLOUD_BASELINE=gemini AUTOYOU_GRID_CI=708 python demo.py")
    print("      -> an efficient cloud baseline + a coal-heavy grid flips some "
          "edge->cloud.")


if __name__ == "__main__":
    main()
