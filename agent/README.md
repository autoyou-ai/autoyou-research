# autoyou_agent - edge-first, capability-routing AI agent

A reference prototype that **runs the paper's decision rule at runtime**: prefer a
local Small Language Model (SLM) on already-owned hardware, fall back to a cloud
frontier model only when the task needs it, and annotate every answer with its
estimated energy / water / carbon footprint.

It demonstrates the thesis of
*"Decentralizing AI Inference to the Edge over Peer-to-Peer Transport"*
(see `../paper/main.pdf`) - and it reuses the **same validated models**
(`../models`) so the runtime numbers and the paper never diverge.

## What it shows

Running `demo.py` over a 10-query personal-assistant workload (no configuration
required - simulation mode, or live if Ollama/keys are present):

```
routed to edge        : 80% of queries (paper f_s = 82%)
system vs all-cloud   : roughly 70% lower energy/water/carbon
```

The two hard-reasoning queries correctly fall back to the cloud, matching the
paper's finding that the "~70%-capable" claim holds for extraction / RAG-QA /
summary / simple-code but **not** for hard reasoning.

## The decision rule (from the paper)

Route a query to the local SLM **iff all** hold; otherwise use a cloud frontier model:

1. **Capability sufficient** - the task class's SLM/frontier capability ratio
   `ρ ≥ α` (default `α = 0.70`).  *(dominant factor)*
2. **Edge available** - an already-owned device is present and reachable.
3. **Grid below routing crossover** - the local grid carbon intensity is low
   enough that edge emits less than the displaced cloud model in its region.
   Against a *frontier* baseline this almost never binds; against an *efficient*
   cloud baseline (e.g. Gemini) a coal-heavy grid will correctly flip it.

Condition (ii) of the paper - that the *displaced* baseline is a frontier model -
is the counterfactual savings are measured against.

## Architecture

```
classify(query) ──► decide_route() ──► edge (Ollama SLM)   ─┐
                          │                                  ├─► footprint accounting
                          └────────────► cloud (OpenAI/      ┘    (../models)
                                          Anthropic/Google)
RAG (WebRAG) grounds rag_qa tasks before generation.
```

| File | Role |
|------|------|
| `agent.py` | capability router + `EdgeFirstAgent` orchestration + RAG hook |
| `providers.py` | Ollama (edge) + OpenAI/Anthropic/Google (cloud) + Mock (sim); lazy, no hard deps; per-tier failover |
| `footprint.py` | per-query energy/water/carbon via the paper's models; routing crossover |
| `demo.py` | scripted workload, prints routing + aggregate savings |

## Run

```bash
# zero-config (simulation, or live if available):
../.venv/bin/python demo.py

# live local SLM:
ollama pull ministral-3:8b   # or ministral-3:3b; any pulled tag is auto-resolved
ollama serve

# live cloud fallback:
export OPENAI_API_KEY=...        # and/or ANTHROPIC_API_KEY / GOOGLE_API_KEY

# explore the honest boundary (efficient baseline + dirty grid flips edge->cloud):
AUTOYOU_CLOUD_BASELINE=gemini AUTOYOU_GRID_CI=708 ../.venv/bin/python demo.py
```

`AUTOYOU_GRID_CI` sets the local grid carbon intensity (gCO₂e/kWh);
`AUTOYOU_CLOUD_BASELINE` ∈ {`gpt-4o`, `gpt-4o-mini`, `claude-3.7-sonnet`,
`gemini`, `o3`} sets the displaced cloud model.

## Honesty notes (carried from the paper)

- Edge does **not** win on raw per-token efficiency (a single-user consumer GPU
  is ~2.4× less efficient per token than a batched data-centre accelerator). It
  wins by **right-sizing**, eliminating **evaporative cooling water** and
  **PUE/idle overhead**, and using **marginal energy** on existing hardware.
- Footprint figures assume an **already-owned, well-utilized** device (marginal
  embodied carbon). A device bought solely for inference and lightly used can be
  carbon-negative-value; the router does not claim savings there.
- All transport in the reference architecture is WebRTC SCTP-over-DTLS, so any
  STUN/TURN relay is a blind ciphertext forwarder. This is a transport claim
  against relays; paired endpoints remain trusted application endpoints.

## Wiring into AutoYou

`providers.OllamaProvider` mirrors AutoYou's local-first Ollama/LiteLLM path;
the cloud chain mirrors its multi-provider fallback. `WebRAG(search_fn=...)`
accepts any retrieval callable (e.g. an `internet_agent`-style web search). The
router is transport-agnostic and can sit behind the same datachannel contract.
