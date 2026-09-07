# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
providers.py - pluggable inference back-ends for autoyou_agent.

Design goals:
  * Local-first: an Ollama SLM (Ministral/Llama 3--8B) is the preferred tier.
  * Multi-provider cloud fallback: OpenAI, Anthropic, Google.
  * Zero hard dependencies: every adapter lazy-imports its SDK and degrades
    gracefully. If nothing is installed/configured, a deterministic MockProvider
    keeps the whole agent runnable and verifiable (sim mode).

Each provider returns a Generation(text, tokens_out, model_id, tier).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Generation:
    text: str
    tokens_out: int
    model_id: str
    tier: str            # "edge" | "cloud" | "mock"
    provider: str


class Provider:
    name = "base"
    tier = "cloud"

    def available(self) -> bool:
        return False

    def generate(self, messages: List[Dict[str, str]], **kw) -> Generation:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class OllamaProvider(Provider):
    """Local SLM via Ollama (the edge tier). Default ministral-3:8b."""
    name = "ollama"
    tier = "edge"

    def __init__(self, model: str = "ministral-3:8b", host: str = None):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    def available(self) -> bool:
        """True only if the Ollama server is up AND a matching model is actually
        pulled. Resolves the concrete pulled tag (e.g. ministral-3:3b) so the
        agent uses what is really present rather than erroring on a missing tag."""
        try:
            import json
            import urllib.request
            with urllib.request.urlopen(self.host + "/api/tags", timeout=0.6) as r:
                if r.status != 200:
                    return False
                tags = json.loads(r.read())
            names = [m.get("name", "") for m in tags.get("models", [])]
            if not names:
                return False
            base = self.model.split(":")[0]
            if self.model in names:                       # exact tag present
                return True
            same_base = [n for n in names if n.split(":")[0] == base]
            if same_base:                                  # resolve to a real tag
                self.model = same_base[0]
                return True
            return False
        except Exception:
            return False

    def generate(self, messages, **kw) -> Generation:
        import json
        import urllib.request
        body = json.dumps({"model": self.model, "messages": messages,
                           "stream": False}).encode()
        req = urllib.request.Request(self.host + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        text = data.get("message", {}).get("content", "")
        tok = data.get("eval_count") or max(1, len(text.split()))
        return Generation(text, int(tok), self.model, "edge", "ollama")


# --------------------------------------------------------------------------- #
class OpenAIProvider(Provider):
    name = "openai"
    tier = "cloud"

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa
            return True
        except Exception:
            return False

    def generate(self, messages, **kw) -> Generation:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(model=self.model, messages=messages)
        text = resp.choices[0].message.content
        tok = getattr(resp.usage, "completion_tokens", None) or len(text.split())
        return Generation(text, int(tok), self.model, "cloud", "openai")


# --------------------------------------------------------------------------- #
class AnthropicProvider(Provider):
    name = "anthropic"
    tier = "cloud"

    def __init__(self, model: str = "claude-3.7-sonnet"):
        self.model = model

    def available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa
            return True
        except Exception:
            return False

    def generate(self, messages, **kw) -> Generation:
        import anthropic
        client = anthropic.Anthropic()
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        turns = [m for m in messages if m["role"] != "system"]
        resp = client.messages.create(model=self.model, max_tokens=1024,
                                       system=sys_msg, messages=turns)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        tok = getattr(resp.usage, "output_tokens", None) or len(text.split())
        return Generation(text, int(tok), self.model, "cloud", "anthropic")


# --------------------------------------------------------------------------- #
class GoogleProvider(Provider):
    name = "google"
    tier = "cloud"

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model = model

    def available(self) -> bool:
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            return False
        try:
            import google.generativeai  # noqa
            return True
        except Exception:
            return False

    def generate(self, messages, **kw) -> Generation:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY")
                        or os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(self.model)
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        resp = model.generate_content(prompt)
        text = resp.text
        return Generation(text, len(text.split()), self.model, "cloud", "google")


# --------------------------------------------------------------------------- #
class MockProvider(Provider):
    """Deterministic, dependency-free stand-in so the agent is always runnable.
    Used for both an 'edge' and a 'cloud' persona in simulation mode."""
    def __init__(self, tier: str = "edge", model: str = "ministral-3:8b-sim"):
        self.name = f"mock-{tier}"
        self.tier = tier
        self.model = model

    def available(self) -> bool:
        return True

    def generate(self, messages, **kw) -> Generation:
        user = next((m["content"] for m in reversed(messages)
                     if m["role"] == "user"), "")
        prefix = ("[edge SLM] " if self.tier == "edge" else "[cloud frontier] ")
        text = prefix + f"(simulated answer to: {user[:80]})"
        return Generation(text, max(1, len(user.split()) * 3), self.model,
                          self.tier, self.name)
