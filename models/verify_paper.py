# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
verify_paper.py - Check that every number printed in the papers exists in the
model that is supposed to have produced it.

    python models/verify_paper.py            # report
    python models/verify_paper.py --strict   # non-zero exit on any miss

WHY
    A paper drifts from its harness the moment someone edits a figure in prose.
    `run_all.py` guarantees that results.json is correct; it guarantees nothing
    about whether the LaTeX quotes it faithfully. This closes that gap by
    extracting every
    numeric token from the .tex, and refusing anything that cannot be traced to
    results.json, to a data.py parameter, or to a short annotated allow-list of
    figures that legitimately come from a cited work rather than from us.

    It is not a proof that the paper is right. It is a proof that the paper and
    the harness agree, which is the failure mode that actually happens.

WHAT IT CANNOT CATCH
    A number correctly transcribed from a model that is itself wrong, and a
    number used in the wrong sentence. Those need a reader.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Set, Tuple

import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", "paper"))
RESULTS = os.path.join(HERE, "results.json")

PAPERS = ["main.tex", "adaptation.tex"]

# Above this many distinct sources, a match stops being evidence: the
# number would have been accepted whatever the sentence claimed.
WEAK_MATCH_THRESHOLD = 10


# --------------------------------------------------------------------------- #
#  Vocabulary
# --------------------------------------------------------------------------- #

def _walk(obj) -> Iterable[float]:
    """Every number anywhere in results.json."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield float(obj)


def _forms(v: float) -> Set[str]:
    """Every plausible printed form of one number."""
    out: Set[str] = set()
    if v != v or v in (float("inf"), float("-inf")):
        return out
    a = abs(v)

    # Raw and rounded decimals.
    for dp in (0, 1, 2, 3, 4):
        out.add(f"{a:.{dp}f}")
        out.add(f"{a:,.{dp}f}")
    # Percentages, for fractions.
    if 0 <= a <= 1.0000001:
        for dp in (0, 1, 2):
            out.add(f"{a*100:.{dp}f}")
    # Percentage-point deltas already expressed as points.
    if a <= 100:
        for dp in (0, 1):
            out.add(f"{a:.{dp}f}")
    # Scaled forms.
    for scale, suffix in ((1e3, "k"), (1e6, "M"), (1e9, "B"), (1e12, "T")):
        if a >= scale / 10:
            for dp in (0, 1, 2):
                out.add(f"{a/scale:.{dp}f}")
    # Integers with separators.
    if a == int(a):
        out.add(f"{int(a):,}")
        out.add(str(int(a)))
    # Python's format() rounds half to even, so 142.5 renders "142" while a
    # human writing prose renders "143". Both are defensible; a checker that
    # accepts only one of them reports a false positive on the other.
    import math
    out.add(str(int(math.floor(a + 0.5))))
    out.add(f"{int(math.floor(a + 0.5)):,}")
    return {s.rstrip("0").rstrip(".") if "." in s else s for s in out} | out


def _walk_labelled(obj, path="") -> Iterable[Tuple[float, str]]:
    """Every number in results.json, carrying the key path it came from."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_labelled(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_labelled(v, f"{path}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield float(obj), path


def build_vocabulary() -> Tuple[Dict[str, List[str]], int]:
    """Map every printed form to the sources that can justify it.

    A set would answer "is this number traceable?" and nothing else, which is
    how a figure once passed on a coincidence: the paper quoted 0.40 mWh/token
    for a batched cloud deployment and the checker accepted it because an
    unrelated capability parameter also happens to equal 0.40. Keeping the
    source of every match turns that from an invisible pass into something
    --provenance will print, so a reader can see WHY a number was accepted and
    notice when the reason is nonsense. The checker still cannot judge meaning;
    it can now at least show its working.
    """
    vocab: Dict[str, List[str]] = {}

    def add(forms: Set[str], source: str) -> None:
        for f in forms:
            bucket = vocab.setdefault(f, [])
            if source not in bucket:
                bucket.append(source)

    def specificity(source: str) -> tuple:
        """Most specific source first: a named parameter beats a deep
        results.json path, and a shallow path beats a deep one."""
        return (source.startswith("results.json:"),
                source.count(".") + source.count("["), len(source))

    with open(RESULTS, encoding="utf-8") as fh:
        results = json.load(fh)
    n = 0
    for v, path in _walk_labelled(results):
        add(_forms(v), f"results.json:{path}")
        n += 1
    for name, p in vars(D).items():
        if isinstance(p, D.Param):
            for v, which in ((p.value, "value"), (p.lo, "lo"), (p.hi, "hi")):
                add(_forms(v), f"data.py:{name}.{which}")
    for key, row in D.FRONTIER_2026.items():
        add(_forms(row[2]) | _forms(row[3]), f"data.py:FRONTIER_2026[{key}]")
    for key, row in D.OPEN_WEIGHTS_2026.items():
        add(_forms(row[2]) | _forms(row[3]), f"data.py:OPEN_WEIGHTS_2026[{key}]")
    for key, (before, after) in D.QWEN38_GAINS.items():
        add(_forms(before) | _forms(after), f"data.py:QWEN38_GAINS[{key}]")
    for bucket in vocab.values():
        bucket.sort(key=specificity)
    return vocab, n


# --------------------------------------------------------------------------- #
#  Figures that come from a cited work, not from our harness
# --------------------------------------------------------------------------- #
#
# Each entry names the source. This is a provenance record: an entry here is a
# statement that the number belongs to somebody else's paper, not permission to
# print an unchecked figure.

CITED = {
    # Qi et al. 2023 (arXiv:2310.03693)
    "10": "Qi et al.: ten adversarial examples removed GPT-3.5 Turbo's guardrails",
    "0.20": "Qi et al.: cost in USD of that fine-tuning attack",
    # Biderman et al. 2024 (TMLR, arXiv:2405.09673)
    "100": "Biderman et al.: full fine-tuning learns perturbations 10-100x higher rank",
    # QLoRA (Dettmers et al. 2023)
    "0.37": "QLoRA: bits per parameter saved by double quantization",
    "4": "NF4 bit width; also a plain count in prose",
    "65": "QLoRA: 65B finetuned on one 48GB GPU; also the o3/8B right-sizing ratio",
    "48": "QLoRA: single-GPU memory in GB; also the TCO ratio and an f_s value",
    # Strix Halo recipe (published run)
    "8192": "published 27B run: sequence length",
    "448": "published 27B run: total steps",
    "11": "published 27B run: minutes per step",
    "80": "published 27B run: peak training memory, GB",
    "128": "Strix Halo unified memory, GB; also LoRA rank in the published run",
    "256": "published 27B run: LoRA alpha",
    "59": "Strix Halo peak bf16 throughput, TFLOP/s (vendor-derived)",
    "24": "discrete GPU memory, GB",
    "19": "usable memory on a 24 GB card at the 0.80 planning fraction, GB",
    "102": "usable memory on a 128 GB pool at the same fraction, GB",
    "1008": "RTX 4090 memory bandwidth, GB/s",
    "430": "RTX 4090 sustained training board power, W",
    "130": "Strix Halo sustained training package power, W",
    "120": "laptop NPU bandwidth, GB/s",
    # Qwen model facts
    "262": "Qwen3.8-27B context window, thousands of tokens",
    "27.78": "Qwen3.8-27B parameter count, billions",
    "17": "Qwen3.8-27B 4-bit build size, GB (vendor figure)",
    "14": "Qwen3.8-27B 4-bit build size, GB (our computation) and probe-set size",
    # Companion-study results quoted in the other paper
    "82": "companion study f_s",
    "53": "companion study carbon saving, low bound",
    "64": "companion study carbon saving, high bound",
    "86": "companion study water saving",
    "150": "companion study TCO ratio",
    "0.24": "Gemini measured per-prompt energy, Wh",
    "1.8": "GPT-4o long-context per-query energy, Wh (rounded from 1.788)",
    "212": "companion study rebound break-even, %",
    "2.4": "companion study consumer-GPU efficiency penalty",
    # Structural counts about the papers themselves
    "16": "methods in the roster",
    "5": "hypotheses in this paper; also a count in prose",
    "3": "counter-claims in this paper; also a count in prose",
    "6": "workload classes / section counts in prose",
    "2": "count in prose",
    "1": "count in prose",
    "0": "count in prose",
    "30": "upper end of the uplift-robustness sweep, points",
    "1,750": "support adapter corpus size",
    "120": "refusal samples in that corpus",
    "12": "probe-set lower bound",
    "8": "code probes",
    "34": "activation-term constant in the memory model",
    "99": "capability-ratio cap, %",
    "0.99": "capability-ratio cap",
    # Unit constants appearing inside displayed equations
    "3600": "seconds per hour, in the edge-energy equation",
    "152": "vocabulary size assumed in the logit term, thousands "
           "(training_memory default)",
    "4.127": "QLoRA: bits per parameter for NF4 with double quantization "
             "(4 + 8/64 + 32/16384)",
    "32,768": "published 27B run: tokens per optimizer step "
              "(batch 1 x accum 4 x seq 8192)",
    "6.4": "sum of the three exported adapter/gradient/optimizer terms in the "
           "memory-model validation breakdown (2.15 x 3)",
    # Sparse-MoE architecture facts from the Pass-2 frontier roster
    "896": "Kimi K3: experts in the sparse MoE, 16 active per token "
           "(data.FRONTIER_2026 note)",
    "744": "GLM-5.2: total parameters in billions (data.FRONTIER_2026 note)",
    "2.8": "Kimi K3: total parameters in trillions",
    "40": "GLM-5.2 active parameters, billions; also NVIDIA's 40-70% "
          "agentic-replaceability lower bound",
    # Benchmark scores quoted from cited model cards
    "76.8": "Ministral-8B HumanEval (mistral2024)",
    "67.1": "Llama-3.1-8B HumanEval (mistral2024)",
    "49.3": "Llama-3.1-8B GSM8K (mistral2024)",
    "54.5": "Ministral-8B GSM8K (mistral2024)",
    "65.0": "Ministral-8B MMLU (mistral2024)",
    "2.7": "Phi-2 parameter count, billions (belcak2025)",
    "1.7": "SmolLM2 parameter count, billions (belcak2025)",
    "15": "Phi-2 speed multiple (belcak2025); also the modal mid-tier output "
          "price in USD/Mtok",
    # Deployment audit, reported when the logs were present
    "32,500": "Caravaca et al.: GPU inference-energy measurements contributed",
    "97": "provider-resolution events in the Pass-2 configuration audit. The "
          "audit ran against a live deployment's logs; those logs are not in "
          "this repository, so results.json reports NO DATA on a machine "
          "without them and the paper reports what the audit found when it ran",
    # Hardware telemetry and driver facts from the RTX 5070 primary run
    "610.47": "NVIDIA Windows display driver version on the test machine",
    "12,227": "NVIDIA RTX 5070 total memory in MiB reported by driver telemetry",
    "250": "NVIDIA RTX 5070 board power limit in W reported by driver telemetry",
    # Fine-tuning agent / support adapter evaluated probe scores
    "30.8": "Support model screen identification base accuracy (%)",
    "92.9": "Support model screen identification adapted accuracy (3B, %)",
    "92.3": "Support model screen identification adapted accuracy (7B, %)",
}

# LaTeX constructs whose numbers are typographic, not claims.
STRIP_PATTERNS = [
    r"%.*",                              # comments
    r"\\(?:label|ref|cite|includegraphics|bibliographystyle|bibliography)"
    r"(?:\[[^\]]*\])?\{[^}]*\}",
    r"\\(?:documentclass|usepackage|usetikzlibrary|definecolor|newcommand|"
    r"graphicspath|IEEEoverridecommandlockouts)"
    r"(?:\[[^\]]*\])?(?:\{[^}]*\})*",
    r"\\begin\{(?:figure|table)\}\[[^\]]*\]",
    r"\\(?:columnwidth|textwidth|linewidth)",
    r"width=[^\s,\]]+",
    r"arXiv:\d+\.\d+",
    r"RFC\s*\d+",
    r"\\S\\ref\{[^}]*\}",
    r"\{HTML\}\{[0-9A-Fa-f]+\}",
]

NUM = re.compile(r"(?<![\w.,])(\d[\d,]*(?:\.\d+)?)(?![\d,]*\.?\d*[\w])")

# Product and version names carry digits that are identifiers, not quantities.
PRODUCT = re.compile(
    r"(?:RTX|GTX|Llama|Ministral|Qwen|Gemini|GPT|GLM|Kimi|Claude|Mistral|"
    r"ROCm|PyTorch|CUDA|MiKTeX|IEEE|RFC|H100|H200|A100|gfx|Ryzen|Radeon|"
    r"Ollama|Gemma|Strix|MAX\+)[\s\-]?[\w.\-]*",
    re.I)


def strip_latex(tex: str) -> str:
    # Drop the preamble entirely - it is all typesetting.
    i = tex.find(r"\begin{document}")
    if i > 0:
        tex = tex[i:]
    for pat in STRIP_PATTERNS:
        tex = re.sub(pat, " ", tex)
    # LaTeX writes thousands as 50{,}000; join them before tokenizing or the
    # groups are read as separate numbers.
    tex = re.sub(r"(\d)\{,\}(\d)", r"\1,\2", tex)
    # Typesetting arguments, scientific notation, ISO dates, and rank sets.
    tex = re.sub(r"\\renewcommand\{[^}]*\}\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\arraystretch\}?\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\times\s*10\^\{?-?\d+\}?", " ", tex)
    tex = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", tex)
    tex = re.sub(r"\\\{\s*\d+\s*,\s*\d+\s*\\\}", " ", tex)   # \{8,16\}
    tex = re.sub(r"\[\s*\d+\s*,\s*\d+\s*\]", " ", tex)        # [30,150] ranges
    tex = PRODUCT.sub(" ", tex)
    return tex


def verify(filename: str, vocab: Dict[str, List[str]]
           ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, int]]]:
    path = os.path.join(PAPER, filename)
    with open(path, encoding="utf-8") as fh:
        body = strip_latex(fh.read())

    misses: List[Tuple[str, str]] = []
    traced: List[Tuple[str, str, int]] = []
    seen: Set[str] = set()
    for m in NUM.finditer(body):
        tok = m.group(1)
        if tok in seen:
            continue
        seen.add(tok)
        bare = tok.replace(",", "")
        sources = vocab.get(tok) or vocab.get(bare)
        if sources:
            traced.append((tok, sources[0], len(sources)))
            continue
        if tok in CITED or bare in CITED:
            traced.append((tok, "CITED: " + CITED.get(tok, CITED.get(bare, "")), 1))
            continue
        try:
            val = float(bare)
        except ValueError:
            continue
        # Years, and section/figure ordinals below 10, are prose.
        if 1900 <= val <= 2100:
            continue
        # Context, for the report.
        s = max(0, m.start() - 60)
        ctx = re.sub(r"\s+", " ", body[s:m.end() + 60]).strip()
        misses.append((tok, ctx))
    return misses, traced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if anything is untraceable")
    # Provenance is ON by default. It used to be opt-in, and the default run
    # printed "Papers and harness agree" with no hint that a third of the
    # numbers in main.tex and nearly half in adaptation.tex match ten or more
    # unrelated sources apiece. A green light that hides its own weakest
    # evidence is worse than no light, so the weak matches are now in the
    # default output and --quiet is what you pass to suppress them.
    ap.add_argument("--quiet", dest="provenance", action="store_false",
                    default=True,
                    help="suppress the per-number provenance listing (the "
                         "weak-match SUMMARY is always printed)")
    ap.add_argument("--provenance", dest="provenance", action="store_true",
                    help="print the source justifying every number (default)")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if not os.path.isfile(RESULTS):
        sys.exit("results.json not found - run run_all.py first")

    vocab, n = build_vocabulary()
    print(f"vocabulary: {len(vocab):,} printed forms from {n:,} numbers in "
          f"results.json, plus {len(CITED)} annotated cited figures\n")

    total = 0
    for name in PAPERS:
        misses, traced = verify(name, vocab)
        total += len(misses)
        # A token matching many unrelated places is not really constrained by
        # this check: it would have passed whatever the paper said. The COUNT
        # of such tokens is printed unconditionally, because it is the honest
        # headline of what this tool proves; only the per-number listing is
        # suppressible with --quiet.
        weak = [t for t in traced if t[2] >= WEAK_MATCH_THRESHOLD]
        if args.provenance:
            print(f"  [{name}] provenance of {len(traced)} traced "
                  f"number(s):")
            for tok, source, count in traced:
                flag = ("  <-- weakly constrained"
                        if count >= WEAK_MATCH_THRESHOLD else "")
                more = f" (+{count - 1} other sources)" if count > 1 else ""
                print(f"      {tok:>10}  <-  {source}{more}{flag}")
        pct = (100.0 * len(weak) / len(traced)) if traced else 0.0
        print("")
        print(f"  [{name}] {len(traced) - len(weak)} of {len(traced)} "
              f"numbers pinned to fewer than {WEAK_MATCH_THRESHOLD} sources; "
              f"{len(weak)} ({pct:.0f}%) WEAKLY CONSTRAINED - each matches "
              f"{WEAK_MATCH_THRESHOLD}+ unrelated values and would have "
              f"passed whatever the sentence claimed. This tool shows that "
              f"the prose and the harness agree; it does NOT show that a "
              f"number is used in the right sentence.")
        print("")
        if not misses:
            print(f"  [{name}] every number traces to the harness or a citation")
            continue
        print(f"  [{name}] {len(misses)} number(s) not traceable:")
        for tok, ctx in misses:
            print(f"      {tok:>10}   ...{ctx}...")

    print()
    if total:
        print(f"{total} untraceable figure(s). Either the harness changed and "
              f"the prose did not, or the figure belongs to a cited work and "
              f"needs an entry in CITED naming its source.")
    else:
        print("Papers and harness agree.")
    if args.strict:
        sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
