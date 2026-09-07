# Copyright (c) 2026 OpenStorey LLC.
# Released under the MIT License. See LICENSE in the repository root.

"""
bench.py - Primary measurement of local inference on the machine it runs on.

    python measure/bench.py --list
    python measure/bench.py                       # default ladder
    python measure/bench.py --models ministral-3:8b qwen3.8:27b
    python measure/bench.py --reps 7 --tokens 256

WHY THIS EXISTS
    The study's edge-energy model has three inputs:

        E_edge = P * (n_o / tau) * (1 + beta) / 3600

    Until now all three were taken from elsewhere: throughput tau from a
    third-party benchmark blog, the prefill factor beta from an author
    assumption, and board power P from a vendor TDP. A reviewer is entitled to
    ask why any of them describe the hardware a household actually owns.

    This harness measures tau and beta directly, on the machine it is run on,
    against the models that machine actually has. P is measured too wherever the
    platform exposes a power sensor; where it does not, the record says so
    explicitly rather than substituting a number.

WHAT IT DOES NOT CLAIM
    This is not a controlled comparison of GPU architectures. It measures one
    serving stack (Ollama) in its default configuration on whatever hardware is
    present. Different quantisations, context lengths, batch sizes and drivers
    will give different numbers, which is why every one of those is recorded
    alongside the result. It is a measurement of a deployment, not of a chip.

PROTOCOL
    * Temperature 0 and a fixed seed, so the token count is stable run to run.
    * One warm-up generation per model, discarded. Ollama's load_duration
      dominates a cold call by two orders of magnitude and is not part of
      steady-state serving.
    * N repetitions; we report the median and the full spread, not the mean,
      because a background scheduler stall produces a long tail rather than
      symmetric noise.
    * Prefill and generation are timed separately by the server, so beta is a
      measured ratio rather than an assumed constant.
    * Raw mode, bypassing the chat template. The template's system prefix is
      identical on every request and is therefore always cached, yet it is still
      counted in prompt_eval_count - which silently inflates prefill throughput
      by 7x on Ministral-3. See generate().
    * Prompt length is swept, not fixed. Prefill time is a fixed cost plus a
      per-token cost; one length cannot separate them, and the regression over
      several lengths is what exposed the caching artefact above.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

def client_url(host: str) -> str:
    """OLLAMA_HOST accepts bind addresses; urllib needs an HTTP client URL."""
    host = host.strip() or "127.0.0.1:11434"
    parts = urllib.parse.urlsplit(host if "://" in host else "http://" + host)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("OLLAMA_HOST must be an HTTP(S) host or URL")
    name = parts.hostname
    if name in ("0.0.0.0", "::"):
        name = "127.0.0.1" if name == "0.0.0.0" else "::1"
    name = f"[{name}]" if ":" in name else name
    port = parts.port or (11434 if "://" not in host else
                          443 if parts.scheme == "https" else 80)
    return urllib.parse.urlunsplit((parts.scheme, f"{name}:{port}",
                                   parts.path.rstrip("/"), "", ""))


OLLAMA = client_url(os.environ.get("OLLAMA_HOST", ""))

# A prompt long enough that prefill is measurable, short enough that it is not
# the whole run. Fixed text so the token count is identical on every machine.
PROMPT = (
    "You are summarising a technical document for an engineer. "
    "Explain, in plain prose and without lists, why moving a small language "
    "model from a data centre to a personal computer changes the energy "
    "arithmetic of a single query. Cover the cooling water, the idle overhead "
    "of shared infrastructure, and the fact that the personal computer was "
    "already powered on for other reasons. Be concrete and avoid slogans."
)

DEFAULT_TOKENS = 192
DEFAULT_REPS = 5

# Monotonic across the whole run so no two requests share a prefix.
_NONCE = 1000


# --------------------------------------------------------------------------- #
#  Ollama
# --------------------------------------------------------------------------- #

def _post(path: str, payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        OLLAMA + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models() -> List[dict]:
    try:
        return _get("/api/tags").get("models", [])
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"cannot reach Ollama at {OLLAMA}: {exc}")


def model_info(model: str) -> Dict[str, object]:
    """Architecture, parameter count and quantisation, from the server.

    Recorded with every measurement because throughput is meaningless without
    it: the same model name at Q4_K_M and at Q8_0 differs by more than the gap
    between two device classes.
    """
    try:
        d = _post("/api/show", {"model": model}, timeout=60)
    except Exception:
        return {"available": False}
    details = d.get("details") or {}
    info = d.get("model_info") or {}
    arch = details.get("family") or info.get("general.architecture")

    def pick(suffix):
        return next((v for k, v in info.items()
                     if k.endswith("." + suffix)), None)

    # Sparsity and multi-token prediction both break the "one full pass
    # over the weights per token" assumption that the bandwidth ladder
    # tests, so they are read from the server metadata rather than
    # guessed from the model name.
    n_exp = pick("expert_count")
    n_used = pick("expert_used_count")
    return {
        "available": True,
        "architecture": arch,
        "parameter_size": details.get("parameter_size"),
        "quantization": details.get("quantization_level"),
        "context_length": pick("context_length"),
        "families": details.get("families"),
        "block_count": pick("block_count"),
        "expert_count": n_exp,
        "expert_used_count": n_used,
        "mixture_of_experts": bool(n_exp and n_used and n_used < n_exp),
        "expert_active_fraction": (round(n_used / n_exp, 4)
                                   if n_exp and n_used else None),
        "nextn_predict_layers": pick("nextn_predict_layers"),
        "per_layer_embedding_input": pick("embedding_length_per_layer_input"),
        # A vision or audio tower is stored in the weight file but is not
        # streamed while decoding text, so it biases any bytes-per-token
        # figure upward. It lives in projector_info rather than
        # model_info on some builds, so check both.
        "multimodal_tower": (any(".vision." in k or ".audio." in k
                                 for k in info)
                             or bool(d.get("projector_info"))
                             or "vision" in (d.get("capabilities") or [])),
        "capabilities": d.get("capabilities"),
    }


def generate(model: str, tokens: int, seed: int = 20260905,
             nonce: Optional[int] = None, prompt_fillers: int = 0) -> dict:
    """One timed generation.

    Two separate caching traps have to be closed here, and the second one is
    invisible unless prompt length is varied.

    1. ``nonce`` defeats *whole-prompt* caching. Ollama reuses the KV cache for
       a repeated prefix, and reports the full prompt length in
       ``prompt_eval_count`` while charging only the uncached remainder to
       ``prompt_eval_duration``. Sending the same prompt five times therefore
       yields an apparent prefill throughput of 40,000+ tok/s, which is a cache
       hit rather than a measurement.

    2. ``raw`` defeats *chat-template* caching, which the nonce alone does not.
       ``/api/generate`` wraps the prompt in the model's chat template, and that
       template's system prefix sits BEFORE the caller's text - so a marker at
       the start of the user content is not at position zero of the token
       stream. The template prefix is identical on every request and is
       therefore always cached. On Ministral-3 the template contributes 552 of
       the 639 reported prompt tokens, and those 552 are counted but never
       computed: the same content measures 6,666 tok/s templated against 949
       tok/s raw. Setting ``raw`` sends the text verbatim, so
       ``prompt_eval_count`` is work actually done.

    Raw mode changes nothing about decode: the same weights are streamed per
    token either way, so generation throughput is unaffected.
    """
    prompt = FILLER * prompt_fillers + PROMPT
    prompt = prompt if nonce is None else f"[run {nonce:04d}] {prompt}"
    return _post("/api/generate", {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "num_predict": tokens,
            "temperature": 0,
            "seed": seed,
            "num_ctx": 4096,
        },
    })


# --------------------------------------------------------------------------- #
#  Power - measured where a sensor exists, absent where it does not
# --------------------------------------------------------------------------- #

def power_sensor() -> Dict[str, object]:
    """Identify a usable power sensor, or report honestly that there is none.

    NVIDIA exposes board power through nvidia-smi with no extra install, so a
    discrete-GPU machine yields a measured P. AMD's amd-smi reports socket power
    on supported parts, but is a ROCm component and is generally absent on a
    Windows consumer install. We do not fall back to a TDP figure and call it a
    measurement.
    """
    for cmd, args, kind in (
        ("nvidia-smi",
         ["--id=0", "--query-gpu=power.draw.instant", "--format=csv,noheader,nounits"], "nvidia"),
        ("nvidia-smi",
         ["--id=0", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], "nvidia"),
        ("amd-smi", ["metric", "-p", "--json"], "amd"),
        ("rocm-smi", ["--showpower", "--json"], "rocm"),
    ):
        try:
            out = subprocess.run([cmd] + args, capture_output=True, text=True,
                                 timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout.strip():
            if kind != "nvidia":
                continue  # These sensors have no implemented numeric reader.
            try:
                value = float(out.stdout.strip())
            except ValueError:
                continue
            if not math.isfinite(value) or value < 0:
                continue
            return {"available": True, "tool": cmd, "kind": kind,
                    "gpu_index": 0, "field": args[1].split("=", 1)[1],
                    "probe_output": out.stdout.strip()[:200]}
    return {
        "available": False,
        "tool": None,
        "reason": ("no power sensor reachable from this interpreter: "
                   "nvidia-smi, amd-smi and rocm-smi are all absent. On this "
                   "platform the board/package rail is only readable through a "
                   "kernel-mode helper (AMD uProf, LibreHardwareMonitor), which "
                   "this harness deliberately does not install. Energy figures "
                   "derived from these timings therefore carry a modelled P and "
                   "say so."),
    }


def read_power_w(sensor: Dict[str, object]) -> Optional[float]:
    if not sensor.get("available") or sensor.get("kind") != "nvidia":
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={sensor.get('gpu_index', 0)}",
             "--query-gpu=" + str(sensor.get("field", "power.draw")),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        if out.returncode != 0:
            return None
        value = float(out.stdout.strip())
        return value if math.isfinite(value) and value >= 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class PowerLog:
    """Sample board power continuously for the duration of one generation.

    Two readings either side of a call do not give energy: decode power ramps,
    and a 30-second generation spends most of its time at a level neither
    endpoint sees. Sampling on a background thread and integrating by the
    trapezium rule turns a power sensor into an energy measurement, which is the
    quantity the study actually models.

    Idle power is measured the same way, before any generation, so the marginal
    accounting the study uses (charge only the power ABOVE what the machine drew
    anyway) is a measurement on both sides rather than an assumption on one.
    """

    def __init__(self, sensor: Dict[str, object], interval: float = 0.10):
        self.sensor = sensor
        self.interval = interval
        self.samples: List[tuple] = []
        self._stop = None
        self._thread = None
        self.start = None
        self.end = None

    @property
    def usable(self) -> bool:
        return bool(self.sensor.get("available")
                    and self.sensor.get("kind") == "nvidia")

    def _run(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            w = read_power_w(self.sensor)
            if w is not None:
                self.samples.append(((t0 + time.perf_counter()) / 2, w))
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.usable:
            import threading
            t0 = time.perf_counter()
            w = read_power_w(self.sensor)
            if w is not None:
                self.samples.append(((t0 + time.perf_counter()) / 2, w))
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.end = time.perf_counter()
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=11)
            t0 = time.perf_counter()
            w = read_power_w(self.sensor)
            if w is not None:
                self.samples.append(((t0 + time.perf_counter()) / 2, w))
        return False

    def result(self) -> Dict[str, object]:
        if len(self.samples) < 2:
            return {"measured": False,
                    "reason": "fewer than two power samples"}
        # Integrate interpolated telemetry strictly over the request window.
        # No unobserved head/tail is silently discarded or called measured.
        if (self.start is None or self.end is None or self.end <= self.start
                or self.samples[0][0] > self.start
                or self.samples[-1][0] < self.end):
            return {"measured": False, "reason": "request not bracketed by power samples"}
        joules = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
            left, right = max(t0, self.start), min(t1, self.end)
            if right > left:
                wl = w0 + (w1 - w0) * (left - t0) / (t1 - t0)
                wr = w0 + (w1 - w0) * (right - t0) / (t1 - t0)
                joules += (wl + wr) / 2.0 * (right - left)
        span = self.end - self.start
        watts = [w for _, w in self.samples]
        return {
            "measured": True,
            "samples": len(self.samples),
            "sampled_s": round(span, 3),
            "window_s": span,
            "trace_s_w": [[t - self.start, w] for t, w in self.samples],
            "max_sample_gap_s": max(b[0] - a[0] for a, b in
                                    zip(self.samples, self.samples[1:])),
            "mean_w": round(joules / span, 2),
            "peak_w": round(max(watts), 2),
            "min_w": round(min(watts), 2),
            "energy_j": round(joules, 3),
            "energy_wh": round(joules / 3600.0, 6),
        }


def measure_idle_power(sensor: Dict[str, object],
                       seconds: float = 8.0) -> Dict[str, object]:
    """Baseline draw with no inference running.

    The study's 'marginal' accounting charges an edge query only the power above
    this line, on the argument that the machine was already on. That argument
    deserves a measured baseline rather than a 60 W assumption.
    """
    log = PowerLog(sensor, interval=0.25)
    if not log.usable:
        return {"measured": False, "reason": "no power sensor"}
    with log:
        time.sleep(seconds)
    return log.result()


# --------------------------------------------------------------------------- #
#  System identification
# --------------------------------------------------------------------------- #

def _wmic(query: str, field: str) -> str:
    """Best-effort Windows hardware lookup; empty string on any failure."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance {query}).{field}"],
            capture_output=True, text=True, timeout=30, check=False)
        return " / ".join(
            line.strip() for line in out.stdout.splitlines() if line.strip())
    except (OSError, subprocess.TimeoutExpired):
        return ""


def system_info() -> Dict[str, object]:
    info = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "ollama_host": OLLAMA,
    }
    if platform.system() == "Windows":
        info["cpu"] = _wmic("Win32_Processor", "Name")
        info["gpu"] = _wmic("Win32_VideoController", "Name")
        ram = _wmic("Win32_ComputerSystem", "TotalPhysicalMemory")
        try:
            info["ram_gb"] = round(int(ram.split(" / ")[0]) / 1e9, 1)
        except (ValueError, IndexError):
            info["ram_gb"] = None
    else:
        info["cpu"] = platform.processor()
    try:
        info["ollama_version"] = _get("/api/version").get("version")
    except Exception:
        info["ollama_version"] = None
    try:
        info["nvidia_gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
             "--format=csv"], capture_output=True, text=True,
            timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


# --------------------------------------------------------------------------- #
#  The measurement
# --------------------------------------------------------------------------- #

def measure_model(model: str, tokens: int, reps: int,
                  sensor: Dict[str, object],
                  sweep: bool = True, prompt_fillers: int = 0) -> Dict[str, object]:
    print(f"  {model:<26} warm-up...", end="", flush=True)
    meta = model_info(model)
    try:
        warm = generate(model, tokens, nonce=0, prompt_fillers=prompt_fillers)
    except Exception as exc:
        print(f" FAILED ({type(exc).__name__})")
        return {"model": model, "ok": False, "model_info": meta,
                "error": str(exc)[:200]}
    load_s = warm.get("load_duration", 0) / 1e9
    print(f" loaded in {load_s:5.1f}s", end="", flush=True)

    runs = []
    for i in range(reps):
        with PowerLog(sensor) as plog:
            wall0 = time.perf_counter()
            r = generate(model, tokens, nonce=i + 1, prompt_fillers=prompt_fillers)
            wall = time.perf_counter() - wall0
        power = plog.result()

        ev_n = r.get("eval_count") or 0
        ev_ns = r.get("eval_duration") or 0
        pp_n = r.get("prompt_eval_count") or 0
        pp_ns = r.get("prompt_eval_duration") or 0
        if ev_n <= 0 or ev_ns <= 0:
            continue
        runs.append({
            "rep": i + 1,
            "wall_s": round(wall, 4),
            "eval_tokens": ev_n,
            "eval_s": ev_ns / 1e9,
            "gen_tok_s": ev_n / (ev_ns / 1e9),
            "prompt_tokens": pp_n,
            "prefill_s": pp_ns / 1e9,
            "prefill_tok_s": (pp_n / (pp_ns / 1e9)) if pp_ns else None,
            # beta in the paper's E_edge equation: the share by which prefill
            # inflates the generation time for this prompt length.
            "beta_prefill_overhead": (pp_ns / ev_ns) if ev_ns else None,
            "power": power,
        })
        print(".", end="", flush=True)

    if not runs:
        print("  no usable runs")
        return {"model": model, "ok": False, "error": "no usable runs"}

    def med(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    def spread(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        if len(vals) < 2:
            return None
        return {"min": min(vals), "max": max(vals),
                "stdev": statistics.stdev(vals)}

    out = {
        "model": model,
        "ok": True,
        "model_info": meta,
        "reps": len(runs),
        "requested_tokens": tokens,
        "prompt_fillers": prompt_fillers,
        "load_s_cold": round(load_s, 2),
        "median_gen_tok_s": round(med("gen_tok_s"), 2),
        "gen_tok_s_spread": spread("gen_tok_s"),
        "median_prefill_tok_s": (round(med("prefill_tok_s"), 1)
                                 if med("prefill_tok_s") else None),
        "median_beta_prefill_overhead": (round(med("beta_prefill_overhead"), 4)
                                         if med("beta_prefill_overhead") else None),
        "median_eval_tokens": med("eval_tokens"),
        "median_prompt_tokens": med("prompt_tokens"),
        "runs": runs,
        "residency": _get("/api/ps").get("models", []),
    }

    # Length-controlled prefill sweep. Without it the 'prefill tok/s' column
    # above is a function of the prompt the harness sent, not a property of
    # the machine.
    if sweep:
        print("\n    prefill sweep", end="", flush=True)
        out["prefill_scaling"] = measure_prefill_scaling(
            model, out["median_gen_tok_s"])
        out["repeat_cache_probe"] = measure_repeat_cache(model)
        ps = out["prefill_scaling"]
        if ps.get("available"):
            f = ps["fit"]
            print(f"  marginal {f['marginal_prefill_tok_s']} tok/s, "
                  f"fixed {f['fixed_overhead_ms']} ms, R2={f['r2']}")
        else:
            print(f"  unavailable: {ps.get('reason')}")
    else:
        out["prefill_scaling"] = {"available": False,
                                  "reason": "sweep disabled"}

    # Energy per query, but only if a sensor actually produced samples. A
    # modelled P must never be silently promoted into this field.
    energies = [r["power"]["energy_wh"] for r in runs
                if r.get("power", {}).get("measured")]
    if energies:
        mean_w = [r["power"]["mean_w"] for r in runs
                  if r.get("power", {}).get("measured")]
        out["measured_energy"] = {
            "measured": True,
            "median_energy_wh_per_query": round(statistics.median(energies), 6),
            "median_mean_power_w": round(statistics.median(mean_w), 2),
            "queries_n": len(energies),
            "note": ("Board power integrated over each generation. This is the "
                     "accelerator rail only, not wall power: host CPU, DRAM and "
                     "PSU losses are outside it, so it is a lower bound on the "
                     "machine's draw and is not comparable to a hyperscaler's "
                     "comprehensive per-prompt figure without adding those."),
        }
    else:
        out["measured_energy"] = {
            "measured": False,
            "reason": "no power sensor on this machine; energy not measured",
        }
    print(f"  {out['median_gen_tok_s']:>7.1f} tok/s")
    return out


# --------------------------------------------------------------------------- #
#  Prefill scaling - the length-controlled measurement
# --------------------------------------------------------------------------- #

# One paragraph of neutral filler, repeated to lengthen the prompt without
# changing its character. Content is irrelevant to prefill cost; token count is
# not, which is the whole point of the sweep.
FILLER = (
    "The following background material is provided for context and does not "
    "change the question being asked. Consumer accelerators share a single "
    "memory pool between the processor and the graphics engine, so the rate at "
    "which weights can be streamed sets an upper bound on how fast tokens can "
    "be produced in a single stream. "
)

SWEEP_FILLERS = (0, 4, 12, 24)
SWEEP_REPS = 3
SWEEP_TOKENS = 16


def _sweep_generate(model: str, n_filler: int, nonce: int,
                    tokens: int = SWEEP_TOKENS, raw: bool = True) -> dict:
    prompt = f"[run {nonce:05d}] " + (FILLER * n_filler) + PROMPT
    return _post("/api/generate", {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": raw,
        "options": {"num_predict": tokens, "temperature": 0,
                    "seed": 20260905, "num_ctx": 4096},
    })


def _sweep_points(model: str, fillers, reps: int, raw: bool):
    """Median (prompt_tokens, prefill_s) at each swept prompt length."""
    global _NONCE
    points, rows = [], []
    for k in fillers:
        recs = []
        for _ in range(reps):
            _NONCE += 1
            try:
                r = _sweep_generate(model, k, _NONCE, raw=raw)
            except Exception as exc:
                return None, None, f"{type(exc).__name__}: {str(exc)[:120]}"
            pp_n = r.get("prompt_eval_count") or 0
            pp_ns = r.get("prompt_eval_duration") or 0
            if pp_n > 0 and pp_ns > 0:
                recs.append((pp_n, pp_ns / 1e9))
            print(".", end="", flush=True)
        if not recs:
            continue
        n_med = statistics.median([n for n, _ in recs])
        s_med = statistics.median([s for _, s in recs])
        points.append((n_med, s_med))
        rows.append({"fillers": k, "prompt_tokens": n_med,
                     "prefill_s_median": round(s_med, 5),
                     "apparent_tok_s": round(n_med / s_med, 1),
                     "reps": len(recs)})
    return points, rows, None


def _ols(xs: List[float], ys: List[float]) -> Dict[str, float]:
    """Least-squares fit y = a + b x, with R^2. Pure stdlib."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return {"intercept_s": a, "slope_s_per_token": b,
            "r2": (1.0 - sse / sst) if sst > 0 else float("nan")}


def measure_repeat_cache(model: str, reps: int = 3) -> Dict[str, object]:
    """The first caching trap, kept as a measurement rather than a memory.

    Send the identical prompt several times with no per-repetition marker and
    the server reuses the KV cache for the whole prefix, while still reporting
    the full prompt length in ``prompt_eval_count``. The result is an apparent
    prefill throughput an order of magnitude above anything the hardware can
    do. This function reproduces that on demand so the figure the paper quotes
    stays a measured one, and so anyone re-running the harness on other
    hardware sees the same trap rather than taking our word for it.

    Nothing here feeds the study's numbers. It is a control.
    """
    try:
        first = _sweep_generate(model, 0, 424242, tokens=8, raw=False)
    except Exception as exc:
        return {"available": False,
                "reason": f"{type(exc).__name__}: {str(exc)[:120]}"}
    rates, n_tok = [], None
    for _ in range(reps):
        try:
            # Same nonce every time: the point is that the prompt repeats.
            r = _sweep_generate(model, 0, 424242, tokens=8, raw=False)
        except Exception:
            break
        pp_n = r.get("prompt_eval_count") or 0
        pp_ns = r.get("prompt_eval_duration") or 0
        if pp_n > 0 and pp_ns > 0:
            rates.append(pp_n / (pp_ns / 1e9))
            n_tok = pp_n
    if not rates:
        return {"available": False, "reason": "no usable repetitions"}
    del first
    return {
        "available": True,
        "reps": len(rates),
        "prompt_tokens_reported": n_tok,
        "apparent_prefill_tok_s": round(statistics.median(rates), 0),
        "note": ("Identical prompt, no per-repetition marker, chat template "
                 "applied. prompt_eval_count reports the full prompt; "
                 "prompt_eval_duration charges only what was recomputed, which "
                 "is nothing. The quotient is not a throughput."),
    }


def measure_prefill_scaling(model: str, gen_tok_s: Optional[float],
                            fillers=SWEEP_FILLERS, reps: int = SWEEP_REPS,
                            n_in: float = 500.0, n_out: float = 300.0,
                            with_template_probe: bool = True
                            ) -> Dict[str, object]:
    """Separate the fixed cost of a request from the marginal cost of a token.

    A single prompt length cannot measure prefill throughput. Prefill time is
    ``a + b*n``: a fixed per-request cost (scheduling, sampler setup, the first
    kernel launches) plus a marginal cost per prompt token. Divide one such
    measurement by its token count and you get ``n/(a + b*n)``, which is not a
    throughput but a function of the prompt you happened to send - it rises with
    n and only approaches 1/b asymptotically. A 95-token prompt and a 639-token
    prompt do not yield the same quantity, so quoting either as "the prefill
    throughput" is a category error.

    Regressing prefill time on prompt length recovers both terms. 1/b is a
    marginal prefill throughput comparable across models, and it lets beta be
    quoted at a *stated* prompt-to-generation ratio - here the study's own
    typical query - instead of at whatever length the harness happened to use.

    The sweep also earns its cost as a diagnostic. Run against the chat-template
    endpoint it returns a large NEGATIVE intercept - minus 485 ms on the 8B -
    which is not a physical quantity. That artefact is what exposed the second
    caching trap: the template's cached system prefix inflates
    ``prompt_eval_count`` without contributing to ``prompt_eval_duration``, so
    the line has to pass below zero to reach the short-prompt point. A single
    prompt length would have reported the contaminated number with no way to
    tell. In raw mode the intercept becomes a small positive fixed cost, which
    is what a per-request overhead should look like.
    """
    global _NONCE
    points, rows, err = _sweep_points(model, fillers, reps, raw=True)
    if err:
        return {"available": False, "reason": err}
    if points is None or len(points) < 3:
        return {"available": False,
                "reason": "fewer than three usable prompt lengths"}

    fit = _ols([p[0] for p in points], [p[1] for p in points])
    b = fit["slope_s_per_token"]
    a = fit["intercept_s"]
    out = {
        "available": True,
        "points": rows,
        "fit": {
            "model": "prefill_s = a + b * prompt_tokens (OLS over medians)",
            "fixed_overhead_ms": round(a * 1000.0, 2),
            "marginal_prefill_tok_s": (round(1.0 / b, 1) if b > 0 else None),
            "r2": round(fit["r2"], 4),
            "lengths_tested": [r["prompt_tokens"] for r in rows],
        },
    }
    # beta at the study's own typical query, rather than at whatever prompt
    # length the ladder happened to produce.
    if gen_tok_s and b > 0:
        decode_s = n_out / gen_tok_s
        out["beta_at_study_typical_query"] = {
            "prompt_tokens": n_in,
            "generated_tokens": n_out,
            "beta": round((a + b * n_in) / decode_s, 4),
            "note": ("beta = prefill_s(500 prompt tokens) / decode_s(300 "
                     "generated tokens), from the fitted line and the measured "
                     "generation throughput. This is the ratio the paper's "
                     "E_edge equation needs, quoted at the paper's own typical "
                     "query rather than at the ladder's prompt length."),
        }

    # The same sweep through the chat-template endpoint, kept as a recorded
    # artefact rather than a remembered anecdote. It is what the harness
    # measured before the template cache was closed, and the negative intercept
    # is the tell.
    if with_template_probe:
        t_points, t_rows, t_err = _sweep_points(model, fillers, reps, raw=False)
        if t_err or not t_points or len(t_points) < 3:
            out["template_cache_probe"] = {"available": False,
                                           "reason": t_err or "too few points"}
        else:
            t_fit = _ols([p[0] for p in t_points], [p[1] for p in t_points])
            tb, ta = t_fit["slope_s_per_token"], t_fit["intercept_s"]
            raw0, tpl0 = rows[0], t_rows[0]
            out["template_cache_probe"] = {
                "available": True,
                "points": t_rows,
                "fit": {
                    "fixed_overhead_ms": round(ta * 1000.0, 2),
                    "marginal_prefill_tok_s": (round(1.0 / tb, 1)
                                               if tb > 0 else None),
                    "r2": round(t_fit["r2"], 4),
                },
                "shortest_prompt": {
                    "raw_prompt_tokens": raw0["prompt_tokens"],
                    "templated_prompt_tokens": tpl0["prompt_tokens"],
                    "template_tokens_added": (tpl0["prompt_tokens"]
                                              - raw0["prompt_tokens"]),
                    "raw_apparent_tok_s": raw0["apparent_tok_s"],
                    "templated_apparent_tok_s": tpl0["apparent_tok_s"],
                    "inflation_factor": round(
                        tpl0["apparent_tok_s"] / raw0["apparent_tok_s"], 2),
                    "prefill_s_raw": raw0["prefill_s_median"],
                    "prefill_s_templated": tpl0["prefill_s_median"],
                },
                "verdict": ("CONTAMINATED" if ta < 0 else "no artefact seen"),
                "note": (
                    "Identical content, identical sweep, template applied. The "
                    "template prefix is counted in prompt_eval_count but is "
                    "cached and never recomputed, so the extra tokens are "
                    "nearly free and the fitted line must pass below zero to "
                    "reach the shortest point. A negative fixed cost is not a "
                    "physical quantity, which is what makes the length sweep a "
                    "detector for this and not merely a better measurement. "
                    "Note the two marginal rates agree: only the intercept and "
                    "the apparent throughput are corrupted."
                ),
            }
    return out


# --------------------------------------------------------------------------- #
#  The bandwidth ladder - testing the premise within one machine
# --------------------------------------------------------------------------- #

def model_sizes() -> Dict[str, int]:
    """On-disk weight size in bytes for every locally installed model."""
    return {m["name"]: m.get("size", 0) for m in list_models()}


def unload(model: str) -> None:
    """Ask the server to evict a model, so the next one starts from a clean
    pool. Without this a ladder over several 17 GB models measures allocator
    pressure as much as it measures the models."""
    try:
        _post("/api/generate", {"model": model, "prompt": "", "keep_alive": 0},
              timeout=120)
    except Exception:
        pass


def measure_ladder(models: List[str], reps: int, tokens: int,
                   peak_bandwidth_gb_s: float,
                   sizes: Dict[str, int]) -> List[Dict[str, object]]:
    """Generation throughput against weight size, across every local model.

    The device-class argument in this study rests on single-stream decode being
    limited by memory bandwidth rather than arithmetic. That premise makes a
    quantitative prediction which one machine can test on its own, with no
    cross-vendor or cross-stack comparison to confound it: if each generated
    token requires streaming the whole weight set once, then

        tau * bytes_per_token ~= B_eff, a constant near the part's peak

    across models that differ by an order of magnitude in size. The product is
    an *effective read bandwidth*, and it has a ceiling that cannot be argued
    with - a part specified at 256 GB/s cannot exceed it while genuinely reading
    every weight.

    The same statistic carries its own falsification: a sparse mixture-of-
    experts model reads only its active experts per token, so its apparent
    effective bandwidth should sit far ABOVE the ceiling. If the statistic did
    not separate dense from sparse it would not be measuring what it claims to.
    """
    rows = []
    for m in models:
        size_b = sizes.get(m, 0)
        gib = size_b / 2**30
        print(f"  {m:<26} {gib:6.2f} GiB ", end="", flush=True)
        meta = model_info(m)
        try:
            generate(m, 8, nonce=0)
        except Exception as exc:
            print(f" FAILED ({type(exc).__name__})")
            rows.append({"model": m, "ok": False, "weights_gib": round(gib, 3),
                         "model_info": meta, "error": str(exc)[:200]})
            continue
        speeds = []
        runs = []
        for i in range(reps):
            try:
                r = generate(m, tokens, nonce=2000 + i)
            except Exception:
                break
            ev_n = r.get("eval_count") or 0
            ev_ns = r.get("eval_duration") or 0
            if ev_n > 0 and ev_ns > 0:
                speeds.append(ev_n / (ev_ns / 1e9))
                runs.append({"eval_tokens": ev_n, "eval_s": ev_ns / 1e9})
            print(".", end="", flush=True)
        residency = _get("/api/ps").get("models", [])
        unload(m)
        if not speeds:
            print("  no usable runs")
            rows.append({"model": m, "ok": False, "weights_gib": round(gib, 3),
                         "model_info": meta, "error": "no usable runs"})
            continue
        tok_s = statistics.median(speeds)
        # GiB -> GB, because the part's bandwidth is specified in GB/s.
        eff = tok_s * gib * (2**30 / 1e9)
        rows.append({
            "model": m,
            "ok": True,
            "model_info": meta,
            "weights_gib": round(gib, 3),
            "weights_gb": round(gib * (2**30 / 1e9), 3),
            "reps": len(speeds),
            "requested_tokens": tokens,
            "median_gen_tok_s": round(tok_s, 2),
            "gen_tok_s_spread": ({"min": min(speeds), "max": max(speeds),
                                  "stdev": statistics.stdev(speeds)}
                                 if len(speeds) > 1 else None),
            "runs": runs,
            "residency": residency,
            "effective_read_gb_s": round(eff, 1),
            "bandwidth_utilisation": round(eff / peak_bandwidth_gb_s, 3),
        })
        print(f"  {tok_s:6.2f} tok/s -> {eff:6.1f} GB/s "
              f"({eff / peak_bandwidth_gb_s * 100:.0f}% of peak)")
    return rows


def tagname(args, sysinfo) -> str:
    """Short filename-safe name for the machine under test."""
    tag = args.tag or (sysinfo.get("gpu") or "unknown").split("/")[0]
    return "".join(c if c.isalnum() else "-"
                   for c in tag).strip("-").lower()[:40]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="models to measure")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--prompt-fillers", type=int, default=0,
                    help="neutral filler paragraphs before the standard prompt")
    ap.add_argument("--ladder", action="store_true",
                    help="bandwidth ladder: generation throughput vs weight "
                         "size across every local model")
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="the part's specified peak memory bandwidth in GB/s, "
                         "used only to report a utilisation percentage")
    ap.add_argument("--ladder-max-gib", type=float, default=20.0,
                    help="skip models whose weights exceed this, to keep the "
                         "ladder inside the pool it is measuring")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the length-controlled prefill regression")
    ap.add_argument("--list", action="store_true", help="list local models")
    ap.add_argument("--tag", default=None,
                    help="short name for this machine, used in the filename")
    args = ap.parse_args()
    if args.reps < 2 or args.tokens < 1 or args.prompt_fillers < 0:
        ap.error("need at least two repetitions, positive tokens, nonnegative fillers")
    if args.bandwidth is not None and args.bandwidth <= 0:
        ap.error("--bandwidth must be positive")

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    available = [m["name"] for m in list_models()]
    if args.list:
        for m in sorted(available):
            print(" ", m)
        return

    models = args.models or [m for m in (
        "ministral-3:3b", "ministral-3:8b", "qwen3.8:27b") if m in available]
    missing = [m for m in models if m not in available]
    if missing:
        print(f"not present locally, skipping: {', '.join(missing)}")
        models = [m for m in models if m in available]
    if not models:
        sys.exit("no requested models are present; run --list")
    os.makedirs(RESULTS, exist_ok=True)

    sysinfo = system_info()
    sensor = power_sensor()
    idle = measure_idle_power(sensor)

    print(f"\nmachine : {sysinfo.get('gpu') or sysinfo.get('cpu')}")
    print(f"ram     : {sysinfo.get('ram_gb')} GB")
    print(f"ollama  : {sysinfo.get('ollama_version')}")
    print(f"power   : {'measured via ' + str(sensor['tool']) if sensor['available'] else 'NO SENSOR - timings only'}")
    print(f"protocol: {args.reps} reps x {args.tokens} tokens, temperature 0, "
          f"fixed seed, warm-up discarded\n")

    if args.ladder:
        if args.bandwidth is None:
            sys.exit("--ladder needs --bandwidth <peak GB/s for this part>")
        sizes = model_sizes()
        cap = args.ladder_max_gib * 2**30
        chosen = args.models or sorted(available)
        skipped = [m for m in chosen if sizes.get(m, 0) > cap]
        chosen = [m for m in chosen if 0 < sizes.get(m, 0) <= cap]
        if skipped:
            print("over the size cap, not attempted: " + ", ".join(skipped))
        print("\nbandwidth ladder: " + str(len(chosen)) + " models, "
              + str(args.reps) + " reps x " + str(args.tokens) + " tokens, "
              + "peak " + str(args.bandwidth) + " GB/s\n")
        rows = measure_ladder(chosen, args.reps, args.tokens, args.bandwidth,
                              sizes)
        path = os.path.join(RESULTS, tagname(args, sysinfo) + "-ladder.json")
        payload = {
            "harness": "measure/bench.py --ladder",
            "experiment": "generation throughput vs weight size, one machine",
            "protocol": {
                "prompt": PROMPT,
                "requested_tokens": args.tokens,
                "reps": args.reps,
                "temperature": 0,
                "seed": 20260905,
                "num_ctx": 4096,
                "warmup_discarded": True,
                "statistic": "median over reps",
                "raw_mode": True,
                "unload_between_models": True,
                "size_cap_gib": args.ladder_max_gib,
                "models_over_cap_not_attempted": skipped,
                "peak_bandwidth_gb_s": args.bandwidth,
                "peak_bandwidth_source": (
                    "part specification, not measured here; used only to "
                    "express each result as a percentage of a ceiling"),
            },
            "system": sysinfo,
            "power_sensor": sensor,
            "models": rows,
        }
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2)
        print("\nwrote " + path)
        return

    results = []
    for m in models:
        results.append(measure_model(m, args.tokens, args.reps, sensor,
                                     sweep=not args.no_sweep,
                                     prompt_fillers=args.prompt_fillers))
        unload(m)

    tag = tagname(args, sysinfo)
    path = os.path.join(RESULTS, f"{tag}.json")
    payload = {
        "harness": "measure/bench.py",
        "protocol": {
            "prompt": PROMPT,
            "requested_tokens": args.tokens,
            "prompt_fillers": args.prompt_fillers,
            "filler": FILLER,
            "energy_scope": "GPU 0 board telemetry, request window only; excludes host and post-request tail",
            "reps": args.reps,
            "temperature": 0,
            "seed": 20260905,
            "num_ctx": 4096,
            "warmup_discarded": True,
            "statistic": "median over reps",
            "prefill_sweep": ("prompt length varied over four values by "
                              "repeating a neutral filler paragraph; "
                              "prefill time regressed on server-reported "
                              "prompt_eval_count to separate the fixed "
                              "per-request cost from the marginal cost "
                              "per prompt token"),
            "raw_mode": True,
            "cache_defeat": ("two traps, both closed: a unique marker per "
                             "repetition defeats whole-prompt caching, and raw "
                             "mode bypasses the chat template whose identical "
                             "system prefix is otherwise always cached while "
                             "still being counted in prompt_eval_count"),
        },
        "system": sysinfo,
        "power_sensor": sensor,
        "idle_power": idle,
        "models": results,
    }
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
