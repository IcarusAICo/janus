"""Latency, token, and cost summaries for demo runs."""
from __future__ import annotations

import json
import math
from pathlib import Path

TYPESAFE_INPUT_USD_PER_MTOK = 0.042
# OpenAI standard short-context list prices, Sep 2026: https://developers.openai.com/api/docs/pricing
OPENAI_USD_PER_MTOK = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-sol": (4.00, 20.00),
}


def rates_for(backend, model=None):
    """(input_usd_per_mtok, output_usd_per_mtok) for a bench backend."""
    if backend == "gpt":
        return OPENAI_USD_PER_MTOK.get(model, OPENAI_USD_PER_MTOK["gpt-5.6-luna"])
    if backend == "local":
        return (0.0, 0.0)
    return (TYPESAFE_INPUT_USD_PER_MTOK, 0.0)


def percentile(values, p):
    """Nearest-rank percentile on a nonempty sequence of numbers."""
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[min(len(ordered), max(1, rank)) - 1]


def summarize(records, wall_seconds, *, input_usd_per_mtok=None, output_usd_per_mtok=0.0):
    """Aggregate call records into the fields written to summary.json."""
    wall = float(wall_seconds)
    if input_usd_per_mtok is None:
        input_usd_per_mtok = TYPESAFE_INPUT_USD_PER_MTOK
    latencies = [float(r.latency_ms) for r in records]
    input_tokens = sum(int(r.input_tokens) for r in records)
    output_tokens = sum(int(getattr(r, "output_tokens", 0) or 0) for r in records)
    calls = len(records)
    estimated = input_tokens / 1_000_000 * input_usd_per_mtok + output_tokens / 1_000_000 * output_usd_per_mtok
    return {
        "calls": calls,
        "wall_seconds": wall,
        "qps": (calls / wall) if wall > 0 else 0.0,
        "latency_p50_ms": percentile(latencies, 50) if latencies else 0.0,
        "latency_p95_ms": percentile(latencies, 95) if latencies else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_usd_per_mtok": input_usd_per_mtok,
        "output_usd_per_mtok": output_usd_per_mtok,
        "estimated_usd": estimated,
        "tokens_per_hour": (input_tokens / wall * 3600) if wall > 0 else 0.0,
        "usd_per_hour": (estimated / wall * 3600) if wall > 0 else 0.0,
    }


def write_summary(path, records, wall_seconds, extra=None):
    payload = summarize(records, wall_seconds)
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
