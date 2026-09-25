"""Long-context execution benchmark over data/longcontext-v1/bench.jsonl (docs/phase4/long-context.md).

For every state and each question-count prefix (1, 2, 4, 8 and the state's full set) one synchronized forward is timed
after warm-up: packed tokens, milliseconds, peak allocated memory. The full request also gives per-question accuracy
and NLL against the known targets, by (length, depth) cell. With `--reference eager` (the default for the qwen family
when the main attention is not eager) the same requests run again on an eager copy of the checkpoint and the logits
are compared (max abs difference, argmax flips). For qwen3_5 checkpoints the hybrid runner's per-level time (state
pass, question blocks, leaves) is recorded by wrapping `janus.hybrid.run_level`. Requests over a budget or out of memory
are recorded as skipped with the reason, not dropped.

    python -m janus.longbench --checkpoint runs/phase1/screen/g2/best.pt --data data/longcontext-v1/bench.jsonl \
        --device cuda:0 --attention flex --max-tokens 65536 --max-state-plus-question 32768 --output runs/phase4/long/g2_flex
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import file_hash, write_json
from .packing import pack_request
from .schema import Request
from .training import load_checkpoint

PREFIXES = (1, 2, 4, 8)


def load_rows(path, cells=None, limit=None):
    """(Request, meta) pairs from bench.jsonl; `cells` filters by length label ("1k"), `limit` keeps the first rows per cell."""
    pairs, per_cell = [], {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = {k: row.get(k) for k in ("cell", "length", "depth", "family")}
        if cells and meta["cell"].split(":")[0] not in cells:
            continue
        per_cell[meta["cell"]] = per_cell.get(meta["cell"], 0) + 1
        if limit and per_cell[meta["cell"]] > limit:
            continue
        pairs.append((Request.from_dict(row), meta))
    if not pairs:
        raise ValueError(f"no rows selected from {path}")
    return pairs


def prefixes(request):
    counts = sorted({c for c in PREFIXES if c <= len(request.questions)} | {len(request.questions)})
    return [(count, replace(request, questions=request.questions[:count])) for count in counts]


@torch.inference_mode()
def run(model, pairs, warmup=2):
    """One timed forward per (state, question prefix); see the module docstring for the record fields."""
    from . import hybrid
    device = model.device
    cuda = device.type == "cuda"

    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)

    level_ms = []
    original = hybrid.run_level

    def timed_run_level(*args, **kwargs):
        synchronize()
        start = time.perf_counter()
        out = original(*args, **kwargs)
        synchronize()
        level_ms.append((time.perf_counter() - start) * 1000)
        return out
    hybrid.run_level = timed_run_level

    def forward(request):
        packed = pack_request(request, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs)
        return packed.token_count, [z.float().cpu().tolist() for z in model(packed)]

    records = []
    try:
        for _ in range(warmup):
            try:
                forward(prefixes(pairs[0][0])[0][1])
            except (ValueError, torch.OutOfMemoryError):
                break
        for request, meta in pairs:
            for count, subset in prefixes(request):
                record = {**meta, "group_id": request.group_id, "questions": count, "full": count == len(request.questions)}
                level_ms.clear()
                synchronize()
                if cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                try:
                    tokens, logits = forward(subset)
                except (ValueError, torch.OutOfMemoryError) as error:
                    if cuda:
                        torch.cuda.empty_cache()
                    records.append({**record, "skipped": f"{type(error).__name__}: {str(error)[:200]}"})
                    continue
                synchronize()
                record.update({"packed_tokens": tokens, "ms": (time.perf_counter() - start) * 1000,
                               "peak_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
                               "logits": logits, "targets": [list(q.target) if q.target else None for q in subset.questions],
                               "kinds": [q.kind for q in subset.questions], "levels_ms": list(level_ms) or None})
                records.append(record)
    finally:
        hybrid.run_level = original
    return records


def _cell_key(record):
    return f"{record['length'] // 1024}k|{record['questions']}"


def summarize(records, reference=None):
    """Aggregate timing by (length, question count), accuracy by (length, depth) cell and family, per-level timing by
    length, and agreement with `reference` records (same group_id and question count) when given."""
    latency = {}
    for r in records:
        cell = latency.setdefault(_cell_key(r), {"length": r["length"], "questions": r["questions"], "ms": [], "peak": [], "tokens": [], "skipped": 0})
        if "skipped" in r:
            cell["skipped"] += 1
            continue
        cell["ms"].append(r["ms"])
        cell["tokens"].append(r["packed_tokens"])
        if r["peak_bytes"] is not None:
            cell["peak"].append(r["peak_bytes"])
    for cell in latency.values():
        ms = cell.pop("ms")
        peak, tokens = cell.pop("peak"), cell.pop("tokens")
        cell.update({"count": len(ms), "p50_ms": float(np.median(ms)) if ms else None, "p95_ms": float(np.quantile(ms, .95)) if ms else None,
                     "peak_bytes_max": max(peak) if peak else None, "packed_tokens_mean": float(np.mean(tokens)) if tokens else None})

    def score(record):
        for z, y, kind in zip(record["logits"], record["targets"], record["kinds"]):
            if y is None:
                continue
            z = torch.tensor(z, dtype=torch.float32)
            p = z.softmax(-1)
            yield kind, float(y[int(p.argmax())]), float(-(torch.tensor(y) * z.log_softmax(-1)).sum())
    accuracy, by_family, by_kind = {}, {}, {}
    for r in records:
        if "skipped" in r or not r["full"]:
            continue
        for kind, correct, nll in score(r):
            for table, key in ((accuracy, r["cell"]), (by_family, r["family"]), (by_kind, kind)):
                entry = table.setdefault(key, {"count": 0, "correct": 0., "nll": 0.})
                entry["count"] += 1
                entry["correct"] += correct
                entry["nll"] += nll
    for table in (accuracy, by_family, by_kind):
        for entry in table.values():
            entry["accuracy"] = entry.pop("correct") / entry["count"]
            entry["nll"] = entry["nll"] / entry["count"]
    levels = {}
    for r in records:
        if r.get("levels_ms"):
            levels.setdefault(f"{r['length'] // 1024}k", []).append(r["levels_ms"])
    levels = {length: {"count": len(rows), "level_p50_ms": [float(np.median(col)) for col in zip(*rows)]}
              for length, rows in levels.items() if len({len(row) for row in rows}) == 1}
    summary = {"latency": latency, "accuracy_by_cell": accuracy, "accuracy_by_family": by_family, "accuracy_by_kind": by_kind,
               "hybrid_levels": levels}
    if reference is not None:
        table = {}
        lookup = {(r["group_id"], r["questions"]): r for r in reference if "skipped" not in r}
        for r in records:
            other = lookup.get((r["group_id"], r["questions"]))
            if "skipped" in r or other is None:
                continue
            entry = table.setdefault(f"{r['length'] // 1024}k", {"compared": 0, "max_abs_logit_diff": 0., "argmax_flips": 0, "questions": 0})
            entry["compared"] += 1
            for a, b in zip(r["logits"], other["logits"]):
                a, b = torch.tensor(a), torch.tensor(b)
                entry["max_abs_logit_diff"] = max(entry["max_abs_logit_diff"], float((a - b).abs().max()))
                entry["argmax_flips"] += int(a.argmax() != b.argmax())
                entry["questions"] += 1
        summary["agreement"] = table
    return summary


def markdown(summary, title=""):
    lines = [f"# {title}", ""] if title else []
    lengths = sorted({c["length"] for c in summary["latency"].values()})
    counts = sorted({c["questions"] for c in summary["latency"].values()})
    lines += ["## Latency p50 / p95 ms by state length and question count (skipped in parentheses)", "",
              "| length | " + " | ".join(f"Q={q}" for q in counts) + " |", "|---|" + "---|" * len(counts)]
    for length in lengths:
        row = []
        for q in counts:
            c = summary["latency"].get(f"{length // 1024}k|{q}")
            row.append("" if c is None else (f"{c['p50_ms']:.0f} / {c['p95_ms']:.0f}" if c["p50_ms"] is not None else "-")
                       + (f" ({c['skipped']} skipped)" if c and c["skipped"] else ""))
        lines.append(f"| {length // 1024}k | " + " | ".join(row) + " |")
    lines += ["", "## Peak allocated GB by state length and question count", "",
              "| length | " + " | ".join(f"Q={q}" for q in counts) + " |", "|---|" + "---|" * len(counts)]
    for length in lengths:
        row = [(f"{c['peak_bytes_max'] / 1e9:.2f}" if c and c["peak_bytes_max"] else "-") for c in
               (summary["latency"].get(f"{length // 1024}k|{q}") for q in counts)]
        lines.append(f"| {length // 1024}k | " + " | ".join(row) + " |")
    cells = summary["accuracy_by_cell"]
    if cells:
        depths = sorted({float(k.split(":")[1]) for k in cells})
        lines += ["", "## Accuracy (NLL) by state length and record depth, full requests", "",
                  "| length | " + " | ".join(f"depth {d}" for d in depths) + " |", "|---|" + "---|" * len(depths)]
        for length in lengths:
            row = []
            for d in depths:
                c = cells.get(f"{length // 1024}k:{d}")
                row.append("-" if c is None else f"{c['accuracy']:.3f} ({c['nll']:.2f}, n={c['count']})")
            lines.append(f"| {length // 1024}k | " + " | ".join(row) + " |")
        lines += ["", "By family: " + ", ".join(f"{k} {v['accuracy']:.3f} (n={v['count']})" for k, v in sorted(summary["accuracy_by_family"].items())),
                  "", "By kind: " + ", ".join(f"{k} {v['accuracy']:.3f} (n={v['count']})" for k, v in sorted(summary["accuracy_by_kind"].items()))]
    if summary.get("agreement"):
        lines += ["", "## Agreement with the reference attention (same requests)", "", "| length | compared | max abs logit diff | argmax flips / questions |", "|---|---|---|---|"]
        for length, a in sorted(summary["agreement"].items(), key=lambda kv: int(kv[0][:-1])):
            lines.append(f"| {length} | {a['compared']} | {a['max_abs_logit_diff']:.2e} | {a['argmax_flips']} / {a['questions']} |")
    if summary.get("hybrid_levels"):
        lines += ["", "## Hybrid runner: p50 ms per level (state pass, question blocks, leaves)", "", "| length | levels |", "|---|---|"]
        for length, a in sorted(summary["hybrid_levels"].items(), key=lambda kv: int(kv[0][:-1])):
            lines.append(f"| {length} | " + ", ".join(f"{v:.0f}" for v in a["level_p50_ms"]) + " |")
    return "\n".join(lines) + "\n"


def benchmark(checkpoint, data, output, device="cpu", attention=None, reference=None, cells=None, limit=None, warmup=2,
              max_tokens=None, max_state_plus_question=None):
    output = Path(output)
    if (output / "benchmark.json").exists():
        raise FileExistsError(output / "benchmark.json")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    pairs = load_rows(data, cells, limit)
    overrides = {"max_tokens": max_tokens, "max_state_plus_question": max_state_plus_question}
    model, metadata = load_checkpoint(checkpoint, device, attention=attention, **overrides)
    settings = {"attention": model.config.attention, "max_tokens": model.config.max_tokens,
                "max_state_plus_question": model.config.state_budget, "backbone_family": model.config.backbone_family, "dtype": model.config.dtype}
    if reference is None:
        reference = "eager" if model.config.backbone_family == "qwen" and settings["attention"] != "eager" else "none"
    records = run(model, pairs, warmup)
    reference_records = None
    if reference != "none":
        del model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        model, _ = load_checkpoint(checkpoint, device, attention=reference, **overrides)
        reference_records = run(model, pairs, warmup)
    summary = summarize(records, reference_records)
    result = {"checkpoint_sha256": file_hash(checkpoint), "data_sha256": file_hash(data), "device": str(device),
              "device_name": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else "CPU",
              "model": metadata["model"], "settings": settings, "reference_attention": reference, "warmup": warmup, "limit": limit,
              "timing_scope": "one warm synchronized forward per (state, question prefix): packing, transfers, forward and CPU logits; "
                              "no network; hybrid level times include their own synchronizations",
              "states": len(pairs), "summary": summary, "records": records}
    if reference_records is not None:
        result["reference_summary"] = summarize(reference_records)
        result["reference_records"] = reference_records
    write_json(output / "benchmark.json", result)
    text = markdown(summary, f"{Path(checkpoint).parent.name}: {settings['attention']} on {result['device_name']}")
    if reference_records is not None:
        text += "\n" + markdown(result["reference_summary"], f"Reference: {reference}")
    (output / "benchmark.md").write_text(text)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/longcontext-v1/bench.jsonl")
    parser.add_argument("--output", required=True, help="Directory for benchmark.json and benchmark.md")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attention", choices=("eager", "sdpa", "flex"), help="Override the checkpoint's attention")
    parser.add_argument("--reference", choices=("none", "eager", "sdpa"), help="Second run for logit agreement (default: eager for the qwen family when --attention is not eager)")
    parser.add_argument("--cells", help="Comma-separated length labels, e.g. 1k,4k (default: all)")
    parser.add_argument("--limit", type=int, help="States per (length, depth) cell")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, help="Per-request budget override (Jev documents 65536)")
    parser.add_argument("--max-state-plus-question", type=int, help="State-plus-longest-question budget override (Jev documents 32768)")
    args = parser.parse_args(argv)
    result = benchmark(args.checkpoint, args.data, args.output, args.device, args.attention, args.reference,
                       [c for c in args.cells.split(",") if c] if args.cells else None, args.limit, args.warmup,
                       args.max_tokens, args.max_state_plus_question)
    print(json.dumps({k: result[k] for k in ("device_name", "settings", "reference_attention", "states")}, indent=2))
    print(markdown(result["summary"]))


if __name__ == "__main__":
    main()
