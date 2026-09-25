"""Serving-latency profiler and equality check for the local server (docs/phase4/latency.md).

    python -m janus.latency profile  --checkpoint runs/.../best.pt --device cuda:0 --tasks banking77,mmlu-pro --limit 20
    python -m janus.latency equality --checkpoint runs/.../best.pt --device cuda:0 --tasks banking77,mmlu-pro --limit 100
    python -m janus.latency sweep    --checkpoint runs/.../best.pt --device cuda:0 --lengths 150,1000,4000,8000,16000

`profile` times one request at a time through the live path (`janus.server.Worker`): tokenize+pack, each backbone level
pass, the head/readout, softmax+copy, the JSON envelope, and the HTTP round trip through an in-process server.
`equality` runs the same requests through the plain path and the fast path and reports the max abs probability
difference and the argmax flips. Requests are the benchmark's own items (demos.bench.tasks). `sweep` times fresh
states (no prefix cache) of the given token lengths with one four-option Choice, with the level passes split and
the eager layer steps grouped by layer type (docs/phase4/latency-track2.md)."""

import argparse
import http.client
import json
import statistics
import threading
import time

import torch

from . import hybrid
from .schema import Request
from .server import Worker, envelope, make_server


def bench_requests(task, limit=100):
    """The benchmark's items as wire-format bodies (what demos/bench/run.py posts)."""
    from demos.bench.run import _api_questions
    from demos.bench.tasks import load_task
    if task == "wiki-hop":  # the local backend scores the Baseball links in batches of 32 five-level Score questions
        from demos.bench.run import baseball_links
        from demos.wikiracing.race import _score_questions, _strip_private
        links = baseball_links()
        state = "Wikispeedia. Current article: Baseball. Target article: Sun."
        return [{"state": state, "questions": _strip_private(_score_questions(links[i:i + 32], "Sun"))}
                for i in range(0, min(len(links), 32 * limit), 32)]
    return [{"state": ex.state, "questions": _api_questions(ex.questions)} for ex in load_task(task, limit=limit)]


class LevelTimer:
    """Records synchronised wall time of every level pass (eager `run_level` or a graph replay) while active."""

    def __init__(self, device):
        self.device, self.levels = device, []

    def __enter__(self):
        self.originals = (hybrid.run_level, hybrid.replay_level)
        hybrid.run_level, hybrid.replay_level = self._wrap(hybrid.run_level, "eager"), self._wrap(hybrid.replay_level, "graph")
        return self

    def __exit__(self, *exc):
        hybrid.run_level, hybrid.replay_level = self.originals

    def _wrap(self, fn, kind):
        def timed(lm, embeds, *args, **kwargs):
            self._sync()
            start = time.perf_counter()
            out = fn(lm, embeds, *args, **kwargs)
            self._sync()
            self.levels.append((kind, tuple(embeds.shape[:2]), (time.perf_counter() - start) * 1e3))
            return out
        return timed

    def _sync(self):
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize(self.device)


class LayerTimer:
    """CUDA-event time of every eager `_layer_step` (janus.hybrid) by layer type while active; graph replays are not
    seen (their layers are inside the graph), captures are skipped."""

    def __init__(self):
        self.events = []  # (layer type, start event, end event)

    def __enter__(self):
        self.original = hybrid._layer_step
        hybrid._layer_step = self._timed
        return self

    def __exit__(self, *exc):
        hybrid._layer_step = self.original

    def _timed(self, layer, *args):
        if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
            return self.original(layer, *args)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.original(layer, *args)
        end.record()
        self.events.append((layer.layer_type, start, end))
        return out

    def by_type(self):
        """Total ms per layer type (after a synchronize)."""
        totals = {}
        for kind, start, end in self.events:
            totals[kind] = totals.get(kind, 0.) + start.elapsed_time(end)
        return totals


def sweep_body(tokenizer, tokens):
    """A request with a state of about `tokens` tokens (long-context bench text cut to length) and one Choice."""
    from itertools import cycle
    rows = (json.loads(line) for line in open("data/longcontext-v1/bench.jsonl"))
    text, ids = "", []
    for row in cycle(r["state"] for r in rows):
        text += row + "\n"
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= tokens:
            break
    state = tokenizer.decode(ids[:tokens])
    return {"state": state, "questions": {"q": {"type": "choice", "instructions": "Which topic does the text cover?",
                                                "criteria": {k: f"topic {k}" for k in "abcd"}}}}


def sweep(checkpoint, device, lengths, fast, repeat=5, with_profiler=False, max_tokens=None, fp8=False, with_http=True, batch_window_ms=4, fp4=False):
    """Fresh-state latency per state length: in-process medians over `repeat` requests after one warm-up, and the
    median HTTP round trip through an in-process server with the prefix cache off (`http`)."""
    server = make_server(checkpoint, port=0, device=device, fast=fast, max_tokens=max_tokens or max(lengths) + 512, fp8=fp8, fp4=fp4,
                         prefix_cache_tokens=0, batch_window_ms=batch_window_ms)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    worker = server.worker
    report = {}
    for tokens in lengths:
        body = sweep_body(worker.model.tokenizer, tokens)
        time_request(worker, body, device)
        records = []
        for _ in range(repeat):
            with LayerTimer() as layers:
                record = time_request(worker, body, device)
            torch.cuda.synchronize(device) if torch.device(device).type == "cuda" else None
            record["layers"] = layers.by_type()
            records.append(record)
        p50 = lambda key: statistics.median(r[key] for r in records)
        summary = {key: p50(key) for key in ("tokens", "pack", "level_total", "head", "softmax", "json", "total")}
        summary["levels"] = records[-1]["levels"]
        summary["layers"] = {k: statistics.median(r["layers"].get(k, 0.) for r in records) for r in records[-1:] for k in r["layers"]}
        if torch.device(device).type == "cuda":
            summary["peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        if with_http:
            summary["http"] = http_round_trip(server, body)
        if with_profiler and torch.device(device).type == "cuda":
            summary.update(cuda_profile(worker, body, device))
        report[tokens] = summary
        print(f"{tokens}: " + json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in summary.items() if k not in ("table", "levels")}))
        print("  levels:", [(kind, shape, round(ms, 1)) for kind, shape, ms in summary["levels"]])
        if "table" in summary:
            print(summary["table"])
    server.shutdown()
    return report


def time_request(worker, body, device):
    """Stage breakdown of one request in ms: pack, levels (list), head (forward minus levels), softmax, json."""
    sync = (lambda: torch.cuda.synchronize(device)) if torch.device(device).type == "cuda" else (lambda: None)
    graphs = getattr(worker.model.backbone, "graphs", None)
    known = len(graphs.graphs) if graphs is not None else 0
    t0 = time.perf_counter()
    request = Request.from_dict(body)
    packed = worker.pack(request)
    t1 = time.perf_counter()
    with LevelTimer(device) as timer, torch.inference_mode():
        logits = worker.model(packed)
        sync()
        t2 = time.perf_counter()
        distributions = worker.probabilities(logits)
        t3 = time.perf_counter()
    payload = json.dumps(envelope(request, distributions, "x", packed.token_count), ensure_ascii=False).encode()
    t4 = time.perf_counter()
    levels = sum(ms for _, _, ms in timer.levels)
    return {"tokens": packed.token_count, "pack": (t1 - t0) * 1e3, "levels": timer.levels, "level_total": levels,
            "head": (t2 - t1) * 1e3 - levels, "softmax": (t3 - t2) * 1e3, "json": (t4 - t3) * 1e3,
            "total": (t4 - t0) * 1e3, "bytes": len(payload),
            "captured": graphs is not None and len(graphs.graphs) > known}  # this request captured a new graph


def http_round_trip(server, body, repeat=5):
    """Median wall time of a POST over one keep-alive connection to an in-process server, ms."""
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port)
    payload = json.dumps(body).encode()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        connection.request("POST", "/v1/systemone", payload, {"Content-Type": "application/json"})
        response = connection.getresponse()
        response.read()
        times.append((time.perf_counter() - start) * 1e3)
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
    connection.close()
    return statistics.median(times)


def cuda_profile(worker, body, device, rows=15):
    """torch.profiler summary of one request: CUDA kernel count and the top kernels by CUDA time."""
    from torch.profiler import ProfilerActivity, profile
    packed = worker.pack(Request.from_dict(body))
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        worker.probabilities(worker.model(packed))
        torch.cuda.synchronize(device)
    events = prof.key_averages()
    kernels = sum(e.count for e in events if e.device_type == torch.autograd.DeviceType.CUDA)
    cuda_ms = sum(e.self_device_time_total for e in events if e.device_type == torch.autograd.DeviceType.CUDA) / 1e3
    classes = {}  # CUDA time by kernel class: GEMMs, attention, the fla scan, everything else (glue)
    for e in events:
        if e.device_type == torch.autograd.DeviceType.CUDA:
            classes[kernel_class(e.key)] = classes.get(kernel_class(e.key), 0.) + e.self_device_time_total / 1e3
    return {"kernel_launches": kernels, "cuda_ms": cuda_ms, "kernel_classes": classes,
            "table": events.table(sort_by="self_device_time_total", row_limit=rows)}


def kernel_class(name):
    n = name.lower()
    if "gemm" in n or "cutlass" in n or "nvjet" in n or "scaled_mm" in n or "cublas" in n:
        return "gemm"
    if "flash" in n or "fmha" in n or "attention" in n:
        return "attention"
    if "chunk_" in n or "gated_delta" in n or "delta_rule" in n or "solve_tril" in n or "wy_" in n or "cumsum" in n and "fla" in n:
        return "fla"
    if "memcpy" in n or "memset" in n:
        return "copy"
    return "other"


def profile(checkpoint, device, tasks, limit, fast, warmup=3, with_http=True, with_profiler=True):
    server = make_server(checkpoint, port=0, device=device, fast=fast)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    worker = server.worker
    report = {}
    for task in tasks:
        bodies = bench_requests(task, limit)
        for body in bodies[:warmup]:
            time_request(worker, body, device)
        records = [time_request(worker, body, device) for body in bodies]
        captures = sum(r["captured"] for r in records)
        p50_all = statistics.median(r["total"] for r in records)
        records = [r for r in records if not r["captured"]] or records  # steady state: capturing requests set aside
        p50 = lambda key: statistics.median(r[key] for r in records)
        summary = {key: p50(key) for key in ("tokens", "pack", "level_total", "head", "softmax", "json", "total")}
        summary.update(captures=captures, p50_with_captures=p50_all)
        depth = max(len(r["levels"]) for r in records)
        summary["levels"] = [statistics.median(r["levels"][d][2] for r in records if len(r["levels"]) > d) for d in range(depth)]
        summary["level_shapes"] = [r["levels"] for r in records[:1]][0]
        if with_http:
            summary["http"] = http_round_trip(server, bodies[0])
            summary["http_overhead"] = summary["http"] - records[0]["total"]
        if with_profiler and torch.device(device).type == "cuda":
            summary.update(cuda_profile(worker, bodies[0], device))
        report[task] = summary
        print(f"{task}: " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items() if k not in ("table", "level_shapes")}))
        if "table" in summary:
            print(summary["table"])
        print("levels of the first request:", summary["level_shapes"])
    server.shutdown()
    return report


def equality(checkpoint, device, tasks, limit, merge=True, graphs=True):
    """Plain path versus fast path on the same requests: max abs probability difference and argmax flips."""
    bodies = [b for task in tasks for b in bench_requests(task, limit)]
    worker = Worker(checkpoint, device, fast=False)
    packs = [worker.pack(Request.from_dict(b)) for b in bodies]
    plain = [worker.submit(p) for p in packs]
    worker.model.backbone.enable_fast(merge, graphs)  # same weights in memory: the plain run is done, the fast path starts here
    return compare(plain, [worker.submit(p) for p in packs])


def accuracy(checkpoint, device, data, limit, fp8=True):
    """The fast path's logits on the first `limit` rows of `data` (a request jsonl with targets), bf16 then with fp8
    GEMMs (enable_fast(fp8=True)) on the same weights: raw and calibrated accuracy and NLL of each, and the max abs
    probability difference and argmax flips between them."""
    from .data import load_requests
    from .metrics import metrics
    from .training import collect_logits
    worker = Worker(checkpoint, device, fast=True)
    requests = load_requests(data)[:limit]
    report = {}
    runs = [("bf16", False)] + ([("fp8", True)] if fp8 else [])
    for name, use_fp8 in runs:
        if use_fp8:
            worker.model.backbone.enable_fast(merge=False, fp8=True)  # merged already; fresh graphs over the fp8 weights
        logits, targets = collect_logits(worker.model, requests)
        report[name] = {"raw": {k: metrics(logits, targets)[k] for k in ("accuracy", "nll")},
                        "calibrated": {k: metrics(logits, targets, worker.temperature)[k] for k in ("accuracy", "nll")},
                        "probabilities": [z.softmax(-1).tolist() for z in logits]}
    if fp8:
        report["fp8_vs_bf16"] = compare([[p] for p in report["bf16"]["probabilities"]], [[p] for p in report["fp8"]["probabilities"]])
    for name in ("bf16", "fp8"):
        report.get(name, {}).pop("probabilities", None)
    return report


def compare(plain, fast):
    worst, flips, questions = 0., 0, 0
    for a, b in zip(plain, fast):
        for pa, pb in zip(a, b):
            questions += 1
            worst = max(worst, max(abs(x - y) for x, y in zip(pa, pb)))
            flips += pa.index(max(pa)) != pb.index(max(pb))
    return {"requests": len(plain), "questions": questions, "max_abs_prob_diff": worst, "argmax_flips": flips}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("profile", "equality", "sweep", "accuracy"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", default="banking77,mmlu-pro")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-http", action="store_true")
    parser.add_argument("--no-profiler", action="store_true")
    parser.add_argument("--no-merge", action="store_true", help="equality: keep the LoRA adapters unmerged")
    parser.add_argument("--no-graphs", action="store_true", help="equality: without the CUDA-graph level passes")
    parser.add_argument("--lengths", default="150,1000,4000,8000,16000", help="sweep: state lengths in tokens")
    parser.add_argument("--profiler", action="store_true", help="sweep: the torch.profiler kernel table per length")
    parser.add_argument("--fp8", action="store_true", help="sweep: fp8 GEMMs (enable_fast(fp8=True))")
    parser.add_argument("--fp4", action="store_true", help="sweep: fp4 expert GEMMs of an NVFP4 MoE checkpoint (enable_fast(fp4=True))")
    parser.add_argument("--batch-window-ms", type=float, default=4, help="sweep: the server's batch window for the HTTP round trip")
    parser.add_argument("--data", default="data/phase1-v1/test.jsonl", help="accuracy: request jsonl with targets")
    args = parser.parse_args(argv)
    tasks = args.tasks.split(",")
    if args.command == "accuracy":
        print(json.dumps(accuracy(args.checkpoint, args.device, args.data, args.limit)))
    elif args.command == "sweep":
        sweep(args.checkpoint, args.device, [int(n) for n in args.lengths.split(",")], args.fast, with_profiler=args.profiler, fp8=args.fp8,
              with_http=not args.no_http, batch_window_ms=args.batch_window_ms, fp4=args.fp4)
    elif args.command == "profile":
        profile(args.checkpoint, args.device, tasks, args.limit, args.fast, with_http=not args.no_http, with_profiler=not args.no_profiler)
    else:
        print(json.dumps(equality(args.checkpoint, args.device, tasks, args.limit, not args.no_merge, not args.no_graphs)))


if __name__ == "__main__":
    main()
