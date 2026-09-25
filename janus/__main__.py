"""CLI for the minimal experiment. Run `python -m janus --help`."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare", help="Prepare source-pinned BANKING77 or offline synthetic data")
    data.add_argument("dataset", choices=("banking77", "synthetic", "study", "phase1", "public", "multilingual", "t3-pilot", "gap-cells", "paraphrase-bank", "t3-volume", "rank", "phase1-rank"))
    data.add_argument("--sources", help="Comma-separated public sources (default: all)")
    data.add_argument("--cells", help="t3-pilot/t3-volume/gap-cells: comma-separated cell names (default: all)")
    data.add_argument("--per-cell", type=int, help="Per cell: t3-pilot generations (default 500), t3-volume generations (default 2000), gap-cells train rows (default 2000)")
    data.add_argument("--train", type=int, help="rank: train rows (default 6000)")
    data.add_argument("--dev", type=int, help="gap-cells: dev rows per cell (default 200); rank: dev rows (default 300)")
    data.add_argument("--calibration", type=int, help="gap-cells: calibration rows per cell (default 200); rank: calibration rows (default 300)")
    data.add_argument("--test", type=int, help="gap-cells: test rows per cell (default 500); rank: test rows (default 1000)")
    data.add_argument("--phase1", default="data/phase1-v1", help="phase1-rank: Phase 1 data directory")
    data.add_argument("--rank", default="data/rank-v1", help="phase1-rank: rank data directory")
    data.add_argument("--workers", type=int, default=8, help="t3-volume/paraphrase-bank: concurrent requests")
    data.add_argument("--output", required=True)
    data.add_argument("--seed", type=int, help="Default 17; t3-volume defaults to 18 because it must differ from the pilot seed")
    data.add_argument("--count", type=int, default=32, help="Synthetic training request count")
    data.add_argument("--no-llm", action="store_true", help="Skip the GPT-5.6 Luna paraphrase slice")
    data.add_argument("--per-family", type=int, default=3000)
    data.add_argument("--paraphrase-fraction", type=float, default=.1)
    train_parser = commands.add_parser("train", help="Train and select the lowest dev-NLL checkpoint")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--dev", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--device")
    for command in ("calibrate", "evaluate"):
        child = commands.add_parser(command)
        child.add_argument("--checkpoint", required=True)
        child.add_argument("--data", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--device", default="cpu")
        child.add_argument("--limit", type=int)
        child.add_argument("--override", action="append", default=[], metavar="KEY=JSON",
                           help="ModelConfig field to override at load, e.g. images=true or batch_states=2 (a smaller card)")
        if command == "evaluate":
            child.add_argument("--calibration")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--request", required=True, help="JSON request file")
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--calibration")
    bench = commands.add_parser("benchmark")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--output", required=True)
    bench.add_argument("--repeats", type=int, default=20)
    comparison = commands.add_parser("compare")
    comparison.add_argument("predictions_a")
    comparison.add_argument("predictions_b")
    baseline = commands.add_parser("zero-shot", help="One-pass pretrained vocabulary-label comparison")
    baseline.add_argument("--data", required=True)
    baseline.add_argument("--calibration-data", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--backbone", default="Qwen/Qwen3-0.6B-Base")
    baseline.add_argument("--revision", default="da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
    baseline.add_argument("--device", default="cuda:0")
    baseline.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    remote = commands.add_parser("remote", help="Evaluate the real Jev API; reads TYPESAFE_API_KEY or env.sh")
    remote.add_argument("--data", required=True)
    remote.add_argument("--output", required=True)
    remote.add_argument("--model", default="jev-1.13.0")
    remote.add_argument("--workers", type=int, default=4)
    remote.add_argument("--limit", type=int)
    remote.add_argument("--env-file", default="env.sh")
    remote.add_argument("--resume", action="store_true")
    remote_cal = commands.add_parser("remote-calibrate", help="Fit Jev temperature using saved calibration responses only")
    for flag in ("calibration-data", "calibration-predictions", "data", "predictions", "output"):
        remote_cal.add_argument("--" + flag, required=True)
    remote_cal.add_argument("--artifact")
    profile = commands.add_parser("profile-panel", help="Warm local request timing including tokenization and packing")
    for flag in ("checkpoint", "data", "output"):
        profile.add_argument("--" + flag, required=True)
    profile.add_argument("--device", default="cpu")
    profile.add_argument("--limit", type=int)
    blocked = commands.add_parser("probe-blocks", help="Blocked null-controlled Jev probes; at most 1,200 calls")
    blocked.add_argument("--data", default="data/study-v1/benchmark.jsonl")
    blocked.add_argument("--output", required=True)
    blocked.add_argument("--model", default="jev-1.13.0")
    blocked.add_argument("--env-file", default="env.sh")
    blocked.add_argument("--seed", type=int, default=17)
    blocked.add_argument("--resume", action="store_true")
    blocked.add_argument("--wave", type=int, choices=(1, 2, 3), default=1, help="1: B0/B2/D1/D3; 2: B1/B3/B4/D4/D5/D7/D1b/B2b; 3: D3 redo with corrected SNLI negation and repeat nulls")
    blocked.add_argument("--predictions", default="runs/study-20260917/janus/predictions.jsonl",
                         help="Wave 2: saved Jev panel predictions used to select interior menus")
    server = commands.add_parser("serve", help="Serve POST /v1/systemone and GET /v1/models over one local checkpoint")
    server.add_argument("--checkpoint", required=True)
    server.add_argument("--calibration")
    server.add_argument("--device", default="cpu")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8080)
    server.add_argument("--model-id", default="janus-local", help="Model identity returned by the server; never a Jev id")
    server.add_argument("--queue-size", type=int, default=32)
    server.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                        help="Serving-only fast paths (merged LoRA, CUDA-graphed level passes); --no-fast is the plain path")
    server.add_argument("--batch-window-ms", type=float, default=4, help="Cross-request batching: wait this long after the first queued request for more (0: drain only)")
    server.add_argument("--batch-max", type=int, default=8, help="Requests per forward_many batch")
    server.add_argument("--rate-limit-rpm", type=float, default=0, help="Requests per minute per bearer token (or client address); 0 is off")
    server.add_argument("--alias", default="", help="Comma-separated extra `model` values accepted, e.g. jev-latest,jev-preview")
    server.add_argument("--overloaded-status", type=int, default=529, help="Status when the queue is full (529 as documented; 503 was the old answer)")
    server.add_argument("--prefix-cache-tokens", type=int, help="State-pass cache capacity in tokens (default 100000 on CUDA, about 3.3 GB on the 4B; 0 elsewhere; 0 disables)")
    server.add_argument("--max-tokens", type=int, help="Override the checkpoint's per-request token budget (and the state-plus-question budget when the checkpoint has none)")
    server.add_argument("--fp8", action="store_true", help="fp8 GEMMs for the projections (CUDA, fast path): faster long states, a measured accuracy change (docs/phase4/latency-track2.md)")
    server.add_argument("--fp4", action="store_true", help="fp4 tensor-core GEMMs for the experts of an NVFP4 MoE checkpoint (Blackwell, fast path): fp4 activations, a measured accuracy change (docs/phase4/a3b-runner-port.md)")
    server.add_argument("--reread-entropy", type=float, nargs="?", const=0.5,
                        help="Ask a multi-option Choice question again with its options reversed and average when its normalised entropy exceeds this (0.5 when given bare; off by default)")
    server.add_argument("--precapture", action=argparse.BooleanOptionalAction, default=True,
                        help="Capture the CUDA graphs of common request shapes (janus/precapture.json) at start-up; --no-precapture skips it")
    args = parser.parse_args(argv)
    if args.command == "serve":
        from .server import serve
        return serve(args.checkpoint, args.host, args.port, args.device, args.calibration, args.model_id, args.queue_size, args.fast,
                     args.batch_window_ms, args.batch_max, args.rate_limit_rpm, args.alias.split(","), args.overloaded_status,
                     args.prefix_cache_tokens, args.max_tokens, args.fp8, args.reread_entropy, args.fp4, args.precapture)
    from .data import write_json
    if args.command == "prepare":
        from .data import prepare_banking, prepare_synthetic
        if args.seed is None:
            args.seed = 18 if args.dataset == "t3-volume" else 17
        if args.dataset == "banking77":
            result = prepare_banking(args.output, args.seed)
        elif args.dataset == "study":
            from .study_data import prepare_study
            result = prepare_study(args.output, seed=args.seed)
        elif args.dataset == "phase1":
            from .synth.build import prepare_phase1
            result = prepare_phase1(args.output, seed=args.seed, per_family=args.per_family,
                                    paraphrase_fraction=args.paraphrase_fraction, llm=not args.no_llm)
        elif args.dataset == "public":
            from .synth.public import prepare_public
            sources = [s for s in args.sources.split(",") if s] if args.sources else None
            result = prepare_public(args.output, sources=sources, seed=args.seed)
        elif args.dataset == "multilingual":
            from .synth.public import prepare_multilingual
            result = prepare_multilingual(args.output, seed=args.seed)
        elif args.dataset == "t3-pilot":
            from .synth.generate import run_pilot
            cells = [c for c in args.cells.split(",") if c] if args.cells else None
            result = run_pilot(args.output, cells=cells, per_cell=args.per_cell or 500, seed=args.seed)
        elif args.dataset == "gap-cells":
            from .synth.gapcells import CELL_NAMES, prepare_gap_cells
            cells = [c for c in args.cells.split(",") if c] if args.cells else CELL_NAMES
            result = prepare_gap_cells(args.output, seed=args.seed, per_cell=args.per_cell or 2000, dev=args.dev or 200,
                                       calibration=args.calibration or 200, test=args.test or 500, cells=cells)
        elif args.dataset == "rank":
            from .synth.rankworlds import prepare_rank
            result = prepare_rank(args.output, seed=args.seed, train=args.train or 6000, dev=args.dev or 300,
                                  calibration=args.calibration or 300, test=args.test or 1000)
        elif args.dataset == "phase1-rank":
            from .synth.rankworlds import prepare_phase1_rank
            result = prepare_phase1_rank(args.output, phase1=args.phase1, rank=args.rank)
        elif args.dataset == "paraphrase-bank":
            from .synth.paraphrase_bank import build_paraphrase_bank
            result = build_paraphrase_bank(args.output, workers=args.workers)
        elif args.dataset == "t3-volume":
            from .synth.volume import VOLUME_CELLS, run_volume
            cells = [c for c in args.cells.split(",") if c] if args.cells else VOLUME_CELLS
            result = run_volume(args.output, cells=cells, per_cell=args.per_cell or 2000, seed=args.seed, workers=args.workers)
            result = {k: v for k, v in result.items() if k != "audit"}
        else:
            prepare_synthetic(args.output, args.count, args.seed)
            result = {"output": args.output, "purpose": "Offline integration only"}
    elif args.command == "train":
        from .training import TrainConfig, train
        config = TrainConfig.from_dict(json.loads(Path(args.config).read_text()))
        for key in ("seed", "max_steps", "device"):
            if getattr(args, key) is not None:
                setattr(config, key, getattr(args, key))
        result = train(args.train, args.dev, args.output, config)
    elif args.command == "calibrate":
        from .evaluation import calibrate
        overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.override)}
        value = calibrate(args.checkpoint, args.data, args.output, args.device, args.limit, **overrides)
        result = {k: value[k] for k in ("temperature", "requests")}
        result.update({"before_nll": value["before"]["nll"], "after_nll": value["after"]["nll"]})
    elif args.command == "evaluate":
        from .evaluation import evaluate
        overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.override)}
        value = evaluate(args.checkpoint, args.data, args.output, args.calibration, args.device, args.limit, **overrides)
        result = {"output": args.output, "requests": value["requests"],
                  "raw": {k: value["raw"][k] for k in ("accuracy", "nll", "brier", "ece")},
                  "calibrated": {k: value["calibrated"][k] for k in ("accuracy", "nll", "brier", "ece")},
                  "uniform_comparison": value["uniform_comparison"]}
    elif args.command == "predict":
        from .evaluation import predict
        from .schema import Request
        request = Request.from_dict(json.loads(Path(args.request).read_text()))
        result = predict(args.checkpoint, request, args.device, args.calibration)
    elif args.command == "benchmark":
        from .benchmark import benchmark
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        result = benchmark(args.checkpoint, args.device, args.repeats)
        write_json(args.output, result)
    elif args.command == "zero-shot":
        from .baseline import evaluate_baseline
        result = evaluate_baseline(args.data, args.calibration_data, args.output,
                                   args.backbone, args.revision, args.device, args.dtype)
    elif args.command == "remote":
        from .remote import evaluate_remote
        result = evaluate_remote(args.data, args.output, model=args.model, workers=args.workers,
                                 limit=args.limit, env_file=args.env_file, resume=args.resume)
    elif args.command == "remote-calibrate":
        from .remote_calibration import calibrate_remote
        value = calibrate_remote(args.calibration_data, args.calibration_predictions, args.data,
                                 args.predictions, args.output, args.artifact)
        result = {k: value[k] for k in ('temperature', 'requests')}
    elif args.command == "profile-panel":
        from .benchmark import profile_panel
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        result = profile_panel(args.checkpoint, args.data, args.device, args.limit)
        write_json(args.output, result)
        result = {k: v for k, v in result.items() if k != 'per_request'}
    elif args.command == "probe-blocks":
        from .blocked_probes import run_blocked_probes
        tokenizer, floor_sources = None, ()
        if args.wave == 2:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
            floor_sources = ("runs/phase1/jev-probes/cache", "runs/study-20260917/janus/cache", Path(args.output) / "cache")
        value = run_blocked_probes(args.data, args.output, model=args.model, env_file=args.env_file,
                                   seed=args.seed, resume=args.resume, wave=args.wave, tokenizer=tokenizer,
                                   floor_sources=floor_sources, predictions_path=args.predictions)
        result = {k: value[k] for k in ("jobs", "http_calls", "failed_jobs", "usage", "latency_p50_seconds")}
    else:
        from .evaluation import compare_runs
        result = compare_runs(args.predictions_a, args.predictions_b)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
