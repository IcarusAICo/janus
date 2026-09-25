# Competitor head-to-head: JevK5 and Laya on our data

Two open competitors, each in its own venv (never the main env), on the same request files we evaluate on.

| competitor | source | weights | limits (honoured, never hidden) |
|---|---|---|---|
| JevK5 | `allebee/jevk5@01d3cc9` (0.2.0) | `alibiserikbay/JevK5` (Qwen3.5-4B merged LoRA, bf16, ~9 GB) | ≤16 options, ≤16,384 input tokens: refused beyond. Choice needs ≥2 options (its server's validation). One forward pass per question. |
| Laya | PyPI `laya[serve]==0.3.11` | `convaiinnovations/laya` bundle: english (512 tok), multilingual (1024), typed-decisions (1024) | Truncates the state silently to its context; we flag every truncated question (`truncated: true`). All questions of a request in one forward pass. |

Neither reads images: image states are recorded as refusals (`image state unsupported`).

## Files

- `setup_venvs.sh`: `.venvs/jevk5` and `.venvs/laya` (Python 3.12, torch 2.11 cu128, which covers the 5090's sm_120 and the 3090's sm_86), and the weights in `.cache/huggingface/hub`. Already run; rerunning is idempotent.
- `run_competitor.py`: runs inside a competitor venv. It sends each of our questions as a client would (our question JSON without `target`) through the competitor's Python API, and writes one row per question: `probabilities` in OUR option order (choice keys, score 0..k-1, noul `[false, true]`), `target`, `group_id`, `family`, `refused` (reason, or null), and `latency_ms` for the request. Latency is synchronised wall time around the API call, after model load and after `--warmup` untimed requests. A `<output>.meta.json` records the model, GPU, load time and refusal/truncation counts. `--limit N` takes the same seeded sample as `python -m janus evaluate --limit N`.
- `score.py` (main env): accuracy, NLL, Brier and ECE via `janus.metrics`, overall, per `family`, and per locale (`massive:<locale>`, `xnli:<lang>`). It reports two views: `answered` (only the questions the competitor answered) and `all` (a refusal counts as uniform). It also gives per-request latency (median/p90/mean) and the share of truncated inputs.
- `jevbench_competitor.sh`: runs `jevk5-serve` / `laya-serve` (their own TypeSafe-style `/v1/systemone` servers) and the public easy/original/hard JevBench tasks through `--adapter typesafe`, with the same cap as `scripts/a3b_local_eval.sh`. Writes `runs/jevbench/<name>.jsonl` and `<name>-summary.json`.

## Final run (GPU)

Run from the repo root. `DEV` is the card that our own numbers are timed on, for comparable latency (the a3b eval uses `cuda:0` = 5090).

```bash
export HF_HUB_CACHE=$PWD/.cache/huggingface/hub HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
DEV=cuda:0; O=runs/competitors
for c in jevk5 laya; do
  P=.venvs/$c/bin/python
  $P scripts/competitors/run_competitor.py $c data/hardtier-v2/test.jsonl        $O/$c/hardtier_v2.jsonl    --device $DEV
  $P scripts/competitors/run_competitor.py $c data/hardtier-judge-v2/test.jsonl  $O/$c/judge_v2.jsonl       --device $DEV
  $P scripts/competitors/run_competitor.py $c data/public-v1/test.jsonl          $O/$c/public.jsonl         --device $DEV --limit 1500
  $P scripts/competitors/run_competitor.py $c data/multilingual-v1/test_massive51.jsonl $O/$c/massive51.jsonl --device $DEV
  $P scripts/competitors/run_competitor.py $c data/multilingual-v1/test_xnli15.jsonl    $O/$c/xnli15.jsonl    --device $DEV
done
# Optional: Laya pinned to one checkpoint instead of its router (router is its documented default):
#   .venvs/laya/bin/python scripts/competitors/run_competitor.py laya data/multilingual-v1/test_xnli15.jsonl $O/laya-multilingual/xnli15.jsonl --device $DEV --model multilingual
python scripts/competitors/score.py $O/*/*.jsonl --json $O/scores.json

# JevBench public (231 items) through each competitor's own server; GPU is the CUDA index (PCI order).
GPU=0 scripts/competitors/jevbench_competitor.sh jevk5 laya
```

`--limit 1500` on `public-v1` matches `a3b_local_eval.sh` (`ev public ... --limit 1500`); drop it for the full 5,116 rows. Our side of each comparison is `runs/phase4/.../<set>/metrics.json` from `python -m janus evaluate`. `score.py` also reads our `predictions.jsonl` (same `probabilities`/`target`/`family`/`group_id` fields), but those probabilities are uncalibrated. Use our calibrated metrics for the headline and `score.py` only for per-locale slices.

## Things to watch on the GPU run

- **JevK5 kernels.** On GPU it uses `flash-linear-attention` (Triton) for the Qwen3.5 linear-attention layers. `causal_conv1d` is not installed (it is not in JevK5's `fast` extra and needs an nvcc build), so transformers uses its PyTorch fallback for that op. Its README quotes ~13 ms on an H100; if our latency is far off that, `uv pip install -p .venvs/jevk5/bin/python causal-conv1d` is the first thing to try, and note it either way. At start-up JevK5 records CUDA graphs for 13 padded lengths up to 4,096 tokens, which takes a minute and extra memory; inputs longer than that run eager.
- **JevK5 on CPU** cannot use fla's Triton kernels ("0 active drivers"): the CPU smoke hid `fla` (`sys.modules['fla'] = None`). This does not matter on GPU.
- **Laya truncation is large on our long sets.** On a 50-row public sample, 45% of questions had their state cut (arena/helpsteer/tabfact). Report `truncated_share` next to its accuracy.
- **Laya rounding.** Laya rounds returned probabilities to 4 decimals, which gives infinite NLL on a confidently wrong 0.0000. `run_competitor.py` shadows `round` inside `laya.agent` so that we score its exact calibrated values (a `ponytail:` comment marks this). The served API (JevBench path) still rounds, as a real client sees it.
- **Laya calibration warning.** The English checkpoint ships a `choice:11+` temperature (0.10) outside its own valid range, and Laya clamps it to 0.5 with a warning. That is Laya's behaviour; we leave it alone.
- **Laya routing.** The router sends short Latin-script non-English states to the English checkpoint (e.g. the French MASSIVE row "musique collectif cieux ouvert"). The chosen checkpoint is saved per row (`checkpoint`) so per-locale results can be explained.
- **Refusals.** Questions with more than 16 options (MASSIVE intent menus can have 20+) and 1-option choices are refused by JevK5 and scored as uniform in the `all` view. Compare coverage as well as accuracy.
- **JevBench ledger.** The typesafe adapter charges Jev's list price in the ledger (about $4.6 per 231 items). No money is spent, but the `--cap-usd 100` still applies. `jevk5-serve` has no token limit of its own (the 16,384 limit is in its in-process adapter), and it rejects more than 16 options with a 400, which JevBench records as a failure.
- **Ports.** 8121 (jevk5) and 8122 (laya); override with `PORT_JEVK5` / `PORT_LAYA`. The script skips a competitor whose `runs/jevbench/<name>.jsonl` already exists.
