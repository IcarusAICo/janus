# Release results (generated from run files, 2026-09-24)

## Held-out sets (calibrated accuracy / ECE)

Our models: `runs/.../<set>/metrics.json`. Competitors: `runs/competitors/scores.json` (their own APIs, the same request
files, RTX 3090). hardtier_v2, judge_v2 and public test come from our own generators and pools (held-out states), so they
favour us; for janus-4b they are also selection numbers (release/FACTS.md). MASSIVE-51 and XNLI-15 are official test splits.
JevK5 refuses MASSIVE's 20-option items (its limit is 16 options).

| set | janus-0.8b | janus-4b (v3-instruct, recommended) | janus-4b (distilled, not chosen) | JevK5 | Laya |
|---|---:|---:|---:|---:|---:|
| hardtier_v2 | 0.672 / 0.019 | 0.828 / 0.017 | 0.797 / 0.018 | 0.671 / 0.073 | 0.420 / 0.208 |
| judge_v2 | 0.530 / 0.154 | 0.870 / 0.074 | 0.740 / 0.092 | 0.670 / 0.123 | 0.470 / 0.217 |
| public | 0.673 / 0.028 | 0.728 / 0.031 | 0.709 / 0.026 | 0.630 / 0.041 | 0.543 / 0.140 |
| massive51 | 0.619 / 0.023 | 0.767 / 0.028 | 0.815 / 0.020 | refused | 0.272 / 0.173 |
| xnli15 | 0.653 / 0.064 | 0.757 / 0.045 | 0.767 / 0.027 | 0.608 / 0.165 | 0.522 / 0.051 |

## JevBench public (231 items; report only, never used to select)

All systems served on the same RTX 3090 through their own `/v1/systemone` servers and JevBench's typesafe adapter.
janus-4b-final / janus-0.8b-final: the release checkpoints (v3-instruct and the distilled 0.8B) on the final serving code
(main 5b1bb22); janus-4b-distill: the 4B candidate not chosen (earlier serving code).

| system | easy | standard | hard | all | hard ECE |
|---|---:|---:|---:|---:|---:|
| jevk5 | 1.000 | 0.972 | 0.730 | 0.861 | 0.062 |
| laya | 0.958 | 0.694 | 0.333 | 0.576 | 0.206 |
| janus-4b-final | 1.000 | 0.972 | 0.712 | 0.853 | 0.097 |
| janus-0.8b-final | 1.000 | 0.708 | 0.423 | 0.632 | 0.166 |
| janus-4b-distill | 1.000 | 0.972 | 0.649 | 0.823 | 0.103 |

## v1.4 what-if (NOT an official score)

sealed accuracy assumed 0.31, GPU $0.69/h, judge tier held out (not in public items)

| system | public acc | I (v1.3) | C (v1.3) | S | cost | v1.3 score | I (v1.4 what-if) | **v1.4 what-if** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| jevk5 | 0.861 | 81.4 | 84.0 | 86.0 | 57.8 | 76.4 | 45.9 | **53.7** |
| laya | 0.576 | 39.8 | 64.0 | 94.6 | 85.3 | 42.7 | 31.8 | **23.4** |
| janus-4b-final | 0.853 | 80.3 | 82.3 | 85.7 | 57.0 | 75.4 | 45.8 | **53.0** |
| janus-0.8b-final | 0.632 | 47.2 | 70.8 | 91.5 | 70.7 | 60.8 | 35.5 | **29.9** |
| janus-4b-distill | 0.823 | 76.4 | 82.2 | 85.6 | 57.7 | 74.6 | 45.4 | **52.2** |

For reference, the official v1.4 board: Jev 1.13.0 63.29, JevK5 v0.2.0 62.04 (their sealed accuracy 0.331).

JevBench HTTP latency (3090, p50 / p95 / max s): JevK5 0.041 / 0.471 / 0.639; Laya 0.013 / 0.024 / 0.211; janus-4b 0.050 /
0.456 / 2.822; janus-0.8b 0.016 / 0.122 / 1.957. Our maxima are first-time CUDA graph captures of a new request shape.

## Latency (median / p90 ms per request; warm, one request at a time, final serving code)

Ours: `scripts/competitors/time_janus.py --limit 300` (runs/final/<card>/latency_janus.txt). Competitors: their own Python
APIs through `scripts/competitors/run_competitor.py` (3090: full files, runs/competitors/<name>/; 5090: the same seeded
300-row samples, runs/final/0/). fanout8 = 100 hardtier_v2 states with 8 questions each (runs/competitors/fanout8.jsonl).
Laya truncates long states to its context (26-35% of the long-state requests), which also shortens its latency there.

RTX 3090:

| request | janus-4b | JevK5 | janus-0.8b | Laya |
|---|---:|---:|---:|---:|
| 1 short question (XNLI) | 46.4 / 47.1 | 40.9 / 41.4 | 11.4 / 12.3 | 6.3 / 8.9 |
| public test (1-2 questions) | 69.9 / 171.3 | 82.0 / 322.1 | 18.5 / 39.9 | 14.5 / 31.4 |
| 1 question, long state (hardtier_v2) | 74.4 / 453.2 | 80.6 / 647.7 | 17.0 / 111.5 | 14.9 / 23.8 |
| 8 questions, one long state | 304.5 / 818.9 | 641.7 / 5194.8 | 81.6 / 178.9 | 66.6 / 110.9 |

RTX 5090:

| request | janus-4b | JevK5 | janus-0.8b | Laya |
|---|---:|---:|---:|---:|
| 1 short question (XNLI) | 17.2 / 17.9 | 18.7 / 19.6 | 5.9 / 6.8 | 6.1 / 7.4 |
| public test (1-2 questions) | 28.1 / 59.7 | 40.3 / 117.8 | 9.5 / 18.0 | 9.1 / 14.8 |
| 1 question, long state (hardtier_v2) | 27.3 / 176.8 | 32.6 / 247.0 | 8.4 / 60.8 | 9.1 / 13.9 |
| 8 questions, one long state | 122.5 / 289.0 | 251.8 / 1964.4 | 40.6 / 105.8 | 26.0 / 56.4 |

Our p90 on long states includes first-time CUDA graph captures of new request shapes (warm-up covers only common
shapes); capturing more shapes at server start is the known fix, not yet done.

## janus-4b fallback temperature (decided 2026-09-24 on our calibration and test sets; JevBench rows are reporting only)

The fallback temperature (used for every request without a calibrated family, JevBench's included) was refit on the
hard-tier calibration rows only (data/hardcal-v1, 1,028 requests / 1,120 questions from the hardtier, hardtier2 and
judge pools; disjoint from training): 1.1025 -> 1.0239. Per-family temperatures unchanged. Re-scored offline from saved
logits (our sets) and by exact re-tempering of the served probabilities (JevBench: p^(T_old/T_new), renormalised):

| set | accuracy | ECE, T 1.1025 | ECE, T 1.0239 |
|---|---:|---:|---:|
| hardtier_v2 test | 0.828 | 0.017 | 0.014 |
| judge_v2 test | 0.870 | 0.074 | 0.068 |
| public test | 0.728 | 0.031 | 0.033 |
| MASSIVE-51 | 0.767 | 0.028 | 0.039 |
| XNLI-15 | 0.757 | 0.045 | 0.055 |
| JevBench standard (72) | 0.972 | 0.020 | 0.015 |
| JevBench hard (111) | 0.712 | 0.097 | **0.071** |

Shipped in runs/publish/hf/janus-4b-v3i/calibration.json (`temperature_note` records the change). Accuracy is unchanged
by any temperature.

## janus-35b-a3b fallback temperature: kept (decided 2026-09-24 on our sets)

The same hard-tier refit for the A3B (data/hardcal-v1 through the NVFP4 checkpoint) gives T 1.0437 -> 1.1945. On our
sets it helps hardtier_v2 (ECE 0.021 -> 0.019) and public test (0.030 -> 0.028) but hurts judge_v2 (0.047 -> 0.119), so
the A3B keeps its fitted 1.0437. For reporting only: JevBench standard ECE would go 0.036 -> 0.045 and hard 0.069 ->
0.049; JevBench did not enter the decision.

## janus-0.8b release checkpoint (distill-v2, no XNLI, label smoothing 0.1; step 1,500 by the usual rule)

Replaces the earlier 0.8B export. runs/publish/qwen35_08b_distill_v2 -> runs/publish/hf/janus-0.8b-v2. Calibrated
accuracy / ECE (earlier 0.8B in brackets): hardtier_v2 0.759 / 0.032 (0.672 / 0.019); judge_v2 0.680 / 0.049 (0.530 /
0.154); public test 0.691 / 0.028 (0.673 / 0.028); MASSIVE-51 0.635 / 0.059 (0.619 / 0.023); XNLI-15 0.655 / 0.044
(0.653 / 0.064; no XNLI in training). JevBench public (3090, runs/jevbench/janus-0.8b-v2.jsonl): easy 1.000, standard
0.778, hard 0.613, all 0.745, hard ECE 0.072; v1.4 what-if 45.8 (Laya 23.4).
Fallback temperature kept at 0.7290: a hard-tier refit (0.8062) helps MASSIVE/XNLI but hurts judge_v2 (0.049 -> 0.115)
and public (0.028 -> 0.041), the same pattern as the A3B.

## Final latency (release checkpoints on the final serving code: precapture on, OOM release-and-retry)

Median / p90 ms per request, warm, one at a time. Ours: `scripts/competitors/time_janus.py --limit 300` on the HF export
folders (runs/final2/<card>/latency.txt; the 4B long-state/8-question rows on the 3090 and the A3B's on the 5090 were
re-run after the OOM fix). Competitors: unchanged from the tables above (their own APIs). Our long-state p90 includes
re-captures after a memory release on the smaller cards.

RTX 5090:

| request | janus-4b | JevK5 | janus-0.8b | Laya | janus-35b-a3b |
|---|---:|---:|---:|---:|---:|
| 1 short question (XNLI) | 17.3 / 18.1 | 18.7 / 19.6 | **5.0 / 5.4** | 6.1 / 7.4 | 35.1 / 37.2 |
| public test (1-2 questions) | 27.9 / 55.7 | 40.3 / 117.8 | **8.7** / 14.4 | 9.1 / 14.8 | 50.9 / 79.1 |
| 1 question, long state (hardtier_v2) | 27.0 / 161.6 | 32.6 / 247.0 | **7.9** / 44.4 | 9.1 / 13.9 | 48.5 / 236.7 |
| 8 questions, one long state | 107.0 / 271.4 | 251.8 / 1964.4 | 34.3 / 88.5 | **26.0 / 56.4** | 205.6 / 381.3 |

RTX 3090 (the A3B needs Blackwell):

| request | janus-4b | JevK5 | janus-0.8b | Laya |
|---|---:|---:|---:|---:|
| 1 short question (XNLI) | 47.0 / 47.5 | 40.9 / 41.4 | 10.7 / 11.2 | **6.3 / 8.9** |
| public test (1-2 questions) | 69.7 / 149.4 | 82.0 / 322.1 | 16.9 / 32.6 | **14.5 / 31.4** |
| 1 question, long state (hardtier_v2) | 75.0 / 462.1 | 80.6 / 647.7 | 16.4 / 94.3 | **14.9 / 23.8** |
| 8 questions, one long state | 337.8 / 857.3 | 641.7 / 5194.8 | 73.4 / 169.6 | **66.6 / 110.9** |

Laya truncates long states to its context (26-35% of the long-state requests), which also shortens its latency there.

## Correction: board-faithful what-if (2026-09-25), supersedes the 54.4 vs 53.7 comparison

JevBench v1.4.2 (upstream 2026-09-24) times self-hosted rows on an RTX PRO 6000 Blackwell (standard + judge requests,
x2 + 0.15 s) and prices Cost as the base model's hosted list price x the server's usage.input_tokens (4B dense: DeepInfra
Qwen3.5-4B $0.03/M). Our earlier what-if used GPU-hours x latency on the RTX 3090, which favoured our latency. Re-run on
the RTX 5090 (closest local card) and scored that way (scripts/jevbench_board_whatif.py; sealed 0.31 assumed; judge tier
held out; not an official score):

| system | public acc | hard | hard ECE | input tokens | S | Cost | **what-if** |
|---|---:|---:|---:|---:|---:|---:|---:|
| JevK5 (runs/jevbench/jevk5-5090) | 0.870 | 0.748 | 0.038 | 699 | 94.6 | 60.4 | **55.7** |
| janus-4b (janus-4b-5090) | 0.853 | 0.712 | 0.077 | 645 | 94.3 | 61.4 | **55.3** |
| janus-0.8b (janus-0.8b-5090) | 0.745 | 0.613 | 0.099 | 645 | 95.4 | 61.4 | **43.3** |

JevK5 on the 5090 answers two more public hard items than on the 3090 (0.748 vs 0.730) and is better calibrated there:
card-to-card bf16 differences of about +-2 items. On this estimate janus-4b is slightly behind JevK5. Official v1.4.2
board: decider-4b v2 64.13, Jev 1.13.0 63.29, JevK5 62.04, Cygnet 61.76.
