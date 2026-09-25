<p align="center">
<pre align="center">
     __                    
 __ / /___ _ ___  __ __ ___
/ // // _ `// _ \/ // /(_-<
\___/ \_,_//_//_/\_,_//___/
</pre>
  <p align="center"><strong>Open typed-decision models: calibrated answers in one forward pass.</strong></p>
</p>

Janus is an independent reconstruction of a "System 1" decision model, built from public descriptions of the
interface. Janus answers typed questions about a piece of state: a ticket, a log, a policy, a JSON record, a long document.
Ask any number of `choice`, `score` and `noul` (yes/no) questions. In **one forward pass**, it returns a calibrated
probability for every option of every question, and generates no tokens. There is nothing to parse and no decoding
loop. The state is encoded once however many questions ride on it, and each question is answered independently of the
others. A `choice` can have up to 255 options. Requests can run to 16,384 tokens in the Docker image, and the limit is
adjustable. Serving is fast: one short question takes about 5 ms on Janus 0.8B and 17 ms on Janus 4B on an RTX 5090,
with CUDA graphs captured at start-up, requests batched together, and a cache that encodes a repeated state once. There
are three sizes (0.8B, 4B and a 35B mixture of experts), and the smallest is trained on data in 51 locales. The weights run
on your own GPU or CPU, offline in a pinned Docker image, and the included server accepts the `POST /v1/systemone`
request shape.

| | |
| --- | --- |
| **Weights** | [`TODO/janus-4b`](https://huggingface.co/TODO/janus-4b) (Qwen3.5-4B + LoRA r16) and [`TODO/janus-0.8b`](https://huggingface.co/TODO/janus-0.8b) (Qwen3.5-0.8B + LoRA r64) and [`TODO/janus-35b-a3b`](https://huggingface.co/TODO/janus-35b-a3b) (Qwen3.6-35B-A3B + LoRA r16, served on NVIDIA's NVFP4 quantization; needs a 32 GB Blackwell GPU), each with a pointer decision head |
| **Readout** | The state is encoded once; each question is an isolated branch; a pointer head scores every option in context. Up to 255 options |
| **Calibration** | Temperatures fitted after training on held-out requests: global, per option count, per task family |
| **Training** | Synthetic decision families and public datasets converted to typed requests. Janus 0.8B: 57,304 requests including 51-locale MASSIVE, distilled from Janus 35B-A3B. Janus 35B-A3B: 53,224 English requests, one epoch. Janus 4B: 51,996 English requests, gold targets, one epoch |
| **Serving** | `pip install` or one Dockerfile; offline at run time; CUDA graphs on GPU; also runs on CPU |

## Results

**JevBench public items.** The 231 public items of [JevBench](https://github.com/fstandhartinger/jevbench), run
through JevBench's own runner and `typesafe` adapter. These are our runs, not an official JevBench score: the
official v1.4 score also pools 308 sealed items that only the maintainers hold. Every released checkpoint is the one
our selection rules picked from our own data. One exception to "report only": at our request, a candidate 4B
(trained further on new multi-step data) was run on these items before we settled on Janus 4B. It scored lower (hard
0.703) and was not chosen; our pre-set procedure had already kept Janus 4B.

| split | n | Janus 0.8B | Janus 4B | Janus 35B-A3B | JevK5 v0.2 | Laya 0.3.11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | 48 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 |
| standard | 72 | 0.778 | 0.972 | **0.986** | 0.972 | 0.694 |
| hard (public half) | 111 | 0.613 | 0.712 | **0.784** | 0.730 | 0.333 |
| all | 231 | 0.745 | 0.853 | **0.892** | 0.861 | 0.576 |
| hard-tier ECE (lower is better) | | 0.072 | 0.071 | 0.069 | **0.062** | 0.206 |

A v1.4 what-if with an assumed sealed accuracy of 0.31 for every system (JevK5's official sealed accuracy is 0.331)
puts Janus 4B at 55.3 and JevK5 at 55.7 (list-price cost and timing on an RTX 5090, as the board scores
self-hosted rows; Janus 0.8B 43.3; release/RESULTS.md). It is not an official score, and it cannot see the sealed items
that decide most of the official one. On the official board, JevK5 v0.2.0 scores
62.04.

**Multilingual, official test splits.** Accuracy / ECE.

| set | Janus 0.8B | Janus 4B | Janus 35B-A3B | JevK5 v0.2 | Laya 0.3.11 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MASSIVE, 51 locales, 20-way intent | 0.635 / 0.059 | **0.767** / 0.039 | TODO | refused (its limit is 16 options) | 0.272 / 0.173 |
| XNLI, 15 languages | 0.655 / 0.044 | **0.757** / 0.055 | TODO | 0.608 / 0.165 | 0.522 / 0.051 |

**Latency.** Median / p90 ms per request, warm, one request at a time, on the final serving code (start-up pre-capture on).
Every system was served by its own code on the same card. Laya truncates long states to its context (26-35% of
long-state requests), which also shortens its latency there. On the smaller cards Janus's long-state p90 includes
graph re-captures after a memory release.

| request | card | Janus 35B-A3B | Janus 4B | JevK5 v0.2 | Janus 0.8B | Laya 0.3.11 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 short question (XNLI) | RTX 5090 | 35.1 / 37.2 | 17.3 / 18.1 | 18.7 / 19.6 | 5.0 / 5.4 | 6.1 / 7.4 |
|  | RTX 3090 | n/a (needs Blackwell) | 47.0 / 47.5 | 40.9 / 41.4 | 10.7 / 11.2 | 6.3 / 8.9 |
| public-dataset test (1-2 questions) | RTX 5090 | 50.9 / 79.1 | 27.9 / 55.7 | 40.3 / 117.8 | 8.7 / 14.4 | 9.1 / 14.8 |
|  | RTX 3090 | n/a (needs Blackwell) | 69.7 / 149.4 | 82.0 / 322.1 | 16.9 / 32.6 | 14.5 / 31.4 |
| 1 question, long state (hard-tier v2) | RTX 5090 | 48.5 / 236.7 | 27.0 / 161.6 | 32.6 / 247.0 | 7.9 / 44.4 | 9.1 / 13.9 |
|  | RTX 3090 | n/a (needs Blackwell) | 75.0 / 462.1 | 80.6 / 647.7 | 16.4 / 94.3 | 14.9 / 23.8 |
| 8 questions, one long state | RTX 5090 | 205.6 / 381.3 | 107.0 / 271.4 | 251.8 / 1,964.4 | 34.3 / 88.5 | 26.0 / 56.4 |
|  | RTX 3090 | n/a (needs Blackwell) | 337.8 / 857.3 | 641.7 / 5,194.8 | 73.4 / 169.6 | 66.6 / 110.9 |

Peak GPU memory in bf16 on the RTX 3090, before start-up pre-capture (which reserves more memory): Janus 0.8B 1.8 GB (150-token state) to 2.2 GB (4k tokens); Janus 4B 9.0 GB
to 9.9 GB. Janus 35B-A3B on an RTX 5090 (NVFP4 experts; measured 2026-09-21): 22.7 GB (150 tokens) to 23.4 GB (4k),
26.0 GB at 16k.

## Install and use

Python 3.11+, PyTorch 2.6+. A CUDA GPU is recommended; everything also runs on CPU, more slowly.

```bash
pip install "janus @ git+https://github.com/TODO/janus@TODO"
```

```python
import janus

m = janus.load("TODO/janus-4b", device="cuda")  # a Hugging Face repo id or a local directory

response = m.predict(
    "Refunds need a receipt and a purchase within 30 days. The customer bought 12 days ago and has no receipt.",
    {
        "refund": {"type": "noul", "instructions": "Is a refund permitted under the policy?"},
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Payments and refunds", "support": "Anything else"}},
        "tone": {"type": "score", "instructions": "How upset is the customer?",
                 "criteria": ["calm", "concerned", "angry"]},
    },
)
response["answers"]["refund"]["noul"]            # p(true)
response["answers"]["route"]["probabilities"]    # {"billing": ..., "support": ...}
response["answers"]["tone"]["score"]             # expected level, 0 to 2
```

`predict` returns exactly what the HTTP server returns: `{"model", "answers", "usage"}`, with `output_tokens`
always 0. `m.predict_batch([request, ...])` takes a list of `/v1/systemone` bodies (`{"state": ..., "questions":
...}`) and runs them through shared batched passes, with the same answers as one request at a time. A request whose
`group_id` starts with a calibrated family (`"tabfact:row-17"`, listed in each model card) gets that family's
temperature.

### Server

```bash
hf download TODO/janus-4b --local-dir janus-4b
JANUS_SERVER_TOKEN=choose-a-secret python -m janus serve \
  --checkpoint janus-4b/model.pt --calibration janus-4b/calibration.json \
  --model-id janus-4b --device cuda --port 8080

curl -s localhost:8080/v1/systemone -H "Authorization: Bearer choose-a-secret" -d '{
  "model": "janus-4b",
  "state": "Order #7120 shows delivered to No. 17; the customer lives at No. 71.",
  "questions": {"what": {"type": "choice", "instructions": "What happened to the parcel?",
                         "criteria": {"delivered": null, "misdelivered": null, "unknown": null}}}}'
```

The server implements `POST /v1/systemone` and `GET /v1/models`, bearer authentication, per-token rate limiting,
cross-request batching, and a prefix cache that pays for a repeated state once. Clients written for the public
`/v1/systemone` API can point their base URL at it; `--alias` accepts other `model` names.

### Docker

```bash
docker build -t janus-serve:4b --build-arg MODEL_REPO=TODO/janus-4b --build-arg MODEL_REVISION=TODO .
docker run --rm --gpus all -p 127.0.0.1:8080:8080 janus-serve:4b
```

The build downloads the pinned checkpoint and Qwen backbone, and the container runs offline. The
[Dockerfile](Dockerfile) header has the CPU build and the options.

## How it works

```
State ── shared causal prefix ─┬─ question + all options + decision ─ softmax over options
                               ├─ question + all levels  + decision ─ distribution and expected level
                               └─ proposition + false/true + decision ─ p(true)
```

- **Shared state prefix.** The state is encoded once per request, however many questions it carries.
- **Isolated question branches.** Each question reads the state and its own tokens only. Position ids restart at the
  end of the state for every branch.
- **Pointer head.** A small head compares the decision position of each branch with each option's representation.
  Nothing is hard-coded in the output layer.
- **Calibration.** The logits are divided by a temperature fitted after training on held-out data: one global, one per
  task family and one per option-count bucket. They are stored in `calibration.json`, bound to the checkpoint by
  its sha256.

## How it was trained

- **Data.** Janus 0.8B: 57,304 training requests (`data/distill-v2`). Janus 4B: the same without the browser and
  multilingual rows (51,996). Janus 35B-A3B: the same without the multilingual rows (53,224). The mix: synthetic decision families (hard-tier policies, multi-step lookups, dates and
  numbers, probability, trade-offs, answer judging), public classification, NLI, preference, table-fact,
  tool-selection and routing sets converted to typed requests, long-context and many-option rows, and
  4,080 multilingual rows from MASSIVE's train split (51 locales). No released model trained on XNLI; its scores
  are zero-shot. Every row is deduplicated by
  state and kept disjoint from evaluation states. Some synthetic families are generated by code with computed
  answers. The LLM-written ones (hard tier, answer judging, broad decision tasks) were written by GPT-5.6 Luna and
  reviewed by GPT-5.6 Terra. The hard-tier generators reject any state that shares a 12-word sequence with a public
  JevBench item.
- **Distillation** (Janus 0.8B only). Janus 35B-A3B labelled every non-multilingual
  training request with its calibrated distribution (`scripts/teacher_label.py`). A one-hot gold target becomes
  `0.5 * gold + 0.5 * teacher`; targets that are already distributions are kept (`scripts/build_distill.py`).
  Multilingual rows keep their gold.
- **Adapter.** LoRA on the attention and Gated DeltaNet projections plus the pointer head, trained with
  cross-entropy against the targets. Configs: [configs/publish/](configs/publish/) and
  [configs/phase4_production/qwen35_4b_prod_v3_instruct.json](configs/phase4_production/qwen35_4b_prod_v3_instruct.json).
  - **Janus 0.8B.** One epoch from scratch, 1,791 steps of 32 requests, with label smoothing 0.1; step 1,500 was
    selected (local RTX 3090, about 5.2 h).
  - **Janus 35B-A3B.** `data/production-v4` (53,224 requests, gold targets only, no multilingual rows), trained in bf16
    against the unquantised Qwen3.6-35B-A3B and served on NVIDIA's NVFP4 experts. LoRA on the projections only, never
    the experts. One epoch in two runs on a rented H200, 48 requests per step: 827 steps stopped by a 6-hour cap, then
    282 steps over the remaining 13,528 requests. Step 282 of the second run was selected (8.0 h in total).
  - **Janus 4B** (v3-instruct). `data/production-v3`: 51,996 requests, gold targets only, no multilingual or browser
    rows. One full epoch of 1,625 steps with an option-order consistency term; step 1,500 (RTX 5090, about 10.4 h).
    - It was first chosen over a partial-epoch distilled 4B on our three internal held-out splits (a tie on dev). So
      for Janus 4B those splits are selection numbers. JevBench did not enter that choice (the distilled run had no
      JevBench numbers yet).
    - Distillation was then tested at a full epoch, under a rule agreed beforehand. It was better on dev, tied on
      hard-tier v2 and public test, and was worse on judge v2, so it was dropped for the 4B.

  Checkpoints were selected by a rule fixed before training (dev NLL plus a held-out confident-error penalty), and
  never on JevBench items.
- **Calibration.** Temperatures fitted after selection on held-out requests (3,412 for the distilled mix) that share
  no state or group with training or selection data.

Export a checkpoint to a Hugging Face layout with `python scripts/export_hf.py CHECKPOINT CALIBRATION OUT_DIR`.

## Tests

```bash
pip install -e '.[dev]'
python -m pytest tests -q   # CPU, with a tiny random model
```

## License and credits

Code: Apache-2.0 ([LICENSE](LICENSE)). Adapter weights: TODO, pending the training-data licence decision
([release/DATA-LICENSES.md](release/DATA-LICENSES.md)). The base models
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B),
and Qwen3.6-35B-A3B (Janus 35B-A3B's base, run as
[nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)), are Apache-2.0.

Training data, with thanks to its authors. Each keeps its own licence; the per-source audit is in DATA-LICENSES.md.
- [MASSIVE](https://github.com/alexa/massive) (Amazon, CC BY 4.0)
- [XNLI](https://github.com/facebookresearch/XNLI) (Meta, CC BY-NC 4.0; evaluation only, no released model trained on it)
- [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI, CC BY 4.0)
- [CLINC150](https://github.com/clinc/oos-eval) (CC BY 3.0)
- [SNLI](https://nlp.stanford.edu/projects/snli/) (Stanford, CC BY-SA 4.0)
- [SST-5](https://nlp.stanford.edu/sentiment/) (Stanford; no licence stated)
- [Civil Comments](https://huggingface.co/datasets/google/civil_comments) (CC0 1.0)
- [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) (NVIDIA, CC BY 4.0)
- [Chatbot Arena human preference 55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) (LMArena; Apache-2.0 per its card; model outputs under their providers' terms)
- [TabFact](https://github.com/wenhuchen/Table-Fact-Checking) (MIT)
- [deepset prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) (Apache-2.0)
- [Wikispeedia](https://snap.stanford.edu/data/wikispeedia.html) (West and Leskovec, WWW 2012; SNAP)
- Wikipedia text (CC BY-SA 4.0)

Synthetic decision families were written by GPT-5.6 Luna and reviewed by GPT-5.6 Terra through OpenAI's API. JevBench
is third-party (MIT) and not included.

"Jev", "System One" and TypeSafe are names of TypeSafe AI's products, used here only to describe interface
compatibility.
