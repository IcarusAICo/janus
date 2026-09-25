"""Write a Hugging Face model repo layout for one checkpoint: model.pt, calibration.json, janus_config.json and a
model card (README.md) whose results are TODO placeholders to fill in by hand. Uploads nothing.

    python scripts/export_hf.py CHECKPOINT CALIBRATION OUT_DIR [--model-id NAME] [--repo-id ORG/NAME]

`janus.load(OUT_DIR)` (or the repo id once uploaded) then serves it.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch

from janus.data import file_hash
from janus.evaluation import read_calibration

CARD = """---
license: apache-2.0
base_model: {base_model}
library_name: janus
pipeline_tag: text-classification
language:
- en
tags:
- janus
- typed-decisions
- calibrated-probabilities
- system-one
- lora
---

# {model_id}

A typed-decision model: give it a state (text, JSON, or images) and any number of `choice`, `score` and `noul`
(yes/no) questions, and it returns a calibrated probability for every option in **one forward pass**, with no text
generation. It is a LoRA adapter plus a small pointer head on top of [{base_model}](https://huggingface.co/{base_model})
(revision `{base_revision}`), trained with the open [janus](https://github.com/TODO/janus) code.

Janus is an independent reconstruction of a "System 1" decision model. It is not affiliated with, endorsed by, or
derived from the weights of TypeSafe or its Jev models.

## Use

```bash
pip install "janus @ git+https://github.com/TODO/janus"
```

```python
import janus

m = janus.load("{repo_id}", device="cuda")   # downloads this repo and the {base_model} backbone
m.predict(
    "Refunds need a receipt and a purchase within 30 days. The customer bought 12 days ago and has no receipt.",
    {{"refund": {{"type": "noul", "instructions": "Is a refund permitted under the policy?"}},
      "route": {{"type": "choice", "instructions": "Which team handles this?",
                "criteria": {{"billing": "Payments and refunds", "support": "Everything else"}}}}}},
)
```

The response is the `/v1/systemone` shape: `{{"model", "answers": {{id: {{"type", ...}}}}, "usage"}}`.
`m.predict_batch([request, ...])` scores many requests with shared forward passes. To serve it over HTTP:
`python -m janus serve --checkpoint model.pt --calibration calibration.json --model-id {model_id}` from a download of
this repo.

## Files

| file | contents |
| --- | --- |
| `model.pt` | trainable tensors only (LoRA + decision head, sha256 `{checkpoint_sha256}`); the backbone is fetched from `{base_model}` at `{base_revision}` |
| `calibration.json` | global, per-family and per-cardinality temperatures, bound to `model.pt` by its sha256 |
| `janus_config.json` | model id and base model reference |

Per-family temperatures apply when a request's `group_id` starts with `family:`. Families calibrated in this
checkpoint: {families}. Every other request uses the global temperature ({temperature:.4f}).

## Training

TODO: data mix, distillation teacher and blend, steps, hardware, wall-clock time.

## Results

TODO: fill in from the evaluation runs. Do not publish numbers that were not measured on this exact checkpoint.

| benchmark | metric | this model |
| --- | --- | --- |
| TODO | TODO | TODO |

## Limitations

TODO

## License and credits

Apache-2.0. The base model {base_model} is by the Qwen team (Apache-2.0).
"""

p = argparse.ArgumentParser()
p.add_argument("checkpoint"); p.add_argument("calibration"); p.add_argument("output")
p.add_argument("--model-id", help="Model id in responses and the card title (default: the output directory's name)")
p.add_argument("--repo-id", help="Hugging Face repo id shown in the card (default: TODO/<model id>)")
a = p.parse_args()

calibration = read_calibration(a.calibration, a.checkpoint)  # raises unless it was fitted for this checkpoint
model = torch.load(a.checkpoint, map_location="cpu", weights_only=True)["metadata"]["model"]
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
shutil.copyfile(a.checkpoint, out / "model.pt")
shutil.copyfile(a.calibration, out / "calibration.json")
model_id = a.model_id or out.name
config = {"model_id": model_id, "base_model": model["backbone"], "base_revision": model["revision"], "format": 1}
(out / "janus_config.json").write_text(json.dumps(config, indent=2) + "\n")
(out / "README.md").write_text(CARD.format(
    model_id=model_id, repo_id=a.repo_id or f"TODO/{model_id}", base_model=model["backbone"],
    base_revision=model["revision"], checkpoint_sha256=file_hash(out / "model.pt"),
    families=", ".join(f"`{f}`" for f in sorted(calibration.get("by_family", {}))) or "none",
    temperature=calibration["temperature"]))
print(out)
