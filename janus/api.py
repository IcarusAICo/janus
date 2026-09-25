"""Python API: `janus.load(repo_or_dir)` returns a model whose answers are exactly those of POST /v1/systemone.

A model directory (or Hugging Face repo, see scripts/export_hf.py) holds `model.pt` (trainable tensors; the base
backbone is referenced by HF id and revision inside it), `calibration.json` (temperatures bound to model.pt's sha256)
and `janus_config.json` (the model id reported in responses).
"""

import copy
import json
from pathlib import Path

import torch

from .schema import Request
from .server import Worker, envelope, input_tokens, null_criteria


class Janus:
    """One loaded checkpoint. `predict` answers one request; `predict_batch` shares forward passes across requests
    (janus.model.forward_many, or the prefix cache on CUDA hybrid backbones). Not thread-safe: use one instance per
    thread, or `python -m janus serve` for concurrent callers. The server's optional option-order re-read is not
    applied here."""

    def __init__(self, directory, device=None, **options):
        directory = Path(directory)
        self.config = json.loads((directory / "janus_config.json").read_text())
        self.model_id = self.config["model_id"]
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # options: Worker's fast, precapture, prefix_cache_tokens, max_tokens, fp8, fp4, batch_max (janus.server.Worker).
        # precapture defaults off here (it adds ~20-50 s at load); `janus serve` turns it on for steady tail latency.
        options.setdefault("precapture", False)
        self.worker = Worker(directory / "model.pt", device, directory / "calibration.json", **options)

    def predict(self, state, questions, group_id=""):
        """`state` and `questions` as in a /v1/systemone body. A `group_id` of the form "family:..." selects that
        family's calibrated temperature when calibration.json has one (listed in the model card); otherwise the
        global temperature applies, as on the server."""
        return self.predict_batch([{"state": state, "questions": questions, "group_id": group_id}])[0]

    def predict_batch(self, requests):
        """A list of /v1/systemone bodies -> a list of responses, in order, `worker.batch_max` requests per pass."""
        worker = self.worker
        parsed = [Request.from_dict(null_criteria(copy.deepcopy(r))) for r in requests]  # copy: the caller's dicts stay as given
        packs = [worker.pack(r) for r in parsed]  # every request is validated before any compute
        distributions = []
        for i in range(0, len(parsed), worker.batch_max):
            chunk = zip(parsed[i:i + worker.batch_max], packs[i:i + worker.batch_max])
            distributions += worker.run_jobs([(p, worker.temperatures(r), worker.key(r), None) for r, p in chunk])
        return [envelope(r, d, self.model_id, input_tokens(p)) for r, d, p in zip(parsed, distributions, packs)]


def load(name, device=None, revision=None, **options):
    """A local model directory, or a Hugging Face repo id (downloaded once into the HF cache)."""
    path = Path(name)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(str(name), revision=revision))
    return Janus(path, device, **options)
