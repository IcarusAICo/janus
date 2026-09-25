"""scripts/export_hf.py -> janus.load(local dir) answers exactly as the HTTP server does on the same checkpoint."""

import json
import subprocess
import sys
import threading
from urllib.request import Request as HTTPRequest, urlopen

import pytest
import torch

import janus
from janus.data import file_hash
from janus.model import DecisionModel, ModelConfig
from janus.server import make_server
from janus.training import checkpoint

EXAMPLE = json.load(open("examples/request.json"))


def test_export_load_predict_matches_server(tmp_path, monkeypatch):
    monkeypatch.delenv("JANUS_SERVER_TOKEN", raising=False)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    ckpt = tmp_path / "best.pt"
    checkpoint(DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, layers=1, head_rank=8, max_tokens=1024)), ckpt, {"step": 0})
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"temperature": 1.3, "checkpoint_sha256": file_hash(ckpt),
                               "by_family": {"fam": {"temperature": .5, "by_cardinality": {"4": 2.}}}}))
    out = tmp_path / "janus-tiny"
    subprocess.run([sys.executable, "-m", "scripts.export_hf", str(ckpt), str(cal), str(out)], check=True)
    assert {p.name for p in out.iterdir()} == {"model.pt", "calibration.json", "janus_config.json", "README.md"}
    assert (out / "README.md").read_text().startswith("---\nlicense: apache-2.0\nbase_model: tiny\n")

    m = janus.load(str(out), device="cpu")
    server = make_server(out / "model.pt", port=0, calibration=out / "calibration.json", model_id="janus-tiny")
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def post(body):
        req = HTTPRequest(f"http://127.0.0.1:{server.server_address[1]}/v1/systemone", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=60) as r:
            return json.load(r)

    four = {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": None, "c": "C", "d": "D"}}
    requests = [EXAMPLE, {**EXAMPLE, "group_id": "fam:1"}, {"state": "Short.", "questions": {"q": four}, "group_id": "fam:2"}]
    try:
        served = [post(r) for r in requests]
    finally:
        server.shutdown()
    assert m.predict(EXAMPLE["state"], EXAMPLE["questions"]) == served[0]
    batch = m.predict_batch(requests)
    assert [b["usage"] for b in batch] == [s["usage"] for s in served]
    for b, s in zip(batch, served):
        for qid, answer in s["answers"].items():
            got = b["answers"][qid]
            assert got.get("choice") == answer.get("choice") and got["type"] == answer["type"]
            for key in ("probabilities", "noul"):
                if key in answer:
                    assert got[key] == pytest.approx(answer[key], abs=1e-6)
    # The family temperature was applied: same state and questions, different group_id, different probabilities.
    assert served[0]["answers"]["route"]["probabilities"] != served[1]["answers"]["route"]["probabilities"]
    assert requests[2]["questions"]["q"]["criteria"]["b"] is None  # the caller's dict is not rewritten
