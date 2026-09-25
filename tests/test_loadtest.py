"""janus.loadtest against the tiny CPU server: four clients for a couple of seconds, plus the pure helpers."""

import json
import threading

import pytest
import torch

from janus.loadtest import load_bodies, main, percentile, run
from janus.model import DecisionModel, ModelConfig
from janus.server import make_server
from janus.training import checkpoint

TOKEN = "test-token-never-print"
EXAMPLE = json.load(open("examples/request.json"))


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    torch.set_num_threads(2)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, layers=1, head_rank=8, max_tokens=1024))
    checkpoint(model, path, {"step": 0})
    server = make_server(path, port=0, model_id="jev-local-test", aliases=["jev-latest"])
    server.token = TOKEN
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{server.server_address[1]}", "server": server}
    server.shutdown()


def test_percentile_and_bodies(tmp_path):
    assert percentile([], .5) is None and percentile([3.], .99) == 3.
    assert percentile([1., 2., 3., 4.], .5) == 3. and percentile([1., 2., 3., 4.], .99) == 4.
    (tmp_path / "one.json").write_text(json.dumps(EXAMPLE))
    (tmp_path / "many.jsonl").write_text("\n".join([json.dumps(EXAMPLE)] * 3) + "\n")
    assert load_bodies(tmp_path / "one.json") == [EXAMPLE] and load_bodies(tmp_path / "many.jsonl") == [EXAMPLE] * 3


def test_four_clients_for_two_seconds(served):
    report = run(served["url"], clients=4, seconds=2, bodies=[EXAMPLE, {**EXAMPLE, "state": "Another state."}], token=TOKEN)
    assert report["model"] == "jev-local-test" and report["errors"] == {} and report["requests"] > 4
    assert report["qps"] > 0 and report["input_tokens_per_s"] > 0 and 1.9 < report["seconds"] < 10
    assert 0 < report["p50_ms"] <= report["p95_ms"] <= report["p99_ms"] and report["mean_ms"] > 0
    assert report["server_stats"]["requests"] >= report["requests"] and report["server_stats"]["batches"] >= 1
    assert report["server_stats"]["mean_batch_size"] >= 1.


def test_errors_are_counted_by_status_and_main_prints_json(served, capsys):
    report = run(served["url"], clients=2, seconds=0.5, bodies=[EXAMPLE], token="wrong", model="jev-local-test")
    assert report["requests"] == 0 and set(report["errors"]) == {"401"} and report["p50_ms"] is None
    report = main(["--url", served["url"], "--clients", "1", "--seconds", "0.5", "--token", TOKEN, "--model", "jev-latest"])
    assert report["requests"] > 0 and report["errors"] == {}
    assert json.loads(capsys.readouterr().out)["model"] == "jev-latest"
