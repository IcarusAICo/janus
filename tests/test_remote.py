"""Exercise the real client; replace only the HTTP transport."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import traceback
from urllib.error import HTTPError, URLError

import pytest

from janus.data import write_jsonl
from janus.evaluation import compare_runs
from janus.schema import Request


def request(group="observation-1"):
    return Request.from_dict({"state": "The parcel is red and delivery was good.", "group_id": group,
        "questions": {
            "banking77:color": {"type": "choice", "instructions": "What color?",
                "criteria": {"a": "Red", "b": "Blue"}, "target": [1, 0]},
            "sst5:rating": {"type": "score", "instructions": "Rate delivery.",
                "criteria": ["Bad", "Good", "Great"], "target": [.1, .8, .1]},
            "snli:truth": {"type": "noul", "instructions": "Is it red?", "target": [0, 1]}}})


def response():
    return {"model": "jev-1.13.0", "answers": {
        "banking77:color": {"type": "choice", "probabilities": {"b": .2, "a": .8}, "choice": "a", "confidence": .8},
        "sst5:rating": {"type": "score", "probabilities": {"0": .1, "1": .8, "2": .1}, "score": 1., "confidence": .8},
        "snli:truth": {"type": "noul", "noul": .75}},
        "usage": {"input_tokens": 123, "output_tokens": 17}}


class HTTPResponse(io.BytesIO):
    status = 200


def install_transport(monkeypatch, payload=None, error=None):
    from janus import remote
    calls = []

    def transport(outgoing, timeout):
        calls.append(outgoing)
        if error:
            raise error
        body = response() if payload is None else payload
        return HTTPResponse(body if isinstance(body, bytes) else json.dumps(body).encode())

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-secret-never-print")
    monkeypatch.setattr(remote, "_http_open", transport)
    return calls


def test_client_payload_excludes_targets_groups_and_uses_fixed_endpoint(monkeypatch):
    from janus.remote import RemoteClient
    calls = install_transport(monkeypatch)
    result = RemoteClient().predict(request())
    outgoing = calls[0]
    payload = json.loads(outgoing.data)
    assert outgoing.full_url == "https://api.typesafe.ai/v1/systemone"
    assert outgoing.method == "POST"
    assert outgoing.get_header("Authorization") == "Bearer test-secret-never-print"
    assert set(payload) == {"model", "state", "questions"}
    assert all(set(q) == {"type", "instructions", "criteria"} for q in payload["questions"].values())
    assert "observation-1" not in outgoing.data.decode()
    assert result.probabilities == ((.8, .2), (.1, .8, .1), (.25, .75))
    assert result.usage == {"input_tokens": 123, "output_tokens": 17}
    assert result.requested_model == result.returned_model == "jev-1.13.0"


@pytest.mark.parametrize("change", [
    lambda p: p["answers"].pop("snli:truth"),
    lambda p: p["answers"].update(extra={"type": "noul", "noul": .5}),
    lambda p: p["answers"]["banking77:color"].update(type="score"),
    lambda p: p["answers"]["banking77:color"]["probabilities"].pop("b"),
    lambda p: p["answers"]["banking77:color"]["probabilities"].update(c=0),
    lambda p: p["answers"]["banking77:color"]["probabilities"].update(a=.1),
    lambda p: p["answers"]["banking77:color"]["probabilities"].update(a=float("nan")),
    lambda p: p["answers"]["banking77:color"]["probabilities"].update(a=-.1),
    lambda p: p["answers"]["banking77:color"].pop("probabilities"),
    lambda p: p["answers"]["snli:truth"].update(noul=1.1),
    lambda p: p["answers"]["snli:truth"].update(noul=True),
    lambda p: p["usage"].update(input_tokens=-1),
    lambda p: p.pop("model"),
])
def test_client_rejects_incomplete_or_invalid_probability_data(monkeypatch, change):
    from janus.remote import RemoteClient, RemoteError
    payload = response()
    change(payload)
    calls = install_transport(monkeypatch, payload)
    with pytest.raises(RemoteError):
        RemoteClient().predict(request())
    assert len(calls) == 1


def test_rounding_is_renormalized_and_recorded(monkeypatch):
    from janus.remote import RemoteClient
    payload = response()
    payload["answers"]["sst5:rating"]["probabilities"] = {"0": .33, "1": .33, "2": .33}
    install_transport(monkeypatch, payload)
    result = RemoteClient().predict(request())
    assert result.probabilities[1] == pytest.approx([1 / 3] * 3)
    assert result.normalization[1] == {"original_sum": .99, "renormalized": True}
    assert result.normalization[0]["renormalized"] is False


def test_credentials_prefer_environment_and_never_execute_env_file(monkeypatch, tmp_path):
    from janus.remote import load_api_key, RemoteError
    env_file = tmp_path / "env.sh"
    env_file.write_text("export TYPESAFE_API_KEY='file-secret' # comment\n")
    monkeypatch.setenv("TYPESAFE_API_KEY", "environment-secret")
    assert load_api_key(env_file) == "environment-secret"
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert load_api_key(env_file) == "file-secret"
    marker = tmp_path / "executed"
    env_file.write_text(f"export TYPESAFE_API_KEY=$(touch {marker})\n")
    with pytest.raises(RemoteError) as caught:
        load_api_key(env_file)
    assert "touch" not in str(caught.value)
    assert not marker.exists()
    env_file.write_text("export TYPESAFE_API_KEY='unterminated-secret\n")
    with pytest.raises(RemoteError) as caught:
        load_api_key(env_file)
    assert "unterminated-secret" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("status", [401, 403, 400, 302])
def test_nonretryable_http_errors_are_sanitized(monkeypatch, status):
    from janus.remote import RemoteClient, RemoteError
    secret = "test-secret-never-print"
    error = HTTPError("https://api.typesafe.ai/v1/systemone", status, secret,
                      {"Authorization": secret}, io.BytesIO(secret.encode()))
    calls = install_transport(monkeypatch, error=error)
    with pytest.raises(RemoteError) as caught:
        RemoteClient().predict(request())
    assert caught.value.status == status
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert len(calls) == 1


def test_network_error_does_not_expose_credentials_or_retry_ambiguous_call(monkeypatch):
    from janus.remote import RemoteClient, RemoteError
    calls = install_transport(monkeypatch, error=URLError("test-secret-never-print"))
    with pytest.raises(RemoteError) as caught:
        RemoteClient().predict(request())
    assert "test-secret-never-print" not in "".join(traceback.format_exception(caught.value))
    assert len(calls) == 1


def test_retries_are_bounded_for_rate_limits(monkeypatch):
    from janus.remote import RemoteClient, RemoteError
    calls = install_transport(monkeypatch, error=HTTPError("https://api.typesafe.ai/v1/systemone", 429, "slow", {}, None))
    with pytest.raises(RemoteError) as caught:
        RemoteClient().predict(request())
    assert caught.value.status == 429
    assert len(calls) == 3


def test_cache_reuses_exact_payload_and_rejects_corruption(monkeypatch, tmp_path):
    from janus.remote import RemoteClient, RemoteError
    calls = install_transport(monkeypatch)
    client = RemoteClient()
    initial = client.predict(request(), cache_dir=tmp_path)
    repeat = client.predict(replace(request("other-group"), questions=tuple(
        replace(q, target=tuple(reversed(q.target))) for q in request().questions)), cache_dir=tmp_path)
    assert len(calls) == 1
    assert initial.cache_hit is False and repeat.cache_hit is True
    assert initial.request_sha256 == repeat.request_sha256
    client.predict(replace(request(), state="Different state"), cache_dir=tmp_path)
    assert len(calls) == 2
    cache_path = tmp_path / f"{initial.request_sha256}.json"
    cache_path.write_text("{")
    with pytest.raises(RemoteError, match="cache"):
        client.predict(request(), cache_dir=tmp_path)
    assert len(calls) == 2


def test_invalid_paid_response_is_cached_without_rebilling(monkeypatch, tmp_path):
    from janus.remote import RemoteClient, RemoteError
    calls = install_transport(monkeypatch, payload=b'{"model": "jev-1.13.0", "answers": {}}')
    client = RemoteClient()
    for _ in range(2):
        with pytest.raises(RemoteError):
            client.predict(request(), cache_dir=tmp_path)
    assert len(calls) == 1


def test_evaluation_resume_and_paired_signatures(monkeypatch, tmp_path):
    import torch
    from janus.remote import evaluate_remote
    calls = install_transport(monkeypatch)
    data = tmp_path / "test.jsonl"
    records = [request(), replace(request("observation-2"), state="Another parcel")]
    write_jsonl(data, [r.to_dict() for r in records])
    output = tmp_path / "remote"
    report = evaluate_remote(data, output, workers=2, price_per_million=.042)
    assert report["successful_requests"] == 2 and report["failed_requests"] == 0
    assert report["raw"]["count"] == 6
    assert report["usage"] == {"input_tokens": 246, "output_tokens": 34}
    assert report["price_per_million_input_tokens_usd"] == .042
    assert report["estimated_cost_usd"] == pytest.approx(246 * .042 / 1_000_000)
    assert report["estimated_new_cost_usd"] == pytest.approx(246 * .042 / 1_000_000)
    assert set(report["by_domain"]) == {"banking77", "sst5", "snli"}
    assert report["latency"]["scope"] == "remote_request_end_to_end_including_retries"
    assert report["latency"]["p95_seconds"] >= report["latency"]["p50_seconds"] > 0
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    local = []
    for i, req in enumerate(records):
        for j, q in enumerate(req.questions):
            row = predictions[i * 3 + j]
            signature = {"state": req.state, "kind": q.kind, "instructions": q.instructions,
                         "options": [(o.key, o.description) for o in q.options]}
            assert row["input_sha256"] == hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            assert row["target"] == torch.tensor(q.target, dtype=torch.float32).tolist()
            assert row["logits"] == pytest.approx([math.log(max(p, 1e-12)) for p in row["probabilities"]])
            local.append({**row, "nll": row["nll"] + 1})
    local_path = tmp_path / "local.jsonl"
    write_jsonl(local_path, local)
    comparison = compare_runs(output / "predictions.jsonl", local_path)
    assert comparison["mean_improvement"] == pytest.approx(1)
    assert comparison["lower"] > 0
    resumed = evaluate_remote(data, output, workers=2, resume=True, price_per_million=.042)
    assert len(calls) == 2
    assert resumed["cache_hits"] == 2
    assert resumed["http_requests_this_run"] == 0
    assert resumed["new_usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert resumed["estimated_cost_usd"] == pytest.approx(246 * .042 / 1_000_000)
    assert resumed["estimated_new_cost_usd"] == 0
    assert (output / "labels.jsonl").exists()
    assert resumed["data_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        evaluate_remote(data, output)
    changed = tmp_path / "changed.jsonl"
    write_jsonl(changed, [replace(request(), state="changed").to_dict()])
    with pytest.raises(ValueError, match="resume"):
        evaluate_remote(changed, output, resume=True)


def test_evaluation_fails_auth_before_scheduling_a_benchmark(monkeypatch, tmp_path):
    from janus.remote import evaluate_remote, RemoteError
    calls = install_transport(monkeypatch, error=HTTPError("https://api.typesafe.ai/v1/systemone", 401, "secret", {}, None))
    data = tmp_path / "test.jsonl"
    write_jsonl(data, [replace(request(str(i)), state=f"State {i}").to_dict() for i in range(12)])
    with pytest.raises(RemoteError) as caught:
        evaluate_remote(data, tmp_path / "remote", workers=4)
    assert caught.value.status == 401
    assert len(calls) == 1
    report = json.loads((tmp_path / "remote" / "metrics.json").read_text())
    assert report["failed_requests"] == 1
    assert report["not_attempted_requests"] == 11


def test_payloads_preserve_each_local_question_and_option_order(monkeypatch, tmp_path):
    from janus.remote import evaluate_remote
    calls = install_transport(monkeypatch)
    original = request()
    reordered = replace(original, group_id="another-group", questions=tuple(reversed([
        replace(q, options=tuple(reversed(q.options)), target=tuple(reversed(q.target)))
        if q.kind == "choice" else q for q in original.questions])))
    data = tmp_path / "data.jsonl"
    write_jsonl(data, [original.to_dict(), reordered.to_dict()])
    report = evaluate_remote(data, tmp_path / "out")
    assert report["estimated_cost_usd"] is None
    assert report["estimated_new_cost_usd"] is None
    rows = [json.loads(line) for line in (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()]
    assert len(calls) == 2 and report["unique_requests"] == 2
    assert list(json.loads(calls[0].data)["questions"]["banking77:color"]["criteria"]) == ["a", "b"]
    assert list(json.loads(calls[1].data)["questions"]["banking77:color"]["criteria"]) == ["b", "a"]
    assert report["successful_requests"] == 2
    assert rows[0]["probabilities"] == [.8, .2]
    assert rows[-1]["keys"] == ["b", "a"]
    assert rows[-1]["probabilities"] == [.2, .8]
    assert rows[-1]["nll"] == pytest.approx(rows[0]["nll"])


def test_preflight_checks_all_question_types(monkeypatch):
    from janus.remote import preflight
    payload = response()
    payload["answers"] = {"choice": payload["answers"]["banking77:color"],
                          "score": payload["answers"]["sst5:rating"],
                          "noul": payload["answers"]["snli:truth"]}
    calls = install_transport(monkeypatch, payload=payload)
    report = preflight()
    assert report["http_status"] == 200
    assert report["model"] == "jev-1.13.0"
    assert report["usage"]["input_tokens"] == 123
    assert set(json.loads(calls[0].data)["questions"]) == {"choice", "score", "noul"}
