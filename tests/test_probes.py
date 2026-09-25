"""Real probe variants, caches and local inference; mock only remote HTTP."""

from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
import math
from urllib.error import HTTPError

import pytest
import torch

from janus.data import file_hash, write_json, write_jsonl
from janus.model import DecisionModel, ModelConfig
from janus.remote import _payload_bytes
from janus.schema import Request


def sample_request(domain="banking77", index=0):
    score = domain == "sst5"
    return Request.from_dict({"state": f"{domain} observation {index}", "group_id": f"{domain}:{index}",
        "questions": {
            f"{domain}:target": {"type": "score" if score else "choice", "instructions": "Select the correct option.",
                "criteria": ["Low", "Medium", "High"] if score else {"a": "First", "b": "Second"},
                "target": [1., 0., 0.] if score else [1., 0.]},
            f"{domain}:other": {"type": "noul", "instructions": "Is the observation positive?", "target": [0., 1.]}}})


def response_for(payload):
    # Known variation: alone shifts 0.1 mass; added siblings shift 0.2.
    count = len(payload["questions"])
    first = .6 if count == 1 else (.5 if count == 5 else .7)
    answers = {}
    for qid, q in payload["questions"].items():
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": .6}
        else:
            p = {"0": first, "1": .9-first, "2": .1} if q["type"] == "score" else {"a": first, "b": 1-first}
            answer = {"type": q["type"], "probabilities": p, "confidence": first}
            answer["score" if q["type"] == "score" else "choice"] = 0 if q["type"] == "score" else "a"
            answers[qid] = answer
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 20}}


class HTTPResponse(io.BytesIO):
    status = 200


def setup_remote_fixture(tmp_path, monkeypatch, *, status=None):
    from janus import remote
    requests = [sample_request(domain, i) for domain in ("banking77", "clinc150", "sst5", "snli") for i in range(4)]
    data = tmp_path / "panel.jsonl"
    baseline = tmp_path / "baseline"
    write_jsonl(data, [r.to_dict() for r in requests])
    for request in requests:
        body = _payload_bytes(request, "jev-1.13.0")
        digest = hashlib.sha256(body).hexdigest()
        write_json(baseline / "cache" / f"{digest}.json", {"version": "1", "request_sha256": digest,
            "requested_model": "jev-1.13.0", "status": 200, "response_body": json.dumps(response_for(json.loads(body))),
            "attempts": 1, "latency_seconds": .2})
    calls = []

    def transport(outgoing, timeout):
        calls.append(outgoing)
        if status is not None:
            raise HTTPError(outgoing.full_url, status, "fake-secret", {}, io.BytesIO(b"fake-secret"))
        return HTTPResponse(json.dumps(response_for(json.loads(outgoing.data))).encode())

    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-secret")
    monkeypatch.setattr(remote, "_http_open", transport)
    return data, baseline, calls


def test_variants_preserve_isolation_inputs_and_align_choice_permutation():
    from janus.probes import build_probe_variants
    original = sample_request()
    variants = build_probe_variants(original)
    assert set(variants) == {"alone", "siblings", "option_permutation"}
    assert variants["alone"].questions == (original.questions[0],)
    assert variants["siblings"].questions[:2] == original.questions
    assert len(variants["siblings"].questions) == 5
    assert len({q.instructions for q in variants["siblings"].questions[2:]}) == 3
    assert all(q.target is None for q in variants["siblings"].questions[2:])
    assert all(r.state == original.state and r.group_id == original.group_id for r in variants.values())
    changed = variants["option_permutation"].questions[0]
    assert changed.id == original.questions[0].id and changed.instructions == original.questions[0].instructions
    assert [(o.key, o.description) for o in changed.options] == [("b", "Second"), ("a", "First")]
    assert changed.target == (0., 1.)
    assert variants["option_permutation"].questions[1:] == original.questions[1:]
    ordinal = sample_request("sst5")
    assert "option_permutation" not in build_probe_variants(ordinal)


def test_selection_is_deterministic_and_four_states_per_domain():
    from janus.probes import select_probe_requests
    requests = [sample_request(domain, i) for domain in ("banking77", "clinc150", "sst5", "snli") for i in range(10)]
    selected = select_probe_requests(requests)
    assert selected == select_probe_requests(requests)
    assert len(selected) == 16
    assert Counter(r.questions[0].id.split(':')[0] for r in selected) == {
        "banking77": 4, "clinc150": 4, "sst5": 4, "snli": 4}
    with pytest.raises(ValueError):
        select_probe_requests(requests[:3])


def test_remote_probes_use_exactly_48_calls_and_full_case_artifacts(tmp_path, monkeypatch):
    from janus.probes import run_remote_probes
    data, baseline, calls = setup_remote_fixture(tmp_path, monkeypatch)
    original_hashes = {path: file_hash(path) for path in (baseline / "cache").glob('*.json')}
    output = tmp_path / "probes"
    report = run_remote_probes(data, baseline, output, workers=4, price_per_million=.042)
    assert len(calls) == report["new_http_attempts"] == report["successful_probe_calls"] == 48
    assert report["failed_probe_calls"] == 0 and report["cached_baselines"] == 16
    assert report["by_probe"]["alone"]["count"] == 16
    assert report["by_probe"]["siblings"]["count"] == 16
    assert report["by_probe"]["option_permutation"]["count"] == 12
    assert report["by_probe"]["repeat"]["count"] == 4
    assert report["by_probe"]["alone"]["mean_total_variation"] == pytest.approx(.1)
    assert report["by_probe"]["siblings"]["mean_total_variation"] == pytest.approx(.2)
    assert report["by_probe"]["option_permutation"]["max_total_variation"] == 0
    assert report["by_probe"]["repeat"]["max_total_variation"] == 0
    assert report["usage"] == {"input_tokens": 4800, "output_tokens": 960}
    assert report["estimated_cost_usd"] == pytest.approx(4800*.042/1e6)
    assert all(file_hash(path) == digest for path,digest in original_hashes.items())
    cases = list((output / "cases").glob('*.json'))
    assert len(cases) == 64
    repeated = 0
    for path in cases:
        case = json.loads(path.read_text())
        assert json.loads(case["payload_utf8"]) == case["payload"]
        assert hashlib.sha256(case["payload_utf8"].encode()).hexdigest() == case["request_sha256"]
        assert case["response"]["model"] == "jev-1.13.0"
        assert set(case["payload"]) == {"model", "state", "questions"}
        assert all("target" not in q for q in case["payload"]["questions"].values())
        assert case["returned_model"] == "jev-1.13.0"
        if case["probe"] == "repeat":
            original = json.loads((output / "cases" / f'{case["case_id"]}--baseline.json').read_text())
            assert case["payload_utf8"] == original["payload_utf8"]
            assert case["cache_hit"] is False
            repeated += 1
    assert repeated == 4
    assert len(list((output / "cache").rglob('*.json'))) == 64
    with pytest.raises(FileExistsError):
        run_remote_probes(data, baseline, output)
    assert len(calls) == 48


def test_probe_http_failures_do_not_retry_or_leak_credentials(tmp_path, monkeypatch):
    from janus.probes import run_remote_probes
    data, baseline, calls = setup_remote_fixture(tmp_path, monkeypatch, status=503)
    output = tmp_path / "probes"
    report = run_remote_probes(data, baseline, output)
    assert len(calls) == report["new_http_attempts"] == 48
    assert report["successful_probe_calls"] == 0 and report["failed_probe_calls"] == 48
    assert all("fake-secret" not in path.read_text() for path in output.rglob('*.json'))


def test_missing_original_cache_cannot_trigger_unplanned_baseline_call(tmp_path, monkeypatch):
    from janus.probes import run_remote_probes
    data, baseline, calls = setup_remote_fixture(tmp_path, monkeypatch)
    next((baseline / "cache").glob('*.json')).unlink()
    with pytest.raises(ValueError, match="baseline cache"):
        run_remote_probes(data, baseline, tmp_path / "probes")
    assert calls == []


def test_local_probe_uses_shared_variants_and_restores_training_state():
    from janus.probes import local_model_probe
    torch.set_num_threads(2)
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation="full", mode="listwise", hidden_size=16,
                                      head_rank=8, layers=1, max_tokens=2048))
    model.train()
    report = local_model_probe(model, [sample_request(), sample_request("sst5")])
    assert model.training is True
    assert report["by_probe"]["alone"]["max_total_variation"] < 1e-6
    assert report["by_probe"]["siblings"]["max_total_variation"] < 1e-6
    assert report["by_probe"]["repeat"]["max_total_variation"] < 1e-6
    assert report["by_probe"]["option_permutation"]["count"] == 1
    assert report["latency_scope"] == "local_forward_only"
