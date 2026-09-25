"""The local /v1/systemone server on a tiny CPU checkpoint, over real HTTP on a free port."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request as HTTPRequest, urlopen

import pytest
import torch

from janus.evaluation import predict
from janus.model import DecisionModel, ModelConfig
from janus.schema import Request
from janus.server import confidence, envelope, make_server
from janus.training import checkpoint

TOKEN = "test-token-never-print"
EXAMPLE = json.load(open("examples/request.json"))


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    torch.set_num_threads(2)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, layers=1, head_rank=8, max_tokens=1024))
    checkpoint(model, path, {"step": 0})
    server = make_server(path, port=0, model_id="janus-local-test", queue_size=32, aliases=["jev-latest"])
    server.token = TOKEN
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{server.server_address[1]}", "checkpoint": path, "server": server}
    server.shutdown()


def call(served, path="/v1/systemone", body=None, method=None, token=TOKEN, raw=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    outgoing = HTTPRequest(served["url"] + path, data=data, headers=headers, method=method or ("POST" if data else "GET"))
    try:
        with urlopen(outgoing, timeout=60) as incoming:
            return incoming.status, json.load(incoming)
    except HTTPError as error:
        return error.code, json.loads(error.read())


def test_envelope_shapes_per_primitive_and_usage(served):
    status, body = call(served, body={"model": "jev-latest", **EXAMPLE})  # an alias; the response names the canonical id
    assert status == 200
    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "janus-local-test"
    assert set(body["answers"]) == set(EXAMPLE["questions"])
    assert body["usage"]["output_tokens"] == 0 and body["usage"]["input_tokens"] > 0
    route = body["answers"]["route"]
    assert set(route) == {"type", "choice", "probabilities", "confidence"} and route["type"] == "choice"
    assert set(route["probabilities"]) == {"payments", "access", "other"}
    assert route["choice"] == max(route["probabilities"], key=route["probabilities"].get)
    assert route["confidence"] == max(route["probabilities"].values())
    score = body["answers"]["frustration"]
    assert set(score) == {"type", "score", "legend", "probabilities", "confidence"} and score["type"] == "score"
    assert list(score["probabilities"]) == ["0", "1", "2"] and score["legend"]["2"] == "very frustrated"
    assert abs(score["score"] - sum(int(k) * v for k, v in score["probabilities"].items())) < 1e-9
    assert 0 <= score["confidence"] <= 1 and score["confidence"] == max(score["probabilities"].values())
    noul = body["answers"]["transfer_problem"]
    assert set(noul) == {"type", "noul"} and noul["type"] == "noul" and 0 <= noul["noul"] <= 1


def test_confidence_rule_on_hand_computed_distribution():
    assert confidence([.1, .8, .1]) == .8
    request = Request.from_dict({"state": "s", "questions": {
        "c": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}},
        "s": {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]},
        "n": {"type": "noul", "instructions": "i"}}})
    out = envelope(request, [[.3, .7], [.1, .8, .1], [.25, .75]], "m", 5)
    assert out["answers"]["c"] == {"type": "choice", "probabilities": {"a": .3, "b": .7}, "choice": "b", "confidence": .7}
    assert out["answers"]["s"]["confidence"] == .8 and abs(out["answers"]["s"]["score"] - 1.) < 1e-12
    assert out["answers"]["n"] == {"type": "noul", "noul": .75}
    assert out["usage"] == {"input_tokens": 5, "output_tokens": 0}


def test_matches_evaluation_predict_on_same_checkpoint(served):
    _, body = call(served, body=EXAMPLE)
    reference = predict(served["checkpoint"], Request.from_dict(EXAMPLE))
    for name, answer in body["answers"].items():
        if answer["type"] == "noul":
            assert abs(answer["noul"] - reference[name]) < 1e-6
        else:
            for key, p in reference[name]["probabilities"].items():
                assert abs(answer["probabilities"][key] - p) < 1e-6


def test_auth_401_and_disabled_when_unset(served):
    assert call(served, body=EXAMPLE, token="wrong")[0] == 401
    status, body = call(served, path="/v1/models", token=None)
    assert status == 401 and body["error"]["type"] == "authentication_error"
    served["server"].token = None
    try:
        assert call(served, path="/v1/models", token=None)[0] == 200
    finally:
        served["server"].token = TOKEN


def test_validation_error_shape_and_statuses(served):
    status, body = call(served, raw=b"{not json")
    assert status == 400 and set(body["error"]) == {"type", "message"}
    status, body = call(served, body={"state": "x", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {}}}})
    assert status == 422 and body["error"]["type"] == "validation_error" and "1" in body["error"]["message"]
    assert call(served, body={"questions": {"q": {"type": "noul", "instructions": "i"}}})[0] == 422  # no state
    assert call(served, body={"state": "x", "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["only"]}}})[0] == 422
    assert call(served, body={"state": "x", "questions": {"q": "nope"}})[0] == 422
    assert call(served, body=[1, 2])[0] == 422
    # Documented: null choice criteria mean "no extra detail"; accepted.
    status, body = call(served, body={"state": "x", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None}}}})
    assert status == 200 and set(body["answers"]["q"]["probabilities"]) == {"a", "b"}


def test_413_on_oversize_request(served):
    status, body = call(served, body={"state": "x" * 2000, "questions": {"q": {"type": "noul", "instructions": "i"}}})
    assert status == 413 and body["error"]["type"] == "request_too_large"


def test_models_endpoint_and_routing(served):
    status, body = call(served, path="/v1/models")
    assert status == 200
    assert body["models"][0]["name"] == "janus-local-test" and set(body["models"][0]) == {"name", "description", "release_date"}
    assert body["limits"]["max_tokens"] == 1024 and body["limits"]["choice_options"] == [1, 255]
    assert body["limits"]["score_levels"] == [2, 10] and body["limits"]["scope"] == "local"
    assert call(served, path="/v1/nothing")[0] == 404
    assert call(served, path="/v1/systemone")[0] == 405
    assert call(served, path="/v1/models", body=EXAMPLE)[0] == 405
    assert call(served, path="/v1/systemone", body=EXAMPLE, method="PUT")[0] == 405


def test_529_when_queue_is_full_and_503_behind_the_flag(served):
    import queue
    worker = served["server"].worker
    original = worker.queue
    worker.queue = queue.Queue(maxsize=1)
    worker.queue.put(None)  # full; never consumed by the loop, which blocks on the original queue
    try:
        status, body = call(served, body=EXAMPLE)
        assert status == 529 and body["error"]["type"] == "overloaded"
        served["server"].overloaded_status = 503
        assert call(served, body=EXAMPLE)[0] == 503
    finally:
        worker.queue = original
        served["server"].overloaded_status = 529
    assert call(served, body=EXAMPLE)[0] == 200


def test_errors_before_body_is_read_close_the_keepalive_connection(served):
    """The SDKs reuse connections; an unread POST body must not be parsed as the next request (stdlib 501)."""
    from http.client import HTTPConnection
    conn = HTTPConnection("127.0.0.1", served["server"].server_address[1], timeout=60)
    conn.request("POST", "/v1/systemone", body=json.dumps(EXAMPLE), headers={"Authorization": "Bearer wrong", "Content-Type": "application/json"})
    response = conn.getresponse()
    assert response.status == 401 and response.getheader("Connection") == "close" and len(response.getheader("x-typesafe-request-id")) == 32
    response.read()
    conn.request("GET", "/v1/models", headers={"Authorization": f"Bearer {TOKEN}"})  # reconnects; previously 501
    response = conn.getresponse()
    assert response.status == 200 and json.load(response)["models"][0]["name"] == "janus-local-test"
    conn.close()


def raw_call(served, body, headers=()):
    """Status, parsed body and response headers over one connection (urlopen hides headers on errors)."""
    from http.client import HTTPConnection
    conn = HTTPConnection("127.0.0.1", served["server"].server_address[1], timeout=60)
    conn.request("POST", "/v1/systemone", body=json.dumps(body), headers={"Authorization": f"Bearer {TOKEN}", **dict(headers)})
    response = conn.getresponse()
    out = response.status, json.load(response), response.headers
    conn.close()
    return out


def test_model_must_be_the_id_or_a_listed_alias(served):
    status, body = call(served, body={"model": "jev-preview", **EXAMPLE})  # not listed in the fixture
    assert status == 422 and body["error"]["type"] == "validation_error"
    assert all(s in body["error"]["message"] for s in ("jev-preview", "janus-local-test", "jev-latest"))
    assert call(served, body={"model": 5, **EXAMPLE})[0] == 422
    for model in ("janus-local-test", "jev-latest"):
        status, body = call(served, body={"model": model, **EXAMPLE})
        assert status == 200 and body["model"] == "janus-local-test"
    assert call(served, body=EXAMPLE)[0] == 200  # absent: the served model
    status, body = call(served, path="/v1/models")
    assert [m["name"] for m in body["models"]] == ["janus-local-test", "jev-latest"]
    assert body["models"][1]["description"] == "Alias of janus-local-test" and set(body["models"][1]) == {"name", "description", "release_date"}
    assert set(body["stats"]) == {"requests", "batches", "batch_sizes", "reread", "mean_batch_size", "queue_depth", "prefix_cache", "memory"}
    assert body["stats"]["prefix_cache"] is None  # the tiny Qwen3 backbone has no state pass to cache


def _distribution(answer, question):
    return [answer["noul"]] if question.kind == "noul" else [answer["probabilities"][o.key] for o in question.options]


def test_per_family_calibration_per_question_and_global_only_file_unchanged(served):
    worker = served["server"].worker
    body = {**EXAMPLE, "group_id": "banking:item-7"}  # family "banking"; questions: route (3 options), transfer_problem (2), frustration (3)
    request = Request.from_dict(body)
    with torch.inference_mode():
        logits = [z.float() for z in worker.model(worker.pack(request))]
    original = worker.calibration
    global_only = {"temperature": 1.7}
    by_family = {"temperature": 1.7, "by_family": {"banking": {"temperature": 0.6, "by_cardinality": {"2": 2.5}}}}
    try:
        worker.calibration = global_only
        assert worker.temperatures(request) == [1.7, 1.7, 1.7]
        _, plain = call(served, body=body)
        worker.calibration = by_family
        assert worker.temperatures(request) == [0.6, 2.5, 0.6]  # 3 options -> bucket "4": family entry; 2 -> bucket "2"
        assert worker.temperatures(Request.from_dict(EXAMPLE)) == [1.7] * 3  # no group_id: global
        assert worker.temperatures(Request.from_dict({**EXAMPLE, "group_id": "other:1"})) == [1.7] * 3
        _, family = call(served, body=body)
    finally:
        worker.calibration = original
    for question, z, t_family in zip(request.questions, logits, [0.6, 2.5, 0.6]):
        for answer, t in ((plain, 1.7), (family, t_family)):
            expected = (z / t).softmax(-1).tolist()
            got = _distribution(answer["answers"][question.id], question)
            assert all(abs(a - b) < 1e-6 for a, b in zip(got, expected[1:] if question.kind == "noul" else expected))
    assert plain["answers"]["route"]["probabilities"] != family["answers"]["route"]["probabilities"]


def test_concurrent_requests_are_batched_and_answer_like_sequential(served):
    worker = served["server"].worker
    bodies = [{**EXAMPLE, "state": f"{EXAMPLE['state']} Attempt number {i}."} for i in range(8)]
    sequential = [call(served, body=b)[1]["answers"] for b in bodies]
    before = call(served, path="/v1/models")[1]["stats"]
    results = [None] * len(bodies)

    def post(i):
        results[i] = call(served, body=bodies[i])[1]["answers"]
    original_window = worker.batch_window
    worker.batch_window = 0.5  # long enough that the eight concurrent arrivals collapse into few batches
    try:
        threads = [threading.Thread(target=post, args=(i,)) for i in range(len(bodies))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        worker.batch_window = original_window
    for got, want in zip(results, sequential):
        for name, answer in want.items():
            question = next(q for q in Request.from_dict(EXAMPLE).questions if q.id == name)
            assert all(abs(a - b) < 1e-6 for a, b in zip(_distribution(got[name], question), _distribution(answer, question)))
    after = call(served, path="/v1/models")[1]["stats"]
    assert after["requests"] - before["requests"] == len(bodies)  # GET /v1/models is not an inference request
    assert after["batches"] - before["batches"] < len(bodies) and max(int(k) for k in after["batch_sizes"]) >= 2
    assert after["mean_batch_size"] == after["requests"] / after["batches"] and after["queue_depth"] == 0


def test_rate_limit_429_with_retry_after(served):
    from janus.ratelimit import TokenBucket
    server = served["server"]
    server.limiter = TokenBucket.per_minute(60)  # 1 per second, burst 1
    try:
        assert raw_call(served, EXAMPLE)[0] == 200
        status, body, headers = raw_call(served, EXAMPLE)
        assert status == 429 and body["error"]["type"] == "rate_limit_error"
        assert headers["Retry-After"] == "1" and 0 < int(headers["retry-after-ms"]) <= 1000 and headers["Connection"] == "close"
        assert call(served, path="/v1/models")[1]["limits"]["requests_per_minute"] == 60  # GET is not limited
    finally:
        server.limiter = None
    assert raw_call(served, EXAMPLE)[0] == 200


def test_token_bucket_refills_per_key():
    from janus.ratelimit import TokenBucket
    now = [0.]
    bucket = TokenBucket(rate=2., burst=2., clock=lambda: now[0])  # 2 per second, burst 2
    assert bucket.take("a") == 0. and bucket.take("a") == 0.
    assert bucket.take("a") == pytest.approx(0.5) and bucket.take("b") == 0.  # keys are independent
    now[0] = 0.5
    assert bucket.take("a") == 0. and bucket.take("a") == pytest.approx(0.5)
    now[0] = 100.
    assert [bucket.take("a") for _ in range(3)] == [0., 0., pytest.approx(0.5)]  # the burst caps the refill
    with pytest.raises(ValueError):
        TokenBucket(0, 1)
    assert TokenBucket.per_minute(6).burst == 1. and TokenBucket.per_minute(1200).rate == 20.


def test_request_id_is_honoured_echoed_and_logged(served, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="janus.server"):
        status, body, headers = raw_call(served, EXAMPLE, [("x-typesafe-request-id", "client-id.42")])
        assert status == 200 and headers["x-typesafe-request-id"] == "client-id.42"
        _, _, headers = raw_call(served, EXAMPLE, [("x-typesafe-request-id", "not an id: spaces")])
        assert len(headers["x-typesafe-request-id"]) == 32 and headers["x-typesafe-request-id"] != "not an id: spaces"
        assert raw_call(served, {"model": "nope", **EXAMPLE}, [("x-typesafe-request-id", "bad-1")])[2]["x-typesafe-request-id"] == "bad-1"
    lines = [r.getMessage() for r in caplog.records if "request_id=client-id.42" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "POST /v1/systemone" in line and "status=200" in line and "model=janus-local-test" in line
    assert "questions=3" in line and "latency_ms=" in line and "input_tokens=" + str(body["usage"]["input_tokens"]) in line
    failed = [r.getMessage() for r in caplog.records if "request_id=bad-1" in r.getMessage()]
    assert len(failed) == 1 and "status=422" in failed[0] and "model=-" in failed[0]


def test_prefix_cache_hit_equals_miss_and_counts(tmp_path):
    """The state-pass cache on the tiny hybrid checkpoint (CPU, fp32): a hit answers exactly like the miss, a batch
    mixing hits and misses answers like the uncached worker, the counters and the token-charged eviction follow."""
    from janus.hybrid import TINY
    from janus.server import PrefixCache, Worker
    torch.manual_seed(0)
    path = tmp_path / "hybrid.pt"
    model = DecisionModel(ModelConfig(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4,
                                      hidden_size=64, layers=4, dtype="float32", attention="sdpa", max_tokens=1024, tree_positions="continue"))
    checkpoint(model, path, {"step": 0})
    plain = Worker(path, fast=False, prefix_cache_tokens=0)
    cached = Worker(path, fast=False, prefix_cache_tokens=5000)
    assert plain.cache is None and cached.cache is not None and cached.scope == plain.scope
    requests = [Request.from_dict(EXAMPLE), Request.from_dict({**EXAMPLE, "state": "A different customer message, longer than the first one."})]
    jobs = [(cached.pack(r), cached.temperatures(r), cached.key(r), None) for r in requests]
    reference = plain._run_many([(p, t, None, None) for p, t, _, _ in jobs])
    miss = cached._run_many(jobs[:1])
    hit = cached._run_many(jobs[:1])
    assert cached.cache.hits == 1 and cached.cache.misses == 1 and cached.cache.snapshot()["entries"] == 1
    assert miss == hit  # the same numbers, not merely close: the branch levels see the same level-0 rows
    for a, b in zip(hit[0], reference[0]):
        assert all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    # A case or whitespace variant of a cached state is a different token sequence, so it must miss, not alias.
    variant = Request.from_dict({**EXAMPLE, "state": "  ".join(EXAMPLE["state"].upper().split())})
    assert cached.key(variant) != cached.key(requests[0])
    mixed = cached._run_many(jobs + jobs[:1])  # one hit, one miss, one duplicate of the hit in the same batch
    assert cached.cache.hits == 2 and cached.cache.misses == 2 and cached.cache.snapshot()["entries"] == 2
    for got, want in zip(mixed, reference + reference[:1]):
        for a, b in zip(got, want):
            assert all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    stats = cached.snapshot()["prefix_cache"]
    assert stats["tokens"] == sum(p.state_length for p, _, _, _ in jobs) and stats["charged_tokens"] > stats["tokens"] > 0
    assert stats["charged_tokens"] <= 5000 and stats["bytes"] > 0
    # Eviction by charge: a capacity below one entry's charge keeps nothing; two entries over capacity keep the recent one.
    small = PrefixCache(1)
    small.put("a", cached.cache.entries[jobs[0][2]][0])
    assert small.snapshot()["entries"] == 0
    charge = cached.cache.entries[jobs[0][2]][1]
    lru = PrefixCache(charge + 1)
    lru.put("a", cached.cache.entries[jobs[0][2]][0])
    lru.put("b", cached.cache.entries[jobs[1][2]][0])
    assert list(lru.entries) == ["b"] and lru.get("a") is None and lru.get("b") is not None
    # The uncached path is what a caller without a key gets, whatever the cache.
    keyless = cached._run_many([(jobs[0][0], jobs[0][1], None, None)])
    assert cached.cache.hits == 2 and all(abs(x - y) < 1e-6 for a, b in zip(keyless[0], reference[0]) for x, y in zip(a, b))


def test_option_order_reread_gated_by_entropy(tmp_path):
    """--reread-entropy on the tiny hybrid checkpoint (CPU): a near-uniform first pass (hot temperature) re-reads the
    Choice questions with reversed options through the same queue and answers the mean of the two views; a confident
    first pass (cold temperature) does not; usage counts both passes and the second pass hits the prefix cache."""
    from dataclasses import replace
    from janus.hybrid import TINY
    from janus.server import normalised_entropy
    torch.manual_seed(0)
    path = tmp_path / "hybrid.pt"
    model = DecisionModel(ModelConfig(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4,
                                      hidden_size=64, layers=4, dtype="float32", attention="sdpa", max_tokens=1024, tree_positions="continue"))
    checkpoint(model, path, {"step": 0})
    server = make_server(path, port=0, fast=False, prefix_cache_tokens=5000, reread_entropy=0.5)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    served = {"url": f"http://127.0.0.1:{server.server_address[1]}", "server": server}
    worker = server.worker
    request = Request.from_dict(EXAMPLE)  # route: 3-option choice; transfer_problem: noul; frustration: score
    assert normalised_entropy([1.]) == 0. and abs(normalised_entropy([.25] * 4) - 1.) < 1e-12 and normalised_entropy([0., 1.]) == 0.
    try:
        worker.calibration = {"temperature": 1e-3}  # near one-hot: no re-read
        _, confident = call(served, body=EXAMPLE)
        assert worker.stats["reread"] == 0 and worker.cache.hits == 0
        worker.calibration = {"temperature": 1e3}  # near uniform: re-read
        _, hedged = call(served, body=EXAMPLE)
    finally:
        worker.calibration = {"temperature": 1e3}
    assert worker.stats["reread"] == 1 and worker.stats["requests"] == 3 and worker.cache.hits == 2  # first passes miss, then hit, the second pass hits
    assert hedged["usage"]["input_tokens"] == 2 * confident["usage"]["input_tokens"]
    # By hand through the worker: the mean of the original and the reversed-option view mapped back to the original keys.
    route = request.questions[0]
    reversed_request = replace(request, questions=(replace(route, options=route.options[::-1]),) + request.questions[1:])
    first, second = (worker._run_many([(worker.pack(r), worker.temperatures(r), worker.key(r), None)])[0] for r in (request, reversed_request))
    assert normalised_entropy(first[0]) > 0.5
    expected = [(a + b) / 2 for a, b in zip(first[0], second[0][::-1])]
    got = _distribution(hedged["answers"]["route"], route)
    assert all(abs(x - y) < 1e-6 for x, y in zip(got, expected)) and abs(sum(got) - 1) < 1e-9
    for name, i in (("transfer_problem", 1), ("frustration", 2)):  # noul and score: the first pass, untouched
        assert all(abs(x - y) < 1e-6 for x, y in zip(_distribution(hedged["answers"][name], request.questions[i]),
                                                     first[i][1:] if i == 1 else first[i]))
    assert set(hedged["answers"]["route"]) == {"type", "choice", "probabilities", "confidence"}  # schema unchanged
    server.shutdown()


def test_out_of_memory_batch_releases_the_cache_and_retries_once(served, monkeypatch, caplog):
    """A batch that raises torch.OutOfMemoryError is retried once after torch.cuda.empty_cache(); a second failure
    falls to the one-at-a-time path, so the request still answers."""
    worker = served["server"].worker
    calls = {"run_many": 0, "empty_cache": 0}
    run_many = worker._run_many

    def failing(jobs):
        calls["run_many"] += 1
        if calls["run_many"] == 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return run_many(jobs)
    monkeypatch.setattr(worker, "_run_many", failing)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1))
    with caplog.at_level("WARNING", logger="janus.server"):
        status, body = call(served, body=EXAMPLE)
    assert status == 200 and calls == {"run_many": 2, "empty_cache": 1}
    assert any("retrying once" in record.message for record in caplog.records)
