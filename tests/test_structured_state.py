"""Structured (object and array) states: accepted shapes round-trip through Request.from_dict/to_dict byte for byte,
serialise into model input deterministically, hash stably, pack, and pass the server's parser and the pattern helpers
unchanged. The shapes that do not round-trip are pinned here too (docs/phase4/robustness.md, "Accepted state shapes")."""

import hashlib
import json
import threading

import pytest
import torch

from janus.data import state_hash
from janus.model import DecisionModel, ModelConfig
from janus.packing import ByteTokenizer, pack_request
from janus.patterns import FakeEvaluator, extract_candidate, intent_route, speculative_fanout
from janus.schema import Request, text
from janus.server import make_server
from janus.training import checkpoint

LONG = "x" * 5_000 + " ünïcödé " + "y" * 5_000  # 20k bytes of state in all: under the 32k state-plus-question budget
SHAPES = {
    "string": "Order 123. The customer wants the second cheapest option.",
    "flat_object": {"order_id": 123, "customer_request": "second cheapest", "vip": True, "note": None},
    "nested_object": {"order": {"id": 123, "lines": [{"sku": "A-1", "qty": 2, "price": 9.99}, {"sku": "B-2", "qty": 1, "price": 0.5}]},
                      "flags": {"rush": False, "gift": True}, "empty": {}, "nothing": []},
    "array": [{"name": "Atlas", "price": 45}, {"name": "Borealis", "price": 70}, "loose string", 3, 4.5, True, None],
    "unicode": {"title": "Résumé — naïve café ☕", "cjk": "東京都", "emoji": "🙂", "rtl": "שלום", "escaped": "quote \" backslash \\ newline \n tab \t"},
    "numbers": {"int": 7, "big": 2 ** 63, "negative": -12, "float": 1.0, "small": 1e-9, "large": 1.5e30, "zero": 0, "neg_zero_float": -0.0},
    "booleans_null": [True, False, None, {"ok": True, "missing": None}],
    "long_strings": {"body": LONG, "list": [LONG[:5000], LONG[5000:]]},
    "nested_deep": {"a": {"b": {"c": {"d": {"e": {"f": [[[[1]]]]}}}}}},
    "keys_needing_sort": {"b": 1, "a": 2, "c": {"z": 1, "y": 2}},
    "empty_object": {},
    "empty_array": [],
}
QUESTIONS = {"pick": {"type": "choice", "instructions": "Which option?", "criteria": {"o0": "Atlas", "o1": "Borealis", "none": None}},
             "ok": {"type": "noul", "instructions": "Is it fine?"},
             "level": {"type": "score", "instructions": "How much?", "criteria": ["low", "high"]}}


def request_dict(state):
    return {"state": state, "questions": {k: ({**v, "criteria": {kk: "" if vv is None else vv for kk, vv in v["criteria"].items()}}
                                              if isinstance(v.get("criteria"), dict) else v) for k, v in QUESTIONS.items()}, "group_id": "t"}


def canonical(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("name", list(SHAPES))
def test_shape_round_trips_serialises_deterministically_hashes_stably_and_packs(name):
    state = SHAPES[name]
    raw = request_dict(state)
    first = Request.from_dict(json.loads(json.dumps(raw)))
    once = first.to_dict()
    second = Request.from_dict(json.loads(json.dumps(once)))
    assert second == first and second.to_dict() == once  # from_dict -> to_dict -> from_dict is a fixed point
    assert json.dumps(once, ensure_ascii=False, sort_keys=True) == json.dumps(second.to_dict(), ensure_ascii=False, sort_keys=True)
    # The model input is the canonical JSON text of the value (sorted keys, no ASCII escaping, no NaN); strings pass through.
    assert first.state == canonical(state) == text(state)
    assert isinstance(first.state, str) and once["state"] == first.state
    if not isinstance(state, str):
        assert json.loads(first.state) == state  # the value survives, key order and whitespace do not
    # Deterministic serialisation and stable hashes across parses and across processes' dict orderings.
    shuffled = json.loads(json.dumps(raw)) if isinstance(state, str) else {**raw, "state": json.loads(json.dumps(state))[::-1] if isinstance(state, list) else dict(reversed(list(state.items())))}
    again = Request.from_dict(shuffled)
    assert again.state == first.state if not isinstance(state, list) else json.loads(again.state) == state[::-1]
    assert state_hash(first.state) == state_hash(second.state) == hashlib.sha256(" ".join(first.state.lower().split()).encode()).hexdigest()
    a = pack_request(first, ByteTokenizer(), "tree", 10 ** 6)
    b = pack_request(second, ByteTokenizer(), "tree", 10 ** 6)
    assert torch.equal(a.input_ids, b.input_ids) and a.token_count == b.token_count > 0
    assert "State:\n" + first.state + "\n" in bytes(t - 1 for t in a.input_ids[0].tolist()).decode()


def test_shapes_that_do_not_round_trip_are_rejected_or_canonicalised():
    # A top-level object with an `images` key is an image state, not a structured state.
    with pytest.raises(ValueError, match="image"):
        Request.from_dict(request_dict({"images": ["not a ref"], "text": "x"}))
    with pytest.raises(ValueError, match="image"):
        Request.from_dict(request_dict({"images": []}))
    # NaN and infinity are not JSON and are refused rather than serialised as `NaN`.
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Request.from_dict(request_dict({"value": bad}))
    # Python-only shapes (never produced by JSON input): int keys become strings and tuples arrays, but a dict mixing
    # str and int keys cannot be sorted and raises TypeError (the server reports it as a 422 "Malformed request").
    assert Request.from_dict(request_dict({3: "three", 1: ("a", "b")})).state == '{"1": ["a", "b"], "3": "three"}'
    with pytest.raises(TypeError):
        Request.from_dict(request_dict({3: "three", "b": (1, 2)}))
    assert Request.from_dict(request_dict({"b": 1, "a": 2})).state == Request.from_dict(request_dict({"a": 2, "b": 1})).state
    # Whitespace-only or empty strings are accepted states (the question must be nonempty, the state may be blank).
    assert Request.from_dict(request_dict("")).state == "" and Request.from_dict(request_dict("   ")).state == "   "
    # A string that looks like JSON stays a string (no double encoding) and is byte-identical after the round trip.
    looks = '{"a": 1}'
    assert Request.from_dict(request_dict(looks)).state == looks
    # A float that is integral keeps its float spelling; -0.0 keeps its sign; large ints are exact.
    assert Request.from_dict(request_dict([1.0, -0.0, 2 ** 63])).state == "[1.0, -0.0, 9223372036854775808]"


def test_pattern_helpers_pass_structured_states_through_unchanged():
    state = SHAPES["nested_object"]
    evaluator = FakeEvaluator({"intent": {"refund": .9, "none": .1}, "amount": {"c0": .8, "c1": .2}})
    route = intent_route(evaluator, state, {"refund": "wants money back", "cancel": "wants to cancel"}, "Neither.")
    fanout = speculative_fanout(evaluator, state, {k: v for k, v in request_dict(state)["questions"].items()})
    extracted = extract_candidate(evaluator, state, "amount", ["9.99", "0.5"])
    assert route["intent"] == "refund" and set(fanout) == set(QUESTIONS) and extracted["value"] == "9.99"
    assert all(r.state == canonical(state) for r in evaluator.requests) and len(evaluator.requests) == 3
    assert all(json.loads(r.state) == state for r in evaluator.requests)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    torch.set_num_threads(2)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, layers=1, head_rank=8, max_tokens=65536,
                                      max_state_plus_question=32768))
    checkpoint(model, path, {"step": 0})
    server = make_server(path, port=0, model_id="jev-local-test", queue_size=8)
    server.token = "test-token"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{server.server_address[1]}", "model": model}
    server.shutdown()


@pytest.mark.parametrize("name", list(SHAPES))
def test_server_parses_every_shape_like_the_library(served, name):
    from urllib.request import Request as HTTPRequest, urlopen
    body = {"state": SHAPES[name], "questions": QUESTIONS}  # null criteria as the API allows them
    outgoing = HTTPRequest(served["url"] + "/v1/systemone", data=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json", "Authorization": "Bearer test-token"}, method="POST")
    with urlopen(outgoing, timeout=60) as incoming:
        assert incoming.status == 200
        answer = json.load(incoming)
    assert set(answer["answers"]) == set(QUESTIONS) and answer["answers"]["pick"]["choice"] in ("o0", "o1", "none")
    local = Request.from_dict(request_dict(SHAPES[name]))
    m = served["model"]
    packed = pack_request(local, m.tokenizer, m.packing_mode, m.config.max_tokens, **m.packing_kwargs)
    assert answer["usage"]["input_tokens"] == packed.token_count + sum(p.token_count for p in packed.branch_packs)
