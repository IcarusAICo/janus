import pytest
import torch

from janus.model import DecisionModel, ModelConfig
from janus.patterns import (NONE, FakeEvaluator, LocalEvaluator, RemoteEvaluator, composite_score,
                          confidence_gated_route, extract_candidate, intent_route, select_arguments,
                          select_function, speculative_fanout)
from janus.schema import Request
from janus.training import checkpoint

STATE = {"message": "My transfer failed three times!!! Refund me, this is a bug."}
TICKET = {"category": {"type": "choice", "instructions": "Kind of ticket?",
                       "criteria": {"bug_report": "A bug", "billing": "Billing", "other": "Other"}},
          "bug_severity": {"type": "score", "instructions": "How severe?", "criteria": ["minor", "major", "critical"]},
          "refund_requested": {"type": "noul", "instructions": "Is a refund requested?"}}


def test_speculative_fanout_returns_every_answer_in_one_request():
    fake = FakeEvaluator({"category": {"bug_report": .7, "billing": .2, "other": .1},
                          "bug_severity": {"0": 0., "1": .5, "2": .5}, "refund_requested": {"true": .9, "false": .1}})
    out = speculative_fanout(fake, STATE, TICKET)
    assert len(fake.requests) == 1 and [q.id for q in fake.requests[0].questions] == list(TICKET)
    assert out["category"].choice == "bug_report" and out["category"].confidence == pytest.approx(.7)
    assert out["bug_severity"].score == pytest.approx(1.5)
    assert out["refund_requested"].noul == pytest.approx(.9)
    assert out["refund_requested"].confidence == pytest.approx(.9)


def test_confidence_gate_takes_both_branches_and_records_the_decision():
    calls = []
    fallback = lambda answer: calls.append(answer.choice) or "human"
    confident = FakeEvaluator({"gated": {"bug_report": .8, "billing": .2, "other": 0.}})
    out = confidence_gated_route(confident, STATE, TICKET["category"], .6, fallback)
    assert out == {**out, "accepted": True, "result": "bug_report", "confidence": pytest.approx(.8)}
    unsure = FakeEvaluator({"gated": {"bug_report": .5, "billing": .5, "other": 0.}})
    out = confidence_gated_route(unsure, STATE, TICKET["category"], .6, fallback)
    assert out["accepted"] is False and out["result"] == "human" and calls == ["bug_report"]
    assert confidence_gated_route(unsure, STATE, TICKET["category"], .5, fallback)["accepted"]  # >= threshold


def test_composite_score_weights_normalised_expected_levels():
    rubrics = {"depth": {"instructions": "Depth?", "criteria": ["none", "some", "deep"]},
               "clarity": {"instructions": "Clarity?", "criteria": ["poor", "fine", "clear", "crisp", "perfect"]}}
    fake = FakeEvaluator({"depth": {"0": 0., "1": 0., "2": 1.}, "clarity": {"0": .5, "1": 0., "2": .5, "3": 0., "4": 0.}})
    out = composite_score(fake, "A resume.", rubrics, {"depth": 3, "clarity": 1})
    # depth = 2/2 = 1.0, clarity = expected level 1 / 4 = .25; weights 3:1 -> (3 + .25) / 4
    assert out["score"] == pytest.approx((3 * 1. + 1 * .25) / 4)
    assert composite_score(fake, "A resume.", rubrics, {"depth": 0, "clarity": 1})["score"] == pytest.approx(.25)
    assert all(a.kind == "score" for a in out["answers"].values())
    with pytest.raises(ValueError):
        composite_score(fake, "A resume.", rubrics, {"depth": 1})
    with pytest.raises(ValueError):
        composite_score(fake, "A resume.", rubrics, {"depth": 0, "clarity": 0})


def test_intent_route_returns_none_when_the_none_option_wins():
    intents = {"order_status": "Where is my order", "complaint": "A complaint"}
    fake = FakeEvaluator({"intent": {"order_status": .1, "complaint": .3, NONE: .6}})
    out = intent_route(fake, "hello?", intents, "None of the listed intents")
    assert out["intent"] is None and out["probability"] == pytest.approx(.6)
    assert [o.key for o in fake.requests[0].questions[0].options] == ["order_status", "complaint", NONE]
    out = intent_route(FakeEvaluator({"intent": {"complaint": 1.}}), "This is broken!", intents, "None")
    assert out["intent"] == "complaint" and out["probability"] == pytest.approx(1.)
    with pytest.raises(ValueError, match="reserved"):
        intent_route(fake, "x", {NONE: "clash"}, "None")


def test_function_calling_selects_tool_then_closed_set_arguments_with_min_confidence():
    tools = {"plot_price": "Plot a price chart", "rolling_correlation": "Correlate two symbols"}
    picked = select_function(FakeEvaluator({"function": {"rolling_correlation": .9, "plot_price": .1}}), "NVDA vs SPY", tools)
    assert picked["function"] == "rolling_correlation" and picked["confidence"] == pytest.approx(.9)
    assert select_function(FakeEvaluator({"function": {NONE: 1.}}), "hi", tools)["function"] is None
    enums = {"symbol": {"NVDA": "Nvidia", "SPY": "S&P 500"}, "benchmark": {"NVDA": "Nvidia", "SPY": "S&P 500"},
             "window": {"1mo": "One month", "1y": "One year"}}
    fake = FakeEvaluator({"symbol": {"NVDA": .95, "SPY": .05}, "benchmark": {"SPY": .7, "NVDA": .3}, "window": {NONE: .6, "1mo": .4}})
    call = select_arguments(fake, "NVDA vs SPY", "rolling_correlation", enums)
    assert call["function"] == "rolling_correlation" and call["arguments"] == {"symbol": "NVDA", "benchmark": "SPY"}
    assert call["confidence"] == pytest.approx(.6)  # the least certain judgement, including the omitted one
    assert len(fake.requests) == 1 and all(q.options[-1].key == NONE for q in fake.requests[0].questions)
    assert select_arguments(fake, "x", "noop", {}) == {"function": "noop", "arguments": {}, "confidence": 1., "answers": {}}


def test_extract_candidate_returns_a_verbatim_span_or_none():
    spans = ["$12.50", "$3.00", "$12.50"]
    fake = FakeEvaluator({"total": {"c1": .8, "c0": .2}})
    out = extract_candidate(fake, "Subtotal $3.00, total $12.50", "total", spans)
    assert out["value"] == "$3.00" and out["confidence"] == pytest.approx(.8)
    assert [o.description for o in fake.requests[0].questions[0].options] == ["$12.50", "$3.00", "None of these is the requested value."]
    assert extract_candidate(FakeEvaluator({"total": {NONE: 1.}}), "no money here", "total", spans)["value"] is None
    assert extract_candidate(fake, "empty", "total", [])["value"] is None and len(fake.requests) == 1
    with pytest.raises(ValueError):  # Choice is capped at 255 options including none
        extract_candidate(fake, "big", "total", [str(i) for i in range(255)])


def test_remote_evaluator_wraps_a_client():
    class Client:
        def predict(self, request, cache_dir=None):
            assert cache_dir == "cache"
            return type("R", (), {"probabilities": tuple((1 / len(q.options),) * len(q.options) for q in request.questions)})()

    out = speculative_fanout(RemoteEvaluator(client=Client(), cache_dir="cache"), STATE, TICKET)
    assert set(out) == set(TICKET) and out["refund_requested"].noul == pytest.approx(.5)


def test_local_evaluator_loads_once_and_runs_every_helper_on_the_tiny_backbone(tmp_path):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    model = DecisionModel(ModelConfig(backbone="tiny", mode="tree", adaptation="full", dtype="float32",
                                      hidden_size=32, layers=1, head_rank=16, max_tokens=4096))
    path = tmp_path / "tiny.pt"
    checkpoint(model, path, {})
    evaluator = LocalEvaluator(path)
    assert evaluator.model is evaluator.model  # loaded once, held on the instance
    out = speculative_fanout(evaluator, STATE, TICKET)
    assert set(out) == set(TICKET)
    for answer in out.values():
        assert sum(answer.probabilities.values()) == pytest.approx(1.) and 0 < answer.confidence <= 1
    assert Request.from_dict({"state": STATE, "questions": TICKET}).questions[0].id == "category"
    gate = confidence_gated_route(evaluator, STATE, TICKET["category"], 2., lambda a: "human")
    assert gate["accepted"] is False and gate["result"] == "human"
    assert 0 <= composite_score(evaluator, STATE, {"sev": {"instructions": "?", "criteria": ["a", "b", "c"]}}, {"sev": 1})["score"] <= 1
    assert intent_route(evaluator, STATE, {"a": "A", "b": "B"}, "None")["intent"] in (None, "a", "b")
    call = select_arguments(evaluator, STATE, "noop", {"x": {"1": "one", "2": "two"}})
    assert set(call["arguments"]) <= {"x"}
    assert extract_candidate(evaluator, STATE, "count", ["three", "four"])["value"] in (None, "three", "four")
