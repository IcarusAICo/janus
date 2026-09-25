"""Offline tests for the demo System One client and metrics. No paid API calls."""
from types import SimpleNamespace

import pytest

from demos.common.metrics import TYPESAFE_INPUT_USD_PER_MTOK, summarize
from demos.common.systemone import Decision, FakeTransport, SystemOne, questions_from_dicts


def test_questions_from_dicts_keep_choice_score_noul_shapes():
    built = questions_from_dicts({
        "route": {"type": "choice", "instructions": "Which team?", "criteria": {"a": "A", "b": "B"}},
        "mood": {"type": "score", "instructions": "How bad?", "criteria": ["calm", "hot"]},
        "urgent": {"type": "noul", "instructions": "Urgent?"},
    })
    assert set(built) == {"route", "mood", "urgent"}
    assert built["route"].criteria == {"a": "A", "b": "B"}
    assert built["mood"].criteria == ["calm", "hot"]
    assert built["urgent"].instructions == "Urgent?"


def test_systemone_records_latency_tokens_and_normalizes_answers():
    transport = FakeTransport({
        "route": {"type": "choice", "choice": "payments",
                  "probabilities": {"payments": 0.8, "other": 0.2}, "confidence": 0.8},
        "urgent": {"type": "noul", "noul": 0.91},
        "mood": {"type": "score", "score": 1.2, "legend": {0: "calm", 1: "hot"},
                 "probabilities": {0: 0.4, 1: 0.6}, "confidence": 0.6},
    }, model="jev-latest", input_tokens=231, latency_ms=120)
    client = SystemOne(backend="typesafe", transport=transport, env_file=None, api_key="k")
    decision = client.decide("the payout failed", {
        "route": {"type": "choice", "instructions": "Team?", "criteria": {"payments": "pay", "other": "else"}},
        "urgent": {"type": "noul", "instructions": "Urgent?"},
        "mood": {"type": "score", "instructions": "Mood?", "criteria": ["calm", "hot"]},
    })
    assert isinstance(decision, Decision)
    assert decision.choice("route") == "payments"
    assert decision.noul("urgent") == pytest.approx(0.91)
    assert decision.score("mood") == pytest.approx(1.2)
    assert decision.record.latency_ms == pytest.approx(120)
    assert decision.record.input_tokens == 231
    assert decision.record.model == "jev-latest"
    assert decision.record.question_ids == ("route", "urgent", "mood")
    assert transport.calls[0]["state"] == "the payout failed"


def test_systemone_local_backend_uses_given_base_url():
    transport = FakeTransport({"x": {"type": "noul", "noul": 0.1}}, base_url_seen=None)
    client = SystemOne(backend="local", base_url="http://127.0.0.1:8080",
                       transport=transport, env_file=None, api_key="tok")
    client.decide("s", {"x": {"type": "noul", "instructions": "X?"}})
    assert transport.base_url == "http://127.0.0.1:8080"


def test_summarize_latency_cost_and_qps():
    records = [
        SimpleNamespace(latency_ms=100, input_tokens=1000, output_tokens=0),
        SimpleNamespace(latency_ms=200, input_tokens=3000, output_tokens=0),
        SimpleNamespace(latency_ms=300, input_tokens=1000, output_tokens=0),
    ]
    out = summarize(records, wall_seconds=2.0)
    assert out["calls"] == 3
    assert out["latency_p50_ms"] == pytest.approx(200)
    assert out["latency_p95_ms"] == pytest.approx(300)
    assert out["input_tokens"] == 5000
    assert out["estimated_usd"] == pytest.approx(5000 / 1_000_000 * TYPESAFE_INPUT_USD_PER_MTOK)
    assert out["qps"] == pytest.approx(1.5)
    assert out["wall_seconds"] == pytest.approx(2.0)


def test_summarize_openai_luna_prices_input_and_output():
    from demos.common.metrics import rates_for

    in_rate, out_rate = rates_for("gpt", "gpt-5.6-luna")
    records = [SimpleNamespace(latency_ms=100, input_tokens=1_000_000, output_tokens=1_000_000)]
    out = summarize(records, wall_seconds=1.0, input_usd_per_mtok=in_rate, output_usd_per_mtok=out_rate)
    assert in_rate == pytest.approx(0.20)
    assert out_rate == pytest.approx(1.20)
    assert out["estimated_usd"] == pytest.approx(1.40)


def test_rates_for_local_is_free():
    from demos.common.metrics import rates_for

    assert rates_for("local", "janus-9b") == (0.0, 0.0)
