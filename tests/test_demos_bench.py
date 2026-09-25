"""Offline tests for Choice benches. No paid API calls."""
from pathlib import Path

import pytest

from demos.bench.run import ece10, score_rows
from demos.bench.tasks import load_systemone_jsonl, synthetic_example, write_synthetic
from demos.common.systemone import FakeTransport, SystemOne


def test_synthetic_label_matches_named_badge():
    item = synthetic_example(17)
    assert item.criteria[item.label] in item.state
    assert item.state.startswith("Choose the exact badge ")


def test_write_and_load_synthetic(tmp_path):
    write_synthetic(tmp_path, train=2, validation=1, test=3, seed=17)
    from demos.bench.tasks import load_jevlike_jsonl

    loaded = load_jevlike_jsonl(tmp_path / "test.jsonl", prefix="t")
    assert len(loaded) == 3
    assert loaded[0].label in loaded[0].criteria
    assert len(loaded[0].criteria) >= 2


@pytest.mark.skipif(not Path("data/public-v1").exists(), reason="needs data/public-v1 (not in the public release)")
def test_mmlu_pro_loader_uses_target_one_hot():
    path = Path("data/public-v1/test_mmlu_pro_1000.jsonl")
    examples = load_systemone_jsonl(path, limit=2)
    assert examples[0].label in examples[0].criteria
    assert len(examples[0].criteria) >= 2


def test_fake_client_scores_top1_top3_and_ece():
    example = synthetic_example(17)
    transport = FakeTransport(
        {
            "answer": {
                "type": "choice",
                "choice": example.label,
                "probabilities": {k: (0.8 if k == example.label else 0.2 / (len(example.criteria) - 1))
                                  for k in example.criteria},
                "confidence": 0.8,
            }
        },
        input_tokens=10,
        latency_ms=5,
    )
    client = SystemOne(backend="typesafe", transport=transport, env_file=None, api_key="k")
    from demos.bench.run import decide_example

    row = decide_example(client, example)
    assert row["correct"] is True
    assert row["top3"] is True
    scored = score_rows([row])
    assert scored["top1"] == 1.0
    assert scored["n"] == 1


def test_shuffled_context_moves_state_and_keeps_label():
    from demos.bench.tasks import shuffled_context

    a = synthetic_example(1)
    b = synthetic_example(2)
    out = shuffled_context([a, b])
    assert out[0].state == b.state
    assert out[0].label == a.label
    assert out[0].label != b.label or a.label == b.label


def test_shuffled_options_preserves_gold_text():
    from demos.bench.tasks import shuffled_options

    item = synthetic_example(17)
    gold = item.criteria[item.label]
    shuffled = shuffled_options([item], seed=3)[0]
    assert shuffled.criteria[shuffled.label] == gold
    assert list(shuffled.criteria) != list(item.criteria) or len(item.criteria) < 3


def test_ece10_matches_single_bin():
    # both in [0.2, 0.3): acc=0.5, conf=0.25, |gap|=0.25
    assert abs(ece10([0.25, 0.25], [True, False]) - 0.25) < 1e-9
    # one example in [0.2, 0.3), acc=1, conf=0.25 → 0.75
    assert abs(ece10([0.25], [True]) - 0.75) < 1e-9


@pytest.mark.skipif(not Path("data/public-v1").exists(), reason="needs data/public-v1 (not in the public release)")
def test_public_family_loader_keeps_noul():
    examples = load_systemone_jsonl("data/public-v1/test.jsonl", limit=1, family="civil")
    assert examples[0].questions["civil:toxic"]["type"] == "noul"


def test_api_questions_drops_list_noul_criteria():
    from demos.bench.run import _api_questions

    out = _api_questions(
        {
            "injected": {
                "type": "noul",
                "instructions": "Injected?",
                "criteria": ["false", "true"],
                "target": [1.0, 0.0],
            }
        }
    )
    assert "target" not in out["injected"]
    assert "criteria" not in out["injected"]


def test_noul_uses_target_yes_mass():
    from demos.bench.run import decide_example
    from demos.bench.tasks import BenchItem

    item = BenchItem(
        example_id="civil:t",
        state="a comment",
        questions={"toxic": {"type": "noul", "instructions": "Toxic?", "target": [0.2, 0.8]}},
    )
    transport = FakeTransport({"toxic": {"type": "noul", "noul": 0.7}}, input_tokens=3, latency_ms=2)
    client = SystemOne(backend="typesafe", transport=transport, env_file=None, api_key="k")
    row = decide_example(client, item)
    assert row["correct"] is True
    assert row["questions"][0]["brier"] == (0.7 - 0.8) ** 2
    scored = score_rows([row])
    assert scored["by_type"]["noul"]["acc"] == 1.0
    assert scored["ece"] > 0


def test_score_argmax_matches_one_hot_target():
    from demos.bench.run import decide_example
    from demos.bench.tasks import BenchItem

    item = BenchItem(
        example_id="hs:t",
        state="prompt",
        questions={
            "helpfulness": {
                "type": "score",
                "instructions": "Helpful?",
                "criteria": ["0", "1", "2"],
                "target": [0.0, 1.0, 0.0],
            }
        },
    )
    transport = FakeTransport(
        {
            "helpfulness": {
                "type": "score",
                "score": 1,
                "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
            }
        },
        input_tokens=3,
        latency_ms=2,
    )
    client = SystemOne(backend="typesafe", transport=transport, env_file=None, api_key="k")
    row = decide_example(client, item)
    assert row["correct"] is True
    assert row["questions"][0]["pred"] == 1


def test_local_score_batch_follows_the_served_budget(monkeypatch):
    import io
    import json
    from types import SimpleNamespace
    from demos.bench import run
    from demos.bench.run import SCORE_BATCH_TOKENS, local_score_batch, score_batch_for

    assert [score_batch_for(n) for n in (4096, 8192, 16384)] == [32, 64, 128]
    assert SCORE_BATCH_TOKENS[128] > 8400 >= SCORE_BATCH_TOKENS[64]
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        body = io.BytesIO(json.dumps({"models": [], "limits": {"max_tokens": 16384}}).encode())
        body.__enter__, body.__exit__ = lambda: body, lambda *a: None
        return body

    monkeypatch.setattr(run, "urlopen", fake_urlopen)
    monkeypatch.setenv("JANUS_SERVER_TOKEN", "local-token")
    client = SimpleNamespace(base_url="http://127.0.0.1:8080", env_file="/nonexistent/env.sh")
    assert local_score_batch(client) == 128 and seen["url"].endswith("/v1/models") and seen["auth"] == "Bearer local-token"
    monkeypatch.setattr(run, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert local_score_batch(client) == 32
