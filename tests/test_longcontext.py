"""Phase 4 long context: the two token budgets, the long-context set, and the execution benchmark."""

import json

import pytest
import torch

from janus.longbench import benchmark, markdown, summarize
from janus.model import DecisionModel, ModelConfig
from janus.packing import ByteTokenizer, pack_request
from janus.schema import Request
from janus.synth.longcontext import DEPTHS, RESERVE, prepare_longcontext
from janus.training import checkpoint, load_checkpoint


def request(state_bytes=100, questions=3, options=4):
    return Request.from_dict({"state": "s" * state_bytes, "questions": {
        f"q{i}": {"type": "choice", "instructions": "Pick one.", "criteria": {f"o{k}": f"option {k}" for k in range(options)},
                  "target": [1.] + [0.] * (options - 1)} for i in range(questions)}})


def test_packer_names_the_budget_it_exceeded():
    r = request()
    packed = pack_request(r, ByteTokenizer(), "tree", 10 ** 6)
    longest = max(sum(e - s for s, e in [(b.start, b.end)] + list(b.leaves)) for b in packed.branches)
    with pytest.raises(ValueError, match=f"exceeding max_tokens={packed.token_count - 1}"):
        pack_request(r, ByteTokenizer(), "tree", packed.token_count - 1)
    with pytest.raises(ValueError, match="exceeding max_state_plus_question=200"):
        pack_request(r, ByteTokenizer(), "tree", 10 ** 6, max_state_plus_question=200)
    # The boundary is exact: state (with its "State:\n" framing) plus the longest question fits, one token less does not.
    assert pack_request(r, ByteTokenizer(), "tree", 10 ** 6, max_state_plus_question=packed.state_length + longest).token_count == packed.token_count
    with pytest.raises(ValueError, match="state-plus-question"):
        pack_request(r, ByteTokenizer(), "tree", 10 ** 6, max_state_plus_question=packed.state_length + longest - 1)
    # Without the second budget the total check fires first, exactly as before.
    with pytest.raises(ValueError, match="exceeding max_tokens"):
        pack_request(r, ByteTokenizer(), "tree", packed.state_length + longest - 1)
    for mode in ("listwise", "independent", "decoder"):
        with pytest.raises(ValueError, match="state-plus-question"):
            pack_request(r, ByteTokenizer(), mode, 10 ** 6, max_state_plus_question=120)
        pack_request(r, ByteTokenizer(), mode, 10 ** 6, max_state_plus_question=10 ** 6)


def test_model_exposes_both_budgets_and_the_override_survives_a_checkpoint(tmp_path):
    config = ModelConfig(backbone="tiny", mode="tree", adaptation="full", hidden_size=32, layers=1, max_tokens=4096)
    model = DecisionModel(config)
    assert config.state_budget == 4096 and model.packing_kwargs["max_state_plus_question"] == 4096
    assert model.backbone.config.max_position_embeddings == 4096
    with pytest.raises(ValueError, match="max_state_plus_question"):
        DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, layers=1, max_tokens=1024, max_state_plus_question=2048))
    weights = tmp_path / "model.pt"
    checkpoint(model, weights, {})
    loaded, _ = load_checkpoint(weights, max_tokens=8192, max_state_plus_question=1024)
    assert loaded.config.max_tokens == 8192 and loaded.packing_kwargs["max_state_plus_question"] == 1024
    # The position table follows the state-plus-question budget, so a request of 8k tokens with a short state runs.
    r = request(state_bytes=200, questions=12, options=8)
    packed = pack_request(r, loaded.tokenizer, "tree", loaded.config.max_tokens, **loaded.packing_kwargs)
    assert packed.token_count > 1024
    assert len(loaded(packed)) == 12
    same, _ = load_checkpoint(weights)
    assert same.config.max_tokens == 4096 and same.config.max_state_plus_question is None


def qwen_tokenizer():
    try:
        from transformers import AutoTokenizer
        from janus.hybrid import HybridBackbone
        from janus.synth.cardinality import QWEN
        path = HybridBackbone.snapshot_dir(QWEN[0])
        assert path
        return AutoTokenizer.from_pretrained(path)
    except Exception:
        pytest.skip("Qwen tokenizer not cached locally")


@pytest.fixture(scope="module")
def small_set(tmp_path_factory):
    out = tmp_path_factory.mktemp("lc") / "set"
    manifest = prepare_longcontext(out, qwen_tokenizer(), per_cell=1, train=3, dev=2, lengths=(1024, 2048), depths=(.1, .9),
                                   tokenizer_name="the-cached-qwen-tokenizer")
    return out, manifest


def test_generated_states_hit_their_cell_and_depth(small_set):
    out, manifest = small_set
    tokenizer = qwen_tokenizer()
    rows = [json.loads(l) for l in (out / "bench.jsonl").read_text().splitlines()]
    assert manifest["counts"] == {"train": 3, "dev": 2, "bench": 4} and len(rows) == 4
    assert sorted(r["cell"] for r in rows) == ["1k:0.1", "1k:0.9", "2k:0.1", "2k:0.9"]
    for row in rows:
        target = row["length"] - RESERVE
        tokens = len(tokenizer.encode(row["state"], add_special_tokens=False))
        assert row["state_tokens"] == tokens and abs(tokens - target) <= max(16, min(.02 * target, RESERVE // 8))
        assert abs(row["record_depth"] - row["depth"]) <= max(.05, 36 / target)
        assert 4 <= row["question_count"] == len(row["questions"]) <= 8
        ident = next(w for w in Request.from_dict(row).questions[0].instructions.replace("?", " ").split() if w.isdigit() and len(w) == 6)
        assert row["state"].count(ident) == 1  # exactly one record carries the id every question names
        for q in row["questions"].values():
            assert q["target"] is not None and abs(sum(q["target"]) - 1) < 1e-6
        # Both budgets hold with the real tokenizer: state plus the longest question under the cell, in the longest layout.
        packed = pack_request(Request.from_dict(row), tokenizer, "tree", 10 ** 6, score_block="full", max_state_plus_question=row["length"])
        assert packed.state_length <= row["length"] - RESERVE + RESERVE // 8 + 3
    train = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert {r["length"] for r in train} <= {1024, 2048} and all(.05 <= r["depth"] <= .95 for r in train)
    assert set(manifest["tokens"]["bench"]) == {"1k:0.1", "1k:0.9", "2k:0.1", "2k:0.9"}
    assert manifest["tokens"]["bench"]["2k:0.9"]["packed_tokens_max"] > manifest["tokens"]["bench"]["2k:0.9"]["state_tokens"]["max"]
    assert list(DEPTHS) == [.1, .5, .9]
    assert manifest["tokenizer"] == "the-cached-qwen-tokenizer" and manifest["token_rule"].startswith("the-cached-qwen-tokenizer tokenizer")


def records_for(length, depth, group, ms, questions=(1, 4), skipped=False, levels=None, shift=0.):
    out = []
    for q in questions:
        r = {"cell": f"{length // 1024}k:{depth}", "length": length, "depth": depth, "family": "rel", "group_id": group,
             "questions": q, "full": q == questions[-1]}
        if skipped:
            out.append({**r, "skipped": "OutOfMemoryError: CUDA"})
            continue
        r.update({"packed_tokens": length + 100 * q, "ms": ms * q, "peak_bytes": 1000 * length * q,
                  "logits": [[1. + shift, 0., 0.]] * q, "targets": [[1., 0., 0.]] * q, "kinds": ["choice"] * q, "levels_ms": levels})
        out.append(r)
    return out


def test_summarize_and_markdown_on_synthetic_timings():
    records = (records_for(1024, .1, "a", 10.) + records_for(1024, .1, "b", 20.) + records_for(1024, .9, "c", 30., levels=[5., 2., 1.])
               + records_for(4096, .5, "d", 100.) + records_for(4096, .5, "e", 0., skipped=True))
    reference = records_for(1024, .1, "a", 10.) + records_for(1024, .1, "b", 20., shift=-3.) + records_for(4096, .5, "d", 100.)
    summary = summarize(records, reference)
    one = summary["latency"]["1k|1"]
    assert one["count"] == 3 and one["p50_ms"] == 20. and one["p95_ms"] == pytest.approx(29.) and one["peak_bytes_max"] == 1024 * 1000
    assert summary["latency"]["4k|4"] == {"length": 4096, "questions": 4, "skipped": 1, "count": 1, "p50_ms": 400., "p95_ms": 400.,
                                          "peak_bytes_max": 4096 * 4000, "packed_tokens_mean": 4496.}
    assert summary["accuracy_by_cell"]["1k:0.1"] == {"count": 8, "accuracy": 1., "nll": pytest.approx(0.5514, abs=1e-3)}
    assert summary["accuracy_by_cell"]["4k:0.5"]["count"] == 4 and summary["accuracy_by_kind"]["choice"]["count"] == 16
    assert summary["hybrid_levels"] == {"1k": {"count": 2, "level_p50_ms": [5., 2., 1.]}}
    assert summary["agreement"]["1k"] == {"compared": 4, "max_abs_logit_diff": 3., "argmax_flips": 5, "questions": 10}
    assert summary["agreement"]["4k"]["argmax_flips"] == 0
    text = markdown(summary, "t")
    assert "| 1k | 20 / 29 | 80 / 116 |" in text and "(1 skipped)" in text and "| 1k | 4 | 3.00e+00 | 5 / 10 |" in text
    assert "| 1k | 5, 2, 1 |" in text and "1.000 (0.55, n=8)" in text


def test_benchmark_cli_path_on_cpu_with_the_tiny_backbone(small_set, tmp_path):
    out, _ = small_set
    torch.manual_seed(0)
    model = DecisionModel(ModelConfig(backbone="tiny", mode="tree", adaptation="full", hidden_size=32, layers=1, max_tokens=16384, attention="sdpa"))
    weights = tmp_path / "tiny.pt"
    checkpoint(model, weights, {})
    result = benchmark(weights, out / "bench.jsonl", tmp_path / "bench", device="cpu", cells=["1k"], limit=1, warmup=1)
    assert result["reference_attention"] == "eager" and result["states"] == 2
    records = [r for r in result["records"] if "skipped" not in r]
    assert records and all(r["peak_bytes"] is None and r["ms"] > 0 for r in records) and all("skipped" not in r for r in result["records"])
    assert {r["questions"] for r in records if r["full"]} and sorted({r["cell"] for r in records}) == ["1k:0.1", "1k:0.9"]
    agreement = result["summary"]["agreement"]["1k"]
    assert agreement["compared"] == len(records) and agreement["max_abs_logit_diff"] < 1e-3 and agreement["argmax_flips"] == 0
    assert set(result["summary"]["accuracy_by_cell"]) == {"1k:0.1", "1k:0.9"} and result["summary"]["hybrid_levels"] == {}
    assert (tmp_path / "bench" / "benchmark.md").read_text().startswith("# ")
    with pytest.raises(FileExistsError):
        benchmark(weights, out / "bench.jsonl", tmp_path / "bench", device="cpu", cells=["1k"], limit=1)
    # A budget too small for the cell is recorded per request, not raised.
    small = benchmark(weights, out / "bench.jsonl", tmp_path / "small", device="cpu", cells=["1k"], limit=1, warmup=0,
                      reference="none", max_tokens=16384, max_state_plus_question=512)
    assert all("state-plus-question" in r["skipped"] for r in small["records"]) and small["summary"]["latency"]["1k|1"]["skipped"] == 2
