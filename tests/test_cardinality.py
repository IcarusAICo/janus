"""Phase 4: level-independent Score packing, the cardinality sweep data, and its report."""

import json
import random

import pytest
import torch

from janus.data import write_jsonl
from janus.model import DecisionModel, ModelConfig
from janus.packing import ByteTokenizer, pack_request
from janus.schema import Request, decode


def example():
    return {"state": "A red card.", "questions": {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"r": "red", "b": "blue", "g": "green"}, "target": [1, 0, 0]},
        "red": {"type": "noul", "instructions": "Is it red?", "target": [0, 1]},
        "intensity": {"type": "score", "instructions": "Intensity?", "criteria": ["low", "medium", "high"], "target": [0, 0, 1]}}}


def text_of(packed, positions=None):
    ids = packed.input_ids[0].tolist()
    if positions is not None:
        ids = [ids[i] for i in positions]
    return bytes(t - 1 for t in ids).decode()


def test_independent_score_leaf_sees_only_its_own_level_and_full_lists_them():
    request = Request.from_dict(example())
    independent = pack_request(request, ByteTokenizer(), "tree", score_block="independent")
    full = pack_request(request, ByteTokenizer(), "tree", score_block="full")
    default = pack_request(request, ByteTokenizer(), "tree")
    assert torch.equal(default.input_ids, independent.input_ids)  # the option's default is the Phase 1 layout
    score = independent.branches[2]
    allowed = independent.allowed
    for index, (start, end) in enumerate(score.leaves):
        visible = text_of(independent, [i for i in range(independent.token_count) if allowed[end - 1, i]])
        own = request.questions[2].options[index].description
        assert f"Level: {own}\nDecision:\n" in visible and "A red card." in visible and "Intensity?" in visible
        for other in ("low", "medium", "high"):
            assert (other in visible) == (other == own)
        assert not any(f"[{k}]" in visible for k in range(3)) and "Levels:" not in visible
        for other_start, other_end in score.leaves:
            if other_start != start:
                assert not allowed[start:end, other_start:other_end].any()
    assert "Levels:\n[0] low\n[1] medium\n[2] high\n" in text_of(full)[full.branches[2].start:full.branches[2].end]
    assert full.token_count > independent.token_count
    # Choice and Noul packing are byte-identical under both settings, including positions and segments.
    others = Request.from_dict({**example(), "questions": {k: v for k, v in example()["questions"].items() if k != "intensity"}})
    a = pack_request(others, ByteTokenizer(), "tree", score_block="independent")
    b = pack_request(others, ByteTokenizer(), "tree", score_block="full")
    assert torch.equal(a.input_ids, b.input_ids) and torch.equal(a.position_ids, b.position_ids)
    assert torch.equal(a.segment_ids, b.segment_ids) and torch.equal(a.parents, b.parents)
    assert torch.equal(full.input_ids[0, :a.token_count], a.input_ids[0])  # the Choice/Noul prefix is unchanged too
    with pytest.raises(ValueError, match="score_block"):
        pack_request(request, ByteTokenizer(), "tree", score_block="none")


def test_score_block_option_flows_through_the_model_and_checkpoints(tmp_path):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    request = Request.from_dict(example())
    for block in ("independent", "full"):
        model = DecisionModel(ModelConfig(backbone="tiny", mode="tree", adaptation="full", hidden_size=32, layers=1,
                                          head_rank=8, score_block=block)).eval()
        assert model.packing_kwargs["score_block"] == block
        packed = pack_request(request, model.tokenizer, model.packing_mode, **model.packing_kwargs)
        logits = [z.detach() for z in model(packed)]
        assert [len(z) for z in logits] == [3, 2, 3]
        answer = decode(request, [z.softmax(-1).tolist() for z in logits])["intensity"]
        assert set(answer["probabilities"]) == {"0", "1", "2"} and 0 <= answer["score"] <= 2
    with pytest.raises(ValueError, match="score_block"):
        DecisionModel(ModelConfig(backbone="tiny", mode="tree", score_block="none"))
    from janus.data import synthetic_requests
    from janus.training import TrainConfig, load_checkpoint, train
    for name, seed in (("train", 17), ("dev", 23)):
        write_jsonl(tmp_path / f"{name}.jsonl", [r.to_dict() for r in synthetic_requests(4, seed=seed)])
    config = ModelConfig(backbone="tiny", mode="tree", adaptation="full", hidden_size=32, layers=1, head_rank=8, score_block="full")
    train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / "run",
          TrainConfig(model=config, epochs=1, accumulation=2, max_steps=2, device="cpu"))
    loaded, metadata = load_checkpoint(tmp_path / "run" / "best.pt")
    assert metadata["model"]["score_block"] == "full" and loaded.packing_kwargs["score_block"] == "full"


def test_scale_worlds_put_the_target_on_the_level_the_record_implies():
    from janus.synth.cardinality import SCALES, scale_world
    seen = {levels: set() for levels in SCALES}
    for seed in range(300):
        rng = random.Random(seed)
        levels = rng.choice(tuple(SCALES))
        world, request = scale_world(rng, levels, "prose" if seed % 2 else "json")
        q = request.questions[0]
        assert q.kind == "score" and len(q.options) == levels and request.group_id.startswith(SCALES[levels][0])
        gold = q.target.index(1.)
        if levels == 2:
            expected = int(world["delivered_day"] <= world["promised_day"])
        elif levels == 7:
            d = world["defective_units"]
            expected = 0 if d > 100 else 1 if d > 50 else 2 if d > 20 else 3 if d > 10 else 4 if d > 5 else 5 if d > 1 else 6
        else:
            h = world["hours_until_deadline"]
            expected = 9 if h <= 0 else 8 if h < 1 else 7 if h < 3 else 6 if h < 6 else 5 if h < 12 else 4 if h < 24 else \
                3 if h < 48 else 2 if h < 72 else 1 if h < 168 else 0
            assert h * 10 != int(h * 10) or h not in (1, 3, 6, 12, 24, 48, 72, 168)  # never on a boundary
        assert gold == expected
        assert str(world[next(k for k in world if k.endswith("_id"))]) in request.state
        seen[levels].add(gold)
    assert all(seen[levels] == set(range(levels)) for levels in SCALES)  # every level is drawn


def test_relative_menu_with_fixed_k_keeps_names_distinct_and_gold_correct():
    from janus.synth.cardinality import name_pool
    from janus.synth.worlds import TEST_NAMES, relative_menu_world
    pool = name_pool(TEST_NAMES)
    assert len(pool) == len(set(pool)) >= 128
    for seed, k in enumerate((2, 4, 128, 41, 42)):
        world, request = relative_menu_world(random.Random(seed), pool, "compact", k=k)
        q = request.questions[0]
        assert len(q.options) == k and len({o.description for o in q.options}) == k
        assert all(o.description.startswith(item["name"] + " $") for o, item in zip(q.options, world["items"]))
        assert len({i["rating"] for i in world["items"]}) == k
        if world["kind"] == "second_cheapest":
            assert world["gold"] == sorted(world["items"], key=lambda i: i["price"])[1]["name"]
        assert world["kind"] != "median_rating" or k % 2 == 1
    sizes = {len(relative_menu_world(random.Random(s), TEST_NAMES, "prose")[1].questions[0].options) for s in range(40)}
    assert sizes <= {4, 5, 6, 7, 8}  # the Phase 1 draw is untouched


def test_prepare_cardinality_writes_cells_manifest_and_enforces_the_budget(tmp_path):
    from janus.synth.cardinality import prepare_cardinality
    manifest = prepare_cardinality(tmp_path / "card", ByteTokenizer(), per_cell=3, train=8, dev=4, calibration=4,
                                   choice_k=(2, 4), score_levels=(2, 7), max_tokens=10 ** 6)
    assert manifest["test_cells"] == {"choice_k2": 3, "choice_k4": 3, "score_l2": 3, "score_l7": 3}
    assert manifest["counts"]["train"] == {"choice": 8, "score": 4}
    rows = [json.loads(l) for l in (tmp_path / "card" / "test.jsonl").read_text().splitlines()]
    assert [r["cell"] for r in rows] == ["choice_k2"] * 3 + ["choice_k4"] * 3 + ["score_l2"] * 3 + ["score_l7"] * 3
    assert all(r["tier"] == "T0" and manifest["tokens"]["test"][r["cell"]]["max"] > 0 for r in rows)
    assert set(manifest["files"]) == {"train.jsonl", "dev.jsonl", "calibration.jsonl", "test.jsonl"}
    with pytest.raises(ValueError, match="over 100"):
        prepare_cardinality(tmp_path / "tight", ByteTokenizer(), per_cell=1, train=1, dev=1, calibration=1,
                            choice_k=(2,), score_levels=(2,), max_tokens=100)


def test_cardinality_report_tabulates_per_cell(tmp_path):
    from janus.phase4_cardinality_report import cells, report

    def rows(right):
        out = []
        for k in (2, 4):
            for i in range(3):
                p = [.1 / (k - 1)] * k
                p[0 if right else 1] = .9
                out.append({"kind": "choice", "cardinality": k, "probabilities": p, "target": [1.] + [0.] * (k - 1),
                            "nll": .2, "calibrated_nll": .3})
                out.append({"kind": "noul", "cardinality": 2, "probabilities": [.5, .5], "target": [0., 1.], "nll": .7})
        out.append({"kind": "score", "cardinality": 3, "probabilities": [.5, .5, 0.], "target": [0., 1., 0.], "nll": .69,
                    "calibrated_nll": .69})
        return out
    root = tmp_path / "runs"
    write_jsonl(root / "arm_a" / "cardinality" / "predictions.jsonl", rows(True))
    write_jsonl(root / "arm_b" / "cardinality" / "predictions.jsonl", rows(False))
    (root / "arm_a" / "phase1").mkdir()
    (root / "arm_a" / "phase1" / "metrics.json").write_text(json.dumps({"raw": {"accuracy": .5, "nll": 1.}}))
    table = cells(root / "arm_a" / "cardinality" / "predictions.jsonl")
    assert set(table) == {("choice", 2), ("choice", 4), ("score", 3)}
    assert table[("choice", 4)] == {"count": 3, "correct": 3., "nll": pytest.approx(.6), "calibrated_nll": pytest.approx(.9), "level_error": 0.}
    assert table[("score", 3)]["level_error"] == pytest.approx(.5)  # expected level 0.5 against gold 1
    text = report(root, arms=("arm_a", "arm_b"), reference=("missing",))
    assert "| 4 | 1.386 | 1.000 / 0.200 / 0.300 | 0.000 / 0.200 / 0.300 |" in text
    assert "| 3 | 1.099 | 0.000 / 0.690 / 0.690 / 0.50 |" in text
    assert "| arm_a | 0.5000 | 1.0000 | n/a | n/a |" in text and "missing" not in text.split("## Choice")[1].split("\n")[2]
