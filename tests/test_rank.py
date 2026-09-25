import json
import random

import torch

from janus.data import load_requests, synthetic_requests, write_jsonl
from janus.schema import Request


def _gold_description(question):
    return question.options[max(range(len(question.target)), key=question.target.__getitem__)].description


def _name(description):
    try:
        return json.loads(description)["name"]
    except ValueError:
        return description.split(":")[0]


def test_rank_world_gold_by_brute_force_per_kind():
    from janus.synth.rankworlds import RANK_KINDS, TRAIN_NAMES, rank_world
    seen_kinds = set()
    for seed in range(300):
        rng = random.Random(seed)
        style = ("json", "prose")[seed % 2]
        world, request = rank_world(rng, TRAIN_NAMES, style)
        items, attribute, kind = world["items"], world["attribute"], world["kind"]
        seen_kinds.add(kind)
        by_name = {i["name"]: i for i in items}
        # Every attribute value is unique, so every rank position is well defined.
        assert all(len({i[a] for i in items}) == len(items) for a in ("price", "rating", "distance_km"))
        beats = (lambda a, b: a[attribute] > b[attribute]) if attribute == "rating" else (lambda a, b: a[attribute] < b[attribute])
        ranked = sorted(items, key=lambda i: -i[attribute] if attribute == "rating" else i[attribute])
        if kind == "pair":
            a, b = by_name[world["reference"]], by_name[world["other"]]
            assert world["gold"] == beats(a, b)
            (q,) = request.questions
            assert q.id == "rank:pair" and q.kind == "noul" and q.target[1] == float(world["gold"])
            assert all(i["name"] in q.instructions and str(i["price"]) in q.instructions for i in items)
        elif kind == "count":
            ref = by_name[world["reference"]]
            assert world["gold"] == sum(beats(i, ref) for i in items)
            (q,) = request.questions
            assert q.id == "rank:count" and q.kind == "choice" and len(q.options) == len(items)
            assert _gold_description(q) == str(world["gold"])
            assert all(i["name"] in q.instructions for i in items)
        else:
            if kind == "kth":
                assert 2 <= world["position"] <= len(items) - 1
                expected = ranked[world["position"] - 1]["name"]
            elif kind == "closest":
                ref = by_name[world["reference"]]
                others = [i for i in items if i["name"] != world["reference"]]
                gaps = sorted(abs(i[attribute] - ref[attribute]) for i in others)
                assert len(gaps) == 1 or gaps[0] != gaps[1]
                expected = min(others, key=lambda i: abs(i[attribute] - ref[attribute]))["name"]
            else:
                assert len(items) % 2 == 1
                expected = sorted(items, key=lambda i: i[attribute])[len(items) // 2]["name"]
            assert world["gold"] == expected
            pick, probe = request.questions
            assert pick.id == "rank:pick" and pick.kind == "choice" and len(pick.options) == len(items)
            assert _name(_gold_description(pick)) == expected
            assert probe.id == "rank:is_gold" and probe.kind == "noul"
        # The state carries the request and a nuisance id only: no attribute values, no non-reference names.
        named = {world["reference"], world["other"]} - {None}
        assert all(i["name"] not in request.state for i in items if i["name"] not in named)
        assert all(str(i["price"]) not in request.state.replace(str(world["order_id"]), "") for i in items)
        assert request.group_id.startswith("rank:")
    assert seen_kinds == set(RANK_KINDS)


def test_rank_world_is_deterministic():
    from janus.synth.rankworlds import TRAIN_NAMES, rank_world
    a = rank_world(random.Random(5), TRAIN_NAMES, "prose")
    b = rank_world(random.Random(5), TRAIN_NAMES, "prose")
    assert a[0] == b[0] and a[1] == b[1]


def test_request_kind_parses_every_rel_and_rank_rendering():
    from janus.synth.rankworlds import TRAIN_NAMES, rank_world, request_kind
    from janus.synth.worlds import relative_menu_world
    for seed in range(120):
        style = ("json", "prose")[seed % 2]
        world, request = rank_world(random.Random(seed), TRAIN_NAMES, style)
        assert request_kind(request.state, "rank") == f"rank:{world['kind']}"
        assert request_kind(request.state) is not None
        world, request = relative_menu_world(random.Random(seed), TRAIN_NAMES, style)
        assert request_kind(request.state, "rel") == f"rel:{world['kind']}"
        assert request_kind(request.state) == f"rel:{world['kind']}"
    assert request_kind("Record 17-0. Color: red. Intensity: low.") is None
    assert request_kind("Record 17-0. Color: red.", "study") is None


def test_prepare_rank_writes_disjoint_splits_with_name_holdout_and_excludes_given_states(tmp_path):
    from janus.data import assert_disjoint
    from janus.synth.rankworlds import prepare_rank
    from janus.synth.worlds import TEST_NAMES, TRAIN_NAMES
    # An "exclude" directory whose states must be redrawn rather than reused.
    other = tmp_path / "phase1"
    other.mkdir()
    write_jsonl(other / "train.jsonl", [r.to_dict() for r in synthetic_requests(3, seed=1)])
    output = tmp_path / "rank"
    manifest = prepare_rank(output, seed=3, train=40, dev=6, calibration=6, test=12, exclude=(other, tmp_path / "missing"))
    splits = {name: load_requests(output / f"{name}.jsonl") for name in ("train", "dev", "calibration", "test")}
    assert_disjoint({**splits, "excluded": load_requests(other / "train.jsonl")})
    assert manifest["dataset"] == "JEV_RANK_V1" and manifest["tier"] == "T0"
    assert manifest["total"] == {"train": 40, "dev": 6, "calibration": 6, "test": 12}
    assert sum(manifest["counts"]["train"].values()) == 40 and manifest["excluded"] == [str(other)]
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert all(r["tier"] == "T0" and r["family"] == "rank" and r["kind"] in manifest["kinds"] for r in rows)
    assert {r["style"] for r in rows} == {"json", "prose"}

    def names(request):
        out = set()
        for q in request.questions:
            if q.id in ("rank:pick",):
                out |= {_name(o.description) for o in q.options}
            elif q.id in ("rank:pair", "rank:count"):
                out |= {n for n in TEST_NAMES + TRAIN_NAMES if n in q.instructions}
        return out

    for name, requests in splits.items():
        pool = TEST_NAMES if name == "test" else TRAIN_NAMES
        for request in requests:
            found = names(request)
            assert found and found <= set(pool)


def test_prepare_phase1_rank_concatenates_and_checks_disjointness(tmp_path):
    from janus.synth.rankworlds import prepare_phase1_rank, prepare_rank
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    for name, seed, count in (("train", 1, 8), ("dev", 2, 3), ("calibration", 3, 3), ("test", 4, 4)):
        write_jsonl(phase1 / f"{name}.jsonl", [dict(r.to_dict(), tier="T0", family="rel") for r in synthetic_requests(count, seed=seed)])
    rank = tmp_path / "rank"
    prepare_rank(rank, seed=5, train=10, dev=2, calibration=2, test=5, exclude=(phase1,))
    out = tmp_path / "mix"
    manifest = prepare_phase1_rank(out, phase1=phase1, rank=rank, evaluation_files=(phase1 / "test.jsonl", rank / "test.jsonl"))
    assert manifest["dataset"] == "JEV_PHASE1_RANK_V1"
    assert manifest["counts"] == {"train": 18, "dev": 5, "calibration": 5}
    assert manifest["families"]["train"] == {"rank": 10, "rel": 8}
    assert set(manifest["checked_against"]) == {str(phase1 / "test.jsonl"), str(rank / "test.jsonl")}
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert [r["family"] for r in rows] == ["rel"] * 8 + ["rank"] * 10
    # A leaked test row is caught.
    leaked = tmp_path / "leaked"
    leaked.mkdir()
    for name in ("train", "dev", "calibration"):
        (leaked / f"{name}.jsonl").write_text((rank / f"{name}.jsonl").read_text())
    (leaked / "train.jsonl").write_text((rank / "train.jsonl").read_text() + (rank / "test.jsonl").read_text())
    import pytest
    with pytest.raises(ValueError, match="overlap"):
        prepare_phase1_rank(tmp_path / "bad", phase1=phase1, rank=leaked, evaluation_files=(rank / "test.jsonl",))


def test_evaluate_reports_request_kind_and_by_kind_rel(tmp_path):
    from janus.evaluation import evaluate
    from janus.model import DecisionModel, ModelConfig
    from janus.synth.rankworlds import TRAIN_NAMES, rank_world
    from janus.synth.worlds import relative_menu_world
    from janus.training import checkpoint
    torch.set_num_threads(2)
    model = DecisionModel(ModelConfig(backbone="tiny", mode="pair", adaptation="full", hidden_size=32, layers=1, pair_width=16))
    checkpoint(model, tmp_path / "m.pt", {"step": 0, "training_group_ids": [], "selection_group_ids": [],
                                          "training_state_hashes": [], "selection_state_hashes": []})
    rows, expected = [], {}
    for seed in range(6):
        world, request = rank_world(random.Random(seed), TRAIN_NAMES, "prose")
        rows.append(request.to_dict())
        expected[request.group_id] = f"rank:{world['kind']}"
        world, request = relative_menu_world(random.Random(seed), TRAIN_NAMES, "json")
        rows.append(request.to_dict())
        expected[request.group_id] = f"rel:{world['kind']}"
    write_jsonl(tmp_path / "test.jsonl", rows)
    report = evaluate(tmp_path / "m.pt", tmp_path / "test.jsonl", tmp_path / "out")
    records = [json.loads(line) for line in (tmp_path / "out" / "predictions.jsonl").read_text().splitlines()]
    assert all(r["request_kind"] == expected[r["group_id"]] for r in records)
    assert all(r["kind"] in {"choice", "noul"} for r in records)  # the question kind key is unchanged
    assert set(report["by_kind_rel"]) == set(expected.values())
    assert sum(v["raw"]["count"] for v in report["by_kind_rel"].values()) == len(records)
    assert set(report) >= {"raw", "calibrated", "uniform", "shuffled_state", "by_kind", "by_family", "by_question",
                           "by_cardinality", "by_kind_rel", "choice_order", "uniform_comparison", "shuffled_state_comparison"}
    write_jsonl(tmp_path / "plain.jsonl", [r.to_dict() for r in synthetic_requests(3, seed=9)])
    plain = evaluate(tmp_path / "m.pt", tmp_path / "plain.jsonl", tmp_path / "plain")
    assert plain["by_kind_rel"] == {}
    first = json.loads((tmp_path / "plain" / "predictions.jsonl").read_text().splitlines()[0])
    assert first["request_kind"] is None


def test_rank_report_tabulates_accuracy_by_kind_from_predictions(tmp_path):
    from janus.phase3_rank_report import accuracy_by_kind, cheapest_share, report
    from janus.synth.rankworlds import TRAIN_NAMES, rank_world
    from janus.synth.worlds import relative_menu_world
    rank_rows, rel_rows = [], []
    for seed in range(8):
        rank_rows.append(rank_world(random.Random(seed), TRAIN_NAMES, "json")[1].to_dict())
        rel_rows.append(relative_menu_world(random.Random(100 + seed), TRAIN_NAMES, "prose")[1].to_dict())
    data = tmp_path / "data"
    write_jsonl(data / "phase1_test.jsonl", rel_rows)
    write_jsonl(data / "rank_test.jsonl", rank_rows)

    def predictions(rows, right, with_kind):
        out = []
        for raw in rows:
            request = Request.from_dict(raw)
            for q in request.questions:
                target = list(q.target)
                gold = target.index(1.)
                pick = gold if right else (gold + 1) % len(target)
                p = [.1 / (len(target) - 1)] * len(target)
                p[pick] = .9
                row = {"group_id": request.group_id, "question_id": q.id, "kind": q.kind, "family": request.group_id.split(":")[0],
                       "keys": [o.key for o in q.options], "target": target, "probabilities": p, "nll": .1, "input_sha256": q.id}
                if with_kind:
                    from janus.synth.rankworlds import request_kind
                    row["request_kind"] = request_kind(request.state, row["family"])
                out.append(row)
        return out
    root = tmp_path / "runs"
    write_jsonl(root / "arm_a" / "phase1" / "predictions.jsonl", predictions(rel_rows, True, True))
    write_jsonl(root / "arm_a" / "rank" / "predictions.jsonl", predictions(rank_rows, True, True))
    write_jsonl(root / "old" / "test" / "predictions.jsonl", predictions(rel_rows, False, False))
    by_kind = accuracy_by_kind(root / "arm_a" / "phase1" / "predictions.jsonl", data / "phase1_test.jsonl", ("rel:pick",))
    assert all(v["correct"] == v["count"] for v in by_kind.values()) and sum(v["count"] for v in by_kind.values()) == 8
    old = accuracy_by_kind(root / "old" / "test" / "predictions.jsonl", data / "phase1_test.jsonl", ("rel:pick",))
    assert set(old) == set(by_kind) and all(v["correct"] == 0 for v in old.values())
    share = cheapest_share(root / "old" / "test" / "predictions.jsonl", data / "phase1_test.jsonl")
    assert share is None or 0 <= share <= 1
    text = report(root, arms=("arm_a",), reference={"old": root / "old"}, phase1_test=data / "phase1_test.jsonl",
                  rank_test=data / "rank_test.jsonl")
    assert "| arm_a |" in text and "| old |" in text and "rel:pick accuracy by kind" in text and "rank accuracy by kind" in text
