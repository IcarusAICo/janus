"""Production mix (janus.synth.production) and per-task calibration (janus.calibration_sets)."""

import json
import math
import random

import pytest
import torch

from janus.data import synthetic_requests, write_jsonl
from janus.packing import ByteTokenizer
from janus.schema import Request


def _synthetic(n, seed, **extra):
    return [dict(r.to_dict(), **extra) for r in synthetic_requests(n, seed=seed)]


def _split_dir(root, rows, dev=2, calibration=2):
    root.mkdir(parents=True)
    write_jsonl(root / "train.jsonl", rows[:-dev - calibration])
    write_jsonl(root / "dev.jsonl", rows[-dev - calibration:-calibration])
    write_jsonl(root / "calibration.jsonl", rows[-calibration:])


def _jevlike_dir(root, prefix, train=12, validation=6, test=4):
    root.mkdir(parents=True)
    counter = 0
    for split, n in (("train", train), ("validation", validation), ("test", test)):
        rows = []
        for _ in range(n):
            options = [f"{prefix} link {counter + i}" for i in range(2 + counter % 4)]
            rows.append({"context": f"Target article: T{counter}\nCurrent article: {prefix} {counter}\nbody text", "options": options,
                         "label": counter % len(options)})
            counter += 1
        write_jsonl(root / f"{split}.jsonl", rows)


def test_jevlike_conversion_matches_the_benchmark_rendering(tmp_path):
    from demos.bench.tasks import load_jevlike_jsonl
    from janus.synth.production import load_jevlike
    _jevlike_dir(tmp_path / "w", "wiki")
    ours = [Request.from_dict(r) for r in load_jevlike(tmp_path / "w" / "test.jsonl", "wikispeedia", "T1")]
    theirs = load_jevlike_jsonl(tmp_path / "w" / "test.jsonl", prefix="wikispeedia")
    assert len(ours) == len(theirs) == 4
    for a, b in zip(ours, theirs):
        assert a.state == b.state and a.to_dict()["questions"] == b.questions and a.group_id.startswith("wikispeedia:test:")


def test_allocate_water_fills_under_the_cap():
    from janus.synth.production import allocate
    counts = {"badges": 2000, "cardinality": 2677, "phase1": 12000, "public": 25311, "families": 9000, "t3": 16526, "wikispeedia": 41291}
    allocation = allocate(counts, 40000, 8000)
    assert allocation["badges"] == 2000 and allocation["cardinality"] == 2677
    assert len({allocation[s] for s in ("phase1", "public", "families", "t3", "wikispeedia")}) == 1
    assert max(allocation.values()) <= 8000 and 39990 <= sum(allocation.values()) <= 40000
    assert allocate({"a": 100, "b": 100000}, 1000, 200) == {"a": 100, "b": 200}


def test_prepare_production_caps_sources_and_keeps_evaluation_files_disjoint(tmp_path):
    from janus.synth.production import prepare_production
    phase1 = _synthetic(14, 1, tier="T0", family="rel")
    public = _synthetic(14, 3, tier="T1", family="massive", source="AmazonScience/massive")
    families = _synthetic(8, 5, tier="T0", family="retrieval")
    cardinality = _synthetic(8, 7, tier="T0", family="quality")
    sources = {"phase1": tmp_path / "p1", "public": tmp_path / "pub", "families": tmp_path / "fam", "cardinality": tmp_path / "card"}
    for name, rows in zip(sources, (phase1, public, families, cardinality)):
        _split_dir(sources[name], rows)
    (tmp_path / "t3").mkdir()
    write_jsonl(tmp_path / "t3" / "routing_text.jsonl", _synthetic(6, 8, tier="T3", family="routing", cell="routing_text", checks={"outcome": "T3"}))
    _jevlike_dir(tmp_path / "wiki", "wiki", train=20)
    _jevlike_dir(tmp_path / "badges", "badge", train=6)
    # A training state coinciding with an evaluation state (in either file format) is dropped and recorded.
    leaked_wiki = json.loads((tmp_path / "wiki" / "train.jsonl").read_text().splitlines()[0])
    write_jsonl(tmp_path / "wiki_eval.jsonl", [leaked_wiki] + [{"context": "elsewhere", "options": ["a", "b"], "label": 1}])
    write_jsonl(tmp_path / "eval.jsonl", [dict(phase1[0], group_id="other")] + _synthetic(3, 11))
    manifest = prepare_production(tmp_path / "out", ByteTokenizer(), sources=sources, t3=(tmp_path / "t3",), wikispeedia=tmp_path / "wiki",
                                  badges=tmp_path / "badges", evaluation_files=(str(tmp_path / "eval.jsonl"), str(tmp_path / "wiki_eval.jsonl")),
                                  size=40, cap=.15, dev_per_source=2, calibration_per_source=3, max_tokens=10 ** 6)
    rows = {split: [json.loads(l) for l in (tmp_path / "out" / f"{split}.jsonl").read_text().splitlines()] for split in ("train", "dev", "calibration")}
    # 7 sources, cap 6: families and cardinality (4 each) take all of theirs, the other five get 6 each.
    assert manifest["counts"]["train"] == len(rows["train"]) == 38 and max(manifest["sources"]["train"].values()) == 6
    assert {d["group_id"] for d in manifest["dropped_for_evaluation_overlap"]} == {phase1[0]["group_id"], "wikispeedia:train:0"}
    assert manifest["sources"]["train"]["wikispeedia"] == 6 and manifest["sources"]["train"]["badges"] == 6
    assert manifest["sources"]["dev"] == {"badges": 2, "cardinality": 2, "families": 2, "phase1": 2, "public": 2, "wikispeedia": 2}
    assert manifest["sources"]["calibration"]["wikispeedia"] == 3 and "t3" not in manifest["sources"]["dev"]
    wiki_states = {r["state"] for r in rows["train"] + rows["dev"] + rows["calibration"] if r["pool"] == "wikispeedia"}
    test_states = {json.loads(l)["context"] for l in (tmp_path / "wiki" / "test.jsonl").read_text().splitlines()}
    assert not wiki_states & test_states and manifest["leakage_check"]["overlap"] == 0
    assert all(Request.from_dict(r).questions[0].kind == "choice" for r in rows["train"] if r["pool"] in ("wikispeedia", "badges"))
    assert manifest["pool"]["public_train_rows_outside_breadth_pool"] == 0
    with pytest.raises(FileExistsError):
        prepare_production(tmp_path / "out", ByteTokenizer(), sources=sources, t3=(), evaluation_files=())


def test_bucket_and_temperature_fallbacks():
    from janus.calibration_sets import bucket, temperature_for
    assert [bucket(k) for k in (2, 3, 4, 5, 8, 9, 16, 17, 64, 200)] == ["2", "4", "4", "8", "8", "16", "16", "32", "64", "256"]
    old = {"temperature": 1.5}
    new = {"temperature": 1.5, "by_family": {"wikispeedia": {"temperature": 0.8, "by_cardinality": {"32": 0.6}}}}
    assert temperature_for(old, "wikispeedia", 20) == 1.5
    assert temperature_for(new, "wikispeedia", 20) == 0.6 and temperature_for(new, "wikispeedia", 3) == 0.8
    assert temperature_for(new, "massive", 3) == 1.5


def test_fit_temperature_by_family_needs_minimum_rows_per_cell():
    from janus.calibration_sets import fit_temperature_by_family
    rng = torch.Generator().manual_seed(0)
    logits, targets, families = [], [], []
    for family, k, n in (("a", 4, 30), ("a", 7, 30), ("a", 2, 5), ("b", 4, 10)):
        for _ in range(n):
            z = torch.randn(k, generator=rng) * 3
            y = torch.zeros(k); y[int(z.argmax())] = 1.  # overconfident logits: the fit wants T > 1
            logits.append(z); targets.append(y); families.append(family)
    fitted = fit_temperature_by_family(logits, targets, families, minimum=20)
    assert set(fitted) == {"a"} and set(fitted["a"]["by_cardinality"]) == {"4", "8"}
    assert all(math.isfinite(t) and t > 0 for t in [fitted["a"]["temperature"], *fitted["a"]["by_cardinality"].values()])
    with pytest.raises(ValueError):
        fit_temperature_by_family(logits, targets, families[:-1])


def test_calibration_file_carries_by_family_and_old_files_still_load(tmp_path):
    from janus.evaluation import calibrate, evaluate, predict, read_calibration
    from janus.model import DecisionModel, ModelConfig
    from janus.training import checkpoint
    torch.set_num_threads(2)
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, layers=1, head_rank=8))
    checkpoint(model, tmp_path / "m.pt", {"step": 0, "training_group_ids": [], "selection_group_ids": [],
                                          "training_state_hashes": [], "selection_state_hashes": []})
    rows = [dict(r.to_dict(), group_id=f"fam{i % 2}:{r.group_id}") for i, r in enumerate(synthetic_requests(16, seed=2))]
    write_jsonl(tmp_path / "calibration.jsonl", rows)
    write_jsonl(tmp_path / "test.jsonl", [dict(r.to_dict(), group_id=f"fam0:{r.group_id}") for r in synthetic_requests(4, seed=3)])
    fitted = calibrate(tmp_path / "m.pt", tmp_path / "calibration.jsonl", tmp_path / "new.json")
    assert set(fitted["by_family"]) == {"fam0", "fam1"} and set(fitted["by_family"]["fam0"]) == {"temperature", "by_cardinality"}
    assert set(fitted["by_family"]["fam0"]["by_cardinality"]) <= {"2", "4"}  # 24 questions per family, 8 per cardinality
    new = read_calibration(tmp_path / "new.json", tmp_path / "m.pt")
    assert isinstance(new["temperature"], float)  # the only entry the server reads
    old = {k: v for k, v in json.loads((tmp_path / "new.json").read_text()).items() if k != "by_family"}
    (tmp_path / "old.json").write_text(json.dumps(old))
    assert "by_family" not in read_calibration(tmp_path / "old.json", tmp_path / "m.pt")
    assert predict(tmp_path / "m.pt", synthetic_requests(1, seed=4)[0], calibration=tmp_path / "new.json")
    bad = dict(json.loads((tmp_path / "new.json").read_text()))
    bad["by_family"]["fam0"]["temperature"] = 0.
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Per-family"):
        read_calibration(tmp_path / "bad.json", tmp_path / "m.pt")
    report = evaluate(tmp_path / "m.pt", tmp_path / "test.jsonl", tmp_path / "eval", tmp_path / "new.json")
    assert math.isfinite(report["calibrated_by_family"]["nll"]) and "calibrated_by_family" in report["by_family"]["fam0"]
    first = json.loads((tmp_path / "eval" / "predictions.jsonl").read_text().splitlines()[0])
    assert math.isfinite(first["calibrated_by_family_nll"])
    plain = evaluate(tmp_path / "m.pt", tmp_path / "test.jsonl", tmp_path / "plain", tmp_path / "old.json")
    assert "calibrated_by_family" not in plain and "calibrated_by_cardinality" in plain


def test_extra_sources_enter_as_their_own_pool_with_dev_and_calibration(tmp_path):
    from janus.synth.production import prepare_production
    sources = {"phase1": tmp_path / "p1", "public": tmp_path / "pub"}
    _split_dir(sources["phase1"], _synthetic(14, 1, tier="T0", family="rel"))
    _split_dir(sources["public"], _synthetic(14, 3, tier="T1", family="massive"))
    hard = _synthetic(12, 21, tier="T2", family="long_policy", pool="whatever", rationale="dropped by KEEP")
    _split_dir(tmp_path / "hard", hard, dev=2, calibration=2)
    (tmp_path / "long").mkdir()
    write_jsonl(tmp_path / "long" / "train.jsonl", _synthetic(5, 31, tier="T0", family="rel"))
    write_jsonl(tmp_path / "long" / "dev.jsonl", _synthetic(2, 32, tier="T0", family="rel"))  # no calibration split
    manifest = prepare_production(tmp_path / "out", ByteTokenizer(), sources=sources, t3=(), wikispeedia=tmp_path / "none",
                                  badges=tmp_path / "none", evaluation_files=(), size=100, cap=.5, dev_per_source=2,
                                  calibration_per_source=2, max_tokens=10 ** 6,
                                  extra={"hardtier": tmp_path / "hard", "longcontext": tmp_path / "long"}, drop=[("longcontext", "rel")])
    assert manifest["sources"]["train"] == {"hardtier": 8, "phase1": 10, "public": 10}  # longcontext train rows (family rel) dropped
    assert manifest["sources"]["dev"] == {"hardtier": 2, "longcontext": 2, "phase1": 2, "public": 2}  # drop touches train only
    assert manifest["sources"]["calibration"] == {"hardtier": 2, "phase1": 2, "public": 2}
    rows = [json.loads(l) for l in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert all(r["pool"] == r["source"] == "hardtier" and "rationale" not in r for r in rows if r["family"] == "long_policy")
