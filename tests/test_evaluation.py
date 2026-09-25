from dataclasses import asdict
import json

import pytest
import torch

from janus.data import synthetic_requests, write_jsonl, state_hash
from janus.evaluation import calibrate, evaluate, predict, compare_runs
from janus.model import DecisionModel, ModelConfig
from janus.training import checkpoint


def test_calibration_and_evaluation_enforce_disjointness_and_model_identity(tmp_path):
    torch.set_num_threads(2)
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, head_rank=16))
    training = synthetic_requests(1, seed=1)
    path = tmp_path / "model.pt"
    checkpoint(model, path, {"training_group_ids": [training[0].group_id],
                            "training_state_hashes": [state_hash(training[0].state)]})
    train_file = tmp_path / "train.jsonl"
    cal_file = tmp_path / "calibration.jsonl"
    test_file = tmp_path / "test.jsonl"
    write_jsonl(train_file, [r.to_dict() for r in training])
    write_jsonl(cal_file, [r.to_dict() for r in synthetic_requests(3, seed=2)])
    write_jsonl(test_file, [r.to_dict() for r in synthetic_requests(3, seed=3)])
    with pytest.raises(ValueError, match="overlap"):
        calibrate(path, train_file, tmp_path / "bad.json")
    calibration = tmp_path / "temperature.json"
    fitted = calibrate(path, cal_file, calibration)
    assert fitted["temperature"] > 0
    with pytest.raises(ValueError, match="overlap"):
        evaluate(path, cal_file, tmp_path / "bad_eval", calibration)
    report = evaluate(path, test_file, tmp_path / "eval", calibration)
    assert report["raw"]["count"] == 9
    assert set(report["by_kind"]) == {"choice", "noul", "score"}
    records = [json.loads(line) for line in (tmp_path / "eval" / "predictions.jsonl").read_text().splitlines()]
    assert len(records) == 9
    assert all(abs(sum(r["probabilities"]) - 1) < 1e-5 for r in records)
    assert report["uniform_comparison"]["groups"] == 3
    output = predict(path, synthetic_requests(1, seed=4)[0], calibration=calibration)
    assert set(output) == {"color", "red", "level"}
    with torch.no_grad():
        next(model.head.parameters()).add_(1)
    second = tmp_path / "different.pt"
    checkpoint(model, second, {})
    with pytest.raises(ValueError, match="checkpoint"):
        predict(second, synthetic_requests(1)[0], calibration=calibration)


def test_paired_comparison_checks_full_inputs_and_duplicate_rows(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    row = {"group_id": "same-state", "question_id": "intent", "keys": ["o0", "o1"],
           "target": [1, 0], "nll": 1., "input_sha256": "menu-a"}
    write_jsonl(a, [row])
    write_jsonl(b, [{**row, "input_sha256": "menu-b"}])
    with pytest.raises(ValueError, match="input"):
        compare_runs(a, b)
    write_jsonl(b, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        compare_runs(a, b)
    write_jsonl(b, [{**row, "nll": 2.}])
    assert compare_runs(a, b)["mean_improvement"] == 1.


def test_evaluate_reports_by_family_from_group_id_prefix(tmp_path):
    torch.set_num_threads(2)
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, layers=1, head_rank=8))
    checkpoint(model, tmp_path / "m.pt", {"step": 0, "training_group_ids": [], "selection_group_ids": [],
                                          "training_state_hashes": [], "selection_state_hashes": []})
    rows = []
    for i, r in enumerate(synthetic_requests(6, seed=5)):
        raw = r.to_dict()
        raw["group_id"] = f"{'fam_a' if i % 2 else 'fam_b'}:{i}"
        rows.append(raw)
    write_jsonl(tmp_path / "test.jsonl", rows)
    report = evaluate(tmp_path / "m.pt", tmp_path / "test.jsonl", tmp_path / "out")
    assert set(report["by_family"]) == {"fam_a", "fam_b"}
    assert report["by_family"]["fam_a"]["raw"]["count"] == 9
    first = json.loads((tmp_path / "out" / "predictions.jsonl").read_text().splitlines()[0])
    assert first["family"] in {"fam_a", "fam_b"}


def test_compare_accuracy_bootstraps_paired_correctness(tmp_path):
    from janus.evaluation import compare_accuracy
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    base = [{"group_id": "g", "question_id": f"q{i}", "keys": ["o0", "o1"], "target": [1, 0],
             "nll": 1., "input_sha256": f"sig{i}"} for i in range(3)]
    # A is right on all three questions; B is right on one.
    write_jsonl(a, [{**r, "probabilities": [.9, .1]} for r in base])
    write_jsonl(b, [{**base[0], "probabilities": [.6, .4]}, {**base[1], "probabilities": [.2, .8]},
                    {**base[2], "probabilities": [.4, .6]}])
    result = compare_accuracy(a, b)
    assert result["groups"] == 1
    assert abs(result["mean_improvement"] - 2 / 3) < 1e-9
    reverse = compare_accuracy(b, a)
    assert abs(reverse["mean_improvement"] + 2 / 3) < 1e-9
    write_jsonl(b, [{**r, "probabilities": [.9, .1], "input_sha256": "other"} for r in base])
    with pytest.raises(ValueError, match="input"):
        compare_accuracy(a, b)


def test_posthoc_calibrators_add_report_sections_with_finite_nll(tmp_path):
    import math
    torch.set_num_threads(2)
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation="full", hidden_size=32, layers=1, head_rank=8))
    checkpoint(model, tmp_path / "m.pt", {"step": 0, "training_group_ids": [], "selection_group_ids": [],
                                          "training_state_hashes": [], "selection_state_hashes": []})
    write_jsonl(tmp_path / "calibration.jsonl", [r.to_dict() for r in synthetic_requests(8, seed=2)])
    write_jsonl(tmp_path / "test.jsonl", [r.to_dict() for r in synthetic_requests(4, seed=3)])
    fitted = calibrate(tmp_path / "m.pt", tmp_path / "calibration.jsonl", tmp_path / "calibration.json")
    assert set(fitted["by_cardinality"]) == {"2", "3", "4"}
    assert len(fitted["histogram"]) == 10 and all(t is None or 0 <= t <= 1 for t in fitted["histogram"])
    written = json.loads((tmp_path / "calibration.json").read_text())
    assert "by_cardinality" in written and "histogram" in written
    report = evaluate(tmp_path / "m.pt", tmp_path / "test.jsonl", tmp_path / "eval", tmp_path / "calibration.json")
    for section in ("calibrated_by_cardinality", "histogram_binned"):
        assert math.isfinite(report[section]["nll"]) and report[section]["count"] == 12
        assert all(math.isfinite(v[section]["nll"]) for v in report["by_kind"].values())
    assert set(report) >= {"raw", "calibrated", "uniform", "shuffled_state", "by_kind", "by_family", "by_question",
                           "by_cardinality", "choice_order", "uniform_comparison", "shuffled_state_comparison"}
    first = json.loads((tmp_path / "eval" / "predictions.jsonl").read_text().splitlines()[0])
    assert math.isfinite(first["calibrated_by_cardinality_nll"]) and math.isfinite(first["histogram_binned_nll"])
    # Without a calibration file neither extra section appears and the old keys are unchanged.
    plain = evaluate(tmp_path / "m.pt", tmp_path / "test.jsonl", tmp_path / "plain")
    assert "calibrated_by_cardinality" not in plain and "histogram_binned" not in plain
