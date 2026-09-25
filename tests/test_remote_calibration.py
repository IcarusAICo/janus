"""Calibration tests use real fitting and artifacts, with no network or model mocks."""

from copy import deepcopy
import hashlib
import json
import math

import pytest
import torch

from janus.data import file_hash, write_jsonl


def write_fixture(directory, name, *, targets=(0, 0, 0, 1), model="jev-1.13.0"):
    requests, predictions = [], []
    for i, target in enumerate(targets):
        state = f"{name} observation {i}"
        group = f"{name}:{i}"
        y = [float(target == 0), float(target == 1)]
        options = [("a", "First"), ("b", "Second")]
        signature = {"state": state, "kind": "choice", "instructions": "Which?", "options": options}
        requests.append({"state": state, "group_id": group, "questions": {
            "domain:q": {"type": "choice", "instructions": "Which?", "criteria": dict(options), "target": y}}})
        p = [.99, .01]
        nll = -sum(value * math.log(probability) for value, probability in zip(y, p))
        predictions.append({"group_id": group, "question_id": "domain:q", "kind": "choice", "cardinality": 2,
            "keys": ["a", "b"], "target": y, "input_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "probabilities": p, "logits": [math.log(value) for value in p], "nll": nll, "calibrated_nll": nll,
            "requested_model": model, "returned_model": model})
    data_path = directory / f"{name}.jsonl"
    prediction_path = directory / f"{name}-predictions.jsonl"
    write_jsonl(data_path, requests)
    write_jsonl(prediction_path, predictions)
    return data_path, prediction_path, requests, predictions


def test_temperature_is_fitted_on_calibration_only_and_raw_artifacts_are_preserved(tmp_path):
    from janus.remote_calibration import calibrate_remote
    cal_data, cal_predictions, _, _ = write_fixture(tmp_path, "calibration")
    panel_data, panel_predictions, _, panel_rows = write_fixture(tmp_path, "panel", targets=(0, 1))
    input_hashes = {path: file_hash(path) for path in (cal_data, cal_predictions, panel_data, panel_predictions)}
    output = tmp_path / "calibrated.jsonl"
    artifact = tmp_path / "temperature.json"
    result = calibrate_remote(cal_data, cal_predictions, panel_data, panel_predictions, output, artifact)
    expected_temperature = math.log(99) / math.log(3)
    assert result["temperature"] == pytest.approx(expected_temperature, rel=2e-3)
    assert result["requests"] == 4
    assert result["questions"] == 4
    assert result["panel_requests"] == 2
    assert result["metric_split"] == "calibration"
    assert result["before"]["nll"] > 1.1
    assert result["after"]["nll"] == pytest.approx(-.75 * math.log(.75) - .25 * math.log(.25), abs=1e-5)
    assert result["after"]["nll"] < result["before"]["nll"]
    assert result["calibration_predictions_sha256"] == input_hashes[cal_predictions]
    assert result["panel_predictions_sha256"] == input_hashes[panel_predictions]
    assert result["calibration_data_sha256"] == input_hashes[cal_data]
    assert result["panel_data_sha256"] == input_hashes[panel_data]
    assert result["calibrated_predictions_sha256"] == file_hash(output)
    assert all(file_hash(path) == digest for path, digest in input_hashes.items())
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    for before, after in zip(panel_rows, rows):
        assert after["raw_probabilities"] == before["probabilities"]
        assert after["probabilities"] == pytest.approx([.75, .25], abs=2e-4)
        assert after["logits"] == before["logits"]
        assert after["nll"] == before["nll"]
        assert after["calibration_temperature"] == result["temperature"]
        expected_nll = -sum(p * math.log(q) for p, q in zip(after["target"], after["probabilities"]))
        assert after["calibrated_nll"] == pytest.approx(expected_nll, abs=1e-6)
    assert json.loads(artifact.read_text()) == result
    changed_data, changed_predictions, _, _ = write_fixture(tmp_path, "different-panel", targets=(1, 1, 1))
    changed = calibrate_remote(cal_data, cal_predictions, changed_data, changed_predictions,
                               tmp_path / "other.jsonl", tmp_path / "other-temperature.json")
    assert changed["temperature"] == result["temperature"]
    with pytest.raises(FileExistsError):
        calibrate_remote(cal_data, cal_predictions, panel_data, panel_predictions, output, artifact)


@pytest.mark.parametrize("overlap", ["normalized_state", "source_group"])
def test_calibration_rejects_panel_state_or_group_leakage(tmp_path, overlap):
    from janus.remote_calibration import calibrate_remote, validate_calibration_splits
    cal_data, cal_predictions, cal_requests, _ = write_fixture(tmp_path, "calibration")
    panel_data, panel_predictions, panel_requests, _ = write_fixture(tmp_path, "panel")
    if overlap == "normalized_state":
        panel_requests[0]["state"] = "  CALIBRATION   OBSERVATION 0  "
    else:
        panel_requests[0]["group_id"] = cal_requests[0]["group_id"]
    write_jsonl(panel_data, panel_requests)
    with pytest.raises(ValueError, match="overlap"):
        validate_calibration_splits(cal_data, panel_data)
    with pytest.raises(ValueError, match="overlap"):
        calibrate_remote(cal_data, cal_predictions, panel_data, panel_predictions,
                         tmp_path / "out.jsonl", tmp_path / "temperature.json")
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[0].update(returned_model="jev-older"),
    lambda rows: rows[0].update(requested_model="jev-other"),
    lambda rows: rows[0].update(input_sha256="different-input"),
    lambda rows: rows[0].update(target=[0., 1.]),
    lambda rows: rows[0].update(keys=["b", "a"]),
    lambda rows: rows[0].update(probabilities=[.9, .9]),
    lambda rows: rows[0].update(probabilities=[float("nan"), .01]),
    lambda rows: rows[0].update(logits=[0., 0.]),
    lambda rows: rows[0].update(calibration_temperature=2.),
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows.pop(),
])
def test_calibration_rejects_mismatched_model_or_prediction_inputs(tmp_path, mutation):
    from janus.remote_calibration import calibrate_remote
    cal_data, cal_predictions, _, cal_rows = write_fixture(tmp_path, "calibration")
    panel_data, panel_predictions, _, _ = write_fixture(tmp_path, "panel")
    mutation(cal_rows)
    # Deliberately allow NaN here to exercise validation of external artifacts.
    cal_predictions.write_text("\n".join(json.dumps(row) for row in cal_rows) + "\n")
    with pytest.raises(ValueError):
        calibrate_remote(cal_data, cal_predictions, panel_data, panel_predictions,
                         tmp_path / "out.jsonl", tmp_path / "temperature.json")
    assert not (tmp_path / "out.jsonl").exists()


def test_calibration_handles_zero_probabilities_and_uses_cpu(tmp_path):
    from janus.remote_calibration import calibrate_remote
    cal_data, cal_predictions, _, cal_rows = write_fixture(tmp_path, "calibration")
    panel_data, panel_predictions, _, panel_rows = write_fixture(tmp_path, "panel")
    for path, rows in [(cal_predictions, cal_rows), (panel_predictions, panel_rows)]:
        for row in rows:
            row["probabilities"] = [1., 0.]
            row["logits"] = [0., math.log(1e-12)]
        write_jsonl(path, rows)
    result = calibrate_remote(cal_data, cal_predictions, panel_data, panel_predictions,
                              tmp_path / "out.jsonl", tmp_path / "temperature.json")
    assert result["log_probability_floor"] == 1e-12
    assert result["fit_device"] == "cpu" and result["fit_threads"] == 4
    assert math.isfinite(result["temperature"])
    assert result["after"]["nll"] < result["before"]["nll"]
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert all(all(0 < p < 1 for p in row["probabilities"]) for row in rows)


def test_output_must_not_alias_any_input(tmp_path):
    from janus.remote_calibration import calibrate_remote
    cal_data, cal_predictions, _, _ = write_fixture(tmp_path, "calibration")
    panel_data, panel_predictions, _, _ = write_fixture(tmp_path, "panel")
    digest = file_hash(panel_predictions)
    with pytest.raises((ValueError, FileExistsError)):
        calibrate_remote(cal_data, cal_predictions, panel_data, panel_predictions,
                         panel_predictions, tmp_path / "temperature.json")
    assert file_hash(panel_predictions) == digest
