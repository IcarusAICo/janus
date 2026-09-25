import hashlib
import importlib.util
import json

import pytest
import torch

from janus.data import file_hash, state_hash, write_json, write_jsonl
from janus.schema import Request


def exposure_module():
    assert importlib.util.find_spec("janus.exposure") is not None, "Exposure audit module is missing"
    from janus import exposure
    return exposure


def request(text, group):
    return Request.from_dict({"state": text, "group_id": group, "questions": {
        "decision": {"type": "noul", "instructions": "Does the proposition hold?", "target": [0, 1]}}})


def digest(states):
    return hashlib.sha256("\n".join(sorted(state_hash(s) for s in set(states))).encode()).hexdigest()


def fixture_run(tmp_path, *, config_overrides=None, selected_step=2, selected_seen=4,
                final_step=4, final_seen=7, seen_hashes=None, rows=None):
    rows = rows or [request("Review A", "sst5:a"), request("Review B", "sst5:b"),
                    request("Premise P and hypothesis A", "snli:p"), request("Premise P and hypothesis B", "snli:p"),
                    request("Assistant request", "clinc150:x")]
    train_path = tmp_path / "train.jsonl"
    write_jsonl(train_path, [r.to_dict() for r in rows])
    run = tmp_path / "run"
    run.mkdir()
    config = {"seed": 7, "data_seed": 19, "train_limit": None, "epochs": 2,
              "accumulation": 2, "max_steps": None, "max_seconds": None, **(config_overrides or {})}
    metadata = {"config": config, "train_sha256": file_hash(train_path)}
    write_json(run / "config.json", metadata)
    checkpoint_metadata = {**metadata, "step": selected_step, "requests_seen": selected_seen}
    if seen_hashes is not None:
        checkpoint_metadata["seen_state_hashes"] = seen_hashes
    torch.save({"weights": {"example": torch.tensor([1.])}, "metadata": checkpoint_metadata}, run / "best.pt")
    write_json(run / "summary.json", {"steps": final_step, "best_step": selected_step, "requests_seen": final_seen,
                                       "budget_limited": False, "elapsed_seconds": 4})
    write_json(run / "history.json", [{"step": 0}, {"step": selected_step, "requests_seen": selected_seen},
                                       {"step": final_step, "requests_seen": final_seen}])
    return run, train_path, rows


def test_reconstructs_selected_and_final_exposure_across_partial_epoch_batches(tmp_path):
    m = exposure_module()
    run, train_path, rows = fixture_run(tmp_path)
    # Seed 7 orders: [4,0,3,1,2], then [2,3,1,4,0]. Step 3 has only one state.
    before = file_hash(run / "best.pt")
    report = m.audit_exposure(run, train_path, tmp_path / "exposure.json")
    selected = report["selected_checkpoint"]
    assert selected["step"] == 2
    assert selected["requests_seen"] == 4
    assert selected["unique_states"] == 4
    assert selected["unique_groups"] == 4
    assert selected["state_hashes_sha256"] == digest([rows[i].state for i in (4, 0, 3, 1)])
    assert selected["domains"]["sst5"]["presented_states"] == 2
    final = report["final"]
    assert final["step"] == 4
    assert final["presented_states"] == final["requests_seen"] == 7
    assert final["unique_states"] == 5
    assert final["unique_groups"] == 4
    assert final["domains"]["snli"]["presented_states"] == 4
    assert final["domains"]["snli"]["unique_states"] == 2
    assert final["domains"]["snli"]["unique_groups"] == 1
    assert report["planned"]["steps"] == 6
    assert report["planned"]["requests_seen"] == 10
    assert report["verification"]["seen_state_hashes"] == "reconstructed_missing_metadata"
    assert file_hash(run / "best.pt") == before
    assert json.loads((tmp_path / "exposure.json").read_text()) == report


def test_balanced_subset_uses_data_seed_before_training_shuffle_and_step_cap(tmp_path):
    m = exposure_module()
    rows = [request(f"s{i}", f"sst5:s{i}") for i in range(3)] + [request(f"n{i}", f"snli:n{i}") for i in range(3)]
    # data_seed19 selects [n2,s2,n0,s0]; seed7 shuffle4 starts [s0,s2].
    run, train_path, _ = fixture_run(tmp_path, rows=rows, config_overrides={"train_limit": 4, "max_steps": 2, "epochs": 3},
                                    selected_step=1, selected_seen=2, final_step=2, final_seen=4)
    report = m.audit_exposure(run, train_path, tmp_path / "exposure.json")
    assert report["full_pool_size"] == 6
    assert report["training_pool"]["unique_states"] == 4
    assert report["training_pool"]["state_hashes_sha256"] == digest(["n2", "s2", "n0", "s0"])
    assert report["selected_checkpoint"]["state_hashes_sha256"] == digest(["s0", "s2"])
    assert report["selected_checkpoint"]["domains"]["sst5"]["presented_states"] == 2
    assert report["planned"]["uncapped_steps"] == 6
    assert report["planned"]["steps"] == 2
    assert report["planned"]["requests_seen"] == 4
    assert report["cap_status"]["step_cap_reached"] is True
    assert report["cap_status"]["configured_plan_completed"] is True


def test_checkpoint_consumed_sets_are_verified_when_available(tmp_path):
    m = exposure_module()
    expected = [state_hash(text) for text in ("Assistant request", "Review A", "Premise P and hypothesis B", "Review B")]
    run, train_path, _ = fixture_run(tmp_path, seen_hashes=expected)
    payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    payload["metadata"]["seen_group_ids"] = ["clinc150:x", "sst5:a", "snli:p", "sst5:b"]
    torch.save(payload, run / "best.pt")
    report = m.audit_exposure(run, train_path, tmp_path / "exposure.json")
    assert report["verification"]["seen_state_hashes"] == "verified"
    assert report["verification"]["seen_group_ids"] == "verified"
    payload["metadata"]["seen_state_hashes"] = [state_hash("Wrong state")]
    torch.save(payload, run / "best.pt")
    with pytest.raises(ValueError, match="seen_state_hashes"):
        m.audit_exposure(run, train_path, tmp_path / "invalid.json")


def test_time_cap_and_zero_step_selected_checkpoint_have_distinct_exposures(tmp_path):
    m = exposure_module()
    run, train_path, _ = fixture_run(tmp_path, selected_step=0, selected_seen=0, final_step=1, final_seen=2,
                                    config_overrides={"max_seconds": 1})
    summary = json.loads((run / "summary.json").read_text())
    summary["budget_limited"] = True
    write_json(run / "summary.json", summary)
    report = m.audit_exposure(run, train_path, tmp_path / "exposure.json")
    assert report["selected_checkpoint"]["unique_states"] == 0
    assert report["final"]["unique_states"] == 2
    assert report["cap_status"]["time_cap_reached"] is True
    assert report["cap_status"]["configured_plan_completed"] is False
    assert report["cap_status"]["reason"] == "max_seconds"


@pytest.mark.parametrize("corruption", ["train_hash", "counter", "best_step", "history"])
def test_audit_rejects_artifacts_that_cannot_describe_the_same_training_run(tmp_path, corruption):
    m = exposure_module()
    run, train_path, _ = fixture_run(tmp_path)
    if corruption == "train_hash":
        write_jsonl(train_path, [request("Different input", "sst5:x").to_dict()])
    elif corruption == "history":
        write_json(run / "history.json", [{"step": 2, "requests_seen": 3}])
    else:
        summary = json.loads((run / "summary.json").read_text())
        summary["requests_seen" if corruption == "counter" else "best_step"] = 3
        write_json(run / "summary.json", summary)
    with pytest.raises(ValueError):
        m.audit_exposure(run, train_path, tmp_path / "invalid.json")
