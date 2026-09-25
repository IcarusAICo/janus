"""Phase 4 honesty: the held-out-aware selection rule, the withheld-family dev split, and the arm report."""

import json
import math
from pathlib import Path
import random

import pytest
import torch

from janus.data import load_requests, state_hash, synthetic_requests, write_json, write_jsonl
from janus.model import ModelConfig
from janus.training import TrainConfig, load_checkpoint, selection_score, train


def withheld_family(count, seed):
    """Same state template as the synthetic training family, different nuisance id, uniform targets: the model has
    nothing to learn here, so any confidence it carries over from training is confident error."""
    rows = []
    for i, request in enumerate(synthetic_requests(count, seed=seed)):
        row = request.to_dict()
        row["state"] = f"Record B-{seed}-{i}. " + row["state"].split(". ", 1)[1]
        for question in row["questions"].values():
            k = len(question["target"])
            question["target"] = [1 / k] * k
        row["group_id"] = state_hash(row["state"])
        rows.append(row)
    return rows


def test_heldout_penalty_changes_the_selected_step(tmp_path):
    torch.set_num_threads(2)
    write_jsonl(tmp_path / "train.jsonl", [r.to_dict() for r in synthetic_requests(16, seed=17)])
    write_jsonl(tmp_path / "dev.jsonl", [r.to_dict() for r in synthetic_requests(6, seed=23)])
    write_jsonl(tmp_path / "heldout.jsonl", withheld_family(6, 31))
    model = ModelConfig(backbone="tiny", hidden_size=32, layers=1, head_rank=8, adaptation="full")
    # A weight this large makes the penalty dominate, so selection must leave the confident late steps for an early,
    # near-uniform one; dev-NLL selection on the same trajectory picks a confident step.
    config = TrainConfig(model=model, epochs=15, accumulation=4, eval_every=6, device="cpu", backbone_lr=.03, head_lr=.03,
                         selection="dev_nll_plus_heldout", heldout_dev=str(tmp_path / "heldout.jsonl"), heldout_weight=1000.)
    result = train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / "run", config)
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    uniform = (math.log(4) + math.log(2) + math.log(3)) / 3  # a 4-way Choice, a Noul and a 3-level Score per state
    for row in history:
        assert row["heldout_uniform_nll"] == pytest.approx(uniform)
        assert row["heldout_penalty"] == pytest.approx(max(0., row["heldout_nll"] - uniform))
        assert row["selection_score"] == pytest.approx(row["dev_nll"] + 1000. * row["heldout_penalty"])
    by_score = min(history, key=lambda r: r["selection_score"])
    by_dev = min(history, key=lambda r: r["dev_nll"])
    assert result["best_step"] == by_score["step"] != by_dev["step"]
    assert by_dev["heldout_penalty"] > .01 > by_score["heldout_penalty"]  # dev selection is confidently wrong on B
    assert result["best_selection_score"] == pytest.approx(by_score["selection_score"])
    assert result["selection"] == "dev_nll_plus_heldout" and result["heldout_requests"] == 6
    _, metadata = load_checkpoint(tmp_path / "run" / "best.pt")
    assert metadata["step"] == result["best_step"] and metadata["heldout_sha256"]
    # The held-out file took part in selection, so evaluation must refuse it like any dev data.
    from janus.evaluation import check_unused
    with pytest.raises(ValueError, match="overlap"):
        check_unused(load_requests(tmp_path / "heldout.jsonl"), metadata)
    assert selection_score(TrainConfig(model=model), .5, 2., 1.) == .5  # dev_nll ignores the held-out terms
    assert selection_score(config, .5, .8, 1.) == .5  # below uniform is never rewarded
    with pytest.raises(ValueError, match="selection"):
        TrainConfig(model=model, selection="heldout").validate()
    with pytest.raises(ValueError, match="heldout_dev"):
        TrainConfig(model=model, selection="dev_nll_plus_heldout").validate()
    bad = TrainConfig(model=model, selection="dev_nll_plus_heldout", heldout_dev=str(tmp_path / "test_heldout.jsonl"))
    with pytest.raises(ValueError, match="train/dev"):
        train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / "bad", bad)


def test_build_heldout_dev_draws_withheld_families_disjoint_from_every_split(tmp_path):
    from janus.phase4_honesty_report import build_heldout_dev
    from janus.synth.build import _generate
    seen = set()
    write_jsonl(tmp_path / "train.jsonl", _generate("post", 10, random.Random("17:post:train"), "train", None, 0., seen))
    write_jsonl(tmp_path / "test_post_unseen.jsonl",
                _generate("post", 20, random.Random("17:post:unseen"), "test_post_unseen", None, 0., seen))
    write_json(tmp_path / "manifest.json", {"counts": {}, "tiers": {"T0": 30}, "holdouts": {"post": "families"}, "files": {}})
    manifest = build_heldout_dev(tmp_path, count=5)
    rows = [json.loads(l) for l in (tmp_path / "dev_post_unseen.jsonl").read_text().splitlines()]
    assert len(rows) == 5 and {r["group_id"].split(":")[1] for r in rows} <= {"8", "9"}
    assert all(r["family"] == "post" and r["tier"] == "T0" and "paraphrased" not in r for r in rows)
    others = {r.group_id for p in ("train", "test_post_unseen") for r in load_requests(tmp_path / f"{p}.jsonl")}
    assert not others & {r["group_id"] for r in rows}
    assert manifest["counts"]["dev_post_unseen"] == {"post": 5} and manifest["tiers"]["T0"] == 35
    assert "dev_post_unseen" in manifest["holdouts"]["post"] and "dev_post_unseen.jsonl" in manifest["files"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        build_heldout_dev(tmp_path)


@pytest.mark.skipif(not Path("data/phase1-v1/dev_post_unseen.jsonl").exists(), reason="Phase 1 data absent")
def test_shipped_heldout_dev_is_recorded_and_disjoint_from_the_withheld_test():
    from janus.data import file_hash
    root = Path("data/phase1-v1")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["files"]["dev_post_unseen.jsonl"] == file_hash(root / "dev_post_unseen.jsonl")
    dev, test = load_requests(root / "dev_post_unseen.jsonl"), load_requests(root / "test_post_unseen.jsonl")
    assert len(dev) == manifest["counts"]["dev_post_unseen"]["post"] == 300
    assert not {r.group_id for r in dev} & {r.group_id for r in test}
    assert not {state_hash(r.state) for r in dev} & {state_hash(r.state) for r in test}


def write_run(root, arm, seed, test_acc, heldout_nll=None, step=500):
    run = Path(root) / arm / f"s{seed}"
    write_json(run / "test" / "metrics.json", {"raw": {"accuracy": test_acc, "nll": 1 - test_acc}})
    write_json(run / "panel" / "metrics.json", {"raw": {"nll": .3, "ece": .02}})
    write_json(run / "mmlu_pro" / "metrics.json", {"raw": {"accuracy": .45, "nll": 1.6}})
    if heldout_nll is not None:
        write_json(run / "test_post_unseen" / "metrics.json", {"raw": {"accuracy": .5, "nll": heldout_nll}, "uniform": {"nll": .896}})
    write_json(run / "summary.json", {"best_step": step, "elapsed_seconds": 6000.})


def test_honesty_report_aggregates_pairs_and_gates(tmp_path):
    from janus.phase4_honesty_report import collect, gate, paired_deltas, report
    for seed, acc, nll in ((17, .82, 1.10), (23, .83, 1.08), (29, .84, 1.09)):
        write_run(tmp_path, "base", seed, acc, nll)
        write_run(tmp_path, "honest", seed, acc - .002, nll - .25, step=250)
        write_run(tmp_path, "lossy", seed, acc - .05, .8)
    write_run(tmp_path, "partial", 17, .9)
    arms = ("base", "honest", "lossy", "partial", "absent")
    runs = collect(tmp_path, arms)
    assert set(runs["honest"]) == {17, 23, 29} and set(runs["partial"]) == {17} and runs["absent"] == {}
    assert paired_deltas(runs["honest"], runs["base"], "test_acc") == pytest.approx([-.002] * 3)
    assert paired_deltas(runs["partial"], runs["base"], "heldout_nll") == []
    assert gate(runs["base"], runs["base"]) == pytest.approx({"heldout_nll": 1.09, "uniform_nll": .896, "seeds": 3, "heldout_ok": False,
                                                              "accuracy_delta": 0., "accuracy_ok": True, "passed": False})
    assert gate(runs["honest"], runs["base"])["passed"] is True
    assert gate(runs["lossy"], runs["base"]) == pytest.approx({"heldout_nll": .8, "uniform_nll": .896, "seeds": 3, "heldout_ok": True,
                                                               "accuracy_delta": -.05, "accuracy_ok": False, "passed": False})
    assert gate(runs["partial"], runs["base"])["passed"] is None
    text = report(tmp_path, arms)
    assert "| base | 0.830 ± 0.010 | 0.170 ± 0.010 | 0.300 ± 0.000 | 0.020 ± 0.000 | 0.500 ± 0.000 | 1.090 ± 0.010 | 0.450 ± 0.000 | 1.600 ± 0.000 | 500, 500, 500 | 100 |" in text
    assert "| partial | 0.900 ± 0.000 (n=1) |" in text and "| absent | n/a | n/a |" in text
    assert "| honest | 3 | -0.002 ± 0.000 |" in text and "| partial | 1 | 0.080 ± 0.000 (n=1) |" in text
    assert "| honest | 3 | 0.840 | 0.896 | pass | -0.002 | pass | pass |" in text
    assert "| lossy | 3 | 0.800 | 0.896 | pass | -0.050 | fail | fail |" in text
    assert "| base | 3 | 1.090 | 0.896 | fail | 0.000 | pass | fail |" in text
    assert "| absent | 0 | n/a | n/a | n/a | n/a | n/a | n/a |" in text
    assert "Uniform NLL on held-out post: 0.896." in text
