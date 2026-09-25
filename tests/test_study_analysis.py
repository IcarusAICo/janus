"""Hand-calculated checks for strict, comparable multi-model study analysis."""

from copy import deepcopy
import importlib.util
import json
import math

import pytest


def analysis():
    assert importlib.util.find_spec("janus.study_analysis") is not None, "Study analysis is missing"
    from janus import study_analysis
    return study_analysis


def record(group="snli:a", question="snli:q", p=(.8, .2), target=(1., 0.),
           kind="choice", remote=False, calibrated=None):
    row = {"group_id": group, "question_id": question, "kind": kind,
           "input_sha256": "a" * 64, "cardinality": len(p),
           "keys": [str(i) for i in range(len(p))], "target": list(target),
           "logits": [math.log(max(v, 1e-300)) for v in p],
           "probabilities": list(p if calibrated is None else calibrated),
           "nll": 777., "calibrated_nll": 888.}
    if remote:
        row.update(requested_model="jev-1.13.0", returned_model="jev-1.13.0")
    return row


def paths(tmp_path, local, remote):
    result = {}
    for name, rows in (("local", local), ("jev", remote)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result[name] = path
    return result


def test_common_nll_floor_caps_local_and_remote_exact_zero_identically(tmp_path):
    m = analysis()
    local = record(p=(1., 0.), target=(0., 1.))
    local["logits"] = [1000., -1000.]
    remote = record(p=(1., 0.), target=(0., 1.), remote=True)
    report = m.analyze_study(paths(tmp_path, [local], [remote]), bootstrap_samples=40)
    for name in ("local", "jev"):
        for mode in ("raw", "calibrated"):
            metrics = report["models"][name][mode]
            assert metrics["nll"] == pytest.approx(27.631021115928547)
            assert metrics["brier"] == 2
            assert metrics["accuracy"] == 0
            assert metrics["ece"] == 1
    assert report["nll_probability_floor"] == 1e-12
    assert report["comparisons"]["local"]["raw"]["nll"]["difference"] == 0


def test_raw_calibrated_and_remote_probabilities_have_distinct_sources(tmp_path):
    m = analysis()
    local = record(p=(.99, .01), calibrated=(.6, .4))
    remote = record(p=(.9, .1), remote=True)
    remote["logits"] = [0., 0.]  # Remote raw must use the original probabilities.
    report = m.analyze_study(paths(tmp_path, [local], [remote]), bootstrap_samples=40)
    assert report["models"]["local"]["raw"]["nll"] == pytest.approx(-math.log(.99))
    assert report["models"]["local"]["calibrated"]["nll"] == pytest.approx(-math.log(.6))
    assert report["models"]["local"]["calibrated"]["brier"] == pytest.approx(.32)
    assert report["models"]["jev"]["raw"]["nll"] == pytest.approx(-math.log(.9))
    assert report["models"]["jev"]["raw"] == report["models"]["jev"]["calibrated"]


def test_unequal_cluster_sizes_keep_decision_weighted_nll_and_reproducibility(tmp_path):
    m = analysis()
    local, remote = [], []
    for i in range(8):
        group = "snli:a" if i < 6 else "snli:b"
        lp = math.exp(-1 if i < 6 else -4)
        rp = math.exp(-2)
        local.append(record(group, f"snli:q{i}", (lp, 1-lp)))
        remote.append(record(group, f"snli:q{i}", (rp, 1-rp), remote=True))
    inputs = paths(tmp_path, local, list(reversed(remote)))
    first = m.analyze_study(inputs, bootstrap_samples=200, seed=41)
    second = m.analyze_study(inputs, bootstrap_samples=200, seed=41)
    ci = first["comparisons"]["local"]["raw"]["nll"]
    assert ci["difference"] == pytest.approx(-.25)
    assert ci["groups"] == 2 and ci["count"] == 8
    assert ci["lower"] == pytest.approx(-1)
    assert ci["upper"] == pytest.approx(2)
    assert ci["favorable_direction"] == "negative"
    assert first == second


def test_accuracy_difference_and_agreement_are_not_ground_truth_substitutes(tmp_path):
    m = analysis()
    local = [record(p=(.9, .1))]
    remote = [record(p=(.1, .9), remote=True)]
    report = m.analyze_study(paths(tmp_path, local, remote), bootstrap_samples=40)
    compared = report["comparisons"]["local"]["raw"]
    assert compared["accuracy"]["difference"] == 1
    assert compared["accuracy"]["lower"] == compared["accuracy"]["upper"] == 1
    assert compared["accuracy"]["favorable_direction"] == "positive"
    assert compared["agreement"] == 0


def test_domain_kind_macro_metrics_and_score_expected_level_error(tmp_path):
    m = analysis()
    local = [record("sst5:s", "sst5:score", (.1, .2, .7), (0., 1., 0.), kind="score"),
             record("snli:a", "snli:q1", (.8, .2)),
             record("snli:b", "snli:q2", (.8, .2))]
    remote = [{**row, "requested_model": "jev-1.13.0", "returned_model": "jev-1.13.0"} for row in local]
    report = m.analyze_study(paths(tmp_path, local, remote), bootstrap_samples=40)
    model = report["models"]["local"]
    assert model["by_domain"]["sst5"]["raw"]["score_expected_level_mae"] == pytest.approx(.6)
    assert model["by_kind"]["score"]["raw"]["count"] == 1
    assert model["raw"]["score_count"] == 1
    assert model["raw"]["score_expected_level_mae"] == pytest.approx(.6)
    assert model["macro_domain"]["raw"]["accuracy"] == pytest.approx(.5)
    assert model["raw"]["accuracy"] == pytest.approx(2/3)
    assert set(report["comparisons"]["local"]["by_domain"]) == {"sst5", "snli"}
    assert set(report["comparisons"]["local"]["by_kind"]) == {"score", "choice"}


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows[0].update(input_sha256="b" * 64),
    lambda rows: rows[0].update(target=[0., 1.]),
    lambda rows: rows[0].update(keys=["1", "0"]),
    lambda rows: rows[0].update(kind="noul"),
    lambda rows: rows[0].update(probabilities=[.2, .2]),
    lambda rows: rows[0].update(logits=[float("nan"), 0.]),
    lambda rows: rows[0].update(target=[-1., 2.]),
    lambda rows: rows[0].update(input_sha256=""),
])
def test_incomplete_mismatched_or_invalid_artifacts_fail_loudly(tmp_path, change):
    m = analysis()
    local = [record(question="snli:q1"), record(question="snli:q2")]
    remote = [{**row, "requested_model": "jev-1.13.0", "returned_model": "jev-1.13.0"} for row in local]
    change(local)
    with pytest.raises(ValueError):
        m.analyze_study(paths(tmp_path, local, remote), bootstrap_samples=40)


def test_machine_report_and_markdown_state_comparison_direction(tmp_path):
    m = analysis()
    report = m.analyze_study(paths(tmp_path, [record()], [record(remote=True)]), bootstrap_samples=40)
    json.dumps(report, allow_nan=False)
    rendered = m.render_markdown(report)
    assert "local minus jev" in rendered.lower()
    assert "negative" in rendered.lower() and "positive" in rendered.lower()
    assert "1e-12" in rendered


def test_nll_sensitivity_reports_zero_targets_as_literal_infinite_nll(tmp_path):
    m = analysis()
    local = record(p=(.999, .001), target=(0., 1.))
    remote = record(p=(1., 0.), target=(0., 1.), remote=True)
    report = m.analyze_study(paths(tmp_path, [local], [remote]), bootstrap_samples=40)
    a, b = report["models"]["local"]["raw"], report["models"]["jev"]["raw"]
    assert a["target_zero_probability_count"] == 0
    assert a["literal_nll"] == pytest.approx(-math.log(.001))
    assert a["literal_nll_is_infinite"] is False
    assert b["target_zero_probability_count"] == 1
    assert b["literal_nll"] is None and b["literal_nll_is_infinite"] is True
    for floor in (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2):
        assert b["nll_sensitivity"][str(floor)] == pytest.approx(-math.log(floor))
    assert a["nll_sensitivity"][str(1e-2)] == pytest.approx(-math.log(.01))
    assert b["brier"] == 2
    assert "infinite" in m.render_markdown(report).lower()


def test_remote_calibration_keeps_original_probabilities_for_raw_comparison(tmp_path):
    m = analysis()
    local = record(p=(.8, .2), calibrated=(.7, .3))
    remote = record(p=(.6, .4), remote=True)
    remote["raw_probabilities"] = [.9, .1]
    report = m.analyze_study(paths(tmp_path, [local], [remote]), bootstrap_samples=40)
    assert report["models"]["jev"]["raw"]["nll"] == pytest.approx(-math.log(.9))
    assert report["models"]["jev"]["calibrated"]["nll"] == pytest.approx(-math.log(.6))
    assert report["models"]["jev"]["has_separate_calibrated_probabilities"] is True
    assert report["comparisons"]["local"]["calibrated"]["nll"]["difference"] == pytest.approx(math.log(.6/.7))
