"""Check standalone plot artifacts without touching source predictions."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


def plots():
    assert importlib.util.find_spec("janus.study_plots") is not None, "Study plotting is missing"
    from janus import study_plots
    return study_plots


def report(source):
    metrics = {"accuracy": .8, "target_zero_probability_count": 0,
               "reliability": [{"count": 3, "confidence": .6, "accuracy": 2/3},
                               {"count": 0, "confidence": None, "accuracy": None},
                               {"count": 7, "confidence": .9, "accuracy": 6/7}],
               "nll_sensitivity": {str(f): .4 for f in (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2)}}
    models = {}
    for name in ("listwise-0.6B", "listwise-1.7B", "frozen", "jev"):
        models[name] = {"path": str(source), "raw": deepcopy(metrics), "calibrated": deepcopy(metrics),
                        "by_domain": {domain: {"raw": deepcopy(metrics), "calibrated": deepcopy(metrics)}
                                      for domain in ("banking77", "snli")}}
    models["jev"]["raw"]["target_zero_probability_count"] = 2
    return {"reference": "jev", "models": models, "nll_probability_floor": 1e-12,
            "nll_sensitivity_floors": [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2]}


def test_generates_three_headless_png_svg_figures_without_mutating_sources(tmp_path):
    m = plots()
    source = tmp_path / "predictions.jsonl"
    source.write_text('{"do_not_modify": true}\n')
    original_bytes = source.read_bytes()
    data = report(source)
    original_report = json.dumps(data, sort_keys=True)
    output = tmp_path / "figures"
    saved = m.plot_study(data, output, reliability_models=["listwise-0.6B", "listwise-1.7B", "jev"])
    assert set(saved) == {"domain_accuracy", "reliability", "nll_sensitivity"}
    for formats in saved.values():
        assert set(formats) == {"png", "svg"}
        assert Path(formats["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert "<svg" in Path(formats["svg"]).read_text()
    assert 'frozen · raw' in Path(saved['nll_sensitivity']['svg']).read_text()
    assert 'frozen · calibrated' in Path(saved['nll_sensitivity']['svg']).read_text()
    assert 'frozen' not in Path(saved['reliability']['svg']).read_text()
    assert 'frozen' in Path(saved['domain_accuracy']['svg']).read_text()
    assert source.read_bytes() == original_bytes
    assert json.dumps(data, sort_keys=True) == original_report
    with pytest.raises(FileExistsError):
        m.plot_study(data, output)


@pytest.mark.parametrize('selected', [False, True])
def test_nll_includes_capitalized_frozen_once_without_changing_reliability(tmp_path, selected):
    data = report(tmp_path / 'unused')
    data['models']['Frozen 0.6B'] = data['models'].pop('frozen')
    selection = ['listwise-0.6B', 'listwise-1.7B', 'jev']
    if selected:
        selection.append('Frozen 0.6B')
    saved = plots().plot_study(data, tmp_path / 'figures', reliability_models=selection)
    sensitivity = Path(saved['nll_sensitivity']['svg']).read_text()
    reliability = Path(saved['reliability']['svg']).read_text()
    assert sensitivity.count('Frozen 0.6B · raw') == 1
    assert sensitivity.count('Frozen 0.6B · calibrated') == 1
    assert ('Frozen 0.6B · raw' in reliability) is selected


def test_unknown_reliability_model_fails_before_writing_figures(tmp_path):
    m = plots()
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="model"):
        m.plot_study(report(tmp_path / "unused"), output, reliability_models=["missing"])
    assert not output.exists()
