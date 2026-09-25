import json

import pytest

from janus.study_report import discover_predictions, partition_predictions, validate_panel, primary_comparisons


def test_discovery_requires_all_expected_results_unless_explicit_partial(tmp_path):
    remote = tmp_path / 'jev'
    remote.mkdir()
    (remote / 'calibrated-predictions.jsonl').write_text('{}\n')
    with pytest.raises(FileNotFoundError, match='Missing study'):
        discover_predictions(tmp_path)
    paths = discover_predictions(tmp_path, partial=True)
    assert paths == {'Jev-1.13.0': remote / 'calibrated-predictions.jsonl'}


def test_optional_prespecified_replication_is_included_if_present(tmp_path):
    remote = tmp_path / 'jev'
    remote.mkdir()
    (remote / 'calibrated-predictions.jsonl').write_text('{}\n')
    repeat = tmp_path / 'listwise-06b-seed23' / 'evaluation'
    repeat.mkdir(parents=True)
    (repeat / 'predictions.jsonl').write_text('{}\n')
    assert 'Listwise 0.6B seed23' in discover_predictions(tmp_path, partial=True)


def test_primary_contrasts_compare_controls_to_trained_primary(monkeypatch):
    import janus.study_analysis
    calls = []
    monkeypatch.setattr(janus.study_analysis, 'analyze_study', lambda *a, **k:
                        calls.append((a, k)) or {'comparisons': {'Frozen 0.6B': {'raw': {}}}})
    paths = {'Listwise 0.6B': 'main', 'Frozen 0.6B': 'frozen', 'Jev-1.13.0': 'remote'}
    result = primary_comparisons(paths, samples=10)
    assert calls == [(({'Listwise 0.6B': 'main', 'Frozen 0.6B': 'frozen'},),
                      {'reference': 'Listwise 0.6B', 'bootstrap_samples': 10})]
    assert 'Frozen 0.6B' in result['comparisons']
    assert primary_comparisons({'Jev-1.13.0': 'remote'}) is None


def test_partition_selection_preserves_complete_rows_and_uses_audit_labels(tmp_path):
    rows = [{'group_id': 'a', 'question_id': 'q', 'input_sha256': 'original'},
            {'group_id': 'b', 'question_id': 'r', 'input_sha256': 'second'}]
    source = tmp_path / 'source.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    labels = {'a': {'q': {'intent_partition': 'unseen'}}, 'b': {'r': {'intent_partition': 'seen'}}}
    result = partition_predictions({'model': source}, labels, 'unseen', tmp_path / 'subset')
    saved = [json.loads(line) for line in result['model'].read_text().splitlines()]
    assert saved == [rows[0]]
    assert len(source.read_text().splitlines()) == 2


def test_panel_validation_rejects_mutually_matching_but_wrong_benchmark(tmp_path):
    import hashlib
    from janus.data import synthetic_requests, write_jsonl
    request = synthetic_requests(1)[0]
    panel = tmp_path / 'panel.jsonl'
    write_jsonl(panel, [request.to_dict()])
    rows = []
    for q in request.questions:
        sig = {'state': request.state, 'kind': q.kind, 'instructions': q.instructions,
               'options': [(o.key, o.description) for o in q.options]}
        rows.append({'group_id': request.group_id, 'question_id': q.id, 'target': list(q.target),
                     'keys': [o.key for o in q.options], 'kind': q.kind,
                     'input_sha256': hashlib.sha256(json.dumps(sig, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
    predictions = tmp_path / 'predictions.jsonl'
    write_jsonl(predictions, rows)
    validate_panel(predictions, panel)
    rows[0]['input_sha256'] = 'a' * 64
    write_jsonl(predictions, rows)
    with pytest.raises(ValueError, match='benchmark'):
        validate_panel(predictions, panel)
    write_jsonl(predictions, rows[1:])
    with pytest.raises(ValueError, match='benchmark'):
        validate_panel(predictions, panel)
