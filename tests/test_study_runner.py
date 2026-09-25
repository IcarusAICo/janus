import json
from pathlib import Path

import pytest

from janus.study_runner import build_steps, run_steps


def test_plan_separates_calibration_test_and_runs_gpu_sequentially(tmp_path):
    steps = build_steps(Path('data/study-v1'), tmp_path, ['listwise', 'small_data'], baselines=True)
    assert [s['name'] for s in steps] == [
        'listwise-06b-seed17/train', 'listwise-06b-seed17/calibrate',
        'listwise-06b-seed17/evaluate', 'small-data-06b-seed17/train',
        'small-data-06b-seed17/calibrate', 'small-data-06b-seed17/evaluate',
        'zero-shot-06b/evaluate', 'zero-shot-17b/evaluate']
    assert 'data/study-v1/dev.jsonl' in steps[0]['command']
    assert 'data/study-v1/calibration.jsonl' in steps[1]['command']
    assert 'data/study-v1/benchmark.jsonl' in steps[2]['command']


def test_runner_skips_complete_stages_and_records_failure(tmp_path):
    complete = tmp_path / 'complete.json'
    complete.write_text('{}')
    steps = [dict(name='complete', command=['skip'], marker=str(complete)),
             dict(name='fails', command=['fails'], marker=str(tmp_path / 'missing'))]
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        raise RuntimeError('failed')
    with pytest.raises(RuntimeError, match='failed'):
        run_steps(steps, tmp_path / 'status.json', execute=execute)
    assert calls == [['fails']]
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['stages'][0]['status'] == 'already_complete'
    assert status['stages'][1]['status'] == 'failed'


def test_runner_requires_completion_artifact_and_honors_deadline(tmp_path):
    step = dict(name='x', command=['x'], marker=str(tmp_path / 'missing'))
    with pytest.raises(RuntimeError, match='completion artifact'):
        run_steps([step], tmp_path / 's.json', execute=lambda *a, **k: None)
    calls = []
    result = run_steps([step], tmp_path / 'expired.json', deadline_utc='2000-01-01T00:00:00+00:00',
                       execute=lambda *a, **k: calls.append(a))
    assert not calls
    assert result['stages'][0]['status'] == 'not_started_deadline'


def completed_study(tmp_path, monkeypatch):
    from dataclasses import asdict
    from janus.data import file_hash, synthetic_requests
    from janus.training import TrainConfig
    from janus import study_runner

    data, root = tmp_path / 'data', tmp_path / 'runs'
    data.mkdir()
    for name, seed in (('train', 17), ('dev', 23), ('calibration', 41), ('benchmark', 42)):
        (data / f'{name}.jsonl').write_text(''.join(json.dumps(r.to_dict())+'\n' for r in synthetic_requests(2, seed)))
    config_path = tmp_path / 'config.json'
    raw_config = {'model': {'backbone': 'tiny', 'adaptation': 'frozen'}, 'epochs': 1}
    config_path.write_text(json.dumps(raw_config))
    monkeypatch.setitem(study_runner.VARIANTS, 'listwise', ('test-run', str(config_path)))
    monkeypatch.setattr(study_runner, 'BACKBONES', [('test', 'Qwen/test', 'pinned-revision')])
    steps = build_steps(data, root, ['listwise'], baselines=True, device='cpu')
    run = root / 'test-run'
    run.mkdir(parents=True)
    config = TrainConfig.from_dict(raw_config)
    config.device = 'cpu'
    (run / 'config.json').write_text(json.dumps({'config': asdict(config),
        'train_sha256': file_hash(data/'train.jsonl'), 'dev_sha256': file_hash(data/'dev.jsonl')}))
    (run / 'summary.json').write_text(json.dumps({'steps': 2, 'best_step': 1}))
    (run / 'best.pt').write_bytes(b'bounded-checkpoint-fixture')
    checkpoint_hash = file_hash(run/'best.pt')
    (run/'calibration.json').write_text(json.dumps({'temperature': 2., 'requests': 2,
        'checkpoint_sha256': checkpoint_hash, 'data_sha256': file_hash(data/'calibration.jsonl')}))
    (run/'evaluation').mkdir()
    (run/'evaluation'/'metrics.json').write_text(json.dumps({'temperature': 2., 'requests': 2,
        'checkpoint_sha256': checkpoint_hash, 'data_sha256': file_hash(data/'benchmark.jsonl')}))
    (run/'evaluation'/'predictions.jsonl').write_text('{"fixture": true}\n')
    baseline = root/'zero-shot-test'
    baseline.mkdir()
    (baseline/'metrics.json').write_text(json.dumps({'backbone': 'Qwen/test', 'revision': 'pinned-revision',
        'requests': 2, 'data_sha256': file_hash(data/'benchmark.jsonl'),
        'calibration_sha256': file_hash(data/'calibration.jsonl')}))
    (baseline/'predictions.jsonl').write_text('{"fixture": true}\n')
    return steps, data, root, config_path


def no_execute(*args, **kwargs):
    raise AssertionError('A completed stage must not execute during provenance checking')


def test_completed_study_is_skipped_only_after_configuration_and_hash_validation(tmp_path, monkeypatch):
    steps, _, _, _ = completed_study(tmp_path, monkeypatch)
    result = run_steps(steps, tmp_path/'status.json', execute=no_execute)
    assert all(row['status'] == 'already_complete' for row in result['stages'])
    assert all(row.get('provenance_verified') is True for row in result['stages'])


@pytest.mark.parametrize('corruption', [
    'config_file', 'config_default', 'device', 'train', 'dev', 'missing_checkpoint',
    'calibration_checkpoint', 'calibration_data', 'calibration_temperature',
    'evaluation_checkpoint', 'evaluation_data', 'evaluation_temperature', 'evaluation_requests',
    'baseline_backbone', 'baseline_revision', 'baseline_data', 'baseline_calibration',
])
def test_completed_stages_reject_changed_scope_instead_of_silently_skipping(tmp_path, monkeypatch, corruption):
    steps, data, root, config_path = completed_study(tmp_path, monkeypatch)
    run = root/'test-run'
    def change(path, field, value):
        obj = json.loads(path.read_text())
        obj[field] = value
        path.write_text(json.dumps(obj))
    if corruption == 'config_file':
        change(config_path, 'epochs', 2)
    elif corruption in ('config_default', 'device'):
        path = run/'config.json'
        obj = json.loads(path.read_text())
        obj['config']['head_lr' if corruption == 'config_default' else 'device'] = .02 if corruption == 'config_default' else 'cuda:7'
        path.write_text(json.dumps(obj))
    elif corruption in ('train', 'dev'):
        path = data/f'{corruption}.jsonl'
        path.write_text(path.read_text()+'\n')
    elif corruption == 'missing_checkpoint':
        (run/'best.pt').unlink()
    elif corruption.startswith('calibration_'):
        field = {'calibration_checkpoint': 'checkpoint_sha256', 'calibration_data': 'data_sha256',
                 'calibration_temperature': 'temperature'}[corruption]
        change(run/'calibration.json', field, -1 if field == 'temperature' else 'wrong')
    elif corruption.startswith('evaluation_'):
        field = {'evaluation_checkpoint': 'checkpoint_sha256', 'evaluation_data': 'data_sha256',
                 'evaluation_temperature': 'temperature', 'evaluation_requests': 'requests'}[corruption]
        change(run/'evaluation'/'metrics.json', field, 3 if field == 'temperature' else 1 if field == 'requests' else 'wrong')
    else:
        field = {'baseline_backbone': 'backbone', 'baseline_revision': 'revision',
                 'baseline_data': 'data_sha256', 'baseline_calibration': 'calibration_sha256'}[corruption]
        change(root/'zero-shot-test'/'metrics.json', field, 'wrong')
    with pytest.raises((ValueError, FileNotFoundError)):
        run_steps(steps, tmp_path/'status.json', execute=no_execute)
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['stages'][-1]['status'] == 'provenance_mismatch'


def test_new_stage_outputs_receive_the_same_provenance_validation(tmp_path, monkeypatch):
    steps, _, _, _ = completed_study(tmp_path, monkeypatch)
    stage = steps[-1]
    marker = Path(stage['marker'])
    bad = json.loads(marker.read_text())
    bad['revision'] = 'unexpected-revision'
    marker.unlink()
    def execute(*args, **kwargs):
        marker.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        run_steps([stage], tmp_path/'status.json', execute=execute)
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['stages'][0]['status'] == 'failed'
