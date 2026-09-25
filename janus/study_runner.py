"""Sequential, artifact-checked study execution on a single GPU.

Run from the repository root. Existing completed stages are skipped; incomplete
training directories are never silently overwritten or resumed without optimizer
state. A deadline prevents new stages, while each training config caps its run.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time


VARIANTS = {
    'listwise': ('listwise-06b-seed17', 'configs/study_listwise.json'),
    'independent': ('independent-06b-seed17', 'configs/study_independent.json'),
    'frozen': ('frozen-06b-seed17', 'configs/study_frozen.json'),
    'small_data': ('small-data-06b-seed17', 'configs/study_small_data.json'),
    'larger': ('listwise-17b-seed17', 'configs/study_larger.json'),
}
BACKBONES = [
    ('06b', 'Qwen/Qwen3-0.6B-Base', 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'),
    ('17b', 'Qwen/Qwen3-1.7B-Base', 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'),
]


def build_steps(data_dir, output, variants=None, baselines=True, device='cuda:0'):
    data_dir, output = Path(data_dir), Path(output)
    prefix = [sys.executable, '-m', 'jev']
    stages = []
    for variant in variants if variants is not None else VARIANTS:
        name, config = VARIANTS[variant]
        run = output / name
        checkpoint = str(run / 'best.pt')
        calibration = str(run / 'calibration.json')
        stages.extend([
            {'name': name + '/train', 'marker': str(run / 'summary.json'),
             'provenance': {'kind': 'train'},
             'command': prefix + ['train', '--config', config, '--train', str(data_dir / 'train.jsonl'),
                        '--dev', str(data_dir / 'dev.jsonl'), '--output', str(run), '--device', device]},
            {'name': name + '/calibrate', 'marker': calibration,
             'provenance': {'kind': 'calibrate'},
             'command': prefix + ['calibrate', '--checkpoint', checkpoint, '--data', str(data_dir / 'calibration.jsonl'),
                        '--output', calibration, '--device', device]},
            {'name': name + '/evaluate', 'marker': str(run / 'evaluation' / 'metrics.json'),
             'provenance': {'kind': 'evaluate', 'calibration_data': str(data_dir / 'calibration.jsonl')},
             'command': prefix + ['evaluate', '--checkpoint', checkpoint, '--data', str(data_dir / 'benchmark.jsonl'),
                        '--calibration', calibration, '--output', str(run / 'evaluation'), '--device', device]},
        ])
    if baselines:
        for size, backbone, revision in BACKBONES:
            name = 'zero-shot-' + size
            run = output / name
            stages.append({'name': name + '/evaluate', 'marker': str(run / 'metrics.json'),
                           'provenance': {'kind': 'zero-shot'},
                           'command': prefix + ['zero-shot', '--data', str(data_dir / 'benchmark.jsonl'),
                                      '--calibration-data', str(data_dir / 'calibration.jsonl'), '--output', str(run),
                                      '--backbone', backbone, '--revision', revision, '--device', device]})
    return stages


def _json_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return value


def _argument(step, flag, default=None):
    command = step['command']
    positions = [i for i, value in enumerate(command) if value == flag]
    if not positions:
        return default
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f'Ambiguous or missing value for {flag}')
    return command[positions[0] + 1]


def _require_argument(step, flag):
    value = _argument(step, flag)
    if value is None:
        raise ValueError(f'Missing provenance input argument {flag}')
    return value


def _check_hash(record, field, path):
    from .data import file_hash
    if record.get(field) != file_hash(path):
        raise ValueError(f'Completed-stage provenance mismatch: {field} for {path}')


def _check_request_count(record, data_path, step):
    with Path(data_path).open() as handle:
        count = sum(bool(line.strip()) for line in handle)
    limit = _argument(step, '--limit')
    if limit is not None:
        if int(limit) < 1:
            raise ValueError('Evaluation limit must be positive')
        count = min(count, int(limit))
    if type(record.get('requests')) is not int or record['requests'] != count:
        raise ValueError('Completed-stage request count differs from the requested data selection')


def _check_temperature(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Completed-stage temperature must be finite and positive')


def validate_completed_stage(step):
    """Verify a built stage against its current inputs, without loading a model.

    Full TrainConfig defaults and CLI overrides are compared to the recorded
    effective training config. Generic manual stages without ``provenance`` retain
    their JSON-marker behavior and return False, explicitly indicating that their
    input provenance was not checked.
    """
    artifact = _json_object(step['marker'])
    provenance = step.get('provenance')
    if provenance is None:
        return False
    kind = provenance['kind']
    if kind == 'train':
        from .training import TrainConfig
        config = TrainConfig.from_dict(_json_object(_require_argument(step, '--config')))
        for key in ('device', 'seed', 'max_steps'):
            value = _argument(step, '--' + key.replace('_', '-'))
            if value is not None:
                setattr(config, key, value if key == 'device' else int(value))
        run = Path(_require_argument(step, '--output'))
        recorded = _json_object(run / 'config.json')
        if recorded.get('config') != asdict(config):
            raise ValueError('Completed-stage effective training configuration differs from the requested configuration')
        _check_hash(recorded, 'train_sha256', _require_argument(step, '--train'))
        _check_hash(recorded, 'dev_sha256', _require_argument(step, '--dev'))
        if not (run / 'best.pt').is_file() or (run / 'best.pt').stat().st_size == 0:
            raise FileNotFoundError('Completed training is missing its selected checkpoint')
        if (type(artifact.get('steps')) is not int or type(artifact.get('best_step')) is not int
                or not 0 <= artifact['best_step'] <= artifact['steps']):
            raise ValueError('Invalid training completion counters')
    elif kind in ('calibrate', 'evaluate'):
        checkpoint = _require_argument(step, '--checkpoint')
        data = _require_argument(step, '--data')
        _check_hash(artifact, 'checkpoint_sha256', checkpoint)
        _check_hash(artifact, 'data_sha256', data)
        _check_request_count(artifact, data, step)
        _check_temperature(artifact.get('temperature'))
        if kind == 'evaluate':
            calibration_path = _argument(step, '--calibration')
            temperature = 1.
            if calibration_path is not None:
                calibration = _json_object(calibration_path)
                _check_hash(calibration, 'checkpoint_sha256', checkpoint)
                _check_temperature(calibration.get('temperature'))
                if provenance.get('calibration_data') is not None:
                    _check_hash(calibration, 'data_sha256', provenance['calibration_data'])
                temperature = calibration['temperature']
            if artifact['temperature'] != temperature:
                raise ValueError('Completed evaluation uses a different calibration temperature')
            if not (Path(_require_argument(step, '--output')) / 'predictions.jsonl').is_file():
                raise FileNotFoundError('Completed evaluation is missing its predictions')
    elif kind == 'zero-shot':
        for field in ('backbone', 'revision'):
            if artifact.get(field) != _require_argument(step, '--' + field):
                raise ValueError(f'Completed baseline has a different {field}')
        data = _require_argument(step, '--data')
        _check_hash(artifact, 'data_sha256', data)
        _check_hash(artifact, 'calibration_sha256', _require_argument(step, '--calibration-data'))
        _check_request_count(artifact, data, step)
        if not (Path(_require_argument(step, '--output')) / 'predictions.jsonl').is_file():
            raise FileNotFoundError('Completed baseline is missing its predictions')
    else:
        raise ValueError(f'Unknown study-stage provenance kind: {kind}')
    return True


def run_steps(steps, status_path, deadline_utc=None, execute=subprocess.run):
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(deadline_utc) if deadline_utc else None
    if deadline and deadline.tzinfo is None:
        raise ValueError('Deadline must include a timezone')
    status = {'started_utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': deadline_utc, 'stages': []}

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n')
        temporary.replace(status_path)

    for step in steps:
        row = {**step}
        status['stages'].append(row)
        if Path(step['marker']).is_file():
            try:
                row['provenance_verified'] = validate_completed_stage(step)
            except Exception as error:
                row.update(status='provenance_mismatch', error_type=type(error).__name__)
                save()
                raise
            row['status'] = 'already_complete'
            save()
            continue
        if deadline and datetime.now(timezone.utc) >= deadline:
            row['status'] = 'not_started_deadline'
            save()
            continue
        row.update(status='running', started_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps({'stage': step['name'], 'status': 'running'}), flush=True)
        start = time.perf_counter()
        try:
            execute(step['command'], check=True)
            if not Path(step['marker']).is_file():
                raise RuntimeError('Stage exited without its completion artifact: ' + step['marker'])
            row['provenance_verified'] = validate_completed_stage(step)
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__, elapsed_seconds=time.perf_counter() - start)
            save()
            raise
        row.update(status='complete', elapsed_seconds=time.perf_counter() - start)
        save()
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/study-v1')
    parser.add_argument('--output', required=True)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--no-baselines', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--deadline-utc')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    stages = build_steps(args.data, args.output, args.variants, not args.no_baselines, args.device)
    if args.dry_run:
        print(json.dumps(stages, indent=2))
    else:
        run_steps(stages, Path(args.output) / 'execution-status.json', args.deadline_utc)


if __name__ == '__main__':
    main()
