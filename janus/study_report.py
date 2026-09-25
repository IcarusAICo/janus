"""Assemble completed study artifacts without loading a model or calling an API."""

import argparse
import hashlib
import json
from pathlib import Path


TRAINED = {
    'Listwise 0.6B': 'listwise-06b-seed17',
    'Independent 0.6B': 'independent-06b-seed17',
    'Frozen 0.6B': 'frozen-06b-seed17',
    'Small-data 0.6B': 'small-data-06b-seed17',
    'Listwise 1.7B': 'listwise-17b-seed17',
}
BASELINES = {'Zero-shot 0.6B': 'zero-shot-06b', 'Zero-shot 1.7B': 'zero-shot-17b'}
OPTIONAL_TRAINED = {'Listwise 0.6B seed23': 'listwise-06b-seed23'}
REFERENCE = 'Jev-1.13.0'


def validate_panel(predictions_path, data_path):
    from .data import load_requests
    expected = {}
    for request in load_requests(data_path):
        for q in request.questions:
            signature = {'state': request.state, 'kind': q.kind, 'instructions': q.instructions,
                         'options': [(o.key, o.description) for o in q.options]}
            expected[(request.group_id, q.id)] = {
                'input_sha256': hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                'kind': q.kind, 'keys': [o.key for o in q.options], 'target': list(q.target)}
    observed = set()
    for line in Path(predictions_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row['group_id'], row['question_id'])
        if (key in observed or key not in expected
                or any(row.get(field) != value for field, value in expected[key].items())):
            raise ValueError('Predictions do not match the audited benchmark inputs and targets')
        observed.add(key)
    if observed != set(expected):
        raise ValueError('Predictions do not cover the complete audited benchmark')


def discover_predictions(root, partial=False):
    root = Path(root)
    expected = {name: root / run / 'evaluation' / 'predictions.jsonl' for name, run in TRAINED.items()}
    expected.update({name: root / run / 'predictions.jsonl' for name, run in BASELINES.items()})
    expected[REFERENCE] = root / 'jev' / 'calibrated-predictions.jsonl'
    missing = {name: path for name, path in expected.items() if not path.is_file()}
    if missing and not partial:
        raise FileNotFoundError('Missing study results: ' + ', '.join(missing))
    found = {name: path for name, path in expected.items() if path.is_file()}
    found.update({name: root / run / 'evaluation' / 'predictions.jsonl'
                  for name, run in OPTIONAL_TRAINED.items()
                  if (root / run / 'evaluation' / 'predictions.jsonl').is_file()})
    if REFERENCE not in found:
        raise FileNotFoundError('Missing calibrated Jev reference')
    return found


def partition_predictions(paths, labels, partition, output):
    from .data import write_jsonl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for index, (name, path) in enumerate(paths.items()):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        rows = [r for r in rows if labels[r['group_id']][r['question_id']]['intent_partition'] == partition]
        if not rows:
            raise ValueError('No observations in requested intent partition')
        target = output / f'model-{index}.jsonl'
        write_jsonl(target, rows)
        result[name] = target
    return result


def primary_comparisons(paths, samples=2000):
    from .study_analysis import analyze_study
    primary = 'Listwise 0.6B'
    local = {name: path for name, path in paths.items() if name != REFERENCE}
    if primary not in local or len(local) < 2:
        return None
    return analyze_study(local, reference=primary, bootstrap_samples=samples)


def assemble_report(root, data_dir='data/study-v1', partial=False, samples=2000, plots=True, output=None):
    from .data import file_hash, write_json
    from .exposure import audit_exposure
    from .study_analysis import analyze_study, render_markdown
    from .study_audit import audit_study
    root, data_dir = Path(root), Path(data_dir)
    paths = discover_predictions(root, partial)
    output = Path(output) if output else root / ('analysis-partial' if partial else 'analysis')
    output.mkdir(parents=True, exist_ok=False)
    audit = audit_study(data_dir, output / 'data-audit.json')
    if not audit['checks']['passed']:
        raise ValueError('Dataset integrity audit failed')
    validate_panel(paths[REFERENCE], data_dir / 'benchmark.jsonl')
    report = analyze_study(paths, reference=REFERENCE, bootstrap_samples=samples)
    contrasts = primary_comparisons(paths, samples)
    report['comparisons_to_primary'] = contrasts['comparisons'] if contrasts else {}
    report['primary_contrast_definition'] = 'Named local model minus Listwise 0.6B; positive accuracy and negative NLL favor that named model'
    if contrasts:
        write_json(output / 'primary-contrasts.json', contrasts)
        (output / 'primary-contrasts.md').write_text(render_markdown(contrasts))
    report['partial'] = partial
    report['training_exposure'] = {}
    report['calibration'] = {}
    calibration_hash = file_hash(data_dir / 'calibration.jsonl')
    report['controls'] = {}
    for name, run in {**TRAINED, **OPTIONAL_TRAINED}.items():
        if name not in paths:
            continue
        report['training_exposure'][name] = audit_exposure(root / run, data_dir / 'train.jsonl',
                                                         output / (run + '-exposure.json'))
        calibration = json.loads((root / run / 'calibration.json').read_text())
        if calibration['data_sha256'] != calibration_hash:
            raise ValueError('Local temperature was fitted on a different calibration dataset')
        report['calibration'][name] = {k: calibration[k] for k in ('temperature', 'requests', 'before', 'after')}
        evaluation = json.loads((root / run / 'evaluation' / 'metrics.json').read_text())
        report['controls'][name] = {k: evaluation[k] for k in
                                    ('uniform', 'shuffled_state', 'choice_order',
                                     'uniform_comparison', 'shuffled_state_comparison')}
    for name, run in BASELINES.items():
        if name in paths:
            baseline = json.loads((root / run / 'metrics.json').read_text())
            if baseline['calibration_sha256'] != calibration_hash:
                raise ValueError('Baseline temperature was fitted on a different calibration dataset')
            report['calibration'][name] = {'temperature': baseline['temperature'],
                                          'requests': audit['splits']['calibration']['states']}
    remote_calibration = json.loads((root / 'jev' / 'calibration.json').read_text())
    if (remote_calibration['calibration_data_sha256'] != calibration_hash
            or remote_calibration['panel_data_sha256'] != file_hash(data_dir / 'benchmark.jsonl')
            or remote_calibration['calibrated_predictions_sha256'] != file_hash(paths[REFERENCE])):
        raise ValueError('Jev calibration provenance does not match the study inputs')
    report['calibration'][REFERENCE] = {k: remote_calibration[k] for k in
        ('temperature', 'requests', 'before', 'after', 'calibration_data_sha256',
         'calibration_predictions_sha256', 'calibrated_predictions_sha256')}
    for partition in ('seen', 'unseen'):
        selected = partition_predictions(paths, audit['benchmark_question_labels'], partition,
                                          output / 'intent-partitions' / partition)
        subset_report = analyze_study(selected, reference=REFERENCE, bootstrap_samples=samples)
        write_json(output / f'intent-{partition}.json', subset_report)
        (output / f'intent-{partition}.md').write_text(render_markdown(subset_report))
    write_json(output / 'report.json', report)
    (output / 'metrics.md').write_text(render_markdown(report))
    if plots:
        from .study_plots import plot_study
        selected = [n for n in ('Listwise 0.6B', 'Listwise 1.7B', REFERENCE) if n in paths]
        plot_study(report, output / 'figures', reliability_models=selected)
    return {'output': str(output), 'models': list(paths), 'partial': partial,
            'decisions': report['count'], 'groups': report['groups']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--data', default='data/study-v1')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--output', help='Fresh report directory; existing output is never overwritten')
    args = parser.parse_args()
    print(json.dumps(assemble_report(args.root, args.data, args.partial, args.samples, not args.no_plots, args.output), indent=2))


if __name__ == '__main__':
    main()
