from janus.__main__ import main


def test_zero_shot_cli(monkeypatch, capsys):
    import janus.baseline
    calls = []
    monkeypatch.setattr(janus.baseline, 'evaluate_baseline', lambda *a, **k: calls.append((a, k)) or {'requests': 3})
    main(['zero-shot', '--data', 'panel', '--calibration-data', 'cal', '--output', 'out',
          '--backbone', 'model', '--revision', 'sha', '--device', 'cpu'])
    assert calls == [(('panel', 'cal', 'out', 'model', 'sha', 'cpu', 'float32'), {})]
    assert '3' in capsys.readouterr().out


def test_remote_cli_exposes_no_key_argument(monkeypatch):
    import janus.remote
    calls = []
    monkeypatch.setattr(janus.remote, 'evaluate_remote', lambda *a, **k: calls.append((a, k)) or {})
    main(['remote', '--data', 'panel', '--output', 'out', '--workers', '2', '--resume'])
    assert calls[0][0] == ('panel', 'out')
    assert calls[0][1]['workers'] == 2
    assert calls[0][1]['resume'] is True


def test_remote_calibration_cli_uses_separate_local_artifacts(monkeypatch):
    import janus.remote_calibration
    calls = []
    monkeypatch.setattr(janus.remote_calibration, 'calibrate_remote', lambda *a, **k:
                        calls.append((a, k)) or {'temperature': 2., 'requests': 8})
    main(['remote-calibrate', '--calibration-data', 'cal-data', '--calibration-predictions', 'cal-pred',
          '--data', 'panel', '--predictions', 'raw', '--output', 'scaled', '--artifact', 'meta'])
    assert calls == [(('cal-data', 'cal-pred', 'panel', 'raw', 'scaled', 'meta'), {})]


def test_profile_panel_cli(monkeypatch, tmp_path):
    import janus.benchmark
    calls = []
    monkeypatch.setattr(janus.benchmark, 'profile_panel', lambda *a, **k:
                        calls.append((a, k)) or {'requests': 3, 'p50_ms': 10.})
    main(['profile-panel', '--checkpoint', 'weights', '--data', 'panel', '--output', str(tmp_path / 'latency.json'),
          '--device', 'cpu', '--limit', '3'])
    assert calls == [(('weights', 'panel', 'cpu', 3), {})]
