import json
from pathlib import Path


def _metrics(acc, nll, tv=.01, flips=1):
    return {"raw": {"accuracy": acc, "nll": nll, "brier": .2, "ece": .05, "count": 10},
            "calibrated": {"accuracy": acc, "nll": nll - .01, "brier": .19, "ece": .04, "count": 10},
            "by_family": {"rel": {"raw": {"accuracy": acc, "nll": nll, "brier": .2, "ece": .05, "count": 5}}},
            "choice_order": {"count": 5, "mean_total_variation": tv, "flip_rate": flips / 5}}


def test_report_tabulates_three_graphs(tmp_path):
    from janus.phase1_report import report
    for g, acc in (("g0", .8), ("g2", .82), ("g4", .81)):
        d = tmp_path / g
        for split in ("panel", "test", "test_post_unseen", "k16"):
            (d / split).mkdir(parents=True)
            (d / split / "metrics.json").write_text(json.dumps(_metrics(acc, .5)))
        (d / "test_post_unseen" / "predictions.jsonl").write_text(json.dumps({"probabilities": [.34, .33, .33]}) + "\n")
        (d / "summary.json").write_text(json.dumps({"elapsed_seconds": 600, "best_step": 100}))
        (d / "panel-latency.json").write_text(json.dumps({"p50_ms": 14., "p95_ms": 18.}))
    text = report(tmp_path)
    assert "| g2 |" in text and "0.8200" in text and "entropy" in text and "rel" in text


def test_report_discovers_graphs_and_shows_compute_accounting_when_present(tmp_path):
    from janus.phase1_report import discover_graphs, report
    for g, compute in (("g2", None), ("g5d2", {"backbone_tokens": 900., "decoder_cross_attention_pairs": 480000.})):
        d = tmp_path / g
        (d / "panel").mkdir(parents=True)
        (d / "panel" / "metrics.json").write_text(json.dumps(_metrics(.8, .5)))
        (d / "summary.json").write_text(json.dumps({"elapsed_seconds": 600, "best_step": 100}))
        latency = {"p50_ms": 14., "p95_ms": 18., **({"compute": compute} if compute else {})}
        (d / "panel-latency.json").write_text(json.dumps(latency))
    (tmp_path / "report.md").write_text("not a run")
    (tmp_path / "empty").mkdir()
    assert discover_graphs(tmp_path) == ("g2", "g5d2")
    text = report(tmp_path)
    assert "mean backbone tokens" in text and "mean decoder cross attention pairs" in text
    assert "| g5d2 | 10.0 | 100 | 14.00 | 18.00 | 900 | 480000 |" in text
    assert "| g2 | 10.0 | 100 | 14.00 | 18.00 | n/a | n/a |" in text
