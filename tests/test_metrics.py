import math

import pytest
import torch

from janus.metrics import distribution_loss, metrics, fit_temperature, paired_nll_bootstrap


def test_uniform_metrics_and_soft_target_losses():
    logits = [torch.zeros(2), torch.zeros(4)]
    targets = [torch.tensor([1., 0.]), torch.tensor([0., 0., 1., 0.])]
    report = metrics(logits, targets)
    assert report["nll"] == pytest.approx((math.log(2) + math.log(4)) / 2)
    assert report["brier"] == pytest.approx((.5 + .75) / 2)
    soft = [torch.tensor([.25, .75])]
    assert distribution_loss([torch.zeros(2)], soft).item() == pytest.approx(math.log(2))
    assert distribution_loss([torch.zeros(2)], soft, "brier").item() == pytest.approx(.125)
    exact = metrics([torch.tensor([1000., -1000.])], [targets[0]])
    assert exact["ece"] == 0
    assert sum(b["count"] for b in exact["reliability"]) == 1


def test_temperature_fits_on_detached_logits_and_can_only_improve_fit_nll():
    logits = [torch.tensor([8., 0.], requires_grad=True) for _ in range(4)]
    targets = [torch.tensor([1., 0.])] * 3 + [torch.tensor([0., 1.])]
    temperature = fit_temperature(logits, targets)
    assert temperature > 1
    before = metrics(logits, targets)["nll"]
    after = metrics(logits, targets, temperature)["nll"]
    assert after < before
    assert all(z.grad is None for z in logits)


def test_bootstrap_resamples_states_not_correlated_questions():
    rows = [{"group_id": "a", "nll": 1., "control_nll": 2.},
            {"group_id": "a", "nll": 1., "control_nll": 2.},
            {"group_id": "b", "nll": 1., "control_nll": 2.}]
    ci = paired_nll_bootstrap(rows, samples=100)
    assert ci["groups"] == 2
    assert ci["mean_improvement"] == 1
    assert ci["lower"] == ci["upper"] == 1


@pytest.mark.parametrize("bad", [[2., 0.], [float("nan"), 0.], [1.], [-.1, 1.1]])
def test_public_math_rejects_invalid_targets(bad):
    for function in (distribution_loss, metrics, fit_temperature):
        with pytest.raises(ValueError):
            function([torch.zeros(2)], [bad])


def test_calibration_rejects_mismatched_collections_and_nonfinite_logits():
    with pytest.raises(ValueError):
        fit_temperature([torch.zeros(2), torch.zeros(2)], [[1., 0.]])
    for function in (distribution_loss, metrics, fit_temperature):
        with pytest.raises(ValueError):
            function([torch.tensor([float("inf"), 0.])], [[1., 0.]])


def test_cluster_bootstrap_keeps_the_reported_decision_weighted_estimand():
    rows = ([{'group_id': 'a', 'nll': 0., 'control_nll': 1.}] * 6
            + [{'group_id': 'b', 'nll': 2., 'control_nll': 0.}] * 2)
    assert paired_nll_bootstrap(rows, samples=100)['mean_improvement'] == pytest.approx(.25)


def test_spherical_and_smoothing_losses():
    from janus.metrics import distribution_loss
    z = [torch.tensor([2., 0., 0.])]
    y = [torch.tensor([1., 0., 0.])]
    p = z[0].softmax(-1)
    assert distribution_loss(z, y, "spherical").item() == pytest.approx(float(-(p * y[0]).sum() / p.norm()))
    smoothed = distribution_loss(z, y, "ce", label_smoothing=.3)
    target = .7 * y[0] + .1
    assert smoothed.item() == pytest.approx(float(-(target * z[0].log_softmax(-1)).sum()))
    # Defaults are unchanged: no smoothing, cross-entropy.
    assert distribution_loss(z, y).item() == pytest.approx(float(-z[0].log_softmax(-1)[0]))
    with pytest.raises(ValueError):
        distribution_loss(z, y, "focal")
    with pytest.raises(ValueError):
        distribution_loss(z, y, "ce", label_smoothing=1.)


def test_per_cardinality_temperature_and_histogram_binning():
    from janus.metrics import apply_histogram_binning, fit_histogram_binning, fit_temperature_by_cardinality
    torch.manual_seed(0)
    logits, targets = [], []
    for _ in range(60):
        z = torch.randn(2) * 4
        logits.append(z)
        targets.append(torch.tensor([1., 0.]) if z[0] > z[1] else torch.tensor([0., 1.]))
    for _ in range(60):
        z = torch.randn(4) * .1
        logits.append(z)
        targets.append(torch.eye(4)[int(torch.randint(4, (1,)))])
    by_k = fit_temperature_by_cardinality(logits, targets)
    assert set(by_k) == {"2", "4"} and by_k["2"] < by_k["4"]
    table = fit_histogram_binning(logits, targets, bins=5)
    assert len(table) == 5 and all(t is None or 0 <= t <= 1 for t in table)
    out = apply_histogram_binning(torch.tensor([.9, .05, .05]), [.5, .5, .5, .5, .5])
    assert out.sum().item() == pytest.approx(1.) and out[0].item() == pytest.approx(.5) and out[1].item() == pytest.approx(.25)
    # An empty bin leaves the distribution untouched.
    assert torch.equal(apply_histogram_binning(torch.tensor([.9, .1]), [None] * 5), torch.tensor([.9, .1]))
