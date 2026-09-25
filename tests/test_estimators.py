import math

import pytest
import torch

from janus.estimators import (
    EstimatorConfig,
    KnownRates,
    ReportPolicy,
    brier_reward,
    estimator_loss,
    log_reward,
    report_grid,
    run_study,
    train_policy,
)


def test_report_grid_has_101_values_and_interior_has_99():
    full = report_grid()
    assert full.shape == (101,) and full.dtype == torch.float64
    assert full[0] == 0 and full[-1] == 1 and torch.allclose(full[1:] - full[:-1], torch.full((100,), .01, dtype=torch.float64))
    interior = report_grid(interior=True)
    assert interior.shape == (99,) and interior[0] == pytest.approx(.01) and interior[-1] == pytest.approx(.99)


def test_brier_reward_is_maximised_at_the_true_rate_in_expectation():
    grid = report_grid()
    eta = .3
    expected = eta * brier_reward(grid, torch.ones_like(grid)) + (1 - eta) * brier_reward(grid, torch.zeros_like(grid))
    assert grid[expected.argmax()] == pytest.approx(.3)


def test_log_reward_rejects_endpoint_reports():
    with pytest.raises(ValueError, match="interior"):
        log_reward(torch.tensor([0., .5], dtype=torch.float64), torch.tensor([1., 1.], dtype=torch.float64))
    value = log_reward(torch.tensor([.25], dtype=torch.float64), torch.tensor([0.], dtype=torch.float64))
    assert value.item() == pytest.approx(math.log(.75))


def test_known_rates_resample_outcomes_each_call():
    rates = KnownRates((.1, .9))
    g = torch.Generator().manual_seed(1)
    c1, y1 = rates.sample(1000, g)
    c2, y2 = rates.sample(1000, g)
    assert c1.shape == (1000,) and y1.dtype == torch.float64
    assert not torch.equal(y1, y2)
    mean_high = y1[c1 == 1].mean().item()
    assert .85 < mean_high < .95


def test_policy_evaluation_matches_the_decomposition():
    grid = report_grid()
    policy = ReportPolicy(contexts=1, grid=grid)
    with torch.no_grad():
        policy.logits.zero_()
        policy.logits[0, 20] = 3.  # favour report .20
        policy.logits[0, 60] = 3.  # and report .60
    rates = torch.tensor([.4], dtype=torch.float64)
    out = policy.evaluate(rates)
    p = policy.probs(torch.tensor([0]))[0]
    mean = (p * grid).sum()
    var = (p * (grid - mean).square()).sum()
    manual = (mean - .4) ** 2 + var + .4 * .6
    assert out["expected_brier"][0].item() == pytest.approx(manual.item())
    assert out["mean_report"][0].item() == pytest.approx(mean.item())
    assert out["policy_mean_brier"][0].item() == pytest.approx(((mean - .4) ** 2 + .4 * .6).item())
    assert out["forecast_error"][0].item() == pytest.approx(abs(mean.item() - .4))


def test_policy_sampling_returns_grid_indices_with_group_dimension():
    policy = ReportPolicy(contexts=3, grid=report_grid())
    g = torch.Generator().manual_seed(0)
    idx = policy.sample(torch.tensor([0, 1, 2, 2]), group=8, rng=g)
    assert idx.shape == (4, 8) and idx.dtype == torch.long and idx.min() >= 0 and idx.max() <= 100


def _policy_and_batch():
    policy = ReportPolicy(contexts=2, grid=report_grid())
    g = torch.Generator().manual_seed(3)
    rates = KnownRates((.2, .8))
    contexts, outcomes = rates.sample(64, g)
    return policy, contexts, outcomes, torch.tensor(rates.rates, dtype=torch.float64), g


def test_exact_loss_is_negative_expected_reward_and_consumes_batch_labels():
    policy, contexts, outcomes, rates, g = _policy_and_batch()
    loss, info = estimator_loss(policy, contexts, outcomes, rates, EstimatorConfig("exact"), g)
    p = policy.probs(contexts)
    manual = -(p * brier_reward(report_grid()[None], outcomes[:, None])).sum(-1).mean()
    assert loss.item() == pytest.approx(manual.item())
    assert info["labels_consumed"] == 64


def test_rloo_advantages_sum_to_zero_and_grpo_has_unit_scale():
    policy, contexts, outcomes, rates, g = _policy_and_batch()
    _, rloo = estimator_loss(policy, contexts, outcomes, rates, EstimatorConfig("rloo", group=8), g)
    _, grpo = estimator_loss(policy, contexts, outcomes, rates, EstimatorConfig("grpo", group=8), g)
    assert abs(rloo["advantages"].sum(-1)).max().item() < 1e-9
    assert grpo["advantages"].std(-1, unbiased=False).mean().item() == pytest.approx(1., abs=.05)
    assert rloo["labels_consumed"] == 64 and grpo["labels_consumed"] == 64


def test_independent_outcomes_consume_group_times_batch_labels():
    policy, contexts, _, rates, g = _policy_and_batch()
    outcomes = (torch.rand(64, 8, generator=g, dtype=torch.float64) < rates[contexts][:, None]).double()
    _, info = estimator_loss(policy, contexts, outcomes, rates, EstimatorConfig("rloo", group=8, outcome_design="independent"), g)
    assert info["labels_consumed"] == 512


def test_sampled_loss_gradient_has_zero_mean_for_constant_reward():
    # A detached constant reward for every sampled report must give a zero expected score-function gradient.
    policy = ReportPolicy(contexts=1, grid=report_grid())
    g = torch.Generator().manual_seed(5)
    contexts = torch.zeros(4096, dtype=torch.long)
    idx = policy.sample(contexts, 1, g)
    logp = policy.log_probs(contexts).gather(1, idx)
    loss = -(torch.full_like(logp, -.25) * logp).mean()
    loss.backward()
    assert policy.logits.grad.abs().max().item() < 5e-3


def test_clip_and_kl_are_off_by_default():
    config = EstimatorConfig("rloo")
    assert config.kl == 0. and config.clip is None and config.outcome_design == "shared"


def test_exact_estimator_recovers_known_rates_within_grid_resolution():
    result = train_policy(KnownRates((.1, .3, .5, .7, .9)), EstimatorConfig("exact"), steps=1500, batch=256, lr=.1, seed=0)
    assert max(result["final"]["forecast_error"]) < .02
    assert result["labels_consumed"] == 1500 * 256


def test_rloo_recovers_known_rates_within_sampling_error():
    result = train_policy(KnownRates((.2, .8)), EstimatorConfig("rloo", group=16), steps=1500, batch=256, lr=.1, seed=0)
    assert max(result["final"]["forecast_error"]) < .06


def test_study_writes_json_and_markdown(tmp_path):
    report = run_study(tmp_path / "study", seeds=1, steps=50)
    assert (tmp_path / "study" / "results.json").exists() and (tmp_path / "study" / "results.md").exists()
    names = {row["estimator"] for row in report["rows"]}
    assert names == {"exact", "rloo", "group_mean", "grpo"}
    assert all("forecast_error_max_mean" in row for row in report["rows"])
