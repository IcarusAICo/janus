"""WP2c neural RL arms on the tiny backbone (CPU only)."""

import json

import pytest
import torch

from janus.data import synthetic_requests, write_jsonl
from janus.estimators import ReportPolicy, report_grid
from janus.model import DecisionModel, ModelConfig
from janus.rl import (ARMS, NON_ARMS, REPORT_ARMS, RLConfig, ReportGridHead, ReportModel, load_rl_checkpoint,
                    report_metrics, rl_step, train_rl)
from janus.schema import Request
from janus.training import TrainConfig, load_checkpoint, train


def tiny_model(seed=17):
    torch.manual_seed(seed)
    config = ModelConfig(backbone="tiny", mode="tree", adaptation="full", dtype="float32", hidden_size=32, layers=2)
    return ReportModel(DecisionModel(config))


def noul_request(state, rate, name="q"):
    return Request.from_dict({"state": state, "questions": {
        name: {"type": "noul", "instructions": "Does the coin land heads?", "target": [1 - rate, rate]}}})


@pytest.fixture(scope="module")
def batch():
    torch.set_num_threads(2)
    return synthetic_requests(2, seed=3)


def test_report_head_grid_matches_estimators_and_starts_uniform():
    head = ReportGridHead(8)
    assert torch.equal(head.grid.double(), report_grid())
    logits = head(torch.randn(3, 8))
    assert logits.shape == (3, 101) and torch.all(logits == 0)


def test_report_model_matches_decision_model_logits(batch):
    model = tiny_model()
    packed = model.pack(batch[0])
    with torch.no_grad():
        logits, reports = model(packed)
        reference = model.model(packed)
    assert len(logits) == len(reference) == 3
    for z, ref in zip(logits, reference):
        assert torch.allclose(z, ref)
    kinds = [q.kind for q in batch[0].questions]
    for kind, r, q in zip(kinds, reports, batch[0].questions):
        assert r.shape == ((1, 101) if kind == "noul" else (len(q.options), 101))
    with pytest.raises(ValueError, match="tree"):
        ReportModel(DecisionModel(ModelConfig(backbone="tiny", mode="listwise", adaptation="full", hidden_size=32)))


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_runs_one_step_with_finite_loss(batch, arm):
    model = tiny_model()
    config = RLConfig(arm=arm, group=4, device="cpu")
    rng = torch.Generator().manual_seed(1)
    loss, info = rl_step(model, batch, config, rng)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    expected_questions = 2 if arm in REPORT_ARMS else 6
    assert info["labels_consumed"] == expected_questions
    assert info["reports_sampled"] == (0 if arm in ("S-CE", "S-Brier", "REPORT-exact") else 4 * expected_questions)
    assert (info["mean_reward"] is None) == (arm in ("S-CE", "S-Brier"))
    assert set(info["loss_terms"]) == {"policy", "auxiliary", "kl"} and info["encoded_tokens"] > 0
    assert (info["auxiliary_loss"] != 0) == (arm in ("SEP", "COUPLED"))


def test_kl_and_clip_paths_run(batch):
    model = tiny_model()
    import copy
    reference = copy.deepcopy(model).requires_grad_(False)
    for arm in ("REPORT-RLOO", "RLVR-cat"):
        loss, info = rl_step(model, batch, RLConfig(arm=arm, group=4, kl=.01, clip=.2), torch.Generator().manual_seed(2), reference)
        assert torch.isfinite(loss) and info["kl_loss"] >= 0
    with pytest.raises(ValueError, match="reference"):
        rl_step(model, batch, RLConfig(arm="REPORT-RLOO", kl=.01))


def test_report_exact_equals_enumerated_expectation():
    model = tiny_model()
    requests = [noul_request("Coin A.", 1.), noul_request("Coin B.", 0.)]
    # Perturb the report head so the policy is not uniform.
    with torch.no_grad():
        model.report_head.out.weight.normal_(0, .5)
        model.report_head.out.bias.normal_(0, .5)
    loss, info = rl_step(model, requests, RLConfig(arm="REPORT-exact", device="cpu"))
    grid = report_grid()
    expected = []
    with torch.no_grad():
        for request, label in zip(requests, (1., 0.)):
            _, reports = model(model.pack(request))
            pi = reports[0][0].double().softmax(-1)
            expected.append((pi * (grid - label).square()).sum())
    assert float(loss.detach()) == pytest.approx(float(torch.stack(expected).mean()), abs=1e-5)
    assert info["mean_reward"] == pytest.approx(-float(loss.detach()), abs=1e-5)
    assert info["outcomes"] == [1, 0] and info["labels_consumed"] == 2


def test_report_rloo_advantages_sum_to_zero_and_grpo_is_normalised():
    model = tiny_model()
    requests = [noul_request("Coin A.", .7), noul_request("Coin B.", .3)]
    _, info = rl_step(model, requests, RLConfig(arm="REPORT-RLOO", group=8), torch.Generator().manual_seed(5))
    assert len(info["advantages"]) == 2
    for advantages in info["advantages"]:
        assert advantages.shape == (8,) and float(advantages.sum()) == pytest.approx(0., abs=1e-6)
    _, info = rl_step(model, requests, RLConfig(arm="REPORT-GRPO", group=8), torch.Generator().manual_seed(5))
    for advantages in info["advantages"]:
        assert float(advantages.mean()) == pytest.approx(0., abs=1e-6)
        if float(advantages.abs().max()) > 0:
            assert float(advantages.square().mean()) == pytest.approx(1., abs=1e-3)


def test_non_arms_are_rejected_by_name():
    for name in NON_ARMS:
        with pytest.raises(ValueError, match=name):
            RLConfig(arm=name)
    with pytest.raises(ValueError, match="non-arm"):
        RLConfig(arm="DETACHED-Brier-reward")
    with pytest.raises(ValueError, match="arm must be one of"):
        RLConfig(arm="unknown")


def test_sep_masks_gradients_between_pathways(batch):
    model = tiny_model()
    with torch.no_grad():
        model.report_head.out.weight.normal_(0, .5)
    _, info = rl_step(model, batch, RLConfig(arm="SEP", group=8), torch.Generator().manual_seed(11))
    report_params = list(model.report_head.parameters())
    decision_params = list(model.model.head.parameters())

    def grads(term):
        model.zero_grad(set_to_none=True)
        term.backward(retain_graph=True)
        return ([p.grad.clone() if p.grad is not None else None for p in report_params],
                [p.grad.clone() if p.grad is not None else None for p in decision_params])

    report_from_correctness, decision_from_correctness = grads(info["loss_terms"]["policy"])
    assert all(g is None or torch.all(g == 0) for g in report_from_correctness)
    assert any(g is not None and g.abs().sum() > 0 for g in decision_from_correctness)
    report_from_brier, decision_from_brier = grads(info["loss_terms"]["auxiliary"])
    assert all(g is None or torch.all(g == 0) for g in decision_from_brier)
    assert any(g is not None and g.abs().sum() > 0 for g in report_from_brier)


def test_report_exact_moves_mean_report_towards_empirical_rate():
    model = tiny_model(seed=3)
    rate = .8
    requests = [noul_request("A coin with a fixed bias.", rate, f"q{i}") for i in range(4)]
    config = RLConfig(arm="REPORT-exact", device="cpu")
    rng = torch.Generator().manual_seed(7)
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    initial = report_metrics(_rows(model, requests), model.grid)["mean_report"]
    assert initial == pytest.approx(.5, abs=1e-6)
    outcomes = []
    for _ in range(40):
        optimizer.zero_grad()
        loss, info = rl_step(model, requests, config, rng)
        loss.backward()
        optimizer.step()
        outcomes += info["outcomes"]
    empirical = sum(outcomes) / len(outcomes)
    final = report_metrics(_rows(model, requests), model.grid)
    assert abs(empirical - rate) < .12
    assert abs(final["mean_report"] - empirical) < abs(initial - empirical)
    assert abs(final["mean_report"] - empirical) < .15


def _rows(model, requests):
    from janus.rl import collect_outputs
    return collect_outputs(model, requests)


def test_report_metrics_match_the_estimator_decomposition():
    grid = report_grid()
    policy = ReportPolicy(contexts=1, grid=grid)
    with torch.no_grad():
        policy.logits.zero_()
        policy.logits[0, 20] = 3.
        policy.logits[0, 60] = 3.
    rates = torch.tensor([.4], dtype=torch.float64)
    reference = policy.evaluate(rates)
    rows = [{"kind": "noul", "logits": torch.zeros(2), "target": torch.tensor([.6, .4]),
             "reports": policy.logits.detach().float()}]
    out = report_metrics(rows, grid.float())
    assert out["expected_brier"] == pytest.approx(float(reference["expected_brier"][0]), abs=1e-5)
    assert out["policy_mean_brier"] == pytest.approx(float(reference["policy_mean_brier"][0]), abs=1e-5)
    assert out["report_variance"] == pytest.approx(float(reference["report_variance"][0]), abs=1e-5)
    assert out["mean_report"] == pytest.approx(float(reference["mean_report"][0]), abs=1e-5)
    assert {"nll", "ece", "brier"} <= set(out["policy_mean"])


@pytest.mark.parametrize("arm", ("REPORT-RLOO", "SEP"))
def test_train_rl_from_a_tiny_checkpoint(tmp_path, arm):
    torch.set_num_threads(2)
    paths = {}
    for name, seed in (("train", 17), ("dev", 23)):
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], [r.to_dict() for r in synthetic_requests(4, seed=seed)])
    base_config = ModelConfig(backbone="tiny", mode="tree", adaptation="full", dtype="float32", hidden_size=32, layers=2)
    train(paths["train"], paths["dev"], tmp_path / "base",
          TrainConfig(model=base_config, epochs=1, accumulation=2, max_steps=1, device="cpu"))
    config = RLConfig(arm=arm, group=4, device="cpu", accumulation=2, max_steps=2, eval_every=1, head_lr=.01,
                      backbone_lr=.001, train_limit=3)
    result = train_rl(tmp_path / "base" / "best.pt", paths["train"], paths["dev"], tmp_path / "run", config)
    assert result["steps"] == 2 and result["labels_consumed"] > 0 and result["reports_sampled"] > 0
    assert result["arm"] == arm and result["train_requests"] == 3
    assert result["selection_metric"] == ("noul_policy_mean_nll" if arm in REPORT_ARMS else "dev_nll")
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["encoded_tokens"] > 0 and summary["updates"] == 2
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert len(history) == 3 and "policy_loss" in history[-1] and "auxiliary_loss" in history[-1]
    # best.pt is an ordinary checkpoint for jev evaluate, and carries the report head for janus.rl.
    plain, metadata = load_checkpoint(tmp_path / "run" / "best.pt")
    assert plain.config.mode == "tree" and metadata["arm"] == arm
    reloaded, _ = load_rl_checkpoint(tmp_path / "run" / "best.pt")
    packed = reloaded.pack(synthetic_requests(1, seed=99)[0])
    with torch.no_grad():
        logits, reports = reloaded(packed)
    assert len(logits) == 3 and reports[1].shape == (1, 101)
