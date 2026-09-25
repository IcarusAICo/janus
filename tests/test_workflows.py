import itertools
import json
import random

import pytest
import torch

from janus.workflows.env import ACTIONS, Grid, brute_force_success, budgeted_grid, distance, episode_request, random_grid, run_episode, success_within


def test_bfs_labels_agree_with_brute_force_rollouts():
    rng = random.Random(3)
    checked = 0
    for _ in range(25):
        grid = random_grid(rng, size=6, walls=5, hazards=4, max_distance=4)
        for action, steps in itertools.product(ACTIONS, range(1, 5)):
            assert success_within(grid, action, steps) == brute_force_success(grid, action, steps)
            checked += 1
    assert checked == 25 * len(ACTIONS) * 4


def test_grid_json_state_round_trips():
    grid = random_grid(random.Random(5))
    again = Grid.from_json(grid.to_json())
    assert again == grid and json.loads(grid.to_json()) == grid.to_state()
    assert Grid.from_state(json.loads(json.dumps(grid.to_state()))) == grid

from janus.workflows.programs import WORKFLOWS, World, generate_worlds, load_worlds, prepare_workflow_data, write_worlds


def corruptions(workflow, world):
    """Every single-question alternative answer, as full answer dicts."""
    request = workflow.request(world)
    for q in request.questions:
        for option in q.options:
            if option.key != world.gold[q.id]:
                yield {**world.gold, q.id: option.key}


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=[w.name for w in WORKFLOWS])
def test_program_reproduces_gold_outcome_and_verifier_is_one_on_gold(workflow):
    rng = random.Random(11)
    for _ in range(20):
        world = workflow.sample(rng)
        request = workflow.request(world)
        assert [q.id for q in request.questions] == list(world.gold)
        for q in request.questions:
            assert world.gold[q.id] in {o.key for o in q.options}
            assert q.target is not None and q.target[[o.key for o in q.options].index(world.gold[q.id])] > 0
        assert workflow.program(world.state, world.gold) == world.outcome
        assert workflow.verify(world, world.outcome) == 1.
        assert workflow.questions(world.state).group_id == request.group_id


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=[w.name for w in WORKFLOWS])
def test_verifier_is_lower_on_corrupted_answers(workflow):
    rng = random.Random(12)
    rewards, lowered_worlds = [], 0
    for _ in range(20):
        world = workflow.sample(rng)
        scores = [workflow.verify(world, workflow.program(world.state, answers)) for answers in corruptions(workflow, world)]
        assert scores and all(0. <= s <= 1. for s in scores)
        rewards += scores
        lowered_worlds += min(scores) < 1.
    assert sum(rewards) / len(rewards) < .9
    # The game's Noul overrides the action Choice, so corrupting the action is harmless whenever the proposed
    # action succeeds; only about half of its worlds have a reward-lowering single corruption.
    assert lowered_worlds >= (8 if workflow.name == "game" else 15)


def test_world_files_round_trip_and_are_disjoint(tmp_path):
    worlds = generate_worlds(10, seed=4)
    assert [w.workflow for w in worlds[:5]] == [w.name for w in WORKFLOWS]
    write_worlds(tmp_path / "w.jsonl", worlds)
    again = load_worlds(tmp_path / "w.jsonl")
    assert again == worlds
    assert all(World.from_dict(w.to_dict()) == w for w in worlds)
    manifest = prepare_workflow_data(tmp_path / "data", train=10, heldout=5, seed=4)
    assert manifest["counts"] == {"train": 10, "heldout": 5}

from janus.model import DecisionModel, ModelConfig
from janus.packing import pack_request
from janus.workflows.rl import (WorkflowRLConfig, evaluate_env, evaluate_workflows, group_advantages, train_env_rl,
                              train_workflow_rl, two_path_logits)


def tiny_model(seed=17):
    torch.manual_seed(seed)
    return DecisionModel(ModelConfig(backbone="tiny", mode="tree", adaptation="full", dtype="float32",
                                     hidden_size=32, layers=1, head_rank=16, max_tokens=4096))


def test_group_advantages_sum_to_zero_per_group():
    rewards = torch.tensor([[1., 0., .5, 1.], [0., 0., 1., .5], [1., 1., 1., 1.]])
    for estimator in ("rloo", "group_mean", "grpo"):
        advantages = group_advantages(rewards, estimator)
        assert advantages.shape == rewards.shape
        torch.testing.assert_close(advantages.sum(-1), torch.zeros(3), atol=1e-6, rtol=0)
    rloo = group_advantages(rewards[0], "rloo")
    torch.testing.assert_close(rloo, rewards[0] - (rewards[0].sum() - rewards[0]) / 3)
    grpo = group_advantages(rewards[:2], "grpo")
    torch.testing.assert_close(grpo.std(-1, unbiased=False), torch.ones(2), atol=1e-4, rtol=0)
    with pytest.raises(ValueError):
        group_advantages(torch.tensor([1.]), "rloo")


def test_two_path_logits_match_forward_and_mask_stops_backbone_gradient():
    torch.set_num_threads(2)
    model = tiny_model()
    workflow = WORKFLOWS[2]
    request = workflow.request(workflow.sample(random.Random(1)))
    model.eval()
    policy, report = two_path_logits(model, request, mask=True)
    reference = model(pack_request(request, model.tokenizer, "tree", 4096))
    for a, b, c in zip(policy, report, reference):
        torch.testing.assert_close(a, c)
        torch.testing.assert_close(b, c)
    from janus.metrics import distribution_loss
    model.zero_grad()
    distribution_loss(report, [q.target for q in request.questions], "brier").backward()
    assert all(p.grad is None or not p.grad.any() for p in model.backbone.parameters())
    assert any(p.grad is not None and p.grad.any() for p in model.head.parameters())
    model.zero_grad()
    distribution_loss(policy, [q.target for q in request.questions], "brier").backward()
    assert any(p.grad is not None and p.grad.any() for p in model.backbone.parameters())


def test_workflow_rl_step_runs_and_fixed_batch_improves():
    torch.set_num_threads(2)
    model = tiny_model()
    worlds = [WORKFLOWS[2].sample(random.Random(7)), WORKFLOWS[3].sample(random.Random(8))]
    config = WorkflowRLConfig(group=8, states_per_step=2, steps=30, backbone_lr=3e-3, head_lr=3e-3, log_every=100, threads=2)
    history = train_workflow_rl(model, WORKFLOWS, config, worlds=worlds)
    assert len(history) == 30
    first, last = history[0], history[-1]
    for key in ("reward_mean", "greedy_reward_mean", "policy_loss", "proper_loss", "loss", "accuracy", "rewards_evaluated", "labels_used"):
        assert key in first
    assert first["rewards_evaluated"] == 16 and last["rewards_evaluated"] == 30 * 16 and last["labels_used"] == 30 * 5
    # The sampled workflow reward is the trained quantity; the masked proper score reaches only the head.
    assert sum(h["reward_mean"] for h in history[-5:]) > sum(h["reward_mean"] for h in history[:5])
    summary = evaluate_workflows(model, worlds)
    assert set(summary["per_workflow"]) == {"moderation", "dom"} and 0. <= summary["greedy_reward_mean"] <= 1.
    supervised = train_workflow_rl(tiny_model(), WORKFLOWS, WorkflowRLConfig(arm="supervised", group=2, states_per_step=2, steps=30,
                                                                             backbone_lr=3e-3, head_lr=3e-3, log_every=100, threads=2), worlds=worlds)
    assert supervised[-1]["proper_loss"] < supervised[0]["proper_loss"] and supervised[-1]["loss"] < supervised[0]["loss"]
    assert supervised[-1]["rewards_evaluated"] == 0 and supervised[-1]["labels_used"] == 30 * 5  # the supervised control consumes no rewards
    only = train_workflow_rl(tiny_model(), WORKFLOWS, WorkflowRLConfig(arm="rl_only", group=2, states_per_step=1, steps=1, threads=2), worlds=worlds[:1])
    # Workflow-reward-only: the proper score is still recorded as a diagnostic but consumes no labels and no gradient.
    assert only[0]["proper_loss"] is not None and only[0]["labels_used"] == 0 and only[0]["loss"] == only[0]["policy_loss"]


def test_env_rl_step_runs():
    torch.set_num_threads(2)
    model = tiny_model()
    starts = [budgeted_grid(random.Random(9), max_distance=3, size=6, walls=4, hazards=3)]
    config = WorkflowRLConfig(group=2, states_per_step=1, steps=2, log_every=100, threads=2)
    history = train_env_rl(model, config, starts=starts)
    assert len(history) == 2
    for key in ("return_mean", "success_rate", "action_accuracy", "noul_brier_bfs", "noul_brier_realised", "policy_loss", "proper_loss"):
        assert key in history[0]
    assert history[0]["rewards_evaluated"] == 2
    realised = train_env_rl(tiny_model(), WorkflowRLConfig(group=2, states_per_step=1, steps=1, noul_target="realised", threads=2), starts=starts)
    assert realised[0]["labels_used"] <= history[0]["labels_used"]
    summary = evaluate_env(model, starts)
    assert 0. <= summary["success_rate"] <= 1. and "noul_brier_bfs" in summary
