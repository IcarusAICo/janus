"""Policy gradient through programs and environments (spec WP4).

What is sampled: for each state, one forward pass gives the decision distributions, and each of G group
members draws one option per question at the sampling temperature. What is rewarded: the verifier score
of the program run on the sampled tuple (WP4a) or the episode return (WP4b), which see only the sampled
tuple. What is differentiated: the sampled options' log-probabilities weighted by a group advantage
(score-function with the RLOO, group-mean or GRPO baseline of `janus.estimators`), plus a proper score on
the full distribution of each question that has a gold target, differentiated directly and never used as
a reward. The proper score acts on the same decision logits that evaluation scores; with
`mask_proper_from_policy` its gradient stops at the backbone and reaches only the decision head, so the
reward alone shapes the backbone while the proper score adjusts the head's affine readout. The COUPLED
objective of the WP2 table is never built: no per-question correctness reward, no proper score in the
advantage."""

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import time

import torch

from ..data import state_hash, write_json
from ..metrics import distribution_loss
from ..packing import pack_request
from ..training import checkpoint, seed_everything
from .env import ACTIONS, budgeted_grid, episode_request, run_episode
from .programs import WORKFLOW_BY_NAME, WORKFLOWS

ARMS = ("rl", "rl_only", "supervised")
ESTIMATORS = ("rloo", "group_mean", "grpo")


@dataclass
class WorkflowRLConfig:
    arm: str = "rl"
    estimator: str = "rloo"
    group: int = 8
    states_per_step: int = 4
    steps: int | None = None
    epochs: int = 1
    temperature: float = 1.
    proper_score: str = "brier"
    proper_weight: float = 1.
    mask_proper_from_policy: bool = True
    backbone_lr: float = 2e-4
    head_lr: float = 1e-3
    weight_decay: float = .01
    seed: int = 17
    device: str = "cpu"
    threads: int = 4
    log_every: int = 10
    env_max_distance: int = 6
    noul_target: str = "bfs"

    def validate(self):
        if self.arm not in ARMS or self.estimator not in ESTIMATORS:
            raise ValueError(f"arm must be one of {ARMS} and estimator one of {ESTIMATORS}")
        if self.proper_score not in {"brier", "ce"} or self.noul_target not in {"bfs", "realised"}:
            raise ValueError("proper_score must be brier or ce; noul_target must be bfs or realised")
        if self.group < 2 or self.states_per_step < 1 or self.epochs < 1 or (self.steps is not None and self.steps < 1):
            raise ValueError("group must be at least 2; states_per_step, epochs and steps must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0 or self.proper_weight < 0:
            raise ValueError("temperature must be positive and proper_weight nonnegative")
        return self

    @property
    def weights(self):
        """(policy weight, proper-score weight, mask) for the arm."""
        if self.arm == "rl":
            return 1., self.proper_weight, self.mask_proper_from_policy
        if self.arm == "rl_only":
            return 1., 0., self.mask_proper_from_policy
        return 0., self.proper_weight, False


def group_advantages(rewards, estimator):
    """Advantages over the last axis, exactly as `janus.estimators.estimator_loss` forms them."""
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    size = rewards.shape[-1]
    if size < 2:
        raise ValueError("Group baselines need at least two samples per group")
    if estimator == "rloo":
        return rewards - (rewards.sum(-1, keepdim=True) - rewards) / (size - 1)
    centred = rewards - rewards.mean(-1, keepdim=True)
    if estimator == "grpo":
        centred = centred / (rewards.std(-1, unbiased=False, keepdim=True) + 1e-8)
    elif estimator != "group_mean":
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    return centred


def decision_logits(model, packed, hidden):
    """The model's readout over given backbone states, for the graphs whose readout is a head over hidden states."""
    mode = model.config.mode
    if mode not in {"tree", "listwise", "independent"}:
        raise ValueError("Workflow training supports the tree, listwise and independent graphs")
    logits = [[None] * k for k in packed.option_counts]
    for branch in packed.branches:
        if mode == "tree":
            values = model.head(hidden[list(branch.option_positions)])
            if branch.kind == "noul":
                logits[branch.question_index] = [values.new_zeros(()), values[0]]
                continue
        else:
            values = model.head(hidden[branch.decision_position], hidden[list(branch.option_positions)])
        for index, value in zip(branch.option_indices, values):
            logits[branch.question_index][index] = value
    return [torch.stack(values) for values in logits]


def two_path_logits(model, request, mask):
    """Policy logits with full gradient and report logits whose gradient stops at the backbone when masked."""
    packed = pack_request(request, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs)
    hidden = model._encode(packed)
    policy = decision_logits(model, packed, hidden)
    report = decision_logits(model, packed, hidden.detach()) if mask else policy
    return policy, report


def sample_indices(logits, group, temperature, generator):
    """[group, questions] option indices: one independent draw per question and group member."""
    columns = [torch.multinomial((z.detach().float().cpu() / temperature).softmax(-1), group, replacement=True, generator=generator)
               for z in logits]
    return torch.stack(columns, 1)


def answers_from_indices(request, indices):
    return {q.id: q.options[int(i)].key for q, i in zip(request.questions, indices)}


def tuple_log_probs(logits, indices, temperature):
    """[group] log-probability of each sampled tuple under the tempered per-question policies."""
    return torch.stack([(z.float() / temperature).log_softmax(-1)[indices[:, qi].to(z.device)]
                        for qi, z in enumerate(logits)], 1).sum(1)


def proper_score(logits, questions, objective):
    """Direct proper score over the questions that carry a target; None when none does."""
    pairs = [(z, torch.tensor(q.target)) for z, q in zip(logits, questions) if q.target is not None]
    if not pairs:
        return None
    return distribution_loss([z for z, _ in pairs], [y for _, y in pairs], objective)


def accuracies(logits, questions):
    return {q.id: float(q.target[int(z.argmax())]) for z, q in zip(logits, questions) if q.target is not None}


def _optimizer(model, config):
    return torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config.backbone_lr},
        {"params": list(model.head.parameters()), "lr": config.head_lr}], weight_decay=config.weight_decay)


def _combine(policy_term, proper_term, weights):
    policy_w, proper_w, _ = weights
    terms = []
    if policy_w and policy_term is not None:
        terms.append(policy_w * policy_term)
    if proper_w and proper_term is not None:
        terms.append(proper_w * proper_term)
    return sum(terms) if terms else None


def workflow_losses(model, batch, config, generator):
    """Per state: one forward, G sampled tuples, G program runs, one verifier score each; yields (loss, record)."""
    weights = config.weights
    for workflow, world in batch:
        request = workflow.request(world)
        policy_logits, report_logits = two_path_logits(model, request, weights[2])
        indices = sample_indices(policy_logits, config.group, config.temperature, generator)
        rewards = torch.tensor([workflow.verify(world, workflow.program(world.state, answers_from_indices(request, row)))
                                for row in indices])
        advantages = group_advantages(rewards, config.estimator).to(policy_logits[0].device)
        policy_term = -(advantages * tuple_log_probs(policy_logits, indices, config.temperature)).mean()
        proper_term = proper_score(report_logits, request.questions, config.proper_score)
        greedy = workflow.verify(world, workflow.program(world.state, answers_from_indices(request, [int(z.argmax()) for z in policy_logits])))
        record = {"workflow": workflow.name, "group_id": request.group_id, "state_hash": state_hash(request.state),
                  "reward_mean": float(rewards.mean()), "greedy_reward": greedy,
                  "policy_loss": float(policy_term.detach()), "proper_loss": float(proper_term.detach()) if proper_term is not None else None,
                  "accuracy": accuracies(policy_logits, request.questions), "rewards_evaluated": config.group if weights[0] else 0,
                  "labels_used": sum(q.target is not None for q in request.questions) if weights[1] else 0}
        yield _combine(policy_term, proper_term, weights), record


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _summarise(step, records, weights):
    accuracy = {}
    for record in records:
        for qid, value in record["accuracy"].items():
            accuracy.setdefault(qid, []).append(value)
    policy_loss, proper_loss = _mean(r["policy_loss"] for r in records), _mean(r["proper_loss"] for r in records)
    loss = (weights[0] * (policy_loss or 0.)) + (weights[1] * (proper_loss or 0.))
    return {"step": step, "reward_mean": _mean(r["reward_mean"] for r in records),
            "greedy_reward_mean": _mean(r["greedy_reward"] for r in records),
            "policy_loss": policy_loss, "proper_loss": proper_loss, "loss": loss,
            "accuracy": {qid: sum(v) / len(v) for qid, v in sorted(accuracy.items())},
            "per_workflow_reward": {name: _mean(r["reward_mean"] for r in records if r["workflow"] == name)
                                    for name in sorted({r["workflow"] for r in records})},
            "rewards_evaluated": sum(r["rewards_evaluated"] for r in records), "labels_used": sum(r["labels_used"] for r in records)}


def _finish(model, config, history, output, seen_groups, seen_states, extra):
    """Write the history, a summary with the information budget, and a checkpoint that `jev evaluate` can load."""
    if output is None:
        return history
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = getattr(model, "start_metadata", {}) or {}
    metadata = {"config": asdict(config), "parameters": model.parameter_counts(), **extra,
                "training_group_ids": sorted(set(seen_groups) | set(start.get("training_group_ids", []))),
                "training_state_hashes": sorted(set(seen_states) | set(start.get("training_state_hashes", []))),
                "selection_group_ids": start.get("selection_group_ids", []), "selection_state_hashes": start.get("selection_state_hashes", []),
                "start_checkpoint_provenance_merged": bool(start),
                "steps": len(history), "rewards_evaluated": history[-1]["rewards_evaluated"], "labels_used": history[-1]["labels_used"]}
    write_json(output / "history.json", history)
    write_json(output / "summary.json", {**metadata, "final": history[-1] if history else None})
    checkpoint(model, output / "final.pt", metadata)
    return history


def train_workflow_rl(model, workflows, config, worlds=None, output=None):
    """WP4a. `worlds` fixes the training states (cycled for `epochs`); otherwise fresh worlds are sampled each step."""
    config.validate()
    seed_everything(config.seed, config.threads)
    model.to(config.device)
    generator = torch.Generator().manual_seed(config.seed)
    rng = random.Random(config.seed)
    workflows = list(workflows)
    by_name = {w.name: w for w in workflows}
    fixed = None if worlds is None else [(by_name[w.workflow], w) for w in worlds]
    if config.steps is None:
        if fixed is None:
            raise ValueError("steps is required when worlds are sampled on the fly")
        steps = math.ceil(len(fixed) / config.states_per_step) * config.epochs
    else:
        steps = config.steps
    optimizer = _optimizer(model, config)
    weights = config.weights
    history, seen_groups, seen_states, order, drawn = [], set(), set(), [], 0
    rewards_evaluated = labels_used = 0
    start = time.perf_counter()
    for step in range(1, steps + 1):
        if fixed is None:
            batch = []
            for _ in range(config.states_per_step):
                workflow = workflows[drawn % len(workflows)]
                batch.append((workflow, workflow.sample(rng)))
                drawn += 1
        else:
            batch = []
            for _ in range(config.states_per_step):
                if not order:
                    order = list(range(len(fixed)))
                    rng.shuffle(order)
                batch.append(fixed[order.pop()])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        records = []
        for loss, record in workflow_losses(model, batch, config, generator):
            seen_groups.add(record["group_id"])
            seen_states.add(record["state_hash"])
            if loss is not None:
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                (loss / len(batch)).backward()
            records.append(record)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        row = _summarise(step, records, weights)
        rewards_evaluated, labels_used = rewards_evaluated + row["rewards_evaluated"], labels_used + row["labels_used"]
        row.update({"rewards_evaluated": rewards_evaluated, "labels_used": labels_used, "elapsed_seconds": time.perf_counter() - start})
        history.append(row)
        if step % config.log_every == 0 or step == steps:
            print(json.dumps(row), flush=True)
    return _finish(model, config, history, output, seen_groups, seen_states, {"kind": "workflow", "workflows": sorted(by_name)})


@torch.inference_mode()
def evaluate_workflows(model, worlds, workflows=WORKFLOWS):
    """Greedy decoding: the argmax tuple through the program, plus per-question accuracy, Brier and NLL."""
    model.eval()
    by_name = {w.name: w for w in workflows}
    rows, per_question = [], {}
    for world in worlds:
        workflow = by_name[world.workflow]
        request = workflow.request(world)
        logits = model(pack_request(request, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs))
        logits = [z.detach().float().cpu() for z in logits]
        answers = answers_from_indices(request, [int(z.argmax()) for z in logits])
        rows.append({"workflow": world.workflow, "reward": workflow.verify(world, workflow.program(world.state, answers))})
        for z, q in zip(logits, request.questions):
            y = torch.tensor(q.target)
            p = z.softmax(-1)
            per_question.setdefault(q.id, []).append({"accuracy": float(y[p.argmax()]), "brier": float((p - y).square().sum()),
                                                      "nll": float(-(y * z.log_softmax(-1)).sum())})
    names = sorted({r["workflow"] for r in rows})
    return {"states": len(rows), "greedy_reward_mean": _mean(r["reward"] for r in rows),
            "per_workflow": {n: _mean(r["reward"] for r in rows if r["workflow"] == n) for n in names},
            "per_question": {qid: {k: _mean(v[k] for v in values) for k in ("accuracy", "brier", "nll")}
                             for qid, values in sorted(per_question.items())}}


def env_policy(model, temperature, generator):
    def policy(grid, remaining):
        request = episode_request(grid, remaining)
        with torch.inference_mode():
            z = model(pack_request(request, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs))[0]
        z = z.detach().float().cpu()
        if temperature is None:
            return ACTIONS[int(z.argmax())]
        return ACTIONS[int(torch.multinomial((z / temperature).softmax(-1), 1, generator=generator))]
    return policy


def env_losses(model, starts, config, generator):
    """G rollouts per start under the sampling policy, then one gradient forward per visited state."""
    weights = config.weights
    model.eval()
    episodes = [[run_episode(grid, budget, env_policy(model, config.temperature, generator)) for _ in range(config.group)]
                for grid, budget in starts]
    returns = torch.tensor([[e["return"] for e in group] for group in episodes])
    advantages = group_advantages(returns, config.estimator)
    model.train()
    visits = sum(len(e["trajectory"]) for group in episodes for e in group)
    for group, advantage_row in zip(episodes, advantages):
        for episode, advantage in zip(group, advantage_row):
            realised = float(episode["success"])
            for grid, remaining, action in episode["trajectory"]:
                request = episode_request(grid, remaining)
                index = ACTIONS.index(action)
                nouls = list(request.questions[1:])
                taken = nouls[index]
                if config.noul_target == "realised":
                    nouls = [replace(taken, target=(1 - realised, realised))]
                    noul_logits_slice = [index]
                else:
                    noul_logits_slice = list(range(len(nouls)))
                policy_logits, report_logits = two_path_logits(model, request, weights[2])
                log_prob = (policy_logits[0].float() / config.temperature).log_softmax(-1)[index]
                policy_term = -advantage.to(log_prob.device) * log_prob
                proper_term = proper_score([report_logits[1 + i] for i in noul_logits_slice], nouls, config.proper_score)
                p_true = policy_logits[1 + index].detach().float().softmax(-1)[1].cpu()
                record = {"group_id": request.group_id, "state_hash": state_hash(request.state),
                          "return": episode["return"], "success": episode["success"], "steps": episode["steps"],
                          "action_accuracy": accuracies(policy_logits[:1], request.questions[:1])["env:action"],
                          "noul_brier_bfs": float(sum((policy_logits[1 + i].detach().float().softmax(-1).cpu() - torch.tensor(q.target)).square().sum()
                                                      for i, q in enumerate(request.questions[1:])) / len(ACTIONS)),
                          "noul_brier_realised": float((p_true - realised) ** 2),
                          "policy_loss": float(policy_term.detach()), "proper_loss": float(proper_term.detach()) if proper_term is not None else None,
                          "labels_used": len(nouls) if weights[1] else 0}
                loss = _combine(policy_term, proper_term, weights)
                yield None if loss is None else loss / visits, record
    yield None, {"episodes": [e for group in episodes for e in group]}


def _summarise_env(step, records, episodes, weights, group_count):
    policy_loss, proper_loss = _mean(r["policy_loss"] for r in records), _mean(r["proper_loss"] for r in records)
    return {"step": step, "return_mean": _mean(e["return"] for e in episodes), "success_rate": _mean(float(e["success"]) for e in episodes),
            "steps_mean": _mean(e["steps"] for e in episodes),
            "action_accuracy": _mean(r["action_accuracy"] for r in records),
            "noul_brier_bfs": _mean(r["noul_brier_bfs"] for r in records), "noul_brier_realised": _mean(r["noul_brier_realised"] for r in records),
            "policy_loss": policy_loss, "proper_loss": proper_loss,
            "loss": weights[0] * (policy_loss or 0.) + weights[1] * (proper_loss or 0.),
            "rewards_evaluated": group_count, "labels_used": sum(r["labels_used"] for r in records)}


def train_env_rl(model, config, starts=None, output=None):
    """WP4b. REINFORCE with a group baseline on the episode return for the action Choice; Brier on the success Nouls."""
    config.validate()
    seed_everything(config.seed, config.threads)
    model.to(config.device)
    generator = torch.Generator().manual_seed(config.seed)
    rng = random.Random(config.seed)
    if config.steps is None:
        if starts is None:
            raise ValueError("steps is required when start grids are sampled on the fly")
        steps = math.ceil(len(starts) / config.states_per_step) * config.epochs
    else:
        steps = config.steps
    optimizer = _optimizer(model, config)
    weights = config.weights
    history, seen_groups, seen_states, order = [], set(), set(), []
    rewards_evaluated = labels_used = 0
    start = time.perf_counter()
    for step in range(1, steps + 1):
        if starts is None:
            batch = [budgeted_grid(rng, max_distance=config.env_max_distance) for _ in range(config.states_per_step)]
        else:
            batch = []
            for _ in range(config.states_per_step):
                if not order:
                    order = list(range(len(starts)))
                    rng.shuffle(order)
                batch.append(starts[order.pop()])
        optimizer.zero_grad(set_to_none=True)
        records, episodes = [], []
        for loss, record in env_losses(model, batch, config, generator):
            if "episodes" in record:
                episodes = record["episodes"]
                continue
            seen_groups.add(record["group_id"])
            seen_states.add(record["state_hash"])
            if loss is not None:
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                loss.backward()
            records.append(record)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        row = _summarise_env(step, records, episodes, weights, len(batch) * config.group)
        rewards_evaluated, labels_used = rewards_evaluated + row["rewards_evaluated"], labels_used + row["labels_used"]
        row.update({"rewards_evaluated": rewards_evaluated, "labels_used": labels_used, "elapsed_seconds": time.perf_counter() - start})
        history.append(row)
        if step % config.log_every == 0 or step == steps:
            print(json.dumps(row), flush=True)
    return _finish(model, config, history, output, seen_groups, seen_states, {"kind": "env"})


def evaluate_env(model, starts):
    """Greedy episodes from each start: success rate, mean return, and Noul Brier against BFS at visited states."""
    model.eval()
    policy = env_policy(model, None, None)
    episodes = [run_episode(grid, budget, policy) for grid, budget in starts]
    briers, action_accuracy = [], []
    with torch.inference_mode():
        for episode in episodes:
            for grid, remaining, _ in episode["trajectory"]:
                request = episode_request(grid, remaining)
                logits = [z.detach().float().cpu() for z in
                          model(pack_request(request, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs))]
                action_accuracy.append(accuracies(logits[:1], request.questions[:1])["env:action"])
                briers += [float((z.softmax(-1) - torch.tensor(q.target)).square().sum()) for z, q in zip(logits[1:], request.questions[1:])]
    return {"episodes": len(episodes), "success_rate": _mean(float(e["success"]) for e in episodes),
            "return_mean": _mean(e["return"] for e in episodes), "steps_mean": _mean(e["steps"] for e in episodes),
            "action_accuracy": _mean(action_accuracy), "noul_brier_bfs": _mean(briers)}
