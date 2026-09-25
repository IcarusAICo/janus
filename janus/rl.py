"""WP2c neural RL arms: a report-grid head on the Noul leaf and the eight arms of the spec table.

Every arm is defined by what is sampled, what is rewarded, and what is differentiated (spec WP2).
Estimator conventions (group size, leave-one-out baseline, GRPO standard-deviation normalisation,
optional KL and clip) mirror janus/estimators.py. Vocabulary: p is the ordinary decision distribution
(softmax of the tree-mode logits), pi_grid the report policy over {0, .01, ..., 1} from the report
head, q a confidence about a selected answer, Y one realised label per question presentation.
"""

from dataclasses import asdict, dataclass
import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .data import assert_disjoint, file_hash, load_requests, state_hash, training_epoch, write_json, write_jsonl
from .estimators import report_grid
from .metrics import metrics
from .packing import pack_request
from .training import balanced_subset, checkpoint, load_checkpoint, seed_everything

ARMS = ("S-CE", "S-Brier", "RLVR-cat", "REPORT-exact", "REPORT-RLOO", "REPORT-GRPO", "SEP", "COUPLED")
SUPERVISED_ARMS = ("S-CE", "S-Brier")
REPORT_ARMS = ("REPORT-exact", "REPORT-RLOO", "REPORT-GRPO")
CATEGORICAL_SAMPLED_ARMS = ("RLVR-cat", "SEP", "COUPLED")
# Spec WP2: two named non-arms, recorded so nobody runs them by accident.
NON_ARMS = {
    "DETACHED-Brier-reward": "a detached full-distribution Brier used as a constant reward for every sampled "
                             "label has zero expected score-function gradient",
    "REINFORCE-Brier-full": "'REINFORCE with Brier on the full distribution' without further specification is "
                            "one of REPORT-RLOO, SEP, or COUPLED and must be named as such",
}
BASELINES = ("rloo", "group_mean", "grpo")


@dataclass
class RLConfig:
    arm: str = "REPORT-exact"
    group: int = 8
    baseline: str = "rloo"          # RLVR-cat, SEP, COUPLED: rloo | group_mean | grpo
    kl: float = 0.                  # KL(pi || pi_initial) weight on the sampled policy's distribution
    clip: float | None = None       # PPO clip against the sampling-time policy (vacuous for one on-policy update)
    coupled_lambda: float = 1.
    grid_size: int = 101
    checkpoint: str = "runs/phase1/screen/g2/best.pt"
    device: str = "cuda:0"
    seed: int = 17
    epochs: int = 1
    accumulation: int = 8
    backbone_lr: float = 2e-4
    head_lr: float = 1e-3
    weight_decay: float = .01
    eval_every: int = 100
    max_steps: int | None = None
    dev_limit: int | None = None
    train_limit: int | None = None
    data_seed: int = 17
    warmup_steps: int = 0
    cosine_decay: bool = False
    max_seconds: float | None = None
    threads: int = 4

    def __post_init__(self):
        self.validate()

    @classmethod
    def from_dict(cls, raw):
        return cls(**dict(raw))

    def validate(self):
        if self.arm in NON_ARMS:
            raise ValueError(f"{self.arm!r} is a named non-arm (spec WP2): {NON_ARMS[self.arm]}")
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.group < 2 or self.kl < 0 or (self.clip is not None and not 0 < self.clip < 1):
            raise ValueError("Invalid estimator settings: group >= 2, kl >= 0, clip in (0, 1) or None")
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        if self.coupled_lambda < 0 or self.grid_size < 2:
            raise ValueError("coupled_lambda must be nonnegative and grid_size at least 2")
        if self.epochs < 1 or self.accumulation < 1 or self.eval_every < 1:
            raise ValueError("epochs, accumulation and eval_every must be positive")
        for name in ("max_steps", "dev_limit", "train_limit"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        return self


class ReportGridHead(nn.Module):
    """Leaf hidden state -> logits over the report grid {0, 1/(n-1), ..., 1}; zero-initialised, so uniform at start."""

    def __init__(self, hidden_size, grid_size=101):
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        self.norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, grid_size)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        grid = report_grid() if grid_size == 101 else torch.arange(grid_size, dtype=torch.float64) / (grid_size - 1)
        self.register_buffer("grid", grid.double())

    def forward(self, reps):
        return self.out(self.norm(reps.float()))


class ReportModel(nn.Module):
    """A tree-mode DecisionModel plus a report-grid head reading the same leaf hidden states.

    forward(packed) -> (logits, reports): `logits` is exactly DecisionModel.forward's tree-mode output
    (Noul as [0, z]); `reports[i]` has one row of grid logits per leaf of question i (one row for Noul)."""

    def __init__(self, model, grid_size=101):
        super().__init__()
        if model.config.mode != "tree":
            raise ValueError("ReportModel requires a DecisionModel in tree mode")
        self.model = model
        self.report_head = ReportGridHead(model.backbone.config.hidden_size, grid_size)

    @property
    def device(self):
        return self.model.device

    @property
    def grid(self):
        return self.report_head.grid

    def pack(self, request):
        return pack_request(request, self.model.tokenizer, self.model.packing_mode, self.model.config.max_tokens,
                            **self.model.packing_kwargs)

    def forward(self, packed):
        model = self.model
        hidden = model._encode(packed)
        model.last_compute = {"backbone_tokens": packed.token_count, "decoder_cross_attention_pairs": 0,
                              "decoder_self_attention_pairs": 0}
        logits = [[None] * k for k in packed.option_counts]
        reports = [None] * len(packed.option_counts)
        for branch in packed.branches:
            leaves = hidden[list(branch.option_positions)]
            z = model.head(leaves)
            reports[branch.question_index] = self.report_head(leaves)
            if branch.kind == "noul":
                logits[branch.question_index] = [z.new_zeros(()), z[0]]
            else:
                for index, value in zip(branch.option_indices, z):
                    logits[branch.question_index][index] = value
        return [torch.stack(values) for values in logits], reports

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "report_head": sum(p.numel() for p in self.report_head.parameters())}


def save_checkpoint(model, path, metadata):
    """best.pt loadable by janus.training.load_checkpoint (ordinary evaluation) with the report head added."""
    checkpoint(model.model, path, {**metadata, "report_head": {"grid_size": model.report_head.out.out_features}})
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["report_head"] = {n: t.detach().cpu() for n, t in model.report_head.state_dict().items()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_rl_checkpoint(path, device="cpu"):
    base, metadata = load_checkpoint(path, device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "report_head" not in payload:
        raise ValueError("Checkpoint has no report head; train it with janus.rl.train_rl")
    model = ReportModel(base, payload["report_head"]["out.weight"].shape[0])
    model.report_head.load_state_dict(payload["report_head"])
    return model.to(device).eval(), metadata


# --- estimator pieces, mirroring janus/estimators.py ---

def _draw_label(target, rng):
    """One realised outcome per question presentation (the shared design of the CPU study)."""
    return int(torch.multinomial(torch.as_tensor(target, dtype=torch.float64), 1, generator=rng))


def _sample(probs, group, rng):
    return torch.multinomial(probs.detach().double().cpu(), group, replacement=True, generator=rng)


def _advantages(rewards, baseline, group):
    if baseline == "rloo":
        return rewards - (rewards.sum() - rewards) / (group - 1)
    advantages = rewards - rewards.mean()
    if baseline == "grpo":
        advantages = advantages / (rewards.std(unbiased=False) + 1e-8)
    return advantages


def _score_function(logp, advantages, clip):
    """mean_g adv_g log pi(a_g); with clip, the PPO surrogate against the sampling-time policy.

    The reference is the policy the samples were drawn from, i.e. the current policy detached, so for
    one on-policy update the ratio is 1 at the point of differentiation and the clip never binds
    (the same single-step property documented in docs/phase1/estimator-study.md)."""
    if clip is None:
        return (advantages * logp).mean()
    ratio = (logp - logp.detach()).exp()
    return torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()


def _kl(log_p, log_ref):
    return (log_p.exp() * (log_p - log_ref)).sum()


def _confidence_grid(kind, answers, grid):
    """Grid of confidences that answer A is correct: the Noul leaf reports P(true), so 'false' mirrors it."""
    if kind == "noul":
        return torch.where(answers[:, None] == 1, grid[None], 1 - grid[None])
    return grid[None].expand(answers.shape[0], -1)


def rl_step(model, batch, config, rng=None, reference=None):
    """One loss over a list of requests for config.arm; returns (loss, info).

    Sampled / rewarded / differentiated per arm follow the spec WP2 table (see the branches below).
    `reference` is a frozen copy of the initial policy, required only when config.kl > 0."""
    if config.kl and reference is None:
        raise ValueError("kl requires the initial-policy reference model")
    grid = model.grid
    policy_terms, auxiliary_terms, kl_terms = [], [], []
    rewards_seen, advantages_out, outcomes = [], [], []
    labels = sampled = tokens = 0
    for request in batch:
        packed = model.pack(request)
        logits, reports = model(packed)
        tokens += packed.token_count
        ref_logits = ref_reports = None
        if reference is not None:
            with torch.no_grad():
                ref_logits, ref_reports = reference(packed)
        for qi, question in enumerate(request.questions):
            if question.target is None:
                raise ValueError("RL training requires targets for all questions")
            if config.arm in REPORT_ARMS and question.kind != "noul":
                continue
            z = logits[qi].float()
            y = torch.tensor(question.target, dtype=torch.float32, device=z.device)
            log_p = z.log_softmax(-1)
            labels += 1
            auxiliary = kl = z.new_zeros(())
            if config.arm == "S-CE":
                # Sampled: nothing. Reward: none. Differentiated: log p_Y (soft targets weight the terms).
                policy = -(y * log_p).sum()
            elif config.arm == "S-Brier":
                # Sampled: nothing. Reward: none. Differentiated: sum_k (p_k - y_k)^2.
                policy = (log_p.exp() - y).square().sum()
            else:
                label = _draw_label(question.target, rng)
                outcomes.append(label)
                if config.arm in REPORT_ARMS:
                    log_pi = reports[qi][0].float().log_softmax(-1)
                    if config.arm == "REPORT-exact":
                        # Sampled: nothing. Reward: Brier of each grid report against Y.
                        # Differentiated: the enumerated expectation sum_j pi_j reward_j over the grid.
                        reward = -(grid - label).square()
                        objective = (log_pi.exp() * reward).sum()
                        rewards_seen.append(float(objective.detach()))
                    else:
                        # Sampled: r_g ~ pi_grid, G per question. Reward: -(r_g - Y)^2.
                        # Differentiated: log pi_grid(r_g) with the leave-one-out baseline (REPORT-RLOO) or the
                        # group mean and standard-deviation normalisation (REPORT-GRPO); rewards detached.
                        idx = _sample(log_pi.exp(), config.group, rng).to(z.device)
                        sampled += config.group
                        reward = -(grid[idx] - label).square().detach()
                        advantages = _advantages(reward, "rloo" if config.arm == "REPORT-RLOO" else "grpo", config.group)
                        objective = _score_function(log_pi[idx], advantages, config.clip)
                        rewards_seen.append(float(reward.mean()))
                        advantages_out.append(advantages.detach().cpu())
                    policy = -objective
                    if config.kl:
                        kl = config.kl * _kl(log_pi, ref_reports[qi][0].float().log_softmax(-1))
                else:
                    # Sampled: A_g ~ p, G per question. Reward: 1[A_g = Y].
                    # Differentiated: log p_{A_g} only (score-function), baseline per config.baseline.
                    idx = _sample(log_p.exp(), config.group, rng).to(z.device)
                    sampled += config.group
                    correct = (idx == label).float().detach()
                    advantages = _advantages(correct, config.baseline, config.group)
                    policy = -_score_function(log_p[idx], advantages, config.clip)
                    rewards_seen.append(float(correct.mean()))
                    advantages_out.append(advantages.detach().cpu())
                    if config.arm == "COUPLED":
                        # Plus lambda * Brier(p, Y) differentiated directly into the same p.
                        one_hot = F.one_hot(torch.tensor(label, device=z.device), z.numel()).float()
                        auxiliary = config.coupled_lambda * (log_p.exp() - one_hot).square().sum()
                    elif config.arm == "SEP":
                        # Confidence q about each sampled answer A_g: the report-grid distribution of A_g's leaf,
                        # scored by the enumerated expected Brier against the detached correctness 1[A_g = Y]
                        # and differentiated directly. The sampled index carries no gradient, so p receives
                        # nothing from this term and the report head nothing from the correctness term.
                        leaf = torch.zeros_like(idx) if question.kind == "noul" else idx
                        log_q = reports[qi].float().log_softmax(-1)[leaf]
                        confidence = _confidence_grid(question.kind, idx, grid)
                        auxiliary = (log_q.exp() * (confidence - correct[:, None]).square()).sum(-1).mean()
                    if config.kl:
                        kl = config.kl * _kl(log_p, ref_logits[qi].float().log_softmax(-1))
            policy_terms.append(policy)
            auxiliary_terms.append(auxiliary)
            kl_terms.append(kl)
    if not policy_terms:
        return torch.zeros(()), {"labels_consumed": 0, "reports_sampled": 0, "encoded_tokens": tokens,
                                 "questions": 0, "mean_reward": None, "policy_loss": 0., "auxiliary_loss": 0.,
                                 "kl_loss": 0., "loss_terms": None, "advantages": [], "outcomes": []}
    policy_loss, auxiliary_loss, kl_loss = (torch.stack(t).mean() for t in (policy_terms, auxiliary_terms, kl_terms))
    loss = policy_loss + auxiliary_loss + kl_loss
    info = {"labels_consumed": labels, "reports_sampled": sampled, "encoded_tokens": tokens,
            "questions": len(policy_terms),
            "mean_reward": sum(rewards_seen) / len(rewards_seen) if rewards_seen else None,
            "policy_loss": float(policy_loss.detach()), "auxiliary_loss": float(auxiliary_loss.detach()),
            "kl_loss": float(kl_loss.detach()),
            "loss_terms": {"policy": policy_loss, "auxiliary": auxiliary_loss, "kl": kl_loss},
            "advantages": advantages_out, "outcomes": outcomes}
    return loss, info


# --- evaluation ---

@torch.inference_mode()
def collect_outputs(model, requests):
    model.eval()
    rows = []
    for request in requests:
        logits, reports = model(model.pack(request))
        for question, z, r in zip(request.questions, logits, reports):
            if question.target is None:
                raise ValueError("Evaluation requires targets for all questions")
            rows.append({"group_id": request.group_id, "id": question.id, "kind": question.kind,
                         "logits": z.detach().float().cpu(), "target": torch.tensor(question.target, dtype=torch.float32),
                         "reports": r.detach().float().cpu()})
    return rows


def _policy_moments(report_logits, grid):
    pi = report_logits.double().softmax(-1)
    mean = (pi * grid).sum(-1)
    variance = (pi * (grid[None] - mean[:, None]).square()).sum(-1)
    return mean, variance


def _forecast_scores(mean, variance, eta, targets):
    noise = eta * (1 - eta)
    mean_logits = torch.stack([(1 - mean).clamp_min(1e-6).log(), mean.clamp_min(1e-6).log()], -1).float()
    return {"count": int(mean.shape[0]), "mean_report": float(mean.mean()), "report_variance": float(variance.mean()),
            "expected_brier": float(((mean - eta).square() + variance + noise).mean()),
            "policy_mean_brier": float(((mean - eta).square() + noise).mean()),
            "policy_mean": metrics(list(mean_logits), targets)}


def report_metrics(rows, grid):
    """Noul report policy scored the WP2a way: sampled-report expected Brier first, policy-mean forecast separately.

    expected_brier = E_pi E_Y (R - Y)^2 = (E[R] - eta)^2 + Var(R) + eta (1 - eta) with eta the target rate;
    policy_mean_brier drops the variance term; policy_mean carries janus.metrics on the mean forecast
    (its brier is the two-vector convention)."""
    noul = [r for r in rows if r["kind"] == "noul"]
    if not noul:
        return None
    grid = grid.double().cpu()
    mean, variance = _policy_moments(torch.stack([r["reports"][0] for r in noul]), grid)
    eta = torch.stack([r["target"][1] for r in noul]).double()
    return _forecast_scores(mean, variance, eta, [torch.stack([1 - e, e]).float() for e in eta])


def selected_confidence(rows, grid):
    """Confidence q about the argmax answer from its leaf's report grid (mirrored for a Noul 'false'), for every question."""
    grid = grid.double().cpu()
    means, variances, etas = [], [], []
    for r in rows:
        answer = int(r["logits"].argmax())
        leaf = 0 if r["kind"] == "noul" else answer
        pi = r["reports"][leaf].double().softmax(-1)
        confidence = _confidence_grid(r["kind"], torch.tensor([answer]), grid)[0]
        mean = (pi * confidence).sum()
        means.append(mean)
        variances.append((pi * (confidence - mean).square()).sum())
        etas.append(r["target"][answer].double())
    mean, variance, eta = torch.stack(means), torch.stack(variances), torch.stack(etas)
    return _forecast_scores(mean, variance, eta, [torch.stack([1 - e, e]).float() for e in eta])


def evaluate_rl(model, requests):
    rows = collect_outputs(model, requests)
    noul = [r for r in rows if r["kind"] == "noul"]
    return {"all": metrics([r["logits"] for r in rows], [r["target"] for r in rows]),
            "noul": metrics([r["logits"] for r in noul], [r["target"] for r in noul]) if noul else None,
            "report": report_metrics(rows, model.grid),
            "selected_confidence": selected_confidence(rows, model.grid)}


def selection_nll(evaluation, arm):
    """Dev NLL used for checkpoint selection: the policy-mean Noul forecast for the REPORT arms, else the ordinary NLL."""
    if arm in REPORT_ARMS:
        if evaluation["report"] is None:
            raise ValueError("REPORT arms need Noul questions in the dev set")
        return evaluation["report"]["policy_mean"]["nll"]
    return evaluation["all"]["nll"]


# --- training driver ---

def train_rl(checkpoint_path, train_path, dev_path, output, config):
    config.validate()
    seed_everything(config.seed, config.threads)
    train_data, dev_data = load_requests(train_path), load_requests(dev_path)
    assert_disjoint({"train": train_data, "dev": dev_data})
    if config.train_limit:
        train_data = balanced_subset(train_data, config.train_limit, config.data_seed)
    if any(q.target is None for r in train_data for q in r.questions):
        raise ValueError("Training requests must have complete targets")
    if any("test" in Path(p).stem.lower() or "calibration" in Path(p).stem.lower() for p in (train_path, dev_path)):
        raise ValueError("Use train/dev splits for checkpoint selection")
    full_dev = dev_data
    if config.dev_limit:
        dev_data = balanced_subset(dev_data, config.dev_limit, config.data_seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base, base_metadata = load_checkpoint(checkpoint_path, config.device)
    model = ReportModel(base, config.grid_size).to(config.device)
    reference = None
    if config.kl:
        reference = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.model.backbone.parameters() if p.requires_grad], "lr": config.backbone_lr},
        {"params": list(model.model.head.parameters()) + list(model.report_head.parameters()), "lr": config.head_lr}],
        weight_decay=config.weight_decay)
    total_steps = math.ceil(len(train_data) / config.accumulation) * config.epochs
    if config.max_steps:
        total_steps = min(total_steps, config.max_steps)

    def lr_multiplier(step):
        if config.warmup_steps and step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        if config.cosine_decay:
            progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
            return .1 + .9 * (1 + math.cos(math.pi * min(1., max(0., progress)))) / 2
        return 1.

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    metadata = {"config": asdict(config), "arm": config.arm, "initial_checkpoint": str(checkpoint_path),
                "initial_checkpoint_sha256": file_hash(checkpoint_path), "initial_step": base_metadata.get("step"),
                "parameters": model.parameter_counts(),
                "train_sha256": file_hash(train_path), "dev_sha256": file_hash(dev_path),
                "training_group_ids": sorted({r.group_id for r in train_data}),
                "selection_group_ids": sorted({r.group_id for r in full_dev}),
                "training_state_hashes": sorted({state_hash(r.state) for r in train_data}),
                "selection_state_hashes": sorted({state_hash(r.state) for r in full_dev}),
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")},
                "device": str(config.device), "device_name": (torch.cuda.get_device_name(model.device)
                           if model.device.type == "cuda" else "CPU")}
    write_json(output / "config.json", metadata)
    start = time.perf_counter()
    initial = evaluate_rl(model, dev_data)
    best_nll, best_step, step = selection_nll(initial, config.arm), 0, 0
    counters = {"requests_seen": 0, "labels_consumed": 0, "reports_sampled": 0, "encoded_tokens": 0}
    seen_groups, seen_states, stop = set(), set(), False
    save_checkpoint(model, output / "best.pt", {**metadata, "step": 0, "dev": initial, **counters,
                                                "seen_group_ids": [], "seen_state_hashes": []})
    history = [{"step": 0, "dev_nll": initial["all"]["nll"], "dev_accuracy": initial["all"]["accuracy"],
                "dev_selection_nll": best_nll}]
    print(json.dumps(history[-1]), flush=True)
    rng = torch.Generator().manual_seed(config.seed)
    order_rng = random.Random(config.seed)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    for epoch in range(config.epochs):
        epoch_data = training_epoch(train_data, config.seed, epoch)
        order = list(range(len(train_data)))
        order_rng.shuffle(order)
        for offset in range(0, len(order), config.accumulation):
            batch = order[offset:offset + config.accumulation]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals = {"loss": 0., "policy_loss": 0., "auxiliary_loss": 0., "kl_loss": 0.}
            rewards = []
            for index in batch:
                request = epoch_data[index]
                counters["requests_seen"] += 1
                seen_groups.add(request.group_id)
                seen_states.add(state_hash(request.state))
                loss, info = rl_step(model, [request], config, rng, reference)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {step}")
                if loss.requires_grad:
                    (loss / len(batch)).backward()
                for key in ("labels_consumed", "reports_sampled", "encoded_tokens"):
                    counters[key] += info[key]
                totals["loss"] += float(loss.detach()) / len(batch)
                for key in ("policy_loss", "auxiliary_loss", "kl_loss"):
                    totals[key] += info[key] / len(batch)
                if info["mean_reward"] is not None:
                    rewards.append(info["mean_reward"])
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            stop = bool((config.max_steps and step >= config.max_steps)
                        or (config.max_seconds and time.perf_counter() - start >= config.max_seconds))
            last = offset + config.accumulation >= len(order) or stop
            if step % config.eval_every == 0 or last:
                validation = evaluate_rl(model, dev_data)
                selection = selection_nll(validation, config.arm)
                row = {"step": step, "epoch": epoch + 1, "train_loss": totals["loss"],
                       "policy_loss": totals["policy_loss"], "auxiliary_loss": totals["auxiliary_loss"],
                       "kl_loss": totals["kl_loss"],
                       "mean_reward": sum(rewards) / len(rewards) if rewards else None, **counters,
                       "head_lr": optimizer.param_groups[1]["lr"], "dev_nll": validation["all"]["nll"],
                       "dev_accuracy": validation["all"]["accuracy"], "dev_selection_nll": selection,
                       "dev_report": validation["report"], "elapsed_seconds": time.perf_counter() - start}
                history.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "dev_report"}), flush=True)
                if selection < best_nll:
                    best_nll, best_step = selection, step
                    save_checkpoint(model, output / "best.pt", {**metadata, "step": step, "dev": validation, **counters,
                                    "seen_group_ids": sorted(seen_groups), "seen_state_hashes": sorted(seen_states)})
                write_json(output / "history.json", history)
            if stop:
                break
        if stop:
            break
    result = {"arm": config.arm, "steps": step, "updates": step, "best_step": best_step,
              "selection_metric": "noul_policy_mean_nll" if config.arm in REPORT_ARMS else "dev_nll",
              "initial_dev_nll": initial["all"]["nll"], "initial_selection_nll": history[0]["dev_selection_nll"],
              "best_selection_nll": best_nll, "dev_requests": len(dev_data), "train_requests": len(train_data),
              **counters, "unique_groups_seen": len(seen_groups), "unique_states_seen": len(seen_states),
              "budget_limited": bool(config.max_seconds and time.perf_counter() - start >= config.max_seconds),
              "elapsed_seconds": time.perf_counter() - start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None}
    write_json(output / "summary.json", result)
    return result


def evaluate_run(checkpoint_path, data_path, output, device="cpu", limit=None):
    """Ordinary metrics (janus.metrics) for all questions and the report-grid scores, written to output/."""
    from .evaluation import check_unused
    torch.set_num_threads(4)
    model, metadata = load_rl_checkpoint(checkpoint_path, device)
    requests = load_requests(data_path)
    check_unused(requests, metadata)
    if limit:
        requests = random.Random(17).sample(requests, min(limit, len(requests)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = collect_outputs(model, requests)
    noul = [r for r in rows if r["kind"] == "noul"]
    grid = model.grid.double().cpu()
    result = {"arm": metadata.get("arm"), "checkpoint_sha256": file_hash(checkpoint_path),
              "data_sha256": file_hash(data_path), "requests": len(requests), "questions": len(rows),
              "all": metrics([r["logits"] for r in rows], [r["target"] for r in rows]),
              "noul": metrics([r["logits"] for r in noul], [r["target"] for r in noul]) if noul else None,
              "report": report_metrics(rows, grid), "selected_confidence": selected_confidence(rows, grid)}
    write_json(output / "metrics.json", result)
    predictions = []
    for r in rows:
        row = {"group_id": r["group_id"], "question_id": r["id"], "kind": r["kind"],
               "probabilities": r["logits"].softmax(-1).tolist(), "target": r["target"].tolist()}
        if r["kind"] == "noul":
            mean, variance = _policy_moments(r["reports"][:1], grid)
            row.update({"mean_report": float(mean[0]), "report_variance": float(variance[0])})
        predictions.append(row)
    write_jsonl(output / "predictions.jsonl", predictions)
    return result


def report_table(root, sets=("test", "test_post_unseen", "panel")):
    """Markdown table over runs/<arm>/summary.json and runs/<arm>/<set>/metrics.json."""
    root = Path(root)
    lines = ["| arm | updates | labels | reports sampled | tokens | best step | " +
             " | ".join(f"{s} nll / acc / ece | {s} noul nll (decision head) / noul nll (report policy mean) | {s} noul report E-Brier / mean-Brier" for s in sets) + " |",
             "| --- | ---: | ---: | ---: | ---: | ---: | " + " | ".join("---: | ---:" for _ in sets) + " |"]
    for arm in ARMS:
        summary_path = root / arm / "summary.json"
        if not summary_path.exists():
            continue
        s = json.loads(summary_path.read_text())
        cells = [arm, str(s["updates"]), str(s["labels_consumed"]), str(s["reports_sampled"]),
                 str(s["encoded_tokens"]), str(s["best_step"])]
        for name in sets:
            path = root / arm / name / "metrics.json"
            if not path.exists():
                cells += ["n/a", "n/a", "n/a"]
                continue
            m = json.loads(path.read_text())
            cells.append(f"{m['all']['nll']:.4f} / {m['all']['accuracy']:.4f} / {m['all']['ece']:.4f}")
            r = m.get("report")
            noul = m.get("noul")
            cells.append(f"{noul['nll']:.4f} / {r['policy_mean']['nll']:.4f}" if noul and r and r.get("policy_mean") else "n/a")
            cells.append(f"{r['expected_brier']:.4f} / {r['policy_mean_brier']:.4f}" if r else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train one WP2c arm from the winner checkpoint")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--dev", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--checkpoint", help="Override the config's starting checkpoint")
    train_parser.add_argument("--device")
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--max-steps", type=int)
    eval_parser = commands.add_parser("evaluate", help="Ordinary and report-grid metrics of a trained arm")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--limit", type=int)
    report_parser = commands.add_parser("report", help="Markdown table over the arms under a run root")
    report_parser.add_argument("root")
    args = parser.parse_args(argv)
    if args.command == "train":
        config = RLConfig.from_dict(json.loads(Path(args.config).read_text()))
        for key in ("device", "seed", "max_steps", "checkpoint"):
            if getattr(args, key) is not None:
                setattr(config, key, getattr(args, key))
        config.validate()
        result = train_rl(config.checkpoint, args.train, args.dev, args.output, config)
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "evaluate":
        value = evaluate_run(args.checkpoint, args.data, args.output, args.device, args.limit)
        print(json.dumps({"output": args.output, "requests": value["requests"],
                          "all": {k: value["all"][k] for k in ("accuracy", "nll", "brier", "ece")},
                          "report": value["report"]}, indent=2, allow_nan=False))
    else:
        print(report_table(args.root))


if __name__ == "__main__":
    main()
