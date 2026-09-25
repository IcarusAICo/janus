"""CPU-sized calibration-estimator study on known Bernoulli rates (spec WP2a).

A tabular actor places probability over a grid of scalar probability reports.
Reports are actions. The actor's distribution over reports is not a forecast of
the outcome; sampled reports are scored individually against outcomes.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch


def report_grid(interior=False):
    grid = torch.arange(101, dtype=torch.float64) / 100
    return grid[1:-1] if interior else grid


def brier_reward(reports, outcomes):
    return -(reports.double() - outcomes.double()).square()


def log_reward(reports, outcomes):
    reports = reports.double()
    if ((reports <= 0) | (reports >= 1)).any():
        raise ValueError("log reward requires interior reports strictly between 0 and 1")
    outcomes = outcomes.double()
    return outcomes * reports.log() + (1 - outcomes) * (1 - reports).log()


@dataclass(frozen=True)
class KnownRates:
    rates: tuple

    def __post_init__(self):
        if not self.rates or any(not 0 <= r <= 1 for r in self.rates):
            raise ValueError("Rates must be probabilities")

    def sample(self, batch, rng):
        contexts = torch.randint(len(self.rates), (batch,), generator=rng)
        eta = torch.tensor(self.rates, dtype=torch.float64)[contexts]
        outcomes = (torch.rand(batch, generator=rng, dtype=torch.float64) < eta).double()
        return contexts, outcomes


class ReportPolicy(torch.nn.Module):
    def __init__(self, contexts, grid):
        super().__init__()
        self.register_buffer("grid", grid.double())
        self.logits = torch.nn.Parameter(torch.zeros(contexts, len(grid), dtype=torch.float64))

    def log_probs(self, contexts):
        return self.logits[contexts].log_softmax(-1)

    def probs(self, contexts):
        return self.log_probs(contexts).exp()

    def sample(self, contexts, group, rng):
        p = self.probs(contexts).detach()
        return torch.multinomial(p, group, replacement=True, generator=rng)

    @torch.no_grad()
    def evaluate(self, rates):
        p = self.probs(torch.arange(self.logits.shape[0]))
        mean = (p * self.grid).sum(-1)
        variance = (p * (self.grid[None] - mean[:, None]).square()).sum(-1)
        noise = rates * (1 - rates)
        return {"mean_report": mean, "report_variance": variance,
                "forecast_error": (mean - rates).abs(),
                "expected_brier": (mean - rates).square() + variance + noise,
                "policy_mean_brier": (mean - rates).square() + noise}


@dataclass(frozen=True)
class EstimatorConfig:
    name: str
    group: int = 8
    outcome_design: str = "shared"
    kl: float = 0.
    clip: float | None = None
    reward: str = "brier"

    def __post_init__(self):
        if self.name not in {"exact", "rloo", "group_mean", "grpo"}:
            raise ValueError("Unknown estimator")
        if self.outcome_design not in {"shared", "independent"}:
            raise ValueError("outcome_design must be shared or independent")
        if self.group < 2 or self.kl < 0 or (self.clip is not None and not 0 < self.clip < 1):
            raise ValueError("Invalid estimator settings")
        if self.reward not in {"brier", "log"}:
            raise ValueError("reward must be brier or log")


def _reward(name):
    return brier_reward if name == "brier" else log_reward


def estimator_loss(policy, contexts, outcomes, rates, config, rng, reference_log_probs=None):
    reward = _reward(config.reward)
    grid = policy.grid
    log_probs = policy.log_probs(contexts)
    batch = contexts.shape[0]
    if config.outcome_design == "shared":
        if outcomes.ndim != 1:
            raise ValueError("shared outcomes have shape [batch]")
        labels = batch
    else:
        if outcomes.shape != (batch, config.group):
            raise ValueError("independent outcomes have shape [batch, group]")
        labels = batch * config.group
    info = {"labels_consumed": labels}
    if config.name == "exact":
        if config.outcome_design == "shared":
            rewards = reward(grid[None].expand(batch, -1), outcomes[:, None].expand(-1, len(grid)))
        else:
            rewards = torch.stack([reward(grid[None].expand(batch, -1), outcomes[:, g][:, None].expand(-1, len(grid)))
                                   for g in range(config.group)]).mean(0)
        objective = (log_probs.exp() * rewards).sum(-1).mean()
        info["mean_reward"] = float(objective.detach())
    else:
        idx = policy.sample(contexts, config.group, rng)
        reports = grid[idx]
        y = outcomes[:, None].expand(-1, config.group) if config.outcome_design == "shared" else outcomes
        rewards = reward(reports, y).detach()
        if config.name == "rloo":
            advantages = rewards - (rewards.sum(-1, keepdim=True) - rewards) / (config.group - 1)
        else:
            advantages = rewards - rewards.mean(-1, keepdim=True)
            if config.name == "grpo":
                advantages = advantages / (rewards.std(-1, unbiased=False, keepdim=True) + 1e-8)
        logp = log_probs.gather(1, idx)
        if config.clip is not None:
            if reference_log_probs is None:
                raise ValueError("clip requires reference_log_probs")
            ratio = (logp - reference_log_probs.gather(1, idx)).exp()
            objective = torch.minimum(ratio * advantages, ratio.clamp(1 - config.clip, 1 + config.clip) * advantages).mean()
        else:
            objective = (advantages * logp).mean()
        info.update({"mean_reward": float(rewards.mean()), "advantages": advantages, "indices": idx})
    loss = -objective
    if config.kl:
        uniform = -math.log(log_probs.shape[-1])
        loss = loss + config.kl * (log_probs.exp() * (log_probs - uniform)).sum(-1).mean()
    return loss, info


def train_policy(rates, config, steps, batch, lr, seed):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    grid = report_grid(interior=config.reward == "log")
    policy = ReportPolicy(len(rates.rates), grid)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    eta = torch.tensor(rates.rates, dtype=torch.float64)
    history, consumed = [], 0
    for step in range(1, steps + 1):
        contexts, outcomes = rates.sample(batch, rng)
        if config.outcome_design == "independent":
            outcomes = (torch.rand(batch, config.group, generator=rng, dtype=torch.float64) < eta[contexts][:, None]).double()
        reference = policy.log_probs(contexts).detach() if config.clip is not None else None
        optimizer.zero_grad()
        loss, info = estimator_loss(policy, contexts, outcomes, eta, config, rng, reference)
        loss.backward()
        optimizer.step()
        consumed += info["labels_consumed"]
        if step % 100 == 0 or step == steps:
            e = policy.evaluate(eta)
            history.append({"step": step, "forecast_error_max": float(e["forecast_error"].max()),
                            "expected_brier_mean": float(e["expected_brier"].mean())})
    final = {k: v.tolist() for k, v in policy.evaluate(eta).items()}
    return {"final": final, "history": history, "labels_consumed": consumed, "config": config.__dict__,
            "rates": list(rates.rates), "seed": seed}


STUDY_RATES = {"on_grid": (.1, .3, .5, .7, .9), "off_grid": (.125, .333, .5, .667, .875)}


def study_configs():
    configs = []
    for group in (4, 16):
        for design in ("shared", "independent"):
            configs.append(EstimatorConfig("exact", group=group, outcome_design=design))
            for name in ("rloo", "group_mean", "grpo"):
                configs.append(EstimatorConfig(name, group=group, outcome_design=design))
    for name in ("rloo", "grpo"):
        configs.append(EstimatorConfig(name, group=16, kl=.01))
        configs.append(EstimatorConfig(name, group=16, clip=.2))
    return configs


def run_study(output, seeds=5, steps=1500, batch=256, lr=.1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, runs = [], []
    for rate_name, rates in STUDY_RATES.items():
        for config in study_configs():
            results = [train_policy(KnownRates(rates), config, steps, batch, lr, seed) for seed in range(seeds)]
            runs.extend({**r, "rates_name": rate_name} for r in results)
            errors = np.array([max(r["final"]["forecast_error"]) for r in results])
            briers = np.array([np.mean(r["final"]["expected_brier"]) for r in results])
            variances = np.array([np.mean(r["final"]["report_variance"]) for r in results])
            rows.append({"rates": rate_name, "estimator": config.name, "group": config.group,
                         "outcome_design": config.outcome_design, "kl": config.kl, "clip": config.clip,
                         "labels_consumed": results[0]["labels_consumed"],
                         "forecast_error_max_mean": float(errors.mean()), "forecast_error_max_sd": float(errors.std()),
                         "expected_brier_mean": float(briers.mean()), "report_variance_mean": float(variances.mean())})
    report = {"seeds": seeds, "steps": steps, "batch": batch, "lr": lr, "grid_resolution": .01, "rows": rows}
    (output / "results.json").write_text(json.dumps({**report, "runs": runs}, indent=2) + "\n")
    lines = ["| rates | estimator | group | outcomes | kl | clip | labels | max forecast error (mean ± sd) | expected Brier | report variance |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['rates']} | {r['estimator']} | {r['group']} | {r['outcome_design']} | {r['kl']} | {r['clip']} | "
                     f"{r['labels_consumed']} | {r['forecast_error_max_mean']:.4f} ± {r['forecast_error_max_sd']:.4f} | "
                     f"{r['expected_brier_mean']:.4f} | {r['report_variance_mean']:.5f} |")
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return report
