"""Proper scoring rules, reliability, and calibration on disjoint observations."""

import math

import numpy as np
import torch
from torch.nn import functional as F


def validated_pairs(logits, targets):
    if not logits or len(logits) != len(targets):
        raise ValueError("Need matching nonempty logits and targets")
    pairs = []
    for z, y in zip(logits, targets):
        z = torch.as_tensor(z, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32, device=z.device)
        if (z.ndim != 1 or y.ndim != 1 or z.shape != y.shape or z.numel() == 0
                or not torch.isfinite(z).all() or not torch.isfinite(y).all()
                or (y < 0).any() or not torch.isclose(y.sum(), y.new_tensor(1.), atol=1e-5, rtol=0)):
            raise ValueError("Expected finite logits and matching normalized target distributions")
        pairs.append((z, y))
    return pairs


def distribution_loss(logits, targets, objective="ce", label_smoothing=0.):
    """Mean proper-score loss over questions: CE, Brier, or spherical (-(p . y) / ||p||).

    Label smoothing replaces y by (1 - s) y + s / K before scoring."""
    if objective not in {"ce", "brier", "spherical"}:
        raise ValueError("objective must be ce, brier or spherical")
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    losses = []
    for z, y in validated_pairs(logits, targets):
        if label_smoothing:
            y = (1 - label_smoothing) * y + label_smoothing / y.numel()
        if objective == "ce":
            losses.append(-(y * F.log_softmax(z.float(), dim=-1)).sum())
        elif objective == "brier":
            losses.append((z.float().softmax(-1) - y).square().sum())
        else:
            p = z.float().softmax(-1)
            losses.append(-(p * y).sum() / p.norm().clamp_min(1e-12))
    return torch.stack(losses).mean()


def metrics(logits, targets, temperature=1.0, bins=10):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    observations = []
    for z, y in validated_pairs(logits, targets):
        z = z.detach().cpu() / temperature
        y = y.detach().cpu()
        p = z.softmax(-1)
        # For soft labels, accuracy is expected correctness under the target.
        observations.append({"confidence": float(p.max()), "correct": float(y[p.argmax()]),
                             "nll": float(-(y * z.log_softmax(-1)).sum()),
                             "brier": float((p - y).square().sum())})
    n = len(observations)
    reliability = []
    for i in range(bins):
        members = [r for r in observations if min(int(r["confidence"] * bins), bins - 1) == i]
        reliability.append({"lower": i / bins, "upper": (i + 1) / bins, "count": len(members),
                            "accuracy": sum(r["correct"] for r in members) / len(members) if members else None,
                            "confidence": sum(r["confidence"] for r in members) / len(members) if members else None})
    ordered = sorted(observations, key=lambda r: r["confidence"], reverse=True)
    selective = []
    for fraction in (.25, .5, .75, 1.):
        count = max(1, math.ceil(n * fraction))
        selective.append({"coverage": count / n, "count": count,
                          "risk": 1 - sum(r["correct"] for r in ordered[:count]) / count})
    return {"count": n, "accuracy": sum(r["correct"] for r in observations) / n,
            "nll": sum(r["nll"] for r in observations) / n,
            "brier": sum(r["brier"] for r in observations) / n,
            "ece": sum(b["count"] / n * abs(b["accuracy"] - b["confidence"])
                       for b in reliability if b["count"]),
            "reliability": reliability, "selective_risk": selective}


def fit_temperature(logits, targets):
    pairs = validated_pairs(logits, targets)
    # Padding allows a single vectorized optimization across dynamic option counts.
    width = max(len(z) for z in logits)
    z = torch.full((len(logits), width), -1e9, dtype=torch.float64)
    y = torch.zeros_like(z)
    for i, (zi, yi) in enumerate(pairs):
        z[i, :len(zi)] = torch.as_tensor(zi).detach().cpu().double()
        y[i, :len(yi)] = torch.as_tensor(yi).detach().cpu().double()
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=.5, max_iter=60, line_search_fn="strong_wolfe")

    def objective():
        return -(y * (z / log_t.clamp(-4.6, 4.6).exp()).log_softmax(-1)).sum(-1).mean()

    initial = float(objective().detach())

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    value = float(log_t.detach().clamp(-4.6, 4.6).exp())
    return value if math.isfinite(value) and float(objective().detach()) <= initial else 1.0


def fit_temperature_by_cardinality(logits, targets, minimum=20):
    """One temperature per option count, keyed by str(K); sparse cardinalities use the global fit."""
    pairs = validated_pairs(logits, targets)
    global_t = fit_temperature(logits, targets)
    by_k = {}
    for k in sorted({len(z) for z, _ in pairs}):
        subset = [(z, y) for z, y in pairs if len(z) == k]
        by_k[str(k)] = (fit_temperature([z for z, _ in subset], [y for _, y in subset])
                        if len(subset) >= minimum else global_t)
    return by_k


def _bin_index(confidence, bins):
    return min(int(float(confidence) * bins), bins - 1)


def fit_histogram_binning(logits, targets, bins=10):
    """Empirical top-label accuracy per confidence bin; None where a bin is empty."""
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    table = [[] for _ in range(bins)]
    for z, y in validated_pairs(logits, targets):
        p = z.detach().cpu().softmax(-1)
        table[_bin_index(p.max(), bins)].append(float(y[p.argmax()]))
    return [sum(v) / len(v) if v else None for v in table]


def apply_histogram_binning(probabilities, table):
    """Set the top probability to its bin accuracy; spread the remainder over the other options in proportion."""
    p = torch.as_tensor(probabilities, dtype=torch.float32).detach().cpu().clone()
    if p.ndim != 1 or p.numel() == 0 or not table:
        raise ValueError("Expected a nonempty probability vector and a nonempty binning table")
    top = int(p.argmax())
    value = table[_bin_index(p[top], len(table))]
    if value is None or p.numel() == 1:
        return p
    others = torch.arange(p.numel()) != top
    rest = float(p[others].sum())
    out = p.clone()
    out[top] = value
    if rest > 0:
        out[others] = p[others] * ((1 - value) / rest)
    else:
        out[others] = (1 - value) / (p.numel() - 1)
    return out


def paired_nll_bootstrap(rows, samples=1000, seed=17):
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row["control_nll"] - row["nll"])
    if not groups:
        raise ValueError("Bootstrap requires nonempty state groups")
    sums = np.array([np.sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(sums), size=len(sums)) for _ in range(samples)]
    estimates = np.array([sums[ix].sum() / counts[ix].sum() for ix in draws])
    return {"groups": len(sums), "mean_improvement": float(sums.sum() / counts.sum()),
            "lower": float(np.quantile(estimates, .025)),
            "upper": float(np.quantile(estimates, .975)), "samples": samples}
