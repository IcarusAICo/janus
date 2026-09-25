"""Strict paired study analysis, computed from probabilities with common NLL floors.

No model loading, GPU work, network access, or file writes occur here. Stored NLLs
are deliberately ignored. The probability floor applies only inside log(p), with
no renormalization after flooring; accuracy, Brier, and calibration bins retain
the unfloored distribution. Literal probability-based NLL is null with an explicit
infinity flag whenever a positive target has zero probability.
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


NLL_FLOOR = 1e-12
SENSITIVITY_FLOORS = (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2)
MODES = ("raw", "calibrated")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _vector(value, size, distribution=False):
    if (not isinstance(value, list) or len(value) != size
            or any(type(v) not in (int, float) for v in value)):
        raise ValueError("Expected a numeric vector matching the option count")
    vector = np.asarray(value, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite numeric values")
    if distribution:
        if (np.any(vector < 0) or np.any(vector > 1)
                or not math.isclose(float(vector.sum()), 1., rel_tol=0., abs_tol=1e-5)):
            raise ValueError("Expected normalized nonnegative probabilities or targets")
        # Correct only accepted floating-point summation drift, before any flooring.
        vector = vector / vector.sum()
    return vector


def _load_predictions(path):
    payload = Path(path).read_bytes()
    records, formats = {}, set()
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(row, dict):
                raise ValueError("Expected a prediction object")
            for field in ("group_id", "question_id", "input_sha256"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"Missing or invalid {field}")
            if not re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"]):
                raise ValueError("Invalid full-input SHA256 signature")
            key = (row["group_id"], row["question_id"])
            if key in records:
                raise ValueError(f"Duplicate state/question pair: {key}")
            keys = row.get("keys")
            if (not isinstance(keys, list) or not keys
                    or any(not isinstance(k, str) or not k for k in keys)
                    or len(set(keys)) != len(keys)):
                raise ValueError("Expected distinct ordered option keys")
            size = len(keys)
            if type(row.get("cardinality")) is not int or row["cardinality"] != size:
                raise ValueError("Cardinality does not match the ordered keys")
            kind = row.get("kind")
            if (kind not in {"choice", "noul", "score"}
                    or kind == "choice" and not 1 <= size <= 255
                    or kind == "noul" and keys != ["false", "true"]
                    or kind == "score" and (not 2 <= size <= 10 or keys != [str(i) for i in range(size)])):
                raise ValueError("Invalid decision kind, cardinality, or ordinal/boolean key order")
            target = _vector(row.get("target"), size, distribution=True)
            logits = _vector(row.get("logits"), size)
            calibrated = _vector(row.get("probabilities"), size, distribution=True)
            remote = "returned_model" in row or "requested_model" in row
            if remote:
                if any(not isinstance(row.get(k), str) or not row[k]
                       for k in ("requested_model", "returned_model")):
                    raise ValueError("Incomplete remote model identity")
                raw = _vector(row.get("raw_probabilities", row["probabilities"]), size, distribution=True)
            else:
                with np.errstate(over="ignore", under="ignore"):
                    raw = np.exp(logits - logits.max())
                raw /= raw.sum()
            formats.add((remote, "raw_probabilities" in row))
            records[key] = {**row, "_target": target, "_raw": raw, "_calibrated": calibrated}
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not records:
        raise ValueError(f"No prediction records in {path}")
    if len(formats) != 1:
        raise ValueError(f"Mixed local/remote or raw/calibrated record schemas in {path}")
    remote, separate = next(iter(formats))
    metadata = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                "source": "remote" if remote else "local",
                "raw_probability_source": "raw_probabilities" if remote and separate else
                                          "probabilities" if remote else "softmax(logits)",
                "calibrated_probability_source": "probabilities",
                "has_separate_calibrated_probabilities": separate if remote else None}
    if remote:
        metadata["requested_models"] = sorted({r["requested_model"] for r in records.values()})
        metadata["returned_models"] = sorted({r["returned_model"] for r in records.values()})
    return records, metadata


def _observations(rows, mode):
    result = []
    for row in rows:
        p, y = row[f"_{mode}"], row["_target"]
        predicted = int(p.argmax())
        positive = y > 0
        zero = bool(np.any(p[positive] == 0))
        literal = None if zero else float(-np.dot(y[positive], np.log(p[positive])))
        sensitivity = {str(f): float(-np.dot(y, np.log(np.maximum(p, f)))) for f in SENSITIVITY_FLOORS}
        expected_level = float(np.dot(p, np.arange(len(p))))
        result.append({"predicted": predicted, "accuracy": float(y[predicted]),
                       "confidence": float(p[predicted]), "nll": sensitivity[str(NLL_FLOOR)],
                       "nll_sensitivity": sensitivity, "brier": float(np.square(p-y).sum()),
                       "literal_nll": literal, "zero_target": zero,
                       "score_mae": float(np.dot(y, np.abs(np.arange(len(p))-expected_level)))
                                    if row["kind"] == "score" else None})
    return result


def _summary(observations):
    count = len(observations)
    result = {"count": count, **{key: float(np.mean([r[key] for r in observations]))
                                for key in ("accuracy", "nll", "brier")}}
    reliability = []
    for i in range(10):
        members = [r for r in observations if min(int(r["confidence"]*10), 9) == i]
        reliability.append({"lower": i/10, "upper": (i+1)/10, "count": len(members),
                            "accuracy": float(np.mean([r["accuracy"] for r in members])) if members else None,
                            "confidence": float(np.mean([r["confidence"] for r in members])) if members else None})
    result["reliability"] = reliability
    result["ece"] = sum(b["count"]/count * abs(b["accuracy"]-b["confidence"])
                        for b in reliability if b["count"])
    ranked = sorted(observations, key=lambda r: r["confidence"], reverse=True)
    result["selective_risk"] = []
    for coverage in (.25, .5, .75, 1.):
        n = max(1, math.ceil(count*coverage))
        result["selective_risk"].append({"coverage": n/count, "count": n,
                                         "risk": 1-float(np.mean([r["accuracy"] for r in ranked[:n]]))})
    score_errors = [r["score_mae"] for r in observations if r["score_mae"] is not None]
    result.update(score_count=len(score_errors),
                  score_expected_level_mae=float(np.mean(score_errors)) if score_errors else None,
                  target_zero_probability_count=sum(r["zero_target"] for r in observations),
                  literal_nll_is_infinite=any(r["zero_target"] for r in observations))
    result["literal_nll"] = None if result["literal_nll_is_infinite"] else float(np.mean([r["literal_nll"] for r in observations]))
    result["nll_sensitivity"] = {str(f): float(np.mean([r["nll_sensitivity"][str(f)] for r in observations]))
                                 for f in SENSITIVITY_FLOORS}
    return result


def _paired(local, reference, rows, indices, samples, seed):
    groups = defaultdict(list)
    for i in indices:
        groups[rows[i]["group_id"]].append(i)
    group_indices = [groups[key] for key in sorted(groups)]
    sums = np.array([[sum(local[i][metric]-reference[i][metric] for i in ix)
                      for metric in ("nll", "accuracy")] for ix in group_indices])
    counts = np.array([len(ix) for ix in group_indices])
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples, 2), dtype=np.float64)
    for sample in range(samples):
        draw = rng.integers(0, len(counts), size=len(counts))
        estimates[sample] = sums[draw].sum(axis=0) / counts[draw].sum()
    difference = sums.sum(axis=0) / counts.sum()
    result = {metric: {"difference": float(difference[j]),
                       "lower": float(np.quantile(estimates[:, j], .025)),
                       "upper": float(np.quantile(estimates[:, j], .975)),
                       "favorable_direction": "negative" if metric == "nll" else "positive",
                       "count": int(counts.sum()), "groups": len(counts), "samples": samples, "seed": seed}
              for j, metric in enumerate(("nll", "accuracy"))}
    result["agreement"] = float(np.mean([local[i]["predicted"] == reference[i]["predicted"] for i in indices]))
    return result


def analyze_study(prediction_paths, reference="jev", bootstrap_samples=2000, seed=17):
    """Return a JSON-safe study report; every model must cover the same decisions.

    ``prediction_paths`` maps display names to prediction JSONL paths. The named
    reference is normally Jev. Remote records are identified by their model fields,
    with optional ``raw_probabilities`` preserving separately calibrated originals.
    Comparisons subtract the reference in the same mode: raw minus raw and
    calibrated minus calibrated. Confidence intervals resample whole source groups
    and retain decision weighting by dividing sampled group sums by sampled counts.
    """
    if (not isinstance(prediction_paths, dict) or not prediction_paths or reference not in prediction_paths
            or any(not isinstance(name, str) or not name for name in prediction_paths)):
        raise ValueError("Provide named prediction paths including the reference")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("bootstrap_samples must be positive and seed nonnegative")
    loaded = {name: _load_predictions(path) for name, path in prediction_paths.items()}
    reference_map = loaded[reference][0]
    keys = sorted(reference_map)
    for name, (records, _) in loaded.items():
        if set(records) != set(reference_map):
            missing, extra = len(set(reference_map)-set(records)), len(set(records)-set(reference_map))
            raise ValueError(f"{name}: decision pairs differ from {reference}: {missing} missing, {extra} extra; no intersection is allowed")
        for key in keys:
            if any(records[key][field] != reference_map[key][field]
                   for field in ("input_sha256", "target", "keys", "kind", "cardinality")):
                raise ValueError(f"{name}: full inputs, ordered keys, or targets differ for {key}")
    rows = [reference_map[key] for key in keys]
    domains, kinds = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        domains[row["question_id"].split(":", 1)[0]].append(i)
        kinds[row["kind"]].append(i)
    observations, models = {}, {}
    for name, (records, metadata) in loaded.items():
        model_rows = [records[key] for key in keys]
        observations[name] = {mode: _observations(model_rows, mode) for mode in MODES}
        model = {**metadata, **{mode: _summary(observations[name][mode]) for mode in MODES}}
        for field, groups in (("by_domain", domains), ("by_kind", kinds)):
            model[field] = {group: {mode: _summary([observations[name][mode][i] for i in ix]) for mode in MODES}
                            for group, ix in sorted(groups.items())}
        model["macro_domain"] = {mode: {"domains": len(domains),
            **{metric: float(np.mean([d[mode][metric] for d in model["by_domain"].values()]))
               for metric in ("accuracy", "nll", "brier", "ece")}} for mode in MODES}
        models[name] = model
    comparisons = {}
    for name in models:
        if name == reference:
            continue
        def compare(ix):
            return {mode: _paired(observations[name][mode], observations[reference][mode], rows, ix,
                                  bootstrap_samples, seed) for mode in MODES}
        comparisons[name] = {**compare(list(range(len(rows)))),
                             "by_domain": {domain: compare(ix) for domain, ix in sorted(domains.items())},
                             "by_kind": {kind: compare(ix) for kind, ix in sorted(kinds.items())}}
    return {"schema_version": 1, "reference": reference, "count": len(rows),
            "groups": len({r["group_id"] for r in rows}), "pairing": "exact full-input and target match; no intersections",
            "reference_has_separate_calibrated_probabilities": models[reference]["has_separate_calibrated_probabilities"],
            "nll_probability_floor": NLL_FLOOR, "nll_sensitivity_floors": list(SENSITIVITY_FLOORS),
            "nll_definition": "-sum(target * log(max(probability, floor))); no renormalization after flooring",
            "literal_nll_definition": "Probability-based NLL; null plus infinity flag for any positive-target zero probability",
            "probability_precision": "float64; accepted sum drift up to 1e-5 normalized before metrics",
            "score_mae_definition": "Mean target-weighted absolute error of the predicted expected zero-based level",
            "accuracy_definition": "Target probability of the first maximum-probability declared option",
            "macro_domain_definition": "Unweighted mean of domain metrics; overall summaries weight decisions",
            "difference_definition": f"local minus {reference}; negative NLL and positive accuracy favor the local model",
            "bootstrap": {"method": "paired source-group cluster percentile bootstrap of sums/counts",
                          "confidence": .95, "samples": bootstrap_samples, "seed": seed,
                          "scope": "evaluation-sample uncertainty, not training-seed or API-repeat variability"},
            "agreement_definition": "Secondary argmax agreement; source targets define correctness",
            "models": models, "comparisons": comparisons}


def render_markdown(report):
    """Render a compact report without hiding zero-probability NLL sensitivity."""
    reference = report["reference"]
    ref = report["models"][reference]
    lines = []
    for mode in MODES:
        count = ref[mode]["target_zero_probability_count"]
        if count:
            lines.append(f"**{reference} {mode} assigns zero probability to {count} positive-target decisions: literal NLL is infinite. Finite NLL comparisons depend on the probability floor.**")
    lines += [f"{report['count']} paired decisions in {report['groups']} source groups. Common NLL floor: {report['nll_probability_floor']:g}, without renormalizing after clipping.",
              "", f"Differences are local minus {reference}: negative NLL and positive accuracy favor the local model.", "",
              "| Model | Probabilities | Accuracy | NLL | Brier | ECE | Score MAE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        for mode in MODES:
            s = model[mode]
            mae = "—" if s["score_expected_level_mae"] is None else f"{s['score_expected_level_mae']:.4f}"
            lines.append(f"| {name} | {mode} | {s['accuracy']:.4f} | {s['nll']:.4f} | {s['brier']:.4f} | {s['ece']:.4f} | {mae} |")
    if report["comparisons"]:
        lines += ["", "| Model | Probabilities | NLL difference [95% CI] | Accuracy difference [95% CI] | Agreement |",
                  "|---|---|---:|---:|---:|"]
        for name, comparison in report["comparisons"].items():
            for mode in MODES:
                a, b = comparison[mode]["nll"], comparison[mode]["accuracy"]
                lines.append(f"| {name} | {mode} | {a['difference']:.4f} [{a['lower']:.4f}, {a['upper']:.4f}] | {b['difference']:.4f} [{b['lower']:.4f}, {b['upper']:.4f}] | {comparison[mode]['agreement']:.4f} |")
    lines += ["", "NLL sensitivity uses the same floor for every model; accuracy and Brier do not change.", "",
              "| Model | Probabilities | " + " | ".join(f"{f:g}" for f in SENSITIVITY_FLOORS) + " |",
              "|---|---|" + "---:|" * len(SENSITIVITY_FLOORS)]
    for name, model in report["models"].items():
        for mode in MODES:
            values = " | ".join(f"{model[mode]['nll_sensitivity'][str(f)]:.4f}" for f in SENSITIVITY_FLOORS)
            lines.append(f"| {name} | {mode} | {values} |")
    return "\n".join(lines) + "\n"
