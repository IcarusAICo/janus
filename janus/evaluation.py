"""Held-out calibration, controls, and per-decision artifacts."""

from dataclasses import replace
import json
import hashlib
import math
from pathlib import Path
import random
import re

import torch

from .calibration_sets import by_family_logits, family_of_request, fit_temperature_by_family
from .data import file_hash, load_requests, shuffled_options, state_hash, write_json, write_jsonl
from .metrics import (apply_histogram_binning, fit_histogram_binning, fit_temperature,
                      fit_temperature_by_cardinality, metrics, paired_nll_bootstrap)
from .packing import pack_request
from .schema import decode
from .synth.rankworlds import request_kind
from .training import collect_logits, load_checkpoint


def question_key(question_id):
    """Question id without a trailing hash segment, e.g. 'snli:relation:abc123' -> 'snli:relation'."""
    parts = question_id.split(":")
    if len(parts) > 2 and re.fullmatch(r"[0-9a-f]{8,}", parts[-1]):
        parts = parts[:-1]
    return ":".join(parts)


def check_unused(requests, metadata, calibration=None):
    used_ids = set(metadata.get("training_group_ids", [])) | set(metadata.get("selection_group_ids", []))
    used_states = set(metadata.get("training_state_hashes", [])) | set(metadata.get("selection_state_hashes", []))
    if calibration:
        used_ids.update(calibration["group_ids"])
        used_states.update(calibration["state_hashes"])
    if any(r.group_id in used_ids or state_hash(r.state) in used_states for r in requests):
        raise ValueError("Data overlap with training, model selection, or calibration observations")


def read_calibration(path, checkpoint_path):
    if path is None:
        return {"temperature": 1., "group_ids": [], "state_hashes": []}
    value = json.loads(Path(path).read_text())
    if value["checkpoint_sha256"] != file_hash(checkpoint_path):
        raise ValueError("Temperature was fitted for a different checkpoint")
    if not math.isfinite(value["temperature"]) or value["temperature"] <= 0:
        raise ValueError("Temperature must be finite and positive")
    if any(not math.isfinite(t) or t <= 0 for t in value.get("by_cardinality", {}).values()):
        raise ValueError("Per-cardinality temperatures must be finite and positive")
    if any(t is not None and not 0 <= t <= 1 for t in value.get("histogram", [])):
        raise ValueError("Histogram bin accuracies must lie in [0, 1]")
    for entry in value.get("by_family", {}).values():  # Phase 4 per-task entries (janus.calibration_sets); absent in older files
        if any(not math.isfinite(t) or t <= 0 for t in [entry["temperature"], *entry.get("by_cardinality", {}).values()]):
            raise ValueError("Per-family temperatures must be finite and positive")
    return value


def _binned_logits(logits, calibration):
    """Log of histogram-binned probabilities, so `metrics` can score the binned variant."""
    table = calibration["histogram"]
    return [apply_histogram_binning(z.softmax(-1), table).clamp_min(1e-12).log() for z in logits]


def _by_cardinality_logits(logits, calibration):
    table = calibration["by_cardinality"]
    return [z / table.get(str(len(z)), calibration["temperature"]) for z in logits]


def _posthoc_variants(logits, targets, calibration, families=None):
    """Metrics for the extra post-hoc calibrators, when the calibration file carries them."""
    out = {}
    if "by_cardinality" in calibration:
        out["calibrated_by_cardinality"] = metrics(_by_cardinality_logits(logits, calibration), targets)
    if "by_family" in calibration and families is not None:
        out["calibrated_by_family"] = metrics(by_family_logits(logits, families, calibration), targets)
    if "histogram" in calibration:
        out["histogram_binned"] = metrics(_binned_logits(logits, calibration), targets)
    return out


def calibrate(checkpoint_path, data_path, output, device="cpu", limit=None, **overrides):
    """`overrides` as in `evaluate` (ModelConfig fields at load, plus `group_tokens`); they change execution only."""
    if Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite calibration artifact: {output}")
    if "test" in Path(data_path).stem.lower():
        raise ValueError("Fit temperature on a calibration split, not test data")
    group_tokens = overrides.pop("group_tokens", 0)
    model, metadata = load_checkpoint(checkpoint_path, device, **overrides)
    requests = load_requests(data_path)
    check_unused(requests, metadata)
    if limit:
        requests = random.Random(17).sample(requests, min(limit, len(requests)))
    logits, targets = collect_logits(model, requests, group_tokens)
    temperature = fit_temperature(logits, targets)
    families = [family_of_request(r) for r in requests for _ in r.questions]
    result = {"temperature": temperature, "by_cardinality": fit_temperature_by_cardinality(logits, targets),
              "by_family": fit_temperature_by_family(logits, targets, families),
              "histogram": fit_histogram_binning(logits, targets), "checkpoint_sha256": file_hash(checkpoint_path),
              "data_sha256": file_hash(data_path), "requests": len(requests),
              "group_ids": sorted({r.group_id for r in requests}),
              "state_hashes": sorted({state_hash(r.state) for r in requests}),
              "before": metrics(logits, targets), "after": metrics(logits, targets, temperature)}
    write_json(output, result)
    return result


@torch.inference_mode()
def predict(checkpoint_path, request, device="cpu", calibration=None):
    temperature = read_calibration(calibration, checkpoint_path)["temperature"]
    model, _ = load_checkpoint(checkpoint_path, device)
    packed = pack_request(request, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs)
    return decode(request, [(z.float() / temperature).softmax(-1).cpu().tolist() for z in model(packed)])


def evaluate(checkpoint_path, data_path, output, calibration=None, device="cpu", limit=None, **overrides):
    """`group_tokens=N` (a keyword or `--override group_tokens=N`) caps the packed tokens per forward_many group, as in
    training; without it the checkpoint's training setting applies."""
    torch.set_num_threads(4)
    group_tokens = overrides.pop("group_tokens", None)
    model, metadata = load_checkpoint(checkpoint_path, device, **overrides)
    if group_tokens is None:
        group_tokens = (metadata.get("config") or {}).get("group_tokens", 0)
    cal = read_calibration(calibration, checkpoint_path)
    requests = load_requests(data_path)
    check_unused(requests, metadata, cal)
    if limit:
        requests = random.Random(17).sample(requests, min(limit, len(requests)))
    if len(requests) < 2:
        raise ValueError("Need at least two requests for a shuffled-state control")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    logits, targets = collect_logits(model, requests, group_tokens)
    # A cyclic permutation has no fixed points; every question sees another state.
    order = list(range(len(requests)))
    random.Random(17).shuffle(order)
    shifted = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    shuffled = [replace(r, state=requests[shifted[i]].state) for i, r in enumerate(requests)]
    control_logits, _ = collect_logits(model, shuffled, group_tokens)
    permuted = [shuffled_options(r, i + 17) for i, r in enumerate(requests)]
    permutation_logits, _ = collect_logits(model, permuted, group_tokens)
    uniform_logits = [torch.zeros_like(z) for z in logits]
    families = [family_of_request(r) for r in requests for _ in r.questions]
    records, by_kind, by_family, by_question, by_cardinality, by_kind_rel, by_kind_question = [], {}, {}, {}, {}, {}, {}
    order_tv, order_flips = [], []
    index = 0
    for request, permuted_request in zip(requests, permuted):
        family = request.group_id.split(":", 1)[0]
        # Relative-menu or rank request kind parsed from the state (e.g. rel:second_cheapest, rank:kth), else None.
        rel_kind = request_kind(request.state, family)
        for q, pq in zip(request.questions, permuted_request.questions):
            z, y = logits[index], targets[index]
            raw = metrics([z], [y])
            control = metrics([control_logits[index]], [y])
            calibrated = metrics([z], [y], cal["temperature"])
            variants = {name: value["nll"] for name, value in _posthoc_variants([z], [y], cal, [families[index]]).items()}
            p = z.softmax(-1)
            p_permuted = permutation_logits[index].softmax(-1)
            mapping = {o.key: i for i, o in enumerate(pq.options)}
            aligned = p_permuted[[mapping[o.key] for o in q.options]]
            tv = float((p - aligned).abs().sum() / 2)
            flip = bool(p.argmax() != aligned.argmax())
            if q.kind == "choice":
                order_tv.append(tv)
                order_flips.append(flip)
            signature = {"state": request.state, "kind": q.kind, "instructions": q.instructions,
                         "options": [(o.key, o.description) for o in q.options]}
            input_sha256 = hashlib.sha256(json.dumps(signature, sort_keys=True,
                                                     ensure_ascii=False).encode()).hexdigest()
            records.append({"group_id": request.group_id, "question_id": q.id, "kind": q.kind, "family": family,
                            "request_kind": rel_kind, "input_sha256": input_sha256,
                            "cardinality": len(q.options), "keys": [o.key for o in q.options],
                            "logits": z.tolist(), "target": y.tolist(),
                            "probabilities": (z / cal["temperature"]).softmax(-1).tolist(),
                            "nll": raw["nll"], "calibrated_nll": calibrated["nll"],
                            **{f"{name}_nll": value for name, value in variants.items()},
                            "control_nll": control["nll"], "uniform_nll": math.log(len(z)),
                            "order_total_variation": tv, "order_choice_flip": flip})
            by_kind.setdefault(q.kind, []).append(index)
            by_family.setdefault(family, []).append(index)
            by_question.setdefault(question_key(q.id), []).append(index)
            by_cardinality.setdefault(str(len(q.options)), []).append(index)
            if rel_kind is not None:
                by_kind_rel.setdefault(rel_kind, []).append(index)
                by_kind_question.setdefault(f"{rel_kind}|{question_key(q.id)}", []).append(index)
            index += 1

    def breakdown(groups):
        return {name: {"raw": metrics([logits[i] for i in ids], [targets[i] for i in ids]),
                       "calibrated": metrics([logits[i] for i in ids], [targets[i] for i in ids], cal["temperature"]),
                       **_posthoc_variants([logits[i] for i in ids], [targets[i] for i in ids], cal, [families[i] for i in ids])}
                for name, ids in groups.items()}

    report = {"checkpoint_sha256": file_hash(checkpoint_path), "data_sha256": file_hash(data_path),
              "requests": len(requests), "temperature": cal["temperature"],
              "raw": metrics(logits, targets), "calibrated": metrics(logits, targets, cal["temperature"]),
              **_posthoc_variants(logits, targets, cal, families),
              "uniform": metrics(uniform_logits, targets), "shuffled_state": metrics(control_logits, targets),
              "by_kind": breakdown(by_kind), "by_family": breakdown(by_family), "by_question": breakdown(by_question), "by_cardinality": breakdown(by_cardinality),
              "by_kind_rel": breakdown(by_kind_rel), "by_kind_question": breakdown(by_kind_question),
              "uniform_comparison": paired_nll_bootstrap([{**r, "control_nll": r["uniform_nll"]} for r in records]),
              "shuffled_state_comparison": paired_nll_bootstrap(records),
              "choice_order": {"count": len(order_tv), "mean_total_variation": sum(order_tv) / len(order_tv) if order_tv else None,
                               "flip_rate": sum(order_flips) / len(order_flips) if order_flips else None}}
    write_jsonl(output / "predictions.jsonl", records)
    write_json(output / "metrics.json", report)
    return report


def compare_runs(predictions_a, predictions_b):
    """Paired raw-NLL comparison; positive improvement favors run A."""
    def load(path):
        rows = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["group_id"], row["question_id"], row.get("input_sha256"))
            if key in rows:
                raise ValueError(f"Duplicate state/question pair in {path}")
            rows[key] = row
        return rows
    a, b = load(predictions_a), load(predictions_b)
    if not a or set(a) != set(b):
        raise ValueError("Paired comparison requires identical state/question groups and identical inputs")
    rows = []
    for key, row in a.items():
        other = b[key]
        if (not row.get("input_sha256") or row["input_sha256"] != other.get("input_sha256")
                or row["target"] != other["target"] or row["keys"] != other["keys"]):
            raise ValueError("Paired comparison requires identical full inputs and targets")
        rows.append({**row, "control_nll": other["nll"]})
    return paired_nll_bootstrap(rows)


def _paired_predictions(predictions_a, predictions_b):
    """Rows of A paired with B under the same checks as compare_runs; no metric is chosen here."""
    def load(path):
        rows = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["group_id"], row["question_id"], row.get("input_sha256"))
            if key in rows:
                raise ValueError(f"Duplicate state/question pair in {path}")
            rows[key] = row
        return rows
    a, b = load(predictions_a), load(predictions_b)
    if not a or set(a) != set(b):
        raise ValueError("Paired comparison requires identical state/question groups and identical inputs")
    pairs = []
    for key, row in a.items():
        other = b[key]
        if (not row.get("input_sha256") or row["input_sha256"] != other.get("input_sha256")
                or row["target"] != other["target"] or row["keys"] != other["keys"]):
            raise ValueError("Paired comparison requires identical full inputs and targets")
        pairs.append((row, other))
    return pairs


def _correct(row):
    p = row["probabilities"]
    return float(row["target"][max(range(len(p)), key=p.__getitem__)])


def compare_accuracy(predictions_a, predictions_b):
    """Paired accuracy comparison; positive improvement means run A is more often correct.

    Same pairing rules as compare_runs and the same source-group bootstrap, applied to
    correct = target[argmax(probabilities)] instead of NLL."""
    rows = [{"group_id": row["group_id"], "nll": -_correct(row), "control_nll": -_correct(other)}
            for row, other in _paired_predictions(predictions_a, predictions_b)]
    return paired_nll_bootstrap(rows)
