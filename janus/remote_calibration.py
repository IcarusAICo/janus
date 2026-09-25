"""Fit one Jev temperature on disjoint calibration observations, using CPU only.

Raw API probabilities are retained. Calibration starts from log(max(p, 1e-12)),
matching the raw remote evaluation convention. Only calibration labels enter the
temperature optimizer; panel labels are checked for artifact alignment, not fit.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from .data import file_hash, load_requests, state_hash
from .metrics import fit_temperature, metrics
from .remote import DEFAULT_MODEL, _atomic_json, _atomic_jsonl

LOG_PROBABILITY_FLOOR = 1e-12


def validate_calibration_splits(calibration_data_path, panel_data_path):
    """Check normalized source states and source groups before any paid requests."""
    calibration = load_requests(calibration_data_path)
    panel = load_requests(panel_data_path)
    calibration_groups = {r.group_id for r in calibration}
    panel_groups = {r.group_id for r in panel}
    calibration_states = {state_hash(r.state) for r in calibration}
    panel_states = {state_hash(r.state) for r in panel}
    if calibration_groups & panel_groups or calibration_states & panel_states:
        raise ValueError("Calibration and panel source groups or normalized states overlap")
    if len(calibration_states) != len(calibration) or len(panel_states) != len(panel):
        raise ValueError("Calibration and panel must each contain unique normalized states")
    return calibration, panel


def _signature(request, question):
    value = {"state": request.state, "kind": question.kind, "instructions": question.instructions,
             "options": [(o.key, o.description) for o in question.options]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _validated_predictions(requests, prediction_path, model):
    try:
        rows = [json.loads(line) for line in Path(prediction_path).read_text().splitlines() if line.strip()]
        indexed = {(row["group_id"], row["question_id"]): row for row in rows}
    except (ValueError, KeyError, TypeError):
        raise ValueError("Invalid remote prediction records") from None
    expected = {(request.group_id, q.id): (request, q) for request in requests for q in request.questions}
    if (len(indexed) != len(rows) or len(expected) != sum(len(r.questions) for r in requests)
            or set(indexed) != set(expected)):
        raise ValueError("Remote predictions require identical unique state/question pairs")
    for key, row in indexed.items():
        request, q = expected[key]
        if row.get("requested_model") != model or row.get("returned_model") != model:
            raise ValueError("Calibration and panel must use the identical requested and returned model version")
        if "raw_probabilities" in row or "calibration_temperature" in row:
            raise ValueError("Expected raw remote predictions, not previously calibrated records")
        if (q.target is None or row.get("target") != torch.tensor(q.target, dtype=torch.float32).tolist()
                or row.get("input_sha256") != _signature(request, q)
                or row.get("keys") != [o.key for o in q.options]
                or row.get("cardinality") != len(q.options) or row.get("kind") != q.kind):
            raise ValueError("Prediction input signature, target, or ordered menu differs from source data")
        p = row.get("probabilities")
        if (not isinstance(p, list) or len(p) != len(q.options)
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in p)
                or not math.isclose(sum(p), 1., rel_tol=0, abs_tol=1e-8)):
            raise ValueError("Expected complete finite normalized raw probabilities")
        logits = [math.log(max(value, LOG_PROBABILITY_FLOOR)) for value in p]
        if row.get("logits") != logits:
            raise ValueError("Raw remote logits must equal log(max(probability, 1e-12))")
    return rows


def calibrate_remote(calibration_data_path, calibration_predictions_path, panel_data_path,
                     panel_predictions_path, output_predictions, artifact_path=None,
                     model=DEFAULT_MODEL):
    """Fit on calibration labels and write separate calibrated panel predictions.

    ``nll`` and ``logits`` remain raw, consistent with local evaluation records.
    ``probabilities`` and ``calibrated_nll`` contain calibrated results, alongside
    ``raw_probabilities`` and ``calibration_temperature``. Before/after metrics in
    the returned artifact refer only to the calibration split. Both input datasets
    and raw prediction files, and the resulting output, are bound by SHA256.
    """
    output_predictions = Path(output_predictions)
    artifact_path = Path(artifact_path) if artifact_path is not None else output_predictions.parent / "calibration.json"
    inputs = [Path(path) for path in (calibration_data_path, calibration_predictions_path,
                                      panel_data_path, panel_predictions_path)]
    if output_predictions.resolve() == artifact_path.resolve() or any(
            output.resolve() == source.resolve() for output in (output_predictions, artifact_path) for source in inputs):
        raise ValueError("Calibration outputs must be separate from all inputs and each other")
    for path in (output_predictions, artifact_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite calibration artifact: {path}")
    calibration_requests, panel_requests = validate_calibration_splits(calibration_data_path, panel_data_path)
    calibration_rows = _validated_predictions(calibration_requests, calibration_predictions_path, model)
    panel_rows = _validated_predictions(panel_requests, panel_predictions_path, model)
    # The optimizer receives only these calibration tensors, never panel labels.
    calibration_logits = [torch.tensor(row["logits"], dtype=torch.float32, device="cpu") for row in calibration_rows]
    calibration_targets = [torch.tensor(row["target"], dtype=torch.float32, device="cpu") for row in calibration_rows]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        temperature = fit_temperature(calibration_logits, calibration_targets)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Fitted calibration temperature must be finite and positive")
        before = metrics(calibration_logits, calibration_targets)
        after = metrics(calibration_logits, calibration_targets, temperature)
        calibrated = []
        for row in panel_rows:
            z = torch.tensor(row["logits"], dtype=torch.float32, device="cpu") / temperature
            target = torch.tensor(row["target"], dtype=torch.float32, device="cpu")
            calibrated.append({**row, "raw_probabilities": list(row["probabilities"]),
                               "probabilities": z.softmax(-1).tolist(),
                               "calibration_temperature": temperature,
                               "calibrated_nll": float(-(target * z.log_softmax(-1)).sum())})
    finally:
        torch.set_num_threads(previous_threads)
    artifact = {"version": "1", "method": "single_scalar_temperature", "temperature": temperature,
                "requested_model": model, "returned_model": model,
                "log_probability_floor": LOG_PROBABILITY_FLOOR, "fit_device": "cpu", "fit_threads": 4,
                "metric_split": "calibration", "requests": len(calibration_requests),
                "questions": len(calibration_rows), "panel_requests": len(panel_requests),
                "panel_questions": len(panel_rows),
                "group_ids": sorted({r.group_id for r in calibration_requests}),
                "state_hashes": sorted({state_hash(r.state) for r in calibration_requests}),
                "panel_group_ids": sorted({r.group_id for r in panel_requests}),
                "panel_state_hashes": sorted({state_hash(r.state) for r in panel_requests}),
                "calibration_data_sha256": file_hash(calibration_data_path),
                "calibration_predictions_sha256": file_hash(calibration_predictions_path),
                "panel_data_sha256": file_hash(panel_data_path),
                "panel_predictions_sha256": file_hash(panel_predictions_path),
                "before": before, "after": after}
    _atomic_jsonl(output_predictions, calibrated)
    artifact["calibrated_predictions_sha256"] = file_hash(output_predictions)
    _atomic_json(artifact_path, artifact)
    return artifact
