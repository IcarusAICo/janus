"""Reconstruct exact state exposure from completed deterministic training runs.

No model is constructed. Checkpoints are loaded on CPU with weights_only=True;
their bytes are never rewritten, preserving any calibration binding to best.pt.
"""

from collections import Counter, defaultdict
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import random

from .data import file_hash, load_requests, state_hash, write_json


def _digest(values):
    return hashlib.sha256("\n".join(sorted(set(values))).encode()).hexdigest()


def _identities(requests):
    return [(r.group_id.split(":", 1)[0], state_hash(r.state), r.group_id) for r in requests]


def _ordered(identities, seed, epochs):
    rng = random.Random(seed)
    for _ in range(epochs):
        order = list(range(len(identities)))
        rng.shuffle(order)
        for index in order:
            yield identities[index]


def _summarize(presentations, domains):
    counts, states, groups = Counter(), defaultdict(set), defaultdict(set)
    for domain, state, group in presentations:
        counts[domain] += 1
        states[domain].add(state)
        groups[domain].add(group)
    all_states = set().union(*states.values()) if states else set()
    all_groups = set().union(*groups.values()) if groups else set()
    by_domain = {domain: {"presented_states": counts[domain], "unique_states": len(states[domain]),
                          "unique_groups": len(groups[domain]), "state_hashes_sha256": _digest(states[domain]),
                          "group_ids_sha256": _digest(groups[domain])} for domain in sorted(domains)}
    return {"presented_states": sum(counts.values()), "requests_seen": sum(counts.values()),
            "unique_states": len(all_states), "unique_groups": len(all_groups),
            "state_hashes_sha256": _digest(all_states), "group_ids_sha256": _digest(all_groups),
            "domains": by_domain}, all_states, all_groups


def _int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def audit_exposure(run_dir, train_path, output):
    """Audit selected-checkpoint and final exposure from completed run artifacts.

    summary.json is required; an active run is not silently treated as completed.
    Any inconsistent recorded hash, counter, pool, or consumed set raises before
    writing an output. Missing legacy consumed-set fields are reconstructed and
    explicitly marked as such. No source arrays are emitted into the report.
    """
    import torch

    run, train_path, output = Path(run_dir), Path(train_path), Path(output)
    paths = {name: run / name for name in ("config.json", "summary.json", "history.json", "best.pt")}
    if output.resolve() in {path.resolve() for path in (*paths.values(), train_path)}:
        raise ValueError("Exposure output cannot overwrite a training artifact")
    for name in ("config.json", "summary.json", "best.pt"):
        if not paths[name].is_file():
            raise ValueError(f"Completed training artifact is missing: {paths[name]}")
    before = {name: file_hash(path) for name, path in paths.items() if path.is_file()}
    saved = json.loads(paths["config.json"].read_text())
    config = saved.get("config", saved)
    summary = json.loads(paths["summary.json"].read_text())
    history = json.loads(paths["history.json"].read_text()) if paths["history.json"].is_file() else []
    checkpoint = torch.load(paths["best.pt"], map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    del checkpoint
    train_digest = file_hash(train_path)
    for label, document in (("config", saved), ("checkpoint", metadata)):
        if "train_sha256" in document and document["train_sha256"] != train_digest:
            raise ValueError(f"{label} train_sha256 disagrees with the supplied training file")
    sampling_defaults = {"seed": 17, "data_seed": 17, "train_limit": None, "epochs": 3,
                         "accumulation": 8, "max_steps": None, "max_seconds": None}
    sampling = {key: config.get(key, default) for key, default in sampling_defaults.items()}
    if "config" in metadata:
        for key, default in sampling_defaults.items():
            if metadata["config"].get(key, default) != sampling[key]:
                raise ValueError(f"Checkpoint and run sampling configurations disagree on {key}")
    epochs = _int(sampling["epochs"], "epochs", 1)
    accumulation = _int(sampling["accumulation"], "accumulation", 1)
    if sampling["max_steps"] is not None:
        _int(sampling["max_steps"], "max_steps", 1)
    if sampling["train_limit"] is not None:
        _int(sampling["train_limit"], "train_limit", 1)
    full_data = load_requests(train_path)
    if sampling["train_limit"]:
        # Reuse the trainer's public sampler: domain insertion order and pop order
        # are part of its seeded behavior, not interchangeable with a random sample.
        from .training import balanced_subset
        training_data = balanced_subset(full_data, sampling["train_limit"], sampling["data_seed"])
    else:
        training_data = full_data
    identities = _identities(training_data)
    full_identities = _identities(full_data)
    domains = {row[0] for row in full_identities}
    pool_summary, pool_states, pool_groups = _summarize(identities, domains)
    full_summary, _, _ = _summarize(full_identities, domains)
    pool_size = len(identities)
    if "train_requests" in summary and summary["train_requests"] != pool_size:
        raise ValueError("Summary train_requests disagrees with the reconstructed training pool")
    for label, document in (("config", saved), ("checkpoint", metadata)):
        for key, expected in (("training_group_ids", pool_groups), ("training_state_hashes", pool_states)):
            if key in document and set(document[key]) != expected:
                raise ValueError(f"{label} {key} disagrees with the reconstructed training pool")
    steps_per_epoch = math.ceil(pool_size / accumulation)
    uncapped_steps = steps_per_epoch * epochs
    planned_steps = min(uncapped_steps, sampling["max_steps"] or uncapped_steps)
    selected_step = _int(metadata["step"], "selected checkpoint step")
    final_step = _int(summary["steps"], "finished steps")
    if not selected_step <= final_step <= planned_steps:
        raise ValueError("Selected/final steps are outside the configured training plan")
    if summary.get("best_step", selected_step) != selected_step:
        raise ValueError("Summary best_step disagrees with the selected checkpoint")

    def count_at_step(step):
        completed, remainder = divmod(step, steps_per_epoch)
        return completed * pool_size + min(remainder * accumulation, pool_size)

    def verify_count(document, expected, label):
        if "requests_seen" in document and document["requests_seen"] != expected:
            raise ValueError(f"{label} requests_seen disagrees with its step and batch schedule")

    selected_count, final_count, planned_count = (count_at_step(step) for step in (selected_step, final_step, planned_steps))
    verify_count(metadata, selected_count, "Checkpoint")
    verify_count(summary, final_count, "Summary")
    for row in history:
        step = _int(row["step"], "history step")
        if step > final_step:
            raise ValueError("History contains a step after the finished summary")
        verify_count(row, count_at_step(step), "History")

    def exposure(count, step):
        result, states, groups = _summarize(islice(_ordered(identities, sampling["seed"], epochs), count), domains)
        if result["requests_seen"] != count:
            raise ValueError("Requested training prefix exceeds the epoch schedule")
        result.update(step=step, equivalent_pool_passes=count / pool_size,
                      training_pool_coverage=len(states) / len(pool_states),
                      full_pool_coverage=len(states) / full_summary["unique_states"])
        return result, states, groups

    selected, selected_states, selected_groups = exposure(selected_count, selected_step)
    final, final_states, final_groups = exposure(final_count, final_step)
    planned, _, _ = exposure(planned_count, planned_steps)
    verification = {}
    for key, expected in (("seen_group_ids", selected_groups), ("seen_state_hashes", selected_states)):
        if key in metadata:
            if set(metadata[key]) != expected:
                raise ValueError(f"Checkpoint {key} disagrees with reconstructed consumed states")
            verification[key] = "verified"
        else:
            verification[key] = "reconstructed_missing_metadata"
    for key, expected in (("unique_groups_seen", len(final_groups)), ("unique_states_seen", len(final_states))):
        if key in summary and summary[key] != expected:
            raise ValueError(f"Summary {key} disagrees with reconstructed final exposure")
        verification[key] = "verified" if key in summary else "reconstructed_missing_metadata"
    verification.update(requests_seen_checkpoint="verified" if "requests_seen" in metadata else "reconstructed_from_step",
                        requests_seen_summary="verified" if "requests_seen" in summary else "reconstructed_from_step",
                        history_rows_checked=len(history), checkpoint_unchanged=True)
    step_cap_reached = bool(sampling["max_steps"] and final_step >= sampling["max_steps"])
    time_cap_reached = bool(summary.get("budget_limited", False))
    complete = final_step == planned_steps
    reason = ("max_steps" if complete and step_cap_reached and planned_steps < uncapped_steps
              else "epochs" if final_step == uncapped_steps
              else "max_seconds" if time_cap_reached else "incomplete_or_other")
    planned.update(steps=planned_steps, uncapped_steps=uncapped_steps, uncapped_requests_seen=pool_size * epochs,
                   steps_per_epoch=steps_per_epoch, epochs=epochs,
                   scope="Configured epochs and max_steps; wall-time cap can reduce actual exposure")
    report = {"schema_version": 1, "run_dir": str(run), "train_path": str(train_path),
              "train_sha256": train_digest, "artifact_sha256": before, "sampling_config": sampling,
              "full_pool_size": len(full_data), "full_pool": full_summary, "training_pool": pool_summary,
              "selected_step": selected_step, "actual_finished_steps": final_step,
              "selected_checkpoint": selected, "final": final, "planned": planned,
              "cap_status": {"max_steps": sampling["max_steps"], "max_seconds": sampling["max_seconds"],
                             "step_cap_reached": step_cap_reached, "time_cap_reached": time_cap_reached,
                             "configured_plan_completed": complete,
                             "epoch_plan_completed": final_step == uncapped_steps,
                             "stopped_before_configured_plan": final_step < planned_steps,
                             "reason": reason},
              "verification": verification,
              "definitions": {"presented_states": "Request presentations including repeats across epochs; excludes development/calibration/test",
                              "unique_states": "Number of distinct normalized state hashes actually consumed",
                              "unique_groups": "Number of distinct source group IDs consumed; multiple SNLI hypotheses may share one group",
                              "state_hashes_sha256": "SHA256 of sorted distinct normalized-state hashes joined by newline, without trailing newline",
                              "group_ids_sha256": "SHA256 of sorted distinct group IDs joined by newline, without trailing newline",
                              "reconstruction": "balanced_subset with data_seed, then persistent random.Random(seed) shuffling each epoch; augmentation preserves input order/state",
                              "requests_seen": "Recorded counter checked against step schedule including each epoch's final partial batch; inferred from step only if absent",
                              "time_cap_reached": "Reported summary budget_limited flag; final evaluation can carry elapsed time past the cap",
                              "planned": "Exposure at the configured epochs/max_steps boundary; does not predict wall-clock throughput"}}
    for name, digest in before.items():
        if file_hash(paths[name]) != digest:
            raise ValueError(f"Training artifact changed during audit: {name}; wait for the run to finish")
    if file_hash(train_path) != train_digest:
        raise ValueError("Training input changed during exposure audit")
    write_json(output, report)
    return report
