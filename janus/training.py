"""Single-device training and portable selected checkpoints. No decoding loop."""

from dataclasses import asdict, dataclass, field, replace
import functools
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import (assert_disjoint, file_hash, load_requests, shuffled_options, state_hash, training_epoch,
                   training_sources, write_json)
from .metrics import distribution_loss, metrics
from .model import DecisionModel, ModelConfig
from .packing import pack_request


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    device: str = "cuda:0"
    seed: int = 17
    epochs: int = 3
    accumulation: int = 8
    backbone_lr: float = 2e-4
    head_lr: float = 1e-3
    weight_decay: float = .01
    objective: str = "ce"
    eval_every: int = 100
    max_steps: int | None = None
    dev_limit: int | None = None
    train_limit: int | None = None
    data_seed: int = 17  # train_limit/dev_limit subsets only; the epoch order follows `seed` (change `seed` to reorder a continuation)
    warmup_steps: int = 0
    cosine_decay: bool = False
    max_seconds: float | None = None
    threads: int = 4
    # WP2 objective-screen arms; defaults reproduce the Phase 1 winner exactly.
    label_smoothing: float = 0.
    consistency_weight: float = 0.
    # Phase 4: split an accumulation batch into forward_many groups by summed packed tokens instead of row count
    # (0 = groups of exactly `model.batch_states` rows, the old behaviour). Memory scales with the tokens in a group,
    # so a fixed row count OOMs on the long-row groups the length sort creates (docs/phase4/training-throughput.md).
    group_tokens: int = 0
    # ponytail: True forwards each option-permuted consistency twin on its own instead of inside the group's
    # forward_many; same gradient, one backbone pass per twin more. Kept only for the equivalence test.
    _consistency_separate: bool = False
    complement_augment: bool = False
    none_rate: float = .2
    # WP5a: masked-next-token warm-up steps on state text before decision training (0 = none).
    mntp_steps: int = 0
    # Phase 4 honesty: "dev_nll" (default) or "dev_nll_plus_heldout", which scores `heldout_dev` (a withheld-family
    # file) at every evaluation and selects on dev_nll + heldout_weight * max(0, heldout_nll - uniform_nll).
    selection: str = "dev_nll"
    heldout_dev: str | None = None
    heldout_weight: float = 1.
    # Phase 4 browser cell: continue from a checkpoint's adapters and head (same ModelConfig) instead of fresh init.
    init_checkpoint: str | None = None

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw)
        raw.pop("winner_run", None)
        raw["model"] = ModelConfig(**raw.get("model", {}))
        return cls(**raw)

    def validate(self):
        if self.objective not in {"ce", "brier", "spherical"}:
            raise ValueError("objective must be ce, brier or spherical")
        if not 0 <= self.label_smoothing < 1 or self.consistency_weight < 0 or not 0 <= self.none_rate <= 1:
            raise ValueError("Invalid training arm settings: label_smoothing in [0, 1), "
                             "consistency_weight >= 0, none_rate in [0, 1]")
        if self.mntp_steps < 0 or self.group_tokens < 0:
            raise ValueError("mntp_steps and group_tokens must be non-negative")
        if self.selection not in {"dev_nll", "dev_nll_plus_heldout"}:
            raise ValueError("selection must be dev_nll or dev_nll_plus_heldout")
        if self.selection == "dev_nll_plus_heldout" and (not self.heldout_dev or self.heldout_weight < 0):
            raise ValueError("dev_nll_plus_heldout needs a heldout_dev path and heldout_weight >= 0")
        return self


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def balanced_subset(requests, limit, seed=17):
    groups = {}
    for request in requests:
        groups.setdefault(request.group_id.split(':', 1)[0], []).append(request)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < min(limit, len(requests)):
        for domain in sorted(groups):
            if groups[domain] and len(selected) < limit:
                selected.append(groups[domain].pop())
    return selected


def token_groups(counts, max_rows, budget=0):
    """Consecutive [lo, hi) index ranges that share one `forward_many`: at most `max_rows` packs each and, when
    `budget` is set, at most `budget` summed tokens -- except a single pack over the budget, which rides alone.
    budget 0 yields exactly the fixed-size ranges of `max_rows`."""
    lo, total = 0, 0
    for i, count in enumerate(counts):
        if i > lo and (i - lo >= max_rows or (budget and total + count > budget)):
            yield lo, i
            lo, total = i, 0
        total += count
    if lo < len(counts):
        yield lo, len(counts)


@torch.no_grad()  # not inference_mode: remote backbones cache tables on first use and inference tensors break the next backward
def collect_logits(model, requests, group_tokens=0):
    model.eval()
    logits, targets = [], []
    packs = [pack_request(r, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs) for r in requests]
    group = max(1, getattr(model.config, "batch_states", 1))
    for lo, hi in token_groups([p.token_count for p in packs], group, group_tokens):
        for per_request in model.forward_many(packs[lo:hi]):
            logits.extend(z.detach().float().cpu() for z in per_request)
    for request in requests:
        for question in request.questions:
            if question.target is None:
                raise ValueError("Training/evaluation requires targets for all questions")
            targets.append(torch.tensor(question.target))
    return logits, targets


def checkpoint(model, path, metadata):
    # Tiny weights must all be stored: no pretrained random initialization exists to reload.
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {n: t.detach().cpu() for n, t in model.state_dict().items()
             if model.config.backbone.startswith("tiny") or n in trainable or model.config.adaptation == "full"}
    payload = {"weights": state, "metadata": {**metadata, "model": asdict(model.config)}}
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu", attention=None, **overrides):
    """`attention` and any other ModelConfig field in `overrides` (e.g. max_tokens, max_state_plus_question) replace
    the stored setting; they change execution, not the weights."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    settings = dict(metadata["model"])
    if attention is not None:
        settings["attention"] = attention
    settings.update({k: v for k, v in overrides.items() if v is not None})
    model = DecisionModel(ModelConfig(**settings))
    load_weights(model, payload["weights"])
    model.to(device).eval()
    return model, metadata


def load_weights(model, weights):
    """The stored (trainable) tensors into a model of the same configuration; anything missing or extra is an error."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys or trainable & set(result.missing_keys):
        raise ValueError("Checkpoint does not match the configured model")
    return model


def _consistency_term(request, permuted, logits, permuted_logits):
    """Mean symmetric KL between original and option-permuted views of each multi-option Choice question."""
    terms = []
    for q, pq, z, pz in zip(request.questions, permuted.questions, logits, permuted_logits):
        if q.kind != "choice" or len(q.options) < 2:
            continue
        mapping = {o.key: i for i, o in enumerate(pq.options)}
        aligned = pz[[mapping[o.key] for o in q.options]]
        lp, lq = z.float().log_softmax(-1), aligned.float().log_softmax(-1)
        terms.append(.5 * ((lp.exp() * (lp - lq)).sum() + (lq.exp() * (lq - lp)).sum()))
    return torch.stack(terms).mean() if terms else None


def selection_score(config, dev_nll, heldout_nll=None, uniform_nll=None):
    """The quantity checkpoint selection minimises. Under dev_nll_plus_heldout the held-out term penalises only
    confident error: NLL above the uniform reference on families the model has not seen."""
    if config.selection == "dev_nll":
        return dev_nll
    return dev_nll + config.heldout_weight * max(0., heldout_nll - uniform_nll)


def train(train_path, dev_path, output, config):
    config.validate()
    if config.epochs < 1 or config.accumulation < 1 or config.eval_every < 1:
        raise ValueError("epochs, accumulation and eval_every must be positive")
    if config.max_steps is not None and config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.dev_limit is not None and config.dev_limit < 1:
        raise ValueError("dev_limit must be positive")
    if config.train_limit is not None and config.train_limit < 1:
        raise ValueError("train_limit must be positive")
    seed_everything(config.seed, config.threads)
    train_data, dev_data = load_requests(train_path), load_requests(dev_path)
    assert_disjoint({"train": train_data, "dev": dev_data})
    sources, pool, data_manifest = training_sources(train_path)
    epoch_builder = training_epoch
    if data_manifest and data_manifest.get('dataset') == 'JEV_MULTI_DOMAIN_V1':
        from .study_data import study_training_sources, study_training_epoch
        sources, pool, data_manifest = study_training_sources(train_path)
        epoch_builder = functools.partial(study_training_epoch, omit=config.none_rate)
    if config.train_limit:
        train_data = balanced_subset(train_data, config.train_limit, config.data_seed)
        if sources is not None:
            ids = {r.group_id for r in train_data}
            sources = [s for s in sources if s['group_id'] in ids]
    if any(q.target is None for r in train_data for q in r.questions):
        raise ValueError("Training requests must have complete targets")
    # Do not permit one input file to masquerade as calibration or test.
    if any("test" in Path(p).stem.lower() or "calibration" in Path(p).stem.lower()
           for p in (train_path, dev_path, config.heldout_dev or "")):
        raise ValueError("Use train/dev splits for checkpoint selection")
    full_dev = dev_data
    heldout_data = []
    if config.selection == "dev_nll_plus_heldout":
        heldout_data = load_requests(config.heldout_dev)
        assert_disjoint({"train": train_data, "dev": dev_data, "heldout_dev": heldout_data})
    if config.dev_limit:
        dev_data = balanced_subset(dev_data, config.dev_limit, config.data_seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = DecisionModel(config.model).to(config.device)
    if config.init_checkpoint:
        load_weights(model, torch.load(config.init_checkpoint, map_location="cpu", weights_only=True)["weights"])
    warmup_history = None
    if config.mntp_steps:
        from .warmup import masked_next_token_warmup
        warmup_history = masked_next_token_warmup(model, train_data, config.mntp_steps, backbone_lr=config.backbone_lr,
                                                  head_lr=config.head_lr, seed=config.seed)
        write_json(output / "warmup.json", warmup_history)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config.backbone_lr},
        {"params": model.head.parameters(), "lr": config.head_lr}], weight_decay=config.weight_decay)
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
    metadata = {"config": asdict(config), "parameters": model.parameter_counts(), "data_manifest": data_manifest,
                "train_sha256": file_hash(train_path), "dev_sha256": file_hash(dev_path),
                "training_group_ids": sorted({r.group_id for r in train_data}),
                "heldout_sha256": file_hash(config.heldout_dev) if heldout_data else None,
                "selection_group_ids": sorted({r.group_id for r in full_dev + heldout_data}),
                "training_state_hashes": sorted({state_hash(r.state) for r in train_data}),
                "selection_state_hashes": sorted({state_hash(r.state) for r in full_dev + heldout_data}),
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")},
                "device": str(config.device), "device_name": (torch.cuda.get_device_name(model.device)
                           if model.device.type == "cuda" else "CPU")}
    write_json(output / "config.json", metadata)
    start = time.perf_counter()
    initial_z, initial_y = collect_logits(model, dev_data, config.group_tokens)
    initial = metrics(initial_z, initial_y)

    def heldout_terms():  # {} under dev_nll; else the held-out NLL, its uniform reference and the penalty
        if not heldout_data:
            return {}
        z, y = collect_logits(model, heldout_data, config.group_tokens)
        nll, uniform = metrics(z, y)["nll"], sum(math.log(len(t)) for t in y) / len(y)
        return {"heldout_nll": nll, "heldout_uniform_nll": uniform, "heldout_penalty": max(0., nll - uniform)}
    initial_heldout = heldout_terms()
    best_score = selection_score(config, initial["nll"], initial_heldout.get("heldout_nll"), initial_heldout.get("heldout_uniform_nll"))
    best_nll, best_step, step = initial["nll"], 0, 0
    requests_seen, seen_groups, seen_states, stop = 0, set(), set(), False
    checkpoint(model, output / "best.pt", {**metadata, "step": 0, "dev": initial,
                                          "requests_seen": 0, "seen_group_ids": [], "seen_state_hashes": []})
    history = [{"step": 0, "dev_nll": best_nll, "dev_accuracy": initial["accuracy"], **initial_heldout,
                "selection_score": best_score}]
    print(json.dumps(history[-1]), flush=True)
    rng = random.Random(config.seed)
    consistency_forwards = complement_added = 0
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    for epoch in range(config.epochs):
        epoch_data = epoch_builder(train_data, config.seed, epoch, sources, pool)
        order = list(range(len(train_data)))
        rng.shuffle(order)
        for offset in range(0, len(order), config.accumulation):
            batch = order[offset:offset + config.accumulation]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            batch_requests = []
            for index in batch:
                request = epoch_data[index]
                requests_seen += 1
                seen_groups.add(request.group_id)
                seen_states.add(state_hash(request.state))
                if config.complement_augment:
                    from .blocked_probes import negate_noul
                    extra = tuple(n for n in (negate_noul(q) for q in request.questions if q.kind == "noul")
                                  if n is not None)
                    if extra:
                        request = replace(request, questions=request.questions + extra)
                        complement_added += len(extra)
                batch_requests.append((index, request))
            packs = [pack_request(r, model.tokenizer, model.packing_mode, config.model.max_tokens, **model.packing_kwargs)
                     for _, r in batch_requests]
            # By length: a group's packs are padded to its longest state, so neighbours in token count share a pass
            # (the batch's loss is the same sum; docs/phase4/training-throughput.md).
            by_length = sorted(range(len(packs)), key=lambda i: packs[i].token_count)
            batch_requests, packs = [batch_requests[i] for i in by_length], [packs[i] for i in by_length]
            # `batch_states` requests share one backbone pass (janus.hybrid batches their prefixes); the group's losses
            # are summed so a single backward covers the shared graph. batch_states 1 reproduces the per-request path.
            group = max(1, getattr(config.model, "batch_states", 1))
            # `lo`/`hi`: not `offset` (the epoch-end check reads it) nor `start` (the clock).
            for lo, hi in token_groups([p.token_count for p in packs], group, config.group_tokens):
                members = batch_requests[lo:hi]
                # Option-order consistency: every multi-option Choice request gets an option-permuted twin, forwarded
                # as a second forward_many group after the originals (its graph stays alive beside the group's until
                # this backward, as before; the twins' own pass halves the per-layer transient of the checkpointed
                # backward, which is what bounds batch_states at 8k-token packs, docs/phase4/training-throughput.md).
                twins = {}  # position in members -> permuted request
                if config.consistency_weight:
                    twins = {i: shuffled_options(r, f"consistency:{step}:{index}") for i, (index, r) in enumerate(members)
                             if any(q.kind == "choice" and len(q.options) > 1 for q in r.questions)}
                twin_packs = [pack_request(t, model.tokenizer, model.packing_mode, config.model.max_tokens, **model.packing_kwargs)
                              for t in twins.values()]
                group_logits = model.forward_many(packs[lo:hi])
                if config._consistency_separate:
                    group_logits += [model(p) for p in twin_packs]
                else:  # the twins' own pass, under the same budget (at group_tokens 0 it is one call, as before)
                    for a, b in token_groups([p.token_count for p in twin_packs], group, config.group_tokens):
                        group_logits += model.forward_many(twin_packs[a:b])
                consistency_forwards += len(twins)
                twin_logits = dict(zip(twins, group_logits[len(members):]))
                loss = 0.
                for i, ((index, request), logits) in enumerate(zip(members, group_logits)):
                    term_loss = distribution_loss(logits, [q.target for q in request.questions], config.objective,
                                                  config.label_smoothing)
                    if i in twins:
                        term_loss = term_loss + config.consistency_weight * _consistency_term(request, twins[i], logits, twin_logits[i])
                    loss = loss + term_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {step}")
                (loss / len(batch)).backward()
                total_loss += float(loss.detach()) / len(batch)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            stop = bool((config.max_steps and step >= config.max_steps)
                        or (config.max_seconds and time.perf_counter() - start >= config.max_seconds))
            last = offset + config.accumulation >= len(order) or stop
            if step % config.eval_every == 0 or last:
                z, y = collect_logits(model, dev_data, config.group_tokens)
                validation = metrics(z, y)
                row = {"step": step, "epoch": epoch + 1, "train_loss": total_loss,
                       "requests_seen": requests_seen, "head_lr": optimizer.param_groups[1]['lr'],
                       "dev_nll": validation["nll"], "dev_accuracy": validation["accuracy"],
                       "elapsed_seconds": time.perf_counter() - start, **heldout_terms()}
                row["selection_score"] = selection_score(config, row["dev_nll"], row.get("heldout_nll"), row.get("heldout_uniform_nll"))
                history.append(row)
                print(json.dumps(row), flush=True)
                if row["selection_score"] < best_score:
                    best_score, best_nll, best_step = row["selection_score"], validation["nll"], step
                    checkpoint(model, output / "best.pt", {**metadata, "step": step, "dev": validation,
                               "requests_seen": requests_seen, "seen_group_ids": sorted(seen_groups),
                               "seen_state_hashes": sorted(seen_states)})
                write_json(output / "history.json", history)
            if stop:
                break
        if stop:
            break
    # The final weights as well, whatever the selection chose: a continued run whose dev is dominated by replay data
    # can end with a worse selection score and still be the checkpoint the new task needs.
    checkpoint(model, output / "last.pt", {**metadata, "step": step, "dev": history[-1], "requests_seen": requests_seen,
                                          "seen_group_ids": sorted(seen_groups), "seen_state_hashes": sorted(seen_states)})
    result = {"steps": step, "best_step": best_step, "initial_dev_nll": initial["nll"],
              "warmup_steps": config.mntp_steps, "warmup_final_loss": warmup_history[-1] if warmup_history else None,
              "best_dev_nll": best_nll, "best_selection_score": best_score, "selection": config.selection,
              "heldout_requests": len(heldout_data), "dev_requests": len(dev_data), "train_requests": len(train_data),
              "requests_seen": requests_seen, "unique_groups_seen": len(seen_groups),
              "unique_states_seen": len(seen_states),
              "consistency_forwards": consistency_forwards, "complement_questions_added": complement_added,
              "budget_limited": bool(config.max_seconds and time.perf_counter() - start >= config.max_seconds),
              "elapsed_seconds": time.perf_counter() - start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None}
    write_json(output / "summary.json", result)
    return result
