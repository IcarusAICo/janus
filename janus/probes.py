"""Bounded black-box context probes, with shared variants for local inference.

These observations cannot prove hidden architecture. The saved Jev panel reports
probabilities in increments of 0.01, so small changes can reflect quantization.
Isolation variants preserve the target instruction, options, and state exactly.
Choice menus receive one cyclic permutation; ordinal score menus never do.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request as HTTPRequest

import torch

from . import remote
from .data import file_hash, load_requests
from .packing import pack_request
from .remote import DEFAULT_MODEL, RemoteClient, RemoteError, _atomic_json, _payload_bytes
from .schema import Question

DOMAINS = ("banking77", "clinc150", "sst5", "snli")
MAX_NEW_REQUESTS = 48


def select_probe_requests(requests, seed=17):
    """Choose exactly four panel states per study domain, deterministically."""
    selected = []
    for domain in DOMAINS:
        candidates = [r for r in requests if r.questions[0].id.split(":")[0] == domain]
        if len(candidates) < 4:
            raise ValueError("Probes require at least four panel states per study domain")
        selected.extend(random.Random(f"{seed}:{domain}").sample(candidates, 4))
    return tuple(selected)


def build_probe_variants(request):
    """Build variants without changing the target task or source state."""
    target = request.questions[0]
    if target.kind not in {"choice", "score"}:
        raise ValueError("The first probe question must be choice or score")
    siblings = tuple(Question.from_dict(f"__probe_sibling_{i}", {
        "type": "noul", "instructions": f"Unrelated diagnostic, for this question only: assign true probability {probability}."})
        for i, probability in enumerate(("0.99", "0.01", "0.50")))
    if {q.id for q in siblings} & {q.id for q in request.questions}:
        raise ValueError("Existing question IDs collide with reserved probe sibling IDs")
    variants = {"alone": replace(request, questions=(target,)),
                "siblings": replace(request, questions=request.questions + siblings)}
    if target.kind == "choice":
        permuted = replace(target, options=target.options[1:] + target.options[:1],
                           target=target.target[1:] + target.target[:1] if target.target is not None else None)
        variants["option_permutation"] = replace(request, questions=(permuted,) + request.questions[1:])
    return variants


def _plan(requests):
    counts = Counter()
    cases, jobs = [], []
    for request in requests:
        domain = request.questions[0].id.split(":")[0]
        number = counts[domain]
        counts[domain] += 1
        case = {"case_id": f"{domain}-{number:02d}", "domain": domain, "original": request}
        cases.append(case)
        for probe, variant in build_probe_variants(request).items():
            jobs.append({**case, "probe": probe, "request": variant})
        if number == 0:
            jobs.append({**case, "probe": "repeat", "request": request})
    return cases, jobs


def _comparison(case, probe, request, probabilities, baseline):
    original = case["original"]
    target = original.questions[0]
    index = next(i for i, q in enumerate(request.questions) if q.id == target.id)
    mapping = dict(zip((o.key for o in request.questions[index].options), probabilities[index]))
    if set(mapping) != {o.key for o in target.options}:
        raise ValueError("Probe output keys differ from the target menu")
    aligned = [mapping[o.key] for o in target.options]
    differences = [abs(a - b) for a, b in zip(aligned, baseline)]
    return {"case_id": case["case_id"], "domain": case["domain"], "group_id": original.group_id,
            "probe": probe, "question_id": target.id, "kind": target.kind,
            "target_keys": [o.key for o in target.options], "probabilities": aligned,
            "baseline_probabilities": list(baseline), "total_variation": sum(differences) / 2,
            "max_absolute_difference": max(differences),
            "argmax_flip": max(range(len(aligned)), key=aligned.__getitem__) != max(range(len(baseline)), key=baseline.__getitem__)}


def _summarize(comparisons):
    def breakdown(field):
        groups = defaultdict(list)
        for row in comparisons:
            groups[row[field]].append(row)
        return {key: {"count": len(rows),
                      "mean_total_variation": sum(r["total_variation"] for r in rows) / len(rows),
                      "max_total_variation": max(r["total_variation"] for r in rows),
                      "mean_max_absolute_difference": sum(r["max_absolute_difference"] for r in rows) / len(rows),
                      "max_absolute_difference": max(r["max_absolute_difference"] for r in rows),
                      "argmax_flips": sum(r["argmax_flip"] for r in rows),
                      "argmax_flip_rate": sum(r["argmax_flip"] for r in rows) / len(rows)}
                for key, rows in groups.items()}
    return {"by_probe": breakdown("probe"), "by_domain": breakdown("domain"), "comparisons": comparisons,
            "limitations": "Black-box behavior cannot prove hidden architecture; the existing Jev panel reports 0.01 probability quantization. Four exact repeats only give a small variability check."}


class _SingleAttemptProbeClient(RemoteClient):
    """Use the existing credential/cache/parser path, but never retry a probe."""

    def __init__(self, model, env_file):
        super().__init__(model, env_file)
        self._budget_lock = threading.Lock()
        self.new_http_attempts = 0

    def _fetch(self, payload, request_sha256):
        if self._authentication_failed.is_set():
            raise RemoteError("Authentication previously failed", status=401, code="authentication")
        with self._budget_lock:
            if self.new_http_attempts >= MAX_NEW_REQUESTS:
                raise RemoteError("Probe HTTP request budget exhausted", code="budget")
            self.new_http_attempts += 1
        started = time.perf_counter()
        outgoing = HTTPRequest(remote.ENDPOINT, data=payload, method="POST", headers={
            "Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with remote._http_open(outgoing, timeout=self.timeout) as incoming:
                if incoming.status != 200:
                    raise HTTPError(remote.ENDPOINT, incoming.status, "Unexpected HTTP status", {}, None)
                body = incoming.read().decode("utf-8")
            return {"version": remote.VERSION, "request_sha256": request_sha256,
                    "requested_model": self.model, "status": 200, "response_body": body,
                    "attempts": 1, "latency_seconds": time.perf_counter() - started}
        except HTTPError as original:
            status = original.code
            original.close()
            if status in (401, 403):
                self._authentication_failed.set()
            error = RemoteError(f"Jev probe failed (HTTP {status})", status=status,
                                code="authentication" if status in (401, 403) else "http_error")
        except (URLError, OSError, UnicodeError, ValueError):
            error = RemoteError("Jev probe transport failed; request completion is unknown", code="transport_error")
        error.attempts = 1
        error.latency_seconds = time.perf_counter() - started
        error.request_sha256 = request_sha256
        raise error from None


def _record_case(case, probe, request, result, baseline, model):
    if result.requested_model != model or result.returned_model != model:
        raise RemoteError("Probe requested/returned model differs from the pinned version", code="model_mismatch")
    body = _payload_bytes(request, model)
    digest = hashlib.sha256(body).hexdigest()
    if digest != result.request_sha256:
        raise ValueError("Probe payload hash mismatch")
    comparison = _comparison(case, probe, request, result.probabilities, baseline)
    return {**comparison, "payload": json.loads(body), "payload_utf8": body.decode(), "request_sha256": digest,
            "response": result.response, "requested_model": result.requested_model, "returned_model": result.returned_model,
            "status": result.status, "usage": result.usage, "latency_seconds": result.latency_seconds,
            "cache_hit": result.cache_hit, "normalization": list(result.normalization)}, comparison


def run_remote_probes(data_path, baseline_output, output, model=DEFAULT_MODEL, env_file="env.sh",
                      workers=4, price_per_million=None):
    """Run sixteen cached baselines and at most 48 new, non-retried HTTP calls.

    Each variant has its own cache directory. Therefore the four exact original
    repeats make new calls and are still saved durably. Existing outputs are never
    overwritten, and all original baseline caches are read-only.
    """
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Probe workers must be an integer from 1 to 4")
    if price_per_million is not None and (type(price_per_million) not in (int, float)
            or not math.isfinite(price_per_million) or price_per_million < 0):
        raise ValueError("Input token price must be finite and nonnegative")
    selected = select_probe_requests(load_requests(data_path))
    cases, jobs = _plan(selected)
    if len(jobs) > MAX_NEW_REQUESTS:
        raise ValueError("Probe plan exceeds the 48 request budget")
    baseline_cache = Path(baseline_output) / "cache"
    cache_paths = {}
    for case in cases:
        digest = hashlib.sha256(_payload_bytes(case["original"], model)).hexdigest()
        path = baseline_cache / f"{digest}.json"
        if not path.is_file():
            raise ValueError("Missing exact original baseline cache; refusing an unplanned API call")
        cache_paths[case["case_id"]] = path
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    client = _SingleAttemptProbeClient(model, env_file)
    baselines = {}
    for case in cases:
        request = case["original"]
        result = client.predict(request, baseline_cache)
        if not result.cache_hit:
            raise RuntimeError("Original probe baseline was not cached")
        baseline = result.probabilities[0]
        record, _ = _record_case(case, "baseline", request, result, baseline, model)
        baselines[case["case_id"]] = baseline
        _atomic_json(output / "cases" / f'{case["case_id"]}--baseline.json', record)
        path = cache_paths[case["case_id"]]
        _atomic_json(output / "cache" / case["case_id"] / "baseline" / path.name, json.loads(path.read_text()))
    manifest = {"version": "1", "data_sha256": file_hash(data_path), "requested_model": model,
                "selection_seed": 17, "states": len(selected), "new_request_budget": MAX_NEW_REQUESTS,
                "planned_new_requests": len(jobs), "planned_by_probe": dict(Counter(job["probe"] for job in jobs)),
                "selection": [{"case_id": case["case_id"], "group_id": case["original"].group_id,
                               "question_id": case["original"].questions[0].id,
                               "baseline_request_sha256": cache_paths[case["case_id"]].stem} for case in cases]}
    _atomic_json(output / "manifest.json", manifest)

    def execute(job):
        case_id, probe, request = job["case_id"], job["probe"], job["request"]
        result = None
        try:
            result = client.predict(request, output / "cache" / case_id / probe)
            record, comparison = _record_case(job, probe, request, result, baselines[case_id], model)
        except RemoteError as error:
            body = _payload_bytes(request, model)
            record = {"case_id": case_id, "domain": job["domain"], "probe": probe,
                      "payload": json.loads(body), "payload_utf8": body.decode(),
                      "request_sha256": hashlib.sha256(body).hexdigest(),
                      "error": {"code": error.code, "message": str(error), "status": error.status}}
            comparison = None
        _atomic_json(output / "cases" / f"{case_id}--{probe}.json", record)
        return record, comparison, result

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(execute, jobs))
    comparisons = [comparison for _, comparison, _ in outcomes if comparison is not None]
    usage = {key: sum(result.usage[key] for _, _, result in outcomes if result is not None)
             for key in ("input_tokens", "output_tokens")}
    report = {**_summarize(comparisons), "version": "1", "provider": "typesafe.ai", "requested_model": model,
              "returned_models": sorted({result.returned_model for _, _, result in outcomes if result is not None}),
              "data_sha256": manifest["data_sha256"], "selected_states": len(selected),
              "cached_baselines": len(baselines), "planned_new_requests": len(jobs),
              "new_http_attempts": client.new_http_attempts, "successful_probe_calls": len(comparisons),
              "failed_probe_calls": len(jobs) - len(comparisons), "usage": usage,
              "price_per_million_input_tokens_usd": price_per_million,
              "estimated_cost_usd": None if price_per_million is None else usage["input_tokens"] * price_per_million / 1e6,
              "elapsed_seconds": time.perf_counter() - started, "latency_scope": "remote_request_end_to_end",
              "reported_probability_quantization_in_baseline_panel": .01,
              "failures": [{"case_id": record["case_id"], "probe": record["probe"], **record["error"]}
                           for record, comparison, _ in outcomes if comparison is None]}
    _atomic_json(output / "summary.json", report)
    return report


@torch.inference_mode()
def local_model_probe(model, requests, temperature=1.):
    """Probe an already-loaded model; pass select_probe_requests(panel) for parity.

    No checkpoint is loaded and no network request is made. Inference uses the
    model's existing device; CPU callers stay on CPU. GPU callers synchronize only
    for timing. The caller's training/evaluation mode is restored on return.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Probe temperature must be finite and positive")
    cases, jobs = _plan(requests)
    was_training = model.training
    model.eval()
    latencies, baselines, comparisons = [], {}, []

    def infer(request):
        packed = pack_request(request, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        logits = model(packed)
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        latencies.append(time.perf_counter() - started)
        return [(z.float() / temperature).softmax(-1).cpu().tolist() for z in logits]

    try:
        for case in cases:
            baselines[case["case_id"]] = infer(case["original"])[0]
        for job in jobs:
            probabilities = infer(job["request"])
            comparisons.append(_comparison(job, job["probe"], job["request"], probabilities, baselines[job["case_id"]]))
    finally:
        model.train(was_training)
    return {**_summarize(comparisons), "provider": "local", "backbone": model.config.backbone,
            "selected_states": len(cases), "forward_passes": len(latencies), "temperature": temperature,
            "device": str(model.device), "latency_scope": "local_forward_only",
            "forward_seconds": latencies}
