"""Direct, label-free Jev evaluation with durable response caching.

Only the documented official HTTPS endpoint is contacted; redirects are disabled.
Every expected probability must be present, finite, and in [0, 1]. Choice/score
distributions whose total differs from one by at most 0.01 (one percentage point)
are explicitly renormalized, with their original sum and correction recorded.
No missing probabilities are inferred. Noul supplies P(true), so P(false)=1-P(true).
Remote latency includes HTTP round trips and retry waits, not local model timing.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request as HTTPRequest, build_opener

import torch

from .data import file_hash, load_requests
from .metrics import metrics, paired_nll_bootstrap
from .schema import Request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"
VERSION = "1"
ROUNDING_TOLERANCE = .01
MAX_ATTEMPTS = 3


class RemoteError(RuntimeError):
    """Sanitized errors never include transport messages, headers, or credentials."""

    def __init__(self, message, *, status=None, code="remote_error"):
        super().__init__(message)
        self.status = status
        self.code = code
        self.attempts = 0
        self.latency_seconds = 0.
        self.cache_hit = False
        self.request_sha256 = None
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.returned_model = None


def load_env_key(name, env_file="env.sh"):
    """Read the environment or a plain shlex assignment; never execute a script."""
    key = os.environ.get(name)
    if key is None:
        try:
            lines = Path(env_file).read_text().splitlines()
        except (OSError, UnicodeError, TypeError):
            raise RemoteError(f"Set {name} or provide a readable env file", code="credentials") from None
        for line in lines:
            if not re.match(rf"^\s*(?:export\s+)?{name}(?:\s|=|$)", line):
                continue
            try:
                tokens = shlex.split(line, comments=True, posix=True)
            except ValueError:
                raise RemoteError("Invalid API key assignment in env file", code="credentials") from None
            if tokens and tokens[0] == "export":
                tokens = tokens[1:]
            if len(tokens) != 1 or not tokens[0].startswith(f"{name}="):
                raise RemoteError("API key must use a plain assignment", code="credentials")
            key = tokens[0].partition("=")[2]
            if any(c in key for c in "$`;|&<>"):
                raise RemoteError("API key must use a literal assignment", code="credentials")
    if not isinstance(key, str) or not key or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in key):
        raise RemoteError(f"Missing or invalid {name}", code="credentials")
    return key


def load_api_key(env_file="env.sh"):
    return load_env_key("TYPESAFE_API_KEY", env_file)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_http_open = build_opener(_NoRedirect()).open


def _payload_bytes(request, model):
    questions = {}
    for question in request.questions:
        if question.id in questions:
            raise ValueError("Duplicate question ID")
        raw = question.to_dict()
        raw.pop("target", None)
        questions[question.id] = raw
    return json.dumps({"model": model, "state": request.state, "questions": questions},
                      ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _json_loads(value):
    return json.loads(value, object_pairs_hook=_unique_object)


def _usage(response):
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                          for k in ("input_tokens", "output_tokens")):
        raise RemoteError("Response has missing or invalid token usage", code="invalid_response")
    return {k: usage[k] for k in ("input_tokens", "output_tokens")}


def _decode_response(response, request):
    if not isinstance(response, dict) or not isinstance(response.get("model"), str) or not response["model"].strip():
        raise RemoteError("Response has missing or invalid model identity", code="invalid_response")
    usage = _usage(response)
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != {q.id for q in request.questions}:
        raise RemoteError("Response question keys do not match the request", code="invalid_response")
    probabilities, normalization = [], []
    for question in request.questions:
        answer = answers[question.id]
        if not isinstance(answer, dict) or answer.get("type") != question.kind:
            raise RemoteError("Response question type does not match the request", code="invalid_response")
        if question.kind == "noul":
            p = answer.get("noul")
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                raise RemoteError("Response contains an invalid noul probability", code="invalid_response")
            if [o.key for o in question.options] != ["false", "true"]:
                raise ValueError("Noul options must be false then true")
            values = (1 - float(p), float(p))
        else:
            mapping = answer.get("probabilities")
            if not isinstance(mapping, dict) or set(mapping) != {o.key for o in question.options}:
                raise RemoteError("Response probability keys or cardinality do not match", code="invalid_response")
            values = tuple(mapping[o.key] for o in question.options)
            if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
                raise RemoteError("Response contains invalid probabilities", code="invalid_response")
        total = sum(values)
        if total <= 0 or abs(total - 1.) > ROUNDING_TOLERANCE + 1e-12:
            raise RemoteError("Response probabilities are not normalized within rounding tolerance", code="invalid_response")
        normalization.append({"original_sum": total, "renormalized": total != 1.})
        probabilities.append(tuple(float(p) / total for p in values))
    return tuple(probabilities), tuple(normalization), usage


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _atomic_jsonl(path, rows):
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


@dataclass(frozen=True)
class RemoteResult:
    probabilities: tuple
    normalization: tuple
    response: dict
    requested_model: str
    returned_model: str
    usage: dict
    latency_seconds: float
    attempts: int
    cache_hit: bool
    request_sha256: str
    status: int = 200


class RemoteClient:
    def __init__(self, model=DEFAULT_MODEL, env_file="env.sh", timeout=60):
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
            raise ValueError("Invalid model name")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        self.model = model
        self.timeout = timeout
        self._api_key = load_api_key(env_file)
        self._lock_guard = threading.Lock()
        self._cache_locks = {}
        self._authentication_failed = threading.Event()

    def __repr__(self):
        return f"RemoteClient(model={self.model!r})"

    def _fetch(self, payload, request_sha256):
        started = time.perf_counter()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self._authentication_failed.is_set():
                error = RemoteError("Authentication previously failed", status=401, code="authentication")
                error.request_sha256 = request_sha256
                raise error
            outgoing = HTTPRequest(ENDPOINT, data=payload, method="POST",
                                   headers={"Authorization": f"Bearer {self._api_key}",
                                            "Content-Type": "application/json", "Accept": "application/json"})
            try:
                with _http_open(outgoing, timeout=self.timeout) as incoming:
                    status = incoming.status
                    if status != 200:
                        raise HTTPError(ENDPOINT, status, "Unexpected HTTP status", {}, None)
                    # Keep the exact response body even when JSON/schema validation fails.
                    body = incoming.read().decode("utf-8")
                return {"version": VERSION, "request_sha256": request_sha256,
                        "requested_model": self.model, "status": status, "response_body": body,
                        "attempts": attempt, "latency_seconds": time.perf_counter() - started}
            except HTTPError as original:
                status = original.code
                original.close()
                if status in (401, 403):
                    self._authentication_failed.set()
                if (status == 429 or 500 <= status <= 599) and attempt < MAX_ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                code = "authentication" if status in (401, 403) else "http_error"
                error = RemoteError(f"Jev request failed (HTTP {status})", status=status, code=code)
            except (URLError, OSError, UnicodeError, ValueError):
                # A network timeout may occur after billing; do not automatically replay it.
                error = RemoteError("Jev transport failed; request completion is unknown", code="transport_error")
            error.attempts = attempt
            error.latency_seconds = time.perf_counter() - started
            error.request_sha256 = request_sha256
            raise error from None

    def predict(self, request, cache_dir=None):
        payload = _payload_bytes(request, self.model)
        digest = hashlib.sha256(payload).hexdigest()
        path = Path(cache_dir) / f"{digest}.json" if cache_dir is not None else None
        with self._lock_guard:
            lock = self._cache_locks.setdefault(digest, threading.Lock())
        with lock:
            cache_hit = path is not None and path.exists()
            if cache_hit:
                try:
                    envelope = _json_loads(path.read_text())
                    if (not isinstance(envelope, dict) or envelope.get("version") != VERSION
                            or envelope.get("request_sha256") != digest
                            or envelope.get("requested_model") != self.model
                            or envelope.get("status") != 200
                            or not isinstance(envelope.get("response_body"), str)
                            or type(envelope.get("attempts")) is not int or not 1 <= envelope["attempts"] <= MAX_ATTEMPTS
                            or type(envelope.get("latency_seconds")) not in (int, float)
                            or not math.isfinite(envelope["latency_seconds"]) or envelope["latency_seconds"] < 0):
                        raise ValueError("Invalid cache envelope")
                except (OSError, UnicodeError, ValueError, TypeError):
                    error = RemoteError("Invalid response cache; refusing to repeat a possibly paid request", code="cache_error")
                    error.cache_hit, error.request_sha256 = True, digest
                    raise error from None
            else:
                envelope = self._fetch(payload, digest)
                if path is not None:
                    _atomic_json(path, envelope)
            response = None
            try:
                try:
                    response = _json_loads(envelope["response_body"])
                except (ValueError, TypeError):
                    raise RemoteError("Jev response is not valid unambiguous JSON", code="invalid_response") from None
                probabilities, normalization, usage = _decode_response(response, request)
            except RemoteError as error:
                error.cache_hit, error.request_sha256 = cache_hit, digest
                error.attempts = envelope["attempts"]
                error.latency_seconds = envelope["latency_seconds"]
                error.status = envelope["status"]
                if isinstance(response, dict):
                    try:
                        error.usage = _usage(response)
                    except RemoteError:
                        pass
                    if isinstance(response.get("model"), str):
                        error.returned_model = response["model"]
                raise error from None
            return RemoteResult(probabilities, normalization, response, self.model, response["model"], usage,
                                envelope["latency_seconds"], envelope["attempts"], cache_hit, digest)


def preflight(model=DEFAULT_MODEL, env_file="env.sh"):
    """Make one small request covering all three supported question types."""
    request = Request.from_dict({"state": "The parcel is red. Delivery was good.", "questions": {
        "choice": {"type": "choice", "instructions": "What color is the parcel?", "criteria": {"a": "Red", "b": "Blue"}},
        "score": {"type": "score", "instructions": "Rate the delivery.", "criteria": ["Bad", "Good", "Great"]},
        "noul": {"type": "noul", "instructions": "Is the parcel red?"}}})
    result = RemoteClient(model, env_file).predict(request)
    return {"http_status": result.status, "requested_model": model, "model": result.returned_model,
            "answers": result.response["answers"], "usage": result.usage,
            "latency_seconds": result.latency_seconds}


def _signature(request, question):
    signature = {"state": request.state, "kind": question.kind, "instructions": question.instructions,
                 "options": [(o.key, o.description) for o in question.options]}
    return hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _report(requests, results, failures, hashes, output, manifest, elapsed, price_per_million):
    predictions, labels, by_kind, by_cardinality, by_domain = [], [], {}, {}, {}
    succeeded, failed, skipped = 0, 0, 0
    for request, digest in zip(requests, hashes):
        for q in request.questions:
            labels.append({"group_id": request.group_id, "question_id": q.id,
                           "input_sha256": _signature(request, q), "keys": [o.key for o in q.options],
                           "target": torch.tensor(q.target, dtype=torch.float32).tolist()})
        if digest not in results:
            failed += int(digest in failures)
            skipped += int(digest not in failures)
            continue
        succeeded += 1
        result = results[digest]
        for q, probabilities, normalization in zip(request.questions, result.probabilities, result.normalization):
            target = torch.tensor(q.target, dtype=torch.float32).tolist()
            logits = [math.log(max(p, 1e-12)) for p in probabilities]
            score = metrics([logits], [target])
            index = len(predictions)
            predictions.append({"group_id": request.group_id, "question_id": q.id, "kind": q.kind,
                                "input_sha256": _signature(request, q), "cardinality": len(q.options),
                                "keys": [o.key for o in q.options], "target": target, "logits": logits,
                                "probabilities": list(probabilities), "nll": score["nll"],
                                "calibrated_nll": score["nll"], "uniform_nll": math.log(len(q.options)),
                                "normalization": normalization, "request_sha256": digest,
                                "requested_model": result.requested_model, "returned_model": result.returned_model})
            by_kind.setdefault(q.kind, []).append(index)
            by_cardinality.setdefault(str(len(q.options)), []).append(index)
            by_domain.setdefault(re.split(r"[:/._]", q.id, maxsplit=1)[0], []).append(index)

    def scores(indices):
        return metrics([predictions[i]["logits"] for i in indices], [predictions[i]["target"] for i in indices])

    def breakdown(groups):
        return {key: {"raw": scores(indices)} for key, indices in groups.items()}

    all_outcomes = {**results, **failures}
    usage = {key: sum(value.usage[key] for value in all_outcomes.values()) for key in ("input_tokens", "output_tokens")}
    new_usage = {key: sum(value.usage[key] for value in all_outcomes.values() if not value.cache_hit) for key in usage}
    latencies = [value.latency_seconds for value in all_outcomes.values() if value.attempts]
    failure_rows = [{"request_sha256": digest, "code": error.code, "http_status": error.status,
                     "message": str(error), "cache_hit": error.cache_hit, "attempts": error.attempts,
                     "latency_seconds": error.latency_seconds, "usage": error.usage}
                    for digest, error in failures.items()]
    report = {"version": VERSION, "provider": "typesafe.ai", "endpoint": ENDPOINT,
              "requested_model": manifest["requested_model"],
              "returned_models": sorted({v.returned_model for v in all_outcomes.values() if v.returned_model}),
              "data_sha256": manifest["data_sha256"], "requests": len(requests), "total_requests": len(requests),
              "unique_requests": len(set(hashes)), "successful_requests": succeeded, "failed_requests": failed,
              "not_attempted_requests": skipped, "cache_hits": sum(v.cache_hit for v in all_outcomes.values()),
              "http_requests_this_run": sum(v.attempts for v in all_outcomes.values() if not v.cache_hit),
              "usage": usage, "new_usage": new_usage,
              "price_per_million_input_tokens_usd": price_per_million,
              "estimated_cost_usd": None if price_per_million is None else usage["input_tokens"] * price_per_million / 1e6,
              "estimated_new_cost_usd": None if price_per_million is None else new_usage["input_tokens"] * price_per_million / 1e6,
              "cost_basis": "Optional user-supplied input token price; output tokens are free; excludes unknown error-response usage",
              "elapsed_seconds_this_run": elapsed,
              "latency": {"scope": "remote_request_end_to_end_including_retries", "includes_cached_original_latencies": True,
                          "count": len(latencies), "p50_seconds": _percentile(latencies, .5),
                          "p95_seconds": _percentile(latencies, .95)},
              "calibration": "Not applied; raw Jev probabilities", "temperature": 1.,
              "rounding_tolerance_absolute": ROUNDING_TOLERANCE,
              "renormalized_questions": sum(r["normalization"]["renormalized"] for r in predictions),
              "log_probability_floor": 1e-12,
              "raw": scores(range(len(predictions))) if predictions else None,
              "by_kind": breakdown(by_kind), "by_cardinality": breakdown(by_cardinality), "by_domain": breakdown(by_domain),
              "uniform_comparison": paired_nll_bootstrap([{**r, "control_nll": r["uniform_nll"]} for r in predictions]) if predictions else None}
    _atomic_jsonl(output / "labels.jsonl", labels)
    _atomic_jsonl(output / "predictions.jsonl", predictions)
    _atomic_jsonl(output / "failures.jsonl", failure_rows)
    _atomic_json(output / "metrics.json", report)
    return report


def evaluate_remote(data_path, output, model=DEFAULT_MODEL, workers=4, limit=None,
                    env_file="env.sh", resume=False, price_per_million=None):
    """Evaluate raw Jev distributions on local labels with a resumable request cache.

    ``limit`` uses the same seed-17 sample as local evaluation. Cost is omitted
    unless an explicit USD/million-input-token price is supplied; output tokens
    are free. An existing output
    directory requires resume=True and an identical dataset/model/selection.
    Authentication is checked using the first real observation before workers are
    scheduled. HTTP 429/5xx receive at most three total attempts (1s, 2s waits).
    Other errors are recorded without retries; auth errors also stop evaluation.
    """
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer from 1 to 32")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if price_per_million is not None and (type(price_per_million) not in (int, float)
                                         or not math.isfinite(price_per_million) or price_per_million < 0):
        raise ValueError("price_per_million must be finite and nonnegative")
    requests = load_requests(data_path)
    if limit is not None:
        requests = random.Random(17).sample(requests, min(limit, len(requests)))
    pairs = [(r.group_id, q.id, _signature(r, q)) for r in requests for q in r.questions]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate state/question pairs in evaluation data")
    if any(q.target is None for r in requests for q in r.questions):
        raise ValueError("Remote evaluation requires local targets for every question")
    hashes = [hashlib.sha256(_payload_bytes(r, model)).hexdigest() for r in requests]
    manifest = {"version": VERSION, "data_sha256": file_hash(data_path), "requested_model": model,
                "limit": limit, "selection_seed": 17, "request_sha256": hashes}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    with (output / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another evaluation is using this output directory") from None
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            try:
                prior = _json_loads(manifest_path.read_text())
            except (ValueError, OSError):
                raise ValueError("Cannot resume an invalid evaluation manifest") from None
            if prior != manifest:
                raise ValueError("Cannot resume with a changed dataset, model, or selection")
        elif resume and any(p.name != ".evaluation.lock" for p in output.iterdir()):
            raise ValueError("Cannot resume an output directory without its manifest")
        _atomic_json(manifest_path, manifest)
        client = RemoteClient(model, env_file)
        unique = dict(zip(hashes, requests))
        results, failures = {}, {}
        started = time.perf_counter()
        fatal = None

        def accept(digest, future=None):
            nonlocal fatal
            try:
                results[digest] = future.result() if future is not None else client.predict(unique[digest], output / "cache")
            except RemoteError as error:
                failures[digest] = error
                if error.status in (401, 403):
                    fatal = error

        pending_items = iter(unique)
        accept(next(pending_items))
        if fatal is None:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}

                def schedule():
                    while len(futures) < workers and fatal is None:
                        digest = next(pending_items, None)
                        if digest is None:
                            break
                        futures[pool.submit(client.predict, unique[digest], output / "cache")] = digest

                schedule()
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        digest = futures.pop(future)
                        accept(digest, future)
                    schedule()
        report = _report(requests, results, failures, hashes, output, manifest,
                         time.perf_counter() - started, price_per_million)
        if fatal is not None:
            raise fatal from None
        return report
