"""Local HTTP server speaking the documented /v1/systemone wire format over our own checkpoint.

The model identity returned is ours (``--model-id``); the weights are never presented as Jev.
Token usage counts our packed sequence under our tokenizer; see docs/api-compat.md.
"""

from dataclasses import replace
from datetime import datetime, timezone
import hmac
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
import uuid

import torch

from .calibration_sets import family_of_request, temperature_for
from .data import file_hash, state_hash
from .evaluation import read_calibration
from .packing import pack_request
from .ratelimit import TokenBucket
from .schema import Request, decode
from .training import load_checkpoint

log = logging.getLogger("janus.server")
MAX_BODY_BYTES = 16 << 20
# Image states over HTTP: inline base64 only (a path would read the server's files), at most MAX_IMAGES per state
# and MAX_IMAGE_BYTES of base64 each; the whole body still stays under MAX_BODY_BYTES.
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 4 << 20
LIMITS = {"choice_options": [1, 255], "score_levels": [2, 10], "questions": [1, None]}
# Sent with any error emitted before the request body is read: on a keep-alive connection (the SDKs
# reuse connections) the unread body would otherwise be parsed as the next request and get a 501.
CLOSE = [("Connection", "close")]
# An incoming x-typesafe-request-id is echoed and logged when it looks like an id; anything else gets a fresh one.
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")
# Request shapes whose CUDA graphs Worker.warm captures at start-up (scripts/precapture_shapes.py picks them from data).
PRECAPTURE = Path(__file__).with_name("precapture.json")


def precapture_bodies(shapes=None):
    """/v1/systemone bodies of `shapes` (default: those in PRECAPTURE): every text a run of `n` one-token words, sized
    so each segment (state, question block, score level) has the token count of the data request it stands for.
    ponytail: the counts are for the Qwen3.5 tokenizer every Janus checkpoint shares; another tokenizer lands in
    nearby buckets (still a warm-up, fewer exact hits)."""
    fill = lambda n: " ".join(["a"] * max(n, 1))
    bodies = []
    for shape in shapes if shapes is not None else json.loads(PRECAPTURE.read_text()):
        questions = {}
        for i, (kind, n, options) in enumerate(shape["questions"]):
            criteria = ({k: "a" for k in options} if kind == "choice" else [fill(m) for m in options] if kind == "score"
                        else {"false": "a", "true": "a"})
            questions[f"q{i}"] = {"type": kind, "instructions": fill(n), "criteria": criteria}
        bodies.append({"state": fill(shape["state"]), "questions": questions})
    return bodies


def confidence(probabilities):
    """Our rule: the largest probability. The public docs define confidence only as a 0-1 statistic of the distribution."""
    return max(probabilities)


def input_tokens(packed):
    return packed.token_count + sum(p.token_count for p in packed.branch_packs)


def normalised_entropy(p):
    """Entropy of a distribution over K outcomes as a fraction of log K (1 is uniform); 0 for K = 1."""
    return -sum(x * math.log(x) for x in p if x > 0) / math.log(len(p)) if len(p) > 1 else 0.


def envelope(request, distributions, model_id, input_tokens):
    answers = decode(request, distributions)
    for question, p in zip(request.questions, distributions):
        if question.kind == "noul":
            answers[question.id] = {"type": "noul", "noul": answers[question.id]}
        else:
            answers[question.id] = {"type": question.kind, **answers[question.id], "confidence": confidence(p)}
    return {"model": model_id, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


def null_criteria(body):
    """In place: a choice option given as null (documented: it needs no extra detail) becomes an empty description."""
    for question in body["questions"].values() if isinstance(body.get("questions"), dict) else ():
        if isinstance(question, dict) and isinstance(question.get("criteria"), dict):
            question["criteria"] = {k: "" if v is None else v for k, v in question["criteria"].items()}
    return body


class PrefixCache:
    """LRU of state-pass snapshots (janus.hybrid.PrefixSnapshot) keyed by the state hash, capacity in tokens.

    A snapshot is charged its state's tokens plus its fixed per-state memory (the DeltaNet recurrent matrices: about
    50 MB on the 4B whatever the length) expressed in tokens of attention keys and values (32 KB per token on the 4B:
    8 attention layers, 4 kv heads of 256 in bf16), so `capacity` bounds the memory whether the states are Doom frames
    or 32k documents: 100,000 tokens is 3.3 GB. Used by the inference thread only."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = {}  # key -> (snapshot, charge); insertion order is the recency order
        self.charged = self.hits = self.misses = 0

    def get(self, key):
        entry = self.entries.pop(key, None)
        if entry is None:
            self.misses += 1
            return None
        self.entries[key] = entry  # back to the most recent end
        self.hits += 1
        return entry[0]

    def put(self, key, snapshot):
        charge = math.ceil(snapshot.bytes * snapshot.tokens / snapshot.kv_bytes)
        if charge > self.capacity:
            return
        while self.entries and self.charged + charge > self.capacity:
            self.charged -= self.entries.pop(next(iter(self.entries)))[1]
        self.entries[key] = (snapshot, charge)
        self.charged += charge

    def clear(self):
        self.entries.clear()
        self.charged = self.hits = self.misses = 0

    def snapshot(self):
        entries = list(self.entries.values())  # copied: HTTP threads read while the inference thread inserts and evicts
        return {"entries": len(entries), "tokens": sum(s.tokens for s, _ in entries), "charged_tokens": self.charged,
                "capacity_tokens": self.capacity, "bytes": sum(s.bytes for s, _ in entries), "hits": self.hits, "misses": self.misses}


PRECAPTURE_RESERVE_BYTES = 4 << 30  # free GPU memory left for serving when precapture stops early


class Worker:
    """One checkpoint, one inference thread, one bounded queue, requests batched across HTTP threads.

    The loop takes one request (blocking), then keeps taking whatever else is queued, waiting at most
    `batch_window_ms` from the first take for more, up to `batch_max`; the batch goes through
    `model.forward_many` in one go. With a window of 0 the loop drains without waiting at all.

    `prefix_cache_tokens` (default 100,000 on CUDA, 0 elsewhere; 0 disables) keeps the state passes of recent states
    on the device (PrefixCache) for backbones with `encode_prefixes` (janus.hybrid): a batch runs the state pass once
    per distinct missing state and the branch levels for every request, so repeated-state workloads (many questions
    over one page or frame) pay the state once. `max_tokens` overrides the checkpoint's request budget at load.
    `fp8`: the projections run as fp8 GEMMs (janus.hybrid.FP8Linear; a measured accuracy change, not bf16 noise); `fp4`: an
    NVFP4 MoE checkpoint's experts run on the fp4 tensor cores (janus.hybrid.fp4_linear, Blackwell) instead of dequantising.
    `reread_entropy` (None: off) gates the option-order second pass of `reread`.

    ponytail: throughput ceiling is one device; the upgrade path is one Worker per device behind the same handler."""

    def __init__(self, checkpoint, device="cpu", calibration=None, queue_size=32, fast=True, batch_window_ms=4, batch_max=8,
                 prefix_cache_tokens=None, max_tokens=None, fp8=False, reread_entropy=None, fp4=False, precapture=True):
        self.model, self.metadata = load_checkpoint(checkpoint, device, max_tokens=max_tokens)
        self.calibration = read_calibration(calibration, checkpoint)
        self.temperature = self.calibration["temperature"]
        self.batch_window = batch_window_ms / 1e3
        self.batch_max = max(1, batch_max)
        # Written by the inference thread only; read racily by GET /v1/models (counters, no lock needed).
        self.stats = {"requests": 0, "batches": 0, "batch_sizes": {}, "reread": 0}
        self.reread_entropy = reread_entropy
        self.max_tokens = self.model.config.max_tokens
        if prefix_cache_tokens is None:
            prefix_cache_tokens = 100_000 if torch.device(device).type == "cuda" else 0  # 3.3 GB on the 4B (PrefixCache)
        self.cache = PrefixCache(prefix_cache_tokens) if prefix_cache_tokens > 0 and hasattr(self.model.backbone, "encode_prefixes") else None
        # The cache key's scope: this checkpoint and the packing settings (the state's tokens depend on both).
        scope = {"checkpoint": file_hash(checkpoint), "mode": self.model.packing_mode, "images": self.model.config.images,
                 "image_max_pixels": self.model.config.image_max_pixels, **{k: v for k, v in self.model.packing_kwargs.items() if k != "vision"}}
        self.scope = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()[:16]
        # fast: the serving-only paths of janus.hybrid (merged LoRA weights, CUDA-graphed level passes); a no-op for
        # other backbones and on CPU (docs/phase4/latency.md).
        if fast and hasattr(self.model.backbone, "enable_fast"):
            self.model.backbone.enable_fast(fp8=fp8, fp4=fp4)
            self.warm(precapture)
        self.queue = queue.Queue(maxsize=queue_size)
        threading.Thread(target=self._loop, daemon=True).start()

    def key(self, request):
        """The prefix cache key of a request's state: the exact text (tokenisation is case- and whitespace-sensitive,
        so the dataset-dedup hash janus.data.state_hash would alias distinct states) plus any image bytes."""
        digest = hashlib.sha256(str(request.state).encode())
        for image in getattr(request.state, "images", ()):
            digest.update(image.data)
        return f"{self.scope}:{digest.hexdigest()}"

    def warm(self, precapture=False):
        """Two requests of common shape through the whole path before serving: kernel autotuning, allocator growth
        and the first CUDA graphs happen here rather than on the first client requests. `precapture` (with CUDA
        graphs) also runs the shapes of `precapture_bodies`, so the level graphs most requests need are captured
        before serving instead of inside a request (0.3 to 2.8 s each on a cold server)."""
        criteria = {k: f"option {k} described in a few words" for k in "abcd"}
        bodies = [{"state": state, "questions": {"c": {"type": "choice", "instructions": "Pick one.", "criteria": criteria},
                                                 "n": {"type": "noul", "instructions": "Is it so?"}}}
                  for state in ("A short state.", " ".join(["The state text repeats a sentence."] * 24))]
        graphs = getattr(self.model.backbone, "graphs", None)
        if precapture and graphs is not None:
            bodies += precapture_bodies()
        start, before = time.monotonic(), torch.cuda.memory_reserved(graphs.device) if graphs is not None else 0
        for i, body in enumerate(bodies):
            # ponytail: precapture stops at a fixed free-memory reserve (PRECAPTURE_RESERVE_BYTES) or at its first
            # out-of-memory, keeping the shapes captured so far; a large model on a small card (the 35B-A3B's 22 GB of
            # NVFP4 weights on 32 GB) then serves with fewer graphs instead of failing to start. The upgrade is a
            # per-model shape budget.
            if i >= 2 and graphs is not None and torch.cuda.mem_get_info(graphs.device)[0] < PRECAPTURE_RESERVE_BYTES:
                log.warning("precapture stopped after %d of %d shapes: free GPU memory under %.1f GB", i - 2, len(bodies) - 2,
                            PRECAPTURE_RESERVE_BYTES / 1e9)
                break
            try:
                request = Request.from_dict(body)
                if self.cache is not None:
                    self.cache.clear()  # every shape runs its state pass (two shapes may share a state)
                self._run_many([(self.pack(request), None, self.key(request), None)])
            except ValueError:  # a tiny budget (tests): a warm-up request over max_tokens is skipped
                pass
            except torch.OutOfMemoryError:
                if i < 2:
                    raise
                log.warning("precapture stopped at shape %d of %d: out of memory", i - 2, len(bodies) - 2)
                torch.cuda.empty_cache()
                break
        if self.cache is not None:
            self.cache.clear()
        if graphs is not None:
            log.info("warm-up: %d requests, %d level graphs, %.1f s, %.2f GB reserved (buffer sets %.2f GB)", len(bodies),
                     len(graphs.graphs), time.monotonic() - start, (torch.cuda.memory_reserved(graphs.device) - before) / 1e9, graphs.bytes / 1e9)

    def pack(self, request):
        return pack_request(request, self.model.tokenizer, self.model.packing_mode, self.max_tokens, **self.model.packing_kwargs)

    def temperatures(self, request):
        """One temperature per question: the calibration file's most specific entry for (family, cardinality), the
        family, then the global one (janus.calibration_sets.temperature_for). The family is the request's group_id
        prefix; a request without one, or a file without `by_family`, gets the global temperature everywhere."""
        family = family_of_request(request)
        return [temperature_for(self.calibration, family, len(q.options), q.kind) for q in request.questions]

    @torch.inference_mode()
    def _run(self, packed, temperatures=None):
        return self.probabilities(self.model(packed), temperatures)

    @torch.inference_mode()
    def _run_many(self, jobs):
        """One batch of (packed, temperatures, key, slot) jobs: with the prefix cache, the state pass once per distinct
        missing key (`encode_prefixes`), then every pack's branch levels from its snapshot (`prepare_many`) and each
        pack's ordinary forward; without it, one `forward_many` over the batch. Then each request's own softmax."""
        packs = [packed for packed, _, _, _ in jobs]
        if self.cache is None or any(key is None for _, _, key, _ in jobs):  # no key: a caller outside the HTTP path
            outputs = self.model.forward_many(packs)
        else:
            snapshots = {}
            for packed, _, key, _ in jobs:
                if key not in snapshots:
                    snapshots[key] = self.cache.get(key)
            # ponytail: a missed short text state under one question is not cached: janus.hybrid runs its state and block
            # as one level (single_pass), 2 passes instead of 3, so a repeat of that state re-runs its tokens (at most
            # 2,048 packed, hybrid.KV_MAX, the graphed level-0 range) where a hit would skip them; longer states,
            # images, several questions and over 32 leaves still go through the cache.
            single = lambda p: getattr(self.model.backbone, "single_pass", False) and len(p.branches) == 1 and p.images is None and p.token_count <= 2048 and len(p.branches[0].leaves) <= 32
            missing = [key for key, snapshot in snapshots.items()
                       if snapshot is None and not all(single(p) for p, _, k, _ in jobs if k == key)]
            if missing:
                first = {key: next(p for p, _, k, _ in jobs if k == key) for key in missing}
                for key, snapshot in zip(missing, self.model.backbone.encode_prefixes([first[key] for key in missing])):
                    snapshots[key] = snapshot
                    self.cache.put(key, snapshot)
            self.model.backbone.prepare_many(packs, [snapshots[key] for _, _, key, _ in jobs])
            outputs = [self.model(packed) for packed in packs]
        return [self.probabilities(logits, temperatures) for logits, (_, temperatures, _, _) in zip(outputs, jobs)]

    def probabilities(self, logits, temperatures=None):
        """Calibrated distributions per question, one device-to-host copy for the whole request."""
        flat = torch.cat(logits).float().cpu()
        if temperatures is None:
            temperatures = [self.temperature] * len(logits)
        return [(z / t).softmax(-1).tolist() for z, t in zip(flat.split([len(z) for z in logits]), temperatures)]

    def _take(self):
        q = self.queue
        jobs = [q.get()]
        deadline = time.monotonic() + self.batch_window
        while len(jobs) < self.batch_max:
            try:
                jobs.append(q.get(timeout=max(0., deadline - time.monotonic())))
            except queue.Empty:
                break
        return jobs

    def _loop(self):
        while True:
            jobs = self._take()
            self.stats["requests"] += len(jobs)
            self.stats["batches"] += 1
            self.stats["batch_sizes"][str(len(jobs))] = self.stats["batch_sizes"].get(str(len(jobs)), 0) + 1
            self.stats["reread"] += sum(slot["reread"] for _, _, _, slot in jobs)
            try:
                results = self._run_many(jobs)
            except torch.OutOfMemoryError:
                # The failed batch's transients are free now and the allocator keeps their segments; released, a retry
                # fits where the first attempt did not (docs/phase4/latency-track2.md, "Serving memory").
                log.warning("out of memory on a batch of %d; releasing captured graphs and the allocator cache, retrying once", len(jobs))
                self.release_memory()
                results = self._retry(jobs)
            except Exception:  # one at a time, so only the failing request answers 500
                results = None
            for i, (packed, temperatures, _, slot) in enumerate(jobs):
                try:
                    slot["result"] = results[i] if results is not None else self._run(packed, temperatures)
                except Exception as error:  # surfaced to the waiting HTTP thread
                    slot["error"] = error
                slot["done"].set()

    def release_memory(self):
        """Drop every captured level graph and its buffer sets, then the allocator cache. A long or wide request that
        precapture left no room for then fits; the graphs it needs are captured again on use."""
        graphs = getattr(self.model.backbone, "graphs", None)
        if graphs is not None:
            for key in list(graphs.outs):
                graphs.evict(key)
        torch.cuda.empty_cache()

    def run_jobs(self, jobs):
        """`_run_many` with one retry after `release_memory` on out-of-memory (the Python API's path)."""
        try:
            return self._run_many(jobs)
        except torch.OutOfMemoryError:
            log.warning("out of memory on %d requests; releasing captured graphs and retrying once", len(jobs))
            self.release_memory()
            return self._run_many(jobs)

    def _retry(self, jobs):
        try:
            return self._run_many(jobs)
        except Exception:
            return None

    def submit(self, packed, temperatures=None, key=None, reread=False):
        slot = {"done": threading.Event(), "reread": reread}
        self.queue.put_nowait((packed, temperatures, key, slot))  # queue.Full -> 529 in the handler
        slot["done"].wait()
        if "error" in slot:
            raise slot["error"]
        return slot["result"]

    def reread(self, request, distributions):
        """Option-order second pass (docs/api-compat.md, "Option-order re-read"): when a multi-option Choice question's
        served distribution has normalised entropy above `reread_entropy`, the request goes through the queue once
        more with those questions' options reversed (the state is unchanged, so the batch shares its prefix-cache
        snapshot) and each flagged question answers the renormalised mean of the two views. Returns the distributions
        and the second pass's input tokens (0 when nothing was flagged)."""
        flagged = [i for i, (q, p) in enumerate(zip(request.questions, distributions))
                   if q.kind == "choice" and normalised_entropy(p) > self.reread_entropy]
        if not flagged:
            return distributions, 0
        questions = list(request.questions)
        for i in flagged:
            questions[i] = replace(questions[i], options=questions[i].options[::-1])
        packed = self.pack(replace(request, questions=tuple(questions)))
        second = self.submit(packed, self.temperatures(request), self.key(request), reread=True)
        distributions = list(distributions)
        for i in flagged:
            mean = [a + b for a, b in zip(distributions[i], second[i][::-1])]
            distributions[i] = [x / sum(mean) for x in mean]
        return distributions, input_tokens(packed)

    def memory(self):
        """Device memory in bytes: the allocator's live and reserved totals, and the LevelGraphs buffer sets (the
        4 GB LRU cap of janus.hybrid.BUFFER_BYTES); None off CUDA. Read racily by GET /v1/models like `stats`."""
        device = self.model.device
        if device.type != "cuda":
            return None
        graphs = getattr(self.model.backbone, "graphs", None)
        return {"allocated": torch.cuda.memory_allocated(device), "reserved": torch.cuda.memory_reserved(device),
                "buffer_sets": graphs.bytes if graphs is not None else 0}

    def snapshot(self):
        n, b = self.stats["requests"], self.stats["batches"]
        return {**self.stats, "mean_batch_size": n / b if b else 0., "queue_depth": self.queue.qsize(),
                "prefix_cache": self.cache.snapshot() if self.cache is not None else None, "memory": self.memory()}


class Handler(BaseHTTPRequestHandler):
    server_version = "janus-local/0.1"
    protocol_version = "HTTP/1.1"
    # Headers and body go out as two sends on an unbuffered socket; with Nagle on, the second waits for the client's
    # delayed ACK (about 40 ms per response on localhost, measured in docs/phase4/latency.md).
    disable_nagle_algorithm = True

    def log_message(self, format, *args):
        log.info("%s " + format, self.address_string(), *args)

    def log_request(self, code="-", size="-"):
        pass  # replaced by the one line per request in _send

    def _begin(self):
        self.started = time.perf_counter()
        given = self.headers.get("x-typesafe-request-id", "")
        self.request_id = given if REQUEST_ID.fullmatch(given) else uuid.uuid4().hex
        self.seen = {}  # model, questions, input_tokens once known; the log line prints "-" for the rest

    def _send(self, status, body, headers=()):
        payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        # The public SDKs read this header into `response.request_id`; the client's own id when it sent one, else ours.
        self.send_header("x-typesafe-request-id", self.request_id)
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)
        seen = self.seen
        log.info("request_id=%s %s %s status=%d model=%s questions=%s input_tokens=%s latency_ms=%.1f", self.request_id,
                 self.command, self.path, status, seen.get("model", "-"), seen.get("questions", "-"),
                 seen.get("input_tokens", "-"), (time.perf_counter() - self.started) * 1e3)

    def _error(self, status, kind, message, headers=()):
        # ponytail: the docs list status codes but no error body; this shape is ours. The SDKs read
        # `error.message` from it (verified against typesafe_sdk 0.7.0 and @typesafe-ai/sdk 0.6.0).
        self._send(status, {"error": {"type": kind, "message": message}}, headers)

    def _authorized(self, headers=()):
        token = self.server.token
        if token is None:
            return True
        if hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
            return True
        self._error(401, "authentication_error", "Missing or invalid API key", headers)
        return False

    def _rate_limited(self):
        limiter = self.server.limiter
        if limiter is None:
            return False
        # One bucket per bearer token (auth already passed, so the header is the token), else per client address.
        wait = limiter.take(self.headers.get("Authorization") if self.server.token else self.client_address[0])
        if not wait:
            return False
        self._error(429, "rate_limit_error", "Rate limit exceeded; retry later",
                    [("Retry-After", str(math.ceil(wait))), ("retry-after-ms", str(math.ceil(wait * 1e3))), *CLOSE])
        return True

    def do_GET(self):
        self._begin()
        if self.path != "/v1/models":
            return self._error(405 if self.path == "/v1/systemone" else 404, "not_found", "No such route")
        if not self._authorized():
            return
        worker, server = self.server.worker, self.server
        cards = [{"name": server.model_id, "description": "Local Janus checkpoint; not TypeSafe Jev", "release_date": server.release_date}]
        cards += [{"name": alias, "description": f"Alias of {server.model_id}", "release_date": server.release_date} for alias in server.aliases]
        limits = {**LIMITS, "max_tokens": worker.max_tokens, "scope": "local", "note": "Limits of this server and checkpoint under our tokenizer"}
        if server.limiter is not None:
            limits["requests_per_minute"] = round(server.limiter.rate * 60)
        self._send(200, {"models": cards, "limits": limits, "stats": worker.snapshot()})

    def do_POST(self):
        self._begin()
        if self.path != "/v1/systemone":
            return self._error(405 if self.path == "/v1/models" else 404, "not_found", "No such route", CLOSE)
        if not self._authorized(CLOSE) or self._rate_limited():
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self._error(400, "invalid_request", "Bad Content-Length", CLOSE)
        if length > MAX_BODY_BYTES:
            return self._error(413, "request_too_large", f"Body exceeds {MAX_BODY_BYTES} bytes", CLOSE)
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            return self._error(400, "invalid_request", "Body is not valid JSON")
        if not isinstance(body, dict):
            return self._error(422, "validation_error", "Body must be a JSON object")
        server = self.server
        model = body.get("model", server.model_id)  # documented as required; absent means the served model
        if model != server.model_id and model not in server.aliases:
            return self._error(422, "validation_error", f"Unknown model {model!r}; this server serves {server.model_id!r}"
                               + (f" (aliases: {', '.join(server.aliases)})" if server.aliases else ""))
        self.seen["model"] = model
        null_criteria(body)
        state = body.get("state")
        if isinstance(state, dict) and "images" in state:
            images = state["images"]
            if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGES:
                return self._error(422, "validation_error", f"An image state carries 1 to {MAX_IMAGES} images")
            if any(not isinstance(ref, dict) or not isinstance(ref.get("base64"), str) for ref in images):
                return self._error(422, "validation_error", "Images must be inline: {\"base64\": ..., \"media_type\": ...}")
            if any(len(ref["base64"]) > MAX_IMAGE_BYTES for ref in images):
                return self._error(413, "request_too_large", f"An image exceeds {MAX_IMAGE_BYTES} base64 bytes")
        worker = self.server.worker
        try:
            request = Request.from_dict(body)
            packed = worker.pack(request)
        except KeyError as error:
            return self._error(422, "validation_error", f"Missing field {error}")
        except ValueError as error:
            if "exceeding max_tokens" in str(error) or "exceeding max_state_plus_question" in str(error):
                return self._error(413, "request_too_large", str(error))
            return self._error(422, "validation_error", str(error))
        except (TypeError, AttributeError) as error:
            return self._error(422, "validation_error", f"Malformed request: {error}")
        self.seen["questions"] = len(request.questions)
        self.seen["input_tokens"] = input_tokens(packed)
        try:
            distributions = worker.submit(packed, worker.temperatures(request), worker.key(request))
            if worker.reread_entropy is not None:
                distributions, extra = worker.reread(request, distributions)
                self.seen["input_tokens"] += extra
        except queue.Full:
            # Retry-After is honoured by the SDKs' 5xx retry; 1 s is a guess at one forward pass.
            return self._error(server.overloaded_status, "overloaded", "Inference queue is full; retry later", [("Retry-After", "1")])
        except Exception as error:
            log.exception("inference failed")
            return self._error(500, "internal_error", type(error).__name__)
        self._send(200, envelope(request, distributions, server.model_id, self.seen["input_tokens"]))

    def _method_not_allowed(self):
        self._begin()
        self._error(405, "method_not_allowed", "Method not allowed")

    do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _method_not_allowed


def make_server(checkpoint, host="127.0.0.1", port=0, device="cpu", calibration=None, model_id="janus-local", queue_size=32,
                fast=True, batch_window_ms=4, batch_max=8, rate_limit_rpm=0, aliases=(), overloaded_status=529,
                prefix_cache_tokens=None, max_tokens=None, fp8=False, reread_entropy=None, fp4=False, precapture=True):
    """Bound but not yet serving; ``server.server_address[1]`` is the port (0 picks a free one).

    `aliases` are the other `model` values a request may carry (the SDKs default to `jev-latest`: list it here to
    accept it); the response always names `model_id`. `rate_limit_rpm` 0 is no limit. `overloaded_status` is what a
    full queue answers (529 as the docs' overload status; 503 for the old behaviour)."""
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.worker = Worker(checkpoint, device, calibration, queue_size, fast, batch_window_ms, batch_max, prefix_cache_tokens, max_tokens, fp8, reread_entropy, fp4, precapture)
    server.model_id = model_id
    server.aliases = [a for a in aliases if a and a != model_id]
    server.limiter = TokenBucket.per_minute(rate_limit_rpm) if rate_limit_rpm else None
    server.overloaded_status = overloaded_status
    server.release_date = datetime.fromtimestamp(Path(checkpoint).stat().st_mtime, timezone.utc).date().isoformat()
    server.token = os.environ.get("JANUS_SERVER_TOKEN") or None
    if server.token is None:
        log.warning("JANUS_SERVER_TOKEN is unset; authentication is disabled")
    return server


def serve(checkpoint, host="127.0.0.1", port=8080, device="cpu", calibration=None, model_id="janus-local", queue_size=32, fast=True,
          batch_window_ms=4, batch_max=8, rate_limit_rpm=0, aliases=(), overloaded_status=529, prefix_cache_tokens=None, max_tokens=None, fp8=False,
          reread_entropy=None, fp4=False, precapture=True):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = make_server(checkpoint, host, port, device, calibration, model_id, queue_size, fast,
                         batch_window_ms, batch_max, rate_limit_rpm, aliases, overloaded_status, prefix_cache_tokens, max_tokens, fp8, reread_entropy, fp4, precapture)
    cache = server.worker.cache
    log.info("serving %s (aliases: %s) on http://%s:%d (max_tokens=%d, batch window %d ms, batch max %d, rate limit %s rpm, prefix cache %s)",
             model_id, ", ".join(server.aliases) or "none", *server.server_address[:2], server.worker.max_tokens,
             batch_window_ms, batch_max, rate_limit_rpm or "off", f"{cache.capacity} tokens" if cache is not None else "off")
    server.serve_forever()
