"""Blocked, null-controlled Jev API probes (spec WP6: B0, B2, D1, D3; wave 2 adds B1, B3, B4, D4, D5, D7, D1b, B2b).

Behavioural measurements with contemporaneous nulls. They narrow a hypothesis
family about the API's behaviour; they do not identify an implementation.
"""

from collections import Counter, defaultdict
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

import numpy as np

from . import remote
from .data import file_hash, load_requests
from .packing import ByteTokenizer
from .remote import DEFAULT_MODEL, RemoteClient, RemoteError, _atomic_json, _payload_bytes
from .schema import Option, Question, Request
from .synth.worlds import TRAIN_NAMES, ordinal_world, posterior_family, posterior_world

DOMAINS = ("banking77", "clinc150", "sst5", "snli")
MAX_CALLS = 1200
INTERIOR = .02
B2_VARIANTS = ("baseline", "repeat", "permutation", "reversal", "added", "replaced")
IRRELEVANT = (
    "The weather was bad that day.", "The office plants need watering.", "A new coffee machine was installed.",
    "The quarterly newsletter is late.", "Traffic on the bridge was heavy.", "The lobby was repainted blue.",
    "A colleague brought cake to work.", "The parking garage is closed on Sundays.", "The stapler is out of staples.",
    "It rained during the afternoon.", "The elevator music changed.", "Someone lost an umbrella in the hall.",
    "The cafeteria now serves soup.", "The window blinds are stuck.", "A pigeon landed on the roof.",
    "The printer on floor two hums.", "The company picnic was postponed.", "The lawn was mowed this morning.",
    "A train passed by at noon.", "The vending machine takes coins only.")
SST5_NOUL = "Does this movie review express positive overall sentiment? Neutral sentiment does not count as positive."
SNLI_NOUL_PREFIX = "Assume the premise is true. Is the hypothesis logically entailed by the premise?"


def domain_of(request):
    return request.questions[0].id.split(":")[0]


def negate_noul(question):
    if question.kind != "noul":
        raise ValueError("Only Noul questions can be negated")
    text = question.instructions
    options = question.options
    for prefix, replacement in (("Does the customer's message express this intent: ", "Is it false that the customer's message expresses this intent: "),
                                ("Does the user's request express this intent: ", "Is it false that the user's request expresses this intent: ")):
        if text.startswith(prefix):
            negated = replacement + text[len(prefix):]
            break
    else:
        if text == SST5_NOUL:
            negated = "Is it false that this movie review expresses positive overall sentiment? Neutral sentiment does not count as positive."
        elif text.startswith(SNLI_NOUL_PREFIX):
            negated = ("Assume the premise is true. Is it false that the hypothesis is logically entailed by the premise? "
                       "Contradiction and insufficient information both mean it is not entailed.")
            def _clause(description):
                return description.split(".", 1)[1].strip() if "." in description else description
            # Negated question: "true" now means the hypothesis is NOT entailed. Keep Yes/No attached to the key.
            options = (Option("false", f"No. {_clause(options[1].description)}".strip()),
                       Option("true", f"Yes. {_clause(options[0].description)}".strip()))
        else:
            return None
    target = tuple(reversed(question.target)) if question.target is not None else None
    return Question(question.id + ":neg", "noul", negated, options, target)


def _gold_index(question):
    if question.target is not None:
        return max(range(len(question.target)), key=question.target.__getitem__)
    return 0


def _with_options(request, question, options):
    target = None
    if question.target is not None:
        by_desc = {o.description: t for o, t in zip(question.options, question.target)}
        target = tuple(by_desc.get(o.description, 0.) for o in options)
        total = sum(target)
        target = tuple(t / total for t in target) if total > 0 else None
    return replace(request, questions=(replace(question, options=options, target=target),))


def b2_variant(request, variant, case_index, protect=()):
    """`protect` lists option descriptions that `replaced` must keep (wave 2 retains them for the residual)."""
    q = request.questions[0]
    options = list(q.options)
    if variant in ("baseline", "repeat"):
        pass
    elif variant == "permutation":
        options = options[1:] + options[:1]
    elif variant == "reversal":
        options = options[::-1]
    elif variant == "added":
        options.append(Option("o_extra", IRRELEVANT[case_index % 20]) if q.kind == "choice"
                       else Option(str(len(options)), "Overwhelmingly positive, beyond any reasonable doubt"))
    elif variant == "replaced":
        if q.kind == "choice":
            gold = _gold_index(q)
            index = max(i for i in range(len(options)) if i != gold and options[i].description not in protect)
            options[index] = Option(options[index].key, IRRELEVANT[(case_index + 7) % 20])
        else:
            middle = len(options) // 2
            options[middle] = Option(options[middle].key, "Neither clearly positive nor clearly negative")
    else:
        raise ValueError(variant)
    if q.kind == "score":
        options = [Option(str(i), o.description) for i, o in enumerate(options)]
    return _with_options(request, q, tuple(options))


def d1_base(request):
    q = request.questions[0]
    gold = _gold_index(q)
    others = [i for i in range(len(q.options)) if i != gold]
    keep = [gold] + others[:3]
    extra = others[3]
    retained = [q.options[i].description for i in keep]
    options = tuple(q.options[i] for i in keep) + (q.options[extra],)
    return _with_options(request, q, options), retained


def d1_variant(base_request, variant):
    q = base_request.questions[0]
    options = list(q.options)
    if variant == "base":
        pass
    elif variant == "removed":
        options = options[:-1]
    elif variant.startswith("e"):
        options[-1] = Option(options[-1].key, IRRELEVANT[int(variant[1:])])
    else:
        raise ValueError(variant)
    return _with_options(base_request, q, tuple(options))


def _pick(requests, predicate, count, seed, label):
    pool = [r for r in requests if predicate(r)]
    if len(pool) < count:
        raise ValueError(f"Need {count} panel states for {label}, found {len(pool)}")
    return random.Random(f"{seed}:{label}").sample(pool, count)


def plan_jobs(requests, seed=17):
    jobs = []

    def add(family, case_id, variant, block, request, retained=None):
        jobs.append({"job_id": f"{family}--{case_id}--{variant}--b{block}", "family": family, "case_id": case_id,
                     "variant": variant, "block": block, "request": request, "retained": retained})

    for domain in DOMAINS:
        for i, r in enumerate(_pick(requests, lambda r, d=domain: domain_of(r) == d, 3, seed, f"B0:{domain}")):
            for block in range(3):
                for rep in range(4):
                    add("B0", f"{domain}-{i}", f"rep{rep}", block, r)
    choice_cases = []
    for domain in ("banking77", "clinc150"):
        choice_cases += _pick(requests, lambda r, d=domain: domain_of(r) == d and r.questions[0].kind == "choice"
                              and len(r.questions[0].options) == 8, 4, seed, f"B2:{domain}")
    score_cases = _pick(requests, lambda r: domain_of(r) == "sst5" and r.questions[0].kind == "score"
                        and len(r.questions[0].options) == 5, 8, seed, "B2:sst5")
    for index, r in enumerate(choice_cases + score_cases):
        kind = r.questions[0].kind
        for block in range(4):
            for variant in B2_VARIANTS:
                add("B2", f"{kind}-{index}", variant, block, b2_variant(r, variant, index))
    for index, r in enumerate(choice_cases):
        base, retained = d1_base(r)
        for variant in ["base", "removed"] + [f"e{n:02d}" for n in range(20)]:
            add("D1", f"choice-{index}", variant, 0, d1_variant(base, variant), retained)
    for domain in DOMAINS:
        pool = [r for r in requests if domain_of(r) == domain and negate_noul(r.questions[1]) is not None]
        for i, r in enumerate(_pick(pool, lambda r: True, 50, seed, f"D3:{domain}")):
            noul = r.questions[1]
            add("D3", f"{domain}-{i}", "affirm", 0, replace(r, questions=(noul,)))
            add("D3", f"{domain}-{i}", "negate", 0, replace(r, questions=(negate_noul(noul),)))
    rng = random.Random(seed)
    by_block = defaultdict(list)
    for job in jobs:
        by_block[job["block"]].append(job)
    ordered = []
    for block in sorted(by_block):
        group = by_block[block]
        rng.shuffle(group)
        ordered.extend(group)
    if len(ordered) > MAX_CALLS:
        raise ValueError("Plan exceeds the call budget")
    return ordered


class _BudgetClient(RemoteClient):
    """Existing credential, cache, and parser path; one attempt per call; hard budget."""

    def __init__(self, model, env_file, budget=MAX_CALLS, timeout=60):
        super().__init__(model, env_file, timeout)
        self.budget = budget
        self.calls = 0
        self._budget_lock = threading.Lock()

    def _fetch(self, payload, request_sha256):
        with self._budget_lock:
            if self.calls >= self.budget:
                raise RemoteError("Probe call budget exhausted", code="budget")
            self.calls += 1
        started = time.perf_counter()
        outgoing = HTTPRequest(remote.ENDPOINT, data=payload, method="POST", headers={
            "Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with remote._http_open(outgoing, timeout=self.timeout) as incoming:
                if incoming.status != 200:
                    raise HTTPError(remote.ENDPOINT, incoming.status, "Unexpected HTTP status", {}, None)
                body = incoming.read().decode("utf-8")
            return {"version": remote.VERSION, "request_sha256": request_sha256, "requested_model": self.model,
                    "status": 200, "response_body": body, "attempts": 1, "latency_seconds": time.perf_counter() - started}
        except HTTPError as original:
            status = original.code
            original.close()
            error = RemoteError(f"Jev probe failed (HTTP {status})", status=status,
                                code="authentication" if status in (401, 403) else "http_error")
        except (URLError, OSError, UnicodeError, ValueError):
            error = RemoteError("Jev probe transport failed; request completion is unknown", code="transport_error")
        error.attempts, error.latency_seconds, error.request_sha256 = 1, time.perf_counter() - started, request_sha256
        raise error from None


WAVE2_JOB_KEYS = ("state_tokens", "questions", "condition", "rep", "option_count", "world_id", "style")


def _job_record(job, model):
    body = _payload_bytes(job["request"], model)
    first = job["request"].questions[0]
    record = {"job_id": job["job_id"], "family": job["family"], "case_id": job["case_id"], "variant": job["variant"],
              "block": job["block"], "retained": job["retained"], "payload": json.loads(body),
              "request_sha256": hashlib.sha256(body).hexdigest(), "keys": [o.key for o in first.options],
              "descriptions": [o.description for o in first.options], "kind": first.kind}
    if first.target is not None:
        record["targets"] = list(first.target)
    record.update({k: job[k] for k in WAVE2_JOB_KEYS if k in job})
    return record


def run_blocked_probes(data_path, output, model=DEFAULT_MODEL, env_file="env.sh", seed=17, resume=False, wave=1,
                       tokenizer=None, floor_sources=(), interior_cases=None,
                       predictions_path="runs/study-20260917/janus/predictions.jsonl"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    requests = load_requests(data_path)
    manifest = {"version": "1", "data_sha256": file_hash(data_path), "requested_model": model, "seed": seed}
    if wave == 1:
        jobs = plan_jobs(requests, seed)
    elif wave == 2:
        manifest["wave"] = 2
        if interior_cases is None:
            manifest["predictions_sha256"] = file_hash(predictions_path)
            interior_cases = select_interior_cases(requests, predictions_path, seed=seed)
        jobs = plan_jobs_wave2(requests, seed, tokenizer, interior_cases)
    elif wave == 3:
        manifest["wave"] = 3
        jobs = plan_jobs_wave3(requests, seed)
    else:
        raise ValueError(f"Unknown wave {wave!r}")
    manifest.update({"planned_jobs": len(jobs), "budget": MAX_CALLS, "families": dict(Counter(j["family"] for j in jobs))})
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Cannot resume with a changed panel, model, or seed")
    _atomic_json(manifest_path, manifest)
    client = _BudgetClient(model, env_file, timeout=180 if wave == 2 else 60)
    records, failed = [], 0
    for job in jobs:
        path = output / "jobs" / f'{job["job_id"]}.json'
        if path.exists():
            records.append(json.loads(path.read_text()))
            failed += "error" in records[-1]
            continue
        record = _job_record(job, model)
        record["started_at"] = time.time()
        try:
            result = client.predict(job["request"], output / "cache" / job["job_id"])
            if result.returned_model != model:
                raise RemoteError(f"Returned model {result.returned_model!r} differs from the pinned version", code="model_mismatch")
            record.update({"response": result.response, "returned_model": result.returned_model, "usage": result.usage,
                           "latency_seconds": result.latency_seconds, "cache_hit": result.cache_hit,
                           "probabilities": list(result.probabilities[0]), "normalization": list(result.normalization)})
        except RemoteError as error:
            if error.code == "model_mismatch":
                raise
            record["error"] = {"code": error.code, "message": str(error), "status": error.status}
            failed += 1
        _atomic_json(path, record)
        records.append(record)
    summary = summarize(records) if wave == 1 else summarize_wave3(records) if wave == 3 else summarize_wave2(records, floor_sources)
    summary.update({"http_calls": client.calls, "failed_jobs": failed, "jobs": len(records), **manifest})
    _atomic_json(output / "summary.json", summary)
    (output / "summary.md").write_text(render_markdown(summary))
    return summary


def aligned(record):
    return dict(zip(record["descriptions"], record["probabilities"]))


def total_variation(base, variant, keys):
    return sum(abs(base[k] - variant[k]) for k in keys) / 2


def rescaling_residual(base, variant, retained):
    keep = [k for k in retained if base.get(k, 0) >= INTERIOR and variant.get(k, 0) >= INTERIOR]
    if len(keep) < 3:
        return None
    lb = np.log([base[k] for k in keep])
    lv = np.log([variant[k] for k in keep])
    lb, lv = lb - lb.mean(), lv - lv.mean()
    scale = float(lv @ lb / (lb @ lb)) if lb @ lb > 0 else float("nan")
    residual = lv - scale * lb
    return {"retained_interior": len(keep), "scale": scale,
            "residual_rms": float(np.sqrt((residual ** 2).mean())),
            "rank_reversal": bool(np.any(np.argsort(-lb) != np.argsort(-lv)))}


def _expected_level(record):
    return sum(i * p for i, p in enumerate(record["probabilities"]))


def _stats(values, prefix):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {f"{prefix}_count": 0}
    return {f"{prefix}_count": len(values), f"{prefix}_mean": float(np.mean(values)), f"{prefix}_max": float(np.max(values))}


def summarize(records):
    ok = [r for r in records if "error" not in r]
    by = defaultdict(lambda: defaultdict(dict))
    for r in ok:
        by[r["family"]][(r["case_id"], r["block"])][r["variant"]] = r
    within, across, identical = [], [], []
    means = defaultdict(list)
    for (case, block), variants in by["B0"].items():
        base = variants.get("rep0")
        if base is None:
            continue
        for name, r in variants.items():
            if name != "rep0":
                tv = total_variation(aligned(base), aligned(r), base["descriptions"])
                within.append(tv)
                identical.append(tv < 5e-3)
        means[case].append(np.mean([r["probabilities"] for r in variants.values()], axis=0))
    for case, rows in means.items():
        for a in rows:
            for b in rows:
                across.append(float(np.abs(a - b).sum() / 2))
    b0 = {"within_block_max_tv": max(within, default=None), "within_block_mean_tv": float(np.mean(within)) if within else None,
          "across_block_max_tv": max(across, default=None), "identical_to_two_decimals_fraction": float(np.mean(identical)) if identical else None}
    b2 = {"choice": defaultdict(list), "score": defaultdict(list)}
    for (case, block), variants in by["B2"].items():
        base = variants.get("baseline")
        if base is None:
            continue
        kind = base["kind"]
        b_al = aligned(base)
        for name, r in variants.items():
            if name == "baseline":
                continue
            v_al = aligned(r)
            common = [d for d in base["descriptions"] if d in v_al]
            row = {"tv": total_variation(b_al, v_al, common),
                   "flip": max(b_al, key=b_al.get) != max(v_al, key=v_al.get)}
            if kind == "choice" and name in ("added", "replaced", "repeat"):
                retained = [d for d in base["descriptions"] if d in v_al]
                res = rescaling_residual(b_al, v_al, retained)
                row["residual_rms"] = res["residual_rms"] if res else None
                row["rank_reversal"] = res["rank_reversal"] if res else None
            if kind == "score" and name == "reversal":
                k = len(base["probabilities"])
                row["expected_level_mismatch"] = abs(_expected_level(r) - (k - 1 - _expected_level(base)))
            b2[kind][name].append(row)
    b2_out = {}
    for kind, variants in b2.items():
        b2_out[kind] = {}
        for name, rows in variants.items():
            out = {**_stats([r["tv"] for r in rows], "tv"), "flips": int(sum(r["flip"] for r in rows)), "count": len(rows)}
            if any("residual_rms" in r for r in rows):
                out.update(_stats([r.get("residual_rms") for r in rows], "residual_rms"))
                out["rank_reversals"] = int(sum(bool(r.get("rank_reversal")) for r in rows))
            if any("expected_level_mismatch" in r for r in rows):
                out.update(_stats([r["expected_level_mismatch"] for r in rows], "expected_level_mismatch"))
            b2_out[kind][name] = out
    d1_res, d1_rev, d1_removed = [], 0, []
    for (case, block), variants in by["D1"].items():
        base = variants.get("base")
        if base is None:
            continue
        b_al = aligned(base)
        for name, r in variants.items():
            if name == "base":
                continue
            res = rescaling_residual(b_al, aligned(r), base["retained"])
            if res is None:
                continue
            (d1_removed if name == "removed" else d1_res).append(res["residual_rms"])
            d1_rev += res["rank_reversal"] and name != "removed"
    d1 = {**_stats(d1_res, "residual_rms"), "rank_reversals": int(d1_rev), **_stats(d1_removed, "removed_residual_rms")}
    gaps = []
    for (case, block), variants in by["D3"].items():
        if "affirm" in variants and "negate" in variants:
            gaps.append(variants["affirm"]["probabilities"][1] + variants["negate"]["probabilities"][1] - 1)
    d3 = {"count": len(gaps)}
    if gaps:
        q = np.quantile(gaps, [.05, .5, .95])
        d3.update({"gap_mean": float(np.mean(gaps)), "gap_sd": float(np.std(gaps)), "gap_q05": float(q[0]),
                   "gap_q50": float(q[1]), "gap_q95": float(q[2]), "fraction_abs_gap_over_0.1": float(np.mean(np.abs(gaps) > .1))})
    latencies = [r["latency_seconds"] for r in ok if "latency_seconds" in r]
    return {"B0": b0, "B2": b2_out, "D1": d1, "D3": d3,
            "latency_p50_seconds": float(np.median(latencies)) if latencies else None,
            "usage": {k: sum(r["usage"][k] for r in ok if "usage" in r) for k in ("input_tokens", "output_tokens")},
            "reading": "Behavioural statistics with contemporaneous nulls (B0 repeats, B2 repeat rows, D1 removed rows). "
                       "They constrain hypotheses about API behaviour; they do not identify an implementation."}


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def render_markdown(summary):
    lines = ["# Blocked Jev probes", "", f"Model {summary.get('requested_model')}, jobs {summary.get('jobs')}, HTTP calls {summary.get('http_calls')}, failed {summary.get('failed_jobs')}.", "",
             "| section | statistic | value |", "| --- | --- | ---: |"]
    if "B0" in summary:
        for key, value in summary["B0"].items():
            lines.append(f"| B0 | {key} | {value} |")
        for kind, variants in summary["B2"].items():
            for name, out in variants.items():
                for key in ("tv_mean", "tv_max", "flips", "residual_rms_mean", "rank_reversals", "expected_level_mismatch_mean"):
                    if key in out:
                        lines.append(f"| B2 {kind} {name} | {key} | {out[key]} |")
        for key, value in summary["D1"].items():
            lines.append(f"| D1 | {key} | {value} |")
        for key, value in summary["D3"].items():
            lines.append(f"| D3 | {key} | {value} |")
    if "B1" in summary:
        for name, out in summary["B1"].items():
            for key, value in out.items():
                lines.append(f"| B1 {name} | {key} | {_fmt(value)} |")
        for name, out in summary["B3"].items():
            for key, value in out.items():
                lines.append(f"| B3 {name} | {key} | {_fmt(value)} |")
        for cell, out in summary["B4"]["cells"].items():
            for key, value in out.items():
                lines.append(f"| B4 {cell} | median_ms_{key} | {_fmt(value)} |")
        for name, out in summary["B4"]["fits"].items():
            for key, value in out.items():
                lines.append(f"| B4 fit {name} | {key} | {_fmt(value)} |")
        for k in sorted(summary["D4"]["by_k"], key=int):
            lines.append(f"| D4 K={k} | median_ms | {_fmt(summary['D4']['by_k'][k])} |")
        lines.append(f"| D4 | ms_per_option | {_fmt(summary['D4']['ms_per_option'])} |")
        for key, value in summary["D5"].items():
            lines.append(f"| D5 | {key} | {_fmt(value)} |")
        for key, value in summary["D7"].items():
            if not isinstance(value, dict):
                lines.append(f"| D7 | {key} | {_fmt(value)} |")
        for family, out in summary["D7"].get("by_family", {}).items():
            for key, value in out.items():
                lines.append(f"| D7 {family} | {key} | {_fmt(value)} |")
        for key, value in summary["D1b"].items():
            lines.append(f"| D1b | {key} | {_fmt(value)} |")
        for name, out in summary["B2b"].items():
            for key, value in out.items():
                lines.append(f"| B2b {name} | {key} | {_fmt(value)} |")
        lines += ["", summary["latency_note"]]
    lines += ["", summary["reading"], ""]
    return "\n".join(lines)


# --- Wave 2 (Plan 2A): B1, B3, B4, D4, D5, D7, D1b, B2b -------------------------------------------------

SECRET = "The access code is ZEBRA-7431."
SECRET_QUESTION = "Does anything you can see mention the access code ZEBRA-7431?"
B1_VARIANTS = ("alone", "repeat", "neutral_sibling", "secret_sibling", "secret_in_state", "secret_in_target")
B3_LOCATIONS = ("state", "instructions", "earlier_option", "later_option")
B4_STATE_TOKENS = (500, 2000, 8000, 20000)
B4_QUESTION_COUNTS = (1, 10, 100, 500)
B4_QUESTION_TOKENS = 48  # nominal tokens per four-option Choice question, used only for the latency fit
D4_COUNTS = (2, 4, 8, 16, 32, 64, 128, 255)
D4_WORDS = ("anchor", "basket", "candle", "dagger", "engine", "feather", "garden", "hammer", "island", "jacket",
            "kettle", "ladder", "marble", "needle", "orange", "pillow", "quiver", "ribbon", "saddle", "tunnel",
            "urchin", "violin", "wagon", "xylophone", "yogurt", "zipper", "arrow", "bottle", "carpet", "dolphin",
            "eagle", "falcon", "goblet", "helmet", "igloo", "jungle", "kitten", "lantern", "meadow", "napkin",
            "otter", "parrot", "quartz", "rocket", "sponge", "trumpet", "umbrella", "vessel", "walnut", "yacht",
            "zebra", "acorn", "beacon", "cactus", "desert", "ember", "forest", "glacier", "harbor", "iceberg",
            "jasmine", "kayak", "lagoon", "mosaic")
PARAGRAPH = ("The customer wrote to say that the transfer had failed twice and that the money had left the account. "
             "They asked whether the fee would be refunded and how long the review would take. ")


def synthetic_state(tokenizer, n_tokens, salt=""):
    """A state of exactly `n_tokens` tokens under `tokenizer`, cut on a token boundary; `salt` changes the first sentence."""
    text = (salt + " " if salt else "") + PARAGRAPH * (n_tokens // 30 + 2)
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    decode = getattr(tokenizer, "decode", None)
    return decode(ids) if decode else text[:n_tokens * 4]


def _noul(qid, instructions, target=None):
    raw = {"type": "noul", "instructions": instructions}
    if target is not None:
        raw["target"] = target
    return Question.from_dict(qid, raw)


def select_interior_cases(requests, predictions_path, count=8, threshold=INTERIOR, seed=17):
    """Interior Choice menus chosen from Jev's saved panel predictions, never from local model outputs.

    A case qualifies when its delivered baseline has at least three options at or above `threshold`. The `count`
    cases with the most such options are used (ties broken by a seeded shuffle); the top four options by baseline
    probability are retained and the fifth-ranked option is the extra option E.
    """
    by_key = {(r.group_id, q.id): (r, q) for r in requests for q in r.questions if q.kind == "choice" and len(q.options) >= 5}
    candidates = []
    for line in Path(predictions_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["group_id"], row["question_id"])
        if key not in by_key:
            continue
        request, question = by_key[key]
        probabilities = dict(zip(row["keys"], row["probabilities"]))
        interior = [o for o in question.options if probabilities.get(o.key, 0) >= threshold]
        if len(interior) >= 3:
            ranked = sorted(question.options, key=lambda o: -probabilities.get(o.key, 0))
            candidates.append((len(interior), request, question, ranked))
    rng = random.Random(f"{seed}:interior")
    rng.shuffle(candidates)
    candidates.sort(key=lambda c: -c[0])
    selected = []
    for _, request, question, ranked in candidates[:count]:
        if len(ranked) > 4:
            selected.append((replace(request, questions=(question,)), [o.description for o in ranked[:4]], ranked[4].description))
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} interior cases available")
    return selected


def plan_jobs_wave2(requests, seed=17, tokenizer=None, interior_cases=None):
    tokenizer = tokenizer or ByteTokenizer()
    jobs = []

    def add(family, case_id, variant, block, request, **extra):
        jobs.append({"job_id": f"{family}--{case_id}--{variant}--b{block}", "family": family, "case_id": case_id,
                     "variant": variant, "block": block, "request": request, "retained": extra.pop("retained", None), **extra})

    # B1: question isolation with a secret sibling, and two positive controls that place the secret in view.
    cases = []
    for domain in DOMAINS:
        cases += _pick(requests, lambda r, d=domain: domain_of(r) == d, 2, seed, f"B1:{domain}")
    for i, r in enumerate(cases):
        target = _noul("probe:secret", SECRET_QUESTION)
        english = _noul("probe:english", "Is the state written in English?")
        variants = {"alone": replace(r, questions=(target,)),
                    "repeat": replace(r, questions=(target,)),
                    "neutral_sibling": replace(r, questions=(target, english)),
                    "secret_sibling": replace(r, questions=(target, _noul("probe:english", SECRET + " Is the state written in English?"))),
                    "secret_in_state": replace(r, state=SECRET + "\n" + r.state, questions=(target,)),
                    "secret_in_target": replace(r, questions=(_noul("probe:secret", SECRET + " " + SECRET_QUESTION),))}
        for block in range(4):
            for name in B1_VARIANTS:
                add("B1", f"{domain_of(r)}-{i}", name, block, variants[name])
    # B3: reference dependence across four locations of the reference price.
    pairs = ((40, 60), (15, 35), (120, 140), (70, 90))
    for t, (lo, hi) in enumerate(pairs):
        a, b = TRAIN_NAMES[2 * t], TRAIN_NAMES[2 * t + 1]
        for r_value in (lo + 2, hi - 2):
            gold = "a" if abs(r_value - lo) < abs(r_value - hi) else "b"
            for location in B3_LOCATIONS:
                instructions = "Which option's price is closest to the reference price?"
                state = "See the options."
                options = {"a": f"{a}: ${lo}", "b": f"{b}: ${hi}"}
                if location == "state":
                    state = f"Reference price: ${r_value}."
                elif location == "instructions":
                    state = "See the question."
                    instructions += f" The reference price is ${r_value}."
                elif location == "earlier_option":
                    options = {"ref": f"Reference card: the reference price is ${r_value}", **options}
                else:
                    options["ref"] = f"Reference card: the reference price is ${r_value}"
                target = [float(k == gold) for k in options]
                request = Request.from_dict({"state": state, "group_id": f"b3:{t}:{r_value}:{location}", "questions": {
                    "probe:closest": {"type": "choice", "instructions": instructions, "criteria": options, "target": target}}})
                for rep in range(6):
                    add("B3", f"task-{t}-r{r_value}", f"{location}:rep{rep}", 0, request)
    # B4: amortisation. `reused` keeps the state byte-identical across reps (rep 0 primes); `fresh` changes its first sentence.
    for state_tokens in B4_STATE_TOKENS:
        for count in B4_QUESTION_COUNTS:
            for condition in ("reused", "fresh"):
                for rep in range(6):
                    salt = f"Note {rep}." if condition == "fresh" else ""
                    state = synthetic_state(tokenizer, state_tokens, salt)
                    questions = {f"q{i}": {"type": "choice", "instructions": f"Which team should handle this message? (variant {rep}-{i})",
                                           "criteria": {"payments": "Payment and transfer problems", "access": "Login problems",
                                                        "billing": "Billing questions", "other": "Anything else"}}
                                 for i in range(count)}
                    request = Request.from_dict({"state": state, "group_id": f"b4:s{state_tokens}:q{count}:{condition}:{rep}", "questions": questions})
                    add("B4", f"s{state_tokens}-q{count}", f"{condition}:rep{rep}", 0, request,
                        state_tokens=state_tokens, questions=count, condition=condition, rep=rep)
    # D4: latency versus option count on one panel state.
    base = _pick(requests, lambda r: domain_of(r) == "banking77", 1, seed, "D4")[0]
    for k in D4_COUNTS:
        options = {f"o{i}": f"Item {i}: {D4_WORDS[i % 64]} {D4_WORDS[(i * 7) % 64]}" for i in range(k)}
        request = Request.from_dict({"state": base.state, "group_id": base.group_id, "questions": {
            "probe:pick": {"type": "choice", "instructions": "Which item best matches the message?", "criteria": options}}})
        for rep in range(6):
            add("D4", f"k{k}", f"rep{rep}", 0, request, option_count=k)
    # D7: JSON versus prose renderings of the same T0 worlds with known targets.
    for i in range(25):
        for style in ("json", "prose"):
            _, ord_request = ordinal_world(random.Random(f"d7:ord:{i}"), 4, False, style)
            add("D7", f"ord-{i}", style, 0, ord_request, world_id=ord_request.group_id, style=style)
            _, post_request = posterior_world(random.Random(f"d7:post:{i}"), posterior_family(0), 0, style)
            add("D7", f"post-{i}", style, 0, post_request, world_id=post_request.group_id, style=style)
    # D1b and B2b: interior menus selected from saved Jev baselines.
    for index, (request, retained, extra) in enumerate(interior_cases or []):
        q = request.questions[0]
        by_desc = {o.description: o for o in q.options}
        keep = [by_desc[d] for d in retained] + [by_desc[extra]]
        base = _with_options(request, q, tuple(keep))
        for variant in ["base", "removed"] + [f"e{n:02d}" for n in range(20)]:
            add("D1b", f"interior-{index}", variant, 0, d1_variant(base, variant), retained=retained)
        for block in range(4):
            for variant in ("baseline", "repeat", "added", "replaced"):
                add("B2b", f"interior-{index}", variant, block, b2_variant(request, variant, index, protect=retained), retained=retained)
    rng = random.Random(seed)
    by_block = defaultdict(list)
    for job in jobs:
        by_block[job["block"]].append(job)
    ordered = []
    for block in sorted(by_block):
        group = by_block[block]
        rng.shuffle(group)
        ordered.extend(group)
    # Stable sort so B4 priming (rep 0) precedes its reuse reps; everything else keeps its shuffled order.
    ordered.sort(key=lambda j: (j["block"], j["rep"] if j["family"] == "B4" else 0))
    if len(ordered) > MAX_CALLS:
        raise ValueError("Plan exceeds the call budget")
    return ordered


def probability_floor(paths):
    """Delivered values (before renormalisation) from every cached envelope under `paths`."""
    values, nouls = [], []
    for root in paths:
        if not Path(root).exists():
            continue
        for path in Path(root).rglob("*.json"):
            try:
                envelope = json.loads(path.read_text())
                body = json.loads(envelope["response_body"]) if "response_body" in envelope else None
            except (ValueError, KeyError, TypeError):
                continue
            if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
                continue
            for answer in body["answers"].values():
                if not isinstance(answer, dict):
                    continue
                if answer.get("type") == "noul":
                    nouls.append(float(answer["noul"]))
                else:
                    values.extend(float(v) for v in answer.get("probabilities", {}).values())
    positive = sorted({v for v in values if v > 0})
    return {"count": len(values) + len(nouls), "choice_score_count": len(values), "noul_count": len(nouls),
            "smallest_nonzero": positive[0] if positive else None,
            "fraction_exact_zero": (sum(v == 0 for v in values) / len(values)) if values else None,
            "noul_min": min(nouls) if nouls else None, "noul_max": max(nouls) if nouls else None,
            "distinct_values_sample": positive[:10]}


def _median(values):
    return float(np.median(values)) if values else None


def _interaction_rows(groups, base_name, null_names=()):
    """Base-versus-variant rows for interior menus: residual over `retained`, TV over common options, argmax flips."""
    rows = defaultdict(list)
    for (case, block), variants in groups.items():
        base = variants.get(base_name)
        if base is None:
            continue
        b_al = aligned(base)
        for name, r in variants.items():
            if name == base_name:
                continue
            v_al = aligned(r)
            common = [d for d in base["descriptions"] if d in v_al]
            res = rescaling_residual(b_al, v_al, base["retained"])
            rows[name].append({"tv": total_variation(b_al, v_al, common),
                               "flip": max(b_al, key=b_al.get) != max(v_al, key=v_al.get),
                               "residual_rms": res["residual_rms"] if res else None,
                               "rank_reversal": res["rank_reversal"] if res else None,
                               "defined": res is not None})
    return rows


def _interaction_stats(rows):
    return {**_stats([r["tv"] for r in rows], "tv"), "flips": int(sum(r["flip"] for r in rows)), "count": len(rows),
            "residual_defined_fraction": float(np.mean([r["defined"] for r in rows])) if rows else None,
            **_stats([r["residual_rms"] for r in rows], "residual_rms"),
            "rank_reversals": int(sum(bool(r["rank_reversal"]) for r in rows))}


def plan_jobs_wave3(requests, seed=17, per_domain=50):
    """D3 redo: affirm, negate, and an exact repeat of each, so the complement gap has a contemporaneous null."""
    jobs = []
    for domain in DOMAINS:
        pool = [r for r in requests if domain_of(r) == domain and negate_noul(r.questions[1]) is not None]
        for i, r in enumerate(_pick(pool, lambda r: True, per_domain, seed, f"D3:{domain}")):
            noul = r.questions[1]
            affirm = replace(r, questions=(noul,))
            negate = replace(r, questions=(negate_noul(noul),))
            for variant, request in (("affirm", affirm), ("negate", negate), ("affirm_repeat", affirm), ("negate_repeat", negate)):
                jobs.append({"job_id": f"D3b--{domain}-{i}--{variant}--b0", "family": "D3b", "case_id": f"{domain}-{i}",
                             "variant": variant, "block": 0, "request": request, "retained": None})
    random.Random(seed).shuffle(jobs)
    if len(jobs) > MAX_CALLS:
        raise ValueError("Plan exceeds the call budget")
    return jobs


def summarize_wave3(records):
    ok = [r for r in records if "error" not in r]
    by = defaultdict(dict)
    for r in ok:
        by[r["case_id"]][r["variant"]] = r["probabilities"][1]
    out = {}
    for domain in DOMAINS + ("all",):
        gaps, noise = [], []
        for case, v in by.items():
            if domain != "all" and not case.startswith(domain + "-"):
                continue
            if {"affirm", "negate", "affirm_repeat", "negate_repeat"} <= set(v):
                gaps.append(v["affirm"] + v["negate"] - 1)
                noise.append((v["affirm"] - v["affirm_repeat"]) + (v["negate"] - v["negate_repeat"]))
        if gaps:
            q = np.quantile(gaps, [.05, .5, .95])
            out[domain] = {"pairs": len(gaps), "gap_mean": float(np.mean(gaps)), "gap_sd": float(np.std(gaps)),
                           "gap_q05": float(q[0]), "gap_q50": float(q[1]), "gap_q95": float(q[2]),
                           "fraction_abs_gap_over_0.1": float(np.mean(np.abs(gaps) > .1)),
                           "repeat_noise_mean_abs": float(np.mean(np.abs(noise))), "repeat_noise_sd": float(np.std(noise)),
                           "fraction_abs_noise_over_0.1": float(np.mean(np.abs(noise) > .1))}
    latencies = [r["latency_seconds"] for r in ok if "latency_seconds" in r]
    return {"D3b": out, "latency_p50_seconds": float(np.median(latencies)) if latencies else None,
            "usage": {k: sum(r["usage"][k] for r in ok if "usage" in r) for k in ("input_tokens", "output_tokens")},
            "reading": "Complement gap g = p(X) + p(not X) - 1 per proposition, with the same two requests repeated as the null. "
                       "Measures logical consistency of the output contract; it does not measure leaf independence."}


def summarize_wave2(records, floor_sources=()):
    ok = [r for r in records if "error" not in r]
    by = defaultdict(list)
    for r in ok:
        by[r["family"]].append(r)
    # B1
    b1 = defaultdict(list)
    for r in by["B1"]:
        b1[r["variant"]].append(r["probabilities"][1])
    b1_out = {name: {"mean": float(np.mean(v)), "max": float(np.max(v)), "min": float(np.min(v)), "count": len(v)}
              for name, v in b1.items()}
    # B3
    b3 = defaultdict(list)
    for r in by["B3"]:
        location = r["variant"].split(":")[0]
        target, p = r["targets"], r["probabilities"]
        gold = max(range(len(target)), key=target.__getitem__)
        ref = r["keys"].index("ref") if "ref" in r["keys"] else None
        b3[location].append((max(range(len(p)), key=p.__getitem__) == gold, p[gold], p[ref] if ref is not None else None))
    b3_out = {}
    for loc in B3_LOCATIONS:
        rows = b3.get(loc, [])
        if not rows:
            continue
        out = {"argmax_accuracy": float(np.mean([a for a, _, _ in rows])), "mean_target_probability": float(np.mean([p for _, p, _ in rows])),
               "count": len(rows)}
        ref_mass = [m for _, _, m in rows if m is not None]
        if ref_mass:
            out["mean_reference_option_probability"] = float(np.mean(ref_mass))
        b3_out[loc] = out
    # B4
    b4 = defaultdict(lambda: {"fresh": [], "priming": [], "reuse": []})
    fresh_rows = []
    for r in by["B4"]:
        key = f"s{r['state_tokens']}-q{r['questions']}"
        bucket = "fresh" if r["condition"] == "fresh" else ("priming" if r["rep"] == 0 else "reuse")
        b4[key][bucket].append(r["latency_seconds"] * 1000)
        if bucket == "fresh":
            fresh_rows.append((r["state_tokens"], r["questions"] * B4_QUESTION_TOKENS, r["latency_seconds"] * 1000))
    fits = {"fresh": {"ms_per_state_token": None, "ms_per_question_token": None, "fixed_ms": None}}
    if len(fresh_rows) >= 3:
        a = np.array([[s, q, 1.] for s, q, _ in fresh_rows])
        b = np.array([t for _, _, t in fresh_rows])
        c = np.linalg.lstsq(a, b, rcond=None)[0]
        fits["fresh"] = {"ms_per_state_token": float(c[0]), "ms_per_question_token": float(c[1]), "fixed_ms": float(c[2])}
    cells = {}
    for s in B4_STATE_TOKENS:
        for q in B4_QUESTION_COUNTS:
            key = f"s{s}-q{q}"
            if key in b4:
                cells[key] = {name: _median(v) for name, v in b4[key].items()}
                cells[key]["counts"] = [len(b4[key][name]) for name in ("fresh", "priming", "reuse")]
    b4_out = {"cells": cells, "fits": fits}
    # D4
    d4 = defaultdict(list)
    for r in by["D4"]:
        d4[str(r["option_count"])].append(r["latency_seconds"] * 1000)
    ks = sorted(d4, key=int)
    slope = None
    if len(ks) >= 2:
        slope = float(np.polyfit([int(k) for k in ks], [_median(d4[k]) for k in ks], 1)[0])
    d4_out = {"by_k": {k: _median(v) for k, v in d4.items()}, "ms_per_option": slope}
    # D7
    pairs = defaultdict(dict)
    for r in by["D7"]:
        pairs[r["case_id"]][r["style"]] = r
    per_family = defaultdict(lambda: {"json_nll": [], "prose_nll": [], "json_acc": [], "prose_acc": []})
    for case, styles in pairs.items():
        if "json" in styles and "prose" in styles:
            fam = per_family[case.split("-")[0]]
            for style in ("json", "prose"):
                p, t = styles[style]["probabilities"], styles[style]["targets"]
                fam[f"{style}_nll"].append(-sum(ti * math.log(max(pi, 1e-12)) for pi, ti in zip(p, t)))
                fam[f"{style}_acc"].append(t[max(range(len(p)), key=p.__getitem__)])

    def d7_stats(fam):
        n = len(fam["json_nll"])
        return {"pairs": n, "nll_json": float(np.mean(fam["json_nll"])) if n else None,
                "nll_prose": float(np.mean(fam["prose_nll"])) if n else None,
                "accuracy_json": float(np.mean(fam["json_acc"])) if n else None,
                "accuracy_prose": float(np.mean(fam["prose_acc"])) if n else None,
                "mean_paired_nll_difference_json_minus_prose": float(np.mean(np.array(fam["json_nll"]) - np.array(fam["prose_nll"]))) if n else None,
                "pairs_json_lower_nll": int(np.sum(np.array(fam["json_nll"]) < np.array(fam["prose_nll"]))) if n else None}
    combined = {k: sum((f[k] for f in per_family.values()), []) for k in ("json_nll", "prose_nll", "json_acc", "prose_acc")}
    d7_out = {**d7_stats(combined), "by_family": {name: d7_stats(fam) for name, fam in sorted(per_family.items())}}
    # D1b and B2b
    grouped = defaultdict(lambda: defaultdict(dict))
    for r in by["D1b"] + by["B2b"]:
        grouped[r["family"]][(r["case_id"], r["block"])][r["variant"]] = r
    d1_rows = _interaction_rows(grouped["D1b"], "base")
    e_rows = sum((rows for name, rows in d1_rows.items() if name.startswith("e")), [])
    removed_rows = d1_rows.get("removed", [])
    d1b = {"variants": len(e_rows), "residual_defined_fraction": float(np.mean([r["defined"] for r in e_rows])) if e_rows else None,
           **_stats([r["residual_rms"] for r in e_rows], "residual_rms"),
           "rank_reversals": int(sum(bool(r["rank_reversal"]) for r in e_rows)),
           **_stats([r["tv"] for r in e_rows], "tv"), "flips": int(sum(r["flip"] for r in e_rows)),
           "removed_count": len(removed_rows),
           "removed_residual_defined_fraction": float(np.mean([r["defined"] for r in removed_rows])) if removed_rows else None,
           **_stats([r["residual_rms"] for r in removed_rows], "removed_residual_rms"),
           "removed_rank_reversals": int(sum(bool(r["rank_reversal"]) for r in removed_rows)),
           **_stats([r["tv"] for r in removed_rows], "removed_tv")}
    b2b = {name: _interaction_stats(rows) for name, rows in _interaction_rows(grouped["B2b"], "baseline").items()}
    latencies = [r["latency_seconds"] for r in ok if "latency_seconds" in r]
    return {"B1": b1_out, "B3": b3_out, "B4": b4_out, "D4": d4_out, "D5": probability_floor(list(floor_sources)), "D7": d7_out,
            "D1b": d1b, "B2b": b2b,
            "latency_p50_seconds": float(np.median(latencies)) if latencies else None,
            "usage": {k: sum(r["usage"][k] for r in ok if "usage" in r) for k in ("input_tokens", "output_tokens")},
            "latency_note": "Client-side end-to-end through a fresh HTTPS connection per call; not server time.",
            "reading": "Behavioural statistics with positive controls (B1) and nulls (B1 alone, B3 state location, B4 fresh, "
                       "D1b removed, B2b repeat); they constrain hypotheses about API behaviour and do not identify an implementation."}
