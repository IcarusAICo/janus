# tests/test_blocked_probes.py
from collections import Counter
import hashlib
import io
import json
import math
from urllib.error import HTTPError

import pytest

from janus.data import write_jsonl
from janus.schema import Question, Request


def panel(n_per_domain=60):
    """Panel-shaped requests: 8-option intents, 5-level sst5, 3-option snli, each with a templated Noul."""
    requests = []
    for domain in ("banking77", "clinc150"):
        who = "customer's message" if domain == "banking77" else "user's request"
        verb = "express"
        for i in range(n_per_domain):
            options = {f"o{k}": f"{domain} intent {i}-{k}" for k in range(8)}
            requests.append(Request.from_dict({"state": f"{domain} state {i}", "group_id": f"{domain}:{i}", "questions": {
                f"{domain}:intent": {"type": "choice", "instructions": f"Which intent best describes the {who}?",
                                     "criteria": options, "target": [1.] + [0.] * 7},
                f"{domain}:matches": {"type": "noul", "instructions": f"Does the {who} {verb} this intent: intent {i}?",
                                      "target": [0., 1.]}}}))
    for i in range(n_per_domain):
        requests.append(Request.from_dict({"state": f"review {i}", "group_id": f"sst5:{i}", "questions": {
            "sst5:sentiment": {"type": "score", "instructions": "Rate the overall sentiment of this movie review.",
                               "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"], "target": [0, 0, 1, 0, 0]},
            "sst5:positive": {"type": "noul", "instructions": "Does this movie review express positive overall sentiment? Neutral sentiment does not count as positive.",
                              "target": [1., 0.]}}}))
    for i in range(n_per_domain):
        requests.append(Request.from_dict({"state": f"Premise: p{i}\nHypothesis: h{i}", "group_id": f"snli:{i}", "questions": {
            f"snli:relation:{i}": {"type": "choice", "instructions": "Assume the premise is true. What relationship does the hypothesis have to the premise?",
                                   "criteria": {"o0": "Entailed", "o1": "Unknown", "o2": "Contradicted"}, "target": [1., 0., 0.]},
            f"snli:entailed:{i}": {"type": "noul", "instructions": "Assume the premise is true. Is the hypothesis logically entailed by the premise? Both contradiction and insufficient information mean it is not entailed.",
                                   "criteria": {"false": "No.", "true": "Yes."}, "target": [0., 1.]}}}))
    return requests


def test_negation_rules_cover_all_four_domains_and_swap_targets():
    from janus.blocked_probes import negate_noul
    for request in panel(1):
        noul = request.questions[1]
        negated = negate_noul(noul)
        assert negated is not None and negated.id == noul.id + ":neg"
        assert negated.instructions != noul.instructions and "false that" in negated.instructions
        assert negated.target == tuple(reversed(noul.target))
    assert negate_noul(Question.from_dict("x", {"type": "noul", "instructions": "Unknown template?"})) is None
    snli = negate_noul(panel(1)[-1].questions[1])
    assert [o.description for o in snli.options] == ["No.", "Yes."]  # Yes/No stay with their keys; only explanatory clauses swap


def test_plan_has_1104_jobs_in_blocks_with_correct_family_counts():
    from janus.blocked_probes import plan_jobs
    jobs = plan_jobs(panel())
    assert len(jobs) == 1104
    families = Counter(j["family"] for j in jobs)
    assert families == {"B0": 144, "B2": 384, "D1": 176, "D3": 400}
    blocks = [j["block"] for j in jobs]
    assert blocks == sorted(blocks)
    assert len({j["job_id"] for j in jobs}) == 1104
    b2 = [j for j in jobs if j["family"] == "B2"]
    assert Counter(j["variant"] for j in b2) == {v: 64 for v in ("baseline", "repeat", "permutation", "reversal", "added", "replaced")}
    assert Counter(j["request"].questions[0].kind for j in b2 if j["variant"] == "baseline") == {"choice": 32, "score": 32}
    d1 = [j for j in jobs if j["family"] == "D1"]
    assert Counter(j["variant"] for j in d1)["base"] == 8 and Counter(j["variant"] for j in d1)["removed"] == 8
    base = next(j for j in d1 if j["variant"] == "base")
    assert len(base["request"].questions[0].options) == 5 and len(base["retained"]) == 4
    assert all(len(j["request"].questions[0].options) == (4 if j["variant"] == "removed" else 5) for j in d1)
    d3 = [j for j in jobs if j["family"] == "D3"]
    assert Counter(j["variant"] for j in d3) == {"affirm": 200, "negate": 200}


def test_b2_variants_change_only_what_they_claim():
    from janus.blocked_probes import plan_jobs
    jobs = [j for j in plan_jobs(panel()) if j["family"] == "B2" and j["block"] == 0]
    by = {}
    for j in jobs:
        by.setdefault(j["case_id"], {})[j["variant"]] = j["request"].questions[0]
    for case, variants in by.items():
        base = variants["baseline"]
        descs = [o.description for o in base.options]
        assert [o.description for o in variants["repeat"].options] == descs
        assert [o.description for o in variants["permutation"].options] == descs[1:] + descs[:1]
        assert [o.description for o in variants["reversal"].options] == descs[::-1]
        assert [o.description for o in variants["added"].options][:-1] == descs and len(variants["added"].options) == len(descs) + 1
        replaced = [o.description for o in variants["replaced"].options]
        assert len(replaced) == len(descs) and sum(a != b for a, b in zip(replaced, descs)) == 1


def _z(desc, salt=""):
    return (int(hashlib.sha256((salt + desc).encode()).hexdigest(), 16) % 1000) / 250 - 2


def independent_responder(payload):
    answers = {}
    for qid, q in payload["questions"].items():
        if q["type"] == "noul":
            text = q["instructions"]
            key = text.split("intent: ")[-1] if "intent: " in text else ("sst5" if "movie review" in text else "snli")
            p = 1 / (1 + math.exp(-_z(key)))
            if "false that" in text:
                p = 1 - p
            answers[qid] = {"type": "noul", "noul": round(p, 2)}
        else:
            items = list(q["criteria"].items()) if isinstance(q["criteria"], dict) else list(enumerate(q["criteria"]))
            z = [_z(d if isinstance(d, str) else json.dumps(d)) for _, d in items]
            m = max(z)
            e = [math.exp(v - m) for v in z]
            p = {str(k): round(v / sum(e), 4) for (k, _), v in zip(items, e)}
            answers[qid] = {"type": q["type"], "probabilities": p, "confidence": .5}
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 20}}


def contextual_responder(payload):
    answers = {}
    for qid, q in payload["questions"].items():
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": .7}
            continue
        items = list(q["criteria"].items()) if isinstance(q["criteria"], dict) else list(enumerate(q["criteria"]))
        salt = str(len(items)) + "".join(sorted(d for _, d in items if isinstance(d, str)))
        z = [_z(d, salt) for _, d in items]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        answers[qid] = {"type": q["type"], "probabilities": {str(k): round(v / sum(e), 4) for (k, _), v in zip(items, e)}, "confidence": .5}
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 20}}


class _Response(io.BytesIO):
    status = 200


def _fixture(tmp_path, monkeypatch, responder, status=None):
    from janus import remote
    data = tmp_path / "panel.jsonl"
    write_jsonl(data, [r.to_dict() for r in panel()])
    calls = []

    def transport(outgoing, timeout):
        calls.append(outgoing)
        if status is not None:
            raise HTTPError(outgoing.full_url, status, "fake-secret", {}, io.BytesIO(b"fake-secret"))
        return _Response(json.dumps(responder(json.loads(outgoing.data))).encode())

    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-secret")
    monkeypatch.setattr(remote, "_http_open", transport)
    return data, calls


def test_run_executes_all_jobs_once_persists_records_and_resumes(tmp_path, monkeypatch):
    from janus.blocked_probes import run_blocked_probes
    data, calls = _fixture(tmp_path, monkeypatch, independent_responder)
    output = tmp_path / "probes"
    report = run_blocked_probes(data, output)
    assert len(calls) == 1104 == report["http_calls"]
    records = list((output / "jobs").glob("*.json"))
    assert len(records) == 1104
    record = json.loads(records[0].read_text())
    assert set(record) >= {"job_id", "family", "variant", "block", "payload", "request_sha256", "response", "returned_model", "usage", "latency_seconds", "started_at"}
    assert hashlib.sha256(json.dumps(record["payload"], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() == record["request_sha256"]
    assert all("fake-secret" not in p.read_text() for p in output.rglob("*.json"))
    again = run_blocked_probes(data, output, resume=True)
    assert len(calls) == 1104 and again["http_calls"] == 0


def test_run_stops_on_model_mismatch_and_records_http_failures(tmp_path, monkeypatch):
    from janus.blocked_probes import run_blocked_probes
    from janus.remote import RemoteError

    def other_model(payload):
        r = independent_responder(payload)
        r["model"] = "jev-9.9.9"
        return r
    data, calls = _fixture(tmp_path, monkeypatch, other_model)
    with pytest.raises(RemoteError, match="model"):
        run_blocked_probes(data, tmp_path / "mismatch")
    assert len(calls) == 1
    data, calls = _fixture(tmp_path / "b", monkeypatch, independent_responder, status=503)
    report = run_blocked_probes(data, tmp_path / "failures")
    assert report["failed_jobs"] == 1104 and len(calls) == 1104


def test_rescaling_residual_is_zero_for_independent_logits_and_positive_for_contextual():
    from janus.blocked_probes import rescaling_residual
    # All four squared-and-renormalized probabilities must stay at or above the 0.02 interior floor.
    base = {"a": .45, "b": .3, "c": .15, "d": .1}
    scaled = {k: v ** 2 for k, v in base.items()}
    total = sum(scaled.values())
    scaled = {k: v / total for k, v in scaled.items()}
    out = rescaling_residual(base, scaled, ["a", "b", "c", "d"])
    assert out["retained_interior"] == 4 and out["scale"] == pytest.approx(2., abs=1e-6)
    assert out["residual_rms"] < 1e-6 and out["rank_reversal"] is False
    contextual = {"a": .3, "b": .45, "c": .15, "d": .1}
    out = rescaling_residual(base, contextual, ["a", "b", "c", "d"])
    assert out["residual_rms"] > .1 and out["rank_reversal"] is True
    assert rescaling_residual(base, {"a": .98, "b": .01, "c": .005, "d": .005}, ["a", "b", "c", "d"]) is None


def test_summary_separates_independent_from_contextual_behaviour(tmp_path, monkeypatch):
    from janus.blocked_probes import run_blocked_probes
    data, _ = _fixture(tmp_path, monkeypatch, independent_responder)
    independent = run_blocked_probes(data, tmp_path / "ind")
    assert independent["B0"]["within_block_max_tv"] < 1e-9
    assert independent["D1"]["residual_rms_mean"] < .05
    assert independent["D1"]["rank_reversals"] == 0
    assert abs(independent["D3"]["gap_mean"]) < .02
    assert independent["B2"]["choice"]["added"]["residual_rms_mean"] < .05
    assert independent["B2"]["score"]["reversal"]["expected_level_mismatch_mean"] < .02
    data, _ = _fixture(tmp_path / "c", monkeypatch, contextual_responder)
    contextual = run_blocked_probes(data, tmp_path / "ctx")
    assert contextual["D1"]["residual_rms_mean"] > independent["D1"]["residual_rms_mean"]
    assert contextual["B2"]["choice"]["repeat"]["tv_mean"] < 1e-9
    md = (tmp_path / "ctx" / "summary.md").read_text()
    assert "| D1 |" in md and "residual" in md


def test_wave2_plan_counts_and_positive_controls():
    from janus.blocked_probes import SECRET, SECRET_QUESTION, plan_jobs_wave2
    interior = [(r, [o.description for o in r.questions[0].options[:4]], r.questions[0].options[4].description) for r in panel()[:8]]
    jobs = plan_jobs_wave2(panel(), interior_cases=interior)
    families = Counter(j["family"] for j in jobs)
    assert families == {"B1": 192, "B3": 192, "B4": 192, "D4": 48, "D7": 100, "D1b": 176, "B2b": 128}
    assert len(jobs) == 1028 and len({j["job_id"] for j in jobs}) == 1028
    blocks = [j["block"] for j in jobs]
    assert blocks == sorted(blocks)
    d1b = [j for j in jobs if j["family"] == "D1b"]
    assert all(len(j["retained"]) == 4 for j in d1b) and Counter(j["variant"] for j in d1b)["removed"] == 8
    b1 = {j["variant"]: j["request"] for j in jobs if j["family"] == "B1" and j["case_id"].endswith("-0") and j["block"] == 0}
    assert b1["alone"].questions[0].instructions == SECRET_QUESTION and SECRET not in b1["alone"].state
    assert SECRET in b1["secret_sibling"].questions[1].instructions and SECRET not in b1["secret_sibling"].state
    assert b1["secret_in_state"].state.startswith(SECRET) and len(b1["secret_in_state"].questions) == 1
    assert b1["secret_in_target"].questions[0].instructions.startswith(SECRET)
    b3 = [j for j in jobs if j["family"] == "B3"]
    assert Counter(j["variant"].split(":")[0] for j in b3) == {"state": 48, "instructions": 48, "earlier_option": 48, "later_option": 48}
    for j in b3:
        q = j["request"].questions[0]
        location = j["variant"].split(":")[0]
        if location.endswith("option"):
            assert len(q.options) == 3 and q.options[0 if location == "earlier_option" else -1].key == "ref"
            assert q.target[0 if location == "earlier_option" else -1] == 0.
        else:
            assert len(q.options) == 2
    b4 = [j for j in jobs if j["family"] == "B4"]
    reused = [j for j in b4 if j["condition"] == "reused" and j["state_tokens"] == 500 and j["questions"] == 10]
    assert len(reused) == 6 and len({j["request"].state for j in reused}) == 1
    fresh = [j for j in b4 if j["condition"] == "fresh" and j["state_tokens"] == 500 and j["questions"] == 10]
    assert len({j["request"].state for j in fresh}) == 6
    positions = {j["job_id"]: i for i, j in enumerate(jobs)}
    for j in reused:
        assert j["rep"] == 0 or positions[j["job_id"]] > positions[reused[0]["job_id"]] or reused[0]["rep"] != 0
    assert max(positions[j["job_id"]] for j in b4 if j["rep"] == 0) < min(positions[j["job_id"]] for j in b4 if j["rep"] == 1)
    d4 = [j for j in jobs if j["family"] == "D4"]
    assert sorted({len(j["request"].questions[0].options) for j in d4}) == [2, 4, 8, 16, 32, 64, 128, 255]
    d7 = [j for j in jobs if j["family"] == "D7"]
    assert Counter(j["style"] for j in d7) == {"json": 50, "prose": 50}
    assert all(q.target is not None for j in d7 for q in j["request"].questions)
    b2b = [j for j in jobs if j["family"] == "B2b" and j["variant"] == "replaced"]
    for j in b2b:
        assert all(d in {o.description for o in j["request"].questions[0].options} for d in j["retained"])


def test_wave2_summary_and_probability_floor(tmp_path, monkeypatch):
    from janus.blocked_probes import probability_floor, run_blocked_probes

    def responder(payload):
        answers = {}
        for qid, q in payload["questions"].items():
            if q["type"] == "noul":
                secret = "ZEBRA-7431" in payload["state"] or "ZEBRA-7431" in q["instructions"].replace("mention the access code ZEBRA-7431", "")
                answers[qid] = {"type": "noul", "noul": .95 if secret else .02}
            else:
                keys = [str(i) for i in range(len(q["criteria"]))] if isinstance(q["criteria"], list) else list(q["criteria"])
                p = {k: round(1 / len(keys), 4) for k in keys}
                p[keys[0]] = round(1 - sum(v for k, v in p.items() if k != keys[0]), 4)
                answers[qid] = {"type": q["type"], "probabilities": p, "confidence": .1}
        return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 20}}

    data, calls = _fixture(tmp_path, monkeypatch, responder)
    interior = [(r, [o.description for o in r.questions[0].options[:4]], r.questions[0].options[4].description) for r in panel()[:8]]
    report = run_blocked_probes(data, tmp_path / "w2", wave=2, interior_cases=interior)
    assert len(calls) == 1028
    assert report["B1"]["secret_sibling"]["mean"] < .1 and report["B1"]["secret_in_state"]["mean"] > .9
    assert set(report["B3"]) == {"state", "instructions", "earlier_option", "later_option"}
    assert report["B4"]["fits"]["fresh"]["ms_per_state_token"] is not None
    assert sorted(report["D4"]["by_k"]) == ["128", "16", "2", "255", "32", "4", "64", "8"]
    assert report["D7"]["pairs"] == 50 and "nll_json" in report["D7"] and "nll_prose" in report["D7"]
    assert "residual_defined_fraction" in report["D1b"] and "added" in report["B2b"]
    assert report["D1b"]["residual_defined_fraction"] == 1. and report["B2b"]["added"]["residual_defined_fraction"] == 1.
    record = json.loads(next((tmp_path / "w2" / "jobs").glob("D7--*.json")).read_text())
    assert "targets" in record and record["style"] in ("json", "prose")
    floor = probability_floor([tmp_path / "w2" / "cache"])
    assert floor["count"] > 0 and floor["noul_min"] == pytest.approx(.02) and floor["smallest_nonzero"] > 0
    md = (tmp_path / "w2" / "summary.md").read_text()
    assert "| B1 secret_sibling |" in md and "| D1b |" in md and "| D5 |" in md
    again = run_blocked_probes(data, tmp_path / "w2", wave=2, interior_cases=interior, resume=True)
    assert len(calls) == 1028 and again["http_calls"] == 0


def test_snli_negation_keeps_yes_no_attached_to_the_right_key():
    from janus.blocked_probes import negate_noul
    snli = panel(1)[-1].questions[1]
    negated = negate_noul(snli)
    assert negated.options[0].key == "false" and negated.options[0].description.startswith("No.")
    assert negated.options[1].key == "true" and negated.options[1].description.startswith("Yes.")


def test_wave3_plan_and_summary_have_repeat_nulls(tmp_path, monkeypatch):
    from janus.blocked_probes import plan_jobs_wave3, run_blocked_probes
    jobs = plan_jobs_wave3(panel())
    assert len(jobs) == 800 and Counter(j["variant"] for j in jobs) == {v: 200 for v in ("affirm", "negate", "affirm_repeat", "negate_repeat")}
    data, calls = _fixture(tmp_path, monkeypatch, independent_responder)
    report = run_blocked_probes(data, tmp_path / "w3", wave=3)
    assert len(calls) == 800 and report["D3b"]["all"]["pairs"] == 200
    assert abs(report["D3b"]["all"]["gap_mean"]) < .02 and report["D3b"]["all"]["repeat_noise_mean_abs"] < 1e-9
