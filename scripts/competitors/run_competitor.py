"""Run JevK5 or Laya over one of our request files, in the competitor's own venv.

    .venvs/laya/bin/python scripts/competitors/run_competitor.py laya data/public-v1/test.jsonl runs/competitors/laya/public.jsonl --device cuda:0
    .venvs/jevk5/bin/python scripts/competitors/run_competitor.py jevk5 data/hardtier-v2/test.jsonl runs/competitors/jevk5/hardtier_v2.jsonl --device cuda:0

Each question is sent exactly as a client would send it (our jsonl question minus `target`) through the competitor's
public Python API. Output: one row per question with the probability vector in OUR option order, the target,
group_id/family, and the request's latency (synchronised wall time around the API call, excluding model load, after
warm-up). A question the competitor cannot answer is written with `refused` set and `probabilities: null`, never
dropped. Stdlib + torch + the competitor's package only: this file must not import janus (the venvs do not have it).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

JEVK5_MAX_OPTIONS, JEVK5_MAX_TOKENS = 16, 16384  # JevK5's documented limits (README; bench/jevk5_direct.py)


def our_keys(q):
    """Option keys in our order (janus/schema.py): choice keys, score indices, noul false/true."""
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def api_question(q):
    return {k: v for k, v in q.items() if k != "target"}


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


class JevK5Runner:
    def __init__(self, device, model):
        from jevk5 import JevK5
        from jevk5.server import normalized
        if device.startswith("cuda"):
            torch.cuda.set_device(device)  # capture() records its CUDA graphs on the current device
        self.normalized = normalized
        self.model = JevK5(model or "alibiserikbay/JevK5", device=device, graphs=device.startswith("cuda"))
        self.name = model or "alibiserikbay/JevK5"

    def refusal(self, state, q):
        """JevK5's limits, checked before the timed call (so an over-long input is never run)."""
        try:
            q = self.normalized(q)  # the served API's own validation
        except ValueError as e:
            return f"invalid: {e}"
        if len(our_keys(q)) > JEVK5_MAX_OPTIONS:
            return f"{len(our_keys(q))} options > {JEVK5_MAX_OPTIONS}"
        from jevk5.runtime import decision_options
        n = len(self.model.encode(state, q["instructions"], [t for _, t in decision_options(q)]))
        return f"{n} input tokens > {JEVK5_MAX_TOKENS}" if n > JEVK5_MAX_TOKENS else None

    def request(self, state, questions):
        """{qid: (probabilities by key, extra)} for the answerable questions; one forward pass per question
        (JevK5's design). Refused questions are left out and reported by refusal()."""
        out = {}
        for qid, q in questions.items():
            probs, tokens = self.model.probabilities(state, self.normalized(q))
            out[qid] = (probs, {"input_tokens": tokens})
        return out


class LayaRunner:
    def __init__(self, device, model):
        import laya
        import laya.agent
        from laya import Router
        from laya.common import build_sequence
        # ponytail: Laya rounds every returned probability to 4 decimals, which turns a small probability into 0
        # and an NLL into infinity. Shadowing `round` in laya.agent returns its exact calibrated values; the
        # arithmetic is otherwise untouched. Remove if a later release stops rounding.
        laya.agent.round = lambda x, n=None: x
        self.build_sequence, self.Agent = build_sequence, laya.agent.Agent
        self.checkpoint = None if model in (None, "router") else model
        self.router = Router(device=device, max_loaded=3)
        self.router.preload([self.checkpoint] if self.checkpoint else ["english", "multilingual"])
        self.name = f"laya {laya.__version__} ({model or 'router'})"

    def refusal(self, state, q):
        try:
            self.Agent._check_question("q", q)
        except ValueError as e:
            return f"invalid: {e}"
        return None

    def truncated(self, agent, state, q):
        """Laya truncates silently (512 tokens English, 1024 multilingual/typed-decisions): flag it."""
        q = self.Agent._to_internal(q)
        cut, _ = self.build_sequence(agent.tok, state, q, agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192))
        full, _ = self.build_sequence(agent.tok, state, q, 10 ** 9, 10 ** 9)
        return len(cut) < len(full)

    def request(self, state, questions):
        result = self.router.predict(state, questions, model=self.checkpoint)
        out = {}
        for qid, a in result["answers"].items():
            probs = {"false": 1 - a["noul"], "true": a["noul"]} if a["type"] == "noul" else a["probabilities"]
            out[qid] = (probs, {"checkpoint": result["routing"]["model"]})
        return out

    def annotate(self, state, questions, out):
        for qid, (_, extra) in out.items():
            extra["truncated"] = self.truncated(self.router.load(extra["checkpoint"]), state, questions[qid])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("competitor", choices=["jevk5", "laya"])
    parser.add_argument("data")
    parser.add_argument("output")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", default=None,
                        help="jevk5: Hub id or folder; laya: router (default), english, multilingual, typed-decisions")
    parser.add_argument("--limit", type=int, default=None, help="seeded sample, identical to janus evaluate --limit")
    parser.add_argument("--warmup", type=int, default=5, help="requests run once untimed before the timed pass")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:  # the same subset as `python -m janus evaluate --limit` (janus/evaluation.py)
        rows = random.Random(17).sample(rows, min(args.limit, len(rows)))
    started = time.perf_counter()
    runner = (JevK5Runner if args.competitor == "jevk5" else LayaRunner)(args.device, args.model)
    load_s = time.perf_counter() - started

    def prepare(row):
        """(state, answerable questions, {qid: refusal reason})."""
        state = row["state"]
        if isinstance(state, dict) and "images" in state:
            return state, {}, {qid: "image state unsupported" for qid in row["questions"]}
        questions, refused = {}, {}
        for qid, q in row["questions"].items():
            q = api_question(q)
            reason = runner.refusal(state, q)
            if reason:
                refused[qid] = reason
            else:
                questions[qid] = q
        return state, questions, refused

    prepared = [prepare(row) for row in rows]
    for state, questions, _ in [p for p in prepared if p[1]][:args.warmup]:
        runner.request(state, questions)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"questions": 0, "refused": 0, "failed": 0, "truncated": 0}
    with output.open("w") as sink:
        for index, (row, (state, questions, refused)) in enumerate(zip(rows, prepared)):
            answers, latency_ms, error = {}, None, None
            if questions:
                try:
                    sync(args.device)
                    t0 = time.perf_counter()
                    answers = runner.request(state, questions)
                    sync(args.device)
                    latency_ms = (time.perf_counter() - t0) * 1e3
                except (ValueError, KeyError, RuntimeError) as e:  # a request the competitor rejects or crashes on
                    error = f"{type(e).__name__}: {str(e)[:300]}"
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
            if answers and hasattr(runner, "annotate"):
                runner.annotate(state, questions, answers)
            for qid, q in row["questions"].items():
                keys = our_keys(q)
                record = {"request_index": index, "group_id": row.get("group_id", ""), "question_id": qid,
                          "kind": q["type"], "family": row.get("family"), "cardinality": len(keys), "keys": keys,
                          "target": q.get("target"), "probabilities": None, "refused": refused.get(qid),
                          "latency_ms": latency_ms, "questions_in_request": len(questions)}
                if qid in answers:
                    probs, extra = answers[qid]
                    p = [float(probs[k]) for k in keys]
                    if not all(math.isfinite(x) and x >= 0 for x in p) or abs(sum(p) - 1) > 1e-3:
                        record["refused"] = f"invalid distribution {p}"
                    else:
                        record["probabilities"] = p
                    record.update(extra)
                elif qid in questions:
                    record["refused"] = error or "no answer returned"
                counts["questions"] += 1
                counts["refused"] += record["refused"] is not None
                counts["failed"] += qid in questions and record["probabilities"] is None
                counts["truncated"] += bool(record.get("truncated"))
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {"competitor": args.competitor, "model": runner.name, "data": args.data, "rows": len(rows),
            "device": args.device, "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
            "torch": torch.__version__, "load_s": load_s, "warmup_requests": args.warmup, **counts}
    Path(str(output) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
