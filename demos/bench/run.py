"""Score System One JSONL with TypeSafe Jev or GPT-5.6. Writes summary JSON, not secrets."""
from __future__ import annotations

import argparse
import json
import time
from urllib.request import Request, urlopen
from pathlib import Path

from demos.bench.tasks import BenchItem, load_task
from demos.common.metrics import rates_for, summarize
from demos.common.record import artifact_path, env_with_keys

TASKS = (
    "synthetic",
    "mmlu-pro",
    "wikispeedia",
    "massive",
    "banking77",
    "banking77-k16",
    "clinc-unseen",
    "helpsteer",
    "civil",
    "injection",
    "wiki-hop",
)


def ece10(confidences, correct):
    """Ten equal-width bins, same as jevlike.eval (confidence == 1.0 is outside the last bin)."""
    n = len(confidences)
    if n == 0:
        return 0.0
    ece = 0.0
    for i in range(10):
        lo = i / 10
        hi = lo + 0.1
        selected = [(c, ok) for c, ok in zip(confidences, correct) if lo <= c < hi]
        if not selected:
            continue
        acc = sum(ok for _, ok in selected) / len(selected)
        conf = sum(c for c, _ in selected) / len(selected)
        ece += (len(selected) / n) * abs(acc - conf)
    return ece


def _api_questions(questions):
    out = {}
    for qid, q in questions.items():
        item = {k: v for k, v in q.items() if k not in {"target"} and not str(k).startswith("_")}
        if item.get("type") == "noul" and not isinstance(item.get("criteria"), dict):
            item.pop("criteria", None)
        out[qid] = item
    return out


def _choice_gold(q):
    keys = list(q["criteria"])
    target = q["target"]
    return keys[max(range(len(target)), key=lambda i: target[i])]


def score_question(qid, q, decision):
    kind = q["type"]
    rec = {"id": qid, "type": kind, "correct": False, "top3": False, "confidence": 0.0}
    if kind == "choice":
        gold = _choice_gold(q)
        choice = decision.choice(qid)
        probs = {str(k): float(v) for k, v in (decision.probabilities(qid) or {}).items()}
        ranked = sorted(probs, key=probs.get, reverse=True)
        rec.update(
            gold=gold,
            pred=choice,
            correct=choice == gold,
            top3=gold in ranked[:3],
            confidence=max(probs.values()) if probs else 0.0,
            n_options=len(q["criteria"]),
        )
        return rec
    if kind == "score":
        target = q["target"]
        gold_i = max(range(len(target)), key=lambda i: target[i])
        probs = decision.probabilities(qid) or {}
        # TypeSafe/GPT may key levels as 0..n or as legend strings.
        by_index = {}
        for k, v in probs.items():
            try:
                by_index[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
        if by_index:
            pred_i = max(by_index, key=by_index.get)
            conf = by_index[pred_i]
        else:
            pred_i = int(round(float(decision.score(qid) or 0)))
            conf = 0.0
        rec.update(gold=gold_i, pred=pred_i, correct=pred_i == gold_i, top3=abs(pred_i - gold_i) <= 1, confidence=conf)
        return rec
    gold = float(q["target"][1] if len(q["target"]) > 1 else q["target"][0])
    pred = float(decision.noul(qid) or 0.0)
    rec.update(
        gold=gold,
        pred=pred,
        correct=(pred >= 0.5) == (gold >= 0.5),
        top3=True,
        confidence=max(pred, 1 - pred),
        brier=(pred - gold) ** 2,
    )
    return rec


def score_rows(rows):
    labeled = [r for r in rows if r.get("error") is None]
    if not labeled:
        return {"n": 0, "top1": 0.0, "top3": 0.0, "ece": 0.0, "errors": len(rows), "by_type": {}}
    qs = [q for r in labeled for q in (r.get("questions") or [r])]
    choice = [q for q in qs if q.get("type", "choice") == "choice"]
    primary = choice or qs
    top1 = [bool(q.get("correct")) for q in primary]
    top3 = [bool(q.get("top3")) for q in primary]
    conf = [float(q.get("confidence") or 0.0) for q in primary]
    by_type = {}
    for q in qs:
        kind = q.get("type") or "choice"
        bucket = by_type.setdefault(kind, {"n": 0, "correct": 0, "brier": []})
        bucket["n"] += 1
        bucket["correct"] += int(bool(q.get("correct")))
        if "brier" in q:
            bucket["brier"].append(q["brier"])
    for kind, bucket in by_type.items():
        bucket["acc"] = bucket["correct"] / bucket["n"] if bucket["n"] else 0.0
        if bucket["brier"]:
            bucket["brier"] = sum(bucket["brier"]) / len(bucket["brier"])
        else:
            bucket.pop("brier", None)
    return {
        "n": len(primary),
        "top1": (sum(top1) / len(top1)) if top1 else 0.0,
        "top3": (sum(top3) / len(top3)) if top3 else 0.0,
        "ece": ece10(conf, top1) if conf else 0.0,
        "errors": len(rows) - len(labeled),
        "by_type": by_type,
    }


def _client(backend, model, env_file, timeout=90):
    if backend in {"typesafe", "local"}:
        from demos.common.systemone import SystemOne

        return SystemOne(backend=backend, model=model, env_file=env_file, timeout=timeout)
    if backend == "gpt":
        from demos.wikiracing.openai import OpenAISystemOne

        return OpenAISystemOne(model=model, env_file=env_file, timeout=timeout)
    raise ValueError("backend must be typesafe, local, or gpt")


def decide_example(client, example: BenchItem):
    decision = client.decide(example.state, _api_questions(example.questions))
    scored = [score_question(qid, q, decision) for qid, q in example.questions.items()]
    choice = next((q for q in scored if q["type"] == "choice"), scored[0])
    return {
        "id": example.example_id,
        "choice": choice.get("pred"),
        "label": choice.get("gold"),
        "correct": all(q["correct"] for q in scored) if scored else False,
        "top3": choice.get("top3", False),
        "confidence": choice.get("confidence", 0.0),
        "n_options": choice.get("n_options", 0),
        "latency_ms": decision.record.latency_ms,
        "input_tokens": decision.record.input_tokens,
        "output_tokens": decision.record.output_tokens,
        "model": decision.record.model,
        "questions": scored,
        "error": None,
    }


def run_examples(examples, client, *, input_usd_per_mtok=None, output_usd_per_mtok=0.0):
    rows = []
    started = time.perf_counter()
    for i, example in enumerate(examples, 1):
        row = None
        last_exc = None
        for _ in range(2):
            try:
                row = decide_example(client, example)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            row = {
                "id": example.example_id,
                "choice": None,
                "label": example.label,
                "correct": False,
                "top3": False,
                "confidence": 0.0,
                "n_options": len(example.criteria),
                "latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "model": getattr(client, "model", None),
                "questions": [],
                "error": f"{type(last_exc).__name__}: {last_exc}",
            }
        rows.append(row)
        print(
            f"{i}/{len(examples)} {row['id']} correct={row['correct']} "
            f"lat={row['latency_ms']:.0f} err={row['error']}",
            flush=True,
        )
    wall = time.perf_counter() - started

    class Rec:
        def __init__(self, row):
            self.latency_ms = row["latency_ms"]
            self.input_tokens = row["input_tokens"]
            self.output_tokens = row["output_tokens"]

    return {
        "accuracy": score_rows(rows),
        "cost": summarize(
            [Rec(r) for r in rows if r["error"] is None],
            wall,
            input_usd_per_mtok=input_usd_per_mtok,
            output_usd_per_mtok=output_usd_per_mtok,
        ),
        "rows": rows,
    }


def baseball_links():
    """Load the frozen Baseball outgoing menu, fetching once if needed."""
    from demos.wikiracing.wikipedia import WikiLink, extract_article_links, fetch_article

    dest = artifact_path("bench", "wiki-hop-baseball-links.json")
    if dest.exists():
        return [WikiLink(**item) for item in json.loads(dest.read_text())]
    _, html = fetch_article("Baseball")
    links = extract_article_links(html, "Baseball")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps([{"title": link.title, "href": link.href, "key": link.key} for link in links]) + "\n")
    return links


# Packed tokens of a wiki-hop Score batch under the Qwen3.5 tokenizer (Baseball links: 2,111 / 4,259 / 8,400), plus 10%.
SCORE_BATCH_TOKENS = {32: 2400, 64: 4700, 128: 9300}


def score_batch_for(max_tokens):
    """The largest Score batch (32, 64 or 128 questions) whose packed wiki-hop request fits `max_tokens`."""
    return max([b for b, tokens in SCORE_BATCH_TOKENS.items() if tokens <= max_tokens] or [32])


def local_score_batch(client):
    """`score_batch_for` the local server's `max_tokens` (GET /v1/models limits, the client's token); 32 when the
    limits cannot be read."""
    from janus.remote import load_env_key

    headers = {}
    for name in ("JANUS_SERVER_TOKEN", "TYPESAFE_API_KEY"):
        try:
            headers["Authorization"] = f"Bearer {load_env_key(name, client.env_file)}"
            break
        except Exception:
            continue
    try:
        with urlopen(Request(client.base_url + "/v1/models", headers=headers), timeout=10) as response:
            return score_batch_for(int(json.load(response)["limits"]["max_tokens"]))
    except Exception:
        return 32


def run_wiki_hop(client, backend, *, input_usd_per_mtok=None, output_usd_per_mtok=0.0, score_batch=None):
    """One frozen Baseball → Scientific American next-click with Score-then-Choice if N>255."""
    import demos.wikiracing.race as race
    from demos.wikiracing.wikipedia import title_key

    prev_batch = race.SCORE_BATCH
    if score_batch:
        race.SCORE_BATCH = score_batch
    elif backend == "gpt":
        race.SCORE_BATCH = 16
    elif backend == "local":
        race.SCORE_BATCH = local_score_batch(client)  # 128 when served with --max-tokens 9300 or more, else 64 or 32
    links = baseball_links()
    gold = "Scientific American"
    gold_key = title_key(gold)
    started = time.perf_counter()
    before = len(getattr(client, "records", []))
    try:
        picked, hop = race.take_hop(client, "Baseball", "Sun", ["Baseball"], links)
    finally:
        race.SCORE_BATCH = prev_batch
    wall = time.perf_counter() - started
    recs = list(getattr(client, "records", []))[before:]
    hit = title_key(picked.title) == gold_key
    top_titles = [t.lower() for t, _ in hop.top]
    row = {
        "id": "wiki-hop:baseball-scientific-american",
        "choice": picked.title,
        "label": gold,
        "correct": hit,
        "top3": gold.lower() in top_titles,
        "confidence": hop.top[0][1] if hop.top else 0.0,
        "n_options": len(links),
        "latency_ms": hop.latency_ms,
        "input_tokens": sum(r.input_tokens for r in recs),
        "output_tokens": sum(r.output_tokens for r in recs),
        "model": getattr(client, "model", None),
        "calls": len(recs),
        "questions": [
            {
                "id": "next",
                "type": "choice",
                "correct": hit,
                "top3": gold.lower() in top_titles,
                "confidence": hop.top[0][1] if hop.top else 0.0,
            }
        ],
        "stage": hop.stage,
        "error": None,
    }

    class Rec:
        def __init__(self, row):
            self.latency_ms = row["latency_ms"]
            self.input_tokens = row["input_tokens"]
            self.output_tokens = row["output_tokens"]

    return {
        "accuracy": score_rows([row]),
        "cost": summarize(
            recs or [Rec(row)], wall, input_usd_per_mtok=input_usd_per_mtok, output_usd_per_mtok=output_usd_per_mtok
        ),
        "rows": [row],
        "n_links": len(links),
        "stage": hop.stage,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--backend", default="typesafe", choices=("typesafe", "local", "gpt"))
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--control", default="none", choices=("none", "shuffled-context", "shuffled-options"))
    p.add_argument("--env-file", default="env.sh")
    p.add_argument("--output", default=None)
    p.add_argument("--score-batch", type=int, default=None, help="wiki-hop: Score questions per call (default: 16 for gpt, from the server limits for local, 128 for typesafe)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    env_with_keys(args.env_file)
    model = args.model or ("gpt-5.6-luna" if args.backend == "gpt" else "jev-latest")
    client = _client(args.backend, model, args.env_file, timeout=180 if args.task == "wiki-hop" else 90)
    in_rate, out_rate = rates_for(args.backend, model)
    if args.task == "wiki-hop":
        payload = run_wiki_hop(client, args.backend, input_usd_per_mtok=in_rate, output_usd_per_mtok=out_rate, score_batch=args.score_batch)
    else:
        examples = load_task(args.task, limit=args.limit, control=args.control)
        payload = run_examples(examples, client, input_usd_per_mtok=in_rate, output_usd_per_mtok=out_rate)
    payload["task"] = args.task
    payload["backend"] = args.backend
    payload["model"] = model
    payload["control"] = args.control
    tag = args.task if args.control == "none" else f"{args.task}-{args.control}"
    dest = Path(args.output) if args.output else artifact_path(
        "bench", f"{tag}-{args.backend}-{model.replace('/', '_')}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2) + "\n")
    dest.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in payload["rows"]))
    acc = payload["accuracy"]
    cost = payload["cost"]
    print(
        json.dumps(
            {
                "output": str(dest),
                "n": acc["n"],
                "top1": round(acc["top1"], 4),
                "top3": round(acc["top3"], 4),
                "ece": round(acc["ece"], 4),
                "by_type": acc.get("by_type"),
                "errors": acc["errors"],
                "latency_p50_ms": round(cost["latency_p50_ms"], 1),
                "estimated_usd": round(cost["estimated_usd"], 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
