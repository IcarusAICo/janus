"""Thin System One client used by every demo.

Talks to TypeSafe or a local `/v1/systemone` server through `typesafe_sdk`.
A `transport=` argument lets tests inject answers without HTTP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
import time

from janus.remote import load_env_key

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
LOCAL_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "jev-latest"


@dataclass
class CallRecord:
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str
    question_ids: tuple[str, ...]


@dataclass
class Decision:
    answers: dict
    record: CallRecord
    raw: object = None

    def choice(self, question_id):
        return self.answers[question_id].get("choice")

    def noul(self, question_id):
        return self.answers[question_id].get("noul")

    def score(self, question_id):
        return self.answers[question_id].get("score")

    def probabilities(self, question_id):
        return self.answers[question_id].get("probabilities") or {}


class FakeTransport:
    """Scripted System One answers for tests. Same method name as TypeSafeClient."""

    def __init__(self, answers, *, model="jev-latest", input_tokens=1, latency_ms=1, base_url_seen=None):
        self._answers = answers
        self.model = model
        self.input_tokens = input_tokens
        self.latency_ms = latency_ms
        self.base_url = base_url_seen
        self.calls = []

    def system_one(self, state, questions, **kwargs):
        self.calls.append({"state": state, "questions": questions, **kwargs})
        answers = {}
        for qid, raw in self._answers.items():
            answers[qid] = SimpleNamespace(**raw)
        return SimpleNamespace(
            model=self.model,
            usage=SimpleNamespace(input_tokens=self.input_tokens, output_tokens=0),
            answers=answers,
        )


def questions_from_dicts(questions):
    """Turn demo dict questions into `typesafe_sdk` Choice/Score/Noul objects."""
    import typesafe_sdk as sdk
    built = {}
    for qid, raw in questions.items():
        kind = raw["type"]
        if kind == "choice":
            built[qid] = sdk.Choice(instructions=raw["instructions"], criteria=raw["criteria"])
        elif kind == "score":
            built[qid] = sdk.Score(instructions=raw["instructions"], criteria=raw["criteria"])
        elif kind == "noul":
            kwargs = {"instructions": raw["instructions"]}
            if raw.get("criteria"):
                kwargs["criteria"] = raw["criteria"]
            built[qid] = sdk.Noul(**kwargs)
        else:
            raise ValueError(f"Unknown question type: {kind}")
    return built


def _as_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(value)


def normalize_answers(raw_answers):
    out = {}
    for qid, answer in _as_dict(raw_answers).items():
        if isinstance(answer, dict):
            data = dict(answer)
        else:
            data = {
                "type": getattr(answer, "type", None),
                "choice": getattr(answer, "choice", None),
                "noul": getattr(answer, "noul", None),
                "score": getattr(answer, "score", None),
                "probabilities": getattr(answer, "probabilities", None),
                "confidence": getattr(answer, "confidence", None),
                "legend": getattr(answer, "legend", None),
            }
        kind = data.get("type")
        out[qid] = {
            "type": kind,
            "choice": data.get("choice") if kind == "choice" else None,
            "noul": data.get("noul") if kind == "noul" else None,
            "score": data.get("score") if kind == "score" else None,
            "probabilities": _as_dict(data.get("probabilities")),
            "confidence": data.get("confidence"),
            "legend": _as_dict(data.get("legend")) or None,
        }
    return out


class SystemOne:
    """One `decide(state, questions)` call. Backend is `typesafe`, `local`, or a fake transport."""

    def __init__(self, backend="typesafe", base_url=None, model=None, transport=None,
                 env_file="env.sh", api_key=None, timeout=30):
        if backend not in {"typesafe", "local"}:
            raise ValueError("backend must be typesafe or local")
        self.backend = backend
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        elif backend == "local":
            self.base_url = LOCAL_BASE_URL
        else:
            self.base_url = TYPESAFE_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.env_file = env_file
        self.timeout = timeout
        self.records = []
        self.transport = transport
        if transport is not None:
            transport.base_url = self.base_url
            self._client = transport
            return
        key_name = "JANUS_SERVER_TOKEN" if backend == "local" else "TYPESAFE_API_KEY"
        try:
            key = api_key if api_key is not None else load_env_key(key_name, env_file)
        except Exception:
            if backend == "local" and api_key is None:
                key = load_env_key("TYPESAFE_API_KEY", env_file)
            else:
                raise
        import typesafe_sdk as sdk
        self._client = sdk.TypeSafeClient(
            api_key=key, base_url=self.base_url, model=self.model,
            retry=sdk.RetryPolicy(max_retries=0), timeout=timeout,
        )

    def decide(self, state, questions):
        typed = questions if questions and not isinstance(next(iter(questions.values())), dict) else questions_from_dicts(questions)
        started = time.perf_counter()
        raw = self._client.system_one(state=state, questions=typed, model=self.model)
        latency_ms = getattr(self._client, "latency_ms", None)
        if latency_ms is None:
            latency_ms = (time.perf_counter() - started) * 1000
        usage = getattr(raw, "usage", None)
        record = CallRecord(
            latency_ms=float(latency_ms),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            model=str(getattr(raw, "model", self.model)),
            question_ids=tuple(questions),
        )
        self.records.append(record)
        return Decision(answers=normalize_answers(raw.answers), record=record, raw=raw)
