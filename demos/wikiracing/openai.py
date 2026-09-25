"""OpenAI GPT-5.6 as a System One stand-in (structured JSON, reasoning none)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time

from demos.common.systemone import CallRecord, Decision, normalize_answers


OPENAI_URL = "https://api.openai.com/v1/responses"


def _schema_for(questions):
    properties = {}
    required = []
    for qid, q in questions.items():
        kind = q["type"] if isinstance(q, dict) else q.type
        if kind == "choice":
            keys = list(q["criteria"] if isinstance(q, dict) else q.criteria)
            properties[qid] = {
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "enum": keys},
                    "probabilities": {
                        "type": "object",
                        "properties": {k: {"type": "number"} for k in keys},
                        "required": keys,
                        "additionalProperties": False,
                    },
                },
                "required": ["choice", "probabilities"],
                "additionalProperties": False,
            }
        elif kind == "score":
            levels = q["criteria"] if isinstance(q, dict) else q.criteria
            idx = [str(i) for i in range(len(levels))]
            properties[qid] = {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "probabilities": {
                        "type": "object",
                        "properties": {i: {"type": "number"} for i in idx},
                        "required": idx,
                        "additionalProperties": False,
                    },
                },
                "required": ["score", "probabilities"],
                "additionalProperties": False,
            }
        else:
            properties[qid] = {
                "type": "object",
                "properties": {"noul": {"type": "number"}},
                "required": ["noul"],
                "additionalProperties": False,
            }
        required.append(qid)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class OpenAISystemOne:
    def __init__(self, model, env_file="env.sh", api_key=None, timeout=60):
        from janus.remote import load_env_key
        self.model = model
        self.timeout = timeout
        self.api_key = api_key or load_env_key("OPENAI_API_KEY", env_file)
        self.records = []

    def decide(self, state, questions):
        import httpx
        raw_questions = questions
        if questions and not isinstance(next(iter(questions.values())), dict):
            raw_questions = {qid: q.model_dump() for qid, q in questions.items()}
        schema = _schema_for(raw_questions)
        prompt = (
            "You are a System One model. Given shared state, answer every question with "
            "calibrated probabilities. Probabilities for a choice or score must sum to 1. "
            f"State:\n{state if isinstance(state, str) else json.dumps(state)}\n\n"
            f"Questions:\n{json.dumps(raw_questions, ensure_ascii=False)}"
        )
        body = {
            "model": self.model,
            "input": prompt,
            "reasoning": {"effort": "none"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "system_one",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                OPENAI_URL, json=body,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
        latency_ms = (time.perf_counter() - started) * 1000
        text = _output_text(payload)
        parsed = json.loads(text)
        answers = {}
        for qid, q in raw_questions.items():
            item = parsed[qid]
            kind = q["type"]
            if kind == "choice":
                answers[qid] = {
                    "type": "choice", "choice": item["choice"],
                    "probabilities": item["probabilities"],
                    "confidence": max(item["probabilities"].values()),
                }
            elif kind == "score":
                answers[qid] = {
                    "type": "score", "score": item["score"],
                    "probabilities": item["probabilities"],
                    "confidence": max(item["probabilities"].values()),
                }
            else:
                answers[qid] = {"type": "noul", "noul": item["noul"]}
        usage = payload.get("usage") or {}
        record = CallRecord(
            latency_ms=latency_ms,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            model=self.model,
            question_ids=tuple(raw_questions),
        )
        self.records.append(record)
        return Decision(answers=normalize_answers(answers), record=record, raw=payload)


def _output_text(payload):
    if payload.get("output_text"):
        return payload["output_text"]
    chunks = []
    for item in payload.get("output") or []:
        for part in item.get("content") or []:
            if part.get("type") in {"output_text", "text"} and part.get("text"):
                chunks.append(part["text"])
    if not chunks:
        raise RuntimeError("OpenAI response had no text")
    return "".join(chunks)
