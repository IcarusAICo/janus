"""Typed input, outcome targets, and deterministic numerical output."""

import base64
from dataclasses import dataclass
import json
import math
from pathlib import Path


def text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def validate_distribution(values, size):
    values = tuple(float(x) for x in values)
    if (len(values) != size or any(not math.isfinite(x) or x < 0 for x in values)
            or not math.isclose(sum(values), 1, abs_tol=1e-5)):
        raise ValueError(f"Expected {size} finite nonnegative probabilities summing to one")
    return values


@dataclass(frozen=True)
class Option:
    key: str
    description: str


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    instructions: str
    options: tuple[Option, ...]
    target: tuple[float, ...] | None = None

    @classmethod
    def from_dict(cls, name, raw):
        kind = raw["type"]
        instructions = text(raw["instructions"])
        if not instructions.strip():
            raise ValueError("Question instructions must be nonempty")
        criteria = raw.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError("Choice needs 1–255 keyed options")
            if any(not isinstance(k, str) or not k for k in criteria):
                raise ValueError("Option keys must be nonempty strings")
            options = tuple(Option(k, text(v)) for k, v in criteria.items())
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score needs 2–10 ordered descriptions")
            options = tuple(Option(str(i), text(v)) for i, v in enumerate(criteria))
        elif kind == "noul":
            if criteria is not None and (not isinstance(criteria, dict)
                                        or set(criteria) - {"true", "false"}):
                raise ValueError("Noul criteria may contain only true and false")
            criteria = criteria or {}
            options = (Option("false", text(criteria.get("false", "No, the proposition is false."))),
                       Option("true", text(criteria.get("true", "Yes, the proposition is true."))))
        else:
            raise ValueError(f"Unknown question type: {kind}")
        target = raw.get("target")
        if target is not None:
            target = validate_distribution(target, len(options))
        return cls(str(name), kind, instructions, options, target)

    def to_dict(self):
        raw = {"type": self.kind, "instructions": self.instructions}
        raw["criteria"] = ([o.description for o in self.options] if self.kind == "score"
                           else {o.key: o.description for o in self.options})
        if self.target is not None:
            raw["target"] = list(self.target)
        return raw


@dataclass(frozen=True)
class Image:
    """One encoded image file (the bytes of a PNG/JPEG/...); `path` is the jsonl-relative reference it came from."""
    data: bytes
    media_type: str = "image/png"
    path: str | None = None

    @classmethod
    def from_dict(cls, raw, base_dir=None):
        if not isinstance(raw, dict) or ("path" in raw) == ("base64" in raw):
            raise ValueError("An image reference is {\"path\": ...} or {\"base64\": ..., \"media_type\": ...}")
        if "path" in raw:
            path = str(raw["path"])
            return cls(Path(base_dir or ".", path).read_bytes(), str(raw.get("media_type", "image/png")), path)
        try:
            data = base64.b64decode(raw["base64"], validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Image base64 is malformed: {error}") from error
        media_type = str(raw.get("media_type", "image/png"))
        if not media_type.startswith("image/"):
            raise ValueError("Image media_type must be image/*")
        return cls(data, media_type)

    def to_dict(self):
        if self.path is not None:
            return {"path": self.path}
        return {"base64": base64.b64encode(self.data).decode(), "media_type": self.media_type}


class ImageState(str):
    """State text plus images. ponytail: a str subclass so every text-only consumer (the "State:" prefix, hashing,
    JSON signatures, regex kinds) keeps working unchanged; the images hang off `.images`. `[image:i]` markers in
    the text place image i; without markers every image precedes the text, in order."""
    __slots__ = ("images",)

    def __new__(cls, text, images):
        self = super().__new__(cls, text)
        self.images = tuple(images)
        return self

    def __reduce__(self):  # pickling (torch.save of a request) keeps the images
        return (ImageState, (str(self), self.images))


def state_from_dict(raw, base_dir=None):
    """A state: a string (or JSON-encoded value) as today, or {"text": ..., "images": [refs]}."""
    if isinstance(raw, dict) and "images" in raw:
        if not isinstance(raw["images"], list) or not raw["images"]:
            raise ValueError("An image state needs a nonempty image list")
        return ImageState(text(raw.get("text", "")), (Image.from_dict(r, base_dir) for r in raw["images"]))
    return text(raw)


def state_to_dict(state):
    if isinstance(state, ImageState):
        return {"text": str(state), "images": [i.to_dict() for i in state.images]}
    return state


@dataclass(frozen=True)
class Request:
    state: str
    questions: tuple[Question, ...]
    group_id: str = ""

    @classmethod
    def from_dict(cls, raw, base_dir=None):
        """`base_dir` resolves image `path` references (the directory of the jsonl file they came from)."""
        questions = raw.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("A request needs a nonempty question map")
        return cls(state_from_dict(raw["state"], base_dir),
                   tuple(Question.from_dict(k, v) for k, v in questions.items()),
                   str(raw.get("group_id", "")))

    def to_dict(self):
        return {"state": state_to_dict(self.state), "questions": {q.id: q.to_dict() for q in self.questions},
                "group_id": self.group_id}


def decode(request, probabilities):
    if len(probabilities) != len(request.questions):
        raise ValueError("Wrong number of question distributions")
    result = {}
    for question, values in zip(request.questions, probabilities):
        p = validate_distribution(values, len(question.options))
        if question.kind == "noul":
            result[question.id] = p[1]
            continue
        output = {"probabilities": {o.key: v for o, v in zip(question.options, p)}}
        if question.kind == "choice":
            output["choice"] = question.options[max(range(len(p)), key=p.__getitem__)].key
        else:
            output["score"] = sum(i * v for i, v in enumerate(p))
            output["legend"] = {o.key: o.description for o in question.options}
        result[question.id] = output
    return result
