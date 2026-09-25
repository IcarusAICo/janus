"""Cookbook patterns as compositions of Choice, Score and Noul over one evaluator.

Every helper builds one ordinary `Request`, asks an evaluator for per-question
distributions, and runs plain code on the answers. The helpers say nothing about
whether a given model judges well; that is empirical (docs/patterns.md).
"""

from dataclasses import dataclass

from .schema import Question, Request

NONE = "none"


@dataclass(frozen=True)
class Answer:
    question: Question
    probabilities: dict  # option key (or "false"/"true" for Noul) -> probability

    @property
    def kind(self):
        return self.question.kind

    @property
    def choice(self):
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def score(self):
        """Expected level index, as `janus.schema.decode` reports it."""
        return sum(i * self.probabilities[o.key] for i, o in enumerate(self.question.options))

    @property
    def noul(self):
        return self.probabilities["true"]

    @property
    def confidence(self):
        # ponytail: the public docs leave the confidence formula unspecified; max probability is the
        # simplest concentration measure. Swap for margin or 1-entropy here if a threshold needs it.
        return max(self.probabilities.values())


def answers(request, distributions):
    """Pair each question with its distribution (one sequence per question, in request order)."""
    if len(distributions) != len(request.questions):
        raise ValueError("Wrong number of question distributions")
    return [Answer(q, {o.key: float(p) for o, p in zip(q.options, d)}) for q, d in zip(request.questions, distributions)]


class LocalEvaluator:
    """Loads a checkpoint once and evaluates like `janus.evaluation.predict`."""

    def __init__(self, checkpoint, calibration=None, device="cpu"):
        from .evaluation import read_calibration
        from .training import load_checkpoint
        self.temperature = read_calibration(calibration, checkpoint)["temperature"]
        self.model, _ = load_checkpoint(checkpoint, device)

    def evaluate(self, request):
        from .packing import pack_request
        m = self.model
        packed = pack_request(request, m.tokenizer, m.packing_mode, m.config.max_tokens, **m.packing_kwargs)
        return answers(request, [(z.float() / self.temperature).softmax(-1).cpu().tolist() for z in m(packed)])


class RemoteEvaluator:
    """Wraps `janus.remote.RemoteClient`; `cache_dir` reuses its durable response cache."""

    def __init__(self, model=None, env_file="env.sh", cache_dir=None, client=None):
        from .remote import DEFAULT_MODEL, RemoteClient
        self.client = client or RemoteClient(model or DEFAULT_MODEL, env_file)
        self.cache_dir = cache_dir

    def evaluate(self, request):
        return answers(request, self.client.predict(request, self.cache_dir).probabilities)


class FakeEvaluator:
    """Scripted distributions by question id (key -> probability); unscripted questions are uniform."""

    def __init__(self, scripted=None):
        self.scripted = scripted or {}
        self.requests = []

    def evaluate(self, request):
        self.requests.append(Request.from_dict(request.to_dict()))  # every helper must build a parseable Request
        rows = []
        for q in request.questions:
            given = self.scripted.get(q.id)
            rows.append([1 / len(q.options)] * len(q.options) if given is None else [given.get(o.key, 0.) for o in q.options])
        return answers(request, rows)


def _request(state, questions):
    return Request.from_dict({"state": state, "questions": questions})


def _choice(instructions, criteria, none_description=None):
    if none_description is not None:
        if NONE in criteria:
            raise ValueError(f"Option key {NONE!r} is reserved for the none option")
        criteria = {**criteria, NONE: none_description}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _one(evaluator, state, qid, question):
    return evaluator.evaluate(_request(state, {qid: question}))[0]


def speculative_fanout(evaluator, state, questions):
    """Ask every question (raw schema dicts by id) in one call; the caller ignores the irrelevant answers."""
    return {a.question.id: a for a in evaluator.evaluate(_request(state, questions))}


def confidence_gated_route(evaluator, state, question, threshold, fallback):
    """Return the answer's choice when confidence >= threshold, else `fallback(answer)`; the gate is recorded."""
    answer = _one(evaluator, state, "gated", question)
    accepted = answer.confidence >= threshold
    return {"accepted": accepted, "confidence": answer.confidence, "answer": answer,
            "result": answer.choice if accepted else fallback(answer)}


def composite_score(evaluator, state, rubrics, weights):
    """Weighted mean of per-rubric expected levels, each normalised to [0, 1] by its top level index.

    `rubrics` maps name -> {"instructions": ..., "criteria": [ordered level descriptions]}.
    """
    if set(weights) != set(rubrics) or not weights or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be nonnegative, keyed exactly by rubric, and not all zero")
    by_name = speculative_fanout(evaluator, state, {name: {"type": "score", **r} for name, r in rubrics.items()})
    total = sum(weights.values())
    score = sum(weights[n] * a.score / (len(a.question.options) - 1) for n, a in by_name.items()) / total
    return {"score": score, "answers": by_name}


def intent_route(evaluator, state, intents, none_description, instructions="What is the primary intent of this message?"):
    """Choice over intents plus a none option; returns the intent (None when none wins) with its mass."""
    answer = _one(evaluator, state, "intent", _choice(instructions, intents, none_description))
    key = answer.choice
    return {"intent": None if key == NONE else key, "probability": answer.probabilities[key], "answer": answer}


def select_function(evaluator, state, tools, none_description="No listed function applies.",
                    instructions="Which function should be called to satisfy this request?"):
    """Choice over tool names (name -> description) plus none; returns the name or None."""
    answer = _one(evaluator, state, "function", _choice(instructions, tools, none_description))
    key = answer.choice
    return {"function": None if key == NONE else key, "confidence": answer.confidence, "answer": answer}


def select_arguments(evaluator, state, tool, enums, none_description="Not stated; use the default."):
    """One Choice per enum argument (arg -> {value: description}), each with none; unstated arguments are omitted.

    `confidence` is the least certain judgement, as one wrong argument invalidates the whole call.
    """
    questions = {arg: _choice(f"For the call to {tool}, which value should the argument {arg!r} take?", values, none_description)
                 for arg, values in enums.items()}
    by_arg = speculative_fanout(evaluator, state, questions) if questions else {}
    arguments = {arg: a.choice for arg, a in by_arg.items() if a.choice != NONE}
    return {"function": tool, "arguments": arguments,
            "confidence": min((a.confidence for a in by_arg.values()), default=1.), "answers": by_arg}


def extract_candidate(evaluator, state, field, candidates, none_description="None of these is the requested value."):
    """Choice over pre-parsed candidate spans plus none; the value returned is one of the spans, verbatim, or None."""
    spans = list(dict.fromkeys(candidates))
    if not spans:
        return {"value": None, "confidence": 1., "answer": None}
    criteria = {f"c{i}": span for i, span in enumerate(spans)}
    answer = _one(evaluator, state, field, _choice(f"Which of these is the {field}?", criteria, none_description))
    key = answer.choice
    return {"value": None if key == NONE else criteria[key], "confidence": answer.confidence, "answer": answer}
