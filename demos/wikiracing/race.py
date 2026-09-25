"""Greedy wiki race: Choice under 255, Score-then-Choice above."""
from __future__ import annotations

from dataclasses import dataclass, field

from demos.wikiracing.wikipedia import WikiLink, title_key

CHOICE_LIMIT = 255
SHORTLIST_K = 32
SCORE_BATCH = 128
SCORE_LEVELS = ["irrelevant", "weak", "related", "strong", "direct_path"]


def shortlist_from_scores(titles, scores, k=SHORTLIST_K):
    def score_for(i, title):
        if f"l{i}" in scores:
            return float(scores[f"l{i}"])
        return float(scores.get(title_key(title), scores.get(title, 0)))
    ranked = sorted(range(len(titles)), key=lambda i: score_for(i, titles[i]), reverse=True)
    return [titles[i] for i in ranked[:k]]


def choose_next(links, target_title, scores=None, choice=None):
    target = target_title.replace("_", " ").lower()
    for link in links:
        if link.title.lower() == target:
            return link, "direct"
    by_key = {link.key: link for link in links}
    if choice:
        if choice in by_key:
            return by_key[choice], "choice"
        lowered = {k.lower(): v for k, v in by_key.items()}
        if choice.lower() in lowered:
            return lowered[choice.lower()], "choice"
    if scores:
        order = shortlist_from_scores([link.title for link in links], scores, k=1)
        if order:
            wanted = order[0].lower()
            for link in links:
                if link.title.lower() == wanted:
                    return link, "score"
    if links:
        return links[0], "fallback"
    raise ValueError("no links")


def _state(current, target, path, n_links):
    return {
        "current": current,
        "target": target,
        "path": path,
        "n_links": n_links,
        "instructions": (
            f"Wikirace: start from {current} and reach {target} using only in-article Wikipedia links. "
            "Pick the link that most reduces remaining hops. Do not invent titles."
        ),
    }


def _choice_question(links, target):
    criteria = {}
    for link in links:
        criteria[link.key] = f"Go to {link.title} toward {target}"
    return {
        "next": {
            "type": "choice",
            "instructions": f"Which article should we open next to reach {target} soonest?",
            "criteria": criteria,
        }
    }


def _score_questions(links, target):
    questions = {}
    for i, link in enumerate(links):
        questions[link.key] = {
            "type": "score",
            "instructions": (
                f"How useful is the outgoing link {link.title} as the next hop toward {target}? "
                "Score independently of sibling links."
            ),
            "criteria": list(SCORE_LEVELS),
        }
        questions[link.key]["_index"] = i  # stripped before send
    return questions


def _strip_private(questions):
    return {qid: {k: v for k, v in q.items() if not k.startswith("_")} for qid, q in questions.items()}


@dataclass
class Hop:
    title: str
    stage: str
    latency_ms: float = 0
    input_tokens: int = 0
    top: list = field(default_factory=list)


def take_hop(client, current, target, path, links):
    """One greedy hop. Returns (WikiLink, Hop)."""
    picked, stage = choose_next(links, target)
    if stage == "direct":
        return picked, Hop(title=picked.title, stage=stage, top=[(picked.title, 1.0)])
    state = _state(current, target, path, len(links))
    if len(links) <= CHOICE_LIMIT:
        decision = client.decide(state, _choice_question(links, target))
        picked, stage = choose_next(links, target, choice=decision.choice("next"))
        probs = decision.probabilities("next")
        top = sorted(((k.replace("_", " "), p) for k, p in probs.items()), key=lambda kv: kv[1], reverse=True)[:5]
        return picked, Hop(title=picked.title, stage=stage, latency_ms=decision.record.latency_ms,
                           input_tokens=decision.record.input_tokens, top=top)
    scores = {}
    latency = 0.0
    tokens = 0
    for start in range(0, len(links), SCORE_BATCH):
        batch = links[start:start + SCORE_BATCH]
        questions = _strip_private(_score_questions(batch, target))
        decision = client.decide(state, questions)
        latency += decision.record.latency_ms
        tokens += decision.record.input_tokens
        for link in batch:
            scores[link.key] = decision.score(link.key) or 0.0
    short = shortlist_from_scores([link.title for link in links], scores, k=SHORTLIST_K)
    by_title = {link.title: link for link in links}
    short_links = [by_title[t] for t in short if t in by_title]
    decision = client.decide(state, _choice_question(short_links, target))
    latency += decision.record.latency_ms
    tokens += decision.record.input_tokens
    picked, stage = choose_next(short_links, target, choice=decision.choice("next"))
    probs = decision.probabilities("next")
    top = sorted(((k.replace("_", " "), p) for k, p in probs.items()), key=lambda kv: kv[1], reverse=True)[:5]
    return picked, Hop(title=picked.title, stage="score_then_choice", latency_ms=latency,
                       input_tokens=tokens, top=top)
