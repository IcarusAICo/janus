"""Luna paraphrase bank for question instructions, with a back-check that each paraphrase asks the same question.

The bank is `{instruction: [accepted paraphrases]}` over every distinct instruction in the source training files.
Acceptance is a second Luna call answering `{"same_question": bool}`; that is a model judgement, not verification,
and the manifest records how many paraphrases it rejected."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

from ..data import write_json

DEFAULT_SOURCES = ("data/phase1-v1/train.jsonl", "data/public-v1/train.jsonl")
PARAPHRASE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["paraphrases"],
                     "properties": {"paraphrases": {"type": "array", "items": {"type": "string"}}}}
CHECK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["same_question"],
                "properties": {"same_question": {"type": "boolean"}}}
PARAPHRASE_INSTRUCTIONS = (
    "Rewrite the given question instruction in {count} different ways. Each rewrite must ask exactly the same question about the "
    "same state and the same answer options: keep every name, label, number, quoted phrase, negation, and scope word; change only "
    "the wording and sentence structure; do not add hints, examples, or answer options. Return JSON with one field, paraphrases, "
    "a list of {count} strings.")
CHECK_INSTRUCTIONS = (
    "You compare two question instructions that will be shown with the same state and the same answer options. Return JSON with one "
    "field, same_question: true only if a careful reader would give the same answer to both for every possible state; false if the "
    "rewrite changes, drops, or adds any condition, name, label, number, negation, or scope.")


def distinct_instructions(paths):
    out = set()
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    for question in json.loads(line)["questions"].values():
                        out.add(question["instructions"])
    return sorted(out)


def paraphrase_instruction(instruction, completer, count=5):
    """Return (accepted, rejected_counts). Blank rewrites and exact repeats are dropped before the back-check."""
    generated, _, _ = completer.complete(PARAPHRASE_INSTRUCTIONS.format(count=count),
                                         json.dumps({"instruction": instruction}, ensure_ascii=False), "paraphrases", PARAPHRASE_SCHEMA)
    accepted, rejected = [], {"blank": 0, "duplicate": 0, "not_same_question": 0}
    seen = {" ".join(instruction.split()).lower()}
    for text in list(generated["paraphrases"])[:count]:
        key = " ".join(str(text).split()).lower()
        if not key:
            rejected["blank"] += 1
            continue
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        check, _, _ = completer.complete(CHECK_INSTRUCTIONS, json.dumps({"original": instruction, "rewrite": text}, ensure_ascii=False),
                                         "same_question", CHECK_SCHEMA)
        if check["same_question"] is True:
            accepted.append(str(text).strip())
        else:
            rejected["not_same_question"] += 1
    return accepted, rejected


def _log(message):
    print(time.strftime("%H:%M:%S"), message, file=sys.stderr, flush=True)


def build_paraphrase_bank(output, sources=DEFAULT_SOURCES, completer=None, count=5, cost_abort_usd=10., workers=8):
    """Write `output` as {instruction: [paraphrases]} and `<stem>.manifest.json` with counts and cost.

    A supplied completer is used sequentially (the plain StructuredCompleter is not thread-safe); without one, a
    PooledCompleter (gpt-5.6-luna, effort none) is driven by a thread pool of `workers`."""
    output = Path(output)
    manifest_path = output.with_name(output.stem + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output)
    instructions = distinct_instructions(sources)
    pooled = completer is None
    if pooled:
        from .generate import PooledCompleter
        completer = PooledCompleter(model="gpt-5.6-luna", effort="none", cache_dir=".cache/synth/openai")
    bank, rejected, aborted, started = {}, {"blank": 0, "duplicate": 0, "not_same_question": 0}, False, time.monotonic()

    def finish(done):
        manifest = {"dataset": "JEV_PARAPHRASES_V1", "sources": [str(s) for s in sources], "model": completer.model, "effort": completer.effort,
                    "count_per_instruction": count, "instructions": len(instructions), "instructions_done": done,
                    "requested": done * count, "accepted": sum(len(v) for v in bank.values()), "rejected": dict(rejected),
                    "rejected_total": sum(rejected.values()), "instructions_with_no_paraphrase": sum(1 for v in bank.values() if not v),
                    "acceptance_rate": sum(len(v) for v in bank.values()) / max(1, done * count),
                    "cost_usd": completer.cost_usd(), "cost_usd_all_calls": completer.cost_usd(include_cached=True),
                    "calls": dict(completer.calls), "aborted": aborted, "wall_minutes": (time.monotonic() - started) / 60,
                    "acceptance_note": "Acceptance is a Luna same_question judgement, not verification."}
        write_json(output, bank)
        write_json(manifest_path, manifest)
        return manifest

    def consume(instruction, result):
        accepted, counts = result
        bank[instruction] = accepted
        for key in rejected:
            rejected[key] += counts[key]

    if not pooled:
        for i, instruction in enumerate(instructions, 1):
            consume(instruction, paraphrase_instruction(instruction, completer, count))
            if completer.cost_usd() > cost_abort_usd:
                aborted = True
                finish(i)
                raise RuntimeError(f"Paraphrase bank cost exceeded {cost_abort_usd} USD; aborted after {i} instructions")
        return finish(len(instructions))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(paraphrase_instruction, instruction, completer, count) for instruction in instructions]
        for i, (instruction, future) in enumerate(zip(instructions, futures), 1):
            consume(instruction, future.result())
            if i % 25 == 0 or i == len(instructions):
                _log(f"paraphrases: {i}/{len(instructions)} instructions, accepted {sum(len(v) for v in bank.values())}, cost so far ${completer.cost_usd():.3f}")
            if completer.cost_usd() > cost_abort_usd:
                pool.shutdown(wait=False, cancel_futures=True)
                aborted = True
                finish(i)
                raise RuntimeError(f"Paraphrase bank cost exceeded {cost_abort_usd} USD; aborted after {i} instructions")
    return finish(len(instructions))
