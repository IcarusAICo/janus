"""Phase 4 robustness set: every perturbation keeps the code-defined gold, twins pair row for row, and the reports."""

import json
import random
import re

import pytest

from janus.data import write_jsonl
from janus.multilingual_report import by_locale, injection_keys, massive_report, observations, paired, robustness_report
from janus.packing import ByteTokenizer
from janus.schema import Request
from janus.synth.families import SENDER_POOL, SURNAMES, evidence_verdict, same_person
from janus.synth.robustness import (FILLER_TOKENS, INJECTIONS, TRANSFORMS, VARIANTS, WORLDS, draw, generate_variant,
                                  irrelevant_text, prepare_robustness, rel_gold)
from janus.synth.worlds import NAMES, relative_menu_world


def pairs(variant, count=30, seed=0):
    """(family, world, twin, variant request, extra) for `count` draws per family of the variant."""
    out = []
    for family in WORLDS[variant]:
        rng, n = random.Random(f"{seed}:{variant}:{family}"), 0
        while n < count:
            style = ("prose", "json", "table")[n % 3] if variant != "format_noise" else "prose"
            world, request = draw(family, rng, style if family in ("retrieval", "evidence", "record_match") else style.replace("table", "prose"))
            result = TRANSFORMS[variant](world, request, rng, ByteTokenizer())
            if result is None:
                continue
            out.append((family, world, *result))
            n += 1
    return out


def _item(description):
    """A rel option line (JSON or prose) back to the item dict."""
    if description.startswith("{"):
        return json.loads(description)
    m = re.fullmatch(r"(\w+): \$(\d+), rated ([\d.]+), (\d+) km away", description)
    return {"name": m[1], "price": int(m[2]), "rating": float(m[3]), "distance_km": int(m[4])}


def gold_keys(request):
    return {q.id: [o.key for o, t in zip(q.options, q.target) if t == max(q.target)] for q in request.questions}


def test_irrelevant_text_is_bounded_and_names_nothing_from_the_worlds():
    rng = random.Random(1)
    for _ in range(20):
        target = rng.randint(*FILLER_TOKENS)
        text, count = irrelevant_text(rng, ByteTokenizer(), target)
        assert target <= count <= target + 200 and 3 <= len(text.split("\n\n")[0].split(". ")) <= 6
        words = set(re.findall(r"[A-Za-z]+", text))
        assert not words & set(NAMES) and not words & set(SENDER_POOL) and not words & set(SURNAMES)


def test_rel_gold_agrees_with_the_world_generator():
    for seed in range(200):
        world, request = relative_menu_world(random.Random(seed), NAMES, "prose")
        gold = next(i for i, item in enumerate(world["items"]) if item["name"] == world["gold"])
        assert rel_gold(world["kind"], world["items"], world["reference"]) == gold
        assert request.questions[0].target[gold] == 1.


def test_prefix_and_suffix_keep_the_state_questions_and_gold():
    for variant, position in (("irrelevant_prefix", "endswith"), ("irrelevant_suffix", "startswith")):
        for family, world, twin, noisy, extra in pairs(variant, count=4):
            assert getattr(noisy.state, position)(twin.state) and len(noisy.state) > len(twin.state)
            assert FILLER_TOKENS[0] <= extra["filler_tokens"] <= FILLER_TOKENS[1]
            assert noisy.questions == twin.questions and gold_keys(noisy) == gold_keys(twin)


def test_injection_names_a_wrong_option_and_changes_nothing_else():
    templates = set()
    for family, world, twin, noisy, extra in pairs("instruction_injection", count=12):
        inj = extra["injection"]
        assert inj["text"] in noisy.state or json.loads(noisy.state)["note"] == inj["text"]  # JSON states escape the quotes
        assert noisy.questions == twin.questions
        question = next(q for q in twin.questions if q.id == inj["question"])
        assert inj["key"] not in gold_keys(twin)[question.id] and inj["key"] in {o.key for o in question.options}
        assert INJECTIONS[inj["template"]].format(key=inj["key"] if question.kind == "choice" else f'"{inj["description"]}"',
                                                  description=inj["description"]) == inj["text"]
        if twin.state.startswith("{"):
            assert json.loads(noisy.state)["note"] == inj["text"] and {**json.loads(noisy.state), "note": None} == {**json.loads(twin.state), "note": None}
        else:
            assert noisy.state.replace(inj["text"] + "\n", "", 1).replace("\n" + inj["text"], "", 1) == twin.state
        templates.add(inj["template"])
    assert templates == set(range(len(INJECTIONS)))


def test_distractors_are_near_duplicates_of_the_gold_that_do_not_qualify():
    for family, world, twin, noisy, extra in pairs("distractor_options", count=25):
        (tq,), (nq,) = twin.questions, noisy.questions  # the Noul is dropped on both sides
        assert nq.id == tq.id and len(nq.options) == len(tq.options) + 2
        gold_text = tq.options[tq.target.index(1.)].description
        assert nq.options[nq.target.index(1.)].description == gold_text
        new = [o.description for i, o in enumerate(nq.options) if i in extra["distractors"]]
        assert len(new) == 2 and all(d != gold_text for d in new)
        if family == "rel":
            assert noisy.state == twin.state
            items = [_item(o.description) for o in nq.options]
            gold_item = next(i for i in world["items"] if i["name"] == world["gold"])
            assert [i["name"] for i in items].count(world["gold"]) == 3
            assert rel_gold(world["kind"], items, world["reference"]) == nq.target.index(1.)
            for d in new:  # same entity, exactly one field changed
                assert [k for k in gold_item if _item(d)[k] != gold_item[k]] in (["price"], ["rating"])
        else:
            entity, probe = world["entity"], world["probe"]
            if noisy.state.startswith("{"):  # a JSON state re-serialises its candidates canonically
                records = [c["record"] for c in json.loads(noisy.state)["candidates"]]
                assert all(json.loads(d) in records for d in new) and len(records) == len(nq.options) - 1
            else:
                assert all(d in noisy.state for d in new) and noisy.state.count("\n[") == len(nq.options) - 1
            assert nq.options[-1].key == "none"
            assert same_person(probe, world["views"][world["gold"]], entity, entity)


def test_missing_evidence_moves_the_gold_to_none_only_in_the_variant():
    for family, world, twin, noisy, extra in pairs("missing_evidence", count=25):
        (tq,), (nq,) = twin.questions, noisy.questions
        assert tq.options[-1].key == nq.options[-1].key == "none" and [o.key for o in tq.options] == [o.key for o in nq.options]
        assert gold_keys(noisy)[nq.id] == ["none"] and gold_keys(twin)[tq.id] != ["none"]
        removed = extra["removed"]
        if family == "rel":
            gold_item = next(i for i in world["items"] if i["name"] == world["gold"])
            option = next(o for o in nq.options if o.key == removed["option"]).description
            assert str(gold_item[removed["field"]]) not in option and gold_item["name"] in option
            assert noisy.state == twin.state
            assert removed["field"] == {"second_cheapest": "price", "median_rating": "rating"}.get(world["kind"], removed["field"])
        else:
            value, entity = world["value"], world["entity"]
            assert re.search(rf"(?<![\w:]){re.escape(value)}(?![\w:])", twin.state)
            answering = [o for o in nq.options[:-1] if entity in o.description and re.search(rf"(?<![\w:]){re.escape(value)}(?![\w:])", o.description)]
            assert not answering and entity in next(o for o in nq.options if o.key == removed["option"]).description
            assert noisy.state != twin.state and all(o.description in noisy.state for o in nq.options[:-1])


def test_format_noise_keeps_every_digit_and_the_options():
    for family, world, twin, noisy, extra in pairs("format_noise", count=15):
        assert noisy.questions == twin.questions
        assert re.sub(r"\D", "", noisy.state) == re.sub(r"\D", "", twin.state)
        assert noisy.state != twin.state and extra["noise"]
        if family == "evidence":  # the recomputed verdict does not depend on the rendering
            assert evidence_verdict(world["claim"], world["records"]) == world["label"]


def test_prepare_writes_paired_files_manifest_and_avoids_existing_rows(tmp_path):
    out = tmp_path / "rob"
    manifest = prepare_robustness(out, ByteTokenizer(), per_variant=6, exclude=None, max_tokens=10 ** 6)
    assert set(manifest["counts"]) == set(VARIANTS) and all(c["pairs"] == 6 for c in manifest["counts"].values())
    assert set(manifest["files"]) == {f"{p}{v}.jsonl" for v in VARIANTS for p in ("", "clean_")}
    for variant in VARIANTS:
        clean = [json.loads(l) for l in (out / f"clean_{variant}.jsonl").read_text().splitlines()]
        noisy = [json.loads(l) for l in (out / f"{variant}.jsonl").read_text().splitlines()]
        assert [r["group_id"] for r in clean] == [r["pair_id"] for r in noisy] == [r["pair_id"] for r in clean]
        assert all(r["group_id"] == f"{r['pair_id']}#{variant}" and r["variant"] == variant for r in noisy)
        assert all(list(r["questions"]) == list(c["questions"]) for r, c in zip(noisy, clean))
        assert all(Request.from_dict(r).questions for r in noisy)
    assert manifest["details"]["instruction_injection"]["templates"] == list(INJECTIONS)
    # An excluded directory that already holds a twin makes the generator redraw it.
    exclude = tmp_path / "data"
    write_jsonl(exclude / "old.jsonl", [json.loads((out / "clean_format_noise.jsonl").read_text().splitlines()[0])])
    again = prepare_robustness(tmp_path / "rob2", ByteTokenizer(), per_variant=6, exclude=exclude, max_tokens=10 ** 6)
    first = [json.loads(l)["group_id"] for l in (tmp_path / "rob2" / "clean_format_noise.jsonl").read_text().splitlines()]
    assert manifest["files"]["clean_format_noise.jsonl"]["sha256"] != again["files"]["clean_format_noise.jsonl"]["sha256"]
    assert json.loads((out / "clean_format_noise.jsonl").read_text().splitlines()[0])["group_id"] not in first
    assert again["excluded"]["files"] == 1
    with pytest.raises(ValueError, match="over 100"):
        prepare_robustness(tmp_path / "tight", ByteTokenizer(), per_variant=6, exclude=None, max_tokens=100)


def test_pairs_are_disjoint_from_each_other_under_one_seen_set():
    seen = set()
    for variant in VARIANTS:
        generate_variant(variant, 6, random.Random(variant), seen, ByteTokenizer(), 10 ** 6)
    assert len(seen) >= 6 * len(VARIANTS) * 2


# ---------------------------------------------------------------------------------------------------------------------
# reports

def _record(locale, p, y, kind="choice", n=0):
    return {"group_id": f"massive:{locale}:{n:08x}", "question_id": f"massive:{kind}", "kind": kind, "keys": [f"o{i}" for i in range(len(p))],
            "probabilities": p, "target": y, "nll": 0.5, "calibrated_nll": 0.25}


def test_multilingual_tables_from_both_row_formats(tmp_path):
    rows = [_record("en-US", [.9, .1], [1., 0.], n=1), _record("en-US", [.8, .2], [0., 1.], n=2),
            _record("de-DE", [.6, .4], [1., 0.], "noul", n=3), {"group_id": "civil:abc", "question_id": "x", "kind": "choice",
                                                                 "keys": ["a"], "probabilities": [1.], "target": [1.], "nll": 0.}]
    write_jsonl(tmp_path / "predictions.jsonl", rows)
    obs = observations(tmp_path / "predictions.jsonl")
    assert [o["locale"] for o in obs] == ["en-US", "en-US", "de-DE"] and all(o["nll"] == .25 for o in obs)
    table = by_locale(obs)
    assert table[("en-US", "choice")] == {"n": 2, "accuracy": .5, "nll": .25, "ece": pytest.approx(.45)}
    assert table[("all", "all")]["n"] == 3 and ("fr-FR", "all") not in table
    bench = [{"id": "massive:fr-FR:1", "error": None, "questions": [{"id": "massive:intent", "type": "choice", "correct": True, "confidence": .7},
                                                                    {"id": "massive:matches", "type": "noul", "correct": False, "confidence": .9}]},
             {"id": "massive:fr-FR:2", "error": "timeout", "questions": []}]
    write_jsonl(tmp_path / "bench.jsonl", bench)
    table = by_locale(observations(tmp_path / "bench.jsonl"))
    assert table[("fr-FR", "all")] == {"n": 2, "accuracy": .5, "nll": None, "ece": pytest.approx(.6)}
    text = massive_report([tmp_path / "predictions.jsonl", tmp_path / "bench.jsonl"])
    assert "| en-US | choice | 2 | 0.500 | 0.250 | 0.450 |" in text and "| fr-FR | all | 2 | 0.500 | n/a | 0.600 |" in text


def test_paired_robustness_table(tmp_path):
    def rows(variant, picks):
        return [{"group_id": f"rel:{i}" + (f"#{variant}" if variant else ""), "question_id": "rel:pick", "keys": ["o0", "o1"],
                 "probabilities": [.8, .2] if pick == "o0" else [.3, .7], "target": [1., 0.], "nll": .1 if pick == "o0" else 1.2}
                for i, pick in enumerate(picks)]
    run = tmp_path / "run"
    write_jsonl(run / "clean_instruction_injection" / "predictions.jsonl", rows(None, ["o0", "o0", "o1", "o0"]))
    write_jsonl(run / "instruction_injection" / "predictions.jsonl", rows("instruction_injection", ["o0", "o1", "o0", "o1"]))
    data = tmp_path / "data"
    write_jsonl(data / "instruction_injection.jsonl", [{"pair_id": f"rel:{i}", "injection": {"question": "rel:pick", "key": "o1"}} for i in range(4)])
    s = paired(run / "clean_instruction_injection" / "predictions.jsonl", run / "instruction_injection" / "predictions.jsonl", injection_keys(data))
    assert s == {"n": 4, "clean_accuracy": .75, "variant_accuracy": .5, "clean_nll": pytest.approx(.375), "variant_nll": pytest.approx(.65),
                 "broken": .5, "fixed": .25, "followed_injection": .5}
    text = robustness_report(run, data)
    assert "| instruction_injection | 4 | 0.750 | 0.500 | -0.250 | 0.375 | 0.650 | 0.500 | 0.250 | 0.500 |" in text
    assert "| irrelevant_prefix | missing |" in text
    write_jsonl(tmp_path / "empty.jsonl", [])
    with pytest.raises(ValueError, match="No paired rows"):
        paired(run / "clean_instruction_injection" / "predictions.jsonl", tmp_path / "empty.jsonl")
