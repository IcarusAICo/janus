import json
import random

import pytest

from janus.schema import Request


def test_relative_menu_world_targets_are_correct_by_brute_force():
    from janus.synth.worlds import NAMES, relative_menu_world
    for seed in range(200):
        rng = random.Random(seed)
        world, request = relative_menu_world(rng, NAMES[:30], style="json")
        items = world["items"]
        by_name = {i["name"]: i for i in items}
        if world["kind"] == "second_cheapest":
            expected = sorted(items, key=lambda i: i["price"])[1]["name"]
        elif world["kind"] == "median_rating":
            assert len(items) % 2 == 1
            expected = sorted(items, key=lambda i: i["rating"])[len(items) // 2]["name"]
        elif world["kind"] == "closest_price_to_named":
            ref = by_name[world["reference"]]["price"]
            expected = min((i for i in items if i["name"] != world["reference"]), key=lambda i: abs(i["price"] - ref))["name"]
        else:
            ref = by_name[world["reference"]]["rating"]
            expected = min((i for i in items if i["rating"] > ref), key=lambda i: i["price"])["name"]
        assert world["gold"] == expected
        choice = request.questions[0]
        assert choice.kind == "choice" and choice.id == "rel:pick"
        gold_index = max(range(len(choice.target)), key=choice.target.__getitem__)
        assert json.loads(choice.options[gold_index].description)["name"] == expected
        assert request.questions[1].kind == "noul" and request.questions[1].id == "rel:is_gold"
        assert all(i["name"] not in request.state for i in items if i["name"] != world["reference"])
        assert len({i["price"] for i in items}) == len(items) and len({i["rating"] for i in items}) == len(items)


def test_relative_menu_prose_style_puts_attributes_in_option_text():
    from janus.synth.worlds import NAMES, relative_menu_world
    world, request = relative_menu_world(random.Random(3), NAMES[:30], style="prose")
    description = request.questions[0].options[0].description
    item = world["items"][0]
    assert item["name"] in description and str(item["price"]) in description and str(item["rating"]) in description


def test_relative_menu_is_deterministic():
    from janus.synth.worlds import NAMES, relative_menu_world
    a = relative_menu_world(random.Random(9), NAMES[:30], "json")
    b = relative_menu_world(random.Random(9), NAMES[:30], "json")
    assert a[0] == b[0] and a[1] == b[1]


def _base(users, minutes, data_loss):
    base = 0 if users == 1 else 1 if users < 10 else 2 if users < 100 else 3 if users < 1000 else 4
    if minutes > 240:
        base = max(base, 2)
    return 4 if data_loss else base


def test_ordinal_world_maps_levels_and_reversal_correctly():
    from janus.synth.worlds import ORD_LEVELS, ordinal_world
    maps = {5: dict(zip(range(5), range(5))), 4: {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}, 3: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}}
    for seed in range(150):
        rng = random.Random(seed)
        levels = rng.choice((3, 4, 5))
        reversed_order = seed % 2 == 0
        world, request = ordinal_world(rng, levels, reversed_order, "prose")
        expected = maps[levels][_base(world["users"], world["minutes"], world["data_loss"])]
        if reversed_order:
            expected = levels - 1 - expected
        score = request.questions[0]
        assert score.kind == "score" and score.id == "ord:severity"
        assert score.target[expected] == 1.
        descriptions = [o.description for o in score.options]
        assert descriptions == (ORD_LEVELS[levels][::-1] if reversed_order else ORD_LEVELS[levels])
        assert not any(c.isdigit() for d in descriptions for c in d)
        assert request.questions[1].kind == "noul"


def test_posterior_world_matches_bayes_rule_and_hides_the_table():
    from janus.synth.worlds import FEATURES, posterior_family, posterior_world
    family = posterior_family(2)
    assert len(family["names"]) == 3 and all(len(row) == 4 for row in family["theta"])
    for seed in range(100):
        world, request = posterior_world(random.Random(seed), family, 2, "json")
        likelihood = []
        for k in range(3):
            l = 1.
            for f, name in enumerate(FEATURES):
                p = family["theta"][k][f]
                l *= p if world["features"][name] else 1 - p
            likelihood.append(l)
        posterior = [l / sum(likelihood) for l in likelihood]
        assert world["posterior"] == pytest.approx(posterior)
        assert request.questions[0].target == pytest.approx(tuple(posterior))
        assert sum(request.questions[0].target) == pytest.approx(1.)
        assert "theta" not in request.state and "0.9" not in request.state
        assert request.group_id.startswith("post:2:")
    assert posterior_family(2) == family and posterior_family(3) != family


class _FakeUsage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _FakeResponse:
    def __init__(self, text):
        self.output_text, self.usage = text, _FakeUsage(120, 30)


class FakeOpenAI:
    """Stands in for openai.OpenAI; `reply` maps the user text to a JSON string."""

    def __init__(self, reply):
        self.calls = []
        parent = self

        class _Responses:
            def create(self, **kwargs):
                parent.calls.append(kwargs)
                return _FakeResponse(reply(kwargs["input"]))
        self.responses = _Responses()


def test_structured_completer_caches_accounts_cost_and_never_stores_secrets(tmp_path, monkeypatch):
    from janus.synth.openai_client import StructuredCompleter
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-secret")
    fake = FakeOpenAI(lambda user: json.dumps({"answer": user.upper()}))
    completer = StructuredCompleter(cache_dir=tmp_path / "cache", client=fake)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
    obj, usage, cached = completer.complete("Echo.", "hello", "echo", schema)
    assert obj == {"answer": "HELLO"} and usage == {"input_tokens": 120, "output_tokens": 30} and cached is False
    again, _, cached = completer.complete("Echo.", "hello", "echo", schema)
    assert again == obj and cached is True and len(fake.calls) == 1
    body = fake.calls[0]
    assert body["model"] == "gpt-5.6-luna" and body["reasoning"] == {"effort": "none"} and body["store"] is False
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["strict"] is True
    assert completer.cost_usd() == pytest.approx(120 * .2 / 1e6 + 30 * 1.2 / 1e6)
    assert all("sk-fake-secret" not in p.read_text() for p in (tmp_path / "cache").rglob("*.json"))


def test_load_env_key_reads_named_variable_from_env_file(tmp_path, monkeypatch):
    from janus.remote import load_env_key
    for name in ("OPENAI_API_KEY", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)  # a sourced env.sh would otherwise win over the file
    env = tmp_path / "env.sh"
    env.write_text("export TYPESAFE_API_KEY=ts-1\nexport OPENAI_API_KEY=sk-2\n")
    assert load_env_key("OPENAI_API_KEY", env) == "sk-2"
    assert load_env_key("TYPESAFE_API_KEY", env) == "ts-1"


def test_paraphrase_is_accepted_only_when_extraction_round_trips(tmp_path, monkeypatch):
    from janus.synth.openai_client import StructuredCompleter
    from janus.synth.render import Paraphraser, evidence
    from janus.synth.worlds import ordinal_world
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    world, request = ordinal_world(random.Random(4), 4, False, "prose")
    truth = evidence(world)

    def faithful(user):
        if user.startswith("REWRITE:"):
            return json.dumps({"text": "Rewritten: " + user})
        return json.dumps(truth)

    def unfaithful(user):
        if user.startswith("REWRITE:"):
            return json.dumps({"text": "Rewritten: " + user})
        wrong = dict(truth)
        wrong["users_affected"] = truth["users_affected"] + 1
        return json.dumps(wrong)

    good = Paraphraser(StructuredCompleter(cache_dir=tmp_path / "a", client=FakeOpenAI(faithful)))
    assert good.paraphrase(world, request.state).startswith("Rewritten:")
    assert good.accepted == 1 and good.rejected == 0
    bad = Paraphraser(StructuredCompleter(cache_dir=tmp_path / "b", client=FakeOpenAI(unfaithful)))
    assert bad.paraphrase(world, request.state) is None
    assert bad.accepted == 0 and bad.rejected == 1


def test_evidence_schemas_are_strict_objects():
    from janus.synth.render import EVIDENCE_SCHEMAS
    for family, schema in EVIDENCE_SCHEMAS.items():
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_prepare_phase1_offline_writes_disjoint_splits_with_holdouts(tmp_path):
    from janus.data import assert_disjoint, load_requests
    from janus.synth.build import prepare_phase1
    from janus.synth.worlds import TEST_NAMES, TRAIN_NAMES
    study = tmp_path / "study"
    study.mkdir()
    from janus.data import synthetic_requests, write_jsonl
    write_jsonl(study / "train.jsonl", [r.to_dict() for r in synthetic_requests(40, seed=1)])
    write_jsonl(study / "dev.jsonl", [r.to_dict() for r in synthetic_requests(10, seed=2)])
    (study / "manifest.json").write_text('{"dataset": "synthetic plumbing only"}')
    output = tmp_path / "phase1"
    manifest = prepare_phase1(output, per_family=30, dev=5, calibration=5, test=10, llm=False, study_dir=study)
    splits = {name: load_requests(output / f"{name}.jsonl") for name in ("train", "dev", "calibration", "test", "test_post_unseen")}
    assert_disjoint(splits)
    assert manifest["dataset"] == "JEV_PHASE1_V1" and manifest["tiers"]["T0"] > 0 and manifest["tiers"]["T1"] > 0
    assert manifest["counts"]["train"]["rel"] == 30 and manifest["counts"]["train"]["study"] == 30
    def item_name(description):
        try:
            return json.loads(description)["name"]
        except ValueError:
            return description.split(":")[0]

    for r in splits["train"]:
        if r.group_id.startswith("rel:"):
            assert all(item_name(o.description) in TRAIN_NAMES for o in r.questions[0].options)
        if r.group_id.startswith("ord:"):
            assert len(r.questions[0].options) in (3, 4)
        if r.group_id.startswith("post:"):
            assert int(r.group_id.split(":")[1]) < 8
    for r in splits["test"]:
        if r.group_id.startswith("rel:"):
            assert all(item_name(o.description) in TEST_NAMES for o in r.questions[0].options)
    assert any(len(r.questions[0].options) == 5 for r in splits["test"] if r.group_id.startswith("ord:"))
    assert all(int(r.group_id.split(":")[1]) >= 8 for r in splits["test_post_unseen"])
    assert manifest["paraphrase"] == {"accepted": 0, "rejected": 0, "requested": 0}
    assert manifest["cost_usd"] == 0.
