from dataclasses import replace
import json
import random

import pytest
import torch

from janus.data import split_banking, banking_request, synthetic_requests, write_jsonl, load_requests, assert_disjoint, shuffled_options
from janus.training import TrainConfig, token_groups, train, load_checkpoint
from janus.model import DecisionModel, ModelConfig
from janus.packing import pack_request
from janus.metrics import distribution_loss


def test_split_excludes_unseen_intents_and_deduplicates_before_augmentation():
    rows = [{"text": f"message {i} for {c}", "category": c} for c in range(20) for i in range(30)]
    rows = [{**r, "category": f"intent_{r['category']}"} for r in rows]
    test = [{"text": f"test {i} for {c}", "category": f"intent_{c}"} for c in range(20) for i in range(2)]
    # Same normalized state in official train/test must belong only to test.
    test.append(dict(rows[0]))
    rows.append({"text": " MESSAGE 0 FOR 0 ", "category": "intent_0"})
    # Conflicting annotations for an identical state must disappear entirely.
    rows += [{"text": "ambiguous", "category": "intent_1"},
             {"text": "ambiguous", "category": "intent_2"}]
    splits, manifest = split_banking(rows, test, [f"intent_{c}" for c in range(20)], heldout_count=4)
    assert len(manifest["unseen_intents"]) == 4
    assert all(r["category"] not in manifest["unseen_intents"]
               for split in ("train", "dev", "calibration") for r in splits[split])
    assert all(r["category"] in manifest["unseen_intents"] for r in splits["test_unseen"])
    sets = [{r["group_id"] for r in values} for values in splits.values()]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            assert not a & b
    assert not any(r["text"] == "ambiguous" for values in splits.values() for r in values)
    assert split_banking(rows, test, [f"intent_{c}" for c in range(20)], heldout_count=4) == (splits, manifest)


def test_banking_targets_and_permutation_stay_with_option_meanings():
    row = {"text": "My transfer failed", "category": "failed_transfer", "group_id": "id"}
    pool = ["failed_transfer", "card_arrival", "lost_card", "cash_withdrawal"]
    request = banking_request(row, pool, seed=7, cardinalities=(4,))
    for version in (request, shuffled_options(request, 44)):
        q = version.questions[0]
        assert len(q.options) == 4
        winner = q.options[q.target.index(1.)].description
        assert winner in {"failed transfer", "None of these intents describes the message."}
    q1, q2 = request.questions[0], shuffled_options(request, 44).questions[0]
    assert {o.key: p for o, p in zip(q1.options, q1.target)} == {o.key: p for o, p in zip(q2.options, q2.target)}


def test_disjoint_checks_use_state_content_even_if_ids_differ():
    a = synthetic_requests(4, seed=1)
    b = [replace(a[0], group_id="different-id")]
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint({"train": a, "dev": b})


def test_tiny_model_overfits_and_checkpoint_roundtrips(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    request = synthetic_requests(1, seed=17)[0]
    config = ModelConfig(backbone="tiny", hidden_size=32, layers=2, head_rank=16,
                         adaptation="full", dtype="float32")
    model = DecisionModel(config)
    packed = pack_request(request, model.tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    for _ in range(60):
        optimizer.zero_grad()
        loss = distribution_loss(model(packed), [q.target for q in request.questions])
        loss.backward()
        optimizer.step()
    assert loss.item() < .02

    # Exercise the public train/select/save/reload pipeline with separate states.
    paths = {}
    for name, seed in (("train", 17), ("dev", 23), ("calibration", 41)):
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], [r.to_dict() for r in synthetic_requests(3, seed=seed)])
    output = tmp_path / "run"
    result = train(paths["train"], paths["dev"], output,
                   TrainConfig(model=config, epochs=1, accumulation=2, max_steps=2, device="cpu"))
    loaded, metadata = load_checkpoint(output / "best.pt", device="cpu")
    assert metadata["step"] == result["best_step"]
    assert metadata["model"]["backbone"] == "tiny"
    assert metadata["training_group_ids"]
    loaded.eval()
    z = loaded(packed)
    again, _ = load_checkpoint(output / "best.pt", device="cpu")
    again.eval()
    for x, y in zip(z, again(packed)):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_init_checkpoint_continues_from_saved_weights(tmp_path):
    torch.set_num_threads(2)
    config = ModelConfig(backbone="tiny", hidden_size=32, layers=1, head_rank=8, adaptation="full", dtype="float32")
    paths = {}
    for name, seed in (("train", 17), ("dev", 23)):
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], [r.to_dict() for r in synthetic_requests(3, seed=seed)])
    first = train(paths["train"], paths["dev"], tmp_path / "first",
                  TrainConfig(model=config, epochs=1, accumulation=1, max_steps=3, device="cpu"))
    saved = json.loads((tmp_path / "first" / "history.json").read_text())
    best = next(h for h in saved if h["step"] == first["best_step"])
    second = train(paths["train"], paths["dev"], tmp_path / "second",
                   TrainConfig(model=config, epochs=1, accumulation=1, max_steps=1, device="cpu",
                               init_checkpoint=str(tmp_path / "first" / "best.pt")))
    # Step 0 of the continued run scores dev exactly where the loaded checkpoint left it; a fresh init does not.
    assert second["initial_dev_nll"] == pytest.approx(best["dev_nll"], abs=1e-6)
    assert first["initial_dev_nll"] != pytest.approx(best["dev_nll"], abs=1e-6)
    with pytest.raises((ValueError, RuntimeError)):  # a different head shape cannot continue from it
        train(paths["train"], paths["dev"], tmp_path / "third",
              TrainConfig(model=replace(config, head_rank=4), epochs=1, accumulation=1, max_steps=1, device="cpu",
                          init_checkpoint=str(tmp_path / "first" / "best.pt")))


def test_epoch_end_is_evaluated_and_last_weights_are_saved(tmp_path):
    torch.set_num_threads(2)
    config = ModelConfig(backbone="tiny", hidden_size=32, layers=1, head_rank=8, adaptation="full", dtype="float32")
    paths = {}
    for name, seed in (("train", 17), ("dev", 23)):
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], [r.to_dict() for r in synthetic_requests(3, seed=seed)])
    result = train(paths["train"], paths["dev"], tmp_path / "run",
                   TrainConfig(model=config, epochs=1, accumulation=2, eval_every=100, device="cpu", max_seconds=3600))
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert result["steps"] == 2 and history[-1]["step"] == 2  # the last partial batch is evaluated, not only step % eval_every
    assert not result["budget_limited"] and history[-1]["elapsed_seconds"] < 60  # the clock is not clobbered by the batch loop
    _, metadata = load_checkpoint(tmp_path / "run" / "last.pt", device="cpu")
    assert metadata["step"] == 2


def test_epoch_regeneration_changes_menus_and_retains_state_groups():
    from janus.data import training_epoch
    pool = [f"intent_{i}" for i in range(10)]
    sources = [{"text": f"message {i}", "category": pool[i % 10], "group_id": f"g{i}"} for i in range(32)]
    initial = [banking_request(row, pool) for row in sources]
    first = training_epoch(initial, 17, 0, sources, pool)
    second = training_epoch(initial, 17, 1, sources, pool)
    assert first == training_epoch(initial, 17, 0, sources, pool)
    assert [r.group_id for r in first] == [r.group_id for r in second]
    assert any(len(a.questions[0].options) != len(b.questions[0].options) for a, b in zip(first, second))
    assert any(a.questions[1].instructions != b.questions[1].instructions for a, b in zip(first, second))


@pytest.mark.parametrize("adaptation", ["frozen", "lora"])
def test_checkpoint_matches_presave_predictions(tmp_path, adaptation):
    from janus.training import checkpoint
    torch.manual_seed(13)
    model = DecisionModel(ModelConfig(backbone="tiny", adaptation=adaptation, hidden_size=32, head_rank=16)).eval()
    packed = pack_request(synthetic_requests(1)[0], model.tokenizer)
    before = model(packed)
    path = tmp_path / "model.pt"
    checkpoint(model, path, {})
    restored, _ = load_checkpoint(path)
    for a, b in zip(before, restored(packed)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_training_limit_is_balanced_and_selection_is_seed_independent():
    from janus.training import balanced_subset
    a = [replace(r, group_id=f'a:{r.group_id}') for r in synthetic_requests(12, seed=1)]
    b = [replace(r, group_id=f'b:{r.group_id}') for r in synthetic_requests(8, seed=2)]
    selected = balanced_subset(a + b, 10, seed=17)
    assert len(selected) == 10
    assert sum(r.group_id.startswith('a:') for r in selected) == 5
    assert selected == balanced_subset(a + b, 10, seed=17)


def test_identical_backbone_and_head_initialization_across_adaptation_controls():
    configs = [ModelConfig(backbone='tiny', adaptation=a, hidden_size=32, head_rank=16)
               for a in ('frozen', 'lora')]
    torch.manual_seed(17)
    frozen = DecisionModel(configs[0])
    torch.manual_seed(17)
    lora = DecisionModel(configs[1])
    for a, b in zip(frozen.head.parameters(), lora.head.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_training_arms_change_the_loss_and_record_counts(tmp_path):
    torch.set_num_threads(2)
    paths = {}
    for name, seed in (("train", 17), ("dev", 23)):
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], [r.to_dict() for r in synthetic_requests(4, seed=seed)])
    config = ModelConfig(backbone="tiny", hidden_size=32, layers=1, head_rank=8, adaptation="full")
    base = train(paths["train"], paths["dev"], tmp_path / "base",
                 TrainConfig(model=config, epochs=1, accumulation=2, max_steps=1, device="cpu"))
    cons = train(paths["train"], paths["dev"], tmp_path / "cons",
                 TrainConfig(model=config, epochs=1, accumulation=2, max_steps=1, device="cpu", consistency_weight=.5))
    assert cons["consistency_forwards"] == 2 and base["consistency_forwards"] == 0
    assert base["complement_questions_added"] == 0
    smooth = train(paths["train"], paths["dev"], tmp_path / "smooth",
                   TrainConfig(model=config, epochs=1, accumulation=2, max_steps=1, device="cpu",
                               label_smoothing=.1, objective="spherical"))
    assert smooth["steps"] == 1
    # Synthetic Nouls have no negation template, so the complement arm adds nothing here.
    comp = train(paths["train"], paths["dev"], tmp_path / "comp",
                 TrainConfig(model=config, epochs=1, accumulation=2, max_steps=1, device="cpu", complement_augment=True))
    assert comp["complement_questions_added"] == 0
    with pytest.raises(ValueError):
        TrainConfig(model=config, none_rate=1.5).validate()
    with pytest.raises(ValueError):
        TrainConfig(model=config, label_smoothing=1.).validate()
    with pytest.raises(ValueError):
        TrainConfig(model=config, consistency_weight=-.1).validate()
    assert "none_rate" in json.loads((tmp_path / "base" / "config.json").read_text())["config"]


def test_complement_augmentation_adds_negated_intent_nouls(tmp_path):
    from janus.blocked_probes import negate_noul
    row = {"text": "My transfer failed", "category": "failed_transfer", "group_id": "id"}
    pool = ["failed_transfer", "card_arrival", "lost_card", "cash_withdrawal"]
    noul = banking_request(row, pool, seed=7).questions[1]
    negated = negate_noul(noul)
    assert negated is not None and negated.target == tuple(reversed(noul.target))
    assert negated.instructions.startswith("Is it false that")


def test_none_rate_changes_omission_frequency():
    pool = [f"intent_{i}" for i in range(10)]
    rows = [{"text": f"t{i}", "category": pool[i % 10], "group_id": f"g{i}"} for i in range(400)]

    def omitted_fraction(rate):
        count = 0
        for row in rows:
            q = banking_request(row, pool, seed=3, omit=rate).questions[0]
            count += any(o.description.startswith("None of these") and t == 1. for o, t in zip(q.options, q.target))
        return count / len(rows)
    assert omitted_fraction(0.) == 0. and .3 < omitted_fraction(.4) < .5
    assert omitted_fraction(.2) == omitted_fraction(.2)
    default = [banking_request(row, pool, seed=3) for row in rows]
    assert default == [banking_request(row, pool, seed=3, omit=.2) for row in rows]
    with pytest.raises(ValueError):
        banking_request(rows[0], pool, omit=1.5)


def test_token_groups_stay_under_the_budget_and_the_row_cap():
    """The greedy split is consecutive and complete, never exceeds `max_rows` rows, and never exceeds the budget
    except for a single row that is over it on its own; budget 0 reproduces the fixed row-count grouping."""
    rng = random.Random(17)
    for _ in range(300):
        counts = [rng.choice([1, 50, 300, 900]) for _ in range(rng.randrange(12))]
        rows, budget = rng.randrange(1, 5), rng.choice([0, 100, 500, 1000])
        groups = list(token_groups(counts, rows, budget))
        assert [i for lo, hi in groups for i in range(lo, hi)] == list(range(len(counts)))
        assert all(hi - lo <= rows for lo, hi in groups)
        assert all(sum(counts[lo:hi]) <= budget or hi - lo == 1 for lo, hi in groups if budget)
        if not budget:
            assert groups == [(lo, min(lo + rows, len(counts))) for lo in range(0, len(counts), rows)]
