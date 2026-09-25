import copy
import math

import pytest
import torch

from janus.schema import Request, decode
from janus.packing import ByteTokenizer, pack_request
from janus.model import DecisionModel, ModelConfig


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(7)


def example():
    return {"state": "A red card.", "questions": {
        "color": {"type": "choice", "instructions": "Color?",
                  "criteria": {"r": "red", "b": "blue"}, "target": [1, 0]},
        "red": {"type": "noul", "instructions": "Is it red?", "target": [0, 1]},
        "intensity": {"type": "score", "instructions": "Intensity?",
                      "criteria": ["low", "medium", "high"], "target": [0, 0, 1]},
    }}


def tiny(mode="listwise", adaptation="full"):
    return DecisionModel(ModelConfig(backbone="tiny", mode=mode, adaptation=adaptation,
                                     hidden_size=32, layers=2, head_rank=16))


def test_decode_uses_declared_labels_and_ordinal_expectation():
    request = Request.from_dict(example())
    result = decode(request, [torch.tensor([.2, .8]), torch.tensor([.3, .7]),
                              torch.tensor([.05, .30, .65])])
    assert result["color"]["choice"] == "b"
    assert result["red"] == pytest.approx(.7)
    assert result["intensity"]["score"] == pytest.approx(1.6)
    assert result["intensity"]["probabilities"] == pytest.approx({"0": .05, "1": .3, "2": .65})
    with pytest.raises(ValueError):
        decode(request, [torch.tensor([1., 1.])] * 3)


@pytest.mark.parametrize("target", [[.2, .2], [-.1, 1.1], [float("nan"), 0], [1], [0, 2]])
def test_invalid_targets_rejected(target):
    raw = example()
    raw["questions"]["color"]["target"] = target
    with pytest.raises(ValueError):
        Request.from_dict(raw)


def test_pack_does_not_encode_question_ids_and_resets_positions():
    raw = example()
    renamed = copy.deepcopy(raw)
    renamed["questions"] = {f"secret-{i}": q for i, q in enumerate(raw["questions"].values())}
    a = pack_request(Request.from_dict(raw), ByteTokenizer())
    b = pack_request(Request.from_dict(renamed), ByteTokenizer())
    assert torch.equal(a.input_ids, b.input_ids)
    for branch in a.branches:
        assert a.position_ids[0, branch.start] == a.state_length
        assert a.allowed[branch.start, :a.state_length].all()
        assert not a.allowed[:a.state_length, branch.start:branch.end].any()
        for other in a.branches:
            if other is not branch:
                assert not a.allowed[branch.start:branch.end, other.start:other.end].any()
    with pytest.raises(ValueError, match="token"):
        pack_request(Request.from_dict(raw), ByteTokenizer(), max_tokens=10)


def test_packed_equals_separate_logits_and_gradients():
    request = Request.from_dict(example())
    model = tiny().eval()
    packed = pack_request(request, ByteTokenizer())
    logits = model(packed)
    loss = sum(x.square().sum() for x in logits)
    loss.backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    separate = [model(pack_request(Request(request.state, (q,)), ByteTokenizer()))[0]
                for q in request.questions]
    for a, b in zip(logits, separate):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    sum(x.square().sum() for x in separate).backward()
    for k, p in model.named_parameters():
        if k in gradients:
            torch.testing.assert_close(p.grad, gradients[k], atol=3e-5, rtol=3e-4)


def test_sibling_mutation_and_reordering_cannot_change_target():
    model = tiny().eval()
    original = Request.from_dict(example())
    logits = model(pack_request(original, ByteTokenizer()))[0]
    changed = example()
    changed["questions"]["red"]["instructions"] = "Secret blue! " * 12
    changed["questions"] = dict(reversed(list(changed["questions"].items())))
    actual = model(pack_request(Request.from_dict(changed), ByteTokenizer()))[-1]
    torch.testing.assert_close(logits, actual, atol=2e-6, rtol=2e-5)


def test_independent_scores_ignore_other_options_but_listwise_can_use_them():
    a = example()
    a["questions"] = {"color": a["questions"]["color"]}
    b = copy.deepcopy(a)
    b["questions"]["color"]["criteria"]["g"] = "green: the card is really blue"
    b["questions"]["color"]["target"] = [1, 0, 0]
    for mode in ("independent", "listwise"):
        model = tiny(mode).eval()
        x = model(pack_request(Request.from_dict(a), ByteTokenizer(), mode))[0]
        y = model(pack_request(Request.from_dict(b), ByteTokenizer(), mode))[0]
        if mode == "independent":
            torch.testing.assert_close(x, y[:2], atol=2e-6, rtol=2e-5)
        else:
            assert abs((x[0] - x[1] - y[0] + y[1]).item()) > 1e-7


def test_frozen_and_lora_only_train_intended_weights():
    request = pack_request(Request.from_dict(example()), ByteTokenizer())
    for adaptation in ("frozen", "lora"):
        model = tiny(adaptation=adaptation)
        sum(z.square().sum() for z in model(request)).backward()
        names = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
        assert all("lora_" in n for n in names)
        assert bool(names) == (adaptation == "lora")
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.head.parameters())
