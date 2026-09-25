import copy

import pytest
import torch

from janus.packing import ByteTokenizer, pack_request
from janus.schema import Request


def example():
    return {"state": "A red card.", "questions": {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"r": "red", "b": "blue", "g": "green"}, "target": [1, 0, 0]},
        "red": {"type": "noul", "instructions": "Is it red?", "target": [0, 1]},
        "intensity": {"type": "score", "instructions": "Intensity?", "criteria": ["low", "medium", "high"], "target": [0, 0, 1]}}}


def test_tree_layout_segments_positions_and_mask():
    packed = pack_request(Request.from_dict(example()), ByteTokenizer(), mode="tree")
    assert len(packed.branches) == 3
    allowed = packed.allowed
    for branch in packed.branches:
        block_len = branch.end - branch.start
        assert packed.position_ids[0, branch.start] == packed.state_length
        assert len(branch.leaves) == (1 if branch.kind == "noul" else len(branch.option_indices))
        assert len(branch.option_positions) == len(branch.leaves)
        for (start, end), decision in zip(branch.leaves, branch.option_positions):
            assert decision == end - 1
            assert packed.position_ids[0, start] == packed.state_length + block_len
            assert allowed[start:end, :packed.state_length].all()
            assert allowed[start:end, branch.start:branch.end].all()
            assert torch.equal(allowed[start:end, start:end], torch.ones(end - start, end - start).tril().bool())
            for other_start, other_end in branch.leaves:
                if other_start != start:
                    assert not allowed[start:end, other_start:other_end].any()
            for other in packed.branches:
                if other is not branch:
                    assert not allowed[start:end, other.start:other.end].any()
        assert not allowed[branch.start:branch.end, branch.leaves[0][0]:branch.leaves[-1][1]].any()
    block = packed.branches[0]
    assert packed.parents[packed.segment_ids[block.leaves[0][0]]] == packed.segment_ids[block.start]


def test_tree_block_contents_by_kind():
    tokenizer = ByteTokenizer()
    packed = pack_request(Request.from_dict(example()), tokenizer, mode="tree")
    text = bytes(t - 1 for t in packed.input_ids[0].tolist()).decode()
    assert "Options:\n[r] red\n[b] blue\n[g] green\n" in text
    assert "Candidate: [r]\nDecision:\n" in text
    assert "Level: low\nDecision:\n" in text and "Question (score):\nIntensity?\nDecision" not in text
    assert "False means: No, the proposition is false.\nTrue means: Yes, the proposition is true.\n" in text
    score_block = text[text.index("Question (score):"):text.index("Level: low")]
    assert "low" not in score_block and "medium" not in score_block


from janus.model import DecisionModel, ModelConfig


def tiny(mode, hidden_size=64, **extra):
    return DecisionModel(ModelConfig(backbone="tiny", mode=mode, adaptation="full", hidden_size=hidden_size, layers=2,
                                     head_rank=16, **extra)).eval()


def run(model, raw):
    return [z.detach() for z in model(pack_request(Request.from_dict(raw), ByteTokenizer(), model.packing_mode))]


def test_tree_outputs_have_the_right_shapes_and_noul_first_logit_is_zero():
    torch.manual_seed(0)
    logits = run(tiny("tree"), example())
    assert [len(z) for z in logits] == [3, 2, 3]
    assert logits[1][0].item() == 0.


def test_tree_choice_leaf_reads_the_block_but_score_levels_are_independent():
    torch.manual_seed(0)
    model = tiny("tree")
    a = example()
    b = copy.deepcopy(a)
    b["questions"]["color"]["criteria"]["g"] = "green: actually a shade of red"
    x, y = run(model, a), run(model, b)
    assert not torch.allclose(x[0][:2], y[0][:2], atol=1e-7)  # option r and b logits change: list is in the block
    c = copy.deepcopy(a)
    c["questions"]["intensity"]["criteria"][2] = "high, extremely so"
    z = run(model, c)
    torch.testing.assert_close(x[2][:2], z[2][:2], atol=1e-6, rtol=1e-5)  # levels 0 and 1 unchanged
    assert not torch.allclose(x[2][2], z[2][2], atol=1e-7)


def test_tree_leaf_logit_has_zero_jacobian_to_sibling_leaves():
    torch.manual_seed(0)
    model = tiny("tree")
    packed = pack_request(Request.from_dict(example()), ByteTokenizer(), "tree")
    embeds = model.backbone.get_input_embeddings()(packed.input_ids).detach().requires_grad_(True)
    logits = model(packed, inputs_embeds=embeds)
    logits[0][0].backward()
    grad = embeds.grad[0].abs().sum(-1)
    block = packed.branches[0]
    first, others = block.leaves[0], block.leaves[1:]
    assert grad[first[0]:first[1]].sum() > 0 and grad[block.start:block.end].sum() > 0
    for start, end in others:
        assert grad[start:end].sum() == 0


def test_set_head_is_exactly_permutation_equivariant_for_choice_and_independent_for_score():
    torch.manual_seed(0)
    model = tiny("set")
    a = example()
    b = copy.deepcopy(a)
    b["questions"]["color"]["criteria"] = {"g": "green", "r": "red", "b": "blue"}
    b["questions"]["color"]["target"] = [0, 1, 0]
    x, y = run(model, a), run(model, b)
    torch.testing.assert_close(y[0], x[0][[2, 0, 1]], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(x[0], torch.zeros(3))
    c = copy.deepcopy(a)
    c["questions"]["color"]["criteria"]["g"] = "green: actually a shade of red"
    assert not torch.allclose(run(model, c)[0][:2], x[0][:2], atol=1e-7)  # set interaction is real
    d = copy.deepcopy(a)
    d["questions"]["intensity"]["criteria"][2] = "high, extremely so"
    torch.testing.assert_close(run(model, d)[2][:2], x[2][:2], atol=1e-6, rtol=1e-5)


def test_pair_head_is_exactly_permutation_equivariant_and_independent_for_score():
    torch.manual_seed(0)
    model = tiny("pair", pair_width=32)
    assert model.packing_mode == "independent"
    a = example()
    b = copy.deepcopy(a)
    b["questions"]["color"]["criteria"] = {"g": "green", "r": "red", "b": "blue"}
    b["questions"]["color"]["target"] = [0, 1, 0]
    x, y = run(model, a), run(model, b)
    torch.testing.assert_close(y[0], x[0][[2, 0, 1]], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(x[0], torch.zeros(3))
    c = copy.deepcopy(a)
    c["questions"]["color"]["criteria"]["g"] = "green: actually a shade of red"
    assert not torch.allclose(run(model, c)[0][:2], x[0][:2], atol=1e-7)  # pairwise interaction is real
    d = copy.deepcopy(a)
    d["questions"]["intensity"]["criteria"][2] = "high, extremely so"
    torch.testing.assert_close(run(model, d)[2][:2], x[2][:2], atol=1e-6, rtol=1e-5)  # Score levels independent
    e = copy.deepcopy(a)
    e["questions"]["red"]["criteria"] = {"true": "Yes, it is red."}
    torch.testing.assert_close(run(model, e)[1][0], x[1][0], atol=1e-6, rtol=1e-5)  # Noul branches independent
    assert [len(z) for z in x] == [3, 2, 3]


def test_tree_and_set_train_and_roundtrip_checkpoints(tmp_path):
    from janus.data import synthetic_requests, write_jsonl
    from janus.training import TrainConfig, load_checkpoint, train
    torch.set_num_threads(2)
    for mode in ("tree", "set", "pair", "decoder"):
        paths = {}
        for name, seed in (("train", 17), ("dev", 23)):
            paths[name] = tmp_path / mode / f"{name}.jsonl"
            write_jsonl(paths[name], [r.to_dict() for r in synthetic_requests(4, seed=seed)])
        config = ModelConfig(backbone="tiny", mode=mode, adaptation="full", hidden_size=32, layers=1, head_rank=8)
        result = train(paths["train"], paths["dev"], tmp_path / mode / "run",
                       TrainConfig(model=config, epochs=1, accumulation=2, max_steps=2, device="cpu"))
        loaded, metadata = load_checkpoint(tmp_path / mode / "run" / "best.pt")
        assert metadata["model"]["mode"] == mode and result["steps"] == 2
        packed = pack_request(synthetic_requests(1, seed=17)[0], loaded.tokenizer, loaded.packing_mode)
        assert [len(z) for z in loaded(packed)] == [4, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernel needs a GPU")
@pytest.mark.parametrize("mode", ["tree", "set"])
def test_flex_matches_eager_for_new_graphs(mode):
    torch.manual_seed(0)
    eager = tiny(mode, hidden_size=128).cuda()
    flex = tiny(mode, hidden_size=128, attention="flex").cuda()
    flex.load_state_dict(eager.state_dict())
    packed = pack_request(Request.from_dict(example()), ByteTokenizer(), eager.packing_mode)
    for x, y in zip(eager(packed), flex(packed)):
        torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)


def test_tree_block_keys_only_and_position_schemes():
    request = Request.from_dict(example())
    tokenizer = ByteTokenizer()
    keys = pack_request(request, tokenizer, "tree", tree_block="keys")
    text = bytes(t - 1 for t in keys.input_ids[0].tolist()).decode()
    assert "[r]\n[b]\n[g]\n" in text and "[r] red" not in text
    full = pack_request(request, tokenizer, "tree")
    assert full.token_count > keys.token_count
    shared = pack_request(request, tokenizer, "tree", tree_positions="shared")
    restart = pack_request(request, tokenizer, "tree", tree_positions="restart")
    cont = pack_request(request, tokenizer, "tree", tree_positions="continue")
    block = shared.branches[0]
    starts = lambda packed: [int(packed.position_ids[0, s]) for s, _ in packed.branches[0].leaves]
    assert len(set(starts(shared))) == 1 and starts(shared)[0] == shared.state_length + (block.end - block.start)
    assert len(set(starts(restart))) == 1 and starts(restart)[0] == restart.state_length
    assert starts(cont) == sorted(starts(cont)) and len(set(starts(cont))) == 3
    assert torch.equal(shared.allowed, restart.allowed) and torch.equal(shared.allowed, cont.allowed)
    # Defaults are the existing layout exactly.
    default = pack_request(request, tokenizer, "tree")
    assert torch.equal(default.input_ids, shared.input_ids) and torch.equal(default.position_ids, shared.position_ids)
    with pytest.raises(ValueError, match="tree_block"):
        pack_request(request, tokenizer, "tree", tree_block="none")
    with pytest.raises(ValueError, match="tree_positions"):
        pack_request(request, tokenizer, "tree", tree_positions="none")


def test_packing_kwargs_follow_the_model_config():
    model = tiny("tree", tree_block="keys", tree_positions="restart")
    assert model.packing_kwargs == {"tree_block": "keys", "tree_positions": "restart", "score_block": "independent", "max_state_plus_question": 2048}
    assert tiny("listwise").packing_kwargs == {"tree_block": "full", "tree_positions": "shared", "score_block": "independent", "max_state_plus_question": 2048}
    with pytest.raises(ValueError, match="tree_block"):
        tiny("tree", tree_block="none")


def test_decoder_mode_outputs_isolation_and_accounting():
    torch.manual_seed(0)
    model = DecisionModel(ModelConfig(backbone="tiny", mode="decoder", adaptation="full", hidden_size=64, layers=2,
                                      head_rank=16, decoder_layers=2, decoder_heads=4)).eval()
    a = example()
    packed = pack_request(Request.from_dict(a), ByteTokenizer(), "decoder")
    assert packed.branches == [] and len(packed.branch_packs) == 3 and packed.branch_packs[0].state_length == 0
    assert all(int(b.position_ids[0, 0]) == 0 and b.parents.tolist() == [0, 0] for b in packed.branch_packs)
    assert model.packing_mode == "decoder"
    logits = [z.detach() for z in model(packed)]
    assert [len(z) for z in logits] == [3, 2, 3]
    assert model.last_compute["backbone_tokens"] == packed.token_count + sum(b.token_count for b in packed.branch_packs)
    assert model.last_compute["decoder_cross_attention_pairs"] == 2 * sum(b.token_count for b in packed.branch_packs) * packed.token_count
    b = copy.deepcopy(a)
    b["questions"]["red"]["instructions"] = "Secret blue! " * 12
    other = [z.detach() for z in model(pack_request(Request.from_dict(b), ByteTokenizer(), "decoder"))]
    torch.testing.assert_close(other[0], logits[0], atol=1e-6, rtol=1e-5)
    c = copy.deepcopy(a)
    c["state"] = "A blue card."
    changed = [z.detach() for z in model(pack_request(Request.from_dict(c), ByteTokenizer(), "decoder"))]
    assert not torch.allclose(changed[0], logits[0], atol=1e-7)
    # The decoder pack costs the same total tokens as the listwise pack, so max_tokens means the same budget.
    listwise = pack_request(Request.from_dict(a), ByteTokenizer(), "listwise")
    assert packed.token_count + sum(b.token_count for b in packed.branch_packs) == listwise.token_count
    with pytest.raises(ValueError, match="token"):
        pack_request(Request.from_dict(a), ByteTokenizer(), "decoder", max_tokens=listwise.token_count - 1)
