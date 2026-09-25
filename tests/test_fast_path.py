import copy

import pytest
import torch

from janus.packing import ByteTokenizer, pack_request
from janus.schema import Request


def example():
    return {"state": "A red card.", "questions": {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"r": "red", "b": "blue"}, "target": [1, 0]},
        "red": {"type": "noul", "instructions": "Is it red?", "target": [0, 1]},
        "intensity": {"type": "score", "instructions": "Intensity?", "criteria": ["low", "medium", "high"], "target": [0, 0, 1]}}}


def legacy_allowed(packed):
    n = packed.token_count
    s = packed.state_length
    allowed = torch.zeros((n, n), dtype=torch.bool)
    allowed[:s, :s] = torch.ones(s, s).tril().bool()
    for branch in packed.branches:
        allowed[branch.start:branch.end, :s] = True
        allowed[branch.start:branch.end, branch.start:branch.end] = torch.ones(branch.end - branch.start, branch.end - branch.start).tril().bool()
    return allowed


def test_segment_metadata_and_dense_mask_match_the_legacy_rule():
    packed = pack_request(Request.from_dict(example()), ByteTokenizer())
    assert packed.segment_ids.shape == (packed.token_count,)
    assert (packed.segment_ids[:packed.state_length] == 0).all()
    for i, branch in enumerate(packed.branches, start=1):
        assert (packed.segment_ids[branch.start:branch.end] == i).all()
    assert packed.parents.tolist() == [0] * (len(packed.branches) + 1)
    assert torch.equal(packed.allowed, legacy_allowed(packed))


def test_mask_mod_materialises_to_the_dense_mask_with_self_only_padding():
    from torch.nn.attention.flex_attention import create_mask
    from janus.attention import tree_mask_mod
    packed = pack_request(Request.from_dict(example()), ByteTokenizer())
    n = packed.token_count
    padded = n + 5
    seg = torch.cat([packed.segment_ids, torch.full((5,), -1, dtype=torch.long)])
    dense = create_mask(tree_mask_mod(seg, packed.parents), None, None, padded, padded, device="cpu")[0, 0]
    assert torch.equal(dense[:n, :n], packed.allowed)
    assert torch.equal(dense[n:, :], torch.eye(padded, dtype=torch.bool)[n:])
    assert not dense[:n, n:].any()


def test_block_mask_matches_torch_reference_without_materialising_the_dense_mask():
    from torch.nn.attention.flex_attention import create_block_mask
    from janus.attention import build_block_mask, tree_mask_mod
    torch.manual_seed(0)
    lengths = [300] + torch.randint(5, 200, (12,)).tolist()
    segment_ids = torch.cat([torch.full((k,), i, dtype=torch.long) for i, k in enumerate(lengths)])
    parents = torch.zeros(len(lengths), dtype=torch.long)
    mask, total = build_block_mask(segment_ids, parents, "cpu", chunk_rows=256)
    seg = torch.cat([segment_ids, torch.full((total - segment_ids.shape[0],), -1, dtype=torch.long)])
    reference = create_block_mask(tree_mask_mod(seg, parents), None, None, total, total, device="cpu")
    assert total % 128 == 0 and total - 128 < segment_ids.shape[0] <= total
    assert torch.equal(mask.kv_num_blocks, reference.kv_num_blocks)
    assert torch.equal(mask.full_kv_num_blocks, reference.full_kv_num_blocks)
    assert torch.equal(mask.to_dense(), reference.to_dense())
    assert torch.equal(mask.to_dense(), reference.to_dense())


from janus.model import DecisionModel, ModelConfig


def tiny(attention, hidden=128):
    return DecisionModel(ModelConfig(backbone="tiny", mode="listwise", adaptation="full",
                                     hidden_size=hidden, layers=2, head_rank=16, attention=attention)).eval()


def test_config_rejects_unknown_attention_and_defaults_to_eager():
    assert ModelConfig().attention == "eager"
    with pytest.raises(ValueError, match="attention"):
        DecisionModel(ModelConfig(backbone="tiny", attention="magic"))


def test_inputs_embeds_path_matches_input_ids_path():
    torch.manual_seed(0)
    model = tiny("eager")
    packed = pack_request(Request.from_dict(example()), ByteTokenizer())
    a = model(packed)
    embeds = model.backbone.get_input_embeddings()(packed.input_ids.to(model.device))
    b = model(packed, inputs_embeds=embeds)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)


def test_sdpa_matches_eager_on_cpu():
    torch.manual_seed(0)
    eager = tiny("eager")
    sdpa = tiny("sdpa")
    sdpa.load_state_dict(eager.state_dict())
    packed = pack_request(Request.from_dict(example()), ByteTokenizer())
    for x, y in zip(eager(packed), sdpa(packed)):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernel needs a GPU")
def test_flex_matches_eager_logits_and_gradients_on_gpu():
    torch.manual_seed(0)
    eager = tiny("eager").cuda()
    flex = tiny("flex").cuda()
    flex.load_state_dict(eager.state_dict())
    packed = pack_request(Request.from_dict(example()), ByteTokenizer())
    a, b = eager(packed), flex(packed)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-4, rtol=1e-4)
    sum(z.square().sum() for z in a).backward()
    sum(z.square().sum() for z in b).backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), flex.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, atol=1e-5, rtol=1e-3, msg=name)


@pytest.mark.parametrize("attention", ["eager", "sdpa"] + (["flex"] if torch.cuda.is_available() else []))
def test_target_logits_have_zero_jacobian_to_sibling_tokens_and_nonzero_to_state(attention):
    torch.manual_seed(0)
    model = tiny(attention)
    if attention == "flex":
        model = model.cuda()
    packed = pack_request(Request.from_dict(example()), ByteTokenizer())
    embeds = model.backbone.get_input_embeddings()(packed.input_ids.to(model.device)).detach().requires_grad_(True)
    logits = model(packed, inputs_embeds=embeds)
    logits[0].sum().backward()
    grad = embeds.grad[0].abs().sum(-1)
    target = packed.branches[0]
    assert grad[:packed.state_length].sum() > 0
    assert grad[target.start:target.end].sum() > 0
    for sibling in packed.branches[1:]:
        assert grad[sibling.start:sibling.end].sum() == 0


def test_load_checkpoint_accepts_attention_override(tmp_path):
    from janus.training import checkpoint, load_checkpoint
    model = tiny("eager")
    checkpoint(model, tmp_path / "m.pt", {"step": 0})
    loaded, _ = load_checkpoint(tmp_path / "m.pt", attention="sdpa")
    assert loaded.config.attention == "sdpa"
    packed = pack_request(Request.from_dict(example()), ByteTokenizer())
    for x, y in zip(model(packed), loaded(packed)):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("state_length,branch_lengths", [(127, (1,)), (128, (1,)), (129, (1,)), (100, (27, 1)), (100, (28, 1)), (100, (29, 1)), (255, (1, 1)), (256, (1,)), (257, (1,))])
def test_block_mask_equals_dense_rule_at_block_boundaries(state_length, branch_lengths):
    """Spec WP0: 127/128/129 tile boundaries, exact block-mask equality with the dense rule."""
    from torch.nn.attention.flex_attention import create_block_mask
    from janus.attention import build_block_mask, tree_mask_mod
    segment_ids = torch.cat([torch.zeros(state_length, dtype=torch.long)] +
                            [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(branch_lengths, start=1)])
    parents = torch.zeros(len(branch_lengths) + 1, dtype=torch.long)
    ours, total = build_block_mask(segment_ids, parents, "cpu")
    seg = torch.cat([segment_ids, torch.full((total - segment_ids.shape[0],), -1, dtype=torch.long)])
    reference = create_block_mask(tree_mask_mod(seg, parents), None, None, total, total, device="cpu")
    assert torch.equal(ours.to_dense(), reference.to_dense())
