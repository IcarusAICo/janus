"""WP5 backbones: block-bidirectional mask, masked-next-token warm-up, masked-diffusion (LLaDA) loader and readout."""

import copy
import json
from pathlib import Path

import os
import pytest
import torch

from janus.attention import build_block_mask, dense_mask, tree_mask_mod
from janus.model import DecisionModel, ModelConfig
from janus.packing import ByteTokenizer, pack_request
from janus.schema import Request


def example():
    return {"state": "A red card.", "questions": {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"r": "red", "b": "blue", "g": "green"}, "target": [1, 0, 0]},
        "red": {"type": "noul", "instructions": "Is it red?", "target": [0, 1]},
        "intensity": {"type": "score", "instructions": "Intensity?", "criteria": ["low", "medium", "high"], "target": [0, 0, 1]}}}


def tiny(directionality="block", attention="eager", **extra):
    return DecisionModel(ModelConfig(backbone="tiny", mode="tree", adaptation="full", hidden_size=64, layers=2, head_rank=16,
                                     directionality=directionality, attention=attention, **extra)).eval()


# --- WP5a: the block rule -------------------------------------------------------------------------------------------

def test_block_rule_on_tree_layout():
    packed = pack_request(Request.from_dict(example()), ByteTokenizer(), mode="tree")
    s, n = packed.state_length, packed.token_count
    causal, block = packed.allowed, packed.mask("block")
    assert torch.equal(causal, dense_mask(packed.segment_ids, packed.parents))  # default stays the causal rule
    assert torch.equal(causal, dense_mask(packed.segment_ids, packed.parents, "causal"))
    assert not (causal & ~block).any()  # block only adds edges
    assert block[:s, :s].all() and not block[:s, s:].any()  # state fully connected, sees no branch
    for branch in packed.branches:
        rows = block[branch.start:branch.end]
        assert rows[:, :s].all() and rows[:, branch.start:branch.end].all()
        assert not rows[:, branch.leaves[0][0]:].any()  # a block never sees its leaves or anything later
        for start, end in branch.leaves:
            leaf = block[start:end]
            assert leaf[:, :s].all() and leaf[:, branch.start:branch.end].all() and leaf[:, start:end].all()
            expected = torch.zeros(n, dtype=torch.bool)
            expected[:s] = expected[branch.start:branch.end] = expected[start:end] = True
            assert torch.equal(leaf, expected[None].expand(end - start, n))  # nothing else: no sibling, no other branch
    with pytest.raises(ValueError, match="directionality"):
        packed.mask("magic")


def test_block_mask_mod_and_padding_match_the_dense_rule():
    from torch.nn.attention.flex_attention import create_mask
    packed = pack_request(Request.from_dict(example()), ByteTokenizer(), mode="tree")
    n = packed.token_count
    seg = torch.cat([packed.segment_ids, torch.full((5,), -1, dtype=torch.long)])
    dense = create_mask(tree_mask_mod(seg, packed.parents, "block"), None, None, n + 5, n + 5, device="cpu")[0, 0]
    assert torch.equal(dense[:n, :n], packed.mask("block"))
    assert torch.equal(dense[n:, :], torch.eye(n + 5, dtype=torch.bool)[n:]) and not dense[:n, n:].any()


@pytest.mark.parametrize("state_length,branch_lengths", [(127, (1,)), (129, (30, 3, 3)), (100, (27, 1, 1)), (300, (40, 5, 5, 200, 7))])
def test_block_mask_equals_torch_reference_in_block_mode(state_length, branch_lengths):
    from torch.nn.attention.flex_attention import create_block_mask
    segment_ids = torch.cat([torch.zeros(state_length, dtype=torch.long)] +
                            [torch.full((k,), i, dtype=torch.long) for i, k in enumerate(branch_lengths, start=1)])
    # Every odd segment is a question block under the state; every even one is a leaf under the previous block.
    parents = torch.tensor([0] + [0 if i % 2 == 1 else i - 1 for i in range(1, len(branch_lengths) + 1)])
    ours, total = build_block_mask(segment_ids, parents, "cpu", chunk_rows=256, directionality="block")
    seg = torch.cat([segment_ids, torch.full((total - segment_ids.shape[0],), -1, dtype=torch.long)])
    reference = create_block_mask(tree_mask_mod(seg, parents, "block"), None, None, total, total, device="cpu")
    assert torch.equal(ours.kv_num_blocks, reference.kv_num_blocks)
    assert torch.equal(ours.full_kv_num_blocks, reference.full_kv_num_blocks)
    assert torch.equal(ours.to_dense(), reference.to_dense())
    from torch.nn.attention.flex_attention import create_mask
    token_level = create_mask(tree_mask_mod(seg, parents, "block"), None, None, total, total, device="cpu")[0, 0]
    assert torch.equal(token_level[:segment_ids.shape[0], :segment_ids.shape[0]], dense_mask(segment_ids, parents, "block"))
    causal = create_mask(tree_mask_mod(seg, parents), None, None, total, total, device="cpu")[0, 0]
    assert token_level.sum() > causal.sum() and not (causal & ~token_level).any()


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_block_mode_leaf_logit_isolation_and_state_bidirectionality(attention):
    torch.manual_seed(0)
    model = tiny("block", attention)
    packed = pack_request(Request.from_dict(example()), ByteTokenizer(), "tree")
    embeds = model.backbone.get_input_embeddings()(packed.input_ids).detach().requires_grad_(True)
    logits = model(packed, inputs_embeds=embeds)
    logits[0][0].backward()
    grad = embeds.grad[0].abs().sum(-1)
    block, s = packed.branches[0], packed.state_length
    first = block.leaves[0]
    assert grad[:s].sum() > 0 and grad[block.start:block.end].sum() > 0 and grad[first[0]:first[1]].sum() > 0
    for start, end in block.leaves[1:]:
        assert grad[start:end].sum() == 0
    for other in packed.branches[1:]:
        assert grad[other.start:other.end].sum() == 0 and grad[other.leaves[0][0]:other.leaves[-1][1]].sum() == 0
    # Within a leaf the decision token now depends on nothing after it (it is last), but the first leaf token sees the
    # decision token: bidirectional inside the leaf.
    embeds2 = model.backbone.get_input_embeddings()(packed.input_ids).detach().requires_grad_(True)
    hidden = model._encode(packed, inputs_embeds=embeds2)
    hidden[first[0]].sum().backward()
    assert embeds2.grad[0][first[1] - 1].abs().sum() > 0
    # State token 0 depends on the last state token in block mode and not in causal mode.
    causal = tiny("causal", attention)
    causal.load_state_dict(model.state_dict())
    changed = example()
    changed["state"] = "A red carD."
    other = pack_request(Request.from_dict(changed), ByteTokenizer(), "tree")
    with torch.no_grad():
        assert not torch.allclose(model._encode(packed)[0], model._encode(other)[0], atol=1e-7)
        torch.testing.assert_close(causal._encode(packed)[0], causal._encode(other)[0], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(torch.stack(model(packed)[0:1]), torch.stack(causal(packed)[0:1]), atol=1e-7)


def test_directionality_is_validated_and_defaults_to_causal():
    assert ModelConfig().directionality == "causal"
    with pytest.raises(ValueError, match="directionality"):
        DecisionModel(ModelConfig(backbone="tiny", directionality="magic"))


def test_gradient_checkpointing_changes_nothing_but_memory():
    torch.manual_seed(0)
    plain = tiny("block", gradient_checkpointing=False)
    checkpointed = tiny("block", gradient_checkpointing=True)
    checkpointed.load_state_dict(plain.state_dict())
    packed = pack_request(Request.from_dict(example()), ByteTokenizer(), "tree")
    for model in (plain, checkpointed):
        model.train()
        sum(z.square().sum() for z in model(packed)).backward()
    for (name, p), (_, q) in zip(plain.named_parameters(), checkpointed.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, atol=1e-6, rtol=1e-5, msg=name)


# --- WP5a: masked-next-token warm-up ---------------------------------------------------------------------------------

def test_masked_next_token_warmup_loss_decreases_over_30_steps():
    from janus.data import synthetic_requests
    from janus.warmup import masked_next_token_warmup, mask_token_id, output_projection, state_pack
    torch.manual_seed(0)
    model = DecisionModel(ModelConfig(backbone="tiny", mode="tree", adaptation="lora", hidden_size=64, layers=2, head_rank=16,
                                      directionality="block", lora_rank=4)).eval()
    requests = synthetic_requests(4, seed=17)
    projection, extra = output_projection(model)
    assert extra and projection.weight.shape == (257, 64)  # untied, initialised from the embedding matrix
    torch.testing.assert_close(projection.weight, model.backbone.get_input_embeddings().weight.float())
    assert mask_token_id(model) == 0
    packed = state_pack(model, requests[0])
    assert packed.branches == [] and (packed.segment_ids == 0).all() and packed.token_count == packed.state_length
    before = {n: p.detach().clone() for n, p in model.backbone.named_parameters() if p.requires_grad}
    history = masked_next_token_warmup(model, requests, steps=30, seed=3)
    assert len(history) == 30 and all(h > 0 for h in history)
    assert sum(history[-5:]) / 5 < sum(history[:5]) / 5
    assert all("lora_" in n for n in before) and before
    assert any(not torch.equal(p, before[n]) for n, p in model.backbone.named_parameters() if n in before)
    assert not model.training and not hasattr(model, "mntp_head")  # the projection is discarded


def test_train_runs_the_warmup_when_configured(tmp_path):
    from janus.data import synthetic_requests, write_jsonl
    from janus.training import TrainConfig, train
    torch.set_num_threads(2)
    for name, seed in (("train", 17), ("dev", 23)):
        write_jsonl(tmp_path / f"{name}.jsonl", [r.to_dict() for r in synthetic_requests(4, seed=seed)])
    config = ModelConfig(backbone="tiny", mode="tree", adaptation="lora", hidden_size=32, layers=1, head_rank=8, directionality="block")
    result = train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / "run",
                   TrainConfig(model=config, epochs=1, accumulation=2, max_steps=1, device="cpu", mntp_steps=3))
    history = json.loads((tmp_path / "run" / "warmup.json").read_text())
    assert len(history) == 3 and result["warmup_steps"] == 3 and result["warmup_final_loss"] == history[-1]
    with pytest.raises(ValueError, match="mntp"):
        TrainConfig(model=config, mntp_steps=-1).validate()


# --- WP5c: masked-diffusion backbone --------------------------------------------------------------------------------

from janus.llada import (ANSWER_TOKENS, LLaDABackbone, TINY_MASK, TINY_NO, TINY_REPOS, TINY_YES, TinyLLaDATokenizer,
                       resolve_mask_token_id)


def cached(name):
    return LLaDABackbone.snapshot_dir(TINY_REPOS[name]) is not None


def llada_model(name, **extra):
    settings = dict(backbone=name, backbone_family="llada", trust_remote_code=True, mode="tree", adaptation="lora", lora_rank=4,
                    hidden_size=64, layers=2, dtype="float32", attention="sdpa", max_tokens=512)
    settings.update(extra)
    return DecisionModel(ModelConfig(**settings)).eval()


@pytest.mark.parametrize("name", ["tiny-llada", "tiny-llada2", "tiny-illada"])
def test_llada_tiny_backbone_readout_isolation_and_positions(name):
    if not cached(name):
        pytest.skip(f"remote code of {TINY_REPOS[name]} is not in the hub cache")
    torch.manual_seed(0)
    model = llada_model(name)
    assert model.config.directionality == "block" and model.backbone.mask_token_id == TINY_MASK
    assert model.backbone.answer_ids == (TINY_YES, TINY_NO)
    a = example()
    packed = pack_request(Request.from_dict(a), model.tokenizer, "tree")
    masked = model.masked_input_ids(packed)
    positions = [p for b in packed.branches for p in b.option_positions]
    assert (masked[0, positions] == TINY_MASK).all()
    keep = torch.ones(packed.token_count, dtype=torch.bool)
    keep[positions] = False
    assert torch.equal(masked[0, keep], packed.input_ids[0, keep])
    with torch.no_grad():
        logits = model(packed)
    assert [len(z) for z in logits] == [3, 2, 3] and logits[1][0].item() == 0.
    # The vocab readout is the LM head's " yes" minus " no" logit at the mask position.
    with torch.no_grad():
        hidden = model._encode(pack_request(Request.from_dict(a), model.tokenizer, "tree").__class__(
            masked, packed.position_ids, packed.segment_ids, packed.parents, packed.branches, packed.state_length, packed.option_counts))
        head = model.backbone.get_output_embeddings().weight
        rows = hidden[list(packed.branches[0].option_positions)]
        expected = rows @ head[TINY_YES] - rows @ head[TINY_NO]
    torch.testing.assert_close(logits[0], expected, atol=1e-5, rtol=1e-4)
    # Isolation: a sibling question's text cannot move the target leaf; the state can.
    b = copy.deepcopy(a)
    b["questions"]["red"]["instructions"] = "Secret blue! " * 8
    with torch.no_grad():
        other = model(pack_request(Request.from_dict(b), model.tokenizer, "tree"))
    torch.testing.assert_close(other[0], logits[0], atol=1e-5, rtol=1e-4)
    c = copy.deepcopy(a)
    c["state"] = "A blue card."
    with torch.no_grad():
        changed = model(pack_request(Request.from_dict(c), model.tokenizer, "tree"))
    assert not torch.allclose(changed[0], logits[0], atol=1e-6)
    # A leaf sees its own block (the option list) but not sibling leaves: change option g, r and b move.
    d = copy.deepcopy(a)
    d["questions"]["color"]["criteria"]["g"] = "green: a shade of red"
    with torch.no_grad():
        listed = model(pack_request(Request.from_dict(d), model.tokenizer, "tree"))
    assert not torch.allclose(listed[0][:2], logits[0][:2], atol=1e-6)
    # Packed positions reach the rotary embedding: the restart scheme gives different logits from shared.
    with torch.no_grad():
        restart = model(pack_request(Request.from_dict(a), model.tokenizer, "tree", tree_positions="restart"))
    assert not torch.allclose(restart[0], logits[0], atol=1e-6)
    # Scalar readout on the same mask position.
    scalar = llada_model(name, llada_readout="scalar")
    scalar.load_state_dict(model.state_dict())
    with torch.no_grad():
        z = scalar(packed)
    assert [len(v) for v in z] == [3, 2, 3] and not torch.allclose(z[0], logits[0])
    # LoRA is what trains, and a backward pass reaches it; gradient checkpointing does not change the logits.
    names = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    sum(v.square().sum() for v in model(packed)).backward()
    assert all(p.grad is not None for n, p in model.backbone.named_parameters() if p.requires_grad)
    checkpointed = llada_model(name, gradient_checkpointing=True)
    checkpointed.load_state_dict(model.state_dict())
    checkpointed.train()
    sum(v.square().sum() for v in checkpointed(packed)).backward()
    with torch.no_grad():
        for x, y in zip(model(packed), checkpointed.eval()(packed)):
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", ["tiny-llada", "tiny-llada2", "tiny-illada"])
def test_llada_warmup_uses_the_checkpoints_lm_head_and_mask_token(name):
    if not cached(name):
        pytest.skip(f"remote code of {TINY_REPOS[name]} is not in the hub cache")
    from janus.data import synthetic_requests
    from janus.warmup import masked_next_token_warmup, mask_token_id, output_projection
    torch.manual_seed(0)
    model = llada_model(name)
    projection, extra = output_projection(model)
    assert extra == [] and projection is model.backbone.get_output_embeddings()
    assert mask_token_id(model) == TINY_MASK
    history = masked_next_token_warmup(model, synthetic_requests(2, seed=17), steps=4, seed=1)
    assert len(history) == 4 and all(h > 0 for h in history)


def test_llada_checkpoint_roundtrip(tmp_path):
    if not cached("tiny-llada"):
        pytest.skip("remote code of GSAI-ML/LLaDA-8B-Base is not in the hub cache")
    from janus.training import checkpoint, load_checkpoint
    torch.manual_seed(0)
    model = llada_model("tiny-llada")
    checkpoint(model, tmp_path / "m.pt", {"step": 0})
    loaded, metadata = load_checkpoint(tmp_path / "m.pt")
    assert metadata["model"]["backbone_family"] == "llada" and metadata["model"]["directionality"] == "block"
    packed = pack_request(Request.from_dict(example()), model.tokenizer, "tree")
    with torch.no_grad():
        for x, y in zip(model(packed), loaded(packed)):
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)


def test_llada_loader_fails_loudly():
    with pytest.raises(ValueError, match="trust_remote_code"):
        DecisionModel(ModelConfig(backbone="tiny-llada", backbone_family="llada", mode="tree"))
    with pytest.raises(ValueError, match="mode tree"):
        DecisionModel(ModelConfig(backbone="tiny-llada", backbone_family="llada", trust_remote_code=True, mode="listwise"))
    with pytest.raises(ValueError, match="(?i)flex"):
        DecisionModel(ModelConfig(backbone="tiny-llada", backbone_family="llada", trust_remote_code=True, mode="tree", attention="flex"))
    with pytest.raises(ValueError, match="llada_readout"):
        DecisionModel(ModelConfig(backbone="tiny", llada_readout="magic"))
    with pytest.raises(ValueError, match="backbone_family"):
        DecisionModel(ModelConfig(backbone="tiny", backbone_family="bert"))
    if not cached("tiny-llada"):
        pytest.skip("remote code of GSAI-ML/LLaDA-8B-Base is not in the hub cache")
    lm = LLaDABackbone._tiny("tiny-llada", 64, 1, 256)

    class NoMask(TinyLLaDATokenizer):
        mask_token_id = None

    class MultiToken(TinyLLaDATokenizer):
        def encode(self, value, add_special_tokens=False):
            return [b + 1 for b in value.encode("utf-8")]

    class OtherMask(TinyLLaDATokenizer):
        mask_token_id = 7

    LLaDABackbone(lm, TinyLLaDATokenizer())
    with pytest.raises(ValueError, match="disagree"):
        LLaDABackbone(lm, OtherMask())
    with pytest.raises(ValueError, match="single token"):
        LLaDABackbone(lm, MultiToken())
    lm.config.mask_token_id = None
    with pytest.raises(ValueError, match="(?i)no mask token"):
        LLaDABackbone(lm, NoMask())

    class NamedOnly(NoMask):  # the LLaDA-8B-Base situation: no mask_token field, an added token named <|mdm_mask|>
        unk_token_id = 0

        def convert_tokens_to_ids(self, name):
            return TINY_MASK if name == "<|mdm_mask|>" else 0
    assert LLaDABackbone(lm, NamedOnly()).mask_token_id == TINY_MASK
    lm.config.mask_token_id = 5
    with pytest.raises(ValueError, match="disagree"):
        LLaDABackbone(lm, NamedOnly())
    lm.config.mask_token_id = 10_000
    with pytest.raises(ValueError, match="outside the embedding table"):
        LLaDABackbone(lm, NoMask())
    lm.config.mask_token_id = TINY_MASK
    lm.get_output_embeddings = lambda: None
    with pytest.raises(ValueError, match="LM head"):
        LLaDABackbone(lm, TinyLLaDATokenizer())


def test_llada_config_files_expose_mask_token_and_head():
    """What the readout relies on, read from the cached config and tokenizer files (no weights)."""
    from transformers import AutoConfig, AutoTokenizer
    from janus.llada import FAMILIES
    expectations = {"GSAI-ML/LLaDA-8B-Base": dict(mask=126336, vocab=126464, layers="n_layers", n=32, hidden="d_model", d=4096, cache=False),
                    "inclusionAI/LLaDA2.0-mini": dict(mask=156895, vocab=157184, layers="num_hidden_layers", n=20, hidden="hidden_size", d=2048, cache=False),
                    # iLLaDA: no mask_token field and no config mask_token_id; <[MASK]> is added token 5, the upstream README's mask_id=5.
                    "GSAI-ML/iLLaDA-8B-Base": dict(mask=5, vocab=155136, layers="num_hidden_layers", n=32, hidden="hidden_size", d=4096, cache=None)}
    seen = 0
    for repo, expected in expectations.items():
        snapshot = LLaDABackbone.snapshot_dir(repo)
        if snapshot is None or not (Path(snapshot) / "tokenizer.json").exists():
            continue
        seen += 1
        config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
        assert getattr(config, expected["layers"]) == expected["n"] and getattr(config, expected["hidden"]) == expected["d"]
        assert config.vocab_size == expected["vocab"]
        assert resolve_mask_token_id(config, tokenizer) == expected["mask"] < config.vocab_size
        assert getattr(config, "mask_token_id", None) in (None, expected["mask"])
        assert tokenizer.mask_token_id in (None, expected["mask"])  # LLaDA-8B-Base and iLLaDA leave mask_token unset
        assert all(len(tokenizer.encode(t, add_special_tokens=False)) == 1 for t in ANSWER_TOKENS)
        assert config.architectures[0] in FAMILIES
        if expected["cache"] is not None:
            assert getattr(config, "use_cache", False) == expected["cache"]
        if repo == "GSAI-ML/iLLaDA-8B-Base":
            assert config.tie_word_embeddings and config.num_key_value_heads == 8 and config.max_position_embeddings == 8192
            assert tokenizer.convert_tokens_to_ids("<[MASK]>") == 5 and tokenizer.encode(" yes", add_special_tokens=False) == [12883]
            assert tokenizer.encode(" no", add_special_tokens=False) == [1287]
    if not seen:
        pytest.skip("no LLaDA tokenizer files in the hub cache")


def test_illada_class_gets_dict_tied_keys_and_runs_under_our_mask():
    """The shipped `_tied_weights_keys` list is rewritten; bool and additive masks with packed positions agree."""
    if not cached("tiny-illada"):
        pytest.skip("remote code of GSAI-ML/iLLaDA-8B-Base is not in the hub cache")
    from transformers import AutoConfig
    from janus.llada import ILLADA_TIED_KEYS, illada_class
    snapshot = LLaDABackbone.snapshot_dir(TINY_REPOS["tiny-illada"])
    hf_config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
    cls = illada_class(hf_config, snapshot)
    assert cls.__name__ == "ILLaDAForCausalLM" and cls._tied_weights_keys == ILLADA_TIED_KEYS
    torch.manual_seed(0)
    lm = LLaDABackbone._tiny("tiny-illada", 64, 2, 256).eval()
    assert lm.lm_head.weight.data_ptr() == lm.get_input_embeddings().weight.data_ptr()  # tied, as the config says
    backbone = LLaDABackbone(lm, TinyLLaDATokenizer())
    assert backbone.family == "illada" and backbone.mask_token_id == TINY_MASK
    n = 12
    allowed = torch.zeros(n, n, dtype=torch.bool)
    allowed[:5, :5] = True
    allowed[5:9, :5] = allowed[5:9, 5:9] = True
    allowed[9:, :5] = allowed[9:, 9:] = True
    ids = torch.randint(1, 257, (1, n))
    positions = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7]])
    with torch.no_grad():
        hidden = backbone.encode(ids, positions, allowed)
        direct = lm(input_ids=ids, attention_mask=allowed[None, None], position_ids=positions, output_hidden_states=True).hidden_states[-1]
        torch.testing.assert_close(hidden, direct, atol=1e-6, rtol=1e-5)
        other = ids.clone()
        other[0, 10] = (other[0, 10] + 1) % 257
        moved = backbone.encode(other, positions, allowed)
    torch.testing.assert_close(moved[0, :9], hidden[0, :9], atol=1e-6, rtol=1e-5)  # the second branch cannot reach the first
    assert not torch.allclose(moved[0, 9:], hidden[0, 9:])


# --- WP5c: WeDLM-8B-Base (diffusion-trained, causal) under the in-library Qwen3 class ------------------------------

WEDLM = "tencent/WeDLM-8B-Base"
QWEN3_8B = "Qwen/Qwen3-8B-Base"


def wedlm_snapshot():
    return LLaDABackbone.snapshot_dir(WEDLM)


def test_wedlm_remote_forward_equals_qwen3_on_the_same_weights():
    """The remote WeDLMForCausalLM and Qwen3ForCausalLM give identical logits from one tiny random state dict.

    WeDLM's remote code targets transformers 4.56: it imports `check_model_inputs` (gone in 5) and its rotary module
    lacks `compute_default_rope_parameters`, which the 5.x weight initialiser calls. Both are shimmed here, in the
    test only; the arm itself never runs the remote code."""
    snapshot = wedlm_snapshot()
    if snapshot is None:
        pytest.skip("remote code of tencent/WeDLM-8B-Base is not in the hub cache")
    import sys
    import transformers.utils.generic as generic
    from transformers import AutoConfig, Qwen3Config, Qwen3ForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    if not hasattr(generic, "check_model_inputs"):
        generic.check_model_inputs = lambda f: f
    tiny = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=16, max_position_embeddings=512, layer_types=["full_attention"] * 2, max_window_layers=2,
                tie_word_embeddings=False, pad_token_id=0, eos_token_id=0)
    torch.manual_seed(0)
    remote_config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, **tiny)
    assert remote_config.qk_norm and not remote_config.attention_bias
    cls = get_class_from_dynamic_module(remote_config.auto_map["AutoModelForCausalLM"], snapshot)
    rotary = sys.modules[cls.__module__].WeDLMRotaryEmbedding
    if not hasattr(rotary, "compute_default_rope_parameters"):
        rotary.compute_default_rope_parameters = lambda self, config, device=None, seq_len=None: self._compute_default_rope_parameters(config, device)
    remote = cls._from_config(remote_config, attn_implementation="eager", dtype=torch.float32).eval()
    qwen = Qwen3ForCausalLM(Qwen3Config(**tiny, rms_norm_eps=remote_config.rms_norm_eps, attention_bias=False, attention_dropout=0.,
                                        rope_parameters={"rope_type": "default", "rope_theta": remote_config.rope_theta})).eval()
    qwen.config._attn_implementation = "eager"
    assert {k: v.shape for k, v in remote.state_dict().items()} == {k: v.shape for k, v in qwen.state_dict().items()}
    qwen.load_state_dict(remote.state_dict(), strict=True)
    ids = torch.randint(1, 300, (1, 12))
    with torch.no_grad():
        torch.testing.assert_close(remote(input_ids=ids, use_cache=False).logits, qwen(input_ids=ids, use_cache=False).logits, atol=0., rtol=0.)
    # The 4-D mask path DecisionModel uses (dict mask, packed positions) matches as well.
    packed = pack_request(Request.from_dict(example()), ByteTokenizer(), "tree")
    n = packed.token_count
    bias = torch.zeros(n, n).masked_fill(~packed.allowed, torch.finfo(torch.float32).min)[None, None]
    with torch.no_grad():
        a = remote(input_ids=packed.input_ids, attention_mask={"full_attention": bias}, position_ids=packed.position_ids, use_cache=False).logits
        b = qwen(input_ids=packed.input_ids, attention_mask={"full_attention": bias}, position_ids=packed.position_ids, use_cache=False).logits
    torch.testing.assert_close(a, b, atol=0., rtol=0.)


def test_wedlm_config_and_index_are_qwen3_8b_shaped():
    """From the cached config.json and safetensors index: same tensor names as Qwen3-8B-Base, config maps to Qwen3Config."""
    from janus.wedlm import config_dict, qwen3_config
    snapshot, qwen_snapshot = wedlm_snapshot(), LLaDABackbone.snapshot_dir(QWEN3_8B)
    if snapshot is None or qwen_snapshot is None:
        pytest.skip("WeDLM-8B-Base or Qwen3-8B-Base config files are not in the hub cache")
    raw = config_dict(snapshot)
    assert raw["model_type"] == "wedlm" and raw["architectures"] == ["WeDLMForCausalLM"] and raw["qk_norm"]
    mapped, reference = qwen3_config(snapshot), qwen3_config(qwen_snapshot)
    assert reference is None  # only wedlm configs are translated
    from transformers import Qwen3Config
    reference = Qwen3Config.from_pretrained(qwen_snapshot)
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                "vocab_size", "rms_norm_eps", "attention_bias", "tie_word_embeddings", "hidden_act"):
        assert getattr(mapped, key) == getattr(reference, key), key
    assert mapped.rope_parameters["rope_theta"] == reference.rope_parameters["rope_theta"] == 1e6
    assert mapped.max_position_embeddings == 16384 and not mapped.use_cache and mapped._commit_hash == raw["_commit_hash"]
    ours, theirs = Path(snapshot) / "model.safetensors.index.json", Path(qwen_snapshot) / "model.safetensors.index.json"
    if ours.exists() and theirs.exists():
        assert set(json.loads(ours.read_text())["weight_map"]) == set(json.loads(theirs.read_text())["weight_map"])


def test_wedlm_typed_checkpoint_loads_as_qwen3_through_decision_model(tmp_path):
    """A directory with a wedlm config.json and Qwen3 weights loads with Qwen3Model, no remote code, weights intact."""
    snapshot = wedlm_snapshot()
    if snapshot is None:
        pytest.skip("config files of tencent/WeDLM-8B-Base are not in the hub cache")
    import shutil
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
    torch.manual_seed(0)
    hf_config = Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=False, pad_token_id=0,
                            eos_token_id=0, rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    lm = Qwen3ForCausalLM(hf_config)
    lm.save_pretrained(tmp_path)
    wedlm = json.loads((Path(snapshot) / "config.json").read_text())
    for key in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "max_position_embeddings", "tie_word_embeddings", "pad_token_id", "eos_token_id"):
        wedlm[key] = getattr(hf_config, key)
    wedlm.update(layer_types=["full_attention"] * 2, max_window_layers=2, rope_theta=1e6)
    (tmp_path / "config.json").write_text(json.dumps(wedlm))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"):
        if (Path(snapshot) / name).exists():
            shutil.copy(Path(snapshot) / name, tmp_path / name)
    if not (tmp_path / "tokenizer.json").exists():
        pytest.skip("WeDLM tokenizer files are not in the hub cache")
    model = DecisionModel(ModelConfig(backbone=str(tmp_path), mode="tree", adaptation="frozen", max_tokens=256, dtype="float32")).eval()
    assert type(model.backbone) is Qwen3Model and model.backbone.config.model_type == "qwen3"
    assert type(model.tokenizer).__name__ == "Qwen2Tokenizer" and model.tokenizer.encode(" yes", add_special_tokens=False) == [9834]
    reference = lm.model.state_dict()
    assert all(torch.equal(reference[k], v) for k, v in model.backbone.state_dict().items()) and len(reference) == len(model.backbone.state_dict())
    packed = pack_request(Request.from_dict(example()), ByteTokenizer(), "tree")  # byte ids fit the 300-row table
    with torch.no_grad():
        logits = model(packed)
    assert [len(z) for z in logits] == [3, 2, 3]


@pytest.mark.skipif(os.environ.get("JANUS_WEIGHT_TESTS") != "1", reason="loads 8B checkpoints; set JANUS_WEIGHT_TESTS=1")
@pytest.mark.parametrize("arm", ["llada_8b_tree", "illada8b_tree"])
def test_real_llada_checkpoints_load_the_shards_faithfully(arm):
    """Guards the transformers-5 remote-class loading bug found 2026-09-18 (iLLaDA came out with 226 of 291 tensors
    re-initialised after a clean loading report). Needs the shards in the project cache."""
    import glob, json
    from safetensors import safe_open
    from janus.model import DecisionModel, ModelConfig
    raw = json.load(open(f"configs/phase3_backbones/{arm}.json"))["model"]; raw["adaptation"] = "frozen"
    lm = DecisionModel(ModelConfig(**raw)).backbone.lm
    snapshot = glob.glob(f".cache/huggingface/hub/models--{raw['backbone'].replace('/', '--')}/snapshots/*")[0]
    index = json.load(open(f"{snapshot}/model.safetensors.index.json"))["weight_map"]
    state = lm.state_dict()
    for key in list(index)[::40]:
        with safe_open(f"{snapshot}/{index[key]}", "pt", device="cpu") as f:
            assert torch.equal(state[key].to(torch.bfloat16), f.get_tensor(key)), key


@pytest.mark.parametrize("name", ["tiny-llada", "tiny-illada"])
def test_llada_backward_after_an_inference_mode_pass(name):
    """The 8B arm failed at its first backward after the step-0 dev pass: the remote rotary cache held inference
    tensors. A forward under inference_mode followed by a training step must work."""
    if not cached(name):
        pytest.skip(f"remote code of {TINY_REPOS[name]} is not in the hub cache")
    torch.manual_seed(0)
    model = llada_model(name, gradient_checkpointing=True)  # the 8B arms checkpoint; the recomputation is where the save happens
    packed = pack_request(Request.from_dict(example()), model.tokenizer, "tree")
    for module in model.backbone.lm.modules():  # the 8B fills its rotary cache lazily on the first forward; force that here
        cache = getattr(module, f"_{type(module).__name__}__cache", None)
        if isinstance(cache, dict):
            cache.clear()
    with torch.inference_mode():
        model.eval()
        model(packed)
    model.train()
    sum(logits.sum() for logits in model(packed)).backward()
