"""Qwen3.5 hybrid backbone (janus.hybrid): the tree-of-batches runner against a replay reference, isolation, padding,
gradients through the prefix state, multi-state batching, the chunked state pass, flex attention, and the
DecisionModel/checkpoint path. CPU, fp32, tiny random weights."""

import functools
import json
from pathlib import Path

import pytest
import torch

from janus.hybrid import LORA_TARGETS, HybridBackbone, TINY, chunk_rule, tree_levels
from janus.model import DecisionModel, ModelConfig
from janus.packing import ByteTokenizer, pack_request
from janus.schema import Request
from janus.training import checkpoint, load_checkpoint
from janus.data import synthetic_requests, write_jsonl


def example():
    return {"state": "A red card.", "questions": {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"r": "red", "b": "blue", "g": "green"}, "target": [1, 0, 0]},
        "red": {"type": "noul", "instructions": "Is it red?", "target": [0, 1]},
        "intensity": {"type": "score", "instructions": "Intensity?", "criteria": ["low", "medium", "high"], "target": [0, 0, 1]}}}


def backbone(attention="sdpa", seed=0, hidden=64, layers=4):
    torch.manual_seed(seed)
    return HybridBackbone(HybridBackbone._tiny(hidden, layers, 512, attention), ByteTokenizer(), attention).eval()


def packed_tree(raw=None, **kwargs):
    kwargs.setdefault("tree_positions", "continue")
    return pack_request(Request.from_dict(raw or example()), ByteTokenizer(), "tree", **kwargs)


def replay(lm, packed, positions):
    """The stock HF forward on the tokens at `positions` as one sequence with their packed positions and a causal mask.

    (Without the explicit mask, transformers reads a position jump as a packed-sequence boundary.)"""
    n = len(positions)
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None]
    if lm.config._attn_implementation == "eager":  # eager adds the mask to the logits, so it must be additive
        causal = torch.zeros(causal.shape).masked_fill(~causal, torch.finfo(torch.float32).min)
    return lm(input_ids=packed.input_ids[:, positions], position_ids=packed.position_ids[:, positions],
              attention_mask=causal, use_cache=False).last_hidden_state[0]


def paths(packed):
    """(leaf span, token positions of state + block + leaf) for every leaf of the tree."""
    state = list(range(packed.state_length))
    for branch in packed.branches:
        block = list(range(branch.start, branch.end))
        for start, end in branch.leaves:
            yield (start, end), state + block + list(range(start, end))


def test_tiny_config_is_the_hybrid_layout():
    bb = backbone()
    assert bb.config.layer_types == ["linear_attention", "full_attention", "linear_attention", "full_attention"]
    assert bb.config.linear_conv_kernel_dim == 4 and bb.config.vocab_size == 257 and bb.config.hidden_size == 64
    assert chunk_rule("cpu").__name__ == "torch_chunk_gated_delta_rule"
    assert type(bb.lm.layers[0].linear_attn.norm).__name__ == "Qwen3_5RMSNormGated"


def test_tree_levels_follow_the_packed_layout():
    packed = packed_tree()
    levels = tree_levels(packed.segment_ids, packed.parents)
    assert levels[0] == [(0, 0, packed.state_length)]
    assert [s for s, _, _ in levels[1]] == [b for b in range(1, len(packed.parents)) if packed.parents[b] == 0]
    assert len(levels) == 3 and len(levels[2]) == sum(len(b.leaves) for b in packed.branches)
    for s, start, end in levels[2]:
        assert (packed.segment_ids[start:end] == s).all() and packed.parents[s] in [t for t, _, _ in levels[1]]
    with pytest.raises(ValueError, match="no tokens"):
        tree_levels(torch.ones(4, dtype=torch.long), torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="contiguous"):
        tree_levels(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0]))


@pytest.mark.parametrize("attention", ["sdpa", "eager"])
@pytest.mark.parametrize("tree_positions", ["continue", "shared"])
def test_runner_matches_replay_of_every_root_to_leaf_path(attention, tree_positions):
    """(a) Every branch position equals the branch run as its own sequence [state + block + leaf] with no cache.

    Three branches of different lengths (choice with three options, noul with one leaf, score with three leaves)."""
    bb = backbone(attention)
    packed = packed_tree(tree_positions=tree_positions)
    lengths = {end - start for b in packed.branches for start, end in b.leaves}
    assert len(lengths) >= 3 and len({b.end - b.start for b in packed.branches}) == 3
    with torch.no_grad():
        ours = bb.encode(packed)
        prefix = replay(bb.lm, packed, list(range(packed.state_length)))
        torch.testing.assert_close(ours[:packed.state_length], prefix, atol=1e-4, rtol=1e-4)
        for (start, end), positions in paths(packed):
            reference = replay(bb.lm, packed, positions)
            torch.testing.assert_close(ours[start:end], reference[-(end - start):], atol=1e-4, rtol=1e-4)
            block = next(b for b in packed.branches if b.leaves and (start, end) in b.leaves)
            length = block.end - block.start
            torch.testing.assert_close(ours[block.start:block.end], reference[packed.state_length:packed.state_length + length],
                                       atol=1e-4, rtol=1e-4)


def test_sibling_isolation():
    """(b) Changing branch 2 (its block and leaves) leaves the state, branch 0 and branch 1 outputs unchanged."""
    bb = backbone()
    a = packed_tree()
    changed = example()
    changed["questions"]["intensity"]["instructions"] = "Secret blue! " * 6
    changed["questions"]["intensity"]["criteria"] = ["very low", "medium-ish", "extremely high"]
    b = packed_tree(changed)
    boundary = a.branches[2].start
    assert b.token_count != a.token_count and b.branches[2].start == boundary
    with torch.no_grad():
        ha, hb = bb.encode(a), bb.encode(b)
    torch.testing.assert_close(hb[:boundary], ha[:boundary], atol=1e-6, rtol=0)
    assert not torch.allclose(hb[-1], ha[-1], atol=1e-3)  # the changed branch itself moved
    # The state text reaches every branch.
    changed = example()
    changed["state"] = "A blue card."
    c = packed_tree(changed)
    with torch.no_grad():
        hc = bb.encode(c)
    for branch in a.branches:
        for start, end in branch.leaves:
            assert not torch.allclose(hc[end - 1], ha[end - 1], atol=1e-4)


def test_leaf_only_change_leaves_siblings_unchanged():
    """(b) with the block held fixed: a score block never lists the levels, so a level description change is leaf-only."""
    bb = backbone()
    a = packed_tree()
    changed = example()
    changed["questions"]["intensity"]["criteria"] = ["low", "medium", "very very high"]
    b = packed_tree(changed)
    block_a, block_b = a.branches[2], b.branches[2]
    assert (block_a.start, block_a.end) == (block_b.start, block_b.end) and block_a.leaves[:2] == block_b.leaves[:2]
    with torch.no_grad():
        ha, hb = bb.encode(a), bb.encode(b)
    last = block_a.leaves[1][1]
    torch.testing.assert_close(hb[:last], ha[:last], atol=1e-6, rtol=0)
    assert not torch.allclose(hb[block_b.leaves[2][1] - 1], ha[block_a.leaves[2][1] - 1], atol=1e-3)


@pytest.mark.parametrize("attention", ["sdpa", "eager"])
def test_right_padding_invariance(attention):
    """(c) Extra padding columns in every level batch change no real-position output."""
    bb = backbone(attention)
    packed = packed_tree()
    with torch.no_grad():
        plain = bb.encode(packed)
        padded = bb.encode(packed, extra_padding=7)
    torch.testing.assert_close(padded, plain, atol=1e-5, rtol=1e-5)


def test_gradients_reach_deltanet_lora_and_the_prefix_pass():
    """(d) A loss on leaf outputs moves LoRA weights inside a DeltaNet layer and flows into the state pass."""
    torch.manual_seed(0)
    model = DecisionModel(ModelConfig(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4,
                                      hidden_size=64, layers=4, dtype="float32", attention="sdpa", max_tokens=512,
                                      tree_positions="continue")).eval()
    names = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert any("linear_attn.in_proj_qkv" in n for n in names) and any("linear_attn.out_proj" in n for n in names)
    assert any("self_attn.q_proj" in n for n in names) and set(LORA_TARGETS) >= {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
    packed = packed_tree()
    embeds = model.backbone.get_input_embeddings()(packed.input_ids).detach().requires_grad_(True)
    logits = model(packed, inputs_embeds=embeds)
    assert [len(z) for z in logits] == [3, 2, 3]
    logits[0].square().sum().backward()
    grads = {n: p.grad for n, p in model.backbone.named_parameters() if p.requires_grad}
    delta = [n for n in grads if "linear_attn" in n and grads[n] is not None and grads[n].abs().sum() > 0]
    attention = [n for n in grads if "self_attn" in n and grads[n] is not None and grads[n].abs().sum() > 0]
    assert delta and attention
    token = embeds.grad[0].abs().sum(-1)
    s, block = packed.state_length, packed.branches[0]
    assert token[:s].sum() > 0 and token[block.start:block.end].sum() > 0  # the prefix and block passes are in the graph
    for start, end in block.leaves:
        assert token[start:end].sum() > 0  # every leaf feeds one of the three logits
    for other in packed.branches[1:]:
        assert token[other.start:other.leaves[-1][1]].sum() == 0  # siblings do not
    # Detaching the prefix removes the LoRA gradient contribution that came through the state (it is not the whole gradient).
    model.zero_grad(set_to_none=True)
    hidden = model._encode(packed)
    hidden[list(block.option_positions)].sum().backward()
    full = {n: p.grad.clone() for n, p in model.backbone.named_parameters() if p.requires_grad and "layers.0.linear_attn" in n}
    assert full and any(g.abs().sum() > 0 for g in full.values())


def three_requests():
    """Three requests whose states (and question sets) differ in length."""
    a = example()
    b = example()
    b["state"] = "A long description of a card that is red, with a border and a number printed on it. " * 2
    del b["questions"]["intensity"]
    c = example()
    c["state"] = "Blue."
    c["questions"]["extra"] = {"type": "noul", "instructions": "Is it a card?", "target": [0, 1]}
    return [Request.from_dict(r) for r in (a, b, c)]


def lora_model(**overrides):
    torch.manual_seed(0)
    settings = dict(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4, hidden_size=64, layers=4,
                    dtype="float32", attention="sdpa", max_tokens=512, tree_positions="continue")
    return DecisionModel(ModelConfig(**{**settings, **overrides})).eval()


def test_encode_many_matches_per_state_encode_and_its_gradients():
    """(f) Three states of different lengths as one batch equal the per-state runner, outputs and LoRA gradients."""
    model = lora_model()
    packs = [pack_request(r, model.tokenizer, "tree", 512, tree_positions="continue") for r in three_requests()]
    assert len({p.state_length for p in packs}) == 3 and len({len(p.parents) for p in packs}) == 3
    bb = model.backbone
    with torch.no_grad():
        for hidden, packed in zip(bb.encode_many(packs), packs):
            torch.testing.assert_close(hidden, bb.encode(packed), atol=1e-4, rtol=1e-4)
    lora = [(n, p) for n, p in bb.named_parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    sum(h.square().sum() for h in bb.encode_many(packs)).backward()
    batched = {n: p.grad.clone() for n, p in lora}
    model.zero_grad(set_to_none=True)
    for packed in packs:
        bb.encode(packed).square().sum().backward()
    for n, p in lora:  # not byte-identical: batched fp32 GEMMs accumulate in another order (1e-5 relative)
        torch.testing.assert_close(batched[n], p.grad, atol=1e-4, rtol=1e-4, msg=n)
    # The model-level hook: forward_many gives the per-pack logits of forward.
    with torch.no_grad():
        for many, packed in zip(model.forward_many(packs), packs):
            for x, y in zip(many, model(packed)):
                torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)
    assert not bb._prepared


def test_chunked_state_pass_matches_unchunked():
    """(g) A state spanning three chunks, with and without gradient checkpointing, equals the one-piece state pass."""
    raw = example()
    raw["state"] = "The card on the table is red and has a gold border. " * 2  # 104 bytes: three chunks of 40
    packed = packed_tree(raw)
    assert 2 * 40 < packed.state_length <= 3 * 40
    whole = backbone()
    chunked = backbone(); chunked.state_chunk_tokens = 40
    with torch.no_grad():
        reference = whole.encode(packed)
        torch.testing.assert_close(chunked.encode(packed), reference, atol=1e-4, rtol=1e-4)
    chunked.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    chunked.train()
    torch.testing.assert_close(chunked.encode(packed), reference, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("chunk", [10 ** 9, 40])
def test_flex_matches_sdpa(chunk, monkeypatch):
    """(h) Flex attention (a BlockMask per level batch) equals sdpa on the same weights, batched and chunked, with an
    ancestor longer than the sdpa key chunk (flex scores every ancestor key in one piece)."""
    from janus import hybrid
    monkeypatch.setattr(hybrid, "ATTN_CHUNK", 8)
    sdpa = backbone("sdpa")
    flex = backbone("flex")
    flex.load_state_dict(sdpa.state_dict())
    sdpa.state_chunk_tokens = flex.state_chunk_tokens = chunk
    assert flex.config._attn_implementation == "jev_flex"
    long = example()
    long["state"] = "The card on the table is red and has a gold border. " * 2
    packs = [packed_tree(long), packed_tree()]
    with torch.no_grad():
        for a, b in zip(flex.encode_many(packs), sdpa.encode_many(packs)):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_piece_query_chunks_match_one_piece(monkeypatch):
    """`_piece` scored in query chunks (PIECE_BYTES bounds its fp32 score matrix) equals the single piece, causal and
    not, with padded keys and a chunk that does not divide the query count."""
    from janus import hybrid
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 4, 7, 8), torch.randn(2, 2, 9, 8), torch.randn(2, 2, 9, 8)
    kv_valid = torch.ones(2, 9, dtype=torch.bool)
    kv_valid[1, 6:] = False
    for causal in (False, True):
        o, lse = hybrid._piece(q, k, v, kv_valid, causal, 0.5)
        monkeypatch.setattr(hybrid, "PIECE_BYTES", 4 * 2 * 4 * 9 * 3)  # three queries per chunk
        chunked = hybrid._piece(q, k, v, kv_valid, causal, 0.5)
        monkeypatch.setattr(hybrid, "PIECE_BYTES", hybrid.PIECE_BYTES * 1000)
        torch.testing.assert_close(chunked[0], o, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(chunked[1], lse, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention has no CPU backward")
def test_flex_gradients_match_sdpa_on_gpu():
    """(h) The gradient through the compiled flex kernel (prefix embeddings, LoRA-free tiny model) equals sdpa's."""
    sdpa = backbone("sdpa").cuda()
    flex = backbone("flex").cuda()
    flex.load_state_dict(sdpa.state_dict())
    sdpa.state_chunk_tokens = flex.state_chunk_tokens = 40
    long = example()
    long["state"] = "The card on the table is red and has a gold border. " * 2
    packs = [packed_tree(long), packed_tree()]
    embeds = [sdpa.lm.embed_tokens(p.input_ids.cuda()).detach().requires_grad_(True) for p in packs]
    grads = []
    for bb in (sdpa, flex):
        for e in embeds:
            e.grad = None
        sum(h.square().sum() for h in bb.encode_many(packs, embeds)).backward()
        grads.append([e.grad.clone() for e in embeds])
    for a, b in zip(*grads):
        torch.testing.assert_close(b, a, atol=1e-3, rtol=1e-3)


def test_unsupported_settings_fail_loudly():
    settings = dict(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", hidden_size=64, layers=4, dtype="float32", max_tokens=512)
    with pytest.raises(ValueError, match="causal"):
        DecisionModel(ModelConfig(**{**settings, "directionality": "block"}))
    with pytest.raises(ValueError, match="state_chunk_tokens"):
        DecisionModel(ModelConfig(**{**settings, "state_chunk_tokens": 0}))
    with pytest.raises(ValueError, match="decoder"):
        DecisionModel(ModelConfig(**{**settings, "mode": "decoder"}))
    with pytest.raises(ValueError, match="backbone_family"):
        DecisionModel(ModelConfig(backbone="tiny", backbone_family="mamba"))
    bb = backbone()
    with pytest.raises(ValueError, match="causal"):
        bb.encode(packed_tree(), directionality="block")


def test_decision_model_end_to_end_and_checkpoint_roundtrip(tmp_path):
    """(e) forward + loss in tree mode, packing_mode, the evaluate-style collect path, gradient checkpointing, checkpoints."""
    from janus.metrics import distribution_loss
    from janus.training import collect_logits
    torch.manual_seed(0)
    config = ModelConfig(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4, hidden_size=64, layers=4,
                         dtype="float32", attention="sdpa", max_tokens=512, tree_positions="continue")
    model = DecisionModel(config).eval()
    assert model.packing_mode == "tree" and model.packing_kwargs == {"tree_block": "full", "tree_positions": "continue", "score_block": "independent", "max_state_plus_question": 512}
    assert model.config.revision is None and model.backbone.config.use_cache is False
    request = Request.from_dict(example())
    packed = pack_request(request, model.tokenizer, model.packing_mode, config.max_tokens, **model.packing_kwargs)
    logits = model(packed)
    assert [len(z) for z in logits] == [3, 2, 3] and logits[1][0].item() == 0.
    loss = distribution_loss(logits, [q.target for q in request.questions], "ce", 0.)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.head.parameters())
    z, y = collect_logits(model, [request, request])
    assert len(z) == 6 and all(torch.equal(a, b) for a, b in zip(z[:3], z[3:]))
    # Gradient checkpointing: same logits and the same gradients.
    checkpointed = DecisionModel(ModelConfig(**{**config.__dict__, "gradient_checkpointing": True}))
    checkpointed.load_state_dict(model.state_dict())
    assert checkpointed.backbone.text_model.gradient_checkpointing
    for m in (model, checkpointed):
        m.train()
        m.zero_grad(set_to_none=True)
        sum(v.square().sum() for v in m(packed)).backward()
    for (name, p), (_, q) in zip(model.named_parameters(), checkpointed.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, atol=1e-6, rtol=1e-5, msg=name)
    with torch.no_grad():
        for x, y in zip(model.eval()(packed), checkpointed.eval()(packed)):
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)
    # Checkpoint round trip (tiny backbones store every weight).
    checkpoint(model, tmp_path / "m.pt", {"step": 3})
    loaded, metadata = load_checkpoint(tmp_path / "m.pt")
    assert metadata["model"]["backbone_family"] == "qwen3_5" and metadata["step"] == 3
    with torch.no_grad():
        for x, y in zip(model(packed), loaded(packed)):
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)


def test_train_loop_runs_on_the_hybrid(tmp_path):
    from janus.data import synthetic_requests, write_jsonl
    from janus.training import TrainConfig, train
    torch.set_num_threads(2)
    for name, seed in (("train", 17), ("dev", 23)):
        write_jsonl(tmp_path / f"{name}.jsonl", [r.to_dict() for r in synthetic_requests(3, seed=seed)])
    config = ModelConfig(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4, hidden_size=32, layers=2,
                         dtype="float32", attention="sdpa", max_tokens=1024, tree_positions="continue")
    result = train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / "run",
                   TrainConfig(model=config, epochs=1, accumulation=3, max_steps=1, device="cpu"))
    assert result["steps"] == 1 and (tmp_path / "run" / "best.pt").exists()
    assert json.loads((tmp_path / "run" / "config.json").read_text())["config"]["model"]["backbone_family"] == "qwen3_5"


@pytest.mark.skipif(not Path("configs/phase3_backbones").exists(), reason="needs configs/phase3_backbones (not in the public release)")
def test_configs_pin_the_hybrid_arms():
    for name, repo in (("qwen35_4b_tree", "Qwen/Qwen3.5-4B-Base"), ("qwen35_9b_tree", "Qwen/Qwen3.5-9B-Base")):
        raw = json.loads(Path(f"configs/phase3_backbones/{name}.json").read_text())
        model = raw["model"]
        assert model["backbone"] == repo and model["backbone_family"] == "qwen3_5" and model["mode"] == "tree"
        assert len(model["revision"]) == 40 and model["attention"] in ("sdpa", "eager") and model["dtype"] == "bfloat16"
        assert model["lora_rank"] == 16 and model["max_tokens"] == 4096 and model.get("directionality", "causal") == "causal"
        assert model["gradient_checkpointing"] is True and raw["max_seconds"] in (7200, 10800)
        assert model["tree_positions"] == "continue"


def test_batched_training_matches_per_request_training(tmp_path):
    """batch_states 3 groups the accumulation batch through forward_many; the trained weights must match batch_states 1."""
    from janus.training import TrainConfig, train
    torch.set_num_threads(2)
    for name, seed in (("train", 17), ("dev", 23)):
        write_jsonl(tmp_path / f"{name}.jsonl", [r.to_dict() for r in synthetic_requests(3, seed=seed)])
    weights = {}
    for batch_states in (1, 3):
        config = ModelConfig(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4, hidden_size=32,
                             layers=2, dtype="float32", attention="sdpa", max_tokens=1024, tree_positions="continue",
                             batch_states=batch_states)
        train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / f"run{batch_states}",
              TrainConfig(model=config, epochs=1, accumulation=3, max_steps=1, device="cpu", seed=17))
        weights[batch_states] = torch.load(tmp_path / f"run{batch_states}" / "best.pt", map_location="cpu", weights_only=False)["weights"]
    for key, value in weights[1].items():
        if value.is_floating_point():
            assert torch.allclose(value, weights[3][key], atol=1e-4, rtol=1e-4), key


def test_group_tokens_matches_row_grouping_and_isolates_an_oversized_row(tmp_path):
    """`group_tokens` splits the accumulation batch by summed packed tokens: at a budget that reproduces the
    batch_states 2 grouping the step trains the same weights, and a row whose pack alone exceeds the budget still
    trains, in a group of its own."""
    from dataclasses import replace
    from janus.training import TrainConfig, token_groups, train
    torch.set_num_threads(2)
    base = dict(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4, hidden_size=32,
                layers=2, dtype="float32", attention="sdpa", max_tokens=2048, tree_positions="continue")
    requests = synthetic_requests(4, seed=17)
    write_jsonl(tmp_path / "train.jsonl", [r.to_dict() for r in requests])
    write_jsonl(tmp_path / "dev.jsonl", [r.to_dict() for r in synthetic_requests(3, seed=23)])
    model = DecisionModel(ModelConfig(**base))
    tokens = functools.partial(pack_request, tokenizer=model.tokenizer, mode=model.packing_mode,
                               max_tokens=2048, **model.packing_kwargs)
    counts = sorted(tokens(r).token_count for r in requests)
    budget = 2 * counts[-1]  # two rows fit, three do not: the same pairs the row count gives
    assert list(token_groups(counts, 8, budget)) == list(token_groups(counts, 2, 0)) == [(0, 2), (2, 4)]
    weights = {}
    for name, states, group_tokens in (("rows", 2, 0), ("tokens", 8, budget)):
        train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / name,
              TrainConfig(model=ModelConfig(**base, batch_states=states), epochs=1, accumulation=4, max_steps=1,
                          device="cpu", seed=17, group_tokens=group_tokens))
        weights[name] = torch.load(tmp_path / name / "last.pt", map_location="cpu", weights_only=False)["weights"]
    for key, value in weights["rows"].items():
        if value.is_floating_point():
            assert torch.allclose(value, weights["tokens"][key], atol=1e-5, rtol=1e-5), key
    # A row three times the size of the others: over the budget on its own, so it neither joins a group nor stops the step.
    long = replace(requests[0], group_id="long", questions=tuple(replace(q, id=f"long:{i}:{q.id}")
                                                                 for i in range(3) for q in requests[0].questions))
    assert tokens(long).token_count > budget
    assert list(token_groups(sorted(counts + [tokens(long).token_count]), 8, budget))[-1] == (4, 5)
    write_jsonl(tmp_path / "long.jsonl", [r.to_dict() for r in (*requests, long)])
    result = train(tmp_path / "long.jsonl", tmp_path / "dev.jsonl", tmp_path / "long_run",
                   TrainConfig(model=ModelConfig(**base, batch_states=8), epochs=1, accumulation=5, max_steps=1,
                               device="cpu", seed=17, group_tokens=budget))
    assert result["steps"] == 1 and result["requests_seen"] == 5


def test_batched_consistency_twins_match_separate_forwards(tmp_path):
    """The option-permuted consistency view rides in the group's forward_many (batch_states 2 originals plus their
    twins in one pass): one step at consistency_weight .1 trains the same weights as forwarding every twin on its own
    (_consistency_separate), the term does move the weights, and the twin count is unchanged."""
    from janus.training import TrainConfig, train
    torch.set_num_threads(2)
    for name, seed in (("train", 17), ("dev", 23)):
        write_jsonl(tmp_path / f"{name}.jsonl", [r.to_dict() for r in synthetic_requests(4, seed=seed)])
    config = ModelConfig(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4, hidden_size=32,
                         layers=2, dtype="float32", attention="sdpa", max_tokens=1024, tree_positions="continue", batch_states=2)
    arms = {"batched": {"consistency_weight": .1}, "separate": {"consistency_weight": .1, "_consistency_separate": True},
            "off": {"consistency_weight": 0.}}
    weights, twins = {}, {}
    for name, arm in arms.items():
        result = train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / name,
                       TrainConfig(model=config, epochs=1, accumulation=4, max_steps=1, device="cpu", seed=17, **arm))
        assert result["best_step"] == 1, name  # best.pt must hold the stepped weights, not the initial ones
        weights[name], twins[name] = torch.load(tmp_path / name / "best.pt", map_location="cpu", weights_only=False)["weights"], result["consistency_forwards"]
    assert twins == {"batched": 4, "separate": 4, "off": 0}
    floating = [k for k, v in weights["batched"].items() if v.is_floating_point()]
    for key in floating:
        assert torch.allclose(weights["batched"][key], weights["separate"][key], atol=1e-5, rtol=1e-5), key
    assert any(not torch.equal(weights["batched"][key], weights["off"][key]) for key in floating)


@pytest.mark.parametrize("options", [1, 3, 40])
def test_branch_levels_match_replay_for_wide_trees(options):
    """(i) Shared-prefix attention (branch_attention): a Choice with 1, 3 and 40 options beside a noul and a score equals
    the per-path replay at every branch position; the block's keys are read once, not once per leaf."""
    bb = backbone()
    raw = example()
    raw["questions"]["color"]["criteria"] = {f"k{i}": f"opt {i}" for i in range(options)}
    raw["questions"]["color"]["target"] = [1] + [0] * (options - 1)
    packed = packed_tree(raw, max_tokens=4096)
    assert len(packed.branches[0].leaves) == options
    with torch.no_grad():
        ours = bb.encode(packed)
        for (start, end), positions in paths(packed):
            reference = replay(bb.lm, packed, positions)
            torch.testing.assert_close(ours[start:end], reference[-(end - start):], atol=1e-4, rtol=1e-4)


def test_prefix_snapshot_hit_equals_miss():
    """(j) encode_prefixes then encode_many(prefixes=...) reproduces the branch outputs of the plain call, for a single
    pack, a batch of hits and a batch mixing hits and misses; a hit leaves zeros at its state positions."""
    model = lora_model()
    bb = model.backbone
    packs = [pack_request(r, model.tokenizer, "tree", 512, tree_positions="continue") for r in three_requests()]
    with torch.no_grad():
        plain = bb.encode_many(packs)
        snapshots = bb.encode_prefixes(packs)
        assert [s.tokens for s in snapshots] == [p.state_length for p in packs] and all(0 < s.kv_bytes < s.bytes for s in snapshots)
        assert all(t.shape[0] == 1 for s in snapshots for pair in s.states for t in pair)
        hits = bb.encode_many(packs, prefixes=snapshots)
        mixed = bb.encode_many(packs, prefixes=[snapshots[0], None, snapshots[2]])
        single = bb.encode(packs[1], prefix=bb.encode_prefix(packs[1]))
        for i, (packed, reference) in enumerate(zip(packs, plain)):
            s = packed.state_length
            for out, hit in ((hits[i], True), (mixed[i], i != 1), (single, True) if i == 1 else (mixed[i], i != 1)):
                torch.testing.assert_close(out[s:], reference[s:], atol=1e-6, rtol=1e-6)
                assert hit == bool((out[:s] == 0).all())
        # The model-level path the server takes: prepare_many with snapshots, then the ordinary per-pack forward.
        bb.prepare_many(packs, snapshots)
        for logits, expected in zip([model(p) for p in packs], model.forward_many(packs)):
            for x, y in zip(logits, expected):
                torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)
        assert not bb._prepared


# --- qwen3_5_moe (Qwen3.6-35B-A3B layout: the same mixers plus a sparse MoE feed-forward) and its NVFP4 weights ---


def moe_backbone(attention="sdpa", seed=0):
    torch.manual_seed(seed)
    return HybridBackbone(HybridBackbone._tiny(64, 4, 512, attention, moe=True), ByteTokenizer(), attention).eval()


@pytest.mark.parametrize("tree_positions", ["continue", "shared"])
def test_moe_runner_matches_hf_forward_and_prefix_hit_equals_miss(tree_positions):
    """The tiny qwen3_5_moe through the tree runner equals the HF Qwen3_5MoeTextModel forward on every root-to-leaf path
    (the shared-prefix branch attention over the state and block levels), and a prefix-snapshot hit equals the miss."""
    bb = moe_backbone()
    assert type(bb.lm).__name__ == "Qwen3_5MoeTextModel" and hasattr(bb.lm.layers[0].mlp, "experts")
    packed = packed_tree(tree_positions=tree_positions)
    with torch.no_grad():
        ours = bb.encode(packed)
        torch.testing.assert_close(ours[:packed.state_length], replay(bb.lm, packed, list(range(packed.state_length))), atol=1e-4, rtol=1e-4)
        for (start, end), positions in paths(packed):
            reference = replay(bb.lm, packed, positions)
            torch.testing.assert_close(ours[start:end], reference[-(end - start):], atol=1e-4, rtol=1e-4)
            block = next(b for b in packed.branches if (start, end) in b.leaves)
            torch.testing.assert_close(ours[block.start:block.end], reference[packed.state_length:block.end - block.start + packed.state_length], atol=1e-4, rtol=1e-4)
        hit = bb.encode_many([packed], prefixes=bb.encode_prefixes([packed]))[0]
        torch.testing.assert_close(hit[packed.state_length:], ours[packed.state_length:], atol=1e-6, rtol=1e-6)


def test_nvfp4_round_trip_and_packed_modules():
    """quantise -> dequantise in the ModelOpt layout: values already on the fp4 grid come back exactly, random weights
    within e2m1's worst step (amax/6, the 4 -> 6 gap) plus the e4m3 block-scale rounding; NVFP4Experts (W4A16 eager
    path) and NVFP4Linear equal the dequantised reference; the fp4 GEMM mode is Blackwell-only and CPU refuses it."""
    from janus.hybrid import NVFP4Experts, NVFP4Linear, _e2m1_decode, nvfp4_dequantize, nvfp4_quantize
    torch.manual_seed(0)
    w = torch.randn(48, 64) * 3
    packed, scale, g = nvfp4_quantize(w)
    assert packed.dtype == torch.uint8 and packed.shape == (48, 32) and scale.dtype == torch.float8_e4m3fn and scale.shape == (48, 4) and g.shape == ()
    amax = w.view(48, 4, 16).abs().amax(-1).repeat_interleave(16, -1)
    assert ((nvfp4_dequantize(packed, scale, g, torch.float32) - w).abs() <= amax / 6 * (1 + 1 / 16) + 1e-6).all()
    codes = torch.randint(0, 16, (48, 64), dtype=torch.uint8)
    codes[:, ::16] = 7  # every block carries a +6, so the block scale is re-derived exactly
    grid = _e2m1_decode(codes) * scale.float().repeat_interleave(16, -1) * g
    torch.testing.assert_close(nvfp4_dequantize(*nvfp4_quantize(grid, g), torch.float32), grid, atol=1e-6, rtol=1e-6)  # fp32 reassociation only
    E, I, H = 8, 32, 64
    gu, dn = torch.randn(E, 2 * I, H), torch.randn(E, H, I)
    stack = lambda ts, rows: tuple(torch.stack(parts) if k < 2 else torch.stack(parts)[:, None, None].expand(E, rows, 1).contiguous()
                                  for k, parts in enumerate(zip(*(nvfp4_quantize(t) for t in ts))))
    experts = NVFP4Experts(*stack(gu, 2 * I), *stack(dn, H))
    assert not any(True for _ in experts.parameters()) and not experts.state_dict()  # buffers only, and nothing persisted
    x, index, weights = torch.randn(10, H), torch.randint(0, E, (10, 2)), torch.rand(10, 2)
    out = experts(x, index, weights)
    GU, DN = experts.gate_up_proj.float(), experts.down_proj.float()
    reference = torch.zeros_like(x)
    for t in range(10):
        for k in range(2):
            gate, up = (x[t] @ GU[index[t, k]].T).chunk(2)
            reference[t] += weights[t, k] * (torch.nn.functional.silu(gate) * up) @ DN[index[t, k]].T
    torch.testing.assert_close(out, reference, atol=1e-3, rtol=1e-3)
    linear = NVFP4Linear(*nvfp4_quantize(torch.randn(32, 64)))
    torch.testing.assert_close(linear(x), x @ linear.weight.float().T, atol=1e-3, rtol=1e-3)
    linear.set_mode(fp4=True)
    with pytest.raises(Exception):  # scaled_mm's fp4 recipe needs a Blackwell device
        linear(x.bfloat16())


def test_moe_lora_trains_only_projections_and_head(tmp_path):
    """One optimizer step on the tiny MoE moves the LoRA adapters of the attention and DeltaNet projections and the head,
    never the experts, the shared expert or the router; `frozen` trains the head alone; a checkpoint of a non-tiny run
    holds only the adapters and the head."""
    from janus.metrics import distribution_loss
    from janus.training import checkpoint
    torch.manual_seed(0)
    settings = dict(backbone="tiny-qwen3_5_moe", backbone_family="qwen3_5_moe", mode="tree", adaptation="lora", lora_rank=4, hidden_size=64,
                    layers=4, dtype="float32", attention="sdpa", max_tokens=512, tree_positions="continue")
    model = DecisionModel(ModelConfig(**settings))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    request = Request.from_dict(example())
    packed = pack_request(request, model.tokenizer, "tree", 512, tree_positions="continue")
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.)
    distribution_loss(model(packed), [q.target for q in request.questions], "ce", 0.).backward()
    optimizer.step()
    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert moved and all(n.startswith("head.") or "lora_" in n for n in moved)
    targets = {n.split(".lora_")[0].rsplit(".", 2)[-2] + "." + n.split(".lora_")[0].rsplit(".", 1)[-1] for n in moved if "lora_" in n}
    assert {"self_attn.q_proj", "linear_attn.in_proj_qkv", "linear_attn.out_proj"} <= targets
    assert not any(part in n for n in moved for part in ("experts", "shared_expert", "mlp.gate"))
    model.config.backbone = "nvidia/Qwen3.6-35B-A3B-NVFP4"  # a real checkpoint stores adapters and head only
    checkpoint(model, tmp_path / "m.pt", {"step": 1})
    keys = set(torch.load(tmp_path / "m.pt", weights_only=True)["weights"])
    assert keys and all(k.startswith("head.") or "lora_" in k for k in keys) and not any("experts" in k for k in keys)
    frozen = DecisionModel(ModelConfig(**{**settings, "adaptation": "frozen"}))
    assert all(n.startswith("head.") for n, p in frozen.named_parameters() if p.requires_grad)
    assert torch.isfinite(distribution_loss(frozen(packed), [q.target for q in request.questions], "ce", 0.))


@pytest.mark.skipif(not Path("configs/phase4_production/qwen36_a3b_v3.json").exists(), reason="needs configs/phase4_production/qwen36_a3b_v3.json (not in the public release)")
def test_v3_recipe_train_step_on_the_tiny_moe(tmp_path):
    """One optimizer step of the v3 production recipe (LoRA on the projections, gradient checkpointing, consistency
    twins, held-out selection) on the tiny MoE runs end to end through janus.training.train on a 4-row train file."""
    from janus.training import TrainConfig, train
    torch.set_num_threads(2)
    recipe = json.loads(Path("configs/phase4_production/qwen36_a3b_v3.json").read_text())
    assert recipe["model"]["backbone_family"] == "qwen3_5_moe" and recipe["model"]["gradient_checkpointing"] and recipe["model"]["adaptation"] == "lora"
    assert recipe["selection"] == "dev_nll_plus_heldout" and recipe["consistency_weight"] == 0.1 and recipe["accumulation"] == 32 and recipe["model"]["max_tokens"] == 8192
    for name, seed, count in (("train", 17, 4), ("dev", 23, 2), ("heldout", 29, 2)):
        write_jsonl(tmp_path / f"{name}.jsonl", [r.to_dict() for r in synthetic_requests(count, seed=seed)])
    model = dict(recipe["model"], backbone="tiny-qwen3_5_moe", revision=None, hidden_size=32, layers=2, dtype="float32", max_tokens=1024, lora_rank=4)
    config = TrainConfig.from_dict(dict(recipe, model=model, device="cpu", accumulation=4, max_steps=1, eval_every=1, warmup_steps=1,
                                        heldout_dev=str(tmp_path / "heldout.jsonl")))
    result = train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / "run", config)
    assert result["steps"] == 1 and result["consistency_forwards"] > 0 and result["heldout_requests"] == 2 and (tmp_path / "run" / "best.pt").exists()
    assert model_family(tmp_path / "run" / "config.json") == "qwen3_5_moe"


def model_family(path):
    return json.loads(Path(path).read_text())["config"]["model"]["backbone_family"]


def test_checkpointing_keeps_no_dequantised_expert_stack_for_backward():
    """With NVFP4 experts swapped into the tiny MoE, autograd under gradient checkpointing saves no tensor the size of
    a dequantised expert stack (the stacks are recomputed inside the layer step in backward); without checkpointing
    it would keep one per layer. The loader refuses that configuration for the real checkpoint."""
    from janus.hybrid import NVFP4Experts, nvfp4_quantize
    torch.manual_seed(0)
    settings = dict(backbone="tiny-qwen3_5_moe", backbone_family="qwen3_5_moe", mode="tree", adaptation="lora", lora_rank=4, hidden_size=64,
                    layers=4, dtype="float32", attention="sdpa", max_tokens=512, tree_positions="continue")
    saved = {}
    for checkpointing in (False, True):
        model = DecisionModel(ModelConfig(**settings, gradient_checkpointing=checkpointing)).train()
        E = 8
        for layer in model.backbone.text_model.layers:
            experts = layer.mlp.experts
            gu, dn = experts.gate_up_proj.detach(), experts.down_proj.detach()
            stack = lambda ts, rows: tuple(torch.stack(parts) if k < 2 else torch.stack(parts)[:, None, None].expand(E, rows, 1).contiguous()
                                          for k, parts in enumerate(zip(*(nvfp4_quantize(t) for t in ts))))
            layer.mlp.experts = NVFP4Experts(*stack(gu, gu.shape[1]), *stack(dn, dn.shape[1]), experts.act_fn)
        stack_numel = E * gu.shape[1] * gu.shape[2]
        big = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: (big.append(t.numel()) if t.numel() >= stack_numel else None, t)[1], lambda t: t):
            logits = model(packed_tree())
            sum(z.sum() for z in logits).backward()
        saved[checkpointing] = len(big)
        assert all(p.grad is not None for n, p in model.named_parameters() if "lora_" in n and "in_proj_qkv" in n)
    assert saved[False] >= 4 and saved[True] == 0, saved


def test_moe_kernels_agree(monkeypatch):
    """The three expert GEMM paths (`JANUS_MOE_KERNEL` grouped | bmm | loop) give the same NVFP4Experts output, and
    the tiny MoE backbone gives the same hidden states under the HF grouped_mm and eager experts; the default is the
    grouped GEMM only from sm_90 on (the L40S, sm_89, stalled in torch's grouped_mm fallback)."""
    from janus.hybrid import NVFP4Experts, moe_kernel, nvfp4_quantize
    torch.manual_seed(0)
    E, I, H = 8, 32, 64
    gu, dn = torch.randn(E, 2 * I, H), torch.randn(E, H, I)
    stack = lambda ts, rows: tuple(torch.stack(parts) if k < 2 else torch.stack(parts)[:, None, None].expand(E, rows, 1).contiguous()
                                  for k, parts in enumerate(zip(*(nvfp4_quantize(t) for t in ts))))
    experts = NVFP4Experts(*stack(gu, 2 * I), *stack(dn, H))
    x, index, weights = torch.randn(40, H).bfloat16(), torch.randint(0, E, (40, 2)), torch.rand(40, 2).bfloat16()
    index[:5, 0] = 3  # an unbalanced expert, and expert 7 possibly unused
    outs = {}
    for kernel in ("grouped", "bmm", "loop"):
        monkeypatch.setenv("JANUS_MOE_KERNEL", kernel)
        assert moe_kernel("cpu") == kernel
        outs[kernel] = experts(x, index, weights).float()
    torch.testing.assert_close(outs["bmm"], outs["loop"], atol=1e-2, rtol=1e-2)  # bf16 GEMMs, another accumulation order
    torch.testing.assert_close(outs["grouped"], outs["loop"], atol=1e-2, rtol=1e-2)
    monkeypatch.setenv("JANUS_MOE_KERNEL", "bad")
    with pytest.raises(ValueError):
        moe_kernel("cpu")
    monkeypatch.delenv("JANUS_MOE_KERNEL")
    assert moe_kernel("cpu") == "bmm"
    bb = moe_backbone()
    packed = packed_tree()
    hidden = {}
    for kernel in ("grouped", "bmm"):
        monkeypatch.setenv("JANUS_MOE_KERNEL", kernel)
        with torch.no_grad():
            hidden[kernel] = bb.encode(packed)
        assert bb.lm.config._experts_implementation == ("grouped_mm" if kernel == "grouped" else "eager")
    torch.testing.assert_close(hidden["grouped"], hidden["bmm"], atol=1e-4, rtol=1e-4)


def test_fused_moe_tile_plan_covers_every_assignment():
    """`janus.moe_w4a16.align` (torch ops only, so checked on the CPU): every token-expert assignment lands in exactly
    one tile row, every tile holds one expert's assignments, padding rows carry the sentinel, the device-side total is
    the padded row count, and the tiles past it belong to no expert."""
    from janus.moe_w4a16 import align
    torch.manual_seed(0)
    S, k, E, block = 40, 2, 8, 16
    index = torch.randint(0, E, (S, k))
    index[:20, 0] = 3  # one hot expert spans several tiles
    sorted_ids, expert_ids, total = align(index, E, block)
    counts = torch.bincount(index.reshape(-1), minlength=E)
    assert int(total) == int(((counts + block - 1) // block * block).sum()) and sorted_ids.shape[0] % block == 0
    real = sorted_ids[sorted_ids < S * k]
    assert real.shape[0] == S * k and real.sort().values.tolist() == list(range(S * k))
    for tile in range(sorted_ids.shape[0] // block):
        rows = sorted_ids[tile * block:(tile + 1) * block]
        rows = rows[rows < S * k]
        if tile * block >= int(total):
            assert rows.numel() == 0 and int(expert_ids[tile]) >= E
        else:
            assert (index.reshape(-1)[rows] == expert_ids[tile]).all()
    monkey = align(torch.full((5, 2), 7), E, block)[2]  # a single expert: one padded tile
    assert int(monkey) == block


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel is Triton on CUDA")
def test_fused_w4a16_moe_matches_dequant_path():
    """The fused Triton kernel (`JANUS_MOE_KERNEL=fused`, the CUDA default) equals the dequantise-and-loop path of
    NVFP4Experts to bf16 accumulation order on the A3B's expert shapes, with skewed routing and an unused expert, and
    under a CUDA graph."""
    from janus.hybrid import NVFP4Experts, nvfp4_quantize
    from janus.moe_w4a16 import fused_moe
    torch.manual_seed(0)
    E, I, H = 16, 512, 2048
    gu, dn = torch.randn(E, 2 * I, H) * 0.05, torch.randn(E, H, I) * 0.05
    stack = lambda ts, rows: tuple(torch.stack(parts) if k < 2 else torch.stack(parts)[:, None, None].expand(E, rows, 1).contiguous()
                                  for k, parts in enumerate(zip(*(nvfp4_quantize(t) for t in ts))))
    experts = NVFP4Experts(*stack(gu, 2 * I), *stack(dn, H)).cuda()
    for S in (3, 100, 700):
        x = (torch.randn(S, H) * 0.5).bfloat16().cuda()
        index = torch.randint(0, E - 1, (S, 8)).cuda()  # expert E-1 never hit
        index[: S // 2, 0] = 2  # a hot expert
        weights = torch.softmax(torch.randn(S, 8), -1).bfloat16().cuda()
        with torch.no_grad():
            out = fused_moe(x, index, weights, experts.gate_up, experts.gate_up_scale, experts.gate_up_global, experts.down, experts.down_scale, experts.down_global)
            reference = experts._eager(x, index, weights, experts.gate_up_proj, experts.down_proj)
        torch.testing.assert_close(out.float(), reference.float(), atol=2e-2, rtol=2e-2)
        assert (out.float() - reference.float()).abs().max() < 0.02 * reference.float().abs().max() + 1e-3
    graph = torch.cuda.CUDAGraph()
    args = (x, index, weights, experts.gate_up, experts.gate_up_scale, experts.gate_up_global, experts.down, experts.down_scale, experts.down_global)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        fused_moe(*args)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph), torch.no_grad():
        replayed = fused_moe(*args)
    graph.replay()
    torch.testing.assert_close(replayed, out)


def test_branch_levels_chunked_by_block_equal_one_pass(monkeypatch):
    """The branch levels run in chunks of BRANCH_CHUNK_ROWS level-1 rows (each with its descendants); one block per
    chunk gives the same hidden states as every block at once, on a cache hit too."""
    from janus import hybrid
    bb = moe_backbone()
    packed = packed_tree()
    assert len(packed.branches) >= 2
    with torch.no_grad():
        whole = bb.encode(packed)
        prefixes = bb.encode_prefixes([packed])
        monkeypatch.setattr(hybrid, "BRANCH_CHUNK_ROWS", 1)
        chunked = bb.encode(packed)
        hit = bb.encode_many([packed], prefixes=prefixes)[0]
    torch.testing.assert_close(chunked, whole, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(hit[packed.state_length:], whole[packed.state_length:], atol=1e-6, rtol=1e-6)  # a hit leaves zeros at the state


def test_packed_branch_training_step_matches_grouped_pieces(tmp_path, monkeypatch):
    """The plain path's branch levels as one varlen call per layer (Packed, the training and evaluate path) train to
    the same parameters as the grouped pieces (`branch_attention`, the serving path): one AdamW step over an
    accumulation batch with batch_states 3, weights equal to 1e-5."""
    from janus import hybrid
    from janus.training import TrainConfig, train
    torch.set_num_threads(2)
    for name, seed in (("train", 17), ("dev", 23)):
        write_jsonl(tmp_path / f"{name}.jsonl", [r.to_dict() for r in synthetic_requests(3, seed=seed)])
    weights = {}
    for packed in (True, False):
        monkeypatch.setattr(hybrid, "BRANCH_PACKED", packed)
        config = ModelConfig(backbone=TINY, backbone_family="qwen3_5", mode="tree", adaptation="lora", lora_rank=4, hidden_size=32,
                             layers=2, dtype="float32", attention="sdpa", max_tokens=1024, tree_positions="continue",
                             batch_states=3, gradient_checkpointing=True)
        train(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / f"run{int(packed)}",
              TrainConfig(model=config, epochs=1, accumulation=3, max_steps=1, device="cpu", seed=17, consistency_weight=.1))
        weights[packed] = torch.load(tmp_path / f"run{int(packed)}" / "best.pt", map_location="cpu", weights_only=False)["weights"]
    for key, value in weights[True].items():
        if value.is_floating_point():
            torch.testing.assert_close(value, weights[False][key], atol=1e-5, rtol=1e-5, msg=key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the flash varlen kernel is CUDA-only")
def test_varlen_flash_matches_dense_on_gpu():
    """`_varlen`'s flash kernel (bf16, GQA, three sequences of unequal query and key counts) equals the dense-mask
    fallback on the same inputs in fp32, forward and gradients, to bf16 rounding."""
    from janus.hybrid import _varlen
    torch.manual_seed(0)
    lq, lk = [5, 1, 7], [12, 9, 7]  # keys beyond the queries are the cached ancestors (the causal rule is bottom-right)
    cu = lambda lens: torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    q = torch.randn(sum(lq), 4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(sum(lk), 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(sum(lk), 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fast = _varlen(q, k, v, cu(lq), cu(lk), max(lq), max(lk), 0.25)
    assert fast.dtype == torch.bfloat16 and fast.shape == (sum(lq), 4, 16)
    fast.float().square().sum().backward()
    grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    q32, k32, v32 = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    slow = _varlen(q32, k32, v32, cu(lq), cu(lk), max(lq), max(lk), 0.25)
    slow.square().sum().backward()
    torch.testing.assert_close(fast.float(), slow, atol=2e-2, rtol=2e-2)
    for a, b in zip(grads, (q32.grad, k32.grad, v32.grad)):
        torch.testing.assert_close(a.float(), b, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="records GPU peak memory and step time")
@pytest.mark.parametrize("batch_states", [1, 4])
def test_packed_branch_peak_memory_and_step_time_on_gpu(batch_states, monkeypatch, capsys):
    """One checkpointed training step (forward, backward) on a synthetic batch of 8k-token packs, bf16 tiny hybrid:
    the packed path's peak memory must not exceed the grouped pieces' (whose fp32 score matrices scale with the
    batch); both peaks and step times are printed (`-s`)."""
    from janus import hybrid
    long = example()
    long["state"] = ("The card on the table is red and has a gold border. " * 160)[:7700]
    packs = [packed_tree(long, max_tokens=8192, max_state_plus_question=8192)] * batch_states
    assert packs[0].token_count > 7800
    torch.manual_seed(0)
    bb = HybridBackbone(HybridBackbone._tiny(64, 4, 16384), ByteTokenizer()).cuda().to(torch.bfloat16)
    bb.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    bb.train()
    peaks, times = {}, {}
    for packed in (False, True):
        monkeypatch.setattr(hybrid, "BRANCH_PACKED", packed)
        for repeat in range(2):  # the second run is timed
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            sum(h.float().square().mean() for h in bb.encode_many(packs)).backward()
            end.record()
            torch.cuda.synchronize()
        bb.zero_grad(set_to_none=True)
        peaks[packed], times[packed] = torch.cuda.max_memory_allocated() / 2 ** 20, start.elapsed_time(end)
    with capsys.disabled():
        print(f"\nbatch_states {batch_states}: grouped pieces {peaks[False]:.0f} MB {times[False]:.0f} ms; "
              f"packed {peaks[True]:.0f} MB {times[True]:.0f} ms")
    assert peaks[True] <= peaks[False]


def test_fused_lora_forward_matches_peft():
    """`_lora_forward` (the training glue's LoRA projection) computes what peft's LoRA Linear does, bf16 base and
    fp32 adapters, on the same input."""
    from janus.hybrid import Fused, _lora_forward, fuse_lora
    torch.manual_seed(0)
    bb = backbone("sdpa").to(torch.bfloat16)
    bb.apply_lora(4)
    module = bb.text_model.layers[1].self_attn.q_proj
    assert module.lora_A["default"].weight.dtype == torch.float32
    with torch.no_grad():
        module.lora_B["default"].weight.normal_()
    x = torch.randn(2, 5, bb.config.hidden_size, dtype=torch.bfloat16)
    reference = module(x)
    torch.testing.assert_close(_lora_forward(module, Fused(compile=False), x), reference, atol=0, rtol=0)
    fuse_lora(bb.lm, Fused(compile=False))
    assert isinstance(module.forward, functools.partial)
    torch.testing.assert_close(module(x), reference, atol=0, rtol=0)
