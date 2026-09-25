"""Serving fast paths (janus.hybrid.LevelGraphs, janus.server.Worker.probabilities, janus.latency): bucketing, padded
level batches equal the plain path, graphed levels equal eager on CUDA, and the equality report."""

import os
from pathlib import Path

import pytest
import torch

from janus.hybrid import BATCH_BUCKETS, KV_MAX, HybridBackbone, LevelGraphs, batch_bucket, chunk_rule, length_bucket
from janus.latency import compare
from janus.packing import ByteTokenizer, pack_request
from janus.schema import Request


def example():
    return {"state": "A red card on the table.", "questions": {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"r": "red", "b": "blue", "g": "green"}},
        "red": {"type": "noul", "instructions": "Is it red?"},
        "intensity": {"type": "score", "instructions": "Intensity?", "criteria": ["low", "medium", "high"]}}}


def backbone(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    return HybridBackbone(HybridBackbone._tiny(64, 4, 512, "sdpa"), ByteTokenizer(), "sdpa").to(device=device, dtype=dtype).eval()


def packed(raw=None):
    return pack_request(Request.from_dict(raw or example()), ByteTokenizer(), "tree", tree_positions="continue")


def test_buckets():
    assert [length_bucket(n) for n in (1, 8, 9, 33, 64, 65, 300, 1024, 1025, 2048)] == [4, 8, 12, 40, 64, 80, 320, 1024, 1280, 2048]
    assert [length_bucket(n, state=True) for n in (1, 64, 129, 1025, 2049, 16001)] == [64, 64, 192, 1280, 3072, 16384]
    assert length_bucket(2049) is None
    assert [batch_bucket(n) for n in (1, 7, 32)] == [1, 8, 32] and batch_bucket(33) is None
    assert BATCH_BUCKETS[-1] * 0 + KV_MAX == 2048


class _NoGraphs:
    """Stands in for LevelGraphs on CPU: the padded batches run eagerly."""

    def run(self, *args):
        return None

    def ensure(self, *args):
        return False  # no buffer set either: an eager level 0 is copied into fresh tensors

    def begin(self):
        pass

    def widen(self, states, key_valid):
        return states, key_valid


def test_bucketed_padding_matches_plain_path():
    bb = backbone()
    plain = bb.encode(packed())
    bb.graphs = _NoGraphs()
    with torch.no_grad():
        padded = bb.encode(packed())
    assert torch.allclose(plain, padded, atol=1e-4)
    with torch.enable_grad():  # gradients enabled: the plain path, whatever `graphs` is
        assert torch.allclose(bb.encode(packed()), plain, atol=1e-4)


def test_single_question_state_and_block_in_one_level_match_three_levels():
    """single_pass (enable_fast): a one-question pack runs state + block as one level; same numbers as three levels,
    alone, bucket-padded, and batched beside a three-question pack (which keeps its three levels)."""
    bb = backbone()
    single = packed({"state": "A red card.", "questions": {"color": example()["questions"]["color"]}})
    with torch.no_grad():
        three = bb.encode_many([single, packed()])
        runs = []
        for graphs in (None, _NoGraphs()):
            bb.graphs, bb.single_pass = graphs, True
            runs += [bb.encode(single), bb.encode_many([single, packed()])]
    for out, ref in zip([runs[0], *runs[1], runs[2], *runs[3]], [three[0], *three] * 2):
        assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_graphed_levels_match_eager_on_cuda():
    bb = backbone("cuda")
    request = packed()
    with torch.inference_mode():
        eager = bb.encode(request)
        bb.graphs = LevelGraphs(bb.text_model, chunk_rule("cuda", bb.allow_fla))
        first = bb.encode(request)  # captures level 0, the block level and the leaf level
        second = bb.encode(request)  # replays them
        keys = set(bb.graphs.graphs)
        third = bb.encode(request)
    assert set(bb.graphs.graphs) == keys and len(keys) == 3
    for out in (first, second, third):
        assert torch.allclose(eager, out, atol=1e-3), (eager - out).abs().max()  # fla chunk kernel over the padded length
    # One question: the block level reads and writes the same buffer set (batch 1 from batch 1), the case where the
    # capture's warm-up used to overwrite its own inputs.
    single = packed({"state": "A red card.", "questions": {"color": example()["questions"]["color"]}})
    with torch.inference_mode():
        bb.graphs = None
        eager = bb.encode(single)
        bb.graphs = LevelGraphs(bb.text_model, chunk_rule("cuda", bb.allow_fla))
        for _ in range(2):
            assert torch.allclose(eager, bb.encode(single), atol=1e-3)


def test_compare_counts_flips_and_max_difference():
    plain = [[[0.6, 0.4], [0.2, 0.3, 0.5]]]
    fast = [[[0.45, 0.55], [0.2, 0.31, 0.49]]]
    report = compare(plain, fast)
    assert report == {"requests": 1, "questions": 2, "max_abs_prob_diff": pytest.approx(0.15), "argmax_flips": 1}


def test_worker_probabilities_single_copy():
    from janus.server import Worker
    worker = Worker.__new__(Worker)
    worker.temperature = 2.
    logits = [torch.tensor([0., 2.]), torch.tensor([1., 1., 1.])]
    expected = [(z / 2).softmax(-1).tolist() for z in logits]
    assert worker.probabilities(logits) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_graphed_levels_with_prefix_snapshots_on_cuda():
    """A snapshot (the level-0 rows copied out of the graph buffers) fed back under graphs: the branch levels replay
    their graphs and match the eager path, alone and mixed with a miss in one batch."""
    bb = backbone("cuda")
    request = packed()
    other = packed({"state": "A blue card, larger than the red one.", "questions": {"red": example()["questions"]["red"]}})
    with torch.inference_mode():
        eager, eager_other = bb.encode_many([request, other])
        bb.graphs = LevelGraphs(bb.text_model, chunk_rule("cuda", bb.allow_fla))
        snapshot = bb.encode_prefix(request)
        s = request.state_length
        for _ in range(2):
            assert torch.allclose(eager[s:], bb.encode(request, prefix=snapshot)[s:], atol=1e-3)
        keys = set(bb.graphs.graphs)
        assert len(keys) == 3
        mixed = bb.encode_many([request, other], prefixes=[snapshot, None])
        assert torch.allclose(eager[s:], mixed[0][s:], atol=1e-3) and torch.allclose(eager_other, mixed[1], atol=1e-3)
        assert set(bb.graphs.graphs) > keys  # the two-pack level shapes captured graphs of their own
        again = bb.encode_many([request, other], prefixes=[snapshot, None])
        assert torch.allclose(mixed[0], again[0], atol=1e-3) and torch.allclose(mixed[1], again[1], atol=1e-3)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="a second CUDA device")
def test_graphs_capture_on_the_model_device_not_the_current_one():
    """The RTX 3090 serving bug: with the model on cuda:1 and the current device left at cuda:0 (nothing sets it,
    as in `janus serve`) the capture used to fail with "operation not permitted when stream is capturing". A fresh
    process, as a server is: a process that has already run graphs on cuda:0 cannot capture on cuda:1 either way."""
    import subprocess
    import sys
    code = """
import sys
import torch
sys.path.insert(0, "tests")
from test_latency import LevelGraphs, backbone, chunk_rule, packed
bb = backbone("cuda:1")
request = packed()
with torch.inference_mode():
    eager = bb.encode(request)
    bb.graphs = LevelGraphs(bb.text_model, chunk_rule("cuda:1", bb.allow_fla))
    for _ in range(2):
        assert torch.allclose(eager, bb.encode(request), atol=1e-3)
assert len(bb.graphs.graphs) == 3 and not bb.graphs.disabled, bb.graphs.graphs.keys()
assert torch.cuda.current_device() == 0
print("captured")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, cwd=str(Path(__file__).resolve().parents[1]),
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    assert result.returncode == 0 and "captured" in result.stdout, result.stderr[-2000:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_failed_capture_falls_back_to_the_plain_path_with_a_warning(caplog, monkeypatch):
    bb = backbone("cuda")
    request = packed()
    with torch.inference_mode():
        eager = bb.encode(request)
        bb.graphs = LevelGraphs(bb.text_model, chunk_rule("cuda", bb.allow_fla))
        monkeypatch.setattr(LevelGraphs, "_capture", lambda self, key, inputs, states: (_ for _ in ()).throw(RuntimeError("operation not permitted when stream is capturing")))
        with caplog.at_level("WARNING", logger="janus.hybrid"):
            first, second = bb.encode(request), bb.encode(request)
    assert bb.graphs.disabled and not bb.graphs.graphs
    assert torch.allclose(eager, first, atol=1e-3) and torch.allclose(eager, second, atol=1e-3)
    assert sum("plain path" in r.message for r in caplog.records) == 1


def test_fused_glue_matches_plain_path():
    """The fused serving glue (Fused: token-major conv taps, fused norms, gates and MLP activation) run eagerly on
    CPU equals the plain path; the flex-free branch and state attention go through the same functions."""
    from janus.hybrid import Fused
    bb = backbone()
    request = packed()
    with torch.inference_mode():
        plain = bb.encode(request)
        bb.fused = Fused(compile=False)
        fused = bb.encode(request)
        bb.state_chunk_tokens = 8  # several state chunks: the cached-keys (masked) branch of state_attention
        chunked = bb.encode(request)
    assert torch.allclose(plain, fused, atol=1e-4), (plain - fused).abs().max()
    assert torch.allclose(plain, chunked, atol=1e-4), (plain - chunked).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile on CUDA")
def test_compiled_fused_glue_matches_plain_path_under_graphs_on_cuda():
    from janus.hybrid import Fused
    bb = backbone("cuda")
    request = packed()
    with torch.inference_mode():
        plain = bb.encode(request)
        bb.fused = Fused()
        compiled = bb.encode(request)
        bb.graphs = LevelGraphs(bb.text_model, chunk_rule("cuda", bb.allow_fla), bb.fused)
        first, second = bb.encode(request), bb.encode(request)
    for out in (compiled, first, second):
        assert torch.allclose(plain, out, atol=1e-3), (plain - out).abs().max()
    assert len(bb.graphs.graphs) == 3


def test_serving_level_matches_plain_path():
    """The serving level (Fused(serve=True), `_serve_level`: residual adds folded into the next norm, merged q/k/v,
    DeltaNet and gate/up projections, l2-normalised q/k out of `_delta_serve`, compiled branch pieces) run eagerly
    on CPU equals the plain path; the merge leaves the modules' weights as views, so the plain path is unchanged."""
    from janus.hybrid import Fused, _merge_projections
    bb = backbone()
    request = packed()
    with torch.inference_mode():
        plain = bb.encode(request)
        _merge_projections(bb.text_model)
        assert all(hasattr(layer.mlp, "serve_gate_up") for layer in bb.text_model.layers)
        again = bb.encode(request)
        bb.fused = Fused(compile=False, serve=True)
        served = bb.encode(request)
    assert torch.equal(plain, again)
    assert torch.allclose(plain, served, atol=1e-4), (plain - served).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile and fla on CUDA")
def test_compiled_serving_level_matches_plain_path_under_graphs_on_cuda():
    """enable_fast's serving level compiled and graphed (the short leaves through fla's fused recurrent kernel)
    equals the plain path."""
    bb = backbone("cuda")
    request = packed()
    with torch.inference_mode():
        plain = bb.encode(request)
        bb.enable_fast()
        first, second = bb.encode(request), bb.encode(request)
    for out in (first, second):
        assert torch.allclose(plain, out, atol=1e-3), (plain - out).abs().max()
    assert bb.fused.serve and len(bb.graphs.graphs) >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 scaled_mm on CUDA")
def test_fp8_linears_run_under_graphs_and_stay_close_on_cuda():
    """FP8Linear (e4m3 weights, per-token activation scales) replaces the projections, captures and replays, and
    stays within e4m3 error of the bf16 path; the head-sized b/a projections are left alone."""
    from janus.hybrid import FP8Linear, Fused
    bb = backbone("cuda", torch.bfloat16)
    request = packed()
    with torch.inference_mode():
        plain = bb.encode(request).float()
        bb.enable_fast(fp8=True)
        assert isinstance(bb.text_model.layers[0].linear_attn.in_proj_qkv, FP8Linear) and isinstance(bb.fused, Fused)
        assert not isinstance(bb.text_model.layers[0].linear_attn.in_proj_b, FP8Linear)
        first, second = bb.encode(request).float(), bb.encode(request).float()
    assert torch.equal(first, second) or torch.allclose(first, second, atol=1e-2)
    assert (plain - first).abs().max() < 0.25 * plain.abs().max(), (plain - first).abs().max()  # e4m3: 3 mantissa bits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs and torch.compile")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_graphs_capture_again_after_every_graph_was_evicted_on_cuda():
    """Evicting every buffer set destroys every graph of the shared pool, which retires the pool in the allocator;
    the next capture must go into a fresh pool (before: an allocator assert disabled the graphs for the process,
    which is how the production server ended on the plain path after its first wide Score request)."""
    bb = backbone("cuda")
    request = packed()
    with torch.inference_mode():
        plain = bb.encode(request)
        bb.enable_fast()
        fast = bb.encode(request)
        graphs = bb.graphs
        assert graphs.graphs and not graphs.disabled
        pool = graphs.pool
        for key in list(graphs.outs):
            graphs.evict(key)
        assert not graphs.graphs and graphs.pool != pool
        again = bb.encode(request)
    assert graphs.graphs and not graphs.disabled
    torch.testing.assert_close(again, fast, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(again, plain, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_adversarial_long_states_padding_cache_and_eviction_on_cuda():
    """The reviewer's adversarial script: a tiny fp32 hybrid under enable_fast() against the plain path at every
    branch position for right-padded unequal-state batches in both orders, states above KV_MAX sharing a capacity
    bucket (singly, batched, with a stale row), prefix-cache hits behind long states, STATE_BUFFER_TOKENS exceeded,
    the chunked masked path with all-padding rows, buffer-set eviction under a tiny budget (the graphs on the evicted
    sets go with them), re-enable, and the gradient-enabled path staying plain."""
    import janus.hybrid as hybrid
    torch.manual_seed(0)
    bb = HybridBackbone(HybridBackbone._tiny(64, 4, 8192, "sdpa"), ByteTokenizer(), "sdpa").to(device="cuda", dtype=torch.float32).eval()

    def pk(n, seed):
        g = torch.Generator().manual_seed(seed)
        state = "".join(chr(97 + int(i)) for i in torch.randint(0, 26, (n,), generator=g))
        body = {"state": state, "questions": {"color": example()["questions"]["color"], "red": example()["questions"]["red"]}}
        return pack_request(Request.from_dict(body), ByteTokenizer(), "tree", max_tokens=8192, tree_positions="continue")

    packs = {"S1": pk(200, 1), "S2": pk(37, 2), "L1": pk(3000, 3), "L2": pk(2500, 4), "L3": pk(2600, 5)}
    with torch.inference_mode():
        plain = {k: bb.encode(v) for k, v in packs.items()}

    def check(tag, names, out, tol=2e-3):
        for name, o in zip(names, out):
            branch = packs[name].segment_ids[:packs[name].token_count].to(o.device) != 0
            worst = (plain[name] - o).abs()[branch].max().item()
            assert worst < tol and not torch.isnan(o).any(), (tag, name, worst)

    bb.enable_fast()
    assert bb.fused is not None and bb.graphs is not None and bb.fast_causal
    with torch.inference_mode():
        check("S1,S2", ["S1", "S2"], bb.encode_many([packs["S1"], packs["S2"]]))
        check("S2,S1", ["S2", "S1"], bb.encode_many([packs["S2"], packs["S1"]]))
        check("L1", ["L1"], [bb.encode(packs["L1"])])
        check("L2", ["L2"], [bb.encode(packs["L2"])])
        check("L3 stale keys from L1", ["L3"], [bb.encode(packs["L3"])])
        check("L1,L2", ["L1", "L2"], bb.encode_many([packs["L1"], packs["L2"]]))
        check("L2,S1", ["L2", "S1"], bb.encode_many([packs["L2"], packs["S1"]]))
        check("S1,L2", ["S1", "L2"], bb.encode_many([packs["S1"], packs["L2"]]))
        check("L3,L2 stale row", ["L3", "L2"], bb.encode_many([packs["L3"], packs["L2"]]))
        snap = bb.encode_prefixes([packs["L1"], packs["L2"]])
        check("hit L1", ["L1"], bb.encode_many([packs["L1"]], prefixes=[snap[0]]))
        check("hit L1, miss L3", ["L1", "L3"], bb.encode_many([packs["L1"], packs["L3"]], prefixes=[snap[0], None]))
        check("miss L3, hit L2", ["L3", "L2"], bb.encode_many([packs["L3"], packs["L2"]], prefixes=[None, snap[1]]))
        check("miss S1, hit L1", ["S1", "L1"], bb.encode_many([packs["S1"], packs["L1"]], prefixes=[None, snap[0]]))
        old = hybrid.STATE_BUFFER_TOKENS
        hybrid.STATE_BUFFER_TOKENS = 1000
        check("STATE_BUFFER_TOKENS exceeded", ["L1"], [bb.encode(packs["L1"])])
        hybrid.STATE_BUFFER_TOKENS = old
        bb.state_chunk_tokens = 1024
        check("chunked L1", ["L1"], [bb.encode(packs["L1"])])
        check("chunked L1,S2", ["L1", "S2"], bb.encode_many([packs["L1"], packs["S2"]]))
        check("chunked S2,L1", ["S2", "L1"], bb.encode_many([packs["S2"], packs["L1"]]))
        bb.state_chunk_tokens = 65536
        # A budget of exactly what is allocated: a batch needing a new long-state set evicts least recently used
        # sets not touched by the request, and the graphs keyed on them; answers unchanged.
        graphs = bb.graphs
        graphs.buffer_bytes = graphs.bytes
        before = set(graphs.outs)
        assert (0, 3, 4096) not in before
        check("L1,L2,L3 evicts", ["L1", "L2", "L3"], bb.encode_many([packs["L1"], packs["L2"], packs["L3"]]))
        assert (0, 3, 4096) in graphs.outs and len(graphs.outs) < len(before) and graphs.bytes <= graphs.buffer_bytes
        assert all(s in graphs.outs for k in graphs.graphs for s in graphs._sets(k) if s[0] == 0)  # no graph on an evicted set
        assert graphs.bytes == sum(graphs.sizes.values()) == sum(t.numel() * t.element_size() for pair in graphs.outs.values() for a, b in pair for t in (a, b))
        check("L1 again after eviction", ["L1"], [bb.encode(packs["L1"])])
        check("S1,S2 again after eviction", ["S1", "S2"], bb.encode_many([packs["S1"], packs["S2"]]))
        bb.graphs.buffer_bytes = 0
        check("no budget at all: eager branches", ["L3"], [bb.encode(packs["L3"])])
        bb.enable_fast()
        check("after re-enable", ["L3"], [bb.encode(packs["L3"])])
    assert torch.allclose(bb.encode(packs["S1"]), plain["S1"], atol=1e-4)  # gradients enabled: the plain path, masked
