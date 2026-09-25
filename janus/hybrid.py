"""Qwen3.5 hybrid backbone (Gated DeltaNet + full attention) behind the packed tree layout.

The packed layout (janus.packing) puts a shared state prefix and several branch segments into one sequence and isolates
siblings with a per-token attention mask. Qwen3.5-*-Base has 24 Gated DeltaNet layers, each a recurrence over the
sequence, so under the packed layout every branch would read the conv/recurrent state left behind by its siblings; no
mask can undo that. This module runs the packed request as a tree of batched passes instead of one sequence:

  level 0   the state prefixes (segment 0 of every pack in the call) as one right-padded batch, run in chunks of
            `state_chunk_tokens` along the sequence, producing per-layer states: key/value tensors for the attention
            layers (concatenated across chunks), and the conv tail (the last kernel-1 conv inputs) plus the recurrent
            matrix for the DeltaNet layers (carried across chunks);
  level d   every segment of every pack whose parent is at level d-1, as ONE right-padded batch. The DeltaNet layers
            continue each row from its parent row's states (index_select along the batch dimension, the pattern of
            `Qwen3_5DynamicCache.reorder_cache`: a conv tail and a recurrent matrix per row, small). The attention
            layers never copy an ancestor's keys per row: each row attends to the keys of its ancestor rows in place
            (`branch_attention`: the queries grouped by ancestor row, one piece per ancestor level and one for the
            row's own tokens, merged through their log-sum-exps, which is the exact softmax over the union), so a
            level keeps only its own tokens' keys and values for the levels below it, and a 255-leaf request or a
            64k state costs one copy of the state's keys, not one per row. That is the serving path; the plain path
            of training and evaluate (a few rows per state) gathers every row's ancestors and runs the level as one
            varlen flash call per layer instead (`packed_attention`, docs/phase4/training-throughput.md).

In tree mode the levels are: state, then all question blocks, then all leaves. Each leaf therefore sees exactly the
state, its own block and itself, i.e. the causal packed mask restricted to one root-to-leaf path. Hidden states after
the final norm are scattered back to the packed positions, so `DecisionModel.forward` reads the option/decision
positions unchanged. `encode_prefixes` returns the level-0 states of packs as PrefixSnapshots and
`encode_many(packs, prefixes=...)` continues from them (the server's state cache): level 0 runs only for the packs
without one, and the rows are assembled in pack order before the branch levels.

Right padding is exact: padded steps are no-ops on the DeltaNet recurrent state (their decay `g` and write gate `beta`
are zeroed, so the step is the identity in both the chunked and the recurrent form), the conv tail is read at each
row's real end, and the attention layers mask padded keys and take explicit positions. The same argument makes a
state that ended in an earlier chunk (a row of pure padding) a no-op, and it is why the state batch is padded on the
right like the branch levels rather than on the left (a left-padded query row would see no key at all). Every state tensor is an ordinary autograd tensor (no HF
cache object, nothing detached), so gradients from the leaf outputs reach the block pass and the state pass. The HF
`Qwen3_5GatedDeltaNet.forward` uses its cache only for single-token decoding (`seq_len == 1`) and otherwise silently
starts from a zero state, so the branch passes do not go through it; the layer's projections, conv weights, gated norm
and output projection are reused as modules and only the sequence mixing is re-expressed here with `initial_state`.
Nothing depends on eval mode (Qwen has no dropout).

Kernels: on CUDA the flash-linear-attention chunk kernel is used when importable, otherwise (and always on CPU) the
transformers torch fallback `torch_chunk_gated_delta_rule`. The gated RMSNorm is always the torch module (fla's fused
version is constructed on the current CUDA device inside the HF layer constructor, which breaks CPU use), and the
causal conv is `F.conv1d` (causal-conv1d is not needed).

Backbone families: `qwen3_5` (Qwen3.5 dense) and `qwen3_5_moe` (Qwen3.6-35B-A3B: the same DeltaNet and gated-attention
sublayers, a sparse MoE feed-forward in place of the MLP, run through the HF sparse block; NVIDIA's ModelOpt NVFP4
checkpoint loads through `load_nvfp4` with the experts kept packed, `NVFP4Experts` / `NVFP4Linear`, in a dequantise-per-call
W4A16 mode or, on Blackwell, the fp4 tensor-core mode `enable_fast(fp4=True)`; docs/phase4/a3b-runner-port.md).

Supported: modes tree, listwise and independent with any tree_block / tree_positions (positions are read from the
packed request and only the attention layers see them); directionality causal only; attention eager, sdpa or flex (at
level 0 the HF attention forward under the level mask, or without one in bf16/fp16 on CUDA where sdpa is the flash
kernel, a BlockMask from the padding pattern under flex, `flex_mask`; at the branch levels, on the plain path of
training and evaluate, one varlen flash call per layer over every row's ancestors and itself, `packed_attention`,
with a dense-mask fallback off CUDA; on the serving path the grouped torch pieces of `_piece`, or flex pieces with
their log-sum-exp under flex); LoRA
through peft on the attention and DeltaNet projections; gradient
checkpointing per layer and level (non-reentrant); several states per call (`encode_many`, reached from
`DecisionModel.forward_many`). Not supported: decoder mode (its branch packs have no state segment), block
directionality (a recurrence is causal).
"""

import functools
import logging
import math
import os
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention
from transformers import AttentionInterface
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5_modeling
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen3_5_moe_modeling
from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5RMSNormGated, Qwen3_5TextConfig, Qwen3_5TextModel,
                                                          torch_chunk_gated_delta_rule)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNormGated, Qwen3_5MoeTextConfig, Qwen3_5MoeTextModel

# fla's FusedRMSNormGated is created on torch.cuda.current_device() inside Qwen3_5GatedDeltaNet.__init__; the torch
# module computes the same function on any device, so every layer built after this import uses it.
qwen3_5_modeling.FusedRMSNormGated = None
qwen3_5_moe_modeling.FusedRMSNormGated = None

# The MoE family (Qwen3.6-35B-A3B, `qwen3_5_moe`): the same DeltaNet and gated-attention sublayers, a sparse MoE feed-forward
# (router, top-k, 256 experts as fused [E, 2I, H] / [E, H, I] parameters, a gated shared expert). LoRA targets the
# projections only (never the experts: peft against transformers 5's fused expert parameters breaks), so LORA_TARGETS is shared.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
log = logging.getLogger("janus.hybrid")
TINY = "tiny-qwen3_5"
TINY_MOE = "tiny-qwen3_5_moe"
TEXT_MODELS = (Qwen3_5TextModel, Qwen3_5MoeTextModel)
TINY_VOCAB = 257
HF_ATTENTION = {"eager": "eager", "sdpa": "sdpa", "flex": "jev_flex"}
TINY_VL_VOCAB = 261  # bytes + 1, then the video, image, vision_start and vision_end placeholders (janus.vision)


def _fla_chunk_rule():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except Exception:
        return None
    return chunk_gated_delta_rule


FLA_CHUNK_RULE = _fla_chunk_rule()
FLA_RECURRENT_RULE = None
if FLA_CHUNK_RULE is not None:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as FLA_RECURRENT_RULE
# Serving (`_serve_delta`): rows of at most RECURRENT_TOKENS take fla's one-kernel recurrent form (the short leaves: 11 us
# against 21 for the chunked form's seven kernels at 4 rows x 12 tokens on the RTX 5090), levels of at most CHUNK32_TOKENS
# tokens take 32-token chunks (a 192-token row: 25 us against 33 at 64; beyond ~1k tokens 64 is faster again). The
# forms differ from the 64-token chunks by fp32 accumulation order only (docs/phase4/latency-track3.md).
# ponytail: both thresholds were measured on the RTX 5090 only; a per-card table is the upgrade if the 3090 disagrees.
RECURRENT_TOKENS = 16
CHUNK32_TOKENS = 1024


def chunk_rule(device, allow_fla=True):
    """The chunked gated delta rule for tensors on `device`: fla's Triton kernel on CUDA when installed, else torch."""
    if allow_fla and FLA_CHUNK_RULE is not None and torch.device(device).type == "cuda":
        return FLA_CHUNK_RULE
    return torch_chunk_gated_delta_rule


def tree_levels(segment_ids, parents):
    """Contiguous segments grouped by depth: level 0 is [(0, start, end)], level d the children of level d-1.

    Raises ValueError when a segment is empty or not contiguous, or when a parent index is not smaller than its child."""
    segment_ids, parents = segment_ids.cpu(), parents.cpu()
    count = parents.shape[0]
    spans, depth = [], []
    for s in range(count):
        where = (segment_ids == s).nonzero()[:, 0]
        if where.numel() == 0:
            raise ValueError(f"segment {s} has no tokens")
        start, end = int(where[0]), int(where[-1]) + 1
        if end - start != where.numel():
            raise ValueError(f"segment {s} is not contiguous")
        parent = int(parents[s])
        if s == 0:
            depth.append(0)
        elif parent >= s:
            raise ValueError("a segment's parent must precede it")
        else:
            depth.append(depth[parent] + 1)
        spans.append((s, start, end))
    levels = [[] for _ in range(max(depth) + 1)]
    for span, d in zip(spans, depth):
        levels[d].append(span)
    return levels


class _KV:
    """The two-method cache Qwen3_5Attention.forward needs: concatenate along the sequence and hand the result back."""

    def __init__(self, key=None, value=None):
        self.key, self.value = key, value

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.key is not None:
            key_states = torch.cat([self.key, key_states], dim=2)
            value_states = torch.cat([self.value, value_states], dim=2)
        self.key, self.value = key_states, value_states
        return key_states, value_states


_COMPILED_FLEX = []


def _flex_forward(module, query, key, value, attention_mask, scaling=None, lse=False, **kwargs):
    """Attention interface for the level batches: torch's flex_attention on a BlockMask (`flex_mask`). Compiled once on
    CUDA (the fused kernel); plain on CPU, where the Inductor C++ kernel does not build for these masks (tests).
    With `lse` (the branch pieces) returns (output [B, H, L, D], log-sum-exp) instead of the HF interface's pair."""
    if query.device.type == "cpu":
        fn = flex_attention
    else:
        if not _COMPILED_FLEX:
            _COMPILED_FLEX.append(torch.compile(flex_attention))
        fn = _COMPILED_FLEX[0]
    # The default tiles at head_dim 256 (Qwen3.5) exceed the 100 KB shared memory of sm_86 and sm_120; 32x32 compiles
    # there and ran 1.6x faster than sdpa at a quarter of its memory on the RTX 3090 (docs/phase4/runner-throughput.md).
    options = {k: 32 for k in ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")} if query.shape[-1] > 128 else None
    out = fn(query, key, value, block_mask=attention_mask, scale=scaling, enable_gqa=True, kernel_options=options,
             return_aux=AuxRequest(lse=True) if lse else None)
    return (out[0], out[1].lse) if lse else (out.transpose(1, 2).contiguous(), None)


AttentionInterface.register(HF_ATTENTION["flex"], _flex_forward)


def _ordered(dense):
    """Per row of a [..., KV] block table: the count of True entries and their indices first (BlockMask's layout)."""
    count = dense.sum(-1, dtype=torch.int32)
    return count, dense.to(torch.int32).argsort(dim=-1, descending=True, stable=True).to(torch.int32)


def flex_mask(kv_valid, cached, length, block=128):
    """BlockMask [B, 1, length, KV] of one level batch: key kv is visible to query q when it is valid and
    kv <= cached + q (the rule of the dense sdpa/eager masks). The block table comes from the padding pattern alone
    (as janus.attention.build_block_mask does for the packed layout), so the dense mask is never materialised."""
    batch, kv_len = kv_valid.shape
    q_blocks, kv_blocks = -(-length // block), -(-kv_len // block)
    padded = F.pad(kv_valid, (0, kv_blocks * block - kv_len)).view(batch, kv_blocks, block)
    i = torch.arange(q_blocks, device=kv_valid.device)[:, None]
    j = torch.arange(kv_blocks, device=kv_valid.device)[None]
    full = ((j + 1) * block - 1 <= cached + i * block)[None] & padded.all(-1)[:, None]
    partial = (j * block <= cached + (i + 1) * block - 1)[None] & padded.any(-1)[:, None] & ~full

    def mask_mod(b, h, q, kv):
        return kv_valid[b, kv.clamp(max=kv_len - 1)] & (kv <= cached + q) & (kv < kv_len)
    kv_num, kv_idx = _ordered(partial)
    full_num, full_idx = _ordered(full)
    return BlockMask.from_kv_blocks(kv_num[:, None], kv_idx[:, None], full_num[:, None], full_idx[:, None],
                                    BLOCK_SIZE=block, mask_mod=mask_mod, seq_lengths=(length, kv_len))


def _delta_mix(mixed, b, a, valid, tail, conv_w, conv_b, A_log, dt_bias, key_dim, head_k_dim, head_v_dim):
    """The glue between a DeltaNet layer's projections and the chunk rule, token-major: the causal conv as K taps over
    the tail-extended sequence (accumulated in fp32, rounded once), silu, the head split, the write gate and the decay.
    mixed [B, L, C] (in_proj_qkv output), b/a [B, L, HV], valid [B, L], tail [B, C, K-1]. Returns (q [B, L, H, Dk],
    k, v [B, L, HV, Dv], g, beta [B, L, HV] fp32/bf16 as the plain path, new tail [B, C, K-1])."""
    B, L, C = mixed.shape
    K = conv_w.shape[-1]
    ext = torch.cat([tail.to(mixed.dtype).transpose(1, 2), mixed], dim=1)  # [B, K-1+L, C]
    index = valid.sum(1)[:, None] + torch.arange(K - 1, device=mixed.device)[None]
    new_tail = ext.gather(1, index[..., None].expand(-1, -1, C)).transpose(1, 2).contiguous()
    w = conv_w[:, 0].float()  # [C, K]
    conv = sum(ext[:, j:j + L].float() * w[:, j] for j in range(K))
    if conv_b is not None:
        conv = conv + conv_b.float()
    act = F.silu(conv).to(mixed.dtype)
    q, k, v = torch.split(act, [key_dim, key_dim, C - 2 * key_dim], dim=-1)
    beta = b.sigmoid() * valid.to(b.dtype)[..., None]
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias) * valid.float()[..., None]
    return q.reshape(B, L, -1, head_k_dim), k.reshape(B, L, -1, head_k_dim), v.reshape(B, L, -1, head_v_dim), g, beta, new_tail


def _gate_norm(core, z, weight, eps):
    """Qwen3_5RMSNormGated.forward over the last dim (the value head), same rounding points."""
    dtype = core.dtype
    h = core.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(dtype)
    return (h * F.silu(z.float())).to(dtype)


def _rmsnorm(x, weight, eps):
    """Qwen3_5RMSNorm.forward."""
    n = x.float()
    n = n * torch.rsqrt(n.pow(2).mean(-1, keepdim=True) + eps)
    return (n * (1.0 + weight.float())).type_as(x)


def _add_rmsnorm(residual, h, weight, eps):
    """(residual + h, its Qwen3_5RMSNorm): the residual add and the next norm in one kernel."""
    x = residual + h
    return x, _rmsnorm(x, weight, eps)


def _silu_mul(gate, up):
    return F.silu(gate) * up


def _add_norm(residual, h, weight, eps, valid):
    """Serving (`_serve_level`): (residual + h, its Qwen3_5RMSNorm) with h None meaning no add, the norm's right-padded
    positions zeroed when `valid` is given (gated_delta_net's masked_fill of its input): the previous layer's residual
    add, the next norm and the DeltaNet input mask in one kernel."""
    x = residual if h is None else residual + h
    n = _rmsnorm(x, weight, eps)
    return x, n if valid is None else torch.where(valid[..., None], n, 0)


def _delta_serve(mixed, b, a, valid, ends, tail, conv_w, conv_b, A_log, dt_bias, key_dim, head_k_dim, head_v_dim):
    """Serving `_delta_mix` (same numbers): a missing tail is zeros, the row ends `ends` (valid.sum(1)) come from the
    level, q and k come out l2-normalised the way fla's l2norm_fwd does it (fp32 from the rounded bf16 value, eps
    1e-6: the rule then runs with use_qk_l2norm_in_kernel=False), and q, k, v each contiguous and computed from the
    conv taps on their own channels, so no full-width activation is written and the rule's input guard copies nothing."""
    B, L, C = mixed.shape
    K = conv_w.shape[-1]
    if tail is None:
        ext = F.pad(mixed, (0, 0, K - 1, 0))
    else:
        ext = torch.cat([tail.to(mixed.dtype).transpose(1, 2), mixed], dim=1)  # [B, K-1+L, C]
    index = ends[:, None] + torch.arange(K - 1, device=mixed.device)[None]
    new_tail = ext.gather(1, index[..., None].expand(-1, -1, C)).transpose(1, 2).contiguous()
    w = conv_w[:, 0].float()  # [C, K]

    def conv(lo, hi, head_dim):  # silu of the causal conv of channels lo:hi, accumulated in fp32, rounded once
        c = sum(ext[:, j:j + L, lo:hi].float() * w[lo:hi, j] for j in range(K))
        if conv_b is not None:
            c = c + conv_b[lo:hi].float()
        return F.silu(c).to(mixed.dtype).view(B, L, -1, head_dim)

    l2 = lambda t: (t.float() * torch.rsqrt(t.float().pow(2).sum(-1, keepdim=True) + 1e-6)).to(t.dtype)
    q, k, v = l2(conv(0, key_dim, head_k_dim)), l2(conv(key_dim, 2 * key_dim, head_k_dim)), conv(2 * key_dim, C, head_v_dim)
    beta = b.sigmoid() * valid.to(b.dtype)[..., None]
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias) * valid.float()[..., None]
    return q, k, v, g, beta, new_tail


def _attn_qk(qg, k, cos, sin, q_weight, k_weight, eps, head_dim):
    """Qwen3_5Attention's q/k path after the projections: the q/gate split, the per-head RMSNorms and rotary.
    qg [B, L, H, 2D] (q_proj output viewed), k [B, L, Hkv, D]. Returns (q [B, H, L, D], k [B, Hkv, L, D], gate [B, L, H*D])."""
    B, L = qg.shape[:2]
    q, gate = torch.chunk(qg, 2, dim=-1)
    q = _rmsnorm(q, q_weight, eps).transpose(1, 2)
    k = _rmsnorm(k, k_weight, eps).transpose(1, 2)
    q, k = qwen3_5_modeling.apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, gate.reshape(B, L, -1)


def _attn_serve(qg, k, cos, sin, q_weight, k_weight, eps):
    """Serving `_attn_qk` (same math, same outputs): rotary as one pointwise over the whole head (cos 1 and sin 0 past
    the rotary dims, the rotate-half partner read by index) instead of HF's concatenations, so q and k are one
    normalise-and-rotate kernel each rather than three."""
    B, L, _, D2 = qg.shape
    D, r = D2 // 2, cos.shape[-1]
    idx = torch.arange(D, device=qg.device)
    partner = torch.where(idx < r // 2, idx + r // 2, torch.where(idx < r, idx - r // 2, idx))
    sign = torch.where(idx < r // 2, -1.0, 1.0)
    cos, sin = F.pad(cos, (0, D - r), value=1.0)[:, :, None], F.pad(sin, (0, D - r), value=0.0)[:, :, None]

    def rope(x, weight):  # [B, L, h, D] -> normed, rotated [B, h, L, D]
        x = _rmsnorm(x, weight, eps)
        return (x * cos + x[..., partner] * sign * sin).to(x.dtype).transpose(1, 2).contiguous()
    return rope(qg[..., :D], q_weight), rope(k, k_weight), qg[..., D:].reshape(B, L, -1)


def _lora(x, weight, bias, a, b, scaling):
    """peft's LoRA Linear forward with one adapter and no dropout, as it computes it: the base projection, the
    low-rank update in the adapter's dtype (fp32 under peft's autocast_adapter_dtype), summed in fp32, rounded once."""
    out = F.linear(x, weight, bias)
    return (out + F.linear(F.linear(x.to(a.dtype), a), b) * scaling).to(out.dtype)


def _lora_forward(module, fused, x):
    """The forward of a peft LoRA Linear through `fused.lora` (training glue): eight launches per projection
    (casts, two skinny fp32 GEMMs each with a split-K reduction, scale, add, cast back) become the GEMMs and one."""
    (adapter,) = module.active_adapters
    return fused.lora(x, module.base_layer.weight, module.base_layer.bias, module.lora_A[adapter].weight,
                      module.lora_B[adapter].weight, module.scaling[adapter])


def fuse_lora(lm, fused):
    """Route every plain peft LoRA Linear of `lm` (one active adapter, no dropout, no variant such as DoRA, not
    merged) through `_lora_forward`; the others keep peft's forward."""
    from peft.tuners.lora.layer import Linear as LoraLinear
    for module in lm.modules():
        if (isinstance(module, LoraLinear) and len(module.active_adapters) == 1 and not module.merged and not module.disable_adapters
                and not module.lora_variant and isinstance(module.lora_dropout[module.active_adapters[0]], nn.Identity)):
            module.forward = functools.partial(_lora_forward, module, fused)


def _gate_out(out, gate):
    """out [B, H, L, D] to [B, L, H*D] times sigmoid(gate)."""
    B, H, L, D = out.shape
    return out.transpose(1, 2).reshape(B, L, H * D).to(gate.dtype) * torch.sigmoid(gate)


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")


def _quantize_rows(x):
    """Per-row e4m3 quantisation of x [M, K]: (x8, scale [M, 1] fp32) with x ~= x8 * scale."""
    x = x.float()
    scale = (x.abs().amax(-1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    return (x / scale).to(torch.float8_e4m3fn), scale


class FP8Linear(nn.Module):
    """A bias-free nn.Linear as an e4m3 weight with per-output-channel scales, applied through torch._scaled_mm with
    per-token activation scales (`Fused.quantize`, one fused kernel): 2.5x the bf16 GEMM rate on the RTX 5090 and half
    the weight bytes. Changes the numbers beyond bf16 rounding (per-token e4m3 activations), so it is opt-in
    (`enable_fast(fp8=True)`, `janus serve --fp8`); the accuracy delta is in docs/phase4/latency-track2.md."""

    def __init__(self, linear, fused):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("FP8Linear takes a bias-free linear")
        w = linear.weight.detach().float()
        self.scale = nn.Parameter((w.abs().amax(1, keepdim=True) / FP8_MAX).clamp(min=1e-12), requires_grad=False)  # [N, 1]
        self.weight = nn.Parameter((w / self.scale).to(torch.float8_e4m3fn), requires_grad=False)
        self.fused = fused

    def forward(self, x):
        x8, scale = self.fused.quantize(x.reshape(-1, x.shape[-1]))
        out = torch._scaled_mm(x8, self.weight.t(), scale_a=scale, scale_b=self.scale.t(), out_dtype=x.dtype)
        return out.view(*x.shape[:-1], -1)


FP4_BLOCK = 16
BMM_CHUNK_TOKENS = 1024  # tokens per padded-bmm expert call (NVFP4Experts): bounds the [E, width, H] gather under skewed routing
FP4_MAX = 6.0  # e2m1's largest magnitude (codes 0..7 are 0, .5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign); a block's amax maps onto it


def _e2m1_decode(codes):
    """uint8 e2m1 codes (0..15) to fp32: bit 3 sign, bits 2-1 exponent, bit 0 mantissa."""
    m = (codes & 1).float()
    e = ((codes >> 1) & 3).float()
    v = torch.where(e == 0, 0.5 * m, (1 + 0.5 * m) * torch.exp2(e - 1))
    return torch.where((codes & 8) != 0, -v, v)


def _e2m1_encode(x):
    """fp32 to uint8 e2m1 codes, round to nearest with ties to even (ModelOpt's rule)."""
    mag = x.abs()
    code = torch.zeros_like(mag, dtype=torch.uint8)
    for threshold, value in ((0.25, 1), (0.75, 2), (1.25, 3), (1.75, 4), (2.5, 5), (3.5, 6), (5.0, 7)):
        code = torch.where(mag > threshold if value % 2 else mag >= threshold, torch.full_like(code, value), code)
    return code | ((x < 0).to(torch.uint8) << 3)


def _nvfp4_dequant(packed, scale, global_scale, dtype):
    """One piece of `nvfp4_dequantize`: packed [..., K/2] uint8, scale [..., K/16] e4m3, global_scale [..., 1] or scalar
    fp32. The element pair of a byte is (low nibble, high nibble) as ModelOpt packs it and torch's float4_e2m1fn_x2 reads it;
    the products are exact in fp32 (e2m1 times e4m3 has 5 significant bits), rounded once to `dtype`."""
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2)  # [..., K]
    values = _e2m1_decode(codes).view(*codes.shape[:-1], -1, FP4_BLOCK) * (scale.float() * global_scale.float())[..., None]
    return values.reshape(codes.shape).to(dtype)


def nvfp4_dequantize(packed, scale, global_scale, dtype=torch.bfloat16, fused=None, chunk_elements=64 << 20):
    """ModelOpt NVFP4 weight to `dtype`, in leading-dimension chunks whose fp32 transients stay near `chunk_elements`
    (an expert stack of the A3B is 805M weights per layer; the caching allocator holds the output only)."""
    if fused is not None:  # one fused kernel, no fp32 transients: the whole stack at once, no chunk copies
        return fused.dequant(packed, scale, global_scale, dtype)
    fn = _nvfp4_dequant
    rows = max(1, chunk_elements // (2 * packed[0].numel())) if packed.dim() > 1 else packed.shape[0]
    if rows >= packed.shape[0]:
        return fn(packed, scale, global_scale, dtype)
    return torch.cat([fn(packed[i:i + rows], scale[i:i + rows], global_scale[i:i + rows] if global_scale.dim() else global_scale, dtype)
                      for i in range(0, packed.shape[0], rows)])


def nvfp4_quantize(w, global_scale=None):
    """A float weight [..., K] (K a multiple of 16) in the ModelOpt NVFP4 layout: (packed uint8 [..., K/2], block scales
    e4m3 [..., K/16], global fp32 scalar). global = amax / (6 * 448) so every block scale fits e4m3; a zero block gets
    scale 1 (ModelOpt); the codes are round-to-nearest-even e2m1 of w / (block scale * global)."""
    w = w.float()
    if global_scale is None:
        global_scale = (w.abs().amax() / (FP4_MAX * FP8_MAX)).clamp(min=torch.finfo(torch.float32).tiny)
    blocks = w.view(*w.shape[:-1], -1, FP4_BLOCK)
    block_scale = blocks.abs().amax(-1) / FP4_MAX / global_scale
    block_scale = torch.where(block_scale == 0, torch.ones_like(block_scale), block_scale).to(torch.float8_e4m3fn)
    codes = _e2m1_encode(blocks / (block_scale.float() * global_scale)[..., None]).reshape(w.shape)
    return (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous(), block_scale, global_scale.reshape(())


def _blocked_scales(scale):
    """[N, K/16] e4m3 block scales in cuBLAS's 128x4-tile (32x4x4 inside) layout, the `SWIZZLE_32_4_4` operand of
    torch.nn.functional.scaled_mm (docs.nvidia.com/cuda/cublas: d-block-scaling-factors-layout)."""
    rows, cols = scale.shape
    r, c = -(-rows // 128) * 128, -(-cols // 4) * 4
    padded = scale
    if (r, c) != (rows, cols):
        padded = torch.zeros(r, c, dtype=scale.dtype, device=scale.device)
        padded[:rows, :cols] = scale
    return padded.view(r // 128, 128, c // 4, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)


def fp4_linear(x, packed, blocked, global_scale, out_dtype):
    """x [M, K] (bf16) times an NVFP4 weight [N, K] through the Blackwell fp4 tensor cores (torch's scaled_mm with the
    two-level NVFP4 recipe): the activations are quantised to NVFP4 on the fly (per-16 e4m3 block scales, a global scale
    of amax / (6 * 448)), which moves the numbers beyond the W4A16 dequantise path; serving only, opt-in
    (`enable_fast(fp4=True)`). `blocked` is the weight's block scales in the swizzled layout (`_blocked_scales`)."""
    from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
    M = x.shape[0]
    pad = (-M) % 16
    if pad:  # ponytail: whole 16-row groups on the activation side, the tile the swizzled scales assume
        x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
    xq, xs, xg = nvfp4_quantize(x)
    out = scaled_mm(xq.view(torch.float4_e2m1fn_x2), packed.view(torch.float4_e2m1fn_x2).t(),
                    [_blocked_scales(xs), xg.reshape(1)], [ScalingType.BlockWise1x16, ScalingType.TensorWise],
                    [blocked, global_scale.reshape(1)], [ScalingType.BlockWise1x16, ScalingType.TensorWise],
                    [SwizzleType.SWIZZLE_32_4_4], [SwizzleType.SWIZZLE_32_4_4], output_dtype=out_dtype)
    return out[:M]


class NVFP4Linear(nn.Module):
    """A bias-free linear whose weight stays packed as ModelOpt NVFP4 (uint8 pairs, e4m3 per-16 block scales, a global
    fp32 scale: 0.56 bytes per weight). Two modes: dequantise to bf16 at every call (`fp4=False`, exact W4A16 on any
    card, the training path on an L40S), or the fp4 tensor-core GEMM (`fp4=True`, Blackwell: `fp4_linear`). The weight
    is a buffer, never a parameter: LoRA, optimisers and checkpoints never see it."""

    def __init__(self, packed, scale, global_scale):
        super().__init__()
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("global_scale", global_scale.reshape(()).float(), persistent=False)
        self.blocked = None  # `_blocked_scales(scale)` under fp4 mode
        self.fused = None
        self.in_features, self.out_features = packed.shape[1] * 2, packed.shape[0]

    def set_mode(self, fp4, fused=None):
        self.fused = fused
        self.blocked = _blocked_scales(self.scale) if fp4 else None

    @property
    def weight(self):
        return nvfp4_dequantize(self.packed, self.scale, self.global_scale, torch.bfloat16, self.fused)

    def forward(self, x):
        if self.blocked is not None:
            return fp4_linear(x.reshape(-1, x.shape[-1]), self.packed, self.blocked, self.global_scale, x.dtype).view(*x.shape[:-1], -1)
        return F.linear(x, self.weight.to(x.dtype))


class NVFP4Experts(nn.Module):
    """The fused expert stack of a Qwen3_5MoeSparseMoeBlock (`gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`) held as
    NVFP4 buffers (`NVFP4Linear`'s layout per expert, the global scale per (expert, projection) since gate, up and down
    carry their own). `forward` has the HF experts' signature (tokens [S, H], top-k indices [S, k], weights [S, k]) and
    runs the kernel `moe_kernel` picks: the fused W4A16 Triton kernel (`janus.moe_w4a16`: the GEMM reads the packed
    weights, no bf16 stack, sync-free so a level graph captures it; the default on CUDA for inference, 10 to 30x the
    dequantise paths on the 5090), or a dequantise-per-call path (the whole stack decoded to bf16 once per call, then
    transformers' grouped GEMM (`torch._grouped_mm` per projection; sm_90+), the padded `torch.bmm` (`_bmm`, the
    default below sm_90, on the CPU, and for training, which the Triton kernel has no backward for) or the per-hit-expert
    loop (`_eager`)). fp4 mode runs the tokens of each hit expert through `fp4_linear` (gate and up separately, their
    global scales differ). The bmm, loop and fp4 paths take a host-side count per call, so no CUDA graph covers them."""

    def __init__(self, gate_up, gate_up_scale, gate_up_global, down, down_scale, down_global, act_fn=F.silu):
        super().__init__()
        for name, t in (("gate_up", gate_up), ("gate_up_scale", gate_up_scale), ("gate_up_global", gate_up_global),
                        ("down", down), ("down_scale", down_scale), ("down_global", down_global)):
            self.register_buffer(name, t, persistent=False)
        self.num_experts, self.hidden_dim, self.intermediate_dim = gate_up.shape[0], gate_up.shape[2] * 2, down.shape[2] * 2
        self.act_fn, self.has_gate, self.has_bias, self.is_transposed = act_fn, True, False, False
        self.fused, self.blocked = None, None

    def set_mode(self, fp4, fused=None):
        self.fused = fused
        I = self.intermediate_dim
        self.blocked = tuple(torch.stack([_blocked_scales(s) for s in scales]) for scales in
                             (self.gate_up_scale[:, :I], self.gate_up_scale[:, I:], self.down_scale)) if fp4 else None

    @property
    def gate_up_proj(self):
        return nvfp4_dequantize(self.gate_up, self.gate_up_scale, self.gate_up_global, torch.bfloat16, self.fused)

    @property
    def down_proj(self):
        return nvfp4_dequantize(self.down, self.down_scale, self.down_global, torch.bfloat16, self.fused)

    def _apply_gate(self, gate_up_out):
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if self.blocked is not None:
            return self._eager(hidden_states, top_k_index, top_k_weights)
        kernel = moe_kernel(hidden_states.device)
        # ponytail: the Triton kernel needs hidden % 128 == 0 and intermediate % 64 == 0 (its tile sizes) and has no
        # backward; other shapes and training take the dequantise paths below.
        if kernel == "fused" and not torch.is_grad_enabled() and self.hidden_dim % 128 == 0 and self.intermediate_dim % 128 == 0:
            from .moe_w4a16 import fused_moe
            return fused_moe(hidden_states, top_k_index, top_k_weights, self.gate_up, self.gate_up_scale, self.gate_up_global,
                             self.down, self.down_scale, self.down_global)
        if kernel == "fused":
            kernel = "grouped" if hidden_states.is_cuda and torch.cuda.get_device_capability(hidden_states.device) >= (9, 0) else "bmm"
        if kernel == "grouped":
            from transformers.integrations.moe import grouped_mm_experts_forward
            return grouped_mm_experts_forward(self, hidden_states.to(torch.bfloat16), top_k_index, top_k_weights).to(hidden_states.dtype)
        if kernel == "bmm":
            # ponytail: token chunks bound the padded width (real routing is skewed: one expert took 1,800 of 4,096
            # tokens on the A3B and the [E, width, H] gather reached 1.9 GB); the stacks are dequantised once for all chunks.
            gate_up, down = self.gate_up_proj, self.down_proj
            return torch.cat([self._bmm(hidden_states[i:i + BMM_CHUNK_TOKENS], top_k_index[i:i + BMM_CHUNK_TOKENS], top_k_weights[i:i + BMM_CHUNK_TOKENS], gate_up, down)
                              for i in range(0, hidden_states.shape[0], BMM_CHUNK_TOKENS)])
        return self._eager(hidden_states, top_k_index, top_k_weights, self.gate_up_proj, self.down_proj)

    def _bmm(self, hidden_states, top_k_index, top_k_weights, gate_up, down):
        """Every expert's tokens padded to the busiest expert's count, the two projections as one `torch.bmm` each
        ([E, M, H] x [E, H, 2I], [E, M, I] x [E, I, H]): about ten kernels per layer whatever E, against six per hit
        expert in the loop. Padded rows read a zero row and are dropped before the scatter-add. One host sync (the pad
        length). ponytail: the padding wastes M_max / M_mean of the GEMM work (about 2x at 8 routed of 256); a
        grouped GEMM has none, but has no fast kernel below sm_90."""
        S, k = top_k_index.shape
        flat_expert = top_k_index.reshape(-1)
        order = torch.argsort(flat_expert, stable=True)
        counts = torch.bincount(flat_expert, minlength=self.num_experts)
        starts = counts.cumsum(0) - counts
        width = int(counts.max())
        slot = torch.arange(S * k, device=order.device) - starts[flat_expert[order]]  # position of each assignment within its expert
        index = torch.full((self.num_experts, width), S, dtype=torch.long, device=order.device)  # S: the zero row
        index[flat_expert[order], slot] = order // k
        x = torch.cat([hidden_states, hidden_states.new_zeros(1, hidden_states.shape[1])])[index]  # [E, width, H]
        h = torch.bmm(x, gate_up.transpose(1, 2).to(x.dtype))  # [E, width, 2I]; the weights follow the activations' dtype as in the loop
        h = self._apply_gate(h)
        y = torch.bmm(h, down.transpose(1, 2).to(x.dtype))  # [E, width, H]
        valid = index.reshape(-1) < S
        weight = torch.zeros(self.num_experts * width, dtype=hidden_states.dtype, device=order.device)
        weight[(flat_expert[order] * width + slot)] = top_k_weights.reshape(-1)[order].to(hidden_states.dtype)
        out = torch.zeros_like(hidden_states)
        out.index_add_(0, index.reshape(-1)[valid], (y.reshape(-1, y.shape[-1]) * weight[:, None])[valid])
        return out

    def _eager(self, hidden_states, top_k_index, top_k_weights, gate_up=None, down=None):
        """Per hit expert: gather its tokens, the two GEMMs (dequantised weights `gate_up`/`down`, else fp4), scatter-add."""
        out = torch.zeros_like(hidden_states)
        I = self.intermediate_dim
        hits = torch.bincount(top_k_index.flatten(), minlength=self.num_experts).tolist()
        for e, count in enumerate(hits):
            if count == 0:
                continue
            token, slot = torch.where(top_k_index == e)
            x = hidden_states[token]
            if gate_up is not None:
                h = self._apply_gate(F.linear(x, gate_up[e].to(x.dtype)))
                y = F.linear(h, down[e].to(x.dtype))
            else:
                gate = fp4_linear(x, self.gate_up[e, :I], self.blocked[0][e], self.gate_up_global[e, 0], x.dtype)
                up = fp4_linear(x, self.gate_up[e, I:], self.blocked[1][e], self.gate_up_global[e, I], x.dtype)
                y = fp4_linear(self.act_fn(gate) * up, self.down[e], self.blocked[2][e], self.down_global[e, 0], x.dtype)
            out.index_add_(0, token, (y * top_k_weights[token, slot, None]).to(out.dtype))
        return out


def moe_kernel(device):
    """Which expert GEMM path runs the MoE block: `JANUS_MOE_KERNEL` (fused | grouped | bmm | loop) when set, else
    `fused` on CUDA (the Triton W4A16 kernel of `janus.moe_w4a16` for NVFP4Experts under inference; under a gradient
    NVFP4Experts takes the dequantise path below, and the HF bf16 experts map it to grouped_mm from sm_90 on, eager
    below) and `bmm` on the CPU. Of the dequantise paths: torch's grouped GEMM from sm_90 on (on the RTX 5090, sm_120,
    it is a fallback that reads its offsets on the host, so no CUDA graph covers it, but end to end it beats the padded
    bmm: 1.59 s against 2.26 s at a 4k state, docs/phase4/a3b-runner-port.md) and the padded bmm below, the CPU
    included: on the L40S (sm_89) `grouped_mm` falls back to something pathologically slow (a step-0 dev pass sat at
    99% utilisation for 4.5 h). `loop` is the per-hit-expert eager loop (`NVFP4Experts._eager`, six kernels per
    expert). The HF experts (bf16 MoE weights) follow the same choice through `config._experts_implementation`
    (grouped_mm or eager)."""
    choice = os.environ.get("JANUS_MOE_KERNEL", "").lower()
    if choice in ("fused", "grouped", "bmm", "loop"):
        return choice
    if choice:
        raise ValueError("JANUS_MOE_KERNEL must be fused, grouped, bmm or loop")
    device = torch.device(device)
    return "fused" if device.type == "cuda" else "bmm"


def load_nvfp4(lm, snapshot, dtype=torch.bfloat16):
    """Fill a meta-device Qwen3_5MoeTextModel from NVIDIA's ModelOpt checkpoint of the A3B (`hf_quant_config.json`:
    experts, shared expert and lm_head W4A16_NVFP4 in per-expert `gate_proj`/`up_proj`/`down_proj` tensors; the attention and
    DeltaNet projections FP8 e4m3 with one fp32 `weight_scale` each; everything else bf16), reading the shards on the CPU.
    The experts become `NVFP4Experts` and the shared expert's linears `NVFP4Linear` (packed as stored); the FP8 projections
    are dequantised to bf16 nn.Linear (1.3 GB more than fp8 on the A3B, and what LoRA and the fp8 serving path expect);
    `input_scale` tensors (ModelOpt's activation calibration) are not used: activations stay in bf16 (W4A16). The vision
    tower, the MTP head and lm_head are skipped. Raises ValueError when a model parameter is left unfilled."""
    import json
    from safetensors import safe_open
    snapshot = Path(snapshot)
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model."
    config = lm.config
    E, I, H = config.num_experts, config.moe_intermediate_size, config.hidden_size
    # The expert stacks are allocated up front and filled per expert (before to_empty, so the bf16 stacks never exist).
    for layer in lm.layers:
        layer.mlp.experts = NVFP4Experts(torch.empty(E, 2 * I, H // 2, dtype=torch.uint8), torch.empty(E, 2 * I, H // FP4_BLOCK, dtype=torch.float8_e4m3fn),
                                         torch.empty(E, 2 * I, 1), torch.empty(E, H, I // 2, dtype=torch.uint8),
                                         torch.empty(E, H, I // FP4_BLOCK, dtype=torch.float8_e4m3fn), torch.empty(E, H, 1), layer.mlp.experts.act_fn)
    lm.to_empty(device="cpu")
    lm.rotary_emb = type(lm.rotary_emb)(config=config)  # to_empty leaves the non-persistent inv_freq buffers uninitialised
    modules = dict(lm.named_modules())
    filled, quantised, by_file = set(), {}, {}
    for key, file in weight_map.items():
        if key.startswith(prefix):
            by_file.setdefault(file, []).append(key)
    for file, keys in sorted(by_file.items()):
        with safe_open(str(snapshot / file), "pt", device="cpu") as f:
            for key in sorted(keys):
                name = key[len(prefix):]
                module, _, leaf = name.rpartition(".")
                if leaf == "input_scale":
                    continue
                t = f.get_tensor(key)
                if ".mlp.experts." in name:
                    parent, _, rest = module.rpartition(".experts.")
                    index, proj = rest.split(".")
                    experts = modules[parent + ".experts"]
                    if proj == "down_proj":
                        target = {"weight": experts.down, "weight_scale": experts.down_scale, "weight_scale_2": experts.down_global}[leaf][int(index)]
                    else:
                        rows = slice(0, I) if proj == "gate_proj" else slice(I, 2 * I)
                        target = {"weight": experts.gate_up, "weight_scale": experts.gate_up_scale, "weight_scale_2": experts.gate_up_global}[leaf][int(index), rows]
                    target.copy_(t.expand(target.shape) if leaf == "weight_scale_2" else t)
                elif leaf in ("weight_scale", "weight_scale_2") or t.dtype in (torch.float8_e4m3fn, torch.uint8):
                    quantised.setdefault(module, {})[leaf] = t
                else:
                    getattr(modules[module], leaf).data = t.to(dtype)
                    filled.add(name)
    for module, tensors in quantised.items():
        parent, _, leaf = module.rpartition(".")
        if tensors["weight"].dtype == torch.uint8:  # NVFP4 (the shared expert): packed, its nn.Linear replaced
            setattr(modules[parent], leaf, NVFP4Linear(tensors["weight"], tensors["weight_scale"], tensors["weight_scale_2"]))
        else:  # FP8 per-tensor (the projections): dequantised once
            modules[module].weight.data = (tensors["weight"].float() * tensors["weight_scale"].float()).to(dtype)
            filled.add(module + ".weight")
    missing = [n for n, _ in lm.named_parameters() if n not in filled]
    if missing:
        raise ValueError(f"{snapshot}: {len(missing)} model tensors not in the shards, e.g. {missing[:5]}")
    return lm


class Fused:
    """Fused glue (`HybridBackbone.enable_fast(compile=True)` for serving, `HybridBackbone.glue` for training on
    CUDA, where `lora` also replaces peft's LoRA forward, `fuse_lora`): the elementwise work between a layer's
    GEMMs and its fla or attention kernel as torch.compile'd functions (Inductor, dynamic shapes, one compile per
    function shared by every layer since the weights are arguments), 3 to 5 kernels per layer instead of 40 to 60.
    Rounding moves at the bf16 level (fp32 inside a fused kernel, one rounding at the end; the plain path rounds
    at every op); measured in docs/phase4/latency-track2.md. `compile=False` runs the same functions eagerly (tests)."""

    RECOMPILE_LIMIT = 64

    def __init__(self, compile=True, serve=False):
        if compile:
            # Every function sees a few input classes (batch 1 or more, fp32 pieces or bf16 sdpa output at _gate_out,
            # the dtype per model) and Dynamo's default of 8 recompiles per function would silently drop to eager in
            # a long-lived server; duck shaping would also tie dims that happen to be equal at the first trace.
            # ponytail: process-wide Dynamo settings, fine for a serving process.
            import torch._dynamo.config as dynamo_config
            import torch.fx.experimental._config as shape_config
            dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, self.RECOMPILE_LIMIT)
            dynamo_config.accumulated_recompile_limit = max(dynamo_config.accumulated_recompile_limit, 8 * self.RECOMPILE_LIMIT)
            shape_config.use_duck_shape = False
        # Serving compiles with automatic dynamic shapes: the model's dims stay static (persistent reductions, fixed
        # strides) and only the batch and length dims that change become symbolic, after one recompile each.
        dynamic = None if serve else True
        wrap = (lambda f, **options: torch.compile(f, dynamic=dynamic, options=options)) if compile else (lambda f, **options: f)
        self.delta_mix, self.gate_norm, self.rmsnorm, self.add_rmsnorm, self.silu_mul, self.attn_qk, self.gate_out, self.quantize, self.dequant, self.lora = map(
            wrap, (_delta_mix, _gate_norm, _rmsnorm, _add_rmsnorm, _silu_mul, _attn_qk, _gate_out, _quantize_rows, _nvfp4_dequant, _lora))
        # serve: run_level takes _serve_level (enable_fast only; training's glue keeps the per-layer functions above)
        self.serve = serve
        self.add_norm, self.scores, self.merge, self.attn_serve = map(wrap, (_add_norm, _scores, _merge_gate, _attn_serve))
        # The conv taps recomputed inside their consumers (the q/k l2-norm reductions) rather than written out first.
        self.delta_serve = wrap(_delta_serve, realize_reads_threshold=16, realize_opcount_threshold=100, realize_acc_reads_threshold=16)
        self.warned = False

    def check(self):
        """One warning when Dynamo has hit its recompile limit on any frame (that function then runs eagerly)."""
        if self.warned:
            return
        from torch._dynamo.utils import counters
        if any("recompile limit" in reason.lower() for reason in counters["unimplemented"]):
            log.warning("torch.compile hit its recompile limit; part of the fused glue now runs eagerly (TORCH_LOGS=recompiles to see why)")
            self.warned = True


def gated_delta_net(m, x, valid, tail, recurrent, rule, fused=None, output_state=True):
    """Sequence mixing of one Qwen3_5GatedDeltaNet layer `m` continuing from a previous state.

    x [B, L, d] (already input-normed), valid [B, L] bool (False = right padding), tail [B, conv_dim, K-1] or None,
    recurrent [B, H, Dk, Dv] or None. Returns (out [B, L, d], new tail, new recurrent), the states taken at each row's
    real end: padded steps have g = beta = 0 and therefore leave the recurrent state untouched. Without `output_state` the new
    recurrent state is None: the kernel then never writes it (2 MB per row per layer on the 4B, 1.3 GB at a
    640-row leaf level; the leaves keep no state)."""
    batch, length, _ = x.shape
    kernel = m.conv_kernel_size
    keep = valid.to(x.dtype)[..., None]
    x = x.masked_fill(~valid[..., None], 0)  # not a multiply: an inf at a padded position would give inf*0 = NaN
    z = m.in_proj_z(x)
    b, a = m.in_proj_b(x), m.in_proj_a(x)
    if fused is not None:
        mixed = m.in_proj_qkv(x)  # [B, L, conv_dim], token-major: no transposes, no conv kernel, no head repeats
        if tail is None:
            tail = mixed.new_zeros(batch, mixed.shape[-1], kernel - 1)
        query, key, value, g, beta, new_tail = fused.delta_mix(mixed, b, a, valid, tail, m.conv1d.weight, m.conv1d.bias,
                                                               m.A_log, m.dt_bias, m.key_dim, m.head_k_dim, m.head_v_dim)
        if rule is torch_chunk_gated_delta_rule and m.num_v_heads // m.num_k_heads > 1:  # fla reads GQA heads in place
            query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
            key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        core, new_recurrent = rule(query, key, value, g=g, beta=beta, initial_state=recurrent, output_final_state=output_state,
                                   use_qk_l2norm_in_kernel=True)
        core = fused.gate_norm(core.reshape(batch, length, -1, m.head_v_dim), z.view(batch, length, -1, m.head_v_dim),
                               m.norm.weight, m.norm.variance_epsilon).reshape(batch, length, -1)
        return m.out_proj(core), new_tail, new_recurrent
    mixed = m.in_proj_qkv(x).transpose(1, 2)  # [B, conv_dim, L]
    if tail is None:
        tail = mixed.new_zeros(batch, mixed.shape[1], kernel - 1)
    conv_in = torch.cat([tail.to(mixed.dtype), mixed], dim=-1)  # [B, conv_dim, K-1+L]
    ends = valid.sum(1)
    index = ends[:, None] + torch.arange(kernel - 1, device=x.device)[None]
    new_tail = conv_in.gather(2, index[:, None, :].expand(-1, conv_in.shape[1], -1))
    mixed = F.silu(F.conv1d(conv_in, m.conv1d.weight, m.conv1d.bias, groups=conv_in.shape[1])).transpose(1, 2)
    query, key, value = torch.split(mixed, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, m.head_k_dim)
    key = key.reshape(batch, length, -1, m.head_k_dim)
    value = value.reshape(batch, length, -1, m.head_v_dim)
    beta = b.sigmoid() * keep
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias) * valid.float()[..., None]
    if m.num_v_heads // m.num_k_heads > 1:
        query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
    core, new_recurrent = rule(query, key, value, g=g, beta=beta, initial_state=recurrent, output_final_state=output_state,
                               use_qk_l2norm_in_kernel=True)
    core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(batch, length, -1)
    return m.out_proj(core), new_tail, new_recurrent


SERVE_CHUNK_TOKENS = 65536  # the state chunk under enable_fast with sdpa (one flash pass per state)
ATTN_CHUNK = 8192  # keys per scored piece at the branch levels: bounds the key axis of the torch attention's score matrix
PIECE_BYTES = 512 << 20  # the fp32 score matrix of one query chunk in _piece: bounds the query axis (a 128-question request
# has 16k block queries and 20k leaf queries against every state key; unbounded, a 16k-token request peaked at 15.6 GB of
# transients, docs/phase4/latency-track2.md "Serving memory under wide Score batches")


def _piece(q, k, v, kv_valid, causal, scaling, scores=None):
    """One attention piece in torch: q [G, Hq, Lq, D] over k, v [G, Hkv, Lk, D] with keys `kv_valid` [G, Lk], and
    (`causal`, the row's own tokens) key j <= query i. Returns (normalised output fp32 [G, Hq, Lq, D], log-sum-exp
    [G, Hq, Lq, 1]); a query without a visible key gets output 0 and lse -inf. GQA folds the query heads of a kv head
    into the query axis, so the keys are never repeated (transformers' sdpa path repeats them 8x per row). The queries
    are scored in chunks whose score matrix stays under PIECE_BYTES (every query row's softmax is independent, so the
    numbers are those of one chunk)."""
    G, Hq, Lq, D = q.shape
    scores = scores or _scores
    step = max(1, PIECE_BYTES // (4 * G * Hq * k.shape[2]))
    if step >= Lq:
        return scores(q, k, v, kv_valid, causal, scaling, 0)
    outs, lses = zip(*(scores(q[:, :, a:a + step], k, v, kv_valid, causal, scaling, a) for a in range(0, Lq, step)))
    return torch.cat(outs, dim=2), torch.cat(lses, dim=2)


def _scores(q, k, v, kv_valid, causal, scaling, offset):
    """`_piece` of one query chunk starting at query `offset` (the causal rule counts from the chunk's first row)."""
    G, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    g = Hq // Hkv
    s = torch.matmul(q.reshape(G, Hkv, g * Lq, D), k.transpose(2, 3)).float() * scaling
    allowed = kv_valid[:, None, None, None, :]
    if causal:
        allowed = allowed & (torch.arange(Lk, device=q.device)[None] <= torch.arange(offset, offset + Lq, device=q.device)[:, None])
    s.view(G, Hkv, g, Lq, Lk).masked_fill_(~allowed, float("-inf"))  # in place: s is this function's own fp32 copy
    m = s.amax(-1, keepdim=True)
    e = (s - m.masked_fill(m == float("-inf"), 0.)).exp_()
    l = e.sum(-1, keepdim=True)
    o = torch.matmul(e.to(v.dtype), v).float() / l.clamp(min=1e-30)
    return o.view(G, Hq, Lq, D), (m + l.log()).view(G, Hq, Lq, 1)


def _merge(pieces):
    """The attention pieces' (output, log-sum-exp) merged into the softmax over the union of their keys."""
    total = torch.logsumexp(torch.cat([lse for _, lse in pieces], dim=-1), dim=-1, keepdim=True)
    return sum((lse - total).exp() * o for o, lse in pieces)


def _row_order(o, lse, back, L):
    """An ancestor piece's output [G, Hq, R * L, D] and log-sum-exp (queries grouped by ancestor row) in row order,
    [B, Hq, L, ...]; `back` is each row's group * R + slot."""
    G, Hq, RL, D = o.shape
    R = RL // L
    return o.view(G, Hq, R, L, D)[back // R, :, back % R], lse.view(G, Hq, R, L, 1)[back // R, :, back % R]


def _merge_gate(own, ancestors, gate):
    """Serving (Fused.merge): the ancestor pieces put in row order (`_row_order`), `_merge`, `_gate_out`."""
    return _gate_out(_merge([own] + [_row_order(o, lse, back, own[0].shape[2]) for o, lse, back in ancestors]), gate)


def _flex_piece(q, k, v, kv_valid, causal, scaling):
    """`_piece` through flex_attention: the block mask from the padding pattern, no score matrix."""
    out, lse = _flex_forward(None, q, k, v, flex_mask(kv_valid, 0 if causal else k.shape[2], q.shape[2]), scaling, lse=True)
    return out.float(), lse.float()[..., None]


BRANCH_PACKED = True  # the plain path (training, evaluate) runs the branch levels as one varlen attention call (Packed)
COMPILE_GLUE = os.environ.get("JANUS_HYBRID_COMPILE", "1") != "0"  # training on CUDA runs the glue as Fused's compiled kernels


class Packed(NamedTuple):
    """The plain path's plan of one branch level (`packed_attention`): every row's keys are its ancestors' valid
    tokens, level by level, then its own, packed back to back without padding (`keys` indexes the flat token axis of
    the ancestor levels' keys followed by the level's own, in row order); `queries` the flat [B * L] indices of the
    valid queries; `cu_q` / `cu_k` the int32 sequence offsets and `max_q` / `max_k` the longest, the varlen layout of
    the flash kernel."""
    keys: torch.Tensor
    queries: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int


def _varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale):
    """Causal attention over packed sequences: q [Tq, Hq, D] and k, v [Tk, Hkv, D] hold the sequences back to back
    (`cu_q`, `cu_k` int32 offsets); query i of a sequence with Lq queries and Lk keys sees keys j <= Lk - Lq + i (its
    cached ancestors, then itself causally). On CUDA in bf16/fp16 the flash kernel (GQA in place, no score matrix, the
    log-sum-exp for its own backward); elsewhere sdpa under the same rule as one dense [Tq, Tk] mask."""
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        return torch.ops.aten._flash_attention_forward(q, k, v, cu_q, cu_k, max_q, max_k, 0.0, True, False, scale=scale)[0]
    # ponytail: the fallback (CPU, fp32) masks every query against every key of the batch; fine for the tests and the
    # tiny models, quadratic in the batch's tokens otherwise. The upgrade is a per-sequence loop.
    lq, lk = cu_q.diff().long(), cu_k.diff().long()
    seg_q = torch.repeat_interleave(torch.arange(len(lq), device=q.device), lq)
    seg_k = torch.repeat_interleave(torch.arange(len(lk), device=q.device), lk)
    pos_q = torch.arange(q.shape[0], device=q.device) - cu_q[seg_q] + (lk - lq)[seg_q]
    pos_k = torch.arange(k.shape[0], device=q.device) - cu_k[seg_k]
    allowed = (seg_q[:, None] == seg_k[None]) & (pos_k[None] <= pos_q[:, None])
    out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                         attn_mask=allowed, scale=scale, enable_gqa=True)
    return out[0].transpose(0, 1)


def packed_attention(m, h, cos, sin, plan, states, fused=None):
    """`branch_attention` of the plain path: the level's rows as one varlen attention call (`_varlen`) over their
    ancestors' keys gathered per row (`plan`: a Packed) and their own. The gather copies an ancestor's keys once per
    descendant row, which the serving path's grouped pieces avoid for its hundreds of leaves; a training or evaluate
    batch has a few rows per state, and one kernel launch per layer instead of one per ancestor level and key
    chunk, with nothing but q, k, v, o and the log-sum-exp kept for the backward."""
    B, L, _ = h.shape
    q, k, v, gate = _attn_proj(m, h, cos, sin, fused)
    Hq, D = q.shape[1], q.shape[3]
    flat = lambda tensors: torch.cat([t.transpose(1, 2).reshape(-1, t.shape[1], D) for t in tensors])
    keys, values = flat((*states[0::2], k))[plan.keys], flat((*states[1::2], v))[plan.keys]
    queries = q.transpose(1, 2).reshape(B * L, Hq, D)[plan.queries]
    out = _varlen(queries, keys, values, plan.cu_q, plan.cu_k, plan.max_q, plan.max_k, m.scaling)
    out = out.new_zeros(B * L, Hq * D).index_copy(0, plan.queries, out.reshape(-1, Hq * D)).view(B, L, Hq * D)
    return m.o_proj(out.to(h.dtype) * torch.sigmoid(gate)), k, v


def _attn_proj(m, h, cos, sin, fused):
    """q [B, H, L, D] (normed, rotated), k [B, Hkv, L, D], v, gate [B, L, H*D] of one Qwen3_5Attention layer."""
    B, L, _ = h.shape
    if fused is not None and fused.serve and hasattr(m, "serve_qkv"):  # one GEMM for q, k, v (enable_fast, _merge_projections)
        qg, k, v = F.linear(h, m.serve_qkv).split(m.serve_split, dim=-1)
    else:
        qg, k, v = m.q_proj(h), m.k_proj(h), m.v_proj(h)
    qg = qg.view(B, L, -1, m.head_dim * 2)
    k = k.view(B, L, -1, m.head_dim)
    v = v.view(B, L, -1, m.head_dim).transpose(1, 2)
    if fused is not None and fused.serve:
        q, k, gate = fused.attn_serve(qg, k, cos, sin, m.q_norm.weight, m.k_norm.weight, m.q_norm.eps)
        return q, k, v, gate
    if fused is not None:
        q, k, gate = fused.attn_qk(qg, k, cos, sin, m.q_norm.weight, m.k_norm.weight, m.q_norm.eps, m.head_dim)
        return q, k, v, gate
    q, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(B, L, -1)
    q = m.q_norm(q.reshape(B, L, -1, m.head_dim)).transpose(1, 2)
    k = m.k_norm(k).transpose(1, 2)
    q, k = qwen3_5_modeling.apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v, gate


def state_attention(m, h, cos, sin, mask, state, fused):
    """The level-0 attention of the fused path (`Qwen3_5Attention.forward` with `_KV`, the glue fused): the row's own
    accumulated keys, `is_causal` with GQA in place when `mask` is None (the flash kernel), else the HF sdpa rule
    (repeated kv under the mask). Returns (out [B, L, d], keys, values)."""
    q, k, v, gate = _attn_proj(m, h, cos, sin, fused)
    if state:
        k, v = torch.cat([state[0], k], dim=2), torch.cat([state[1], v], dim=2)
    if mask is None:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] > 1, scale=m.scaling, enable_gqa=True)
    else:
        from transformers.integrations.sdpa_attention import repeat_kv
        out = F.scaled_dot_product_attention(q, repeat_kv(k, m.num_key_value_groups), repeat_kv(v, m.num_key_value_groups), attn_mask=mask, scale=m.scaling)
    return m.o_proj(fused.gate_out(out, gate)), k, v


def branch_attention(m, h, cos, sin, valid, plan, states, flex, fused=None):
    """Sequence mixing of one Qwen3_5Attention layer `m` on a branch level: every row attends to its ancestors' tokens
    and to its own, causally. One piece per ancestor level with the queries grouped by ancestor row (`plan`), so an
    ancestor's keys are read in place, once, however many rows descend from it; one piece for the rows' own tokens;
    the pieces merged through their log-sum-exps (the exact softmax over the union). h [B, L, d] (input-normed),
    cos/sin [B, L, R], valid [B, L]; plan = (kv_valid_0 [B_0, L_0], slots_0 [B_0, R_0], back_0 [B], ...) per ancestor
    level, slots the row ids grouped by ancestor row (padded with row 0), back each row's (group, slot) as group * R + slot;
    states = (k_0, v_0, k_1, v_1, ...) the ancestor levels' own keys and values [B_l, Hkv, L_l, D].
    Returns (out [B, L, d], k [B, Hkv, L, D], v)."""
    B, L, _ = h.shape
    q, k, v, gate = _attn_proj(m, h, cos, sin, fused)
    serve = fused is not None and fused.serve
    piece = _flex_piece if flex else functools.partial(_piece, scores=fused.scores) if serve else _piece
    pieces = [piece(q, k, v, valid, True, m.scaling)]
    for level, (kv_valid, slots, back) in enumerate(zip(plan[0::3], plan[1::3], plan[2::3])):
        k_a, v_a = states[2 * level], states[2 * level + 1]
        G, R = slots.shape
        Hq, D = q.shape[1], q.shape[3]
        qg = q[slots.flatten()].view(G, R, Hq, L, D).permute(0, 2, 1, 3, 4).reshape(G, Hq, R * L, D)
        # ponytail: a long ancestor (the state) is scored in key chunks, each a piece; flex needs no chunking.
        step = k_a.shape[2] if flex else ATTN_CHUNK
        for start in range(0, k_a.shape[2], step):
            o, lse = piece(qg, k_a[:, :, start:start + step], v_a[:, :, start:start + step],
                           kv_valid[:, start:start + step], False, m.scaling)
            pieces.append((o, lse, back) if serve else _row_order(o, lse, back, L))
    if serve:  # the row order, the merge and the output gate in one compiled function
        return m.o_proj(fused.merge(pieces[0], pieces[1:], gate)), k, v
    out = _merge(pieces)
    if fused is not None:
        return m.o_proj(fused.gate_out(out, gate)), k, v
    out = out.to(h.dtype).transpose(1, 2).reshape(B, L, -1) * torch.sigmoid(gate)
    return m.o_proj(out), k, v


def _layer_step(layer, x, valid, mask, cos, sin, rule, parent, keep, depth, flex, fused, *state):
    """One decoder layer on one level: returns (hidden, *new state tensors). DeltaNet layers continue from `state`
    rows `parent` (None: the rows as they are); the per-row copies are made here, one layer at a time, and dropped
    with the layer unless `keep`. Attention layers at level 0 (`depth` 0) run the HF forward over the row's own
    accumulated keys (`mask`: the level mask, `state`: (key, value) so far); at branch levels `mask` is the grouping
    plan and `state` the ancestor levels' keys and values (`branch_attention`). `fused` (a Fused, serving) swaps the
    glue for its compiled kernels. Pure in its inputs (checkpointable)."""
    residual = x
    h = layer.input_layernorm(x) if fused is None else fused.rmsnorm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
    if layer.layer_type == "linear_attention":
        if parent is not None:
            state = tuple(t.index_select(0, parent) for t in state)
        tail, recurrent = state if state else (None, None)
        h, tail, recurrent = gated_delta_net(layer.linear_attn, h, valid, tail, recurrent, rule, fused, keep)
        new_state = (tail, recurrent)
    elif depth == 0:
        if parent is not None:
            state = tuple(t.index_select(0, parent) for t in state)
        if fused is not None and isinstance(mask, (torch.Tensor, type(None))):  # a flex BlockMask keeps the HF attention
            h, key, value = state_attention(layer.self_attn, h, cos, sin, mask, state, fused)
            new_state = (key, value)
        else:
            kv = _KV(*state)
            h, _ = layer.self_attn(hidden_states=h, position_embeddings=(cos, sin), attention_mask=mask, past_key_values=kv)
            new_state = (kv.key, kv.value)
    elif isinstance(mask, Packed):
        h, key, value = packed_attention(layer.self_attn, h, cos, sin, mask, state, fused)
        new_state = (key, value)
    else:
        h, key, value = branch_attention(layer.self_attn, h, cos, sin, valid, mask, state, flex, fused)
        new_state = (key, value)
    if fused is None:
        x = residual + h
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    else:
        x, h = fused.add_rmsnorm(residual, h, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps)
        if hasattr(layer.mlp, "experts"):  # qwen3_5_moe: the HF sparse block (router, experts, gated shared expert), GEMM-bound
            x = x + layer.mlp(h)
        else:
            x = x + layer.mlp.down_proj(fused.silu_mul(layer.mlp.gate_proj(h), layer.mlp.up_proj(h)))
    return (x, *new_state) if keep else (x,)


def run_level(lm, embeds, positions, valid, mask, states, rule, checkpointing=False, parent=None, keep=True, depth=0, flex=False, fused=None):
    """All layers over one level batch. `states` has one tuple per layer (empty tuples at the first state chunk)."""
    cos, sin = lm.rotary_emb(embeds, positions)
    if fused is not None and fused.serve and not checkpointing and not flex:
        return _serve_level(lm, embeds, valid, mask, cos, sin, states, rule, parent, keep, depth, fused)
    x, new_states = embeds, []
    for layer, state in zip(lm.layers[: lm.config.num_hidden_layers], states):
        if checkpointing:
            out = torch.utils.checkpoint.checkpoint(_layer_step, layer, x, valid, mask, cos, sin, rule, parent, keep, depth, flex, fused, *state, use_reentrant=False)
        else:
            out = _layer_step(layer, x, valid, mask, cos, sin, rule, parent, keep, depth, flex, fused, *state)
        x, new_state = out[0], tuple(out[1:])
        new_states.append(new_state)
    return lm.norm(x), new_states


def _serve_delta(m, h, valid, ends, tail, recurrent, rule, fused, output_state):
    """`gated_delta_net` on the serving path: h already normed and masked (`_add_norm`), the four input projections
    one GEMM (`serve_proj`, or the modules when they could not be merged), the glue `_delta_serve`."""
    batch, length, _ = h.shape
    if hasattr(m, "serve_proj"):
        mixed, z, b, a = F.linear(h, m.serve_proj).split(m.serve_split, dim=-1)
    else:
        mixed, z, b, a = m.in_proj_qkv(h), m.in_proj_z(h), m.in_proj_b(h), m.in_proj_a(h)
    query, key, value, g, beta, new_tail = fused.delta_serve(mixed, b, a, valid, ends, tail, m.conv1d.weight, m.conv1d.bias,
                                                             m.A_log, m.dt_bias, m.key_dim, m.head_k_dim, m.head_v_dim)
    if rule is torch_chunk_gated_delta_rule and m.num_v_heads // m.num_k_heads > 1:  # fla reads GQA heads in place
        query = query.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
        key = key.repeat_interleave(m.num_v_heads // m.num_k_heads, dim=2)
    if rule is FLA_CHUNK_RULE and length <= RECURRENT_TOKENS:
        rule = FLA_RECURRENT_RULE
    elif rule is FLA_CHUNK_RULE and batch * length <= CHUNK32_TOKENS:
        rule = functools.partial(rule, chunk_size=32)
    core, new_recurrent = rule(query, key, value, g=g, beta=beta, initial_state=recurrent, output_final_state=output_state,
                               use_qk_l2norm_in_kernel=False)
    core = fused.gate_norm(core.reshape(batch, length, -1, m.head_v_dim), z.reshape(batch, length, -1, m.head_v_dim),
                           m.norm.weight, m.norm.variance_epsilon).reshape(batch, length, -1)
    return m.out_proj(core), new_tail, new_recurrent


def _serve_level(lm, x, valid, mask, cos, sin, states, rule, parent, keep, depth, fused):
    """`run_level` on the serving fast path (a Fused with `serve`, from enable_fast): the same layers with each
    layer's residual add folded into the next norm (`_add_norm`, also the DeltaNet input mask), q/k/v, the DeltaNet
    input projections and gate/up as one GEMM each where enable_fast merged them (`_merge_projections`), and the
    DeltaNet glue `_delta_serve`. Training and evaluate never come here (no Fused with `serve`)."""
    new_states, pending, ends = [], None, valid.sum(1)
    for layer, state in zip(lm.layers[: lm.config.num_hidden_layers], states):
        if parent is not None and (layer.layer_type == "linear_attention" or depth == 0):
            state = tuple(t.index_select(0, parent) for t in state)
        norm = layer.input_layernorm
        if layer.layer_type == "linear_attention":
            x, h = fused.add_norm(x, pending, norm.weight, norm.eps, valid)
            tail, recurrent = state if state else (None, None)
            h, *new_state = _serve_delta(layer.linear_attn, h, valid, ends, tail, recurrent, rule, fused, keep)
        else:
            x, h = fused.add_norm(x, pending, norm.weight, norm.eps, None)
            if depth == 0:
                h, *new_state = state_attention(layer.self_attn, h, cos, sin, mask, state, fused)
            else:
                h, *new_state = branch_attention(layer.self_attn, h, cos, sin, valid, mask, state, False, fused)
        norm = layer.post_attention_layernorm
        x, h = fused.add_norm(x, h, norm.weight, norm.eps, None)
        mlp = layer.mlp
        if hasattr(mlp, "serve_gate_up"):
            pending = mlp.down_proj(fused.silu_mul(*F.linear(h, mlp.serve_gate_up).chunk(2, dim=-1)))
        elif hasattr(mlp, "experts"):
            pending = mlp(h)
        else:
            pending = mlp.down_proj(fused.silu_mul(mlp.gate_proj(h), mlp.up_proj(h)))
        new_states.append(tuple(new_state) if keep else ())
    return fused.add_norm(x, pending, lm.norm.weight, lm.norm.eps, None)[1], new_states


def _merge_projections(lm):
    """Serving (enable_fast): the projections that read the same input as one weight each, q/k/v of an attention layer
    (`serve_qkv`), the DeltaNet's qkv/z/b/a (`serve_proj`), the MLP's gate/up (`serve_gate_up`); the modules'
    weights become views into it, so nothing is stored twice and the plain path reads the same numbers. Only plain
    bias-free nn.Linear modules are merged (not peft LoRA, fp8 or fp4); earlier merges are dropped first."""
    groups = (("self_attn", ("q_proj", "k_proj", "v_proj"), "serve_qkv"),
              ("linear_attn", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), "serve_proj"),
              ("mlp", ("gate_proj", "up_proj"), "serve_gate_up"))
    for layer in lm.layers[: lm.config.num_hidden_layers]:
        for owner, names, attr in groups:
            module = getattr(layer, owner, None)
            if module is None:
                continue
            for stale in (attr, "serve_split"):
                if hasattr(module, stale):
                    delattr(module, stale)
            linears = [getattr(module, name, None) for name in names]
            if not all(type(linear) is nn.Linear and linear.bias is None for linear in linears):
                continue
            sizes = [linear.weight.shape[0] for linear in linears]
            with torch.inference_mode(False), torch.no_grad():  # ordinary tensors: a gradient pass may still read the views
                merged = torch.cat([linear.weight for linear in linears])
                for linear, weight in zip(linears, merged.split(sizes)):
                    linear.weight = nn.Parameter(weight, requires_grad=linear.weight.requires_grad)
            setattr(module, attr, merged)
            module.serve_split = sizes


BATCH_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)
BRANCH_CHUNK_ROWS = 32  # level-1 rows (with their descendants) per pass of the branch levels (HybridBackbone._levels)
KV_MAX = 2048
WIDE_KEYS = 256  # the shortest level-0 key length the branch graphs are keyed on (LevelGraphs.widen)
# ponytail: the branch pieces then score up to twice the state's keys (masked); cheap next to the GEMMs up to KV_MAX.
STATE_BUFFER_TOKENS = 65536  # rows x key capacity of one long-state level-0 buffer set (2 GB on the 4B); beyond, branch levels run eagerly
BUFFER_BYTES = 4 << 30  # all buffer sets of one LevelGraphs together; least recently used sets (and their graphs) go first


def length_bucket(n, state=False):
    """Padded level length: a multiple of 4 up to 32, of 8 up to 64, of 16 up to 256, of 64 up to 1024, of 256 up to
    KV_MAX; None beyond. The state level (`state`) uses multiples of 64 up to 1024 (cuBLAS's 64-row tiles: a 157-token
    level costs the same at 160 and 192 but 1.4x more at 256 on the RTX 5090, docs/phase4/latency-track3.md), then of
    256. Branch levels pad finely: on the RTX 5090 a level batch is compute-bound from about 100 tokens on
    (docs/phase4/latency.md), so their padding is paid for. A state beyond KV_MAX is not graphed itself but its keys
    are still bucketed (multiples of 1024) so the branch levels behind it are: the bucket pads the key range the
    branch pieces score (masked, cheap), never the state pass (`batch_index` leaves that at its real length)."""
    if state and n > KV_MAX:
        return -(-n // 1024) * 1024
    step = (64 if n <= 1024 else 256) if state else 4 if n <= 32 else 8 if n <= 64 else 16 if n <= 256 else 64 if n <= 1024 else 256
    padded = -(-n // step) * step
    return padded if padded <= KV_MAX else None


def kv_capacity(kv_len):
    """Key capacity of the buffer set that holds `kv_len` keys: KV_MAX, else the next power of two."""
    return KV_MAX if kv_len <= KV_MAX else 1 << (kv_len - 1).bit_length()


def batch_bucket(n):
    return next((b for b in BATCH_BUCKETS if b >= n), None)


class LevelGraphs:
    """CUDA graphs of `run_level` over bucketed level shapes; serving only (docs/phase4/latency.md).

    Every level batch is padded to a (batch, length) bucket (exact: right padding is a no-op, see the module
    docstring), so a request touches a small set of shapes and each shape is captured once from the eager code on
    its first use. Level d writes its states into the buffer set `outs[(d, B_d)]` (one per level and batch bucket:
    DeltaNet tails and recurrent states, and the level's own keys and values in buffers KV_MAX long); a later level's
    graph reads them in place (the DeltaNet states through index_select on a static parent index, the keys through
    the grouping plan), so nothing is copied between levels and an eager level (a batch beyond the buckets) reads
    the same views. All graphs share one memory pool: every result is copied to a static buffer inside the capture,
    so replay order does not matter. A state longer than KV_MAX runs eagerly and `_assemble` copies its rows into a
    level-0 set of a larger key capacity (`kv_capacity`, keyed with the set), so the branch levels behind it replay
    their graphs as they do behind a cached state.
    Memory: the sets together stay under `buffer_bytes` (BUFFER_BYTES): allocating one evicts the least recently
    used sets not touched by the current request (`begin` marks a request) together with every graph that reads or
    writes them (`_sets`), and an allocation that still fails with out-of-memory gives False, which `_assemble`
    answers with fresh tensors (eager branch levels) rather than an error.
    ponytail: the per-batch-bucket recurrent buffers cost 48 MB per row on the 9B (24 layers of [32, 128, 128] fp32),
    2.5 GB at batch 32; the upgrade path is fewer, coarser buckets or bf16 states. A long-state level-0 set costs
    32 KB per key per row on the 4B (512 MB per 16k row); STATE_BUFFER_TOKENS bounds one set, BUFFER_BYTES all."""

    def __init__(self, lm, rule, fused=None, buffer_bytes=BUFFER_BYTES):
        self.lm, self.rule, self.fused = lm, rule, fused
        self.buffer_bytes, self.bytes, self.sizes, self.touched, self.generation = buffer_bytes, 0, {}, {}, 0
        # The model's device, made current around every capture and replay: torch.cuda.graph and Stream() work on the
        # *current* device, which is cuda:0 unless set, so a model on cuda:1 used to hit "operation not permitted when
        # stream is capturing" at the first capture (the RTX 3090 serving bug: a device mismatch, not a kernel).
        self.device = lm.embed_tokens.weight.device
        with torch.cuda.device(self.device):
            self.pool = torch.cuda.graph_pool_handle()
        self.graphs, self.outs = {}, {}
        self.disabled = False  # set after a failed capture: every later level runs eagerly (warning logged once)

    def _shape(self, x, mask, states, depth):
        """The graph key of a level, or None when the shape is not graphed (batch beyond the buckets, too many keys,
        states not produced by a graphed level). Level 0 keys on (source batch, accumulated keys); a branch level on
        every ancestor level's (batch, length, rows per ancestor) from the plan."""
        batch, length = x.shape[:2]
        if batch not in BATCH_BUCKETS or length_bucket(length, depth == 0) != length:
            return None
        source = None if not states[0] else next(s[0].shape[0] for s, layer in zip(states, self.lm.layers) if layer.layer_type == "linear_attention")
        if depth == 0:
            kv_prev = 0 if source is None else next(s[0].shape[2] for s, layer in zip(states, self.lm.layers) if layer.layer_type != "linear_attention")
            if kv_prev + length > KV_MAX:
                return None
            expected = self._views(0, source, kv_prev) if source is not None else None
            key = (0, source, batch, length, kv_prev)
        else:
            ancestors = tuple((valid.shape[0], valid.shape[1], slots.shape[1]) for valid, slots in zip(mask[0::3], mask[1::3]))
            expected = self._ancestor_views(depth, source, ancestors)
            key = (depth, source, batch, length, ancestors)
        if expected is not None and (expected is False or any(a.data_ptr() != b.data_ptr() or a.shape != b.shape
                                                              for s, e in zip(states, expected) for a, b in zip(s, e))):
            return None
        return key

    def run(self, x, pos, valid, mask, states, parent, keep, depth=0):
        """(hidden, new states) for one bucketed level, or None when it has no graph."""
        key = self._shape(x, mask, states, depth) if not self.disabled else None
        if key is None:
            return None
        if parent is None and key[1] is not None:
            parent = torch.arange(x.shape[0], device=x.device)
        key = (*key, keep)
        inputs = [x, pos, valid, parent, *(mask if depth else (mask,))]
        entry = self.graphs.get(key)
        if entry is None:
            try:
                with torch.cuda.device(self.device):
                    entry = self.graphs[key] = self._capture(key, inputs, states)
            except Exception as error:  # a capture that cannot complete on this device: the plain path from here on
                log.warning("CUDA graph capture failed (%s: %s); serving on the plain path", type(error).__name__, str(error).splitlines()[0])
                self.disabled = True
                return None
        graph, statics = entry
        for static, value in zip(statics, inputs):
            if value is not None:
                static.copy_(value)
        with torch.cuda.device(self.device):
            graph.replay()
        batch, length = x.shape[:2]
        return statics[-1], self._views(depth, batch, (key[4] if depth == 0 else 0) + length) if keep else [() for _ in states]

    def begin(self):
        """Marks a new request: the sets it touches from here on are not evicted for its duration."""
        self.generation += 1

    def _views(self, depth, batch, kv_len):
        """Per-layer views of the buffer set of (`depth`, `batch`, capacity of `kv_len`): the DeltaNet states and the
        first `kv_len` keys; False when the set does not exist yet."""
        key = (depth, batch, kv_capacity(kv_len))
        if key not in self.outs:
            return False
        self.touched[key] = self.generation
        return [(a[:batch], b[:batch]) if layer.layer_type == "linear_attention" else (a[:batch, :, :kv_len], b[:batch, :, :kv_len])
                for (a, b), layer in zip(self.outs[key], self.lm.layers)]

    def ensure(self, depth, batch, template, kv_len):
        """`_views` of (`depth`, `batch`, `kv_len`), allocating the buffer set from the shapes of `template` (per-layer
        state tuples of any batch size) when it does not exist yet; False when it cannot be allocated (the budget
        after evicting what may go, or the device's memory)."""
        key = (depth, batch, kv_capacity(kv_len))
        if key not in self.outs:
            shapes = [[(batch, *t.shape[1:]) if layer.layer_type == "linear_attention" else (batch, t.shape[1], key[2], t.shape[3]) for t in pair]
                      for pair, layer in zip(template, self.lm.layers)]
            need = sum(math.prod(shape) * t.element_size() for pair, per in zip(template, shapes) for t, shape in zip(pair, per))
            self._make_room(need)
            for attempt in (0, 1):
                try:
                    self.outs[key] = [tuple(torch.zeros(shape, dtype=t.dtype, device=t.device) for t, shape in zip(pair, per))
                                      for pair, per in zip(template, shapes)]
                    break
                except torch.OutOfMemoryError:
                    if attempt or not self._make_room(self.buffer_bytes):  # everything that may go, then give up
                        log.warning("no room for a %s-row buffer set of %d keys (%.1f GB); branch levels behind it run eagerly", batch, key[2], need / 1e9)
                        return False
            self.sizes[key], self.bytes = need, self.bytes + need
        return self._views(depth, batch, kv_len)

    def widen(self, states, key_valid):
        """Level-0 rows as the branch levels see them: when `states` are views of a level-0 buffer set, the same set's
        views at a power-of-two key length (at least WIDE_KEYS) and the key validity padded with False, so the branch
        graphs key on that coarse length rather than on the state's 64-token bucket (a new state length reuses the
        branch graphs instead of capturing them again; the extra keys are masked, a cheap part of the branch pieces).
        Anything else (fresh tensors after a failed allocation) is returned as it is."""
        batch, length = key_valid.shape
        wide = max(WIDE_KEYS, 1 << (length - 1).bit_length())
        views = self._views(0, batch, wide) if wide > length and kv_capacity(wide) == kv_capacity(length) else False
        if views is False or any(a.data_ptr() != b.data_ptr() for s, v in zip(states, views) for a, b in zip(s, v)):
            return states, key_valid
        return views, F.pad(key_valid, (0, wide - length))

    def _make_room(self, need):
        """Evicts least recently used sets untouched by the current request until `need` bytes fit the budget;
        returns whether anything was evicted."""
        evicted = False
        for key in sorted(self.outs, key=lambda k: self.touched.get(k, 0)):
            if self.bytes + need <= self.buffer_bytes:
                break
            if self.touched.get(key, 0) < self.generation:
                self.evict(key)
                evicted = True
        return evicted

    def evict(self, key):
        """Drops a buffer set and every graph that reads or writes it."""
        self.bytes -= self.sizes.pop(key)
        del self.outs[key]
        self.touched.pop(key, None)
        for graph_key in [k for k in self.graphs if key in self._sets(k)]:
            del self.graphs[graph_key]
        if not self.graphs:
            # The allocator retires a private pool when its last graph is destroyed (use_count 0, "freeable"); a
            # capture into the retired id then fails an internal assert and `run` would disable every graph for the
            # process. A fresh pool here; the retired one's blocks go on the next empty_cache (every capture's entry).
            with torch.cuda.device(self.device):
                self.pool = torch.cuda.graph_pool_handle()

    @staticmethod
    def _sets(key):
        """The buffer sets a graph key reads (its ancestors' or the previous chunk's) and writes (its own)."""
        depth, source, batch, length = key[:4]
        if depth == 0:
            reads = {(0, source, kv_capacity(key[4]))} if source is not None else set()
            return reads | {(0, batch, kv_capacity(key[4] + length))}
        return {(d, b, kv_capacity(l)) for d, (b, l, _) in enumerate(key[4])} | {(depth, batch, kv_capacity(length))}

    def _ancestor_views(self, depth, source, ancestors):
        """The `states` a branch level reads: DeltaNet states of level depth-1 (batch `source`), keys and values of
        every ancestor level; False when a buffer set is missing."""
        levels = [self._views(d, batch, length) for d, (batch, length, _) in enumerate(ancestors)]
        if any(v is False for v in levels):
            return False
        previous = levels[depth - 1]  # its DeltaNet entries are the level's rows whatever the key length
        return [prev if layer.layer_type == "linear_attention" else tuple(t for level in levels for t in level[i])
                for i, (prev, layer) in enumerate(zip(previous, self.lm.layers))]

    def _capture(self, key, inputs, states):
        depth, source, batch, length, keep = key[0], key[1], key[2], key[3], key[-1]
        kv_len = (key[4] if depth == 0 else 0) + length
        statics = [None if t is None else t.clone() for t in inputs]
        mask = statics[4] if depth == 0 else tuple(statics[4:])

        def level():
            hidden, new_states = run_level(self.lm, statics[0], statics[1], statics[2], mask, states, self.rule, False, statics[3], keep, depth, False, self.fused)
            if keep:
                views = self.ensure(depth, batch, new_states, kv_len)
                if views is False:
                    raise torch.OutOfMemoryError("no buffer set for the level's states")  # run(): the plain path from here on
                _copy_all([t for pair in views for t in pair], [t for pair in new_states for t in pair])
            return hidden

        # Warm-up on a side stream (Triton autotuning, cuBLAS workspaces), then capture; the outputs land in statics.
        # When a level reads and writes the same buffer set (a state chunk after the first: source == batch at level 0)
        # the warm-up overwrites its own inputs, so they are put back before the capture (a replay reads every input
        # before its end-of-level writes). A branch level writes its own level's set, never one it reads.
        snapshot = [tuple(t.clone() for t in s) for s in states] if keep and depth == 0 and source == batch else None
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            hidden = level()
            if snapshot is not None:
                for view, saved in zip(states, snapshot):
                    for t, c in zip(view, saved):
                        t.copy_(c)
        torch.cuda.current_stream().wait_stream(stream)
        statics.append(torch.empty_like(hidden))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            statics[-1].copy_(level())
        return graph, statics


def _copy_all(dst, src):
    """d.copy_(s) for every pair; the contiguous pairs as one multi-tensor kernel per dtype (a level's DeltaNet states
    into its buffer set: two launches instead of two per layer), the strided ones (key slices) one by one."""
    groups = {}
    for d, s in zip(dst, src):
        if d.is_contiguous() and s.is_contiguous() and d.dtype == s.dtype and d.shape == s.shape:
            pair = groups.setdefault(d.dtype, ([], []))
            pair[0].append(d)
            pair[1].append(s)
        else:
            d.copy_(s)
    for d, s in groups.values():
        torch._foreach_copy_(d, s)


def replay_level(lm, embeds, positions, valid, mask, states, rule, parent, keep, graphs, depth=0):
    """`run_level` through `graphs` (LevelGraphs); None when the shape has no graph. Same signature prefix as
    run_level so a profiler can wrap both (janus.latency.LevelTimer)."""
    return graphs.run(embeds, positions, valid, mask, states, parent, keep, depth)


class HybridBackbone(nn.Module):
    """A Qwen3_5TextModel behind the interface DecisionModel uses for its backbone (see module docstring)."""

    def __init__(self, lm, tokenizer, attention="sdpa", allow_fla=True, vision=None, state_chunk_tokens=4096):
        super().__init__()
        if not isinstance(lm, TEXT_MODELS):
            raise ValueError(f"HybridBackbone needs a Qwen3_5TextModel or Qwen3_5MoeTextModel, got {type(lm).__name__}")
        if attention not in ("eager", "sdpa", "flex"):
            raise ValueError("qwen3_5 backbones support attention eager, sdpa or flex")
        if state_chunk_tokens < 1:
            raise ValueError("state_chunk_tokens must be positive")
        # JANUS_HYBRID_FLA=0 forces the torch fallback without a code edit (e.g. if the Triton build fails on a card).
        allow_fla = allow_fla and os.environ.get("JANUS_HYBRID_FLA", "1") != "0"
        self.lm, self.tokenizer, self.attention, self.allow_fla = lm, tokenizer, attention, allow_fla
        self.state_chunk_tokens = state_chunk_tokens
        self.vision = vision  # janus.vision.Vision when the config enables image states, else None
        self._prepared = {}  # hidden states from prepare_many, read once by encode
        self.graphs = None  # LevelGraphs after enable_fast (serving)
        self.fused = None  # Fused after enable_fast(compile=True) (serving)
        self.glue = None  # Fused for training on CUDA (COMPILE_GLUE), made on the first step with gradients
        self.fast_causal = False  # enable_fast: level 0 without a mask under sdpa (serving only)
        self.single_pass = False  # enable_fast: a pack with one question runs its state and block as one level (_levels)
        lm.config._attn_implementation = HF_ATTENTION[attention]
        for layer in lm.layers:
            if layer.layer_type != "linear_attention":
                continue
            mixer = layer.linear_attn
            if not isinstance(mixer.norm, (Qwen3_5RMSNormGated, Qwen3_5MoeRMSNormGated)):  # a layer built before this module was imported
                norm = Qwen3_5RMSNormGated(mixer.head_v_dim, eps=mixer.layer_norm_epsilon).to(mixer.norm.weight.device)
                with torch.no_grad():
                    norm.weight.copy_(mixer.norm.weight)
                mixer.norm = norm
            # The stock forward (used by tests as the replay reference) must not hit a Triton kernel on CPU tensors.
            mixer.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            mixer.recurrent_gated_delta_rule = qwen3_5_modeling.torch_recurrent_gated_delta_rule
        self.config = lm.config
        self.config.use_cache = False

    @classmethod
    def load(cls, config):
        """From a ModelConfig: the language model of a Qwen3.5 checkpoint, or a random tiny instance for `tiny-qwen3_5`."""
        chunk = getattr(config, "state_chunk_tokens", 4096)
        # The HF loader validates its attention name; flex is set on the text config afterwards (in __init__).
        hf_attention = "sdpa" if config.attention == "flex" else config.attention
        if config.backbone in (TINY, TINY_MOE):
            from .packing import ByteTokenizer
            if not config.images:
                return cls(cls._tiny(config.hidden_size, config.layers, config.max_tokens, hf_attention, config.backbone == TINY_MOE), ByteTokenizer(),
                           config.attention, state_chunk_tokens=chunk)
            from .vision import Vision, tiny_full_model, tiny_processor
            full = tiny_full_model(cls._tiny_config(config.hidden_size, config.layers, config.max_tokens, hf_attention, TINY_VL_VOCAB), hf_attention)
            vision = Vision(full.model.visual, tiny_processor(), full.config.image_token_id, full.config.vision_start_token_id, full.config.vision_end_token_id)
            return cls(full.model.language_model, ByteTokenizer(), config.attention, vision=vision, state_chunk_tokens=chunk)
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
        hf_config = AutoConfig.from_pretrained(config.backbone, revision=config.revision, trust_remote_code=False)
        if getattr(hf_config, "model_type", None) != config.backbone_family:
            raise ValueError(f"{config.backbone} is a {getattr(hf_config, 'model_type', None)} checkpoint, not {config.backbone_family}")
        load = dict(revision=config.revision, trust_remote_code=False, attn_implementation=hf_attention, dtype=getattr(torch, config.dtype))
        vision = None
        if config.backbone_family == "qwen3_5_moe":
            from huggingface_hub import snapshot_download
            snapshot = Path(snapshot_download(config.backbone, revision=config.revision, local_files_only=True))
            if (snapshot / "hf_quant_config.json").exists():
                # ModelOpt NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4): the experts stay packed (load_nvfp4); the bf16 stacks
                # of the original would not fit any local card or the host, so the model is built on the meta device.
                if config.adaptation != "frozen" and not config.gradient_checkpointing:
                    # Without checkpointing autograd saves every layer's dequantised expert stack for the backward
                    # (1.6 GB x 40 layers on the A3B); with it a stack lives only inside its layer step, twice.
                    raise ValueError("training on an NVFP4 MoE checkpoint needs gradient_checkpointing: true")
                with torch.device("meta"):
                    lm = Qwen3_5MoeTextModel(hf_config.text_config)
                lm.config._attn_implementation = hf_attention
                lm = load_nvfp4(lm, snapshot, getattr(torch, config.dtype))
            else:
                from transformers import Qwen3_5MoeForCausalLM
                causal = Qwen3_5MoeForCausalLM.from_pretrained(config.backbone, **load)
                lm = causal.model
                del causal
        elif config.images:
            # The checkpoint class itself: the language model plus the vision tower and merger (janus.vision).
            from .vision import Vision
            full = Qwen3_5ForConditionalGeneration.from_pretrained(config.backbone, **load)
            lm = full.model.language_model
            vision = Vision.from_pretrained(full, config.backbone, config.revision, config.image_max_pixels)
            del full
        else:
            # The checkpoint class is the vision-language Qwen3_5ForConditionalGeneration; Qwen3_5ForCausalLM takes its
            # text_config, loads the model.language_model.* weights under model.* and ignores model.visual.* and mtp.*.
            causal = Qwen3_5ForCausalLM.from_pretrained(config.backbone, **load)
            lm = causal.model
            del causal  # the (tied) LM head is not part of the decision model
        lm.config._commit_hash = getattr(hf_config, "_commit_hash", None) or config.revision
        tokenizer = AutoTokenizer.from_pretrained(config.backbone, revision=config.revision, trust_remote_code=False)
        return cls(lm, tokenizer, config.attention, vision=vision, state_chunk_tokens=chunk)

    @classmethod
    def _tiny(cls, hidden, layers, max_tokens, attention="sdpa", moe=False):
        """A random text model with layer_types alternating linear_attention / full_attention, byte vocabulary; `moe`:
        the qwen3_5_moe layout (8 experts, 2 routed per token, a shared expert) with the same mixers."""
        config = cls._tiny_config(hidden, layers, max_tokens, attention, moe=moe)
        return Qwen3_5MoeTextModel(config) if moe else Qwen3_5TextModel(config)

    @staticmethod
    def _tiny_config(hidden, layers, max_tokens, attention="sdpa", vocab=TINY_VOCAB, moe=False):
        if hidden % 4:
            raise ValueError("tiny-qwen3_5 needs hidden_size divisible by 4")
        common = dict(
            vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=2, head_dim=hidden // 4,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=hidden // 4, linear_value_head_dim=hidden // 4,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention" if i % 2 == 0 else "full_attention" for i in range(layers)],
            max_position_embeddings=max_tokens, attention_dropout=0.0, use_cache=False, tie_word_embeddings=False)
        if moe:
            text_config = Qwen3_5MoeTextConfig(moe_intermediate_size=hidden // 2, shared_expert_intermediate_size=hidden // 2,
                                               num_experts=8, num_experts_per_tok=2, **common)
        else:
            text_config = Qwen3_5TextConfig(intermediate_size=2 * hidden, **common)
        text_config._attn_implementation = "sdpa" if attention == "flex" else attention  # flex is set after construction
        return text_config

    @staticmethod
    def snapshot_dir(repo):
        """The cached snapshot of a repo's config and tokenizer files, without network access; None when absent."""
        from huggingface_hub import snapshot_download
        cache = os.environ.get("HF_HUB_CACHE") or str(Path(__file__).resolve().parent.parent / ".cache" / "huggingface" / "hub")
        try:
            return snapshot_download(repo, cache_dir=cache, local_files_only=True,
                                     allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt"])
        except Exception:
            return None

    @property
    def text_model(self):
        """The Qwen3_5TextModel, through the peft wrapper when LoRA is applied."""
        return self.lm.get_base_model() if hasattr(self.lm, "get_base_model") else self.lm

    def get_input_embeddings(self):
        return self.text_model.embed_tokens

    def get_output_embeddings(self):
        return None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False})

    def apply_lora(self, rank):
        from peft import LoraConfig, get_peft_model
        self.lm = get_peft_model(self.lm, LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
                                                     target_modules=LORA_TARGETS, bias="none"))

    def enable_fast(self, merge=True, graphs=True, compile=True, fp8=False, fp4=False):
        """Serving-only fast paths. `graphs`: on CUDA with sdpa or eager attention, the level passes run as CUDA graphs
        over bucketed shapes (LevelGraphs; inference only, encode_many keeps the plain path when gradients are
        enabled). `merge`: the LoRA adapters are folded into the bf16 weights (one GEMM per projection instead of
        three, about 3 ms per level on the 9B). The fold rounds the adapter's update once to the weights' bf16 grid;
        measured on the benchmark requests that moves the probabilities by as much as the graphs' padding does
        (docs/phase4/latency.md, "Equality"), so it is on by default and switchable here. `compile`: on CUDA the
        per-layer glue runs as torch.compile'd kernels (`Fused`; inference only). `fp8` (CUDA, needs `compile`): the
        layers' projections become FP8Linear, which moves the numbers beyond bf16 rounding; off by default.
        `fp4` (Blackwell, qwen3_5_moe NVFP4 checkpoints): the packed experts and shared experts run on the fp4 tensor
        cores with fp4 activations (`fp4_linear`) instead of dequantising to bf16 per call; off by default."""
        self.graphs = None  # earlier captures replay kernels bound to the weights and glue of their moment
        if merge and hasattr(self.lm, "merge_and_unload"):
            self.lm = self.lm.merge_and_unload()
        device = self.text_model.embed_tokens.weight.device
        self.fast_causal = True
        self.single_pass = True
        self.fused = Fused(serve=True) if compile and device.type == "cuda" else None
        for module in self.text_model.modules():
            if isinstance(module, (NVFP4Linear, NVFP4Experts)):
                module.set_mode(fp4, self.fused)
        if fp8:
            if self.fused is None:
                raise ValueError("fp8 needs the compiled glue on CUDA (enable_fast(compile=True))")
            skipped = set()
            for layer in self.text_model.layers[: self.config.num_hidden_layers]:
                for module in (layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn, layer.mlp):
                    for name in FP8_TARGETS:
                        linear = getattr(module, name, None)
                        if isinstance(linear, nn.Linear):
                            setattr(module, name, FP8Linear(linear, self.fused))
                        elif linear is not None:
                            skipped.add(f"{name} ({type(linear).__name__})")
            if skipped:
                log.warning("fp8: left in bf16, not a plain nn.Linear (unmerged LoRA?): %s", ", ".join(sorted(skipped)))
        if self.fused is not None:
            _merge_projections(self.text_model)
        if device.type == "cuda" and self.attention == "sdpa":
            # One chunk per state: only a chunk with cached keys needs a mask (the masked path is the 6x slower
            # memory-efficient kernel over repeated kv), and the flash kernel needs no score matrix, which is what the
            # chunking was for. ponytail: 64k covers --max-tokens 65536 with about 8 GB of activations on the 4B;
            # longer states fall back to masked chunks.
            self.state_chunk_tokens = max(self.state_chunk_tokens, SERVE_CHUNK_TOKENS)
        if graphs and device.type == "cuda" and self.attention != "flex":
            self.graphs = LevelGraphs(self.text_model, chunk_rule(device, self.allow_fla), self.fused)

    def _flash(self, device, dtype):
        """Whether sdpa runs as the flash kernel here: CUDA in bf16/fp16 (the plain path in training and evaluate
        then takes the same causal level 0 as serving's `fast_causal`; fp32, the tiny tests, keeps the masked kernel)."""
        return self.attention == "sdpa" and torch.device(device).type == "cuda" and dtype in (torch.float16, torch.bfloat16)

    def _mask(self, kv_valid, cached, length, dtype):
        """The attention mask of one level batch: key kv is visible to query q when valid and kv <= cached + q.
        None on the serving fast path (`fast_causal`, set by enable_fast; inference only) under sdpa when nothing is
        cached: the rule is then plain causal over a right-padded batch (a valid query sees keys <= q, all valid),
        which HF's sdpa path runs as `is_causal=True` without a mask, i.e. the flash kernel with GQA in place instead
        of `repeat_kv` plus the masked memory-efficient kernel (4k tokens: 0.8 ms against 4.2; 16k: 11 against 62,
        docs/phase4/latency-track2.md), and likewise on the plain path whenever sdpa is the flash kernel (`_flash`:
        CUDA in bf16/fp16, training and evaluate); fp32 keeps the masked kernel."""
        if self.attention == "flex":
            return flex_mask(kv_valid, cached, length)
        if cached == 0 and ((self.attention == "sdpa" and self.fast_causal and not torch.is_grad_enabled()) or self._flash(kv_valid.device, dtype)):
            return None
        device = kv_valid.device
        keys, queries = torch.arange(kv_valid.shape[1], device=device), torch.arange(length, device=device)
        allowed = kv_valid[:, None, None, :] & (keys[None, None, None, :] <= cached + queries[None, None, :, None])
        if self.attention == "sdpa":
            return allowed
        return torch.zeros(allowed.shape, device=device, dtype=dtype).masked_fill_(~allowed, torch.finfo(dtype).min)

    def prepare_many(self, packs, prefixes=None):
        """Encode several packs as one batch and hold each result for the next `encode` call on that pack
        (DecisionModel.forward_many keeps its per-pack readout and calls encode through the ordinary forward).
        `prefixes`: per pack a PrefixSnapshot of its state (`encode_prefixes`) or None."""
        # ponytail: keyed by id(pack); every entry is consumed by the forward that follows, so nothing outlives a batch.
        self._prepared = dict(zip(map(id, packs), self.encode_many(packs, prefixes=prefixes)))

    def encode(self, packed, inputs_embeds=None, directionality="causal", extra_padding=0, prefix=None):
        """Final-norm hidden states [n, d] at the packed positions (module docstring); `encode_many` of one pack."""
        if inputs_embeds is None and id(packed) in self._prepared:
            return self._prepared.pop(id(packed))
        return self.encode_many([packed], [inputs_embeds], directionality, extra_padding, [prefix])[0]

    def encode_prefix(self, packed):
        """`encode_prefixes` of one pack."""
        return self.encode_prefixes([packed])[0]

    def encode_prefixes(self, packs):
        """The state pass of several packs as one batch, one PrefixSnapshot per pack: the per-layer states after the
        state's last token (DeltaNet conv tail and recurrent matrix, attention keys and values), which
        `encode_many(packs, prefixes=...)` continues from without running the state again. The rows are copied out of
        the level batch (and out of the graph buffers), so a snapshot outlives the call."""
        return self._levels(packs, snapshot=True)

    def encode_many(self, packs, inputs_embeds=None, directionality="causal", extra_padding=0, prefixes=None):
        """Final-norm hidden states, one [n_i, d] tensor per pack, computed level by level over all packs at once: the
        state prefixes as one right-padded batch in chunks of `state_chunk_tokens`, then every branch level of every
        pack merged into one batch (module docstring).

        `inputs_embeds` is a list parallel to `packs` (None or [1, n_i, d] entries); `extra_padding` adds that many
        padding columns to every batch (tests of right-padding invariance); `prefixes` (parallel to `packs`) gives
        packs whose state pass is a PrefixSnapshot already: level 0 runs only for the others, and a pack with a
        snapshot gets zeros at its state positions (nothing reads them)."""
        if directionality != "causal":
            raise ValueError("qwen3_5 backbones are causal (the DeltaNet layers are a recurrence); directionality must be causal")
        return self._levels(packs, inputs_embeds, extra_padding, prefixes)

    def _levels(self, packs, inputs_embeds=None, extra_padding=0, prefixes=None, snapshot=False):
        lm = self.text_model
        device = lm.embed_tokens.weight.device
        embeds, all_positions, levels, parents, offset = [], [], [], [], 0
        for packed, given in zip(packs, inputs_embeds or [None] * len(packs)):
            n = packed.token_count
            e = lm.embed_tokens(packed.input_ids[:, :n].to(device))[0] if given is None else given[0, :n]
            if packed.images is not None:
                # An image state: the placeholder rows become the vision tower's merged features (janus.vision).
                if self.vision is None:
                    raise ValueError("the request has images but the backbone was loaded without them (ModelConfig.images)")
                mask = packed.input_ids[0, :n].to(device) == self.vision.image_token_id
                e = e.masked_scatter(mask[:, None].expand_as(e), self.vision.features(*packed.images).to(e.dtype))
            # [1, n] text positions or [3, n] (t, h, w) with images; rotary_emb expands text positions to 3 axes itself.
            all_positions.append(packed.position_ids[:, :n].to(device).expand(3, -1))
            embeds.append(e)
            parents.append(packed.parents.tolist())
            tree = tree_levels(packed.segment_ids, packed.parents)
            # ponytail: one question (one depth-1 segment right after the state) runs state + block as one causal level-0
            # row, the DeltaNet recurrence simply continuing: 3 level passes -> 2. Serving only (enable_fast, no
            # gradients), text states without a snapshot, at most BRANCH_CHUNK_ROWS leaves (one leaf pass), not for
            # encode_prefixes; everything else keeps the three levels. The merged row is no prefix-cache snapshot
            # (janus.server.Worker leaves such states uncached).
            if (self.single_pass and not torch.is_grad_enabled() and not snapshot and packed.images is None
                    and not (prefixes and prefixes[len(parents) - 1] is not None)
                    and len(tree) > 1 and len(tree[1]) == 1 and tree[1][0][1] == tree[0][0][2]
                    and (len(tree) == 2 or len(tree[2]) <= BRANCH_CHUNK_ROWS)):
                block = tree[1][0][0]
                tree = [[(0, tree[0][0][1], tree[1][0][2])]] + tree[2:]
                parents[-1] = [0 if p == block else p for p in parents[-1]]
            for depth, segments in enumerate(tree):
                if depth == len(levels):
                    levels.append([])
                levels[depth] += [(len(parents) - 1, s, start + offset, end + offset) for s, start, end in segments]
            offset += n
        counts = [e.shape[0] for e in embeds]
        embeds, all_positions = torch.cat(embeds), torch.cat(all_positions, dim=1)
        dtype = embeds.dtype
        rule = chunk_rule(device, self.allow_fla)
        checkpointing = bool(getattr(lm, "gradient_checkpointing", False) and lm.training and torch.is_grad_enabled())
        graphs = self.graphs if self.graphs is not None and not torch.is_grad_enabled() else None
        if graphs is not None:
            graphs.begin()
        serving = self.fused is not None and not torch.is_grad_enabled()
        if torch.is_grad_enabled() and self.glue is None and COMPILE_GLUE and device.type == "cuda":
            self.glue = Fused()
            fuse_lora(lm, self.glue)
        fused = self.fused if serving else (self.glue if torch.is_grad_enabled() else None)
        flex = self.attention == "flex"
        packed = BRANCH_PACKED and graphs is None and not serving and not flex  # the plain path's branch levels (Packed)
        # sdpa in bf16/fp16 on CUDA is the flash kernel (no score matrix), so the state runs as one chunk without a mask.
        chunk = max(self.state_chunk_tokens, embeds.shape[0]) if self._flash(device, dtype) else self.state_chunk_tokens
        layers = lm.layers[: lm.config.num_hidden_layers]
        if hasattr(layers[0].mlp, "experts"):  # qwen3_5_moe: the HF experts (bf16 weights) take the same kernel choice
            kernel = moe_kernel(device)
            if kernel == "fused":  # NVFP4Experts only; the bf16 experts take the dequantise paths' device rule
                kernel = "grouped" if device.type == "cuda" and torch.cuda.get_device_capability(device) >= (9, 0) else "bmm"
            lm.config._experts_implementation = "grouped_mm" if kernel == "grouped" else "eager"
        values, positions, replays = [], [], []

        def batch_index(spans, state=False):
            """Right-padded [B, L] token indices and their validity for the given (start, end) spans; under `graphs`
            both dimensions are padded to their buckets (extra rows are all padding and read parent row 0)."""
            length = max(end - start for start, end in spans) + extra_padding
            rows = len(spans)
            if graphs is not None:
                bucket = length_bucket(length, state) if not state or length <= KV_MAX else None  # a longer state is eager: real length
                rows, length = batch_bucket(rows) or rows, bucket or length
            # Built on the CPU and moved once: two copies per level instead of two kernels per row, and the valid
            # entries' flat offsets are known here, so reading a level's outputs needs no boolean-mask sync.
            index = torch.zeros(rows, length, dtype=torch.long)
            valid = torch.zeros(rows, length, dtype=torch.bool)
            flat, tokens = [], []
            for row, (start, end) in enumerate(spans):
                index[row, : end - start] = torch.arange(start, end)
                valid[row, : end - start] = True
                flat.append(torch.arange(row * length, row * length + end - start))
                tokens.append(torch.arange(start, end))
            # Serving copies without a stream sync, so the next level's inputs are built while the previous one runs.
            move = lambda t: t.to(device, non_blocking=graphs is not None)
            return move(index), move(valid), move(torch.cat(flat)), torch.cat(tokens)

        def run(index, valid, flat, tokens, mask, states, parent, keep, depth):
            """One batch through all layers (a graph replay when the shape has one); returns the new states."""
            x = embeds[index] * valid[..., None].to(dtype)
            pos = all_positions[:, index] * valid
            out = replay_level(lm, x, pos, valid, mask, states, rule, parent, keep, graphs, depth) if graphs is not None else None
            hidden, states = out if out is not None else run_level(lm, x, pos, valid, mask, states, rule, checkpointing, parent, keep, depth, flex, fused)
            replays.append(out is not None)
            values.append(hidden.reshape(-1, hidden.shape[-1]).index_select(0, flat))
            positions.append(tokens)
            return states

        # Level 0: every state without a snapshot as one row, in chunks along the sequence; a row past its end is all
        # padding (a no-op). Its states are per row: the DeltaNet states and the row's own keys, accumulated over chunks.
        # ponytail: rows are padded to the longest state of the batch; callers that batch states of similar length waste less.
        prefixes = list(prefixes or [None] * len(packs))
        misses = [i for i, p in enumerate(prefixes) if p is None]
        rows = None
        if misses:
            spans = [(levels[0][i][2], levels[0][i][3]) for i in misses]
            longest = max(end - start for start, end in spans)
            states, key_valid = [() for _ in layers], None
            for begin in range(0, longest, chunk):
                index, valid, flat, tokens = batch_index([(min(start + begin, end), min(start + begin + chunk, end)) for start, end in spans], state=True)
                key_valid = valid if key_valid is None else torch.cat([key_valid, valid], dim=1)
                mask = self._mask(key_valid, key_valid.shape[1] - valid.shape[1], valid.shape[1], dtype)
                states = run(index, valid, flat, tokens, mask, states, None, snapshot or len(levels) > 1 or begin + chunk < longest, 0)
            rows = (states, key_valid)
        if snapshot:
            return [PrefixSnapshot([tuple(t[j:j + 1, :, :end - start].clone() if layer.layer_type != "linear_attention" else t[j:j + 1].clone()
                                          for t in s) for s, layer in zip(states, layers)], end - start,
                                   [layer.layer_type != "linear_attention" for layer in layers])
                    for j, (start, end) in enumerate(spans)]
        if len(levels) == 1:
            history = []
        else:
            history = [self._assemble(rows, misses, prefixes, graphs, bool(replays and replays[-1]))]  # (states per layer, key validity [n, S])
            if graphs is not None:
                history[0] = graphs.widen(*history[0])
        # Branch levels: every row continues its parent's DeltaNet states (index_select) and attends to the keys of
        # its ancestor rows in place, through a grouping plan per ancestor level (branch_attention); a level's own
        # keys and values are kept for the levels below it.
        # ponytail: the branch levels run in chunks of BRANCH_CHUNK_ROWS level-1 rows, each with its descendants (the
        # chunks are independent: a row reads its ancestors only). A kept block row's DeltaNet states are 63 MB on
        # the A3B (30 layers of [32, 128, 128] fp32): 8 GB at 128 Score questions, which did not fit beside the
        # 22 GB of weights on a 32 GB card; 32 rows are 2 GB, and 32 is the widest graph bucket, so a wide level
        # replays graphs instead of one eager pass. The upgrade is a chunk size from the free memory.
        block_of = {}
        for depth, segments in enumerate(levels[1:], 1):
            for i, s, _, _ in segments:
                block_of[(i, s)] = (i, s) if depth == 1 else block_of[(i, parents[i][s])]
        blocks = levels[1] if len(levels) > 1 else []
        all_levels = levels
        chunk_rows = BRANCH_CHUNK_ROWS if graphs is not None else max(len(blocks), 1)  # serving only: training and evaluate run one pass
        for first in range(0, len(blocks), chunk_rows):
            chunk = {(i, s) for i, s, _, _ in blocks[first:first + chunk_rows]}
            levels = [all_levels[0]] + [[seg for seg in segments if block_of[(seg[0], seg[1])] in chunk] for segments in all_levels[1:]]
            levels = levels[:next((d for d, segments in enumerate(levels) if not segments), len(levels))]
            history = history[:1]
            self._branch_levels(levels, parents, history, batch_index, run, device, graphs, packed)
        if fused is not None:
            fused.check()
        hidden = embeds.new_zeros(embeds.shape[0], embeds.shape[-1])
        if positions:
            hidden = hidden.index_copy(0, torch.cat(positions).to(device), torch.cat(values))
        return list(hidden.split(counts))

    def _branch_levels(self, levels, parents, history, batch_index, run, device, graphs, packed=False):
        """The branch levels (`levels[1:]`, every row of each a segment of one pack) below the level-0 rows in
        `history[0]`, appending each level's (states, validity) to `history`. `packed`: the plan of a level is a
        Packed (one varlen call per layer) instead of the grouped pieces' plan."""
        packs = history[0][1].shape[0]
        # per level, the rows' token counts (the packed plan is built here; serving needs none, and no device sync)
        lengths = [history[0][1].sum(1).tolist() if packed else None]
        move = lambda t: t.to(device, non_blocking=graphs is not None)
        kinds = [layer.layer_type for layer in self.text_model.layers[: self.text_model.config.num_hidden_layers]]
        ancestry = torch.zeros(packs, 0, dtype=torch.long)
        row_of = {(i, 0): i for i in range(packs)}
        for depth, segments in enumerate(levels[1:], 1):
            index, valid, flat, tokens = batch_index([(start, end) for _, _, start, end in segments])
            batch = index.shape[0]
            parent = torch.tensor([row_of[(i, parents[i][s])] for i, s, _, _ in segments] + [0] * (batch - len(segments)))
            anc = torch.cat([ancestry[parent], parent[:, None]], dim=1)
            lengths.append([end - start for _, _, start, end in segments] + [0] * (batch - len(segments)))
            plan = []
            for level in range(depth) if not packed else ():
                a = anc[:, level]
                counts_per = torch.bincount(a, minlength=history[level][1].shape[0])
                per = int(counts_per.max())
                per = (batch_bucket(per) or per) if graphs is not None else per
                order = torch.argsort(a, stable=True)
                slot = torch.empty_like(a)
                slot[order] = torch.arange(batch) - (counts_per.cumsum(0) - counts_per)[a[order]]
                slots = torch.zeros(counts_per.shape[0], per, dtype=torch.long)
                slots[a, slot] = torch.arange(batch)
                plan += [history[level][1], move(slots), move(a * per + slot)]
            if packed:
                # Every row's keys in order: its ancestor at each level (that level's valid tokens), then itself.
                shapes = [history[level][1].shape for level in range(depth)] + [valid.shape]
                offsets = torch.tensor([0] + [b * l for b, l in shapes]).cumsum(0)
                rows = [torch.cat([torch.arange(offsets[level] + a * shapes[level][1], offsets[level] + a * shapes[level][1] + lengths[level][a])
                                   for level, a in enumerate((*anc[b].tolist(), b))]) for b in range(batch)]
                counts = torch.tensor([len(r) for r in rows])
                cu_k = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
                cu_q = F.pad(torch.tensor(lengths[depth]).cumsum(0), (1, 0)).to(torch.int32)
                plan = Packed(torch.cat(rows).to(device), flat, cu_q.to(device), cu_k.to(device), max(lengths[depth]), int(counts.max()))
            states = [history[depth - 1][0][j] if kind == "linear_attention" else tuple(t for level in range(depth) for t in history[level][0][j])
                      for j, kind in enumerate(kinds)]
            states = run(index, valid, flat, tokens, plan if packed else tuple(plan), states, move(parent), depth + 1 < len(levels), depth)
            history.append((states, valid))
            ancestry = anc
            row_of = {(i, s): row for row, (i, s, _, _) in enumerate(segments)}

    def _assemble(self, rows, misses, prefixes, graphs, replayed=False):
        """The level-0 rows in pack order, (states per layer, key validity [n, S]): the batch of misses `rows` as it
        is when every pack missed and the pass `replayed` a graph (or there are none), else every pack's row (from
        the batch or its snapshot) copied into fresh tensors padded to the longest state, or under `graphs` into the
        level-0 buffer set the next level's graph reads (also for an eager long state, so its branch levels replay)."""
        if len(misses) == len(prefixes) and (graphs is None or replayed):
            return rows
        layers = self.text_model.layers[: self.text_model.config.num_hidden_layers]
        sources = []
        for i, prefix in enumerate(prefixes):
            if prefix is not None:
                sources.append((prefix.states, prefix.tokens))
            else:
                j = misses.index(i)
                states, key_valid = rows
                # Clones: a miss row may live in the buffer set this batch writes into.
                sources.append(([tuple(t[j:j + 1].clone() for t in s) for s in states], int(key_valid[j].sum())))
        n, longest = len(sources), max(tokens for _, tokens in sources)
        device = sources[0][0][0][0].device
        batch, length = n, longest
        views = False
        if graphs is not None:
            batch, length = batch_bucket(n), length_bucket(longest, True)
            if batch is not None and length is not None and batch * kv_capacity(length) <= STATE_BUFFER_TOKENS:
                views = graphs.ensure(0, batch, sources[0][0], length)
            batch, length = batch or n, length or longest
        if views is False:
            views = [tuple(torch.zeros(batch, *t.shape[1:-2], length, t.shape[-1], dtype=t.dtype, device=device) if layer.layer_type != "linear_attention"
                           else torch.zeros(batch, *t.shape[1:], dtype=t.dtype, device=device) for t in s)
                     for s, layer in zip(sources[0][0], layers)]
        key_valid = torch.zeros(batch, length, dtype=torch.bool, device=device)
        for i, (states, tokens) in enumerate(sources):
            key_valid[i, :tokens] = True
            for (a, b), (src_a, src_b), layer in zip(views, states, layers):
                if layer.layer_type == "linear_attention":
                    a[i].copy_(src_a[0])
                    b[i].copy_(src_b[0])
                else:
                    a[i, :, :tokens].copy_(src_a[0, :, :tokens])
                    b[i, :, :tokens].copy_(src_b[0, :, :tokens])
        return views, key_valid


class PrefixSnapshot:
    """The per-layer states after a state's last token (HybridBackbone.encode_prefixes): DeltaNet (conv tail, recurrent
    matrix) and attention (keys, values [1, H, tokens, D]) per layer, on the device. `bytes` is its memory, `kv_bytes`
    the part that grows with the token count (the attention keys and values)."""

    def __init__(self, states, tokens, attention):
        self.states, self.tokens = states, tokens
        self.bytes = sum(t.numel() * t.element_size() for s in states for t in s)
        self.kv_bytes = sum(t.numel() * t.element_size() for s, a in zip(states, attention) if a for t in s)
