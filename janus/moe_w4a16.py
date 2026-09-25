"""Fused W4A16 expert GEMMs for NVFP4Experts (janus.hybrid): a Triton kernel that reads the packed e2m1 weights and
their e4m3 block scales inside the GEMM, so no bf16 expert stack is ever materialised (the dequantise-per-call path
decodes 1.6 GB per layer per level pass on the A3B, about 1 s per request; here the GEMM reads the 0.56 bytes per
weight once). One launch computes silu(gate) * up for the tokens of every expert (the tile grid is sorted by expert,
`align`), a second the down projection times the routing weight; the per-token sum over the routed experts is a
reshape-sum in fp32. Everything is shape-static and sync-free (the tile count is read on the device), so a level pass
that uses it can be captured as a CUDA graph. Any card Triton supports (sm_80+); serving and forward only (no autograd).

Numerics: e2m1 code times (e4m3 block scale times the fp32 global scale) is exact in fp32 and rounded once to bf16, the
same value the dequantise path feeds its GEMM; the bf16 dot accumulates in fp32 (tl.dot), so the two paths agree to
bf16 accumulation order (test_fused_w4a16_moe_matches_dequant_path).
"""
import torch
import triton
import triton.language as tl



@triton.jit
def _decode_e2m1(nibble):
    """uint8 e2m1 codes (0..15) to fp32 times 2^-14: bits 2..0 placed as an fp16 exponent/mantissa give the value
    scaled by 2^-14 exactly (a denormal for the 0 and 0.5 codes), the sign goes to bit 15. The 2^14 is folded into
    the scale by the caller (exact: a power of two)."""
    bits = ((nibble & 7).to(tl.uint16) << 9) | ((nibble & 8).to(tl.uint16) << 12)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _expand_scales(s, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    """e4m3 block scales [BLOCK_K/16, BLOCK_N] (one contiguous, vectorised load) to fp32 [BLOCK_K/2, BLOCK_N]: each
    scale repeated for the 8 byte rows (16 elements) of its block. A broadcast the compiler folds; the same tile
    loaded with a `k // 8` index costs 30 to 60% more (scalar loads)."""
    return tl.reshape(tl.broadcast_to(tl.reshape(s.to(tl.float32), (BLOCK_K // 16, 1, BLOCK_N)), (BLOCK_K // 16, 8, BLOCK_N)), (BLOCK_K // 2, BLOCK_N))


@triton.jit
def _w4a16_moe_kernel(a_ptr, b_ptr, bs_ptr, bg_ptr, c_ptr, w_ptr, sorted_ids_ptr, expert_ids_ptr, total_ptr,
                      N, K, num_valid, stride_am, stride_be, stride_bn, stride_bse, stride_bsn, stride_bge, stride_cm,
                      TOP_K: tl.constexpr, GATED: tl.constexpr, MUL_WEIGHT: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """C[assignment, n] = A[row(assignment), :] @ W[expert(tile), n, :]^T (times the routing weight of the assignment
    under MUL_WEIGHT), or silu(gate) * up over the row pairs (n, N + n) of a gate_up stack under GATED. A row of A is
    assignment // TOP_K (the token, first GEMM) or the assignment itself (second GEMM). W is [E, N_total, K/2] uint8
    (low nibble = even k), its scales [E, N_total, K/16] e4m3 and [E, N_total] fp32 (times 2^14, see _decode_e2m1).
    The byte tile [BLOCK_K/2, BLOCK_N] is loaded once (contiguous along k, vectorised) and contracted as two dots, the
    even elements (low nibbles) against A's even columns and the odd against the odd: a permutation of K."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    if pid_m * BLOCK_M >= tl.load(total_ptr):
        return
    expert = tl.load(expert_ids_ptr + pid_m)
    ids = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    mask_m = ids < num_valid
    rows = ids // TOP_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_kb = tl.arange(0, BLOCK_K // 2)
    a_ptrs = a_ptr + rows[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + expert * stride_be + offs_n[None, :] * stride_bn + offs_kb[:, None]
    bs_ptrs = bs_ptr + expert * stride_bse + offs_n[None, :] * stride_bsn + tl.arange(0, BLOCK_K // 16)[:, None]
    g = tl.load(bg_ptr + expert * stride_bge + offs_n)[None, :]
    if GATED:
        g2 = tl.load(bg_ptr + expert * stride_bge + N + offs_n)[None, :]
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        a_even, a_odd = tl.split(tl.reshape(a, (BLOCK_M, BLOCK_K // 2, 2)))
        b = tl.load(b_ptrs)
        s = _expand_scales(tl.load(bs_ptrs), BLOCK_K, BLOCK_N) * g
        acc = tl.dot(a_even, (_decode_e2m1(b & 15) * s).to(tl.bfloat16), acc)
        acc = tl.dot(a_odd, (_decode_e2m1(b >> 4) * s).to(tl.bfloat16), acc)
        if GATED:
            b = tl.load(b_ptrs + N * stride_bn)
            s = _expand_scales(tl.load(bs_ptrs + N * stride_bsn), BLOCK_K, BLOCK_N) * g2
            acc2 = tl.dot(a_even, (_decode_e2m1(b & 15) * s).to(tl.bfloat16), acc2)
            acc2 = tl.dot(a_odd, (_decode_e2m1(b >> 4) * s).to(tl.bfloat16), acc2)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K // 2
        bs_ptrs += BLOCK_K // 16
    if GATED:
        acc = acc * tl.sigmoid(acc) * acc2
    if MUL_WEIGHT:
        acc = acc * tl.load(w_ptr + ids, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    tl.store(c_ptr + ids[:, None] * stride_cm + offs_n[None, :], acc.to(tl.bfloat16), mask=mask_m[:, None])


def align(top_k_index, num_experts, block):
    """The tile plan of one MoE call, without a host sync: (sorted_ids [tiles * block] of assignment ids grouped by
    expert and padded per expert to `block` with the sentinel S*k, expert_ids [tiles] per tile, total [1] the padded
    row count on the device). `tiles` is the worst case (every expert padded), so the shapes are static; tiles past
    `total` exit at once in the kernel."""
    S, k = top_k_index.shape
    device = top_k_index.device
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(num_experts, dtype=torch.long, device=device).index_add_(0, flat, torch.ones_like(flat))
    padded = (counts + block - 1) // block * block
    cum = padded.cumsum(0)
    first = (cum - padded)[flat[order]] + torch.arange(S * k, device=device) - (counts.cumsum(0) - counts)[flat[order]]
    tiles = (S * k + num_experts * (block - 1) + block - 1) // block
    sorted_ids = torch.full((tiles * block,), S * k, dtype=torch.long, device=device)
    sorted_ids[first] = order
    expert_ids = torch.searchsorted(cum, torch.arange(tiles, device=device) * block, right=True)
    return sorted_ids, expert_ids, cum[-1:]


def _launch(a, packed, scale, global_scale, out, weights, plan, top_k, gated, block_m, block_n=64, block_k=128, num_warps=4, num_stages=3):
    sorted_ids, expert_ids, total = plan
    N = out.shape[1]
    K = a.shape[1]
    assert packed.shape[2] * 2 == K and K % block_k == 0 and N % block_n == 0 and block_k % 16 == 0, (packed.shape, K, N)
    assert a.stride(1) == 1 and out.stride(1) == 1 and packed.stride(2) == 1 and scale.stride(2) == 1, "inner strides must be 1"
    grid = (sorted_ids.shape[0] // block_m * (N // block_n),)
    _w4a16_moe_kernel[grid](a, packed, scale, global_scale, out, weights if weights is not None else out, sorted_ids, expert_ids, total,
                            N, K, a.shape[0] * top_k, a.stride(0), packed.stride(0), packed.stride(1), scale.stride(0), scale.stride(1),
                            global_scale.stride(0), out.stride(0), TOP_K=top_k, GATED=gated, MUL_WEIGHT=weights is not None,
                            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages)


def fused_moe(x, top_k_index, top_k_weights, gate_up, gate_up_scale, gate_up_global, down, down_scale, down_global, plan=None):
    """NVFP4Experts.forward through the fused kernels: x [S, H] bf16, the routing [S, k], the packed stacks
    (`NVFP4Experts` buffers; the global scales [E, N, 1] fp32, scaled by 2^14 here per call: cheap, E*N floats).
    Returns [S, H] in x's dtype."""
    S, k = top_k_index.shape
    E, I, H = down.shape[0], down.shape[2] * 2, x.shape[1]
    dtype = x.dtype
    x = x.to(torch.bfloat16)
    # Tile rows by the mean rows per expert (measured on the 5090, docs/phase4/a3b-fast-serving.md): small tiles with a
    # deep K block when a few tokens hit each expert (weight-bandwidth-bound), 64-row tiles with K=64 once tens do.
    per_expert = S * k / E
    block_m, block_k = (16, 128) if per_expert < 8 else (32, 128) if per_expert < 48 else (64, 64)
    if plan is None:
        plan = align(top_k_index, E, block_m)
    h = torch.empty(S * k, I, dtype=torch.bfloat16, device=x.device)
    _launch(x, gate_up, gate_up_scale, (gate_up_global * 16384.0).reshape(E, -1), h, None, plan, k, True, block_m, 64, block_k)
    y = torch.empty(S * k, H, dtype=torch.bfloat16, device=x.device)
    _launch(h, down, down_scale, (down_global * 16384.0).reshape(E, -1), y, top_k_weights.reshape(-1).to(torch.bfloat16), plan, 1, False, block_m, 64, block_k)
    return y.view(S, k, H).sum(1).to(dtype)
