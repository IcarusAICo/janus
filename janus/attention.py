"""Tree attention layout: one definition shared by the dense reference and the block-sparse kernel."""

import torch
from torch.nn.attention.flex_attention import BlockMask


DIRECTIONALITIES = ("causal", "block")


def _rule(q, kv, sq, sk, parents, directionality):
    """The one attention rule. A key is visible when it is state, the query's own segment, or the query's parent segment.

    causal: additionally kv <= q (the Phase 1 layout, unchanged).
    block:  no order constraint, so the state, each question block, and each leaf are fully connected internally
            (spec WP5a); cross-segment edges are exactly the causal ones. Padding (segment -1) attends only to itself.
    """
    if directionality not in DIRECTIONALITIES:
        raise ValueError(f"directionality must be one of {DIRECTIONALITIES}")
    parent = parents[sq.clamp(min=0)]
    ok = (sk >= 0) & ((sk == 0) | (sk == sq) | (sk == parent))
    if directionality == "causal":
        ok = ok & (kv <= q)
    return torch.where(sq < 0, q == kv, ok)


def tree_mask_mod(segment_ids, parents, directionality="causal"):
    """Return a FlexAttention mask_mod. Padded positions (segment -1) attend only to themselves."""
    if directionality not in DIRECTIONALITIES:
        raise ValueError(f"directionality must be one of {DIRECTIONALITIES}")

    def mask_mod(b, h, q, kv):
        return _rule(q, kv, segment_ids[q], segment_ids[kv], parents, directionality)
    return mask_mod


def _mask_rows(segment_ids, parents, start, stop, directionality="causal"):
    """Rows [start, stop) of the dense mask, computed on the device of segment_ids."""
    device = segment_ids.device
    q = torch.arange(start, stop, device=device)[:, None]
    kv = torch.arange(segment_ids.shape[0], device=device)[None, :]
    return _rule(q, kv, segment_ids[start:stop, None], segment_ids[None, :], parents, directionality)


def dense_mask(segment_ids, parents, directionality="causal"):
    return _mask_rows(segment_ids, parents, 0, segment_ids.shape[0], directionality)


def padded_length(n, block_size=128):
    return ((n + block_size - 1) // block_size) * block_size


def _ordered(dense):
    """Per row: count of True entries and their column indices first, ascending (as BlockMask expects)."""
    count = dense.sum(-1, dtype=torch.int32)
    indices = dense.to(torch.int32).argsort(dim=-1, descending=True, stable=True).to(torch.int32)
    return count[None, None], indices[None, None]


def build_block_mask(segment_ids, parents, device, block_size=128, chunk_rows=2048, directionality="causal"):
    """Exact block-sparse mask from the segment metadata without materialising the n-by-n mask at once.

    torch's create_block_mask evaluates the dense mask in one piece (9 GB at 30k tokens), so the
    rows are reduced to block granularity a chunk at a time. Full blocks skip the mask_mod in the kernel.
    """
    n = segment_ids.shape[0]
    total = padded_length(n, block_size)
    seg = torch.cat([segment_ids, torch.full((total - n,), -1, dtype=torch.long)]).to(device)
    parents = parents.to(device)
    blocks = total // block_size
    full = torch.zeros((blocks, blocks), dtype=torch.bool, device=device)
    partial = torch.zeros((blocks, blocks), dtype=torch.bool, device=device)
    chunk_rows = max(block_size, chunk_rows // block_size * block_size)
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        rows = _mask_rows(seg, parents, start, stop, directionality).view(-1, block_size, blocks, block_size)
        lo, hi = start // block_size, stop // block_size
        full[lo:hi] = rows.all(dim=3).all(dim=1)
        partial[lo:hi] = rows.any(dim=3).any(dim=1) & ~full[lo:hi]
    kv_num, kv_idx = _ordered(partial)
    full_num, full_idx = _ordered(full)
    mask = BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=block_size,
                                    mask_mod=tree_mask_mod(seg, parents, directionality), seq_lengths=(total, total))
    return mask, total


def pad_packed(packed, total, pad_id):
    extra = total - packed.token_count
    input_ids = torch.cat([packed.input_ids, torch.full((1, extra), pad_id, dtype=packed.input_ids.dtype)], dim=1)
    position_ids = torch.cat([packed.position_ids, torch.zeros((1, extra), dtype=packed.position_ids.dtype)], dim=1)
    return input_ids, position_ids
