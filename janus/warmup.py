"""LLM2Vec-style masked-next-token warm-up on state text under the block-bidirectional mask (spec WP5a).

The sequence is the packed state text alone (segment 0). A random 15% of positions 1..n-1 are replaced by the mask
token; the loss is the cross-entropy of each original token read from the hidden state one position earlier, which
keeps the pretrained next-token geometry while the attention becomes bidirectional. The output projection is the
backbone's LM head when it has one, otherwise an untied linear map initialised from the input embedding matrix that is
trained during the warm-up and discarded afterwards. LoRA (or full) backbone parameters train throughout.
"""

from dataclasses import replace
import random

import torch
from torch import nn
import torch.nn.functional as F

from .packing import PackedRequest


def state_pack(model, request):
    """The state text as one causal-free segment: every token belongs to segment 0."""
    tokens = model.tokenizer.encode("State:\n" + request.state + "\n", add_special_tokens=False)
    n = len(tokens)
    if n > model.config.max_tokens:
        tokens, n = tokens[:model.config.max_tokens], model.config.max_tokens
    return PackedRequest(torch.tensor([tokens]), torch.tensor([list(range(n))]), torch.zeros(n, dtype=torch.long),
                         torch.zeros(1, dtype=torch.long), [], n, ())


def mask_token_id(model):
    backbone_mask = getattr(model.backbone, "mask_token_id", None)
    if backbone_mask is not None:
        return int(backbone_mask)
    for attribute in ("mask_token_id", "pad_token_id"):
        value = getattr(model.tokenizer, attribute, None)
        if value is not None:
            return int(value)
    return 0


def output_projection(model):
    """(projection module, its new trainable parameters). Ties to the LM head when the backbone has one."""
    getter = getattr(model.backbone, "get_output_embeddings", None)
    head = getter() if getter is not None else None
    if head is not None:
        return head, []
    weight = model.backbone.get_input_embeddings().weight
    projection = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=torch.float32)
    with torch.no_grad():
        projection.weight.copy_(weight)
    return projection, list(projection.parameters())


def masked_next_token_warmup(model, requests, steps, mask_rate=.15, backbone_lr=2e-4, head_lr=1e-3, seed=17):
    """Run `steps` single-request updates and return the loss history (one float per step)."""
    if steps < 1 or not requests:
        raise ValueError("The warm-up needs at least one step and one request")
    if not 0 < mask_rate < 1:
        raise ValueError("mask_rate must lie in (0, 1)")
    projection, extra = output_projection(model)
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": backbone_parameters, "lr": backbone_lr}] if backbone_parameters else []
    if extra:
        groups.append({"params": extra, "lr": head_lr})
    if not groups:
        raise ValueError("Nothing to train: the backbone is frozen and the LM head is tied to it")
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    mask_id = mask_token_id(model)
    device = model.device
    order, history = [], []
    was_training = model.training
    model.train()
    for _ in range(steps):
        if not order:
            order = list(range(len(requests)))
            rng.shuffle(order)
        packed = state_pack(model, requests[order.pop()])
        ids = packed.input_ids[0]
        n = ids.shape[0]
        if n < 2:
            raise ValueError("State text must have at least two tokens")
        count = max(1, round(mask_rate * (n - 1)))
        chosen = torch.randperm(n - 1, generator=generator)[:count] + 1
        masked = ids.clone()
        masked[chosen] = mask_id
        hidden = model._encode(replace(packed, input_ids=masked[None]), directionality="block")
        rows = hidden[(chosen - 1).to(hidden.device)]
        logits = projection(rows.to(projection.weight.dtype)).float()
        loss = F.cross_entropy(logits, ids[chosen].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite warm-up loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for group in groups for p in group["params"]], 1.)
        optimizer.step()
        history.append(float(loss.detach()))
    model.train(was_training)
    return history
