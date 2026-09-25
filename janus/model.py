"""Qwen backbone with a dynamic pointer readout and optional adapter training."""

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, Qwen3Config, Qwen3Model

from .attention import DIRECTIONALITIES, build_block_mask, pad_packed
from .packing import SCORE_BLOCKS, TREE_BLOCKS, TREE_POSITIONS, ByteTokenizer

HYBRID_FAMILIES = ("qwen3_5", "qwen3_5_moe")  # janus.hybrid: Qwen3.5 dense and Qwen3.6-A3B MoE, the same tree runner


@dataclass
class ModelConfig:
    backbone: str = "Qwen/Qwen3-0.6B-Base"
    revision: str | None = None
    mode: str = "listwise"
    adaptation: str = "lora"
    head_rank: int = 128
    lora_rank: int = 16
    hidden_size: int = 64
    layers: int = 2
    dtype: str = "bfloat16"
    max_tokens: int = 2048
    # Budget for the state plus any one question's tokens (Jev documents 32k against 64k per request); None follows
    # max_tokens. It is also the largest position a pack uses, so it, not max_tokens, is checked against the
    # backbone's position limit.
    max_state_plus_question: int | None = None
    attention: str = "eager"
    set_layers: int = 2
    set_width: int = 256
    set_heads: int = 4
    # Pair graph: width of the per-option projection and of the pairwise comparison features.
    pair_width: int = 256
    # Tree graph variants: what the Choice block lists and where leaf positions start.
    tree_block: str = "full"
    tree_positions: str = "shared"
    # Tree graph: whether the Score block lists the levels ("full") or each level leaf alone carries its description
    # ("independent", the Phase 1 layout; docs/phase4/score-and-cardinality.md). Ignored outside tree mode.
    score_block: str = "independent"
    # Decoder graph: shallow decision decoder over cached state features.
    decoder_layers: int = 2
    decoder_heads: int = 8
    # WP5a: "causal" is the Phase 1 mask; "block" is fully connected inside the state, each block, and each leaf.
    directionality: str = "causal"
    # WP5c: "qwen" loads AutoModel with trust_remote_code=False; "llada" loads a masked-diffusion checkpoint through
    # janus.llada (remote code, so trust_remote_code must be set explicitly), forces block directionality, and reads each
    # leaf at a mask token: "vocab" uses the LM head's " yes" minus " no" logit, "scalar" the ordinary ScalarHead.
    backbone_family: str = "qwen"
    trust_remote_code: bool = False
    llada_readout: str = "vocab"
    # Activation checkpointing in the backbone (8B-class runs); no effect on results, only on memory and time.
    gradient_checkpointing: bool = False
    # Image states (janus.vision; qwen3_5 only): load the checkpoint's vision tower. An image is resized to at most
    # image_max_pixels before patching (Qwen3.5: 32x32 pixels per token, so 1024x1024 is at most 1024 tokens).
    images: bool = False
    image_max_pixels: int = 1024 * 1024
    # qwen3_5 runner (janus.hybrid): the state pass runs in chunks of this many tokens (memory, not results), and
    # `batch_states` packs go through the backbone as one batch in training and evaluation (forward_many).
    state_chunk_tokens: int = 4096
    batch_states: int = 1

    @property
    def state_budget(self):
        return self.max_tokens if self.max_state_plus_question is None else self.max_state_plus_question


class PointerHead(nn.Module):
    def __init__(self, hidden_size, rank):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.scale = math.sqrt(rank)

    def forward(self, decision, options):
        query = self.query(self.norm(decision.float()))
        keys = self.key(self.norm(options.float()))
        return (keys * query).sum(-1) / self.scale


class ScalarHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, reps):
        return self.out(self.norm(reps.float()))[:, 0]


class SetHead(nn.Module):
    def __init__(self, hidden_size, width, layers, heads):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, width)
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0., batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, 1)

    def forward(self, reps, interact):
        x = self.proj(self.norm(reps.float()))[None]
        if interact:
            x = self.encoder(x)
        return self.out(x)[0, :, 0]


class PairHead(nn.Module):
    """Pairwise comparison features between every ordered option pair before the readout.

    Each option representation is projected to `width`; for every ordered pair (i, j) an MLP reads
    [h_i, h_j, h_i - h_j, h_i * h_j]; option i aggregates by the mean over j != i, concatenates with h_i,
    and a second MLP gives its logit. Permutation-equivariant by construction. Without interaction
    (Score, Noul, or a single option) the aggregate is zero and the readout is a scalar function of h_i."""

    def __init__(self, hidden_size, width, heads=None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, width)
        self.pair = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, width))
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, reps, interact):
        h = self.proj(self.norm(reps.float()))
        n, width = h.shape
        if interact and n > 1:
            hi, hj = h[:, None].expand(n, n, width), h[None].expand(n, n, width)
            features = self.pair(torch.cat([hi, hj, hi - hj, hi * hj], -1))
            keep = (~torch.eye(n, dtype=torch.bool, device=h.device)).to(features.dtype)[..., None]
            aggregate = (features * keep).sum(1) / (n - 1)
        else:
            aggregate = torch.zeros_like(h)
        return self.out(torch.cat([h, aggregate], -1))[:, 0]


class DecoderHead(nn.Module):
    """Branch tokens self-attend causally and cross-attend to the state features; pointer readout on top."""

    def __init__(self, hidden_size, layers, heads, rank):
        super().__init__()
        layer = nn.TransformerDecoderLayer(hidden_size, heads, 2 * hidden_size, dropout=0., batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.pointer = PointerHead(hidden_size, rank)

    def forward(self, state_features, branch_hidden, option_positions, decision_position):
        n = branch_hidden.shape[0]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=branch_hidden.device), diagonal=1)
        out = self.decoder(branch_hidden[None].float(), state_features[None].float(), tgt_mask=causal)[0]
        return self.pointer(out[decision_position], out[list(option_positions)])


class DecisionModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.mode not in {"listwise", "independent", "tree", "set", "pair", "decoder"}:
            raise ValueError("Unknown option interaction mode")
        if config.tree_block not in TREE_BLOCKS or config.tree_positions not in TREE_POSITIONS:
            raise ValueError(f"tree_block must be one of {TREE_BLOCKS} and tree_positions one of {TREE_POSITIONS}")
        if config.score_block not in SCORE_BLOCKS:
            raise ValueError(f"score_block must be one of {SCORE_BLOCKS}")
        if config.decoder_layers < 1 or config.decoder_heads < 1:
            raise ValueError("decoder_layers and decoder_heads must be positive")
        if config.adaptation not in {"frozen", "lora", "full"}:
            raise ValueError("adaptation must be frozen, lora or full")
        if config.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        if config.attention not in {"eager", "sdpa", "flex"}:
            raise ValueError("attention must be eager, sdpa or flex")
        if not 0 < config.state_budget <= config.max_tokens:
            raise ValueError("max_state_plus_question must be positive and at most max_tokens")
        if config.directionality not in DIRECTIONALITIES:
            raise ValueError(f"directionality must be one of {DIRECTIONALITIES}")
        if config.backbone_family not in {"qwen", "llada", *HYBRID_FAMILIES}:
            raise ValueError("backbone_family must be qwen, llada, qwen3_5 or qwen3_5_moe")
        if config.llada_readout not in {"vocab", "scalar"}:
            raise ValueError("llada_readout must be vocab or scalar")
        if config.images and config.backbone_family != "qwen3_5":
            raise ValueError("images are only supported by backbone_family qwen3_5")
        implementation = {"eager": "eager", "sdpa": "sdpa", "flex": "flex_attention"}[config.attention]
        self.config = config
        if config.backbone_family == "llada":
            from .llada import LLaDABackbone
            if config.mode != "tree":
                raise ValueError("llada backbones read leaves at mask tokens and need mode tree")
            # Masked-diffusion pretraining is bidirectional; the causal rule would be a different model.
            config.directionality = "block"
            self.backbone = LLaDABackbone.load(config)
            self.tokenizer = self.backbone.tokenizer
            config.revision = self.backbone.config._commit_hash or config.revision
            if config.state_budget > self.backbone.config.max_position_embeddings:
                raise ValueError("max_tokens exceeds the pretrained position limit")
        elif config.backbone_family in HYBRID_FAMILIES:
            # Hybrid Gated DeltaNet + attention (qwen3_5_moe: plus the sparse MoE feed-forward): the packed request runs
            # as a tree of batched passes (janus.hybrid).
            from .hybrid import HybridBackbone
            if config.mode == "decoder":
                raise ValueError("qwen3_5 backbones need a state segment in every pack; decoder mode is not supported")
            if config.directionality != "causal":
                raise ValueError("qwen3_5 backbones are causal (DeltaNet recurrence); directionality must be causal")
            self.backbone = HybridBackbone.load(config)
            self.tokenizer = self.backbone.tokenizer
            config.revision = getattr(self.backbone.config, "_commit_hash", None) or config.revision
            if config.state_budget > self.backbone.config.max_position_embeddings:
                raise ValueError("max_tokens exceeds the pretrained position limit")
        elif config.backbone == "tiny":
            hf_config = Qwen3Config(vocab_size=257, hidden_size=config.hidden_size,
                                   intermediate_size=config.hidden_size * 2,
                                   num_hidden_layers=config.layers, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=config.hidden_size // 4,
                                   max_position_embeddings=config.state_budget,
                                   attention_dropout=0.0, use_cache=False)
            hf_config._attn_implementation = implementation
            self.backbone = Qwen3Model(hf_config)
            self.tokenizer = ByteTokenizer()
        else:
            from .wedlm import qwen3_config
            # tencent/WeDLM-8B-Base: Qwen3-shaped weights behind a "wedlm" config; janus/wedlm.py maps it to Qwen3Config.
            mapped = qwen3_config(config.backbone, config.revision)
            self.backbone = AutoModel.from_pretrained(
                config.backbone, revision=config.revision, trust_remote_code=False,
                attn_implementation=implementation, dtype=getattr(torch, config.dtype),
                **({"config": mapped} if mapped is not None else {}))
            resolved = getattr(self.backbone.config, "_commit_hash", None)
            config.revision = resolved or config.revision
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.backbone, revision=config.revision, trust_remote_code=False)
            if config.state_budget > self.backbone.config.max_position_embeddings:
                raise ValueError("max_tokens exceeds the pretrained position limit")
        self.backbone.config.use_cache = False
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Initialize the readout before adapters consume RNG, matching control heads.
        hidden = self.backbone.config.hidden_size
        if config.mode == "tree":
            self.head = ScalarHead(hidden)
        elif config.mode == "set":
            self.head = SetHead(hidden, config.set_width, config.set_layers, config.set_heads)
        elif config.mode == "pair":
            self.head = PairHead(hidden, config.pair_width)
        elif config.mode == "decoder":
            self.head = DecoderHead(hidden, config.decoder_layers, config.decoder_heads, config.head_rank)
        else:
            self.head = PointerHead(hidden, config.head_rank)
        self.last_compute = None
        if config.adaptation == "frozen":
            self.backbone.requires_grad_(False)
        elif config.adaptation == "lora" and config.backbone_family in ("llada", *HYBRID_FAMILIES):
            self.backbone.apply_lora(config.lora_rank)
        elif config.adaptation == "lora":
            from peft import LoraConfig, get_peft_model
            self.backbone = get_peft_model(self.backbone, LoraConfig(
                r=config.lora_rank, lora_alpha=config.lora_rank * 2, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def packing_mode(self):
        return "independent" if self.config.mode in ("set", "pair") else self.config.mode

    @property
    def packing_kwargs(self):
        kwargs = {"tree_block": self.config.tree_block, "tree_positions": self.config.tree_positions,
                  "score_block": self.config.score_block, "max_state_plus_question": self.config.state_budget}
        if self.config.images:
            kwargs["vision"] = self.backbone.vision
        return kwargs

    def masked_input_ids(self, packed):
        """llada: the backbone's mask token at every leaf decision position (the readout position)."""
        input_ids = packed.input_ids.clone()
        for branch in packed.branches:
            input_ids[0, list(branch.option_positions)] = self.backbone.mask_token_id
        return input_ids

    def _encode(self, packed, inputs_embeds=None, directionality=None):
        """Backbone hidden states [n, d] for one packed sequence under its segment mask.

        `directionality` defaults to the config's; the warm-up passes "block" explicitly."""
        directionality = directionality or self.config.directionality
        device = self.device
        dtype = self.backbone.get_input_embeddings().weight.dtype
        n = packed.token_count
        input_ids, position_ids = packed.input_ids, packed.position_ids
        if self.config.backbone_family == "llada":
            allowed = packed.mask(directionality).to(device)
            return self.backbone.encode(input_ids.to(device), position_ids.to(device), allowed, inputs_embeds)[0, :n]
        if self.config.backbone_family in HYBRID_FAMILIES:
            return self.backbone.encode(packed, inputs_embeds, directionality)
        extra = {}
        if self.config.attention == "flex":
            block_mask, total = build_block_mask(packed.segment_ids, packed.parents, device, directionality=directionality)
            input_ids, position_ids = pad_packed(packed, total, 0)
            if inputs_embeds is not None:
                pad = torch.zeros((1, total - n, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = torch.cat([inputs_embeds, pad], dim=1)
            mask = block_mask
            if dtype == torch.float32:
                # The default Triton tiles for fp32 at head_dim 128 exceed the shared memory of sm_120 (RTX 5090).
                extra["kernel_options"] = {"BLOCK_M": 32, "BLOCK_N": 32}
        else:
            allowed = packed.mask(directionality).to(device)
            if self.config.attention == "sdpa":
                mask = allowed[None, None]
            else:
                mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
                mask.masked_fill_(~allowed, torch.finfo(dtype).min)
                mask = mask[None, None]
        kwargs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None else {"input_ids": input_ids.to(device)}
        return self.backbone(position_ids=position_ids.to(device), attention_mask={"full_attention": mask},
                             use_cache=False, **extra, **kwargs).last_hidden_state[0, :n]

    def forward(self, packed, inputs_embeds=None):
        if self.config.mode == "decoder":
            if inputs_embeds is not None:
                raise ValueError("inputs_embeds is not supported in decoder mode")
            state_features = self._encode(packed)
            logits = [None] * len(packed.option_counts)
            branch_tokens, self_pairs = 0, 0
            for pack in packed.branch_packs:
                branch = pack.branches[0]
                branch_tokens += pack.token_count
                self_pairs += pack.token_count * pack.token_count
                logits[branch.question_index] = self.head(state_features, self._encode(pack),
                                                          branch.option_positions, branch.decision_position)
            self.last_compute = {"backbone_tokens": packed.token_count + branch_tokens,
                                 "decoder_cross_attention_pairs": self.config.decoder_layers * branch_tokens * packed.token_count,
                                 "decoder_self_attention_pairs": self.config.decoder_layers * self_pairs}
            return logits
        llada = self.config.backbone_family == "llada"
        if llada and inputs_embeds is None:
            packed = replace(packed, input_ids=self.masked_input_ids(packed))
        hidden = self._encode(packed, inputs_embeds)
        self.last_compute = {"backbone_tokens": packed.token_count, "decoder_cross_attention_pairs": 0, "decoder_self_attention_pairs": 0}
        logits = [[None] * k for k in packed.option_counts]
        if self.config.mode == "tree":
            for branch in packed.branches:
                rows = hidden[list(branch.option_positions)]
                z = self.backbone.answer_logits(rows) if llada and self.config.llada_readout == "vocab" else self.head(rows)
                if branch.kind == "noul":
                    logits[branch.question_index] = [z.new_zeros(()), z[0]]
                else:
                    for index, value in zip(branch.option_indices, z):
                        logits[branch.question_index][index] = value
        elif self.config.mode in ("set", "pair"):
            per_question = {}
            for branch in packed.branches:
                per_question.setdefault(branch.question_index, []).append(
                    (branch.option_indices[0], hidden[branch.decision_position], branch.kind))
            for qi, rows in per_question.items():
                rows.sort(key=lambda row: row[0])
                z = self.head(torch.stack([row[1] for row in rows]), interact=rows[0][2] == "choice")
                for (index, _, _), value in zip(rows, z):
                    logits[qi][index] = value
        else:
            for branch in packed.branches:
                scores = self.head(hidden[branch.decision_position], hidden[list(branch.option_positions)])
                for index, value in zip(branch.option_indices, scores):
                    logits[branch.question_index][index] = value
        return [torch.stack(values) for values in logits]

    def forward_many(self, packs):
        """Logits for several packed requests, one list per pack, as `[self(p) for p in packs]` computes them; a
        backbone with `prepare_many` (janus.hybrid) first encodes all of them in one batch of passes."""
        if hasattr(self.backbone, "prepare_many"):
            self.backbone.prepare_many(packs)
        return [self(p) for p in packs]

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
