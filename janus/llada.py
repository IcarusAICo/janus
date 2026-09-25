"""Masked-diffusion backbones (LLaDA family) behind the tree layout: loader, verification, and mask-token readout.

Spec WP5c: each leaf ends in a mask token whose one-step prediction over a two-token vocabulary gives the leaf logit.
Two checkpoints are supported, each through its own calling convention:
  llada   GSAI-ML/LLaDA-8B-Base (`LLaDAModelLM`): bool `attention_bias[B,1,T,T]`, no `position_ids` input, so the
          packed positions are applied by a patched rotary forward.
  llada2  inclusionAI/LLaDA2.0-mini (`LLaDA2MoeModelLM`): additive 4-D `attention_mask` plus `position_ids`.
  illada  GSAI-ML/iLLaDA-8B-Base (`ILLaDAForCausalLM`): the llada2 calling convention (additive or bool 4-D
          `attention_mask` plus `position_ids`, own rotary class, no rope shim); tied LM head; mask token <[MASK]> = 5
          (the upstream README's `mask_id=5`). Its remote code declares `_tied_weights_keys` as a list, which
          transformers 5 rejects, so the class attribute is rewritten to the dict form before instantiation.
Everything the readout needs (mask token id, LM head, single-token answers) is verified at construction and raises
ValueError with the reason when absent. Remote code loads only when the model config says `trust_remote_code: true`.
"""

import os
from pathlib import Path
import types

import torch
from torch import nn

ANSWER_TOKENS = (" yes", " no")
# Names the checkpoints give their mask token when the tokenizer's `mask_token` field is unset (LLaDA-8B-Base lists
# <|mdm_mask|> only as an additional special token; its model card calls it [MDM_MASK]).
MASK_TOKEN_NAMES = ("<|mdm_mask|>", "[MDM_MASK]", "<|mask|>", "<[MASK]>")
FAMILIES = {"LLaDAModelLM": "llada", "LLaDA2MoeModelLM": "llada2", "ILLaDAForCausalLM": "illada"}
TINY_REPOS = {"tiny-llada": "GSAI-ML/LLaDA-8B-Base", "tiny-llada2": "inclusionAI/LLaDA2.0-mini", "tiny-illada": "GSAI-ML/iLLaDA-8B-Base"}
TINY_VOCAB, TINY_YES, TINY_NO, TINY_MASK = 300, 257, 258, 259
LORA_TARGETS = {"llada": ["q_proj", "k_proj", "v_proj", "attn_out"], "llada2": ["query_key_value", "dense"],
                "illada": ["q_proj", "k_proj", "v_proj", "o_proj"]}
ILLADA_TIED_KEYS = {"lm_head.weight": "model.embed_tokens.weight"}


def illada_class(hf_config, repo, revision=None):
    """The checkpoint's `ILLaDAForCausalLM` class, with `_tied_weights_keys` in the dict form transformers 5 expects.

    The shipped code (written for 4.57) declares the list `["lm_head.weight"]`; `post_init` in transformers 5 calls
    `.keys()` on it and fails. The mapping ties the head to the embedding, which is what `tie_word_embeddings: true` means."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    reference = hf_config.auto_map["AutoModelForCausalLM"]
    cls = get_class_from_dynamic_module(reference, repo, revision=revision)
    if not isinstance(getattr(cls, "_tied_weights_keys", None), dict):
        cls._tied_weights_keys = dict(ILLADA_TIED_KEYS)
    return cls


def ensure_default_rope():
    """transformers 5 removed ROPE_INIT_FUNCTIONS['default'], which LLaDA2.0's remote code (written for 4.57) looks up.

    The registered function is the 4.x default: inverse frequencies from rope_theta over the rotary dimension, scaling 1.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def default_rope(config, device=None, seq_len=None, **kwargs):
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        declared = getattr(config, "rotary_dim", None)
        if declared is not None and declared != dim:
            raise ValueError(f"rotary_dim {declared} disagrees with head_dim * partial_rotary_factor = {dim}")
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS["default"] = default_rope


def load_shards(lm, repo, revision=None):
    """Load a cached checkpoint's safetensors shards into `lm` with strict key accounting, then tie the LM head when the
    config says so. Raises ValueError listing the keys that were neither loaded nor tied, or that the model lacks."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    snapshot = snapshot_download(repo, revision=revision, local_files_only=True, allow_patterns=["*.safetensors", "*.safetensors.index.json"])
    files = sorted(Path(snapshot).glob("*.safetensors"))
    if not files:
        raise ValueError(f"{repo}: no safetensors shards in {snapshot}")
    expected = set(lm.state_dict())
    loaded = set()
    for file in files:
        with safe_open(str(file), "pt", device="cpu") as f:
            shard = {k: f.get_tensor(k) for k in f.keys()}
        unexpected = set(shard) - expected
        if unexpected:
            raise ValueError(f"{repo}: shard {file.name} has keys the model lacks: {sorted(unexpected)[:5]}")
        lm.load_state_dict(shard, strict=False)
        loaded |= set(shard)
    tied = getattr(lm.config, "tie_word_embeddings", False)
    if tied and lm.get_output_embeddings() is not None:
        lm.get_output_embeddings().weight = lm.get_input_embeddings().weight
        loaded.add("lm_head.weight")
    missing = {k for k in expected - loaded if not k.endswith("inv_freq") and not k.endswith("rotary_emb.inv_freq")}
    if tied:
        missing = {k for k in missing if not k.endswith("lm_head.weight")}
    if missing:
        raise ValueError(f"{repo}: {len(missing)} model tensors not in the shards, e.g. {sorted(missing)[:5]}")
    return lm


def resolve_mask_token_id(config, tokenizer):
    """The mask token id from every source that names one; all of them must agree.

    Sources: the tokenizer's mask_token, an added token with one of MASK_TOKEN_NAMES, the config's mask_token_id."""
    sources = {}
    tokenizer_mask = getattr(tokenizer, "mask_token_id", None)
    if tokenizer_mask is not None:
        sources["tokenizer mask_token"] = int(tokenizer_mask)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unknown = getattr(tokenizer, "unk_token_id", None)
    if convert is not None:
        for name in MASK_TOKEN_NAMES:
            token_id = convert(name)
            if token_id is not None and token_id != unknown:
                sources[f"added token {name}"] = int(token_id)
                break
    config_mask = getattr(config, "mask_token_id", None)
    if config_mask is not None:
        sources["config mask_token_id"] = int(config_mask)
    if not sources:
        raise ValueError("No mask token: the tokenizer has no mask_token, none of "
                         f"{MASK_TOKEN_NAMES} is an added token, and the config has no mask_token_id")
    if len(set(sources.values())) > 1:
        raise ValueError(f"Mask token sources disagree: {sources}")
    return next(iter(sources.values()))


def _single_token(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Answer token {text!r} must be a single token for the mask-position readout; got {ids}")
    return int(ids[0])


def _patch_rotary(rotary):
    """LLaDA-8B's RotaryEmbedding.forward takes positions 0..T-1; this instance reads them from `_jev_positions`."""
    original = rotary.forward

    def forward(q, k):
        positions = getattr(rotary, "_jev_positions", None)
        if positions is None:
            return original(q, k)
        if q.shape[-2] != k.shape[-2]:
            raise ValueError("Packed positions need query and key lengths to match (no cache)")
        q_, k_ = (q.float(), k.float()) if rotary.config.rope_full_precision else (q, k)
        with torch.autocast(q.device.type, enabled=False):
            pos_sin, pos_cos = rotary.get_rotary_embedding(int(positions.max()) + 1, q_.device)
            if torch.is_grad_enabled() and pos_sin.is_inference():
                # LLaDA caches these tables on first use; computed under the dev pass's inference_mode they are
                # inference tensors and the first training backward fails. Clone outside inference mode.
                pos_sin, pos_cos = pos_sin.clone(), pos_cos.clone()
            pos_sin, pos_cos = pos_sin[:, :, positions, :].type_as(q_), pos_cos[:, :, positions, :].type_as(q_)
            q_ = rotary.apply_rotary_pos_emb(pos_sin, pos_cos, q_)
            k_ = rotary.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)
    rotary.forward = forward


class TinyLLaDATokenizer:
    """Byte tokenizer for the tiny test instances: bytes 1..256, ' yes' and ' no' single ids, a mask id."""
    vocab_size = TINY_VOCAB
    mask_token_id = TINY_MASK
    pad_token_id = 0

    def encode(self, value, add_special_tokens=False):
        if value == ANSWER_TOKENS[0]:
            return [TINY_YES]
        if value == ANSWER_TOKENS[1]:
            return [TINY_NO]
        return [b + 1 for b in value.encode("utf-8")]


class LLaDABackbone(nn.Module):
    """A LLaDA checkpoint's ForCausalLM module behind the interface DecisionModel uses for its backbone."""

    def __init__(self, lm, tokenizer):
        super().__init__()
        family = FAMILIES.get(type(lm).__name__)
        if family is None:
            raise ValueError(f"Unsupported masked-diffusion architecture {type(lm).__name__}; expected one of {sorted(FAMILIES)}")
        self.lm, self.tokenizer, self.family = lm, tokenizer, family
        cfg = lm.config
        embeddings = lm.get_input_embeddings()
        if not isinstance(embeddings, nn.Embedding):
            raise ValueError("Checkpoint does not expose an input embedding module")
        head = lm.get_output_embeddings()
        if head is None or not hasattr(head, "weight") or tuple(head.weight.shape) != tuple(embeddings.weight.shape):
            raise ValueError("Checkpoint does not expose an LM head with one row per vocabulary entry; the mask-token readout needs it")
        if getattr(head, "bias", None) is not None:
            raise ValueError("LM head with a bias is not supported by the two-row readout")
        mask_id = resolve_mask_token_id(cfg, tokenizer)
        if not 0 <= int(mask_id) < embeddings.num_embeddings:
            raise ValueError(f"mask token id {mask_id} lies outside the embedding table ({embeddings.num_embeddings} rows)")
        self.mask_token_id = int(mask_id)
        self.answer_ids = tuple(_single_token(tokenizer, text) for text in ANSWER_TOKENS)
        if max(self.answer_ids) >= embeddings.num_embeddings:
            raise ValueError("Answer token ids lie outside the embedding table")
        if family == "llada":
            hidden, max_positions = cfg.d_model, cfg.max_sequence_length
            if not getattr(cfg, "rope", False):
                raise ValueError("LLaDA positions are applied through RoPE; this checkpoint has rope disabled")
            self._rotaries = [block.rotary_emb for block in lm.model.transformer.blocks]
            for rotary in self._rotaries:
                _patch_rotary(rotary)
        else:
            hidden, max_positions = cfg.hidden_size, cfg.max_position_embeddings
        self.config = types.SimpleNamespace(hidden_size=hidden, max_position_embeddings=max_positions, use_cache=False,
                                            vocab_size=embeddings.num_embeddings, _commit_hash=getattr(cfg, "_commit_hash", None),
                                            model_type=getattr(cfg, "model_type", family))
        cfg.use_cache = False
        self.lora_targets = LORA_TARGETS[family]

    @classmethod
    def load(cls, config):
        """Load from a ModelConfig; `tiny-llada` and `tiny-llada2` build random 2-layer instances of the real code."""
        if not config.trust_remote_code:
            raise ValueError("backbone_family llada executes the checkpoint's remote code; set trust_remote_code: true explicitly")
        if config.attention == "flex":
            raise ValueError("FlexAttention is not available for LLaDA backbones (they run their own attention); use sdpa or eager")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        ensure_default_rope()
        if config.backbone in TINY_REPOS:
            return cls(cls._tiny(config.backbone, config.hidden_size, config.layers, config.max_tokens), TinyLLaDATokenizer())
        hf_config = AutoConfig.from_pretrained(config.backbone, revision=config.revision, trust_remote_code=True)
        architectures = getattr(hf_config, "architectures", None) or []
        family = FAMILIES.get(architectures[0]) if architectures else None
        if family is None:
            raise ValueError(f"{config.backbone} declares architectures {architectures}; expected one of {sorted(FAMILIES)}")
        dtype = getattr(torch, config.dtype)
        if family == "illada":
            remote = illada_class(hf_config, config.backbone, config.revision)
        else:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            remote = get_class_from_dynamic_module(hf_config.auto_map["AutoModelForCausalLM"], config.backbone, revision=config.revision)
        # LLaDA-8B's class runs its own attention and rejects an attn_implementation; the other two take sdpa/eager.
        attention = {"llada": None, "llada2": "sdpa"}.get(family, config.attention)
        # transformers 5.3 mishandles these remote classes' weights: `from_pretrained` re-initialises most tensors after
        # a clean loading report for iLLaDA (226 of 291 differed from the shards) and raises inside
        # `mark_tied_weights_as_initialized` for LLaDA-8B. So build from the config and load the shards directly.
        lm = remote._from_config(hf_config, dtype=dtype, **({} if attention is None else {"attn_implementation": attention}))
        load_shards(lm, config.backbone, config.revision)
        tokenizer = AutoTokenizer.from_pretrained(config.backbone, revision=config.revision, trust_remote_code=True)
        return cls(lm, tokenizer)

    @staticmethod
    def snapshot_dir(repo):
        """The cached snapshot of a repo's non-weight files, without network access; None when absent."""
        from huggingface_hub import snapshot_download
        cache = os.environ.get("HF_HUB_CACHE") or str(Path(__file__).resolve().parent.parent / ".cache" / "huggingface" / "hub")
        try:
            return snapshot_download(repo, cache_dir=cache, local_files_only=True,
                                     allow_patterns=["config.json", "*.py", "tokenizer*", "special_tokens_map.json"])
        except Exception:
            return None

    @classmethod
    def _tiny(cls, name, hidden, layers, max_tokens):
        from transformers import AutoConfig, AutoModelForCausalLM
        snapshot = cls.snapshot_dir(TINY_REPOS[name])
        if snapshot is None:
            raise FileNotFoundError(f"{name} needs the cached remote code of {TINY_REPOS[name]} (HF_HUB_CACHE)")
        if name == "tiny-llada":
            overrides = dict(d_model=hidden, n_heads=4, n_kv_heads=4, n_layers=layers, mlp_hidden_size=2 * hidden, vocab_size=TINY_VOCAB,
                             embedding_size=TINY_VOCAB, mask_token_id=TINY_MASK, max_sequence_length=max_tokens,
                             pad_token_id=0, eos_token_id=0, attention_dropout=0., residual_dropout=0., embedding_dropout=0.)
        elif name == "tiny-illada":
            overrides = dict(hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers, num_attention_heads=4,
                             num_key_value_heads=2, vocab_size=TINY_VOCAB, max_position_embeddings=max_tokens, pad_token_id=0)
        else:
            overrides = dict(hidden_size=hidden, intermediate_size=2 * hidden, moe_intermediate_size=hidden // 2, num_experts=4,
                             num_experts_per_tok=2, n_group=1, topk_group=1, num_hidden_layers=layers, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=hidden // 4, rotary_dim=hidden // 8, vocab_size=TINY_VOCAB,
                             max_position_embeddings=max_tokens, first_k_dense_replace=1, num_shared_experts=1, pad_token_id=0)
        hf_config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, dtype="float32", **overrides)
        if name == "tiny-illada":
            return illada_class(hf_config, snapshot)._from_config(hf_config, dtype=torch.float32, attn_implementation="sdpa")
        extra = {"attn_implementation": "sdpa"} if name == "tiny-llada2" else {}
        return AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True, dtype=torch.float32, **extra)

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm.get_output_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if self.family == "llada":
            import sys
            strategies = sys.modules[type(self.lm.config).__module__].ActivationCheckpointingStrategy
            self.lm.model.set_activation_checkpointing(strategies.whole_layer)
        else:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False})

    def apply_lora(self, rank):
        from peft import LoraConfig, get_peft_model
        self.lm = get_peft_model(self.lm, LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
                                                     target_modules=self.lora_targets, bias="none"))

    def encode(self, input_ids, position_ids, allowed, inputs_embeds=None):
        """Final-norm hidden states [1, n, d] under a bool `allowed[n, n]` (True = attend) and packed positions."""
        n = allowed.shape[0]
        if position_ids.shape != (1, n) or (input_ids is not None and input_ids.shape != (1, n)):
            raise ValueError("encode expects one sequence of n tokens with positions [1, n]")
        ids = None if inputs_embeds is not None else input_ids
        if self.family == "llada":
            for rotary in self._rotaries:
                rotary._jev_positions = position_ids[0]
            try:
                out = self.lm(input_ids=ids, inputs_embeds=inputs_embeds, attention_bias=allowed[None, None],
                              output_hidden_states=True, use_cache=False, return_dict=True)
            finally:
                for rotary in self._rotaries:
                    rotary._jev_positions = None
        else:
            dtype = self.get_input_embeddings().weight.dtype
            bias = torch.zeros((1, 1, n, n), device=allowed.device, dtype=dtype).masked_fill_(~allowed[None, None], torch.finfo(dtype).min)
            out = self.lm(input_ids=ids, inputs_embeds=inputs_embeds, attention_mask=bias, position_ids=position_ids,
                          output_hidden_states=True, use_cache=False, return_dict=True)
        if getattr(out, "hidden_states", None) is None:
            raise RuntimeError("Checkpoint did not return hidden states; the readout needs the final-norm hidden states")
        return out.hidden_states[-1]

    def answer_logits(self, rows):
        """logit(' yes') - logit(' no') from the two LM-head rows at the given hidden states [k, d]."""
        weight = self.get_output_embeddings().weight[list(self.answer_ids)]
        logits = rows.to(weight.dtype) @ weight.T
        return (logits[:, 0] - logits[:, 1]).float()
