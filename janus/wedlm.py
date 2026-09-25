"""tencent/WeDLM-8B-Base under the in-library Qwen3 class (spec WP5c, diffusion-trained causal arm).

WeDLM-8B-Base is initialised from Qwen3-8B and trained for diffusion decoding under standard causal attention. Its
safetensors hold exactly the Qwen3ForCausalLM tensor names, shapes and dtypes (399 bf16 tensors, 8,190,735,360
parameters, verified from the Hub safetensors headers), and its remote `WeDLMForCausalLM` forward pass gives the same
logits as `Qwen3ForCausalLM` on the same weights (tests/test_backbones.py). The remote code also does not import under
transformers 5 (it uses `check_model_inputs` and a 4.x rotary-embedding API). So the arm loads the checkpoint with
`Qwen3Model` and `backbone_family: "qwen"`; only the config needs translating, because `config.json` says
`model_type: "wedlm"` with an `auto_map`, which `AutoModel` refuses without remote code.
"""

import inspect

from transformers import PretrainedConfig, Qwen3Config

WEDLM_MODEL_TYPE = "wedlm"


def config_dict(backbone, revision=None):
    """The raw `config.json` of a checkpoint; executes no code."""
    values, _ = PretrainedConfig.get_config_dict(backbone, revision=revision)
    return values


def qwen3_config(backbone, revision=None):
    """A Qwen3Config for a WeDLM checkpoint, or None when the checkpoint is not one (model_type != "wedlm").

    Raises ValueError when the WeDLM config declares anything the Qwen3 class cannot reproduce."""
    values = config_dict(backbone, revision)
    if values.get("model_type") != WEDLM_MODEL_TYPE:
        return None
    if not values.get("qk_norm", False):
        raise ValueError("WeDLM checkpoint without qk_norm is Qwen2-shaped, not Qwen3-shaped")
    if values.get("use_sliding_window") or set(values.get("layer_types") or ["full_attention"]) != {"full_attention"}:
        raise ValueError("WeDLM checkpoint uses sliding-window layers; the packed-tree mask needs full attention everywhere")
    if values.get("rope_scaling") is not None:
        raise ValueError("WeDLM checkpoint with rope_scaling is not supported")
    params = set(inspect.signature(Qwen3Config.__init__).parameters) - {"self", "kwargs"}
    kwargs = {key: value for key, value in values.items() if key in params}
    kwargs["rope_parameters"] = {"rope_type": "default", "rope_theta": values["rope_theta"]}
    kwargs["use_cache"] = False
    mapped = Qwen3Config(**kwargs)
    mapped._commit_hash = values.get("_commit_hash")
    return mapped
