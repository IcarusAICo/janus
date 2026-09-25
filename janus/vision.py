"""Image states for the Qwen3.5 backbone (janus.hybrid): the vision tower, the placeholder layout, the 3-axis positions.

A state may carry images (`janus.schema.ImageState`). The stock `Qwen3_5ForConditionalGeneration` lays an image out as
`<|vision_start|>`, N `<|image_pad|>` tokens, `<|vision_end|>`, with N = (grid_t * grid_h * grid_w) / merge_size^2 from
the image processor, replaces the N placeholder embeddings by the vision tower's merged features, and gives them
3-axis rotary positions (`get_rope_index`): the temporal axis is the start position, height and width count rows and
columns of the merged grid from it, and the text after the image continues at start + max(rows, columns). Everything
here reproduces that; the runner (`HybridBackbone.encode`) only sees embeddings and positions.

Deepstack: transformers 5.3's Qwen3.5 has no deepstack path (the modular file deletes `deepstack_visual_indexes` and
the merger list from the Qwen3-VL parent) and the Base checkpoints ship `deepstack_visual_indexes: []`; the vision
features enter the language model once, at the embedding, and `load` refuses a checkpoint that says otherwise.
"""

import io
import re

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

MARKER = re.compile(r"\[image:(\d+)\]")


class Vision(nn.Module):
    """The vision tower and image processor of one checkpoint, plus the placeholder token ids."""

    def __init__(self, visual, processor, image_token_id, vision_start_token_id, vision_end_token_id):
        super().__init__()
        if not isinstance(visual, Qwen3_5VisionModel):
            raise ValueError(f"Vision needs a Qwen3_5VisionModel, got {type(visual).__name__}")
        if getattr(visual.config, "deepstack_visual_indexes", None):
            raise ValueError("this checkpoint injects deepstack features into language layers; the runner does not")
        self.visual, self.processor = visual, processor
        self.image_token_id, self.vision_start_token_id, self.vision_end_token_id = image_token_id, vision_start_token_id, vision_end_token_id
        self.merge = visual.spatial_merge_size
        # ponytail: the tower is frozen (LoRA covers the language model only); the upgrade path is LoRA on visual.blocks.
        visual.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, full, repo, revision=None, max_pixels=None):
        """From a loaded Qwen3_5ForConditionalGeneration `full` and the repo's preprocessor config."""
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(repo, revision=revision, trust_remote_code=False)
        if max_pixels is not None:
            processor.size = {"shortest_edge": processor.size["shortest_edge"], "longest_edge": max_pixels}
        config = full.config
        return cls(full.model.visual, processor, config.image_token_id, config.vision_start_token_id, config.vision_end_token_id)

    def prepare(self, images):
        """(pixel_values [patches, C*T*P*P], grid_thw [k, 3]) for the images (janus.schema.Image) of a state."""
        from PIL import Image as PILImage
        pil = [PILImage.open(io.BytesIO(image.data)).convert("RGB") for image in images]
        out = self.processor(images=pil, return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"]

    def features(self, pixel_values, grid_thw):
        """Merged image features [sum of image tokens, d_text], as `Qwen3_5Model.get_image_features` (concatenated)."""
        weight = self.visual.patch_embed.proj.weight
        out = self.visual(pixel_values.to(weight.device, weight.dtype), grid_thw=grid_thw.to(weight.device), return_dict=True)
        return out.pooler_output

    def layout(self, state, encode, start=0):
        """The state's token ids and 3-axis positions [3, S] (lists), starting at text position `start`.

        `[image:i]` markers place image i at its first mention (later mentions of the same image are dropped, as a
        reference to an image already shown); images never mentioned come first, in order; a marker for an image
        that does not exist is an error."""
        pixel_values, grid_thw = self.prepare(state.images)
        text = str(state)
        pieces, last, seen = [], 0, set()
        for m in MARKER.finditer(text):
            index = int(m.group(1))
            if index >= len(state.images):
                raise ValueError(f"[image:{index}] refers to a missing image; the state has {len(state.images)}")
            pieces.append(text[last:m.start()])
            if index not in seen:
                pieces.append(index)
                seen.add(index)
            last = m.end()
        pieces.append(text[last:])
        pieces = [i for i in range(len(state.images)) if i not in seen] + pieces
        tokens, axes, pos = [], ([], [], []), start

        def text_tokens(ids):
            nonlocal pos
            tokens.extend(ids)
            for axis in axes:
                axis.extend(range(pos, pos + len(ids)))
            pos += len(ids)

        for piece in pieces:
            if isinstance(piece, str):
                if piece:
                    text_tokens(encode(piece))
                continue
            t, h, w = (int(v) for v in grid_thw[piece])
            if t != 1:
                raise ValueError("only still images are supported")
            h, w = h // self.merge, w // self.merge
            text_tokens([self.vision_start_token_id])
            tokens.extend([self.image_token_id] * (h * w))
            axes[0].extend([pos] * (h * w))
            axes[1].extend(pos + r for r in range(h) for _ in range(w))
            axes[2].extend(pos + c for _ in range(h) for c in range(w))
            pos += max(h, w)
            text_tokens([self.vision_end_token_id])
        return tokens, [list(a) for a in axes], (pixel_values, grid_thw)


def state_embeddings(backbone, state, tokenizer):
    """(inputs_embeds [1, S, d], position_ids [3, S], image_token_mask [S]) for an ImageState on a HybridBackbone."""
    vision = backbone.vision
    tokens, axes, images = vision.layout(state, lambda s: tokenizer.encode(s, add_special_tokens=False))
    embed = backbone.get_input_embeddings()
    ids = torch.tensor(tokens, device=embed.weight.device)
    embeds = embed(ids)
    mask = ids == vision.image_token_id
    embeds = embeds.masked_scatter(mask[:, None].expand_as(embeds), vision.features(*images).to(embeds.dtype))
    return embeds[None], torch.tensor(axes, device=ids.device), mask


def tiny_processor(patch_size=4, merge_size=2, min_pixels=64, max_pixels=4096):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    return Qwen2VLImageProcessorFast(patch_size=patch_size, merge_size=merge_size, temporal_patch_size=2,
                                     image_mean=[.5] * 3, image_std=[.5] * 3,
                                     size={"shortest_edge": min_pixels, "longest_edge": max_pixels})


def tiny_full_model(text_config, attention="sdpa"):
    """A random Qwen3_5ForConditionalGeneration around `text_config` (its last four ids are the video, image,
    vision_start and vision_end tokens) with a 2-block vision tower (patch 4, merge 2)."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    vocab = text_config.vocab_size
    vision_config = Qwen3_5VisionConfig(depth=2, hidden_size=32, intermediate_size=64, num_heads=4, in_channels=3,
                                        patch_size=4, spatial_merge_size=2, temporal_patch_size=2,
                                        out_hidden_size=text_config.hidden_size, num_position_embeddings=16)
    config = Qwen3_5Config(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), image_token_id=vocab - 3,
                           vision_start_token_id=vocab - 2, vision_end_token_id=vocab - 1, video_token_id=vocab - 4,
                           tie_word_embeddings=False)
    config._attn_implementation = attention
    config.text_config._attn_implementation = attention
    config.vision_config._attn_implementation = attention
    return Qwen3_5ForConditionalGeneration(config)
