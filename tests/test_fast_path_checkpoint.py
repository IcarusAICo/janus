from pathlib import Path

import pytest
import torch

CHECKPOINT = Path("runs/study-20260917/listwise-06b-seed17/best.pt")
PANEL = Path("data/study-v1/benchmark.jsonl")

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and CHECKPOINT.exists() and PANEL.exists()),
                                reason="needs the study checkpoint, the panel, and a GPU")


def test_flex_fp32_matches_eager_reference_on_a_panel_sample(tmp_path):
    from scripts.fast_path_check import compare_panel
    out = compare_panel(CHECKPOINT, PANEL, attention="flex", dtype="float32", limit=40, cache_dir=tmp_path)
    assert out["max_abs_logit_diff"] < 1e-3
    assert out["argmax_flips"] == 0


def test_sibling_perturbations_leave_target_unchanged_under_flex(tmp_path):
    from scripts.fast_path_check import sibling_gates
    out = sibling_gates(CHECKPOINT, PANEL, attention="flex", cases=10)
    assert out["max_target_total_variation"] < 1e-5


def test_lora_and_head_gradients_match_between_eager_and_flex_on_the_checkpoint():
    """Spec WP0 gate: gradients on head, LoRA and shared state path on the unchanged checkpoint."""
    import random
    from janus.data import load_requests
    from janus.packing import pack_request
    from janus.training import load_checkpoint
    requests = random.Random(17).sample(load_requests(PANEL), 3)
    grads = {}
    for attention in ("eager", "flex"):
        model, _ = load_checkpoint(CHECKPOINT, "cuda:0", attention=attention)
        model.train()
        model.zero_grad(set_to_none=True)
        for request in requests:
            packed = pack_request(request, model.tokenizer, model.packing_mode if hasattr(model, "packing_mode") else model.config.mode,
                                  model.config.max_tokens)
            loss = sum((z.float() * torch.arange(len(z), device=z.device)).sum() + z.float().square().sum() for z in model(packed))
            loss.backward()
        grads[attention] = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        del model
        torch.cuda.empty_cache()
    assert grads["eager"].keys() == grads["flex"].keys() and any("lora" in n for n in grads["eager"])
    for name, g in grads["eager"].items():
        torch.testing.assert_close(grads["flex"][name], g, atol=1e-4, rtol=1e-3, msg=name)


def test_sibling_perturbations_have_an_eager_baseline(tmp_path):
    """The flex sibling residual is compared against the same perturbations under eager."""
    from scripts.fast_path_check import sibling_gates
    eager = sibling_gates(CHECKPOINT, PANEL, attention="eager", cases=10)
    flex = sibling_gates(CHECKPOINT, PANEL, attention="flex", cases=10)
    assert eager["max_target_total_variation"] < 1e-5 and flex["max_target_total_variation"] < 1e-5
