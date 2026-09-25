import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from janus.baseline import render_prompt, candidate_token_ids, label_logits
from janus.data import synthetic_requests
from janus.packing import ByteTokenizer


def test_label_prompt_contains_options_but_not_targets_or_question_ids():
    request = synthetic_requests(1)[0]
    question = request.questions[0]
    prompt = render_prompt(request, question)
    assert question.instructions in prompt
    assert all(o.description in prompt for o in question.options)
    assert 'target' not in prompt
    assert request.group_id not in prompt
    assert prompt.endswith('Answer:')
    ids = candidate_token_ids(ByteTokenizer(), len(question.options))
    assert ids == [66, 67, 68, 69]


def test_selected_label_logits_match_full_vocabulary_forward():
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=257, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).eval()
    tokens = torch.tensor([[1, 7, 12, 26]])
    ids = [66, 67, 68]
    with torch.inference_mode():
        expected = model(tokens).logits[0, -1, ids]
        actual = label_logits(model, tokens, ids)
    torch.testing.assert_close(actual, expected)


def test_baseline_artifact_identifies_device_and_calibration_count(monkeypatch, tmp_path):
    import janus.baseline as baseline
    from janus.data import write_jsonl
    config = Qwen3Config(vocab_size=257, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'eager'
    monkeypatch.setattr(baseline.AutoTokenizer, 'from_pretrained', lambda *a, **k: ByteTokenizer())
    monkeypatch.setattr(baseline.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: Qwen3ForCausalLM(config))
    panel, calibration = tmp_path / 'panel.jsonl', tmp_path / 'cal.jsonl'
    write_jsonl(panel, [r.to_dict() for r in synthetic_requests(2, seed=1)])
    write_jsonl(calibration, [r.to_dict() for r in synthetic_requests(2, seed=2)])
    result = baseline.evaluate_baseline(panel, calibration, tmp_path / 'result', backbone='tiny', revision='test', device='cpu')
    assert result['device'] == 'cpu'
    assert result['device_name'] == 'CPU'
    assert result['calibration_requests'] == 2
