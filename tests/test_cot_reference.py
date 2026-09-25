import torch
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

from janus.cot_reference import (PresencePenalty, fewshot_prefix, generate, parse_answer, render_question, token_counts,
                               write_table)
from janus.data import synthetic_requests, write_json

LETTERS = 'ABCDEFGHIJ'


def test_parser_takes_the_last_pattern_match_after_the_thinking_block():
    text = '<think>\nMaybe A. The answer is (B)? No.\n</think>\n\nSo the answer is (C). Actually, **the answer is D**.'
    assert parse_answer(text, LETTERS) == ('D', 'pattern')
    assert parse_answer('<think>\n...\n</think>\n\nThe answer is \\boxed{\\text{E}}.', LETTERS) == ('E', 'pattern')
    assert parse_answer('Answer: F', LETTERS) == ('F', 'pattern')
    assert parse_answer('The answer is (K).', LETTERS) == (None, 'none')  # out of range for ten options
    assert parse_answer('The answer is Answer.', LETTERS) == (None, 'none')


def test_parser_falls_back_to_a_standalone_letter_then_to_the_thinking_text():
    assert parse_answer('<think>\nx\n</think>\n\nI would pick G here.', LETTERS) == ('G', 'letter')  # last standalone
    assert parse_answer('<think>\nthe answer is (H) but let me check', LETTERS) == ('H', 'pattern')  # truncated thinking
    assert parse_answer('<think>\nnothing decided\n</think>\n\n', LETTERS) == (None, 'none')
    assert parse_answer('<think>\nA common view is that I should', LETTERS) == (None, 'none')  # no fallback in reasoning
    assert parse_answer('Hello world', LETTERS) == (None, 'none')


def test_question_rendering_hides_the_target():
    request = synthetic_requests(1)[0]
    question = request.questions[0]
    text = render_question(request, question)
    assert request.state in text and question.instructions in text
    assert all(f'{l}. {o.description}' in text for l, o in zip(LETTERS, question.options))
    assert 'target' not in text
    prefix = fewshot_prefix(synthetic_requests(2))
    assert prefix.count('Answer: ') == sum(len(r.questions) for r in synthetic_requests(2))


def test_presence_penalty_only_touches_generated_tokens():
    scores = torch.zeros(2, 10)
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 7]])  # prompt length 2
    out = PresencePenalty(1.5, 2)(ids, scores)
    expected = torch.zeros(2, 10)
    expected[0, [3, 4]] = -1.5
    expected[1, [7]] = -1.5
    torch.testing.assert_close(out, expected)
    assert PresencePenalty(1.5, 4)(ids, scores) is scores  # nothing generated yet


def test_token_counts_stop_at_the_first_stop_id_and_measure_thinking():
    assert token_counts([5, 6, 9, 7, 0, 3], stop={0}, think_end=9) == (4, 3, True)
    assert token_counts([5, 6, 7], stop={0}, think_end=9) == (3, 3, False)
    assert token_counts([0, 5], stop={0}, think_end=9) == (0, 0, False)


class FakeTokenizer:
    """Bytes as tokens, left padding, the batch interface generate() uses; 0 pads, 254 ends thinking, 255 stops."""
    pad_token_id = 0
    padding_side = 'right'

    def convert_tokens_to_ids(self, token):
        return {'</think>': 254, '<|im_end|>': 255, '<|endoftext|>': 0}[token]

    def __call__(self, texts, return_tensors, padding, add_special_tokens):
        rows = [[b + 1 for b in t.encode()] for t in texts]
        width = max(map(len, rows))
        ids = torch.tensor([[0] * (width - len(r)) + r for r in rows])
        return BatchEncoding({'input_ids': ids, 'attention_mask': (ids != 0).long()})

    def decode(self, ids):
        return bytes(i - 1 for i in ids if 0 < i < 254).decode(errors='replace')


def tiny_model():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, attention_dropout=0., pad_token_id=0)
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config).eval()


def test_batched_generation_is_seeded_padded_and_counted():
    model, tokenizer = tiny_model(), FakeTokenizer()
    prompts = ['short', 'a much longer prompt here', 'mid size']
    rows = generate(model, tokenizer, prompts, max_new_tokens=12, seed=3, batch_size=2)
    again = generate(model, tokenizer, prompts, max_new_tokens=12, seed=3, batch_size=2)
    assert [r['text'] for r in rows] == [r['text'] for r in again]
    assert len(rows) == 3 and tokenizer.padding_side == 'left'
    for row in rows:
        assert 0 <= row['thinking_tokens'] <= row['new_tokens'] <= 12
        assert len(row['text']) <= row['new_tokens']  # one byte-token decodes to at most one character
    assert [r['text'] for r in generate(model, tokenizer, prompts, max_new_tokens=12, seed=4, batch_size=2)] != [r['text'] for r in rows]


def test_table_reads_generative_and_likelihood_metrics(tmp_path, monkeypatch):
    import janus.cot_reference as module
    monkeypatch.setattr(module, 'CELLS', [('gen', {'4B post': '{out}/gen'}), ('lik', {'4B post': '{out}/lik', '9B post': '{out}/missing'}),
                                          ('card', {'9B post': 0.825})])
    write_json(tmp_path / 'gen' / 'metrics.json', {'accuracy': 0.5, 'unparseable_share': 0.1, 'truncated_share': 0.25})
    write_json(tmp_path / 'lik' / 'metrics.json', {'raw': {'accuracy': 0.25}})
    table = write_table(tmp_path)
    assert table == {'gen': {'4B post': (0.5, 0.1, 0.25)}, 'lik': {'4B post': 0.25, '9B post': None}, 'card': {'9B post': 0.825}}
    text = (tmp_path / 'table.md').read_text()
    assert '| gen |  | 0.500 (unparsed 0.10, truncated 0.25) |  |  |' in text and '| lik |  | 0.250 |  |  |' in text and '0.825' in text
