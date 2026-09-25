"""Reference protocols for the post-trained Qwen3.5 weights on MMLU-Pro (evaluation only; docs/phase4/post-trained.md).

Three protocols beside janus.baseline's zero-shot label-likelihood rule, on the same items:
  thinking  generative chain of thought in the checkpoint's thinking mode, the model card's sampling, answer parsed
            from the text (parse_answer);
  direct    the same question with thinking disabled through the chat template, a few new tokens, the same parser;
  fewshot   label-likelihood with five calibration exemplars in the prompt (protocol change, no weight change).
"""

import argparse
import json
from pathlib import Path
import re
import string
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from .data import assert_disjoint, file_hash, load_requests, write_json, write_jsonl

# The Qwen3.5 model card's recommendation for thinking mode and for "instruct (non-thinking) mode for reasoning tasks".
SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 1.5}
INSTRUCTION = {
    'thinking': 'Please reason step by step, then finish with "The answer is (X)", where X is the letter of the correct option.',
    'direct': 'Answer with only "The answer is (X)", where X is the letter of the correct option.',
}
# Primary parse: the last "answer is (X)" / "Answer: X" / \boxed{X} (letter optionally in parentheses, bold or \text{}).
ANSWER = re.compile(r'(?:(?i:answer is|answer:)|\\boxed\{)\s*[(*{]*(?:\\text\{)?\s*([A-Z])(?![A-Za-z])')
# Fallback: the last standalone capital letter. ponytail: "I" as a pronoun can hit for ten-option items; the report
# counts how often the fallback decided, so the ceiling is visible.
LETTER = re.compile(r'(?<![A-Za-z])([A-Z])(?![A-Za-z])')
STOP = ('<|im_end|>', '<|endoftext|>')


def parse_answer(text, letters):
    """(letter, how) with how in pattern / letter / none. The ANSWER pattern over the content after the last </think>,
    then the standalone-letter fallback over that content, then the ANSWER pattern over the whole text (a truncated
    thinking block may still have named an answer). The fallback never reads reasoning: "A" is also an article there."""
    content = text.rsplit('</think>', 1)[-1] if '</think>' in text else ''
    for how, pattern, chunk in (('pattern', ANSWER, content), ('letter', LETTER, content), ('pattern', ANSWER, text)):
        found = [m.group(1) for m in pattern.finditer(chunk) if m.group(1) in letters]
        if found:
            return found[-1], how
    return None, 'none'


def render_question(request, question):
    options = '\n'.join(f'{letter}. {option.description}' for letter, option in zip(string.ascii_uppercase, question.options))
    return f'{question.instructions}\n\n{request.state}\n\n{options}'


def chat_prompt(tokenizer, text, thinking):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=thinking)


class PresencePenalty(LogitsProcessor):
    """The OpenAI/vLLM presence penalty: subtract `penalty` from every token already generated (the prompt excluded)."""

    def __init__(self, penalty, prompt_length):
        self.penalty, self.prompt_length = penalty, prompt_length

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] == 0 or self.penalty == 0:
            return scores
        present = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, generated, True)
        return scores - self.penalty * present.to(scores.dtype)


def token_counts(ids, stop, think_end):
    """(new tokens before the first stop/pad id, tokens up to and including </think>, thinking closed?)."""
    length = next((i for i, t in enumerate(ids) if t in stop), len(ids))
    ids = ids[:length]
    closed = think_end in ids
    return length, ids.index(think_end) + 1 if closed else length, closed


@torch.inference_mode()
def generate(model, tokenizer, prompts, max_new_tokens, seed, batch_size=32, sampling=SAMPLING):
    """Batched sampling over left-padded prompts. Returns per prompt: text, new_tokens, thinking_tokens, thinking_closed.
    The seed is reset per batch, so a run is reproducible for a fixed batch size."""
    tokenizer.padding_side = 'left'
    think_end = tokenizer.convert_tokens_to_ids('</think>')
    stop = {tokenizer.convert_tokens_to_ids(t) for t in STOP} | {tokenizer.pad_token_id}
    rows = []
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[start:start + batch_size], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        prompt_length = batch['input_ids'].shape[1]
        torch.manual_seed(seed + start)
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=True, temperature=sampling['temperature'],
                             top_p=sampling['top_p'], top_k=sampling['top_k'], eos_token_id=sorted(stop),
                             pad_token_id=tokenizer.pad_token_id,
                             logits_processor=LogitsProcessorList([PresencePenalty(sampling['presence_penalty'], prompt_length)]))
        for ids in out[:, prompt_length:].tolist():
            length, thinking, closed = token_counts(ids, stop, think_end)
            rows.append({'text': tokenizer.decode(ids[:length]), 'new_tokens': length,
                         'thinking_tokens': thinking, 'thinking_closed': closed})
    return rows


def load(backbone, revision, device, dtype='bfloat16', attention='sdpa'):
    if str(device).startswith('cuda'):
        torch.cuda.set_device(torch.device(device))  # see janus.baseline: fla builds its norm on the current device
    tokenizer = AutoTokenizer.from_pretrained(backbone, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(backbone, revision=revision, dtype=getattr(torch, dtype),
                                                 attn_implementation=attention, trust_remote_code=False).to(device).eval()
    return model, tokenizer


def provenance(backbone, revision, device, dtype, data_path):
    return {'backbone': backbone, 'revision': revision, 'device': str(device), 'dtype': dtype,
            'device_name': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
            'data_sha256': file_hash(data_path)}


def evaluate_generative(data_path, output, backbone, revision, device='cuda:0', mode='thinking', max_new_tokens=2048,
                        batch_size=32, seed=17, limit=None, dtype='bfloat16'):
    """MMLU-Pro accuracy of sampled text answers; unparseable answers count as wrong."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requests = load_requests(data_path)[:limit]
    model, tokenizer = load(backbone, revision, device, dtype)
    items = [(r, q) for r in requests for q in r.questions]
    prompts = [chat_prompt(tokenizer, render_question(r, q) + '\n\n' + INSTRUCTION[mode], mode == 'thinking') for r, q in items]
    started = time.perf_counter()
    rows = generate(model, tokenizer, prompts, max_new_tokens, seed, batch_size)
    wall = time.perf_counter() - started
    records = []
    for (request, question), row in zip(items, rows):
        letters = string.ascii_uppercase[:len(question.options)]
        predicted, how = parse_answer(row['text'], letters)
        target = letters[max(range(len(question.target)), key=question.target.__getitem__)]
        records.append({'group_id': request.group_id, 'question_id': question.id, 'target': target,
                        'predicted': predicted, 'parsed_by': how, 'correct': predicted == target, **row})
    n = len(records)
    report = {**provenance(backbone, revision, device, dtype, data_path), 'mode': mode, 'sampling': SAMPLING,
              'instruction': INSTRUCTION[mode], 'seed': seed, 'max_new_tokens': max_new_tokens, 'batch_size': batch_size,
              'items': n, 'accuracy': sum(r['correct'] for r in records) / n,
              'unparseable_share': sum(r['predicted'] is None for r in records) / n,
              'parsed_by': {how: sum(r['parsed_by'] == how for r in records) for how in ('pattern', 'letter', 'none')},
              'thinking_closed_share': sum(r['thinking_closed'] for r in records) / n,
              'truncated_share': sum(r['new_tokens'] >= max_new_tokens for r in records) / n,
              'mean_thinking_tokens': sum(r['thinking_tokens'] for r in records) / n,
              'mean_new_tokens': sum(r['new_tokens'] for r in records) / n,
              'wall_seconds': wall, 'wall_seconds_per_item': wall / n}
    write_jsonl(output / 'predictions.jsonl', records)
    write_json(output / 'metrics.json', report)
    return report


def fewshot_prefix(exemplars):
    """Rendered exemplar prompts, each followed by its gold letter."""
    from .baseline import render_prompt
    parts = []
    for request in exemplars:
        for question in request.questions:
            gold = string.ascii_uppercase[max(range(len(question.target)), key=question.target.__getitem__)]
            parts.append(render_prompt(request, question) + ' ' + gold + '\n\n')
    return ''.join(parts)


@torch.inference_mode()
def evaluate_fewshot(data_path, calibration_path, output, backbone, revision, device='cuda:0', shots=5, dtype='bfloat16', limit=None):
    """janus.baseline's label-likelihood rule with the first `shots` calibration items as exemplars; the temperature is
    fitted on the remaining calibration items."""
    from .baseline import candidate_token_ids, label_logits, render_prompt
    from .metrics import fit_temperature, metrics
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requests, calibration = load_requests(data_path)[:limit], load_requests(calibration_path)
    assert_disjoint({'test': requests, 'calibration': calibration})
    exemplars, calibration = calibration[:shots], calibration[shots:]
    model, tokenizer = load(backbone, revision, device, dtype, attention='eager')
    prefix = fewshot_prefix(exemplars)

    def collect(rows):
        logits, targets = [], []
        for request in rows:
            for question in request.questions:
                ids = candidate_token_ids(tokenizer, len(question.options))
                tokens = tokenizer.encode(prefix + render_prompt(request, question), add_special_tokens=False)
                logits.append(label_logits(model, torch.tensor([tokens], device=device), ids).cpu())
                targets.append(torch.tensor(question.target))
        return logits, targets

    calibration_z, calibration_y = collect(calibration)
    with torch.inference_mode(False), torch.enable_grad():
        temperature = fit_temperature([z.clone() for z in calibration_z], [y.clone() for y in calibration_y])
    started = time.perf_counter()
    logits, targets = collect(requests)
    report = {**provenance(backbone, revision, device, dtype, data_path), 'baseline': 'few-shot candidate vocabulary logits',
              'shots': shots, 'exemplars': [r.group_id for r in exemplars], 'prefix_tokens': len(tokenizer.encode(prefix, add_special_tokens=False)),
              'calibration_sha256': file_hash(calibration_path), 'calibration_requests': len(calibration),
              'requests': len(requests), 'temperature': temperature, 'raw': metrics(logits, targets),
              'calibrated': metrics(logits, targets, temperature), 'wall_seconds': time.perf_counter() - started}
    items = [(r.group_id, q.id) for r in requests for q in r.questions]
    write_jsonl(output / 'predictions.jsonl', [{'group_id': g, 'question_id': q, 'logits': z.tolist(), 'target': y.tolist()}
                                               for (g, q), z, y in zip(items, logits, targets)])
    write_json(output / 'metrics.json', report)
    return report


# Decomposition table: (label, {column: metrics.json path}). Phase 3 paths hold the rows measured before this track.
CELLS = [
    ('zero-shot label-likelihood', {'4B Base': 'runs/phase3/mmlu-size/qwen35-4b', '4B post': 'runs/phase3/mmlu-size/qwen35-4b-instruct',
                                    '9B Base': 'runs/phase3/mmlu-size/qwen35-9b', '9B post': 'runs/phase3/mmlu-size/qwen35-9b-instruct'}),
    ('5-shot label-likelihood', {'4B Base': '{out}/4b/base_fewshot', '4B post': '{out}/4b/post_fewshot',
                                 '9B Base': '{out}/9b/base_fewshot', '9B post': '{out}/9b/post_fewshot'}),
    ('decision model (Phase 1 data, no MMLU-Pro)', {'4B Base': 'runs/phase3/backbones/qwen35_4b_tree/mmlu_pro', '4B post': '{out}/qwen35_4b_post_tree/mmlu_pro',
                                                    '9B Base': 'runs/phase3/backbones/qwen35_9b_tree/mmlu_pro', '9B post': '{out}/qwen35_9b_post_tree/mmlu_pro'}),
    ('direct answer, thinking off (generative)', {'4B post': '{out}/4b/post_direct', '9B post': '{out}/9b/post_direct'}),
    ('chain of thought, thinking on, 300 calibration items', {'4B post': '{out}/4b/post_thinking_300', '9B post': '{out}/9b/post_thinking_300'}),
    ('chain of thought, thinking on, 1,000 items', {'4B post': '{out}/4b/post_thinking_1000', '9B post': '{out}/9b/post_thinking_1000'}),
    ('chain of thought, 8,192-token budget, 300 calibration items', {'4B post': '{out}/4b/post_thinking_300_8k', '9B post': '{out}/9b/post_thinking_300_8k'}),
    ('model card (thinking, external)', {'4B post': 0.791, '9B post': 0.825}),
]
COLUMNS = ('4B Base', '4B post', '9B Base', '9B post')


def read_cell(path):
    """accuracy, or (accuracy, unparseable share, truncated share) for a generative run; None when not landed."""
    path = Path(path) / 'metrics.json'
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    if 'accuracy' in m:
        return (m['accuracy'], m['unparseable_share'], m['truncated_share'])
    return m['raw']['accuracy']


def fmt(cell):
    if cell is None:
        return ''
    if isinstance(cell, tuple):
        return f'{cell[0]:.3f} (unparsed {cell[1]:.2f}, truncated {cell[2]:.2f})'
    return f'{cell:.3f}'


def write_table(out):
    """runs/.../table.md and table.json from whatever metrics have landed; empty cells are still running."""
    out = Path(out)
    table = {}
    for label, cells in CELLS:
        table[label] = {c: (v if isinstance(v, float) else read_cell(str(v).format(out=out))) for c, v in cells.items()}
    lines = ['| protocol | ' + ' | '.join(COLUMNS) + ' |', '| --- |' + ' ---: |' * len(COLUMNS)]
    for label, row in table.items():
        lines.append(f'| {label} | ' + ' | '.join(fmt(row.get(c)) for c in COLUMNS) + ' |')
    (out / 'table.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'table.json', table)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('thinking', 'direct', 'fewshot', 'table'))
    parser.add_argument('--data')
    parser.add_argument('--calibration-data', help='fewshot: exemplars and temperature fit')
    parser.add_argument('--output')
    parser.add_argument('--root', default='runs/phase4/posttrained', help='table: where the runs live')
    parser.add_argument('--backbone', default='Qwen/Qwen3.5-4B')
    parser.add_argument('--revision', default='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    if args.mode == 'table':
        write_table(args.root)
        print((Path(args.root) / 'table.md').read_text())
        return
    if args.mode == 'fewshot':
        report = evaluate_fewshot(args.data, args.calibration_data, args.output, args.backbone, args.revision, args.device, limit=args.limit)
    else:
        report = evaluate_generative(args.data, args.output, args.backbone, args.revision, args.device, args.mode,
                                     args.max_new_tokens or (2048 if args.mode == 'thinking' else 8),
                                     args.batch_size, args.seed, args.limit)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict) or k in ('parsed_by', 'raw')}, indent=2))


if __name__ == '__main__':
    main()
