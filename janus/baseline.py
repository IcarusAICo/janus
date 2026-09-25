"""One-pass vocabulary-label scoring without decision training or text decoding."""

import hashlib
import json
from pathlib import Path
import string
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import file_hash, load_requests, write_json, write_jsonl
from .metrics import fit_temperature, metrics


def candidate_token_ids(tokenizer, count):
    if not 1 <= count <= 26:
        raise ValueError('The single-letter baseline supports 1–26 options')
    letters = string.ascii_uppercase[:count]
    for prefix in (' ', ''):
        tokenized = [tokenizer.encode(prefix + letter, add_special_tokens=False) for letter in letters]
        if all(len(ids) == 1 for ids in tokenized):
            ids = [ids[0] for ids in tokenized]
            if len(set(ids)) == count:
                return ids
    raise ValueError('Candidate letters must each map to a distinct single vocabulary token')


def render_prompt(request, question):
    options = '\n'.join(f'{letter}. [{option.key}] {option.description}'
                        for letter, option in zip(string.ascii_uppercase, question.options))
    return ('Choose the best answer to the question using the state. Answer with only the option letter.\n\n'
            f'State:\n{request.state}\n\nQuestion:\n{question.instructions}\n\nOptions:\n{options}\n\nAnswer:')


def label_logits(model, tokens, ids):
    # Avoid allocating [sequence, full-vocabulary] logits when only a few rows are needed.
    hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state[0, -1].float()
    weights = model.get_output_embeddings().weight[ids].float()
    return torch.mv(weights, hidden)


@torch.inference_mode()
def evaluate_baseline(data_path, calibration_path, output, backbone='Qwen/Qwen3-0.6B-Base',
                      revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd', device='cuda:0', dtype='float32'):
    from .data import assert_disjoint
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    requests, calibration_requests = load_requests(data_path), load_requests(calibration_path)
    assert_disjoint({'test': requests, 'calibration': calibration_requests})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(backbone, revision=revision, trust_remote_code=False)
    if str(device).startswith('cuda'):
        # Qwen3.5 (hybrid) layers build their fused gated norm on the *current* CUDA device when flash-linear-attention
        # is installed; make that the evaluation device so nothing touches another GPU.
        torch.cuda.set_device(torch.device(device))
    # Qwen3.5-*-Base checkpoints are Qwen3_5ForConditionalGeneration; AutoModelForCausalLM maps them to the text-only
    # Qwen3_5ForCausalLM (language-model weights, tied LM head, vision tower ignored).
    model = AutoModelForCausalLM.from_pretrained(backbone, revision=revision,
                dtype=getattr(torch, dtype), attn_implementation='eager', trust_remote_code=False).to(device).eval()

    def collect(rows):
        logits, targets, times = [], [], []
        for request in rows:
            start = time.perf_counter()
            for question in request.questions:
                ids = candidate_token_ids(tokenizer, len(question.options))
                prompt = tokenizer.encode(render_prompt(request, question), add_special_tokens=False)
                if len(prompt) > model.config.max_position_embeddings:
                    raise ValueError('Baseline input exceeds the pretrained context limit')
                tokens = torch.tensor([prompt], device=device)
                logits.append(label_logits(model, tokens, ids).cpu())
                targets.append(torch.tensor(question.target))
            if str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
        return logits, targets, times

    # Warm up before reporting network-free, sequential-question, end-to-end request latency.
    collect(requests[:2])
    calibration_z, calibration_y, _ = collect(calibration_requests)
    # LBFGS needs gradients for the temperature, but never for backbone outputs.
    with torch.inference_mode(False), torch.enable_grad():
        temperature = fit_temperature([z.clone() for z in calibration_z], [y.clone() for y in calibration_y])
    logits, targets, times = collect(requests)
    records, by_domain = [], {}
    index = 0
    for request in requests:
        for q in request.questions:
            z, y = logits[index], targets[index]
            signature = {'state': request.state, 'kind': q.kind, 'instructions': q.instructions,
                         'options': [(o.key, o.description) for o in q.options]}
            signature = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            records.append({'group_id': request.group_id, 'question_id': q.id, 'kind': q.kind,
                            'input_sha256': signature, 'keys': [o.key for o in q.options],
                            'cardinality': len(q.options), 'logits': z.tolist(), 'target': y.tolist(),
                            'probabilities': (z / temperature).softmax(-1).tolist(),
                            'nll': metrics([z], [y])['nll'],
                            'calibrated_nll': metrics([z], [y], temperature)['nll']})
            by_domain.setdefault(q.id.split(':', 1)[0], []).append(index)
            index += 1
    report = {'backbone': backbone, 'revision': revision, 'baseline': 'one-pass candidate vocabulary logits',
              'device': str(device), 'device_name': torch.cuda.get_device_name(device)
              if str(device).startswith('cuda') else 'CPU',
              'data_sha256': file_hash(data_path), 'calibration_sha256': file_hash(calibration_path),
              'requests': len(requests), 'calibration_requests': len(calibration_requests),
              'temperature': temperature, 'dtype': dtype,
              'raw': metrics(logits, targets), 'calibrated': metrics(logits, targets, temperature),
              'by_domain': {d: {'raw': metrics([logits[i] for i in ix], [targets[i] for i in ix]),
                               'calibrated': metrics([logits[i] for i in ix], [targets[i] for i in ix], temperature)}
                            for d, ix in by_domain.items()},
              'latency': {'scope': 'warm local end-to-end requests, sequential question forwards, no network',
                          'p50_ms': float(np.median(times)), 'p95_ms': float(np.quantile(times, .95))}}
    write_jsonl(output / 'predictions.jsonl', records)
    write_json(output / 'metrics.json', report)
    return report
