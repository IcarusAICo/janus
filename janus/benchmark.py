"""Synchronized forward-only timings of packed and repeated-state computation."""

from dataclasses import replace
import random
import time

import numpy as np
import torch

from .packing import pack_request
from .schema import Request
from .training import load_checkpoint


@torch.inference_mode()
def profile_panel(checkpoint_path, data_path, device='cpu', limit=None, warmup=3):
    """Warm network-free request latency including packing and CPU probabilities."""
    from .data import file_hash, load_requests
    if limit is not None and limit < 1 or warmup < 0:
        raise ValueError('limit must be positive and warmup nonnegative')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, metadata = load_checkpoint(checkpoint_path, device)
    requests = load_requests(data_path)
    if not requests:
        raise ValueError('Panel must contain requests')
    if limit is not None:
        requests = random.Random(17).sample(requests, min(limit, len(requests)))

    def synchronize():
        if model.device.type == 'cuda':
            torch.cuda.synchronize(model.device)

    def forward(request):
        packed = pack_request(request, model.tokenizer, model.packing_mode, model.config.max_tokens, **model.packing_kwargs)
        probabilities = [z.float().softmax(-1).cpu().tolist() for z in model(packed)]
        return packed.token_count, probabilities, getattr(model, 'last_compute', None)

    for index in range(warmup):
        forward(requests[index % len(requests)])
    synchronize()
    if model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(model.device)
    records = []
    for request in requests:
        synchronize()
        start = time.perf_counter()
        tokens, _, compute = forward(request)
        synchronize()
        records.append({'group_id': request.group_id, 'question_ids': [q.id for q in request.questions],
                        'questions': len(request.questions), 'packed_tokens': tokens,
                        'milliseconds': (time.perf_counter() - start) * 1000,
                        **({'compute': dict(compute)} if compute else {})})
    times = [r['milliseconds'] for r in records]
    # Mean per-request compute accounting when the model reports it (backbone tokens; decoder cross-attention pairs).
    keys = sorted({k for r in records for k in r.get('compute', {})})
    compute = {k: float(np.mean([r['compute'][k] for r in records if k in r.get('compute', {})])) for k in keys}
    return {'checkpoint_sha256': file_hash(checkpoint_path), 'data_sha256': file_hash(data_path),
            'device': str(device), 'device_name': torch.cuda.get_device_name(model.device)
            if model.device.type == 'cuda' else 'CPU', 'model': metadata['model'],
            'timing_scope': 'warm sequential local requests: tokenization, packing, transfers, forward, softmax and CPU list',
            'includes_tokenization': True, 'includes_network': False, 'includes_model_loading': False,
            'temperature': 1., 'warmup': warmup, 'requests': len(requests),
            'decisions': sum(len(r.questions) for r in requests),
            **({'compute': compute} if compute else {}),
            'p50_ms': float(np.median(times)), 'p95_ms': float(np.quantile(times, .95)),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(model.device) if model.device.type == 'cuda' else None,
            'per_request': records}


@torch.inference_mode()
def benchmark(checkpoint_path, device="cpu", repeats=20, warmup=3,
              question_counts=(1, 4, 8), state_words=(32, 128, 512)):
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    model, metadata = load_checkpoint(checkpoint_path, device)
    torch.set_num_threads(4)
    result = {"device": str(device), "device_name": torch.cuda.get_device_name(model.device)
              if model.device.type == "cuda" else "CPU", "model": metadata["model"],
              "timing_scope": "forward only, pretokenized, sequential separate-question baseline",
              "repeats": repeats, "warmup": warmup, "measurements": []}

    def synchronize():
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)

    def measure(packs):
        for _ in range(warmup):
            for packed in packs:
                model(packed)
        synchronize()
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        times = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            for packed in packs:
                model(packed)
            synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return {"p50_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95)),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None,
                "processed_tokens": sum(p.token_count for p in packs)}

    for words in state_words:
        for count in question_counts:
            raw = {"state": "red " * words, "questions": {str(i): {
                "type": "choice", "instructions": "Which color is mentioned?",
                "criteria": {"a": "red", "b": "blue", "c": "green", "d": "yellow"}}
                for i in range(count)}}
            request = Request.from_dict(raw)
            try:
                packed = pack_request(request, model.tokenizer, model.packing_mode, model.config.max_tokens,
                                      **model.packing_kwargs)
                separate = [pack_request(replace(request, questions=(q,)), model.tokenizer, model.packing_mode,
                                         model.config.max_tokens, **model.packing_kwargs) for q in request.questions]
            except ValueError as error:
                result["measurements"].append({"state_words": words, "questions": count, "skipped": str(error)})
                continue
            # Check semantic equivalence at the precision actually being timed.
            packed_z = model(packed)
            separate_z = [model(p)[0] for p in separate]
            max_error = max(float((a - b).abs().max()) for a, b in zip(packed_z, separate_z))
            result["measurements"].append({"state_words": words, "state_tokens": packed.state_length,
                 "questions": count, "max_logit_difference": max_error,
                 "packed": measure([packed]), "separate": measure(separate)})
    return result
