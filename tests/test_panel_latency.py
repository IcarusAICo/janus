import torch

from janus.benchmark import profile_panel
from janus.data import synthetic_requests, write_jsonl
from janus.model import DecisionModel, ModelConfig
from janus.training import checkpoint


def test_profile_counts_full_requests_and_reports_timing_boundary(tmp_path):
    torch.set_num_threads(2)
    model = DecisionModel(ModelConfig(backbone='tiny', adaptation='full', hidden_size=32, head_rank=16))
    weights = tmp_path / 'model.pt'
    checkpoint(model, weights, {})
    data = tmp_path / 'panel.jsonl'
    write_jsonl(data, [r.to_dict() for r in synthetic_requests(4)])
    result = profile_panel(weights, data, device='cpu', limit=3, warmup=1)
    assert result['requests'] == 3
    assert result['decisions'] == 9
    assert result['p50_ms'] > 0
    assert result['p95_ms'] >= result['p50_ms']
    assert result['includes_tokenization'] is True
    assert result['includes_network'] is False
    assert len(result['per_request']) == 3
    assert all(r['packed_tokens'] > 0 for r in result['per_request'])
