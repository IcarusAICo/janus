#!/usr/bin/env bash
# scripts/competitors/setup_venvs.sh: one venv per competitor (never the main env), pinned sources, CUDA 12.8 torch
# (sm_120 for the 5090 and sm_86 for the 3090 are both in the cu128 wheels), weights into the repo's HF cache.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_CACHE="$PWD/.cache/huggingface/hub"
JEVK5_REF=${JEVK5_REF:-01d3cc99e11d5a080d9d15dd232eaa89e0941c1d}   # allebee/jevk5 main, 2026-09-23
LAYA_VERSION=${LAYA_VERSION:-0.3.11}
TORCH_INDEX=https://download.pytorch.org/whl/cu128
for name in jevk5 laya; do
  [ -x .venvs/$name/bin/python ] || uv venv -q --python 3.12 .venvs/$name
  uv pip install -q -p .venvs/$name/bin/python torch --index-url $TORCH_INDEX
done
uv pip install -q -p .venvs/jevk5/bin/python "jevk5[fast] @ git+https://github.com/allebee/jevk5@$JEVK5_REF" --extra-index-url $TORCH_INDEX
uv pip install -q -p .venvs/laya/bin/python "laya[serve]==$LAYA_VERSION" --extra-index-url $TORCH_INDEX
# Weights: JevK5 (merged 4B + jevk5_config.json temperature); Laya's bundle repo holds all three checkpoints.
.venvs/jevk5/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('alibiserikbay/JevK5'))"
.venvs/laya/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('convaiinnovations/laya'))"
for name in jevk5 laya; do
  .venvs/$name/bin/python -c "import torch, transformers; print('$name', torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.get_arch_list() if torch.cuda.is_available() else 'no cuda visible')"
done
