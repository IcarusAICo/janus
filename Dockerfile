# A Janus checkpoint behind POST /v1/systemone, offline at run time.
#
# CUDA (default; linux/amd64; torch 2.11.0 wheels for CUDA 12.8, NVIDIA driver R570+):
#   docker build -t janus-serve:4b --build-arg MODEL_REPO=TODO/janus-4b --build-arg MODEL_REVISION=<commit sha> .
#   docker run --rm --gpus all -p 127.0.0.1:8080:8080 janus-serve:4b
#
# CPU (float32 or bfloat16 as saved; slow on the 4B):
#   docker build -t janus-serve:0.8b-cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
#     --build-arg MODEL_REPO=TODO/janus-0.8b --build-arg MODEL_REVISION=<commit sha> .
#   docker run --rm -p 127.0.0.1:8080:8080 janus-serve:0.8b-cpu
#
# A local export (scripts/export_hf.py OUT_DIR) instead of the Hub: add `--build-context model=OUT_DIR`, which replaces
# the `model` stage below, so MODEL_REPO/MODEL_REVISION are not used.
#
# Weights are downloaded once, here: the Janus repo at a full commit sha (model.pt checked against MODEL_SHA256 when
# given) and the Qwen backbone at the revision pinned in janus_config.json. Nothing is fetched at run time.
# Arguments after the image name are appended to `python -m janus serve` (e.g. --no-fast, --alias jev-latest, or
# --max-tokens N to replace the default 16,384-token request limit; the checkpoints were trained on requests up to 9,216).
# Set JANUS_SERVER_TOKEN to require a bearer token; unset, the server accepts any caller.
# On a GPU the server pre-captures the CUDA graphs of 269 common request shapes before it answers requests (about 55 s for the
# 4B, 29 s for the 0.8B); GET /v1/models returns 200 once it is ready. Append --no-precapture to skip it.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS base
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install torch==2.11.0 --index-url "${TORCH_INDEX}" \
 && pip install transformers==5.3.0 peft==0.19.1 numpy 'huggingface_hub[hf_xet]>=1.0' \
 && if [ "${TORCH_INDEX##*/}" != cpu ]; then pip install flash-linear-attention==0.5.2 triton==3.6.0; fi

FROM base AS model
ARG MODEL_REPO=TODO/janus-4b
ARG MODEL_REVISION=main
RUN hf download "${MODEL_REPO}" --revision "${MODEL_REVISION}" --local-dir /model && rm -rf /model/.cache

FROM base
ARG MODEL_SHA256=
WORKDIR /opt/janus
COPY pyproject.toml README.md LICENSE ./
COPY janus janus
RUN pip install --no-deps . && python -c "import janus.server"
COPY --from=model . /models/janus
ENV HF_HOME=/models/hf
RUN if [ -n "${MODEL_SHA256}" ]; then echo "${MODEL_SHA256}  /models/janus/model.pt" | sha256sum -c -; fi \
 && python -c "import json; c = json.load(open('/models/janus/janus_config.json')); \
from huggingface_hub import snapshot_download; snapshot_download(c['base_model'], revision=c['base_revision'])" \
 && rm -rf /models/hf/xet
RUN useradd --create-home --uid 10001 janus
USER janus

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 CMD python -c "import os, urllib.request as u; \
u.urlopen(u.Request('http://127.0.0.1:%s/v1/models' % os.environ['PORT'], headers={'Authorization': 'Bearer ' + os.environ.get('JANUS_SERVER_TOKEN', '')}), timeout=4)"
# The device is cuda when torch sees a GPU, else cpu; the model id comes from janus_config.json.
ENTRYPOINT ["sh", "-c", "exec python -m janus serve --checkpoint /models/janus/model.pt --calibration /models/janus/calibration.json \
--model-id \"$(python -c 'import json; print(json.load(open(\"/models/janus/janus_config.json\"))[\"model_id\"])')\" \
--device \"$(python -c 'import torch; print(\"cuda\" if torch.cuda.is_available() else \"cpu\")')\" --host 0.0.0.0 --port \"$PORT\" --max-tokens 16384 \"$@\"", "janus-serve"]
