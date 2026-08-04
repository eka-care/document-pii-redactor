# Eka-PII-redaction — single container running text + visual PII redaction,
# plus the React demo UI (web/), served by the same FastAPI process.
#
# Base image ships torch + CUDA so the same image runs on GPU (if --gpus is
# passed) or CPU. Tesseract (text OCR) and libGL/glib (opencv used by ultralytics)
# are installed as system deps.
#
# Build:  docker build -t eka-pii-redaction .
# Run (GPU):  docker run --gpus all -p 7860:7860 eka-pii-redaction
# Run (CPU):  docker run -e EKA_PII_DEVICE=cpu -p 7860:7860 eka-pii-redaction
#
# The API + UI listen on :7860 (HF Spaces' conventional Docker port). Models
# are pulled from the Hugging Face repo on first start (set EKA_PII_HF_REPO;
# mount a HF cache volume to avoid re-downloading).
FROM node:20-slim AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# transformers>=5.0 (required by pyproject.toml) imports torch APIs
# (torch.distributed.tensor.DTensor, torch.float8_e8m0fnu) that don't exist
# before torch ~2.9; pinned to 2.12.0 to match the version verified working
# in local dev.
FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
COPY --from=web-builder /web/dist /app/web/dist

# Install the package + the API server extra. --break-system-packages: this
# base image's Python is externally-managed (PEP 668); safe here since the
# container is single-purpose and isolated.
RUN pip install --no-cache-dir --break-system-packages ".[server]"

EXPOSE 7860
CMD ["uvicorn", "eka_pii_redaction.server:app", "--host", "0.0.0.0", "--port", "7860"]
