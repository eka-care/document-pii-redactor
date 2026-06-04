# Eka-PII-redaction — single container running text + visual PII redaction.
#
# Base image ships torch + CUDA so the same image runs on GPU (if --gpus is
# passed) or CPU. Tesseract (text OCR) and libGL/glib (opencv used by ultralytics)
# are installed as system deps.
#
# Build:  docker build -t eka-pii-redaction .
# Run (GPU):  docker run --gpus all -p 8080:8080 eka-pii-redaction
# Run (CPU):  docker run -e EKA_PII_DEVICE=cpu -p 8080:8080 eka-pii-redaction
#
# The API listens on :8080. Models are pulled from the Hugging Face repo on first
# start (set EKA_PII_HF_REPO; mount a HF cache volume to avoid re-downloading).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

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

# Install the package + the API server extra.
RUN pip install --no-cache-dir ".[server]"

EXPOSE 8080
CMD ["uvicorn", "eka_pii_redaction.server:app", "--host", "0.0.0.0", "--port", "8080"]
