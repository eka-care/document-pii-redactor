"""Minimal FastAPI service wrapping ImagePIIRedactor — lets the Docker image run
the whole thing as one container.

Endpoints:
    GET  /health                      -> {"status": "ok", "device": ...}
    GET  /entities                    -> {"text": [...], "visual": [...]}
    POST /detect  (multipart: file)   -> {"entities": [...]}
    POST /redact  (multipart: file, mode=solid|blur|pixelate) -> image/png

Config via env vars (read at startup):
    EKA_PII_HF_REPO        Hugging Face repo id or local dir (default ekacare/pii-redactors)
    EKA_PII_DETECT_VISUAL  "1"/"0"  (default 1)
    EKA_PII_DEVICE         "cuda"/"cpu" (default auto)
    EKA_PII_EXCLUDE        comma-separated categories to exclude (default none)
"""
from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .image import DEFAULT_HF_REPO, ImagePIIRedactor

app = FastAPI(title="Eka-PII-redaction", version="0.1.0")
_redactor: ImagePIIRedactor | None = None
_text_redactor = None  # built lazily on first /detect-text or /redact-text call


class _TextIn(BaseModel):
    text: str


class _RedactTextIn(BaseModel):
    text: str
    mask: str = "[REDACTED]"


def _get_text():
    global _text_redactor
    if _text_redactor is None:
        from .taxonomy import TEXT_REDACTABLE
        from .text import TextPIIRedactor

        raw = [s for s in os.environ.get("EKA_PII_EXCLUDE", "").split(",") if s]
        # EKA_PII_EXCLUDE may name visual categories (for the image model); keep
        # only the ones the text model knows so construction doesn't error.
        exclude = [e for e in raw if e in TEXT_REDACTABLE]
        _text_redactor = TextPIIRedactor(
            hf_repo=os.environ.get("EKA_PII_HF_REPO", DEFAULT_HF_REPO),
            device=os.environ.get("EKA_PII_DEVICE") or None,
            exclude_entities=exclude or None,
        )
    return _text_redactor


def _get() -> ImagePIIRedactor:
    global _redactor
    if _redactor is None:
        exclude = [s for s in os.environ.get("EKA_PII_EXCLUDE", "").split(",") if s]
        _redactor = ImagePIIRedactor(
            hf_repo=os.environ.get("EKA_PII_HF_REPO", DEFAULT_HF_REPO),
            detect_visual=os.environ.get("EKA_PII_DETECT_VISUAL", "1") == "1",
            device=os.environ.get("EKA_PII_DEVICE") or None,
            exclude_entities=exclude or None,
        )
    return _redactor


@app.on_event("startup")
def _warm():
    _get()  # load models at startup so the first request is fast


@app.get("/health")
def health():
    return {"status": "ok", "device": _get().device,
            "detect_visual": _get().detect_visual}


@app.get("/entities")
def entities():
    return ImagePIIRedactor.list_entities()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    ents = _get().detect(data)
    return JSONResponse({"entities": [e.to_dict() for e in ents]})


@app.post("/redact")
async def redact(file: UploadFile = File(...), mode: str = Form("solid")):
    data = await file.read()
    img = _get().redact(data, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/entities-text")
def entities_text():
    from .text import TextPIIRedactor
    return {"text": TextPIIRedactor.list_entities()}


@app.post("/detect-text")
def detect_text(body: _TextIn):
    spans = _get_text().detect(body.text)
    return {"spans": [s.to_dict() for s in spans]}


@app.post("/redact-text")
def redact_text(body: _RedactTextIn):
    return {"text": _get_text().redact(body.text, mask=body.mask)}
