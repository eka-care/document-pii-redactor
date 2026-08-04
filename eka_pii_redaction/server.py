"""Minimal FastAPI service wrapping ImagePIIRedactor — lets the Docker image run
the whole thing as one container.

Endpoints:
    GET  /health                      -> {"status": "ok", "device": ...}
    GET  /entities                    -> {"text": [...], "visual": [...]}
    POST /detect  (multipart: file)   -> {"entities": [...]}
    POST /redact  (multipart: file, mode=solid|blur|pixelate) -> image/png
    POST /deidentify (multipart: file) -> {"image": <base64 png>, "mapping": ...}
    POST /anonymize  (multipart: file) -> image/png
    POST /detect-text | /redact-text | /deidentify-text | /anonymize-text (JSON)

Config via env vars (read at startup):
    EKA_PII_HF_REPO        Hugging Face repo id or local dir (default ekacare/pii-redactors)
    EKA_PII_DETECT_VISUAL  "1"/"0"  (default 1)
    EKA_PII_DEVICE         "cuda"/"cpu" (default auto)
    EKA_PII_EXCLUDE        comma-separated categories to exclude (default none)
"""
from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .image import DEFAULT_HF_REPO, ImagePIIRedactor

app = FastAPI(title="Eka-PII-redaction", version="0.1.0")
_redactor: ImagePIIRedactor | None = None
_text_redactor = None  # built lazily on first /detect-text or /redact-text call


def _parse_exclude(exclude: str | None) -> list[str] | None:
    """Comma-separated query param -> list, or None if empty/absent."""
    if not exclude:
        return None
    return [s for s in exclude.split(",") if s]


def _parse_hex_color(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


class _TextIn(BaseModel):
    text: str
    exclude: list[str] | None = None


class _RedactTextIn(BaseModel):
    text: str
    mask: str = "[REDACTED]"
    exclude: list[str] | None = None


class _DeidTextIn(BaseModel):
    text: str
    exclude: list[str] | None = None
    mapping: dict | None = None  # a prior call's mapping, to continue numbering


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
async def detect(file: UploadFile = File(...), exclude: str | None = Query(None)):
    data = await file.read()
    ents = _get().detect(data, exclude_entities=_parse_exclude(exclude))
    return JSONResponse({"entities": [e.to_dict() for e in ents]})


@app.post("/redact")
async def redact(
    file: UploadFile = File(...),
    mode: str = Form("solid"),
    color: str = Form("000000"),
    exclude: str | None = Query(None),
):
    data = await file.read()
    img = _get().redact(
        data, mode=mode, color=_parse_hex_color(color),
        exclude_entities=_parse_exclude(exclude),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/entities-text")
def entities_text():
    from .text import TextPIIRedactor
    return {"text": TextPIIRedactor.list_entities()}


@app.post("/detect-text")
def detect_text(body: _TextIn):
    spans = _get_text().detect(body.text, exclude_entities=body.exclude)
    return {"spans": [s.to_dict() for s in spans]}


@app.post("/redact-text")
def redact_text(body: _RedactTextIn):
    text = _get_text().redact(body.text, mask=body.mask, exclude_entities=body.exclude)
    return {"text": text}


@app.post("/deidentify-text")
def deidentify_text(body: _DeidTextIn):
    from .pseudonym import PseudonymMapping

    result = _get_text().deidentify(
        body.text, exclude_entities=body.exclude,
        mapping=PseudonymMapping.from_dict(body.mapping),
    )
    return {"text": result.text, "mapping": result.mapping.to_dict()}


@app.post("/anonymize-text")
def anonymize_text(body: _TextIn):
    return {"text": _get_text().anonymize(body.text, exclude_entities=body.exclude)}


@app.post("/deidentify")
async def deidentify(file: UploadFile = File(...), exclude: str | None = Query(None)):
    # JSON (base64 PNG + mapping) rather than an image response — the mapping
    # has to ride along with the image it de-identifies.
    import base64

    data = await file.read()
    result = _get().deidentify(data, exclude_entities=_parse_exclude(exclude))
    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    return JSONResponse({
        "image": base64.b64encode(buf.getvalue()).decode("ascii"),
        "mapping": result.mapping.to_dict(),
    })


@app.post("/anonymize")
async def anonymize(file: UploadFile = File(...), exclude: str | None = Query(None)):
    data = await file.read()
    img = _get().anonymize(data, exclude_entities=_parse_exclude(exclude))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# Serve the built React app (web/dist) at "/". Mounted last so it never shadows
# the API routes above — StaticFiles only matches what FastAPI's router doesn't.
_WEB_DIST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "dist")
if os.path.isdir(_WEB_DIST):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
