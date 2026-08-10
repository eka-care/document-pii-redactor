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
    EKA_PII_HF_REPO        Hugging Face repo id or local dir (default ekacare/document-pii-redactor)
    EKA_PII_DETECT_VISUAL  "1"/"0"  (default 1)
    EKA_PII_DEVICE         "cuda"/"cpu" (default auto)
    EKA_PII_CATEGORIES     comma-separated categories to detect (default: all)
"""
from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .image import DEFAULT_HF_REPO, ImagePIIRedactor

app = FastAPI(title="document-pii-redactor", version="0.1.0")
_redactor: ImagePIIRedactor | None = None
_text_redactor = None  # built lazily on first /detect-text or /redact-text call


def _parse_categories(categories: str | None) -> list[str] | None:
    """Comma-separated query param -> list, or None (= all) if empty/absent."""
    if not categories:
        return None
    return [s for s in categories.split(",") if s]


def _parse_hex_color(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


class _TextIn(BaseModel):
    text: str
    categories: list[str] | None = None  # None = all


class _RedactTextIn(BaseModel):
    text: str
    mask: str = "[REDACTED]"
    categories: list[str] | None = None


class _DeidTextIn(BaseModel):
    text: str
    categories: list[str] | None = None
    mapping: dict | None = None  # a prior call's mapping, to continue numbering
    strategy: str = "counter"    # "counter" (Person_1) | "hash" (Person_a3f9c1)
    secret: str | None = None    # optional salt for the hash strategy


def _env_categories(only_text: bool = False) -> list[str] | None:
    raw = [s for s in os.environ.get("EKA_PII_CATEGORIES", "").split(",") if s]
    if not raw:
        return None
    if only_text:
        # The env list may name visual categories (for the image model); keep
        # only the ones the text model knows so construction doesn't error.
        from .taxonomy import TEXT_REDACTABLE
        raw = [c for c in raw if c in TEXT_REDACTABLE]
    return raw or None


def _get_text():
    global _text_redactor
    if _text_redactor is None:
        from .text import TextPIIRedactor

        _text_redactor = TextPIIRedactor(
            hf_repo=os.environ.get("EKA_PII_HF_REPO", DEFAULT_HF_REPO),
            device=os.environ.get("EKA_PII_DEVICE") or None,
            categories=_env_categories(only_text=True),
        )
    return _text_redactor


def _get() -> ImagePIIRedactor:
    global _redactor
    if _redactor is None:
        _redactor = ImagePIIRedactor(
            hf_repo=os.environ.get("EKA_PII_HF_REPO", DEFAULT_HF_REPO),
            detect_visual=os.environ.get("EKA_PII_DETECT_VISUAL", "1") == "1",
            device=os.environ.get("EKA_PII_DEVICE") or None,
            categories=_env_categories(),
            text_score_threshold=float(
                os.environ.get("EKA_PII_TEXT_THRESHOLD", "0.75")),
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
async def detect(file: UploadFile = File(...), categories: str | None = Query(None)):
    data = await file.read()
    ents = _get().detect(data, categories=_parse_categories(categories))
    return JSONResponse({"entities": [e.to_dict() for e in ents]})


@app.post("/redact")
async def redact(
    file: UploadFile = File(...),
    mode: str = Form("solid"),
    color: str = Form("000000"),
    categories: str | None = Query(None),
):
    data = await file.read()
    ents = _get().detect(data, categories=_parse_categories(categories))
    img = _get().redact(data, ents, mode=mode, color=_parse_hex_color(color))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/entities-text")
def entities_text():
    from .text import TextPIIRedactor
    return {"text": TextPIIRedactor.list_entities()}


@app.post("/detect-text")
def detect_text(body: _TextIn):
    spans = _get_text().detect(body.text, categories=body.categories)
    return {"spans": [s.to_dict() for s in spans]}


@app.post("/redact-text")
def redact_text(body: _RedactTextIn):
    spans = _get_text().detect(body.text, categories=body.categories)
    return {"text": _get_text().redact(body.text, spans, mask=body.mask)}


@app.post("/deidentify-text")
def deidentify_text(body: _DeidTextIn):
    from .pseudonym import PseudonymMapping

    spans = _get_text().detect(body.text, categories=body.categories)
    result = _get_text().deidentify(
        body.text, spans, mapping=PseudonymMapping.from_dict(body.mapping),
        strategy=body.strategy, secret=body.secret)
    return {"text": result.text, "mapping": result.mapping.to_dict()}


@app.post("/anonymize-text")
def anonymize_text(body: _TextIn):
    spans = _get_text().detect(body.text, categories=body.categories)
    return {"text": _get_text().anonymize(body.text, spans)}


@app.post("/deidentify")
async def deidentify(file: UploadFile = File(...),
                     categories: str | None = Query(None),
                     strategy: str = Query("counter"),
                     secret: str | None = Query(None)):
    # JSON (base64 PNG + mapping) rather than an image response — the mapping
    # has to ride along with the image it de-identifies.
    import base64

    data = await file.read()
    ents = _get().detect(data, categories=_parse_categories(categories))
    result = _get().deidentify(data, ents, strategy=strategy, secret=secret)
    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    return JSONResponse({
        "image": base64.b64encode(buf.getvalue()).decode("ascii"),
        "mapping": result.mapping.to_dict(),
    })


@app.post("/anonymize")
async def anonymize(file: UploadFile = File(...), categories: str | None = Query(None)):
    data = await file.read()
    ents = _get().detect(data, categories=_parse_categories(categories))
    img = _get().anonymize(data, ents)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# Serve the built React app (web/dist) at "/". Mounted last so it never shadows
# the API routes above — StaticFiles only matches what FastAPI's router doesn't.
_WEB_DIST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "dist")
if os.path.isdir(_WEB_DIST):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
