"""Streamlit app to test the Eka image PII redactor.

Run:
    pip install -e .            # or: pip install eka-pii-redaction  (needs Python >=3.10)
    pip install streamlit
    streamlit run streamlit_app.py

Two actions:
  - Detect  : annotate (draw) labeled bounding boxes around all detected PII.
  - Redact  : hide each detected PII region (solid / blur / pixelate).

The model repo is downloaded via huggingface_hub, which uses your cached HF
login (`huggingface-cli login`) or the HF_TOKEN env var automatically — no token
field in the UI.
"""
from __future__ import annotations

import io
import time

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from eka_pii_redaction.image import DEFAULT_HF_REPO
from eka_pii_redaction.taxonomy import TEXT_ENTITIES, VISUAL_ENTITIES

st.set_page_config(page_title="Eka PII Redactor — Image", layout="wide")

# Fixed defaults — the app always runs with this configuration.
HF_REPO = DEFAULT_HF_REPO
DEVICE = "auto"
DETECT_VISUAL = True
VISUAL_SCORE_THRESHOLD = 0.25
OCR_LANG = None

# A distinct color per coarse L1 group, for the detection overlay.
L1_COLORS = {
    "person": (220, 38, 38),
    "location": (37, 99, 235),
    "date_time": (217, 119, 6),
    "contact": (5, 150, 105),
    "uid": (124, 58, 237),
    "device_net": (8, 145, 178),
    "credential": (190, 18, 60),
    "entity": (100, 116, 139),
    "biometric_visual": (219, 39, 119),
    "unknown": (75, 85, 99),
}


@st.cache_resource(show_spinner=False)
def load_redactor(hf_repo: str, detect_visual: bool, device: str,
                  visual_score_threshold: float):
    """Build (and cache) an ImagePIIRedactor for the given config.

    Keeps BOTH models loaded: the LayoutLMv3 text detector and (when
    detect_visual is on) the YOLO visual detector.
    """
    from eka_pii_redaction import ImagePIIRedactor

    return ImagePIIRedactor(
        hf_repo,
        detect_visual=detect_visual,
        device=(None if device == "auto" else device),
        visual_score_threshold=visual_score_threshold,
    )


def draw_boxes(img: Image.Image, entities) -> Image.Image:
    """Overlay labeled bounding boxes (colored by L1 group) on a copy of img."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(13, out.width // 90))
    except Exception:
        font = ImageFont.load_default()

    for e in entities:
        x0, y0, x1, y1 = e.bbox
        color = L1_COLORS.get(e.l1, L1_COLORS["unknown"])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = e.category
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(label) * 6, 12
        ly = max(0, y0 - th - 4)
        draw.rectangle([x0, ly, x0 + tw + 6, ly + th + 4], fill=color)
        draw.text((x0 + 3, ly + 2), label, fill=(255, 255, 255), font=font)
    return out


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def entity_rows(entities):
    return [
        {
            "kind": e.kind,
            "category": e.category,
            "l1": e.l1,
            "text": e.text or "",
            "score": round(e.score, 3) if e.score is not None else None,
            "bbox": list(e.bbox),
        }
        for e in entities
    ]


# ---------------------------------------------------------------- sidebar --- #
st.sidebar.markdown("**Redaction settings**")
redact_mode = st.sidebar.selectbox("Mode", ["solid", "blur", "pixelate"], index=1)
hex_color = st.sidebar.color_picker(
    "Fill color (solid)", value="#000000", disabled=redact_mode != "solid"
)
redact_color = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

st.sidebar.markdown("**Exclude categories** (never detect/redact)")
exclude_text = st.sidebar.multiselect("Text categories", TEXT_ENTITIES, default=[])
exclude_visual = st.sidebar.multiselect("Visual categories", VISUAL_ENTITIES, default=[])
exclude_entities = list(exclude_text) + list(exclude_visual)

# ---------------------------------------------------------------- main ----- #
st.title("🛡️ Eka Image PII Redactor")

# Load (or reuse cached) models — both text and visual stay resident.
try:
    with st.spinner("Loading models (first run downloads weights — may take a while)…"):
        redactor = load_redactor(HF_REPO, DETECT_VISUAL, DEVICE, VISUAL_SCORE_THRESHOLD)
    st.sidebar.success(f"Model ready · device={redactor.device}")
except Exception as exc:  # noqa: BLE001 — surface load errors to the user
    st.error(f"Failed to load model: {exc}")
    st.info(
        "If the repo is gated/private, log in first with `huggingface-cli login` "
        "or set the HF_TOKEN env var before launching."
    )
    st.stop()

uploaded = st.file_uploader(
    "Upload a document image", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"]
)
if uploaded is None:
    st.info("⬆️ Upload an image to begin.")
    st.stop()

image = Image.open(uploaded).convert("RGB")

# ---- run (starts automatically on upload) ---- #
excl = exclude_entities or None
with st.spinner("Detecting…"):
    t0 = time.time()
    entities = redactor.detect(image, exclude_entities=excl, ocr_lang=OCR_LANG)
    detect_s = time.time() - t0

n_text = sum(1 for e in entities if e.kind == "text")
n_visual = sum(1 for e in entities if e.kind == "visual")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Entities", len(entities))
m2.metric("Text", n_text)
m3.metric("Visual", n_visual)
m4.metric("Detect time", f"{detect_s:.1f}s")

with st.spinner("Redacting…"):
    redacted = redactor.redact(
        image, mode=redact_mode, color=redact_color,
        exclude_entities=excl, ocr_lang=OCR_LANG,
    )
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Detections")
    st.image(draw_boxes(image, entities), use_container_width=True)
with col_b:
    st.subheader(f"Redacted ({redact_mode})")
    st.image(redacted, use_container_width=True)
    st.download_button(
        "⬇️ Download redacted PNG", data=to_png_bytes(redacted),
        file_name="redacted.png", mime="image/png", use_container_width=True,
    )

st.subheader("Detected entities")
if entities:
    st.dataframe(entity_rows(entities), use_container_width=True, hide_index=True)
else:
    st.info("No PII entities detected.")
