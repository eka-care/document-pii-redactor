"""Streamlit app to test the Eka PII redactors — image and text modalities.

Run:
    pip install -e .            # needs Python >=3.10
    pip install streamlit
    streamlit run streamlit_app.py

Two tabs:
  - Image : redact PII regions in a document image (auto-runs on upload).
  - Text  : detect PII in pasted text and highlight each span (color-coded by group).

Both models are loaded once at startup and kept resident (cached) for the session.
The model repo is downloaded via huggingface_hub, which uses your cached HF login
(`huggingface-cli login`) or the HF_TOKEN env var automatically.
"""
from __future__ import annotations

import html
import io
import time

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from eka_pii_redaction.image import DEFAULT_HF_REPO
from eka_pii_redaction.taxonomy import TEXT_ENTITIES, TEXT_REDACTABLE, VISUAL_ENTITIES

st.set_page_config(page_title="Eka PII Redactor — Tester", layout="wide")

# Fixed model config — the app always loads with this configuration (no UI knobs).
HF_REPO = DEFAULT_HF_REPO
DEVICE = "auto"
DETECT_VISUAL = True
VISUAL_SCORE_THRESHOLD = 0.25
OCR_LANG = None

# A distinct color per coarse L1 group, used by both the image overlay and the
# text highlighter.
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


# ---------------------------------------------------------------- loaders --- #
# Both redactors are cached resources: loaded once and kept resident for the whole
# session (no lazy per-action loading).
@st.cache_resource(show_spinner=False)
def load_image_redactor(hf_repo: str, detect_visual: bool, device: str,
                        visual_score_threshold: float):
    from eka_pii_redaction import ImagePIIRedactor

    return ImagePIIRedactor(
        hf_repo,
        detect_visual=detect_visual,
        device=(None if device == "auto" else device),
        visual_score_threshold=visual_score_threshold,
    )


@st.cache_resource(show_spinner=False)
def load_text_redactor(hf_repo: str, device: str):
    from eka_pii_redaction import TextPIIRedactor

    return TextPIIRedactor(hf_repo, device=(None if device == "auto" else device))


# ---------------------------------------------------------------- shared --- #
def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _rgb(group: str) -> str:
    c = L1_COLORS.get(group, L1_COLORS["unknown"])
    return f"rgb({c[0]},{c[1]},{c[2]})"


def legend_html(groups) -> str:
    chips = "".join(
        f'<span style="background:{_rgb(g)};color:#fff;padding:1px 7px;'
        f'border-radius:4px;margin:0 6px 4px 0;display:inline-block;font-size:.8rem;">'
        f'{html.escape(g)}</span>'
        for g in groups
    )
    return f'<div style="margin:4px 0 10px;">{chips}</div>'


# ---------------------------------------------------------------- image ----- #
def draw_boxes(img: Image.Image, entities) -> Image.Image:
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


# ---------------------------------------------------------------- text ------ #
def highlight_spans(text: str, spans) -> str:
    """Return HTML with each PII span wrapped in a color-coded <mark>."""
    parts, prev = [], 0
    for sp in sorted(spans, key=lambda s: s.start):
        if sp.start < prev:
            continue
        parts.append(html.escape(text[prev:sp.start]))
        score = f"{sp.score:.3f}" if sp.score is not None else "?"
        parts.append(
            f'<mark style="background:{_rgb(sp.l1)};color:#fff;padding:0 3px;'
            f'border-radius:3px;" title="{html.escape(sp.category)} · {score}">'
            f'{html.escape(text[sp.start:sp.end])}</mark>'
        )
        prev = sp.end
    parts.append(html.escape(text[prev:]))
    body = "".join(parts)
    return (
        '<div style="white-space:pre-wrap;line-height:2;font-size:1.05rem;'
        'border:1px solid #ddd;border-radius:8px;padding:12px;">' + body + "</div>"
    )


def span_rows(spans):
    return [
        {
            "category": s.category,
            "l1": s.l1,
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "score": round(s.score, 3) if s.score is not None else None,
        }
        for s in spans
    ]


# ---------------------------------------------------------------- startup -- #
# Load BOTH models up front and keep them resident (cached) for the session.
try:
    with st.spinner("Loading models (first run downloads weights — may take a while)…"):
        image_redactor = load_image_redactor(
            HF_REPO, DETECT_VISUAL, DEVICE, VISUAL_SCORE_THRESHOLD)
        text_redactor = load_text_redactor(HF_REPO, DEVICE)
    st.sidebar.success(
        f"Models ready · image={image_redactor.device} · text={text_redactor.device}")
except Exception as exc:  # noqa: BLE001 — surface load errors to the user
    st.error(f"Failed to load models: {exc}")
    st.info(
        "If the repo is gated/private, log in first with `huggingface-cli login` "
        "or set the HF_TOKEN env var before launching."
    )
    st.stop()

st.title("🛡️ Eka PII Redactor — Tester")
tab_img, tab_txt = st.tabs(["🖼️ Image", "📝 Text"])

# ================================================================ IMAGE ==== #
with tab_img:
    st.subheader("Image modality")
    st.caption("Upload a document image — it is detected and redacted automatically.")

    rc1, rc2 = st.columns([2, 1])
    redact_mode = rc1.selectbox("Redaction mode", ["solid", "blur", "pixelate"], index=1)
    hex_color = rc2.color_picker(
        "Fill color (solid)", value="#000000", disabled=redact_mode != "solid")
    redact_color = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

    with st.expander("Exclude categories (never detect/redact)"):
        exclude_text = st.multiselect(
            "Text categories", TEXT_ENTITIES, default=[], key="img_excl_text")
        exclude_visual = st.multiselect(
            "Visual categories", VISUAL_ENTITIES, default=[], key="img_excl_visual")
    img_exclude = (list(exclude_text) + list(exclude_visual)) or None

    uploaded = st.file_uploader(
        "Upload a document image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    )

    if uploaded is None:
        st.info("⬆️ Upload an image to begin.")
    else:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("Detecting…"):
            t0 = time.time()
            entities = image_redactor.detect(
                image, exclude_entities=img_exclude, ocr_lang=OCR_LANG)
            detect_s = time.time() - t0

        n_text = sum(1 for e in entities if e.kind == "text")
        n_visual = sum(1 for e in entities if e.kind == "visual")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entities", len(entities))
        m2.metric("Text", n_text)
        m3.metric("Visual", n_visual)
        m4.metric("Detect time", f"{detect_s:.1f}s")

        with st.spinner("Redacting…"):
            redacted = image_redactor.redact(
                image, entities, mode=redact_mode, color=redact_color)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Detections")
            st.image(draw_boxes(image, entities), use_container_width=True)
        with col_b:
            st.caption(f"Redacted ({redact_mode})")
            st.image(redacted, use_container_width=True)
            st.download_button(
                "⬇️ Download redacted PNG", data=to_png_bytes(redacted),
                file_name="redacted.png", mime="image/png", use_container_width=True,
            )

        st.caption("Detected entities")
        if entities:
            st.dataframe(entity_rows(entities), use_container_width=True, hide_index=True)
        else:
            st.info("No PII entities detected.")

# ================================================================ TEXT ===== #
with tab_txt:
    st.subheader("Text modality")
    st.caption("Detects PII in raw text and highlights each span (color-coded by group).")

    with st.expander("Exclude categories (never detect)"):
        txt_exclude = st.multiselect(
            "Text categories", TEXT_REDACTABLE, default=[], key="txt_excl") or None

    # An editable example loaded by default — packed with varied PII so the demo
    # shows several categories at once. Users can edit or replace it freely.
    EXAMPLE_TEXT = (
        "DISCHARGE SUMMARY\n"
        "Patient: Mr. John Doe (Male, 45 yrs), DOB 12-03-1979, Blood group O+.\n"
        "Address: 14 MG Road, Indiranagar, Bangalore, Karnataka 560038.\n"
        "Aadhaar: 1234 5678 9012  |  PAN: ABCDE1234F  |  ABHA: 14-1234-5678-9012.\n"
        "MRN/UHID: UH00219834. Insurance policy: TPA-IND-99213.\n"
        "Contact: +91 98765 43210, email john.doe@example.com.\n"
        "Treating physician: Dr. Asha Menon (NMC Reg. 2011/04/1123).\n"
        "Payment received to UPI johndoe@okhdfcbank, A/C 50100123456789."
    )
    text_in = st.text_area(
        "Text to scan (an example is preloaded — edit or replace it)",
        value=EXAMPLE_TEXT, height=220,
    )
    run_txt = st.button(
        "🔍 Detect PII", type="primary", use_container_width=True,
        disabled=not text_in.strip(), key="run_txt",
    )

    if run_txt and text_in.strip():
        with st.spinner("Detecting…"):
            t0 = time.time()
            spans = text_redactor.detect(text_in, exclude_entities=txt_exclude)
            detect_s = time.time() - t0

        m1, m2 = st.columns(2)
        m1.metric("Spans", len(spans))
        m2.metric("Detect time", f"{detect_s:.2f}s")

        if spans:
            groups = sorted({s.l1 for s in spans})
            st.markdown(legend_html(groups), unsafe_allow_html=True)
            st.markdown(highlight_spans(text_in, spans), unsafe_allow_html=True)
            st.caption("Detected spans")
            st.dataframe(span_rows(spans), use_container_width=True, hide_index=True)
        else:
            st.info("No PII detected.")
    else:
        st.info("✏️ Edit the preloaded example (or paste your own) and press **Detect PII**.")
