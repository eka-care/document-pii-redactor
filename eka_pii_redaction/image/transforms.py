"""Image-side de-identification/anonymization: grouping + in-place rendering.

The detector emits one entity per OCR word (see layoutlmv3.py), but pseudonym
consistency needs logical entities — "John Doe" must become one "Person_1",
not "Person_1 Person_1". `group_text_entities` merges same-category words that
sit on one line within a small horizontal gap; the substitute value is then
rendered into the merged box over a background-colored fill. The result
intentionally looks patched — edits should be detectable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..entities import PIIEntity
from ..pseudonym import PseudonymMapping
from ..text.transforms import anonymize_value

# Two boxes are "same line" if they overlap vertically by at least half the
# shorter box; words merge when the horizontal gap is under this fraction of
# the line height (typical inter-word gaps are ~0.25-0.5x height).
_MIN_VERTICAL_OVERLAP = 0.5
_MAX_GAP_FACTOR = 0.6

# Placeholder labels for visual-only entities under de-identification — the
# reader should know something was there without seeing it. Anonymization
# fills these solid black instead: biometrics have nothing to generalize, so
# destruction is the anonymization.
VISUAL_PLACEHOLDER_LABELS: dict[str, str] = {
    "face_photo": "[FACE]", "signature": "[SIGNATURE]", "seal_stamp": "[STAMP]",
    "fingerprint_thumb_impression": "[FINGERPRINT]", "qr_barcode": "[QR]",
    "logo": "[LOGO]",
}


@dataclass
class EntityGroup:
    """One logical text entity: merged words, union bbox, joined text."""
    category: str
    bbox: tuple[int, int, int, int]
    text: str


@dataclass
class ImageDeidResult:
    """De-identified image plus the mapping the caller may store securely."""
    image: Image.Image
    mapping: PseudonymMapping


def _same_line(a: tuple, b: tuple) -> bool:
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    shorter = min(a[3] - a[1], b[3] - b[1])
    return shorter > 0 and overlap >= _MIN_VERTICAL_OVERLAP * shorter


def group_text_entities(entities: list[PIIEntity]) -> list[EntityGroup]:
    """Merge adjacent same-category words into logical entities.

    Words are walked in (line, x) order; each either extends the previous
    group of its category (same line, gap <= _MAX_GAP_FACTOR x line height)
    or starts a new one. Visual entities are not grouped.
    """
    words = sorted(
        (e for e in entities if e.kind == "text"),
        key=lambda e: ((e.bbox[1] + e.bbox[3]) / 2, e.bbox[0]),
    )
    groups: list[EntityGroup] = []
    open_by_category: dict[str, EntityGroup] = {}

    for w in words:
        x0, y0, x1, y1 = w.bbox
        prev = open_by_category.get(w.category)
        if prev is not None and _same_line(prev.bbox, w.bbox):
            line_height = min(prev.bbox[3] - prev.bbox[1], y1 - y0)
            if 0 <= x0 - prev.bbox[2] <= _MAX_GAP_FACTOR * line_height:
                prev.bbox = (min(prev.bbox[0], x0), min(prev.bbox[1], y0),
                             max(prev.bbox[2], x1), max(prev.bbox[3], y1))
                prev.text = f"{prev.text} {w.text or ''}".strip()
                continue
        group = EntityGroup(category=w.category, bbox=(x0, y0, x1, y1),
                            text=(w.text or "").strip())
        groups.append(group)
        open_by_category[w.category] = group
    return groups


# ---------------------------------------------------------------- render --- #
def _background_color(img: Image.Image, bbox: tuple) -> tuple[int, int, int]:
    """Median color of a 2px ring just outside the box (clamped to image)."""
    W, H = img.size
    x0, y0, x1, y1 = bbox
    rx0, ry0 = max(0, x0 - 2), max(0, y0 - 2)
    rx1, ry1 = min(W, x1 + 2), min(H, y1 + 2)
    region = np.asarray(img.crop((rx0, ry0, rx1, ry1)), dtype=np.uint8)
    if region.size == 0:
        return (255, 255, 255)
    mask = np.ones(region.shape[:2], dtype=bool)
    ix0, iy0 = x0 - rx0, y0 - ry0
    ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)
    mask[max(0, iy0):max(0, iy1), max(0, ix0):max(0, ix1)] = False
    ring = region[mask]
    if ring.size == 0:
        ring = region.reshape(-1, region.shape[-1])
    return tuple(int(v) for v in np.median(ring, axis=0)[:3])


def _load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fitted_font(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int):
    size = max(8, int(box_h * 0.75))
    font = _load_font(size)
    while size > 8 and draw.textlength(text, font=font) > box_w:
        size -= 1
        font = _load_font(size)
    return font


def render_text_in_box(img: Image.Image, bbox: tuple, replacement: str) -> None:
    """Erase the box to its surrounding background color and draw `replacement`."""
    draw = ImageDraw.Draw(img)
    bg = _background_color(img, bbox)
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], fill=bg)
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    ink = (20, 20, 20) if luminance > 140 else (245, 245, 245)
    font = _fitted_font(draw, replacement, max(1, x1 - x0 - 2), max(1, y1 - y0))
    top = draw.textbbox((0, 0), replacement, font=font)
    text_h = top[3] - top[1]
    draw.text((x0 + 1, y0 + max(0, ((y1 - y0) - text_h) // 2) - top[1]),
              replacement, fill=ink, font=font)


def render_visual_placeholder(img: Image.Image, bbox: tuple, category: str) -> None:
    """Neutral gray fill + centered category label (de-identification only)."""
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], fill=(232, 232, 232), outline=(180, 180, 180))
    label = VISUAL_PLACEHOLDER_LABELS.get(category, "[REMOVED]")
    font = _fitted_font(draw, label, max(1, x1 - x0 - 4), max(1, (y1 - y0) // 3))
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((x0 + max(0, ((x1 - x0) - tw) // 2), y0 + max(0, ((y1 - y0) - th) // 2) - tb[1]),
              label, fill=(120, 120, 120), font=font)


# -------------------------------------------------------------- appliers --- #
def apply_deidentify(img: Image.Image, entities: list[PIIEntity],
                     mapping: PseudonymMapping) -> Image.Image:
    out = img.convert("RGB").copy()
    for group in group_text_entities(entities):
        pseudonym = mapping.pseudonym_for(group.category, group.text)
        render_text_in_box(out, group.bbox, pseudonym)
    for e in entities:
        if e.kind == "visual":
            render_visual_placeholder(out, e.bbox, e.category)
    return out


def apply_anonymize(img: Image.Image, entities: list[PIIEntity]) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for group in group_text_entities(entities):
        render_text_in_box(out, group.bbox, anonymize_value(group.category, group.text))
    for e in entities:
        if e.kind == "visual":
            draw.rectangle(list(e.bbox), fill=(0, 0, 0))
    return out
