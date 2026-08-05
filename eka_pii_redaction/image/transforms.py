"""Image-side de-identification/anonymization: grouping + in-place rendering.

The detector emits one entity per OCR word (see layoutlmv3.py), but pseudonym
consistency needs logical entities — "John Doe" must become one "Person_1",
not "Person_1 Person_1". Words are grouped three ways: into visual lines
(tolerant of mixed OCR box heights), into same-line runs per category, and
finally into blocks — same-category runs on tightly stacked consecutive lines
(a header's multi-line qualification list) merge into one region. Substitute
values are rendered over a background-colored fill at the size of the
original words (not the union box), so labels sit naturally beside the
surrounding print. The result still reads as edited — by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..entities import PIIEntity
from ..pseudonym import PseudonymMapping
from ..text.transforms import anonymize_value

# Line clustering: a word joins a line when its vertical center is within
# this fraction of the line's height from the line's center. Same-line words
# merge when the horizontal gap is under _MAX_GAP_FACTOR x line height —
# generous on purpose, because an under-merged run renders as pseudonym
# confetti while an over-merged one just erases a slightly larger area.
_LINE_CENTER_TOLERANCE = 0.6
_MAX_GAP_FACTOR = 1.5
# Block merge: same-category groups on consecutive lines join when the
# vertical gap is under this fraction of the text height and they overlap
# horizontally. Tight enough that table rows (padded apart) stay separate.
_MAX_BLOCK_GAP_FACTOR = 0.7
_MIN_BLOCK_X_OVERLAP = 0.5

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
    word_heights: list[int] = field(default_factory=list)

    @property
    def type_height(self) -> int:
        """The size the original text was printed at (not the union bbox).

        85th percentile of member word heights, after dropping outliers under
        half the tallest word: punctuation gets tiny boxes that drag a median
        down, while boxes without ascenders/descenders under-measure the type
        — the tall end of the distribution is closest to the true print size.
        """
        if not self.word_heights:
            return self.bbox[3] - self.bbox[1]
        tallest = max(self.word_heights)
        kept = sorted(h for h in self.word_heights if h >= 0.5 * tallest)
        return kept[min(len(kept) - 1, int(0.85 * len(kept)))]


@dataclass
class ImageDeidResult:
    """De-identified image plus the mapping the caller may store securely."""
    image: Image.Image
    mapping: PseudonymMapping


def _cluster_lines(words: list[PIIEntity]) -> list[list[PIIEntity]]:
    """Bucket words into visual lines, tolerant of mixed box heights.

    A word joins an existing line when its vertical center is close to the
    line's running center (relative to the taller of the two heights).
    Returns lines top-to-bottom, each line's words left-to-right — reading
    order, which also makes pseudonym numbering follow reading order.
    """
    lines: list[list[PIIEntity]] = []
    centers: list[float] = []
    heights: list[float] = []
    for w in sorted(words, key=lambda e: (e.bbox[1] + e.bbox[3]) / 2):
        c = (w.bbox[1] + w.bbox[3]) / 2
        h = w.bbox[3] - w.bbox[1]
        for i in range(len(lines)):
            if abs(c - centers[i]) <= _LINE_CENTER_TOLERANCE * max(heights[i], h):
                lines[i].append(w)
                centers[i] += (c - centers[i]) / len(lines[i])
                heights[i] = max(heights[i], h)
                break
        else:
            lines.append([w])
            centers.append(c)
            heights.append(h)
    order = sorted(range(len(lines)), key=lambda i: centers[i])
    return [sorted(lines[i], key=lambda e: e.bbox[0]) for i in order]


def _merge_blocks(groups: list[EntityGroup]) -> list[EntityGroup]:
    """Merge same-category groups stacked on tightly consecutive lines.

    A multi-line run (a header's qualification list, a wrapped address) reads
    as ONE entity; without this it renders as a column of numbered labels.
    Groups arrive in reading order, so each new group only needs to check the
    blocks already kept. Table rows don't merge — their row padding exceeds
    the gap threshold.
    """
    merged: list[EntityGroup] = []
    for g in groups:
        target = None
        for m in merged:
            if m.category != g.category:
                continue
            h = min(m.type_height, g.type_height)
            gap = g.bbox[1] - m.bbox[3]
            x_overlap = min(m.bbox[2], g.bbox[2]) - max(m.bbox[0], g.bbox[0])
            min_width = min(m.bbox[2] - m.bbox[0], g.bbox[2] - g.bbox[0])
            if (-0.3 * h <= gap <= _MAX_BLOCK_GAP_FACTOR * h
                    and x_overlap >= _MIN_BLOCK_X_OVERLAP * min_width):
                target = m
                break
        if target is None:
            merged.append(g)
        else:
            target.bbox = (min(target.bbox[0], g.bbox[0]), min(target.bbox[1], g.bbox[1]),
                           max(target.bbox[2], g.bbox[2]), max(target.bbox[3], g.bbox[3]))
            target.text = f"{target.text} {g.text}".strip()
            target.word_heights.extend(g.word_heights)
    return merged


def _mostly_inside(inner: tuple, outer: tuple, threshold: float = 0.7) -> bool:
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return area > 0 and (ix * iy) / area >= threshold


def group_text_entities(entities: list[PIIEntity]) -> list[EntityGroup]:
    """Merge word runs into logical entities; groups return in reading order.

    Words are clustered into visual lines (mixed OCR box heights on one
    printed line must not break a name apart), then walked left-to-right: a
    word extends its category's open group when the horizontal gap is within
    _MAX_GAP_FACTOR x line height, even across interleaved words of other
    categories. Same-category groups on tightly stacked lines then merge
    into blocks. Visual entities are not grouped, and text words sitting
    mostly inside a visual entity's box are dropped — they're part of that
    graphic (a logo's lettering), already covered by its placeholder.
    """
    visual_boxes = [e.bbox for e in entities if e.kind == "visual"]
    words = [e for e in entities if e.kind == "text"
             and not any(_mostly_inside(e.bbox, v) for v in visual_boxes)]
    groups: list[EntityGroup] = []
    for line in _cluster_lines(words):
        line_height = max(e.bbox[3] - e.bbox[1] for e in line)
        open_by_category: dict[str, EntityGroup] = {}
        for w in line:
            x0, y0, x1, y1 = w.bbox
            prev = open_by_category.get(w.category)
            if prev is not None and 0 <= x0 - prev.bbox[2] <= _MAX_GAP_FACTOR * line_height:
                prev.bbox = (min(prev.bbox[0], x0), min(prev.bbox[1], y0),
                             max(prev.bbox[2], x1), max(prev.bbox[3], y1))
                prev.text = f"{prev.text} {w.text or ''}".strip()
                prev.word_heights.append(y1 - y0)
                continue
            group = EntityGroup(category=w.category, bbox=(x0, y0, x1, y1),
                                text=(w.text or "").strip(),
                                word_heights=[y1 - y0])
            groups.append(group)
            open_by_category[w.category] = group
    return _merge_blocks(groups)


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


# Resolved once: first scalable font that actually loads on this system.
# PIL's bare load_default() is a fixed ~10px bitmap that IGNORES the size
# argument — silently falling back to it renders every label tiny.
_FONT_CANDIDATES = [
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Debian/Ubuntu
    "/System/Library/Fonts/Supplemental/Arial.ttf",      # macOS
    "/System/Library/Fonts/Helvetica.ttc",               # macOS fallback
    "Arial.ttf",
]
_font_path: str | None = None
_font_searched = False


def _load_font(size: int):
    global _font_path, _font_searched
    if not _font_searched:
        _font_searched = True
        for cand in _FONT_CANDIDATES:
            try:
                ImageFont.truetype(cand, 12)
                _font_path = cand
                break
            except OSError:
                continue
    if _font_path is not None:
        return ImageFont.truetype(_font_path, size)
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1: scalable
    except TypeError:
        return ImageFont.load_default()


def _fitted_font(draw: ImageDraw.ImageDraw, text: str, box_w: int,
                 start_size: int):
    """Font at `start_size`, shrunk only as needed to fit box_w."""
    size = max(8, start_size)
    font = _load_font(size)
    while size > 8 and draw.textlength(text, font=font) > box_w:
        size -= 1
        font = _load_font(size)
    return font


def _bucket_em_sizes(groups: list[EntityGroup]) -> list[int]:
    """Per-group font size, quantized to three document-relative buckets.

    Fully dynamic sizing amplified OCR noise — a garbage box spanning two
    stacked lines rendered a billboard label. Instead: the document's body
    size is the word-count-weighted median of group type heights, and every
    label is small (0.8x), body (1x), or large (1.6x) by how its own type
    height compares. Uniform sizes also simply read better. The 1.2 factor
    compensates DejaVu's ~0.72 cap-height-to-em ratio so body labels match
    neighboring print.
    """
    if not groups:
        return []
    heights: list[int] = []
    for g in groups:
        heights.extend([g.type_height] * max(1, len(g.word_heights)))
    heights.sort()
    body = heights[len(heights) // 2]
    sizes = []
    for g in groups:
        ratio = g.type_height / body if body else 1.0
        # OCR box heights vary ~±25% line to line (ascenders/descenders), so
        # the body bucket is wide: only genuinely tiny print (footer fine
        # print) goes small, only headline-scale text goes large.
        bucket = 0.8 if ratio < 0.55 else (1.6 if ratio > 1.5 else 1.0)
        sizes.append(max(8, int(1.2 * bucket * body)))
    return sizes


def _padded(bbox: tuple, img: Image.Image) -> tuple[int, int, int, int]:
    """Expand the box to cover antialiased edges/ascenders that OCR boxes clip."""
    x0, y0, x1, y1 = bbox
    pad = max(2, int(0.2 * (y1 - y0)))
    W, H = img.size
    return (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))


def erase_box(img: Image.Image, bbox: tuple) -> tuple[int, int, int]:
    """Fill the (padded) box with its surrounding background color; return it."""
    box = _padded(bbox, img)
    bg = _background_color(img, box)
    ImageDraw.Draw(img).rectangle(list(box), fill=bg)
    return bg


def draw_label_in_box(img: Image.Image, bbox: tuple, replacement: str,
                      bg: tuple[int, int, int],
                      type_height: int | None = None,
                      em_size: int | None = None) -> None:
    """Draw `replacement` in the box at `em_size` (or the box height).

    `type_height` (the group's word height) decides baseline vs centered
    placement; `em_size` comes from the document-level buckets.
    """
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    ink = (20, 20, 20) if luminance > 140 else (245, 245, 245)
    height = min(type_height or (y1 - y0), y1 - y0)
    font = _fitted_font(draw, replacement, max(1, x1 - x0 - 2),
                        em_size or max(8, int(1.2 * height)))
    if (y1 - y0) <= 1.6 * height:
        # Single-line region: sit the label on the original baseline (OCR
        # boxes include descender space, so the baseline is a bit above the
        # bottom edge). Centering instead reads as superscript next to the
        # surrounding print.
        draw.text((x0 + 1, y1 - max(1, int(0.18 * height))),
                  replacement, fill=ink, font=font, anchor="ls")
    else:
        # Merged multi-line block: vertical center is the natural placement.
        top = draw.textbbox((0, 0), replacement, font=font)
        text_h = top[3] - top[1]
        draw.text((x0 + 1, y0 + max(0, ((y1 - y0) - text_h) // 2) - top[1]),
                  replacement, fill=ink, font=font)


def render_text_in_box(img: Image.Image, bbox: tuple, replacement: str) -> None:
    """Erase the box to its surrounding background color and draw `replacement`."""
    draw_label_in_box(img, bbox, replacement, erase_box(img, bbox))


def fill_visual_placeholder(img: Image.Image, bbox: tuple) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle(list(bbox), fill=(232, 232, 232), outline=(180, 180, 180))


def label_visual_placeholder(img: Image.Image, bbox: tuple, category: str) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    label = VISUAL_PLACEHOLDER_LABELS.get(category, "[REMOVED]")
    font = _fitted_font(draw, label, max(1, x1 - x0 - 4), max(8, (y1 - y0) // 4))
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text((x0 + max(0, ((x1 - x0) - tw) // 2), y0 + max(0, ((y1 - y0) - th) // 2) - tb[1]),
              label, fill=(120, 120, 120), font=font)


def render_visual_placeholder(img: Image.Image, bbox: tuple, category: str) -> None:
    """Neutral gray fill + centered category label (de-identification only)."""
    fill_visual_placeholder(img, bbox)
    label_visual_placeholder(img, bbox, category)


def _group_line_clusters(groups: list[EntityGroup]) -> list[list[int]]:
    """Cluster group indices into visual lines (top-to-bottom, left-to-right)."""
    idxs = sorted(range(len(groups)),
                  key=lambda i: ((groups[i].bbox[1] + groups[i].bbox[3]) / 2,
                                 groups[i].bbox[0]))
    lines: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    for i in idxs:
        y0, y1 = groups[i].bbox[1], groups[i].bbox[3]
        for j, (ly0, ly1) in enumerate(spans):
            overlap = min(y1, ly1) - max(y0, ly0)
            if overlap >= 0.3 * min(y1 - y0, ly1 - ly0):
                lines[j].append(i)
                spans[j] = (min(ly0, y0), max(ly1, y1))
                break
        else:
            lines.append([i])
            spans.append((y0, y1))
    for line in lines:
        line.sort(key=lambda i: groups[i].bbox[0])
    return lines


def _content_limit(img: Image.Image, y0: int, y1: int, x_from: int, x_to: int,
                   bg: tuple[int, int, int]) -> int:
    """First x in [x_from, x_to] where non-background content starts.

    Labels may extend past their erased box into empty space, but must stop
    where real (un-erased) print begins.
    """
    x_to = min(x_to, img.width)
    y1 = min(y1, img.height)
    if x_to <= x_from or y1 <= y0:
        return x_from
    strip = np.asarray(img.crop((x_from, y0, x_to, y1)), dtype=int)
    diff = np.abs(strip[:, :, :3] - np.array(bg[:3])).max(axis=2) > 30
    col_fraction = diff.mean(axis=0)
    hits = np.where(col_fraction > 0.12)[0]
    return x_from + (int(hits[0]) if len(hits) else strip.shape[1])


def _draw_labels(out: Image.Image, groups: list[EntityGroup],
                 backgrounds: list[tuple[int, int, int]], sizes: list[int],
                 substitute) -> None:
    """Draw every label at its bucket size, reflowing within each line.

    A replacement label is often wider than the word it replaces
    ("Gender_1" vs "female,"). Rather than shrinking to the original box —
    which broke size uniformity — each line lays labels out left-to-right:
    a label starts at its own box (or after the previous label), extends
    into empty space, and only shrinks when actual un-erased content is in
    the way. Iteration order stays reading order, so pseudonym numbering is
    unaffected.
    """
    draw = ImageDraw.Draw(out)
    for line in _group_line_clusters(groups):
        cursor = 0
        for i in line:
            g, bg, em = groups[i], backgrounds[i], sizes[i]
            text = substitute(g)
            x0, y0, x1, y1 = g.bbox
            luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            ink = (20, 20, 20) if luminance > 140 else (245, 245, 245)

            start_x = max(x0, cursor)
            font = _load_font(em)
            width = draw.textlength(text, font=font)
            erased_end = _padded(g.bbox, out)[2]
            scan_from = max(erased_end, start_x)
            limit = _content_limit(out, y0, y1, scan_from,
                                   int(start_x + width + 8), bg)
            avail = max(8, limit - start_x - 2)
            if width > avail:
                font = _fitted_font(draw, text, avail, em)
                width = draw.textlength(text, font=font)

            if (y1 - y0) <= 1.6 * g.type_height:
                # Single-line: sit on the original baseline (OCR boxes include
                # descender space; centering reads as superscript).
                draw.text((start_x + 1, y1 - max(1, int(0.18 * g.type_height))),
                          text, fill=ink, font=font, anchor="ls")
            else:
                tb = draw.textbbox((0, 0), text, font=font)
                text_h = tb[3] - tb[1]
                draw.text((start_x + 1, y0 + max(0, ((y1 - y0) - text_h) // 2) - tb[1]),
                          text, fill=ink, font=font)
            cursor = int(start_x + width + max(6, em // 3))


# -------------------------------------------------------------- appliers --- #
def apply_deidentify(img: Image.Image, entities: list[PIIEntity],
                     mapping: PseudonymMapping) -> Image.Image:
    # Strict fill-then-label ordering across BOTH kinds: any fill drawn after
    # a label can wipe it (a large [LOGO] placeholder once erased half of the
    # brand pseudonyms drawn next to it).
    out = img.convert("RGB").copy()
    groups = group_text_entities(entities)
    visuals = [e for e in entities if e.kind == "visual"]
    sizes = _bucket_em_sizes(groups)

    backgrounds = [erase_box(out, g.bbox) for g in groups]
    for v in visuals:
        fill_visual_placeholder(out, v.bbox)

    _draw_labels(out, groups, backgrounds, sizes,
                 lambda g: mapping.pseudonym_for(g.category, g.text))
    for v in visuals:
        label_visual_placeholder(out, v.bbox, v.category)
    return out


def apply_anonymize(img: Image.Image, entities: list[PIIEntity]) -> Image.Image:
    out = img.convert("RGB").copy()
    groups = group_text_entities(entities)
    visuals = [e for e in entities if e.kind == "visual"]
    sizes = _bucket_em_sizes(groups)

    backgrounds = [erase_box(out, g.bbox) for g in groups]
    draw = ImageDraw.Draw(out)
    for v in visuals:
        draw.rectangle(list(v.bbox), fill=(0, 0, 0))

    _draw_labels(out, groups, backgrounds, sizes,
                 lambda g: anonymize_value(g.category, g.text))
    return out
