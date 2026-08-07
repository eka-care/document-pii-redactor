"""ImagePIIRedactor — detect & redact PII in document IMAGES (the image modality).

    from eka_pii_redaction import ImagePIIRedactor   # or: from eka_pii_redaction.image import ...

    redactor = ImagePIIRedactor(detect_visual=True)          # GPU if available
    entities = redactor.detect("page.jpg")                   # list[PIIEntity]
    redacted = redactor.redact("page.jpg")                   # PIL.Image

One Hugging Face repo holds the weights, organized BY MODALITY:

    <hf_repo>/
        image/
            layoutlmv3/      # text-PII-in-image token classifier (config, safetensors, processor)
            yolo/best.pt     # visual-entity detector (YOLO11m)
        text/                # (reserved) future text-only PII model

`detect_visual` controls whether the YOLO weights are downloaded + loaded at all.
"""
from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from PIL import Image, ImageDraw, ImageFilter

from ..entities import PIIEntity
from ..taxonomy import ALL_ENTITIES, TEXT_ENTITIES, VISUAL_ENTITIES, validate_entities
from .layoutlmv3 import LayoutLMv3Detector

DEFAULT_HF_REPO = "ekacare/pii-redactors"
# Weight locations within the (single) model repo, organized by modality.
LAYOUTLMV3_SUBDIR = "image/layoutlmv3"
YOLO_WEIGHTS = "image/yolo/best.pt"

ImageInput = Union[str, Path, bytes, Image.Image]


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(str(image)).convert("RGB")


def _resolve_sources(hf_repo: str, detect_visual: bool, cache_dir: Optional[str]):
    """Return (layoutlmv3_dir, yolo_weights_path|None).

    If `hf_repo` is an existing local directory it is used as-is (offline); else
    the needed files are pulled from the Hugging Face Hub (one repo)."""
    local = Path(hf_repo)
    if local.is_dir():
        lm_dir = local / LAYOUTLMV3_SUBDIR
        if not lm_dir.is_dir():
            lm_dir = local  # allow pointing straight at a layoutlmv3 dir
        yolo = local / YOLO_WEIGHTS
        return str(lm_dir), (str(yolo) if detect_visual and yolo.exists() else None)

    from huggingface_hub import hf_hub_download, snapshot_download

    snap = snapshot_download(
        repo_id=hf_repo, allow_patterns=f"{LAYOUTLMV3_SUBDIR}/*", cache_dir=cache_dir
    )
    lm_dir = os.path.join(snap, *LAYOUTLMV3_SUBDIR.split("/"))
    yolo_path = None
    if detect_visual:
        yolo_path = hf_hub_download(
            repo_id=hf_repo, filename=YOLO_WEIGHTS, cache_dir=cache_dir
        )
    return lm_dir, yolo_path


class ImagePIIRedactor:
    """Detect and redact PII (text + visual) in document images."""

    AVAILABLE_ENTITIES = ALL_ENTITIES
    TEXT_ENTITIES = TEXT_ENTITIES
    VISUAL_ENTITIES = VISUAL_ENTITIES

    def __init__(
        self,
        hf_repo: str = DEFAULT_HF_REPO,
        *,
        detect_visual: bool = True,
        device: Optional[str] = None,
        categories: Optional[Iterable[str]] = None,
        visual_score_threshold: float = 0.25,
        ocr_lang: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            hf_repo: Hugging Face repo id (or a local dir) holding the weights.
                One id for the user; multiple weights downloaded inside.
            detect_visual: if False, the YOLO weights are neither downloaded nor
                loaded — only text-in-image PII is detected.
            device: "cuda" / "cpu". None -> auto (cuda if available, else cpu).
            categories: the categories to detect. Default None = all of them.
            visual_score_threshold: YOLO confidence cutoff for visual entities.
            ocr_lang: Tesseract language/script spec (e.g. "eng", "eng+Devanagari").
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detect_visual = detect_visual
        self.ocr_lang = ocr_lang
        self.visual_score_threshold = visual_score_threshold
        self._categories = (set(validate_entities(categories))
                            if categories is not None else None)

        lm_dir, yolo_path = _resolve_sources(hf_repo, detect_visual, cache_dir)
        self.layoutlmv3 = LayoutLMv3Detector(lm_dir, self.device)

        self.yolo = None
        if detect_visual:
            if not yolo_path:
                raise FileNotFoundError(
                    f"detect_visual=True but no visual weights found at "
                    f"'{YOLO_WEIGHTS}' in {hf_repo}."
                )
            from .yolo11m import YOLODetector
            self.yolo = YOLODetector(
                yolo_path, self.device, score_threshold=visual_score_threshold
            )

    # ------------------------------------------------------------------ #
    @classmethod
    def list_entities(cls) -> dict:
        """Return all selectable categories, grouped by kind."""
        return {"text": list(TEXT_ENTITIES), "visual": list(VISUAL_ENTITIES)}

    def _active_categories(self, per_call: Optional[Iterable[str]]):
        if per_call is not None:
            return set(validate_entities(per_call))
        return self._categories  # None -> all categories

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_ocr_input(words, boxes) -> None:
        if (words is None) != (boxes is None):
            raise ValueError(
                "Bring-your-own OCR needs BOTH words and boxes (or neither).")
        if words is None:
            return
        if len(words) != len(boxes):
            raise ValueError(
                f"words and boxes must be the same length "
                f"(got {len(words)} words, {len(boxes)} boxes).")
        for i, b in enumerate(boxes):
            if len(b) != 4:
                raise ValueError(
                    f"boxes[{i}] must be [x0, y0, x1, y1], got {b!r}.")
            x0, y0, x1, y1 = b
            if x1 < x0 or y1 < y0:
                raise ValueError(
                    f"boxes[{i}] is inverted ({b!r}); expected x0<=x1, y0<=y1.")

    def detect(
        self,
        image: ImageInput,
        *,
        categories: Optional[Iterable[str]] = None,
        ocr_lang: Optional[str] = None,
        words: Optional[list[str]] = None,
        boxes: Optional[list] = None,
    ) -> list[PIIEntity]:
        """Detect PII entities (text + visual). Each PIIEntity has `category`,
        `bbox` (pixels), `text`, `l1`, `score`, `kind`.

        `categories` selects which categories to detect (default: all);
        given per-call it overrides the constructor's selection.

        Bring your own OCR by passing `words` and `boxes` together — one
        `[x0, y0, x1, y1]` box per word, in ORIGINAL-IMAGE pixel
        coordinates. The built-in Tesseract step is skipped (`ocr_lang` is
        ignored) and your exact boxes come back on the emitted text
        entities. The visual detector is unaffected.
        """
        self._validate_ocr_input(words, boxes)
        img = _load_image(image)
        wanted = self._active_categories(categories)
        lang = ocr_lang if ocr_lang is not None else self.ocr_lang

        if self.yolo is not None:
            # Run the text (OCR + LayoutLMv3) and visual (YOLO) detectors
            # concurrently rather than sequentially. Both release the GIL during
            # heavy compute, and the text path's Tesseract OCR is CPU-bound, so it
            # overlaps the YOLO inference instead of waiting for it.
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_text = ex.submit(self.layoutlmv3.detect, img, lang,
                                   words, boxes)
                f_visual = ex.submit(self.yolo.detect, img)
                results = f_text.result() + f_visual.result()
        else:
            results = self.layoutlmv3.detect(img, ocr_lang=lang,
                                             words=words, boxes=boxes)

        if wanted is None:
            return results
        return [e for e in results if e.category in wanted]

    # ------------------------------------------------------------------ #
    def redact(
        self,
        image: ImageInput,
        entities: list[PIIEntity],
        *,
        mode: str = "solid",
        color: tuple[int, int, int] = (0, 0, 0),
        pad: int = 2,
    ) -> Image.Image:
        """Return a copy of the image with the given PII regions redacted.

        Args:
            entities: a `detect()` result — detection is always the explicit
                first step; every transform consumes its output.
            mode: "solid" (fill with `color`), "blur", or "pixelate".
            color: fill color for mode="solid".
            pad: pixels to expand each box by (covers anti-aliasing at edges).
        """
        img = _load_image(image).copy()
        W, H = img.size

        for e in entities:
            x0, y0, x1, y1 = e.bbox
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            if mode == "solid":
                ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], fill=color)
            elif mode == "blur":
                region = img.crop((x0, y0, x1, y1))
                region = region.filter(ImageFilter.GaussianBlur(radius=max(4, (x1 - x0) // 8)))
                img.paste(region, (x0, y0))
            elif mode == "pixelate":
                region = img.crop((x0, y0, x1, y1))
                small = region.resize((max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)))
                img.paste(small.resize(region.size, Image.NEAREST), (x0, y0))
            else:
                raise ValueError(f"unknown redaction mode: {mode!r}")
        return img

    # ------------------------------------------------------------------ #
    def deidentify(
        self,
        image: ImageInput,
        entities: list[PIIEntity],
        *,
        mapping=None,
    ):
        """De-identify: consistent pseudonyms rendered in place of text PII.

        `entities` is a `detect()` result. OCR'd words are merged into
        logical entities ("John Doe" -> one "Person_1"), erased to the
        surrounding background color, and the pseudonym is drawn into the
        box. Visual-only entities (faces, signatures, ...) become neutral
        labeled placeholders. Returns `ImageDeidResult(image, mapping)`;
        pass a prior `mapping` to keep numbering stable across pages of one
        record.
        """
        from ..pseudonym import PseudonymMapping
        from .transforms import ImageDeidResult, apply_deidentify

        if mapping is None:
            mapping = PseudonymMapping()
        img = _load_image(image)
        return ImageDeidResult(image=apply_deidentify(img, entities, mapping),
                               mapping=mapping)

    def anonymize(self, image: ImageInput,
                  entities: list[PIIEntity]) -> Image.Image:
        """Anonymize: one-way in-place replacement, no mapping anywhere.

        `entities` is a `detect()` result. Text PII is rendered as
        generalized values/unnumbered tokens (same rules as
        `TextPIIRedactor.anonymize`); visual-only entities are filled solid
        black — biometrics have nothing to generalize, so destruction is
        the anonymization.
        """
        from .transforms import apply_anonymize

        return apply_anonymize(_load_image(image), entities)
