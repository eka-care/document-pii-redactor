"""LayoutLMv3 detector — finds text-based PII *within document images*.

Runs Tesseract OCR (via the LayoutLMv3 image processor), classifies each OCR word
with the LayoutLMv3 token classifier, and emits one entity per labeled word
(BIO prefix stripped to a normalised category) with a pixel bounding box.

This is the *image* modality. (A future *text* modality will redact PII inside
plain-text blobs, with no image — see `document_pii_redactor.text`.)
"""
from __future__ import annotations

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

from ..entities import PIIEntity
from ..taxonomy import l1_group


def normalize_boxes_to_1000(boxes, width: int, height: int) -> list[list[int]]:
    """Convert pixel-coordinate word boxes to LayoutLMv3's 0..1000 space.

    Values are rounded and clamped, so boxes that touch (or slightly
    overshoot) the image edge stay valid.
    """
    out = []
    for x0, y0, x1, y1 in boxes:
        out.append([
            max(0, min(1000, round(x0 / width * 1000))),
            max(0, min(1000, round(y0 / height * 1000))),
            max(0, min(1000, round(x1 / width * 1000))),
            max(0, min(1000, round(y1 / height * 1000))),
        ])
    return out


class LayoutLMv3Detector:
    """Detects text PII in an image via OCR + LayoutLMv3 token classification."""

    def __init__(self, model_dir: str, device: str):
        self.device = device
        self.model = (
            AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
        )
        # apply_ocr=True -> the image processor runs Tesseract to get words + boxes.
        self.processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=True)
        self.id2label = self.model.config.id2label

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _classify_words(self, image: Image.Image, words: list[str],
                        boxes: list[list[int]], max_length: int = 512,
                        stride: int = 128) -> tuple[list[str], list[Optional[float]]]:
        """Tokenize words+boxes (no OCR), run the model with long-doc chunking,
        and map subword predictions back to per-word (label, score)."""
        encoding = self.processor(
            image, words, boxes=boxes,
            truncation=True, padding="max_length", max_length=max_length,
            stride=stride, return_overflowing_tokens=True, return_tensors="pt",
        )
        encoding.pop("overflow_to_sample_mapping", None)
        n_chunks = encoding["input_ids"].shape[0]
        if isinstance(encoding["pixel_values"], list):
            encoding["pixel_values"] = torch.stack(encoding["pixel_values"], dim=0)
        on_device = {k: v.to(self.device) for k, v in encoding.items()}

        logits = self.model(**on_device).logits          # (n_chunks, T, C)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1).cpu().tolist()
        pred_prob = probs.max(-1).values.cpu().tolist()

        n_words = len(words)
        word_label_id: list[Optional[int]] = [None] * n_words
        word_score: list[Optional[float]] = [None] * n_words
        for ci in range(n_chunks):
            wids = encoding.word_ids(batch_index=ci)
            for ti, wid in enumerate(wids):
                if wid is None or word_label_id[wid] is not None:
                    continue
                word_label_id[wid] = pred_ids[ci][ti]
                word_score[wid] = float(pred_prob[ci][ti])

        labels = [self.id2label[i] if i is not None else "O" for i in word_label_id]
        return labels, word_score

    # ------------------------------------------------------------------ #
    def detect(self, image: Image.Image, ocr_lang: Optional[str] = None,
               words: Optional[list[str]] = None,
               boxes: Optional[list] = None) -> list[PIIEntity]:
        """Return text PII entities (pixel bboxes) for one image.

        Pass `words` + `boxes` (word-level boxes in ORIGINAL-IMAGE pixel
        coordinates) to bring your own OCR: Tesseract is skipped, the words
        go straight to the classifier, and your exact pixel boxes come back
        on the emitted entities. Without them, the built-in Tesseract path
        runs (`ocr_lang` applies only there).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        W, H = image.size

        if words is not None:
            # Bring-your-own OCR: normalize the pixel boxes to the 0..1000
            # space LayoutLMv3 expects; keep the originals for the output.
            pixel_boxes = [tuple(int(v) for v in b) for b in boxes]
            norm_boxes = normalize_boxes_to_1000(boxes, W, H)
        else:
            # Built-in path: Tesseract via the image processor -> words +
            # 0..1000 boxes, converted to pixels for the output.
            prev_lang = self.processor.image_processor.ocr_lang
            self.processor.image_processor.ocr_lang = ocr_lang
            try:
                feats = self.processor.image_processor(image, return_tensors=None)
            finally:
                self.processor.image_processor.ocr_lang = prev_lang
            words = feats["words"][0]
            norm_boxes = feats["boxes"][0]
            pixel_boxes = [
                (int(x0 / 1000 * W), int(y0 / 1000 * H),
                 int(x1 / 1000 * W), int(y1 / 1000 * H))
                for x0, y0, x1, y1 in norm_boxes
            ]
        if not words:
            return []

        # Classify each word (toggle OCR off for the tokenization step).
        self.processor.image_processor.apply_ocr = False
        try:
            labels, scores = self._classify_words(image, words, norm_boxes)
        finally:
            self.processor.image_processor.apply_ocr = True

        # Emit one entity per labeled word (NO grouping/merging). Just strip
        # the BIO prefix (B-/I-) so the category is normalised.
        entities: list[PIIEntity] = []
        for w, px, lab, sc in zip(words, pixel_boxes, labels, scores):
            if lab == "O":
                continue
            cat = lab[2:] if lab[:2] in ("B-", "I-") else lab
            entities.append(PIIEntity(
                category=cat, kind="text", bbox=px,
                l1=l1_group(cat), text=w,
                score=round(sc, 4) if sc is not None else None,
            ))
        return entities
