"""LayoutLMv3 detector — finds text-based PII *within document images*.

Runs Tesseract OCR (via the LayoutLMv3 image processor), classifies each OCR word
with the LayoutLMv3 token classifier, and merges BIO tags into entity spans with
pixel bounding boxes.

This is the *image* modality. (A future *text* modality will redact PII inside
plain-text blobs, with no image — see `eka_pii_redaction.text`.)
"""
from __future__ import annotations

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

from ..entities import PIIEntity
from ..taxonomy import l1_group


def _union(a: list[int], b: list[int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


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
    def detect(self, image: Image.Image, ocr_lang: Optional[str] = None
               ) -> list[PIIEntity]:
        """Return text PII entities (pixel bboxes) for one image."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        W, H = image.size

        # 1) Tesseract OCR via the image processor -> words + 0..1000 boxes.
        prev_lang = self.processor.image_processor.ocr_lang
        self.processor.image_processor.ocr_lang = ocr_lang
        try:
            feats = self.processor.image_processor(image, return_tensors=None)
        finally:
            self.processor.image_processor.ocr_lang = prev_lang
        words = feats["words"][0]
        boxes = feats["boxes"][0]
        if not words:
            return []

        # 2) Classify each word (toggle OCR off for the tokenization step).
        self.processor.image_processor.apply_ocr = False
        try:
            labels, scores = self._classify_words(image, words, boxes)
        finally:
            self.processor.image_processor.apply_ocr = True

        # 3) Merge BIO tags into entity spans (union the word boxes), then convert
        #    0..1000 boxes to pixel coordinates of the original image.
        entities: list[PIIEntity] = []
        cur = None

        def flush():
            if cur is None:
                return
            x0, y0, x1, y1 = cur["box"]
            px = (int(x0 / 1000 * W), int(y0 / 1000 * H),
                  int(x1 / 1000 * W), int(y1 / 1000 * H))
            sc = cur["scores"]
            entities.append(PIIEntity(
                category=cur["category"], kind="text", bbox=px,
                l1=l1_group(cur["category"]), text=" ".join(cur["words"]),
                score=round(sum(sc) / len(sc), 4) if sc and all(s is not None for s in sc) else None,
            ))

        for w, box, lab, sc in zip(words, boxes, labels, scores):
            if lab == "O":
                flush(); cur = None
                continue
            tag, cat = lab[0], lab[2:]
            if tag == "B" or cur is None or cur["category"] != cat:
                flush()
                cur = {"category": cat, "words": [w], "box": list(box), "scores": [sc]}
            else:  # I- continuing same category
                cur["words"].append(w)
                cur["box"] = _union(cur["box"], list(box))
                cur["scores"].append(sc)
        flush()
        return entities
