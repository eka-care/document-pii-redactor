"""MiniLM detector — finds PII inside plain-text strings (the TEXT modality).

Runs a multilingual MiniLM token classifier (BertForTokenClassification head over
an XLMRobertaTokenizerFast) on raw text and maps the per-token BIO predictions back
to character spans using the fast tokenizer's offset mapping. No OCR, no image.

This is the text counterpart to image/layoutlmv3.py.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ..taxonomy import l1_group
from .redactor import TextPIISpan


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) to exclude leading/trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _flatten_chunks(chunk_offsets, chunk_labels, chunk_scores):
    """Flatten per-chunk (offsets, labels, scores) into a single de-duplicated,
    char-start-sorted sequence.

    Long inputs are tokenized into overlapping (strided) chunks, so a token can
    appear in two chunks. We key tokens by their (char_start, char_end) and keep
    the FIRST occurrence (earlier chunk wins), dropping zero-width tokens (special
    + padding tokens, whose offsets are (0, 0))."""
    seen: dict[tuple[int, int], tuple[str, Optional[float]]] = {}
    for offs, labs, scs in zip(chunk_offsets, chunk_labels, chunk_scores):
        for (cs, ce), lab, sc in zip(offs, labs, scs):
            if ce <= cs:
                continue
            key = (cs, ce)
            if key not in seen:
                seen[key] = (lab, sc)
    items = sorted(seen.items(), key=lambda kv: kv[0])
    offsets = [k for k, _ in items]
    labels = [v[0] for _, v in items]
    scores = [v[1] for _, v in items]
    return offsets, labels, scores


def merge_bio_spans(text, offsets, labels, scores) -> "list[TextPIISpan]":
    """Merge a flat token sequence of (offset, BIO label, score) into entity spans.

    Labels are uppercase BIO (e.g. "B-EMAIL"); the category is `label[2:].lower()`,
    matching the taxonomy keys. A "B-" tag, a category change, or a non-entity
    token ends the current span. Each emitted span's char range is whitespace-
    trimmed and its score is the mean of its tokens' max-probabilities.
    """
    spans: list[TextPIISpan] = []
    cur = None  # {category, start, end, scores}

    def flush():
        nonlocal cur
        if cur is None:
            return
        s, e = _trim(text, cur["start"], cur["end"])
        if e > s:
            sc = [x for x in cur["scores"] if x is not None]
            spans.append(TextPIISpan(
                category=cur["category"], start=s, end=e,
                l1=l1_group(cur["category"]), text=text[s:e],
                score=round(sum(sc) / len(sc), 4) if sc else None,
            ))
        cur = None

    for (cs, ce), lab, sc in zip(offsets, labels, scores):
        if ce <= cs:
            continue
        if lab == "O":
            flush()
            continue
        tag, cat = lab[0], lab[2:].lower()
        if tag == "B" or cur is None or cur["category"] != cat:
            flush()
            cur = {"category": cat, "start": cs, "end": ce, "scores": [sc]}
        else:  # I- continuing the same category
            cur["end"] = max(cur["end"], ce)
            cur["scores"].append(sc)
    flush()
    return spans


class MiniLMDetector:
    """Detects PII in a plain-text string via a MiniLM token classifier."""

    def __init__(self, model_dir: str, device: str,
                 max_length: int = 512, stride: int = 128):
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "TextPIIRedactor needs a fast tokenizer (offset mapping); "
                f"{model_dir} loaded a slow one."
            )
        self.model = (
            AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
        )
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def detect(self, text: str) -> "list[TextPIISpan]":
        """Return character-span PII entities for one text string."""
        if not text or not text.strip():
            return []

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            stride=self.stride,
            return_tensors="pt",
        )
        offset_mapping = enc.pop("offset_mapping").tolist()   # (n_chunks, T, 2)
        enc.pop("overflow_to_sample_mapping", None)
        on_device = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**on_device).logits               # (n_chunks, T, C)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1).cpu().tolist()
        pred_scores = probs.max(-1).values.cpu().tolist()

        chunk_offsets = [[tuple(o) for o in chunk] for chunk in offset_mapping]
        chunk_labels = [[self.id2label[i] for i in ids] for ids in pred_ids]

        offsets, labels, scores = _flatten_chunks(
            chunk_offsets, chunk_labels, pred_scores)
        return merge_bio_spans(text, offsets, labels, scores)
