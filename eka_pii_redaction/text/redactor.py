"""TextPIIRedactor — detect & redact PII inside plain-text blobs (the TEXT modality).

STATUS: planned / not yet implemented.

This is the text-only counterpart to `eka_pii_redaction.image.ImagePIIRedactor`.
It operates on raw strings (no image, no OCR) and returns character-span
annotations + a redacted string. The weights will live under `text/` in the same
single model repo, mirroring the `image/` layout:

    <hf_repo>/
        image/   layoutlmv3/, yolo/best.pt     # image modality (implemented)
        text/    <text-pii-model>              # text modality (this)

Intended API (parallels the image modality):

    from eka_pii_redaction.text import TextPIIRedactor

    r = TextPIIRedactor("ekacare/pii-redactors")
    spans = r.detect("John Doe, DOB 1990-01-01, lives at ...")
    #   -> list[TextPIISpan]: {category, start, end, text, l1, score}
    clean = r.redact("...", mask="[REDACTED]")   # -> str with PII replaced

The category taxonomy is shared with the image modality (see
`eka_pii_redaction.taxonomy`), minus the visual-only categories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TextPIISpan:
    """A PII span within a text blob (character offsets, not pixel boxes)."""
    category: str
    start: int
    end: int
    l1: str
    text: str
    score: Optional[float] = None


class TextPIIRedactor:
    """Planned text-only PII redactor. Not yet implemented."""

    _NOT_IMPLEMENTED = (
        "TextPIIRedactor (text modality) is not implemented yet. Use "
        "eka_pii_redaction.image.ImagePIIRedactor for document images. "
        "Track progress in the repo roadmap."
    )

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self._NOT_IMPLEMENTED)

    def detect(self, text: str) -> "list[TextPIISpan]":  # pragma: no cover
        raise NotImplementedError(self._NOT_IMPLEMENTED)

    def redact(self, text: str, mask: str = "[REDACTED]") -> str:  # pragma: no cover
        raise NotImplementedError(self._NOT_IMPLEMENTED)
