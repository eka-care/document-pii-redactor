"""The PIIEntity record returned by detection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class PIIEntity:
    """A single detected PII item.

    Attributes:
        category:  fine-grained category name (e.g. "primary_subject_name",
                   "signature"). One of taxonomy.ALL_ENTITIES.
        kind:      "text" (from OCR + token classifier) or "visual" (from the
                   image detector).
        bbox:      pixel coordinates [x0, y0, x1, y1] in the ORIGINAL image.
        l1:        coarse L1 group ("person", "location", ..., or
                   "biometric_visual").
        text:      the OCR text for text entities; None for visual entities.
        score:     detector confidence (visual: YOLO confidence in [0,1]; text:
                   mean per-token softmax probability, or None).
    """
    category: str
    kind: str               # "text" | "visual"
    bbox: tuple[int, int, int, int]
    l1: str
    text: Optional[str] = None
    score: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d
