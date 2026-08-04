"""Eka-PII-redaction — detect and redact PII in documents, organized by modality.

- `eka_pii_redaction.image.ImagePIIRedactor` — PII in document images.
- `eka_pii_redaction.text.TextPIIRedactor`   — PII in plain-text strings.

`ImagePIIRedactor` and `TextPIIRedactor` are also re-exported at the top level.
"""
from .entities import PIIEntity
from .image import ImagePIIRedactor
from .pseudonym import PseudonymMapping
from .taxonomy import ALL_ENTITIES, TEXT_ENTITIES, TEXT_REDACTABLE, VISUAL_ENTITIES
from .text import TextPIIRedactor, TextPIISpan

__version__ = "0.2.3"
__all__ = [
    "ImagePIIRedactor",
    "TextPIIRedactor",
    "TextPIISpan",
    "PIIEntity",
    "PseudonymMapping",
    "ALL_ENTITIES",
    "TEXT_ENTITIES",
    "TEXT_REDACTABLE",
    "VISUAL_ENTITIES",
]
