"""Eka-PII-redaction — detect and redact PII in documents, organized by modality.

- `eka_pii_redaction.image.ImagePIIRedactor` — PII in document images (implemented).
- `eka_pii_redaction.text.TextPIIRedactor`  — PII in plain-text blobs (planned).

`ImagePIIRedactor` is also re-exported at the top level for convenience.
"""
from .entities import PIIEntity
from .image import ImagePIIRedactor
from .taxonomy import ALL_ENTITIES, TEXT_ENTITIES, VISUAL_ENTITIES

__version__ = "0.1.0"
__all__ = [
    "ImagePIIRedactor",
    "PIIEntity",
    "ALL_ENTITIES",
    "TEXT_ENTITIES",
    "VISUAL_ENTITIES",
]
