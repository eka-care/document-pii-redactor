"""document-pii-redactor — detect and redact PII in documents, organized by modality.

- `document_pii_redactor.image.ImagePIIRedactor` — PII in document images.
- `document_pii_redactor.text.TextPIIRedactor`   — PII in plain-text strings.

`ImagePIIRedactor` and `TextPIIRedactor` are also re-exported at the top level.
"""
from .entities import PIIEntity
from .image import ImagePIIRedactor
from .pseudonym import PseudonymMapping
from .taxonomy import ALL_ENTITIES, TEXT_ENTITIES, TEXT_REDACTABLE, VISUAL_ENTITIES
from .text import TextPIIRedactor, TextPIISpan

__version__ = "0.9.0"
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
