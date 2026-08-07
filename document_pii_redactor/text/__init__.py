"""Text modality: detect & redact PII inside plain-text strings."""
from .redactor import DEFAULT_HF_REPO, TextPIIRedactor, TextPIISpan

__all__ = ["TextPIIRedactor", "TextPIISpan", "DEFAULT_HF_REPO"]
