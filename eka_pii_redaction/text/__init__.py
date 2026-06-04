"""Text modality (planned): detect & redact PII inside plain-text blobs."""
from .redactor import TextPIIRedactor, TextPIISpan

__all__ = ["TextPIIRedactor", "TextPIISpan"]
