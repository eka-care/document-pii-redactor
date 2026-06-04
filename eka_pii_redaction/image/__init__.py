"""Image modality: detect & redact PII in document images."""
from .redactor import DEFAULT_HF_REPO, ImagePIIRedactor

__all__ = ["ImagePIIRedactor", "DEFAULT_HF_REPO"]
