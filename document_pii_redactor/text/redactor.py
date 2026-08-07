"""TextPIIRedactor — detect & redact PII inside plain-text strings (TEXT modality).

The text counterpart to `document_pii_redactor.image.ImagePIIRedactor`. Operates on
raw strings (no image, no OCR) via a multilingual MiniLM token classifier and
returns character-span annotations and/or a redacted string. The weights live
under `text/minilm/` in the same single HF repo as the image models:

    <hf_repo>/
        image/   layoutlmv3/, yolo/best.pt     # image modality
        text/    minilm/                        # text modality (this)

    from document_pii_redactor.text import TextPIIRedactor

    r = TextPIIRedactor("ekacare/document-pii-redactor")
    spans = r.detect("John Doe, DOB 1990-01-01, lives at ...")
    #   -> list[TextPIISpan]: {category, start, end, l1, text, score}
    clean = r.redact("...", mask="[REDACTED]")   # -> str with PII replaced

The category taxonomy is shared with the image modality (see
`document_pii_redactor.taxonomy`), minus the visual-only categories.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..taxonomy import TEXT_REDACTABLE, validate_entities

DEFAULT_HF_REPO = "ekacare/document-pii-redactor"
# Text model location within the (single) model repo, organized by modality.
MINILM_SUBDIR = "text/minilm"


@dataclass
class TextPIISpan:
    """A PII span within a text blob (character offsets, not pixel boxes)."""
    category: str
    start: int
    end: int
    l1: str
    text: str
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "category": self.category, "start": self.start, "end": self.end,
            "l1": self.l1, "text": self.text, "score": self.score,
        }


def apply_mask(text: str, spans: "Iterable[TextPIISpan]", mask: str = "[REDACTED]") -> str:
    """Replace each span's characters in `text` with `mask`.

    `mask` may contain `{category}` and/or `{l1}` placeholders, filled per span;
    a literal mask (no `{`) is inserted verbatim. Spans are applied left-to-right
    on character offsets; any span overlapping an already-consumed region is
    skipped so offsets stay valid.
    """
    out: list[str] = []
    prev = 0
    for sp in sorted(spans, key=lambda s: s.start):
        if sp.start < prev:
            continue
        out.append(text[prev:sp.start])
        out.append(mask.format(category=sp.category, l1=sp.l1) if "{" in mask else mask)
        prev = sp.end
    out.append(text[prev:])
    return "".join(out)


def _resolve_text_source(hf_repo: str, cache_dir: Optional[str]) -> str:
    """Return the local dir holding the text model.

    If `hf_repo` is an existing local directory it is used as-is (offline); else
    the `text/minilm` files are pulled from the Hugging Face Hub."""
    local = Path(hf_repo)
    if local.is_dir():
        d = local / MINILM_SUBDIR
        return str(d if d.is_dir() else local)
    from huggingface_hub import snapshot_download

    snap = snapshot_download(
        repo_id=hf_repo, allow_patterns=f"{MINILM_SUBDIR}/*", cache_dir=cache_dir
    )
    return os.path.join(snap, *MINILM_SUBDIR.split("/"))


class TextPIIRedactor:
    """Detect and redact PII inside plain-text strings."""

    AVAILABLE_ENTITIES = TEXT_REDACTABLE

    def __init__(
        self,
        hf_repo: str = DEFAULT_HF_REPO,
        *,
        device: Optional[str] = None,
        categories: Optional[Iterable[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            hf_repo: Hugging Face repo id (or a local dir) holding the weights.
            device: "cuda" / "cpu". None -> auto (cuda if available, else cpu).
            categories: the categories to detect. Default None = all of them.
        """
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._categories = (set(validate_entities(categories))
                            if categories is not None else None)

        # Imported here so a missing torch/transformers only bites at construction,
        # and the pure helpers above stay importable for tests.
        from .minilm import MiniLMDetector

        model_dir = _resolve_text_source(hf_repo, cache_dir)
        self.minilm = MiniLMDetector(model_dir, self.device)

    @classmethod
    def list_entities(cls) -> list[str]:
        """Return all categories the text modality can detect/redact."""
        return list(TEXT_REDACTABLE)

    def _active_categories(self, per_call: Optional[Iterable[str]]):
        if per_call is not None:
            return set(validate_entities(per_call))
        return self._categories  # None -> all categories

    def detect(
        self, text: str, *, categories: Optional[Iterable[str]] = None
    ) -> "list[TextPIISpan]":
        """Detect PII spans in `text`. Each TextPIISpan has `category`, `start`,
        `end` (char offsets), `l1`, `text`, `score`.

        `categories` selects which categories to detect (default: all);
        given per-call it overrides the constructor's selection.
        """
        wanted = self._active_categories(categories)
        spans = self.minilm.detect(text)
        if wanted is None:
            return spans
        return [s for s in spans if s.category in wanted]

    def redact(
        self,
        text: str,
        spans: Iterable[TextPIISpan],
        *,
        mask: str = "[REDACTED]",
    ) -> str:
        """Return `text` with every span in `spans` replaced by `mask`.

        `spans` is a `detect()` result — detection is always the explicit
        first step; every transform consumes its output. `mask` may contain
        `{category}` / `{l1}` placeholders (see `apply_mask`).
        """
        return apply_mask(text, spans, mask=mask)

    def deidentify(
        self,
        text: str,
        spans: Iterable[TextPIISpan],
        *,
        mapping=None,
        strategy: str = "counter",
        secret: Optional[str] = None,
    ):
        """De-identify: replace each entity with a consistent pseudonym.

        `spans` is a `detect()` result. Two pseudonym strategies:
        - "counter" (default): sequential `Person_1` pseudonyms, scoped to
          the mapping — pass a prior result's `mapping` to keep numbering
          stable across pages/documents of one record.
        - "hash": globally deterministic `Person_a3f9c1` tokens (md5 of the
          normalized value; salt with `secret` to resist dictionary
          reversal of guessable values) — same value, same token, across
          all documents, no mapping threading needed.

        Either way the entity->pseudonym mapping is returned (never
        persisted here) so an authorized caller can re-link later.
        Returns `TextDeidResult(text, mapping)`.
        """
        from ..pseudonym import PseudonymMapping
        from .transforms import TextDeidResult, apply_pseudonyms

        if mapping is None:
            mapping = PseudonymMapping()
        return TextDeidResult(
            text=apply_pseudonyms(text, spans, mapping,
                                  strategy=strategy, secret=secret),
            mapping=mapping)

    def anonymize(self, text: str, spans: Iterable[TextPIISpan]) -> str:
        """Anonymize: one-way replacement with quasi-identifier generalization.

        `spans` is a `detect()` result. Ages become 10-year buckets, dates
        keep only the year, fine geography collapses to [LOCATION], and
        direct identifiers become unnumbered category tokens. No mapping
        exists anywhere — by design this is not reversible, unlike
        `deidentify()`.
        """
        from .transforms import apply_anonymization

        return apply_anonymization(text, spans)
