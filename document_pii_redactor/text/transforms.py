"""Pure span-transform functions for text de-identification and anonymization.

Model-free on purpose: `TextPIIRedactor.deidentify()`/`.anonymize()` run
`detect()` and then apply these, and the unit tests exercise them directly
with synthetic spans. Span-walk mechanics mirror `redactor.apply_mask`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..pseudonym import PseudonymMapping, label_for
from .redactor import TextPIISpan

# Anonymization rule groups (see taxonomy.py for the full category list).
_GEO_KEEP = {"country", "state_province"}          # already coarse — kept verbatim
_GEO_COLLAPSE = {"city_district", "street_address",
                 "postal_zip_pin_code", "geocode_coordinates"}
_DATE_CATEGORIES = {"date_of_birth", "death_date", "other_date_time"}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_INT_RE = re.compile(r"\d+")


@dataclass
class TextDeidResult:
    """De-identified text plus the mapping the caller may store securely."""
    text: str
    mapping: PseudonymMapping


def anonymize_value(category: str, value: str) -> str:
    """One-way replacement for a single span — no mapping, no consistency.

    Quasi-identifiers are generalized (age buckets, year-only dates, coarse
    geography); direct identifiers collapse to an unnumbered category token.
    """
    if category in _GEO_KEEP:
        return value
    if category == "age":
        m = _INT_RE.search(value)
        if m:
            lo = (int(m.group()) // 10) * 10
            return f"{lo}–{lo + 9}"
        return "[AGE]"
    if category in _DATE_CATEGORIES:
        m = _YEAR_RE.search(value)
        return m.group() if m else "[DATE]"
    if category in _GEO_COLLAPSE:
        return "[LOCATION]"
    return f"[{label_for(category).upper()}]"


def merge_adjacent_spans(text: str, spans: Iterable[TextPIISpan]) -> list[TextPIISpan]:
    """Merge same-category spans separated by whitespace only.

    BIO decoding sometimes restarts mid-entity, splitting "Mr. John Doe" into
    two spans — without merging, de-identification numbers them separately
    ("Person_1 Person_2"). Whitespace-only separators are safe to bridge; a
    comma is not ("John, Asha" is two people, not one).
    """
    merged: list[TextPIISpan] = []
    for sp in sorted(spans, key=lambda s: s.start):
        prev = merged[-1] if merged else None
        if (prev is not None and sp.category == prev.category
                and sp.start >= prev.end and not text[prev.end:sp.start].strip()):
            scores = [s for s in (prev.score, sp.score) if s is not None]
            merged[-1] = TextPIISpan(
                category=prev.category, start=prev.start, end=sp.end,
                l1=prev.l1, text=text[prev.start:sp.end],
                score=round(min(scores), 4) if len(scores) == 2 else None,
            )
        else:
            merged.append(sp)
    return merged


def _replace_spans(text: str, spans: Iterable[TextPIISpan], substitute) -> str:
    """Rebuild `text` with each merged span replaced by substitute(span).

    Left-to-right on character offsets; spans overlapping an already-consumed
    region are skipped so offsets stay valid (same rule as apply_mask).
    """
    out: list[str] = []
    prev = 0
    for sp in merge_adjacent_spans(text, spans):
        if sp.start < prev:
            continue
        out.append(text[prev:sp.start])
        out.append(substitute(sp))
        prev = sp.end
    out.append(text[prev:])
    return "".join(out)


def apply_pseudonyms(text: str, spans: Iterable[TextPIISpan],
                     mapping: PseudonymMapping, *, strategy: str = "counter",
                     secret: str | None = None) -> str:
    """De-identify: consistent pseudonyms, recorded in `mapping`.

    strategy "counter": sequential per-label pseudonyms (Person_1) — scoped
    to the mapping, which threads numbering across pages.
    strategy "hash": globally deterministic tokens (Person_a3f9c1, md5 of
    the normalized value, optionally salted with `secret`) — same value
    gets the same token across documents with no mapping to thread.
    """
    if strategy == "counter":
        substitute = (lambda sp:
                      mapping.pseudonym_for(sp.category, text[sp.start:sp.end]))
    elif strategy == "hash":
        from ..pseudonym import token_for

        def substitute(sp):
            value = text[sp.start:sp.end]
            return mapping.record_token(
                sp.category, value, token_for(sp.category, value, secret))
    else:
        raise ValueError(
            f"strategy must be 'counter' or 'hash', got {strategy!r}")
    return _replace_spans(text, spans, substitute)


def apply_anonymization(text: str, spans: Iterable[TextPIISpan]) -> str:
    """Anonymize: generalize/collapse every span; nothing is recorded."""
    return _replace_spans(
        text, spans,
        lambda sp: anonymize_value(sp.category, text[sp.start:sp.end]))
