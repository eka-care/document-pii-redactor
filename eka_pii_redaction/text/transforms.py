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


def _replace_spans(text: str, spans: Iterable[TextPIISpan], substitute) -> str:
    """Rebuild `text` with each span replaced by substitute(span).

    Left-to-right on character offsets; spans overlapping an already-consumed
    region are skipped so offsets stay valid (same rule as apply_mask).
    """
    out: list[str] = []
    prev = 0
    for sp in sorted(spans, key=lambda s: s.start):
        if sp.start < prev:
            continue
        out.append(text[prev:sp.start])
        out.append(substitute(sp))
        prev = sp.end
    out.append(text[prev:])
    return "".join(out)


def apply_pseudonyms(text: str, spans: Iterable[TextPIISpan],
                     mapping: PseudonymMapping) -> str:
    """De-identify: consistent numbered pseudonyms, recorded in `mapping`."""
    return _replace_spans(
        text, spans,
        lambda sp: mapping.pseudonym_for(sp.category, text[sp.start:sp.end]))


def apply_anonymization(text: str, spans: Iterable[TextPIISpan]) -> str:
    """Anonymize: generalize/collapse every span; nothing is recorded."""
    return _replace_spans(
        text, spans,
        lambda sp: anonymize_value(sp.category, text[sp.start:sp.end]))
