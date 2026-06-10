# Text PII Modality + Streamlit Tabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a plain-text PII modality (`TextPIIRedactor`) backed by the `text/minilm` model in `ekacare/pii-redactors`, wired into the package exports, FastAPI server, README, and the Streamlit tester (as a Text tab alongside an Image tab).

**Architecture:** Mirror the existing image modality 1:1 — a thin model wrapper (`MiniLMDetector`, like `LayoutLMv3Detector`) plus an orchestrator (`TextPIIRedactor`, like `ImagePIIRedactor`). The model is a `BertForTokenClassification` BIO tagger over an `XLMRobertaTokenizerFast`; the fast tokenizer's `offset_mapping` gives character spans directly, so detection emits the existing `TextPIISpan` dataclass. All span-assembly logic lives in pure functions so it can be unit-tested without downloading the model.

**Tech Stack:** Python ≥3.10, PyTorch, transformers ≥5 (fast tokenizer), huggingface_hub, FastAPI, Streamlit, pytest.

---

## File Structure

| File | Responsibility |
|---|---|
| `eka_pii_redaction/taxonomy.py` | (modify) add `mac_address → device_net`; add `TEXT_REDACTABLE`. |
| `eka_pii_redaction/text/minilm.py` | (create) `MiniLMDetector` + pure span-assembly functions. |
| `eka_pii_redaction/text/redactor.py` | (replace stub) `TextPIIRedactor` + `TextPIISpan` + pure `apply_mask`. |
| `eka_pii_redaction/text/__init__.py` | (modify) export `TextPIIRedactor`, `TextPIISpan`, `DEFAULT_HF_REPO`. |
| `eka_pii_redaction/__init__.py` | (modify) re-export `TextPIIRedactor`, `TextPIISpan`. |
| `eka_pii_redaction/server.py` | (modify) add `/detect-text`, `/redact-text`. |
| `pyproject.toml` | (modify) add `sentencepiece` dependency. |
| `streamlit_app.py` | (replace) Image + Text tabs, lazy per-tab loading. |
| `README.md`, `scripts/build_hf_repo.py` | (modify) document text modality as implemented. |
| `tests/test_taxonomy.py` | (create) taxonomy unit tests. |
| `tests/test_text_minilm.py` | (create) pure span-assembly unit tests. |
| `tests/test_text_redactor.py` | (create) `apply_mask` unit tests. |
| `tests/test_package_exports.py` | (create) import-surface tests. |
| `tests/test_text_integration.py` | (create) env-gated real-model smoke test. |

All `pytest` commands are run from the repo root: `/Users/rushabh/orbi/code/Eka-PII-redactors`.

---

### Task 1: Taxonomy — add `mac_address` and `TEXT_REDACTABLE`

**Files:**
- Modify: `eka_pii_redaction/taxonomy.py`
- Test: `tests/test_taxonomy.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_taxonomy.py`:

```python
from eka_pii_redaction.taxonomy import (
    ALL_ENTITIES, TEXT_ENTITIES, TEXT_REDACTABLE, VISUAL_ENTITIES,
    TEXT_L2_TO_L1, l1_group,
)


def test_mac_address_added_to_device_net():
    assert "mac_address" in TEXT_L2_TO_L1
    assert l1_group("mac_address") == "device_net"
    assert "mac_address" in TEXT_ENTITIES
    assert "mac_address" in ALL_ENTITIES


def test_text_redactable_is_text_only():
    # Text model predicts the text categories, never the visual ones.
    assert set(TEXT_REDACTABLE) == set(TEXT_ENTITIES)
    for v in VISUAL_ENTITIES:
        assert v not in TEXT_REDACTABLE


def test_text_redactable_matches_text_model_label_set():
    # The 48 lowercased entity types emitted by text/minilm/labels.json.
    expected = {
        "aadhaar_12_digit", "abha_address", "abha_number_14_digit", "age",
        "api_key_token", "bank_account_number", "blood_type", "brandname",
        "city_district", "country", "date_of_birth", "death_date",
        "device_serial_identifier", "driving_licence_no", "email", "fax",
        "gender", "geocode_coordinates", "health_plan_beneficiary_no",
        "high_entropy_secret", "iban", "insurance_tpa_policy_no", "ip_address",
        "mac_address", "mrn_uhid", "national_id",
        "occupation_designation_education_level", "other_date_time", "other_id",
        "other_person_name", "pan", "passport_no", "password", "phone_mobile",
        "pmjay_ayushman_id", "postal_zip_pin_code", "practitioner_reg_no_npi_nmc",
        "primary_subject_name", "religion_ethnicity_cast", "sexual_orientation",
        "ssn", "state_province", "street_address", "tax_id", "upi_id",
        "vehicle_id_licence_plate", "voter_id", "web_url",
    }
    assert set(TEXT_REDACTABLE) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_taxonomy.py -v`
Expected: FAIL — `ImportError: cannot import name 'TEXT_REDACTABLE'` (and `mac_address` missing).

- [ ] **Step 3: Implement the taxonomy changes**

In `eka_pii_redaction/taxonomy.py`, add one entry to `TEXT_L2_TO_L1` next to the other `device_net` entries (after the `ip_address` / `device_serial_identifier` lines):

```python
    "ip_address": "device_net", "device_serial_identifier": "device_net",
    "mac_address": "device_net",
```

Then, immediately after the `TEXT_ENTITIES` definition, add:

```python
# Categories the TEXT modality (plain-string redactor) can emit. The text model
# predicts every text category but none of the visual ones, so this is just the
# text entity list — named separately so the text API can advertise its own set.
TEXT_REDACTABLE: list[str] = list(TEXT_ENTITIES)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_taxonomy.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add eka_pii_redaction/taxonomy.py tests/test_taxonomy.py
git commit -m "taxonomy: add mac_address (device_net) and TEXT_REDACTABLE"
```

---

### Task 2: `MiniLMDetector` — pure span-assembly functions (TDD)

These three pure functions do all the BIO→character-span logic and need no model. The `MiniLMDetector` class (Task 3) only feeds them tokenizer/model output.

**Files:**
- Create: `eka_pii_redaction/text/minilm.py`
- Test: `tests/test_text_minilm.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_text_minilm.py`:

```python
from eka_pii_redaction.text.minilm import _trim, _flatten_chunks, merge_bio_spans


def test_trim_strips_surrounding_whitespace():
    text = "  Asha  "
    assert _trim(text, 0, len(text)) == (2, 6)


def test_merge_simple_bi_span_trims_leading_space():
    text = "Email john@x.com now"
    offsets = [(0, 5), (5, 10), (10, 12), (12, 16), (16, 20)]
    labels = ["O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "O"]
    scores = [0.9, 0.8, 0.7, 0.6, 0.95]
    spans = merge_bio_spans(text, offsets, labels, scores)
    assert len(spans) == 1
    s = spans[0]
    assert s.category == "email"
    assert s.l1 == "contact"
    assert (s.start, s.end) == (6, 16)          # leading space (5->6) trimmed
    assert s.text == "john@x.com"
    assert s.score == round((0.8 + 0.7 + 0.6) / 3, 4)


def test_b_tag_starts_new_span_even_when_adjacent():
    text = "Asha Bob"
    offsets = [(0, 4), (4, 8)]
    labels = ["B-PRIMARY_SUBJECT_NAME", "B-OTHER_PERSON_NAME"]
    scores = [0.9, 0.9]
    spans = merge_bio_spans(text, offsets, labels, scores)
    assert [s.category for s in spans] == [
        "primary_subject_name", "other_person_name"]


def test_specials_and_O_tokens_skipped():
    text = "hi name"
    offsets = [(0, 0), (0, 2), (2, 7), (0, 0)]   # <s>, "hi"=O, " name", </s>
    labels = ["O", "O", "B-OTHER_PERSON_NAME", "O"]
    scores = [None, 0.9, 0.8, None]
    spans = merge_bio_spans(text, offsets, labels, scores)
    assert len(spans) == 1
    assert (spans[0].start, spans[0].end) == (3, 7)
    assert spans[0].text == "name"


def test_flatten_dedups_overlap_first_chunk_wins():
    chunk_offsets = [[(0, 3), (3, 6)], [(3, 6), (6, 9)]]
    chunk_labels = [["B-AGE", "I-AGE"], ["B-GENDER", "B-COUNTRY"]]
    chunk_scores = [[0.9, 0.8], [0.5, 0.7]]
    offs, labs, scs = _flatten_chunks(chunk_offsets, chunk_labels, chunk_scores)
    assert offs == [(0, 3), (3, 6), (6, 9)]
    assert labs == ["B-AGE", "I-AGE", "B-COUNTRY"]   # (3,6): first chunk wins
    assert scs == [0.9, 0.8, 0.7]


def test_flatten_skips_zero_width_tokens():
    chunk_offsets = [[(0, 0), (0, 3), (0, 0)]]
    chunk_labels = [["O", "B-AGE", "O"]]
    chunk_scores = [[None, 0.9, None]]
    offs, labs, scs = _flatten_chunks(chunk_offsets, chunk_labels, chunk_scores)
    assert offs == [(0, 3)]
    assert labs == ["B-AGE"]
    assert scs == [0.9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_text_minilm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eka_pii_redaction.text.minilm'`.

- [ ] **Step 3: Create the module with the pure functions**

Create `eka_pii_redaction/text/minilm.py` with this content (the `MiniLMDetector` class is added in Task 3; write the whole file now so imports resolve, but the class body in Task 3 — for this task include ONLY the imports and the three pure functions):

```python
"""MiniLM detector — finds PII inside plain-text strings (the TEXT modality).

Runs a multilingual MiniLM token classifier (BertForTokenClassification head over
an XLMRobertaTokenizerFast) on raw text and maps the per-token BIO predictions back
to character spans using the fast tokenizer's offset mapping. No OCR, no image.

This is the text counterpart to image/layoutlmv3.py.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ..taxonomy import l1_group
from .redactor import TextPIISpan


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) to exclude leading/trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _flatten_chunks(chunk_offsets, chunk_labels, chunk_scores):
    """Flatten per-chunk (offsets, labels, scores) into a single de-duplicated,
    char-start-sorted sequence.

    Long inputs are tokenized into overlapping (strided) chunks, so a token can
    appear in two chunks. We key tokens by their (char_start, char_end) and keep
    the FIRST occurrence (earlier chunk wins), dropping zero-width tokens (special
    + padding tokens, whose offsets are (0, 0))."""
    seen: dict[tuple[int, int], tuple[str, Optional[float]]] = {}
    for offs, labs, scs in zip(chunk_offsets, chunk_labels, chunk_scores):
        for (cs, ce), lab, sc in zip(offs, labs, scs):
            if ce <= cs:
                continue
            key = (cs, ce)
            if key not in seen:
                seen[key] = (lab, sc)
    items = sorted(seen.items(), key=lambda kv: kv[0])
    offsets = [k for k, _ in items]
    labels = [v[0] for _, v in items]
    scores = [v[1] for _, v in items]
    return offsets, labels, scores


def merge_bio_spans(text, offsets, labels, scores) -> "list[TextPIISpan]":
    """Merge a flat token sequence of (offset, BIO label, score) into entity spans.

    Labels are uppercase BIO (e.g. "B-EMAIL"); the category is `label[2:].lower()`,
    matching the taxonomy keys. A "B-" tag, a category change, or a non-entity
    token ends the current span. Each emitted span's char range is whitespace-
    trimmed and its score is the mean of its tokens' max-probabilities.
    """
    spans: list[TextPIISpan] = []
    cur = None  # {category, start, end, scores}

    def flush():
        nonlocal cur
        if cur is None:
            return
        s, e = _trim(text, cur["start"], cur["end"])
        if e > s:
            sc = [x for x in cur["scores"] if x is not None]
            spans.append(TextPIISpan(
                category=cur["category"], start=s, end=e,
                l1=l1_group(cur["category"]), text=text[s:e],
                score=round(sum(sc) / len(sc), 4) if sc else None,
            ))
        cur = None

    for (cs, ce), lab, sc in zip(offsets, labels, scores):
        if ce <= cs:
            continue
        if lab == "O":
            flush()
            continue
        tag, cat = lab[0], lab[2:].lower()
        if tag == "B" or cur is None or cur["category"] != cat:
            flush()
            cur = {"category": cat, "start": cs, "end": ce, "scores": [sc]}
        else:  # I- continuing the same category
            cur["end"] = max(cur["end"], ce)
            cur["scores"].append(sc)
    flush()
    return spans
```

Note: this imports `TextPIISpan` from `.redactor`, which already exists (the stub defines it). Task 4 keeps that dataclass intact.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_text_minilm.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add eka_pii_redaction/text/minilm.py tests/test_text_minilm.py
git commit -m "text: add MiniLM BIO->char-span assembly (pure functions)"
```

---

### Task 3: `MiniLMDetector` class — model wrapper

Adds the model-loading + inference class to `minilm.py`. No unit test (it needs the real model — covered by the env-gated integration test in Task 6).

**Files:**
- Modify: `eka_pii_redaction/text/minilm.py` (append the class)

- [ ] **Step 1: Append the `MiniLMDetector` class**

Append to the end of `eka_pii_redaction/text/minilm.py`:

```python
class MiniLMDetector:
    """Detects PII in a plain-text string via a MiniLM token classifier."""

    def __init__(self, model_dir: str, device: str,
                 max_length: int = 512, stride: int = 128):
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "TextPIIRedactor needs a fast tokenizer (offset mapping); "
                f"{model_dir} loaded a slow one."
            )
        self.model = (
            AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
        )
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def detect(self, text: str) -> "list[TextPIISpan]":
        """Return character-span PII entities for one text string."""
        if not text or not text.strip():
            return []

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            stride=self.stride,
            return_tensors="pt",
        )
        offset_mapping = enc.pop("offset_mapping").tolist()   # (n_chunks, T, 2)
        enc.pop("overflow_to_sample_mapping", None)
        on_device = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**on_device).logits               # (n_chunks, T, C)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1).cpu().tolist()
        pred_scores = probs.max(-1).values.cpu().tolist()

        chunk_offsets = [[tuple(o) for o in chunk] for chunk in offset_mapping]
        chunk_labels = [[self.id2label[i] for i in ids] for ids in pred_ids]

        offsets, labels, scores = _flatten_chunks(
            chunk_offsets, chunk_labels, pred_scores)
        return merge_bio_spans(text, offsets, labels, scores)
```

- [ ] **Step 2: Verify the module still imports**

Run: `python -c "import eka_pii_redaction.text.minilm as m; print(m.MiniLMDetector)"`
Expected: prints `<class 'eka_pii_redaction.text.minilm.MiniLMDetector'>` (no error).

- [ ] **Step 3: Re-run the pure-function tests (no regression)**

Run: `python -m pytest tests/test_text_minilm.py -v`
Expected: PASS (6 passed).

- [ ] **Step 4: Commit**

```bash
git add eka_pii_redaction/text/minilm.py
git commit -m "text: add MiniLMDetector model wrapper (load + chunked inference)"
```

---

### Task 4: `TextPIIRedactor` — replace the stub

Replaces the `NotImplementedError` stub in `text/redactor.py` with the real orchestrator. Keeps the existing `TextPIISpan` dataclass and adds a pure `apply_mask` helper (unit-tested) plus the class.

**Files:**
- Modify: `eka_pii_redaction/text/redactor.py`
- Test: `tests/test_text_redactor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_text_redactor.py`:

```python
from eka_pii_redaction.text.redactor import (
    DEFAULT_HF_REPO, TextPIISpan, apply_mask,
)


def _span(cat, start, end, l1, txt):
    return TextPIISpan(category=cat, start=start, end=end, l1=l1, text=txt, score=0.9)


def test_default_hf_repo_constant():
    assert DEFAULT_HF_REPO == "ekacare/pii-redactors"


def test_apply_mask_default_token():
    text = "Name Asha here"
    spans = [_span("primary_subject_name", 5, 9, "person", "Asha")]
    assert apply_mask(text, spans) == "Name [REDACTED] here"


def test_apply_mask_category_placeholder():
    text = "Name Asha here"
    spans = [_span("primary_subject_name", 5, 9, "person", "Asha")]
    assert apply_mask(text, spans, mask="[{category}]") == "Name [primary_subject_name] here"


def test_apply_mask_l1_placeholder():
    text = "Name Asha here"
    spans = [_span("primary_subject_name", 5, 9, "person", "Asha")]
    assert apply_mask(text, spans, mask="<{l1}>") == "Name <person> here"


def test_apply_mask_multiple_spans_in_order():
    text = "a@b.com and Asha"
    spans = [
        _span("email", 0, 7, "contact", "a@b.com"),
        _span("primary_subject_name", 12, 16, "person", "Asha"),
    ]
    assert apply_mask(text, spans) == "[REDACTED] and [REDACTED]"


def test_apply_mask_unsorted_spans_handled():
    text = "a@b.com and Asha"
    spans = [
        _span("primary_subject_name", 12, 16, "person", "Asha"),
        _span("email", 0, 7, "contact", "a@b.com"),
    ]
    assert apply_mask(text, spans) == "[REDACTED] and [REDACTED]"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_text_redactor.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_mask'` / `'DEFAULT_HF_REPO'`.

- [ ] **Step 3: Replace `text/redactor.py`**

Replace the ENTIRE contents of `eka_pii_redaction/text/redactor.py` with:

```python
"""TextPIIRedactor — detect & redact PII inside plain-text strings (TEXT modality).

The text counterpart to `eka_pii_redaction.image.ImagePIIRedactor`. Operates on
raw strings (no image, no OCR) via a multilingual MiniLM token classifier and
returns character-span annotations and/or a redacted string. The weights live
under `text/minilm/` in the same single HF repo as the image models:

    <hf_repo>/
        image/   layoutlmv3/, yolo/best.pt     # image modality
        text/    minilm/                        # text modality (this)

    from eka_pii_redaction.text import TextPIIRedactor

    r = TextPIIRedactor("ekacare/pii-redactors")
    spans = r.detect("John Doe, DOB 1990-01-01, lives at ...")
    #   -> list[TextPIISpan]: {category, start, end, l1, text, score}
    clean = r.redact("...", mask="[REDACTED]")   # -> str with PII replaced

The category taxonomy is shared with the image modality (see
`eka_pii_redaction.taxonomy`), minus the visual-only categories.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ..taxonomy import TEXT_REDACTABLE, validate_entities

DEFAULT_HF_REPO = "ekacare/pii-redactors"
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
        exclude_entities: Optional[Iterable[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            hf_repo: Hugging Face repo id (or a local dir) holding the weights.
            device: "cuda" / "cpu". None -> auto (cuda if available, else cpu).
            exclude_entities: categories to never detect/redact. Default: none.
        """
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._excluded = set(validate_entities(exclude_entities))

        # Imported here so a missing torch/transformers only bites at construction,
        # and the pure helpers above stay importable for tests.
        from .minilm import MiniLMDetector

        model_dir = _resolve_text_source(hf_repo, cache_dir)
        self.minilm = MiniLMDetector(model_dir, self.device)

    @classmethod
    def list_entities(cls) -> list[str]:
        """Return all categories the text modality can detect/redact."""
        return list(TEXT_REDACTABLE)

    def _active_exclusions(self, extra: Optional[Iterable[str]]) -> set:
        e = set(validate_entities(extra)) if extra else set()
        return self._excluded | e

    def detect(
        self, text: str, *, exclude_entities: Optional[Iterable[str]] = None
    ) -> "list[TextPIISpan]":
        """Detect PII spans in `text`. Each TextPIISpan has `category`, `start`,
        `end` (char offsets), `l1`, `text`, `score`.

        `exclude_entities` (per-call) is unioned with the constructor's exclusions.
        """
        excluded = self._active_exclusions(exclude_entities)
        return [s for s in self.minilm.detect(text) if s.category not in excluded]

    def redact(
        self,
        text: str,
        *,
        mask: str = "[REDACTED]",
        exclude_entities: Optional[Iterable[str]] = None,
    ) -> str:
        """Return `text` with every detected PII span replaced by `mask`.

        `mask` may contain `{category}` / `{l1}` placeholders (see `apply_mask`).
        """
        spans = self.detect(text, exclude_entities=exclude_entities)
        return apply_mask(text, spans, mask=mask)
```

Note: `MiniLMDetector` is imported lazily inside `__init__` (not at module top) so `apply_mask`/`TextPIISpan` stay importable without torch+transformers. `minilm.py` imports `TextPIISpan` from this module at its top — that's fine because importing `redactor` does not import `minilm`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_text_redactor.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Run the full unit suite (no regression)**

Run: `python -m pytest tests/test_taxonomy.py tests/test_text_minilm.py tests/test_text_redactor.py -v`
Expected: PASS (15 passed).

- [ ] **Step 6: Commit**

```bash
git add eka_pii_redaction/text/redactor.py tests/test_text_redactor.py
git commit -m "text: implement TextPIIRedactor (detect/redact/list_entities)"
```

---

### Task 5: Package exports

Wire `TextPIIRedactor` / `TextPIISpan` into the `text` package `__init__` and the top-level package `__init__`.

**Files:**
- Modify: `eka_pii_redaction/text/__init__.py`
- Modify: `eka_pii_redaction/__init__.py`
- Test: `tests/test_package_exports.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_package_exports.py`:

```python
def test_text_package_exports():
    from eka_pii_redaction.text import (
        TextPIIRedactor, TextPIISpan, DEFAULT_HF_REPO,
    )
    assert DEFAULT_HF_REPO == "ekacare/pii-redactors"
    assert TextPIIRedactor.list_entities()  # non-empty list


def test_top_level_exports():
    import eka_pii_redaction as pkg
    assert hasattr(pkg, "TextPIIRedactor")
    assert hasattr(pkg, "TextPIISpan")
    assert hasattr(pkg, "ImagePIIRedactor")
    for name in ("TextPIIRedactor", "TextPIISpan", "ImagePIIRedactor"):
        assert name in pkg.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_package_exports.py -v`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_HF_REPO'` from `eka_pii_redaction.text` (and top-level missing `TextPIIRedactor`).

- [ ] **Step 3: Update `text/__init__.py`**

Replace the contents of `eka_pii_redaction/text/__init__.py` with:

```python
"""Text modality: detect & redact PII inside plain-text strings."""
from .redactor import DEFAULT_HF_REPO, TextPIIRedactor, TextPIISpan

__all__ = ["TextPIIRedactor", "TextPIISpan", "DEFAULT_HF_REPO"]
```

- [ ] **Step 4: Update top-level `__init__.py`**

In `eka_pii_redaction/__init__.py`, update the docstring line for text (it currently says "planned"), add the import, and extend `__all__`. The full new file:

```python
"""Eka-PII-redaction — detect and redact PII in documents, organized by modality.

- `eka_pii_redaction.image.ImagePIIRedactor` — PII in document images.
- `eka_pii_redaction.text.TextPIIRedactor`   — PII in plain-text strings.

`ImagePIIRedactor` and `TextPIIRedactor` are also re-exported at the top level.
"""
from .entities import PIIEntity
from .image import ImagePIIRedactor
from .taxonomy import ALL_ENTITIES, TEXT_ENTITIES, TEXT_REDACTABLE, VISUAL_ENTITIES
from .text import TextPIIRedactor, TextPIISpan

__version__ = "0.1.0"
__all__ = [
    "ImagePIIRedactor",
    "TextPIIRedactor",
    "TextPIISpan",
    "PIIEntity",
    "ALL_ENTITIES",
    "TEXT_ENTITIES",
    "TEXT_REDACTABLE",
    "VISUAL_ENTITIES",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_package_exports.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add eka_pii_redaction/__init__.py eka_pii_redaction/text/__init__.py tests/test_package_exports.py
git commit -m "exports: surface TextPIIRedactor + TextPIISpan at package top level"
```

---

### Task 6: Env-gated real-model integration smoke test

A single end-to-end test that downloads the real model and verifies detection +
redaction. Skipped by default so the unit suite stays offline/fast; run it with
`RUN_TEXT_INTEGRATION=1`.

**Files:**
- Create: `tests/test_text_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_text_integration.py`:

```python
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TEXT_INTEGRATION") != "1",
    reason="set RUN_TEXT_INTEGRATION=1 to run the real-model text smoke test",
)


@pytest.fixture(scope="module")
def redactor():
    from eka_pii_redaction import TextPIIRedactor
    return TextPIIRedactor(device="cpu")


def test_detect_finds_email_and_name(redactor):
    text = "Patient John Doe, email john.doe@example.com, lives in Bangalore."
    spans = redactor.detect(text)
    cats = {s.category for s in spans}
    assert "email" in cats
    # Every span's text slice matches its offsets exactly.
    for s in spans:
        assert text[s.start:s.end] == s.text
        assert 0 <= s.start < s.end <= len(text)


def test_redact_removes_email_text(redactor):
    text = "Reach me at john.doe@example.com please."
    out = redactor.redact(text, mask="[{category}]")
    assert "john.doe@example.com" not in out
    assert "[email]" in out
```

- [ ] **Step 2: Run it (real model — downloads weights on first run)**

Run: `RUN_TEXT_INTEGRATION=1 python -m pytest tests/test_text_integration.py -v`
Expected: PASS (2 passed). First run downloads the `text/minilm` weights.

- [ ] **Step 3: Verify it skips by default**

Run: `python -m pytest tests/test_text_integration.py -v`
Expected: 2 skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/test_text_integration.py
git commit -m "test: env-gated real-model text smoke test"
```

---

### Task 7: Add `sentencepiece` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, in the `[project].dependencies` list, add `sentencepiece` after the `huggingface_hub` line:

```python
    "huggingface_hub>=0.23",
    # XLM-R tokenizer backing the text MiniLM model (fast path loads from
    # tokenizer.json, but the slow-tokenizer fallback needs sentencepiece).
    "sentencepiece>=0.1.99",
    "pillow>=9.0",
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add sentencepiece for the text MiniLM tokenizer"
```

---

### Task 8: FastAPI server — text endpoints

Add `/detect-text` and `/redact-text`, backed by a lazily-built shared
`TextPIIRedactor` (mirrors the existing image lazy singleton).

**Files:**
- Modify: `eka_pii_redaction/server.py`

- [ ] **Step 1: Add the text redactor singleton + endpoints**

In `eka_pii_redaction/server.py`:

1. Add `from pydantic import BaseModel` to the imports (after the FastAPI import line).

2. After the existing `_redactor: ImagePIIRedactor | None = None` line, add:

```python
_text_redactor = None  # built lazily on first /detect-text or /redact-text call


class _TextIn(BaseModel):
    text: str


class _RedactTextIn(BaseModel):
    text: str
    mask: str = "[REDACTED]"


def _get_text():
    global _text_redactor
    if _text_redactor is None:
        from .taxonomy import TEXT_REDACTABLE
        from .text import TextPIIRedactor

        raw = [s for s in os.environ.get("EKA_PII_EXCLUDE", "").split(",") if s]
        # EKA_PII_EXCLUDE may name visual categories (for the image model); keep
        # only the ones the text model knows so construction doesn't error.
        exclude = [e for e in raw if e in TEXT_REDACTABLE]
        _text_redactor = TextPIIRedactor(
            hf_repo=os.environ.get("EKA_PII_HF_REPO", DEFAULT_HF_REPO),
            device=os.environ.get("EKA_PII_DEVICE") or None,
            exclude_entities=exclude or None,
        )
    return _text_redactor
```

3. At the end of the file, add the endpoints:

```python
@app.get("/entities-text")
def entities_text():
    from .text import TextPIIRedactor
    return {"text": TextPIIRedactor.list_entities()}


@app.post("/detect-text")
def detect_text(body: _TextIn):
    spans = _get_text().detect(body.text)
    return {"spans": [s.to_dict() for s in spans]}


@app.post("/redact-text")
def redact_text(body: _RedactTextIn):
    return {"text": _get_text().redact(body.text, mask=body.mask)}
```

- [ ] **Step 2: Verify the module imports**

Run: `python -c "import eka_pii_redaction.server as s; print([r.path for r in s.app.routes if hasattr(r,'path')])"`
Expected: the printed list includes `/detect-text`, `/redact-text`, and `/entities-text` (alongside `/detect`, `/redact`, `/entities`, `/health`).

- [ ] **Step 3: Manual smoke test (optional, needs server extra + model)**

```bash
pip install -e ".[server]"
EKA_PII_DEVICE=cpu uvicorn eka_pii_redaction.server:app --port 8080 &
sleep 5
curl -s -X POST http://localhost:8080/detect-text \
  -H 'Content-Type: application/json' \
  -d '{"text":"Reach John at john@x.com"}'
# -> {"spans":[{"category":"primary_subject_name",...},{"category":"email",...}]}
kill %1
```
Expected: JSON with at least an `email` span.

- [ ] **Step 4: Commit**

```bash
git add eka_pii_redaction/server.py
git commit -m "server: add /detect-text, /redact-text, /entities-text endpoints"
```

---

### Task 9: Streamlit app — Image + Text tabs

Replace `streamlit_app.py` with a two-tab version. Each tab loads ONLY its own
model, and only when its run button is clicked, via separate `@st.cache_resource`
loaders — so image-only users never download the text model and vice-versa.

**Files:**
- Replace: `streamlit_app.py`

- [ ] **Step 1: Replace `streamlit_app.py`**

Replace the ENTIRE file with:

```python
"""Streamlit app to test the Eka PII redactors — image and text modalities.

Run:
    pip install -e .            # needs Python >=3.10
    pip install streamlit
    streamlit run streamlit_app.py

Two tabs:
  - Image : detect/annotate or redact PII regions in a document image.
  - Text  : detect PII in pasted text and highlight each span (color-coded by group).

The model repo is downloaded via huggingface_hub, which uses your cached HF login
(`huggingface-cli login`) or the HF_TOKEN env var automatically.
"""
from __future__ import annotations

import html
import io
import time

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from eka_pii_redaction.image import DEFAULT_HF_REPO
from eka_pii_redaction.taxonomy import TEXT_ENTITIES, TEXT_REDACTABLE, VISUAL_ENTITIES

st.set_page_config(page_title="Eka PII Redactor — Tester", layout="wide")

# A distinct color per coarse L1 group, used by both the image overlay and the
# text highlighter.
L1_COLORS = {
    "person": (220, 38, 38),
    "location": (37, 99, 235),
    "date_time": (217, 119, 6),
    "contact": (5, 150, 105),
    "uid": (124, 58, 237),
    "device_net": (8, 145, 178),
    "credential": (190, 18, 60),
    "entity": (100, 116, 139),
    "biometric_visual": (219, 39, 119),
    "unknown": (75, 85, 99),
}


# ---------------------------------------------------------------- loaders --- #
@st.cache_resource(show_spinner=False)
def load_image_redactor(hf_repo: str, detect_visual: bool, device: str,
                        visual_score_threshold: float):
    from eka_pii_redaction import ImagePIIRedactor

    return ImagePIIRedactor(
        hf_repo,
        detect_visual=detect_visual,
        device=(None if device == "auto" else device),
        visual_score_threshold=visual_score_threshold,
    )


@st.cache_resource(show_spinner=False)
def load_text_redactor(hf_repo: str, device: str):
    from eka_pii_redaction import TextPIIRedactor

    return TextPIIRedactor(hf_repo, device=(None if device == "auto" else device))


# ---------------------------------------------------------------- shared --- #
def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _rgb(group: str) -> str:
    c = L1_COLORS.get(group, L1_COLORS["unknown"])
    return f"rgb({c[0]},{c[1]},{c[2]})"


def legend_html(groups) -> str:
    chips = "".join(
        f'<span style="background:{_rgb(g)};color:#fff;padding:1px 7px;'
        f'border-radius:4px;margin:0 6px 4px 0;display:inline-block;font-size:.8rem;">'
        f'{html.escape(g)}</span>'
        for g in groups
    )
    return f'<div style="margin:4px 0 10px;">{chips}</div>'


# ---------------------------------------------------------------- image ----- #
def draw_boxes(img: Image.Image, entities) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(13, out.width // 90))
    except Exception:
        font = ImageFont.load_default()

    for e in entities:
        x0, y0, x1, y1 = e.bbox
        color = L1_COLORS.get(e.l1, L1_COLORS["unknown"])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = e.category
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(label) * 6, 12
        ly = max(0, y0 - th - 4)
        draw.rectangle([x0, ly, x0 + tw + 6, ly + th + 4], fill=color)
        draw.text((x0 + 3, ly + 2), label, fill=(255, 255, 255), font=font)
    return out


def entity_rows(entities):
    return [
        {
            "kind": e.kind,
            "category": e.category,
            "l1": e.l1,
            "text": e.text or "",
            "score": round(e.score, 3) if e.score is not None else None,
            "bbox": list(e.bbox),
        }
        for e in entities
    ]


# ---------------------------------------------------------------- text ------ #
def highlight_spans(text: str, spans) -> str:
    """Return HTML with each PII span wrapped in a color-coded <mark>."""
    parts, prev = [], 0
    for sp in sorted(spans, key=lambda s: s.start):
        if sp.start < prev:
            continue
        parts.append(html.escape(text[prev:sp.start]))
        score = f"{sp.score:.3f}" if sp.score is not None else "?"
        parts.append(
            f'<mark style="background:{_rgb(sp.l1)};color:#fff;padding:0 3px;'
            f'border-radius:3px;" title="{html.escape(sp.category)} · {score}">'
            f'{html.escape(text[sp.start:sp.end])}</mark>'
        )
        prev = sp.end
    parts.append(html.escape(text[prev:]))
    body = "".join(parts)
    return (
        '<div style="white-space:pre-wrap;line-height:2;font-size:1.05rem;'
        'border:1px solid #ddd;border-radius:8px;padding:12px;">' + body + "</div>"
    )


def span_rows(spans):
    return [
        {
            "category": s.category,
            "l1": s.l1,
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "score": round(s.score, 3) if s.score is not None else None,
        }
        for s in spans
    ]


# ---------------------------------------------------------------- sidebar --- #
st.sidebar.title("⚙️ Configuration")
hf_repo = st.sidebar.text_input("HF repo id (or local dir)", value=DEFAULT_HF_REPO)
device = st.sidebar.selectbox(
    "Device", ["auto", "cpu", "cuda"], index=0,
    help="auto = CUDA if available, else CPU. Shared by both tabs.",
)
if st.sidebar.button("🔄 Clear model cache", use_container_width=True):
    load_image_redactor.clear()
    load_text_redactor.clear()
    st.sidebar.success("Cache cleared — models reload on next run.")

st.title("🛡️ Eka PII Redactor — Tester")
tab_img, tab_txt = st.tabs(["🖼️ Image", "📝 Text"])

# ================================================================ IMAGE ==== #
with tab_img:
    st.subheader("Image modality")

    detect_visual = st.checkbox(
        "Detect visual entities (YOLO)", value=True,
        help="Off = text-in-image PII only; skips the YOLO weights + inference.",
    )
    c1, c2 = st.columns(2)
    visual_score_threshold = c1.slider(
        "Visual score threshold", 0.0, 1.0, 0.25, 0.05, disabled=not detect_visual,
    )
    ocr_lang = c2.text_input(
        "OCR language (Tesseract)", value="",
        help='e.g. "eng" or "eng+Devanagari". Blank = default.',
    ).strip() or None

    with st.expander("Exclude categories (never detect/redact)"):
        exclude_text = st.multiselect("Text categories", TEXT_ENTITIES, default=[])
        exclude_visual = st.multiselect(
            "Visual categories", VISUAL_ENTITIES, default=[], disabled=not detect_visual,
        )
    img_exclude = (list(exclude_text) + list(exclude_visual)) or None

    action = st.radio(
        "Action", ["Detect (annotate boxes)", "Redact"], horizontal=True,
        help="Detect draws labeled boxes around PII. Redact hides each PII region.",
    )
    is_redact = action == "Redact"

    redact_mode, redact_color = "solid", (0, 0, 0)
    if is_redact:
        with st.container(border=True):
            rc1, rc2 = st.columns([2, 1])
            redact_mode = rc1.selectbox("Mode", ["solid", "blur", "pixelate"], index=0)
            hex_color = rc2.color_picker(
                "Fill color (solid)", value="#000000", disabled=redact_mode != "solid"
            )
            redact_color = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

    uploaded = st.file_uploader(
        "Upload a document image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    )
    run_img = st.button(
        f"🔍 {'Detect & annotate' if not is_redact else 'Detect & redact'}",
        type="primary", use_container_width=True, disabled=uploaded is None,
        key="run_img",
    )

    if uploaded is not None and not run_img:
        st.image(Image.open(uploaded).convert("RGB"),
                 caption="Original — press the button to run", use_container_width=True)
    elif uploaded is not None and run_img:
        image = Image.open(uploaded).convert("RGB")
        try:
            with st.spinner("Loading image model (first run downloads weights)…"):
                redactor = load_image_redactor(
                    hf_repo, detect_visual, device, visual_score_threshold)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load image model: {exc}")
            st.stop()

        with st.spinner("Detecting…"):
            t0 = time.time()
            entities = redactor.detect(image, exclude_entities=img_exclude, ocr_lang=ocr_lang)
            detect_s = time.time() - t0

        n_text = sum(1 for e in entities if e.kind == "text")
        n_visual = sum(1 for e in entities if e.kind == "visual")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entities", len(entities))
        m2.metric("Text", n_text)
        m3.metric("Visual", n_visual)
        m4.metric("Detect time", f"{detect_s:.1f}s")

        if is_redact:
            with st.spinner("Redacting…"):
                redacted = redactor.redact(
                    image, mode=redact_mode, color=redact_color,
                    exclude_entities=img_exclude, ocr_lang=ocr_lang,
                )
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Detections")
                st.image(draw_boxes(image, entities), use_container_width=True)
            with col_b:
                st.caption(f"Redacted ({redact_mode})")
                st.image(redacted, use_container_width=True)
                st.download_button(
                    "⬇️ Download redacted PNG", data=to_png_bytes(redacted),
                    file_name="redacted.png", mime="image/png",
                    use_container_width=True,
                )
        else:
            st.caption("Annotated detections")
            st.image(draw_boxes(image, entities), use_container_width=True)
            st.download_button(
                "⬇️ Download annotated PNG", data=to_png_bytes(draw_boxes(image, entities)),
                file_name="annotated.png", mime="image/png",
            )

        st.caption("Detected entities")
        if entities:
            st.dataframe(entity_rows(entities), use_container_width=True, hide_index=True)
        else:
            st.info("No PII entities detected.")
    else:
        st.info("⬆️ Upload an image to begin.")

# ================================================================ TEXT ===== #
with tab_txt:
    st.subheader("Text modality")
    st.caption("Detects PII in raw text and highlights each span (color-coded by group).")

    with st.expander("Exclude categories (never detect)"):
        txt_exclude = st.multiselect("Text categories", TEXT_REDACTABLE, default=[]) or None

    sample = ("Patient John Doe (DOB 1990-01-01), Aadhaar 1234 5678 9012, "
              "email john.doe@example.com, phone +91 98765 43210, lives in Bangalore.")
    text_in = st.text_area("Text to scan", value=sample, height=180)
    run_txt = st.button(
        "🔍 Detect PII", type="primary", use_container_width=True,
        disabled=not text_in.strip(), key="run_txt",
    )

    if run_txt and text_in.strip():
        try:
            with st.spinner("Loading text model (first run downloads weights)…"):
                tredactor = load_text_redactor(hf_repo, device)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load text model: {exc}")
            st.stop()

        with st.spinner("Detecting…"):
            t0 = time.time()
            spans = tredactor.detect(text_in, exclude_entities=txt_exclude)
            detect_s = time.time() - t0

        m1, m2 = st.columns(2)
        m1.metric("Spans", len(spans))
        m2.metric("Detect time", f"{detect_s:.2f}s")

        if spans:
            groups = sorted({s.l1 for s in spans})
            st.markdown(legend_html(groups), unsafe_allow_html=True)
            st.markdown(highlight_spans(text_in, spans), unsafe_allow_html=True)
            st.caption("Detected spans")
            st.dataframe(span_rows(spans), use_container_width=True, hide_index=True)
        else:
            st.info("No PII detected.")
    else:
        st.info("✏️ Enter text and press **Detect PII**.")
```

- [ ] **Step 2: Verify the file parses (syntax check)**

Run: `python -m py_compile streamlit_app.py && echo OK`
Expected: prints `OK`.

- [ ] **Step 3: Manual run (visual verification)**

Run: `streamlit run streamlit_app.py`
Expected: two tabs (🖼️ Image, 📝 Text). The Text tab, with the prefilled sample, after pressing **Detect PII** shows colored highlights over `John Doe`, the Aadhaar number, the email, the phone, and `Bangalore`, plus a legend and a span table. The Image tab behaves as before.

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "streamlit: split into Image + Text tabs; lazy per-tab model loading"
```

---

### Task 10: Documentation

Update `README.md` (and the model-card README in `scripts/build_hf_repo.py`) so the
text modality is documented as implemented rather than "planned".

**Files:**
- Modify: `README.md`
- Modify: `scripts/build_hf_repo.py` (the `README` string)

- [ ] **Step 1: Update `README.md` — Quickstart**

After the existing image Quickstart block (the line `redactor.redact("page.jpg", mode="blur").save("blurred.png")  # solid|blur|pixelate`) and before the line `\`detect()\` / \`redact()\` accept a file path...`, insert a new text subsection:

````markdown
### Text-only PII (plain strings)

```python
from eka_pii_redaction import TextPIIRedactor

r = TextPIIRedactor("ekacare/pii-redactors")     # GPU if available, else CPU

# 1) Character-span entities
for s in r.detect("John Doe, DOB 1990-01-01, john@x.com"):
    print(s.category, s.l1, s.start, s.end, s.text, s.score)

# 2) Redacted string (mask supports {category} / {l1} placeholders)
r.redact("Call John at john@x.com", mask="[REDACTED]")   # -> "Call [REDACTED] at [REDACTED]"
r.redact("Call John at john@x.com", mask="[{category}]")  # -> "Call [primary_subject_name] at [email]"
```

`TextPIIRedactor` runs a multilingual MiniLM token classifier on raw text — no OCR,
no image — and returns `TextPIISpan(category, start, end, l1, text, score)` with
**character** offsets.
````

- [ ] **Step 2: Update `README.md` — Structure section**

In the "Structure (by modality)" code block, change the `text/` lines from "planned" to current, and update the model-repo tree. Replace:

```
  text/                             # TEXT modality (planned)
    redactor.py  -> TextPIIRedactor  (redact PII inside plain-text blobs, no image)
```
with:
```
  text/                             # TEXT modality (implemented)
    redactor.py  -> TextPIIRedactor  (PII inside plain-text strings, no image)
    minilm.py    -> MiniLM token classifier (char-span detector)
```

And in the model-repo tree below it, replace:
```
  text/  …                          # reserved for the text-only model
```
with:
```
  text/  minilm/                    # multilingual MiniLM text-PII model
```

- [ ] **Step 3: Update `README.md` — remove the Roadmap note and update endpoints**

Replace the Roadmap blockquote:
```
> **Roadmap:** `eka_pii_redaction.text.TextPIIRedactor` will redact PII inside raw
> strings (character spans, no OCR), reusing the shared category taxonomy.
```
with:
```
Both modalities share the category taxonomy (`eka_pii_redaction.taxonomy`); the
text model also detects `mac_address` (device_net).
```

In the container "Endpoints" line, append the text endpoints. Replace:
```
Endpoints: `GET /health`, `GET /entities`, `POST /detect` (multipart `file`),
`POST /redact` (multipart `file`, `mode`). Env: `EKA_PII_HF_REPO`,
```
with:
```
Endpoints: `GET /health`, `GET /entities`, `GET /entities-text`,
`POST /detect` / `POST /redact` (multipart `file`), and
`POST /detect-text` / `POST /redact-text` (JSON `{"text": ...}`).
Env: `EKA_PII_HF_REPO`,
```

- [ ] **Step 4: Update the model-card README in `scripts/build_hf_repo.py`**

In `scripts/build_hf_repo.py`, in the `README` string, replace the line:
```
- `text/` — reserved for the future text-only PII model.
```
with:
```
- `text/minilm/` — multilingual MiniLM token classifier for **text PII in plain
  strings** (48 categories, char spans).
```

- [ ] **Step 5: Verify docs reference real symbols**

Run: `python -c "from eka_pii_redaction import TextPIIRedactor; print(TextPIIRedactor('x') if False else 'symbols ok')"`
Expected: prints `symbols ok` (import resolves; constructor not actually called).

- [ ] **Step 6: Commit**

```bash
git add README.md scripts/build_hf_repo.py
git commit -m "docs: document the text modality as implemented"
```

---

### Task 11: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire offline unit suite**

Run: `python -m pytest tests/ -v`
Expected: all unit tests PASS, the 2 integration tests SKIPPED (17 passed, 2 skipped).

- [ ] **Step 2: Optional — run the real-model integration test once**

Run: `RUN_TEXT_INTEGRATION=1 python -m pytest tests/test_text_integration.py -v`
Expected: 2 passed (downloads the text model on first run).

- [ ] **Step 3: Confirm the working tree is clean**

Run: `git status --short`
Expected: empty (everything committed).

---

## Self-Review notes

- **Spec coverage:** taxonomy `mac_address`+`TEXT_REDACTABLE` (T1); `MiniLMDetector` (T2–T3); `TextPIIRedactor` detect/redact/list_entities (T4); exports (T5); server `/detect-text`+`/redact-text` (T8); `sentencepiece` dep (T7); Streamlit two tabs + detection-only highlight (T9); README/build-script docs (T10). All spec sections map to a task.
- **Detection-only text tab:** T9 has no redaction controls in the Text tab, per the approved design; `redact()` still exists in the library (T4) and server (T8) for parity.
- **Type consistency:** `TextPIISpan(category, start, end, l1, text, score)` and `.to_dict()` are used identically across T2/T4/T8/T9; `apply_mask(text, spans, mask)` and `merge_bio_spans(text, offsets, labels, scores)` signatures match their call sites.
