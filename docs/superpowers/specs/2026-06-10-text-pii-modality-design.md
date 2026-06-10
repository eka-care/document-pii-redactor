# Text PII modality + Streamlit tabs — design

Date: 2026-06-10
Branch: `add-streamlit-tester`

## Goal

Add the **text** PII modality (`TextPIIRedactor`) as a first-class counterpart to
the existing **image** modality (`ImagePIIRedactor`), wired into every surface the
image modality already touches: the package exports, the FastAPI server, the
Streamlit tester (as a dedicated tab), and the README. Replace the existing
`eka_pii_redaction/text/redactor.py` stub (currently raises `NotImplementedError`)
with a real implementation.

## Model facts (verified against `ekacare/pii-redactors`)

The text model lives at `text/minilm/` in the same single HF repo used by the
image modality:

```
text/minilm/
  config.json            # BertForTokenClassification, hidden_size 384, 97 labels
  labels.json            # the 97 BIO labels
  model.safetensors
  tokenizer.json         # -> loads as XLMRobertaTokenizerFast (fast, has offsets)
  sentencepiece.bpe.model
  special_tokens_map.json
  tokenizer_config.json  # tokenizer_class: XLMRobertaTokenizer
  eval_report.json       # overall F1 ~0.74; multilingual (en + 11 Indian langs)
```

- It is a **plain-text BIO NER tagger** — no OCR, no image. Operates on raw
  strings and yields **character spans**.
- The fast tokenizer supports `return_offsets_mapping=True`, giving exact
  character offsets back into the original string. Verified: loads and predicts
  correctly even on transformers 4.x (only the image LayoutLMv3 processor needs 5.x).
- Labels are **uppercase BIO** (`B-AADHAAR_12_DIGIT`). Lowercasing `label[2:]`
  matches the taxonomy category keys exactly.
- It predicts **48** entity types = the image model's 47 text categories **plus
  `mac_address`**, which is not yet in `taxonomy.py`.

## Architecture — mirror the image modality 1:1

The image side is `redactor.py` (orchestrator) + `layoutlmv3.py` (model wrapper).
We mirror this exactly for text rather than building one combined redactor, so each
modality stays independently loadable and the `text/` package shape matches `image/`.

Rejected alternative: a single `PIIRedactor` that does both text+image. Rejected
because the two modalities have different inputs (str vs image), different outputs
(char spans vs pixel bboxes), and different weights; the repo is already organized
by modality and the README documents that intent.

### Files

| File | Change |
|---|---|
| `eka_pii_redaction/taxonomy.py` | Add `mac_address -> device_net` to `TEXT_L2_TO_L1`. Add `TEXT_REDACTABLE` = the text-only category list for the text API's `list_entities()` (currently identical to `TEXT_ENTITIES`). |
| `eka_pii_redaction/text/minilm.py` | **New** `MiniLMDetector(model_dir, device)`. Loads `AutoModelForTokenClassification` + `AutoTokenizer(use_fast=True)`. `detect(text)` tokenizes with `return_offsets_mapping=True` + overflow/stride chunking for long docs, runs the model, maps subword predictions back to character spans, merges BIO into `TextPIISpan`s. Mirrors `LayoutLMv3Detector`. |
| `eka_pii_redaction/text/redactor.py` | Replace the stub with real `TextPIIRedactor(hf_repo=DEFAULT_HF_REPO, *, device=None, exclude_entities=None, cache_dir=None)`. Methods: `detect(text, *, exclude_entities=None)`, `redact(text, *, mask="[REDACTED]", exclude_entities=None)`, classmethod `list_entities()`. Keep the existing `TextPIISpan` dataclass. |
| `eka_pii_redaction/text/__init__.py` | Export `TextPIIRedactor`, `TextPIISpan`, `DEFAULT_HF_REPO`. |
| `eka_pii_redaction/__init__.py` | Re-export `TextPIIRedactor`, `TextPIISpan` at top level. |
| `eka_pii_redaction/server.py` | Add `POST /detect-text` and `POST /redact-text`, backed by a lazily-built shared `TextPIIRedactor`. Add `/entities` for text or extend the existing one. |
| `pyproject.toml` | Add `sentencepiece` (XLM-R tokenizer dependency). |
| `streamlit_app.py` | Restructure into `st.tabs(["Image", "Text"])`; lazy per-tab model loading. |
| `README.md` + `scripts/build_hf_repo.py` README block | Move text modality from "planned/roadmap" to documented + usage. |

### MiniLMDetector detection algorithm

1. Tokenize `text` with `return_offsets_mapping=True`, `return_overflowing_tokens=True`,
   `truncation=True`, `padding="max_length"`, `max_length=512`, `stride=128`,
   `return_tensors="pt"`. Each chunk's offset_mapping references original-string
   char positions.
2. For each chunk: softmax logits -> per-token argmax label + max-prob score.
3. Collect non-special (`offset end > start`), non-`O` token predictions as
   `(char_start, char_end, category, bio_tag, score)`. Dedup tokens that recur in
   the stride-overlap region (key by `(char_start, char_end)`, first wins).
4. Sort by `char_start`; merge BIO into spans: start a new span on `B-`/category
   change/gap, extend on `I-` of the same category, unioning char ranges and
   collecting scores.
5. Trim leading/trailing whitespace inside each span's char range (sentencepiece
   offsets include the leading space). Emit `TextPIISpan(category, start, end, l1,
   text=text[start:end], score=mean(scores))`.

### TextPIIRedactor.redact

Replace each detected span `[start:end]` with `mask`, walking spans right-to-left
(or splicing sorted, non-overlapping spans) so offsets stay valid. `mask` may
contain `{category}` and/or `{l1}` placeholders, formatted per span (default the
literal `"[REDACTED]"`). `exclude_entities` is unioned with the constructor's.

### Server endpoints

- `POST /detect-text` — body `{"text": "..."}` (or form field) -> `{"spans": [...]}`.
- `POST /redact-text` — body `{"text": "...", "mask": "[REDACTED]"}` -> `{"text": "<redacted>"}`.
- Lazy singleton `TextPIIRedactor`, configured from the same env vars
  (`EKA_PII_HF_REPO`, `EKA_PII_DEVICE`, `EKA_PII_EXCLUDE`).

### Streamlit text tab (detection-only)

- `st.tabs(["🖼️ Image", "📝 Text"])`. Each tab loads only its own model, on its
  run button click, via a separate `@st.cache_resource` loader — image-only users
  never download the text model, and vice-versa.
- Image tab: existing detect/redact flow, behavior unchanged.
- Text tab: `st.text_area` input -> "Detect" -> render the input with each PII span
  wrapped in a colored `<mark>` (background = `L1_COLORS` group color, via
  `st.markdown(unsafe_allow_html=True)`, with the surrounding text HTML-escaped),
  plus a color legend and the entity table (`category, l1, text, start, end, score`).
  No redaction controls in the UI.

## Out of scope / explicit notes

- The image LayoutLMv3 model will now have `mac_address` as a *selectable* exclude
  category that it never actually emits — harmless, flagged intentionally.
- No retraining or model changes.
- `device_net` already has a color in the Streamlit `L1_COLORS` map.

## Success criteria

- `from eka_pii_redaction import TextPIIRedactor` works; `detect()` returns
  character-accurate `TextPIISpan`s on multilingual input; `redact()` returns a
  masked string.
- Streamlit app shows two tabs; text tab highlights detected spans with per-group
  colors; image tab unchanged.
- Server exposes `/detect-text` + `/redact-text`.
- README documents the text modality as implemented.
