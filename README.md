<div align="center">

# document-pii-redactor

**Detect PII in document images and plain text — then redact, anonymize, or de-identify it.**
Built for Indian documents, light enough to run on CPU.

[![PyPI](https://img.shields.io/pypi/v/document-pii-redactor?color=2f6feb)](https://pypi.org/project/document-pii-redactor/)
[![Python](https://img.shields.io/pypi/pyversions/document-pii-redactor?color=2f6feb)](https://pypi.org/project/document-pii-redactor/)
[![License](https://img.shields.io/badge/license-Apache--2.0-2f6feb)](#license--citation)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-live_demo-ffd21e)](https://huggingface.co/spaces/ekacare/document-pii-redactor)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-model_weights-ffd21e)](https://huggingface.co/ekacare/document-pii-redactor)

</div>

Most PII redactors stop at plain text. This one also handles **document
images** — both the text in them (names, addresses, IDs, dates, phones, …)
and **visual entities** (signatures, stamps, QR codes, face photos,
fingerprints). The models are trained on **Indian names, documents, and
contexts**, and the text model works across **Indian languages**. The main
contribution is the **PII token classifier** — OCR is just the pluggable
input stage in front of it. It defaults to lightweight **Tesseract**, which
works well for PDFs and good-quality images; for more difficult or blurred
images,
[**Bring-your-own OCR**](#quickstart) lets a model like [**Nemotron OCR**](https://huggingface.co/nvidia/nemotron-ocr-v2)
(or Textract, Google Vision, etc) plug straight in for better results
([example notebook](examples/byo_ocr_nemotron.ipynb)).
## What it does

`detect()` is the core primitive — it finds every PII entity with its
location, category, and confidence. The three transforms consume its output:

| action | result | reversible? |
|---|---|---|
| `detect()` | structured entity list (category, location, text, confidence) | — |
| `redact()` | destroy — mask text, or black-out / blur / pixelate image regions | no |
| `anonymize()` | generalize — age → 10-year bucket, dates → year, fine geography → `[LOCATION]`, everything else → unnumbered tokens | no |
| `deidentify()` | consistent pseudonyms — sequential `Person_1`, or globally deterministic hash tokens `Person_a3f9c1` (a.k.a. PII tokenization) — plus the entity→pseudonym mapping | via the returned mapping |

> Anonymization is best-effort removal/generalization of *detected*
> identifiers — not a k-anonymity guarantee or a compliance determination.

## Install

```bash
pip install "document-pii-redactor[visual]"  # full pipeline, used by the quickstart below (AGPL-3.0 — see License)
pip install document-pii-redactor            # text + OCR pipelines only, no visual entities (permissive licenses)
pip install "document-pii-redactor[server]"  # + FastAPI service
```

Tesseract (`apt-get install tesseract-ocr` / `brew install tesseract`) is
needed only for the built-in image OCR — not for text-only use or
bring-your-own-OCR.

## Quickstart

Load once, detect once — every transform consumes the `detect()` result:

```python
from document_pii_redactor import ImagePIIRedactor, TextPIIRedactor

image_redactor = ImagePIIRedactor("ekacare/document-pii-redactor")
entities = image_redactor.detect("report.png")              # built-in Tesseract OCR

# …or bring your own OCR — pass words + pixel boxes, Tesseract is skipped
# and your exact boxes come back on the detected entities:
entities = image_redactor.detect("report.png", words=["John", "Doe"],
                                 boxes=[[100, 20, 140, 40], [145, 20, 180, 40]])

text_redactor = TextPIIRedactor("ekacare/document-pii-redactor")
text  = "Mr. John Doe, 45 yrs, DOB 12-03-1979, Indiranagar, Bangalore. Contact: +91 98765 43210."
spans = text_redactor.detect(text)
```

**Redact** — destroy:

```python
image_redactor.redact("report.png", entities, mode="blur").save("redacted.png")  # or "solid" / "pixelate"

text_redactor.redact(text, spans)
# '[REDACTED], [REDACTED] yrs, DOB [REDACTED], [REDACTED], [REDACTED]. Contact: [REDACTED].'
```

**Anonymize** — generalize, one-way, no mapping kept. Ages become 10-year
buckets, dates keep only the year, fine geography collapses to `[LOCATION]`
(state and country survive), everything else becomes an unnumbered token:

```python
image_redactor.anonymize("report.png", entities).save("anonymized.png")

text_redactor.anonymize(text, spans)
# '[PERSON], 40–49 yrs, DOB 1979, [LOCATION], [LOCATION]. Contact: [PHONE].'
```

**De-identify** — pseudonymize. Same value → same pseudonym throughout the
document, and the entity→pseudonym mapping comes back for authorized
re-linking. `strategy="hash"` gives tokens that stay stable across documents
with no mapping to thread (`secret=` salts the hash):

```python
deid = image_redactor.deidentify("report.png", entities)    # .image + .mapping
deid.image.save("deidentified.png")

text_redactor.deidentify(text, spans).text
# 'Person_1, Age_1 yrs, DOB Date_1, City_1, City_2. Contact: Phone_1.'

text_redactor.deidentify(text, spans, strategy="hash").text
# 'Person_539681, Age_6c8349 yrs, DOB Date_7f19c4, City_d12704, City_60c7d5. Contact: Phone_d57003.'
```

Good to know:

- Transforms **require** the `detect()` result as their second argument —
  detection is always the explicit first step, and runs the models exactly once.
- `categories=[...]` on `detect()` limits which of the 53 PII categories are
  found (default: all).
- Sequential pseudonyms are scoped to the returned `mapping` — pass
  `mapping=result.mapping` on the next page of the same record to keep
  numbering consistent. Hash tokens need no threading.

For a runnable end-to-end walkthrough, see
[`examples/quickstart.ipynb`](examples/quickstart.ipynb) — and
[`examples/byo_ocr_nemotron.ipynb`](examples/byo_ocr_nemotron.ipynb) for
plugging in [Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2)
as the OCR for difficult or blurred scans.

<details>
<summary><b>API reference</b></summary>

### ImagePIIRedactor

```python
ImagePIIRedactor(hf_repo, *, detect_visual=True, device=None,
                  categories=None, visual_score_threshold=0.25,
                  ocr_lang=None, cache_dir=None)
```

| arg | meaning |
|---|---|
| `hf_repo` | HF repo id **or** a local dir with the model layout below. |
| `detect_visual` | If `False`, visual entities (QR codes, face photos, signatures, etc.) are **not** downloaded or loaded — text PII only. |
| `device` | `"cuda"` / `"cpu"`. `None` → auto (CUDA if available). |
| `categories` | The categories to detect. Default `None` = all of them. |
| `visual_score_threshold` | Visual-entity confidence cutoff. |
| `ocr_lang` | Tesseract language/script, e.g. `"eng"`, `"eng+Devanagari"`. |

### detect

```python
detect(image, *, categories=None, ocr_lang=None,
       words=None, boxes=None) -> list[PIIEntity]
```

`words` + `boxes` (pixel-coordinate word boxes, passed together) bring your
own OCR: the Tesseract step is skipped and your boxes pass through to the
entities. Without them the built-in Tesseract path runs, honoring `ocr_lang`.

Each `PIIEntity` has:

| field | description |
|---|---|
| `category` | fine category, e.g. `primary_subject_name`, `signature` |
| `kind` | `"text"` or `"visual"` |
| `bbox` | `[x0, y0, x1, y1]` in original-image **pixels** |
| `l1` | coarse group: `person`/`location`/`contact`/`uid`/… or `biometric_visual` |
| `text` | OCR text (text entities) or `None` (visual) |
| `score` | confidence in `[0,1]` |

### redact

```python
redact(image, entities, *, mode="solid", color=(0,0,0), pad=2) -> PIL.Image
```

`mode` ∈ `solid` | `blur` | `pixelate`. All transforms (both modalities)
take a `detect()` result as their second argument — `entities` for images,
`spans` for text.

### deidentify

```python
deidentify(image_or_text, entities_or_spans, *,
           mapping=None, strategy="counter", secret=None)
```

`strategy="counter"` (default) mints sequential `Person_1` pseudonyms scoped
to `mapping`; `strategy="hash"` mints globally deterministic `Person_a3f9c1`
tokens (md5 of the normalized value, salted with `secret`). Returns a result
with `.mapping` either way. Text also has `redact(text, spans, mask=...)`
with `{category}` / `{l1}` placeholders, and `anonymize(text, spans)`.

### list_entities

```python
ImagePIIRedactor.list_entities() -> {"text": [...], "visual": [...]}
TextPIIRedactor.list_entities()  -> [...]
```

</details>

<details>
<summary><b>PII categories (47 text + 6 visual)</b></summary>

- **Text (47):** person (name, age, gender, occupation, …), location
  (address, city, state, postcode, …), date_time, contact (phone, email,
  web_url, fax), uid (aadhaar, pan, passport, mrn/uhid, abha, insurance
  policy, bank/iban/upi, …), brandname, device_net, and credential
- **Visual (6):** `signature` · `seal_stamp` · `qr_barcode` · `face_photo` ·
  `fingerprint_thumb_impression` · `logo`

`list_entities()` returns the exact set.

</details>

<details>
<summary><b>Self-hosting — Docker container & HF Space</b></summary>

The Docker image bundles torch, Tesseract, the FastAPI service, and the
React demo UI (served at `/`). Same image runs on GPU or CPU; it is also
what runs on the [demo Space](https://huggingface.co/spaces/ekacare/document-pii-redactor).

```bash
docker build -t document-pii-redactor .

# GPU
docker run --gpus all -p 7860:7860 \
  -e EKA_PII_HF_REPO=ekacare/document-pii-redactor \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  document-pii-redactor

# CPU
docker run -e EKA_PII_DEVICE=cpu -p 7860:7860 document-pii-redactor
```

Open `http://localhost:7860` for the UI.

**Endpoints** — image (multipart `file`, optional `categories` query param,
comma-separated, absent = all): `POST /detect`, `POST /redact`
(`mode`/`color` form fields), `POST /anonymize`, `POST /deidentify`
(also `strategy`/`secret` query params; returns JSON
`{"image": <base64 png>, "mapping": ...}`). Text (JSON
`{"text": ..., "categories": [...]}`): `POST /detect-text`,
`POST /redact-text`, `POST /anonymize-text`, `POST /deidentify-text`
(also accepts `"mapping"`, `"strategy"`, `"secret"`). Plus `GET /health`,
`GET /entities`, `GET /entities-text`.

**Env**: `EKA_PII_HF_REPO`, `EKA_PII_DETECT_VISUAL`, `EKA_PII_DEVICE`,
`EKA_PII_CATEGORIES`.

```bash
curl -F file=@page.jpg http://localhost:7860/detect
curl -F file=@page.jpg -F mode=blur http://localhost:7860/redact -o redacted.png
```

**Deploying the demo Space**: push the repo to the Space's git remote with
HF's Space front matter (`title` / `sdk: docker` / `app_port: 7860` / …)
prepended to `README.md` for that push — the Space needs it to render its
card, but GitHub and PyPI would show it as literal text. The Space reads the
model via an `HF_TOKEN` repository secret.

</details>

<details>
<summary><b>How it works & repo structure</b></summary>

Image modality:

1. **Text-in-image:** Tesseract OCR — or your own OCR's words + boxes — →
   token classifier → per-word BIO labels → merged spans.
2. **Visual:** a detector over the page → boxes + categories.
3. **Transforms:** fill / blur / pixelate boxes (redact), or erase-and-render
   replacement values in place (de-identify / anonymize).

The library is organized by **modality**, so additional models slot in
cleanly:

```
document_pii_redactor/
  taxonomy.py, entities.py          # shared
  image/                            # IMAGE modality
    redactor.py  -> ImagePIIRedactor
    layoutlmv3.py -> text-PII-in-image detector
    yolo11m.py    -> visual-entity detector
  text/                             # TEXT modality
    redactor.py  -> TextPIIRedactor  (PII inside plain-text strings, no image)
    minilm.py    -> text-PII token classifier (char-span detector)
```

The model repo mirrors this: `image/layoutlmv3/`, `image/yolo/best.pt`,
`text/minilm/`. Both modalities share the category taxonomy
(`document_pii_redactor.taxonomy`).

</details>

## License & citation

**Code**: [Apache-2.0](LICENSE) (keep the [NOTICE](NOTICE) attribution when
redistributing). The optional `[visual]` extra installs
[ultralytics](https://github.com/ultralytics/ultralytics) (**AGPL-3.0**).
**Model weights** are licensed per model, following each base's license —
the text pipeline is fully permissive; the image models inherit
restrictions:

| weights | fine-tuned from | license |
|---|---|---|
| `text/minilm/` (plain-text PII) | Multilingual MiniLM (MIT) | **CC-BY-4.0** — free use incl. commercial, credit Eka Care |
| `image/layoutlmv3/` (text-in-image PII) | `microsoft/layoutlmv3-base` | **CC-BY-NC-SA-4.0** — **non-commercial only**, ShareAlike |
| `image/yolo/best.pt` (visual entities) | YOLO11m (Ultralytics) | **AGPL-3.0** |

If you use this library or the models, please cite us:

```bibtex
@software{document_pii_redactor,
  author = {{Eka Care}},
  title  = {document-pii-redactor: detect, redact, de-identify, or anonymize
            PII in document images and plain text},
  year   = {2026},
  url    = {https://github.com/eka-care/document-pii-redactor},
  note   = {Model weights: https://huggingface.co/ekacare/document-pii-redactor}
}
```
