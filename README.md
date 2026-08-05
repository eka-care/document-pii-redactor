# Eka-PII-redaction

Detect **PII in document images and plain text** — both **text** PII (names,
addresses, IDs, dates, phone/email, …) and **visual** entities (signatures,
stamps/seals, QR/barcodes, face photos, fingerprints, logos) — then **redact**,
**de-identify**, or **anonymize** it, behind a single class per modality.

- **Three modes, one detector.**
  **Redact** destroys the value (mask/black-out — nothing kept).
  **De-identify** replaces each entity with a consistent pseudonym
  (`Person_1`) and returns the entity→pseudonym mapping so an authorized
  key-holder can re-link later.
  **Anonymize** is one-way: ages become 10-year buckets, dates keep only the
  year, fine geography collapses, and names/IDs become unnumbered tokens —
  no mapping exists anywhere.
- **One install, one Hugging Face repo.** The package internally pulls the
  model weights it needs from one HF repo; you only ever reference one
  repo id.
- **Text + visual** in one call, or text-only if you don't need the detector.
- **CPU or GPU** — auto-selects CUDA if available, else CPU.
- **Choose what to process** — exclude any categories; everything is on by default.
- Returns a structured **entity list** (`text`, `bbox`, `category`, …) and/or a
  transformed image/string.

> Note: anonymization here is best-effort removal/generalization of
> *detected* identifiers. It is not a k-anonymity guarantee and not, by
> itself, a compliance determination.

## Install

From PyPI:

```bash
pip install eka-pii-redaction          # core library
pip install "eka-pii-redaction[server]"  # + FastAPI service
```

From source:

```bash
git clone https://github.com/eka-care/Eka-PII-redactors.git
cd Eka-PII-redactors
pip install -e .            # core library
pip install -e ".[server]"  # + FastAPI service
```

System dependency: **Tesseract OCR** — required for the **image** modality
(it OCRs the document before the text-in-image classifier runs on the
words); not needed for the text-only modality.

```bash
# Debian/Ubuntu
sudo apt-get install -y tesseract-ocr
# macOS
brew install tesseract
```

(The provided Docker image installs everything for you — see below.)

## Quickstart

**`detect()` is the core primitive** — it finds every PII entity with its
location, category, and confidence. The transforms (redact / anonymize /
de-identify) are consumers of its output: run detection once, feed the same
result into whichever transform you need.

### Detect — images

```python
from eka_pii_redaction import ImagePIIRedactor

# Loads the models from one HF repo; GPU if available, else CPU.
redactor = ImagePIIRedactor(
    "ekacare/pii-redactors",
    detect_visual=True,          # set False to skip visual entities (QR codes,
                                  # face photos, signatures, etc.) — text PII only
    # device="cpu",              # force CPU (default: auto)
    # exclude_entities=["logo", "brandname"],  # never detect these
)

entities = redactor.detect("page.jpg")
for e in entities:
    print(e.kind, e.category, e.bbox, e.text, e.score)
```

### Detect — plain text

```python
from eka_pii_redaction import TextPIIRedactor

r = TextPIIRedactor("ekacare/pii-redactors")     # GPU if available, else CPU

spans = r.detect("John Doe, DOB 1990-01-01, john@x.com")
for s in spans:
    print(s.category, s.l1, s.start, s.end, s.text, s.score)
```

`TextPIIRedactor` runs a lightweight multilingual token classifier on raw text —
no OCR, no image — and returns `TextPIISpan(category, start, end, l1, text, score)`
with **character** offsets. `detect()` (image) accepts a file path, raw
`bytes`, or a `PIL.Image`.

### Feed detections into transforms

Every transform takes a `detect()` result as its second argument (`entities`
for images, `spans` for text). Detection is always the explicit first step —
it runs once, and every transform consumes its output.

```python
# --- Image: one detection, three outputs ---
entities = redactor.detect("page.jpg")

redactor.redact("page.jpg", entities, mode="blur").save("redacted.png")
redactor.anonymize("page.jpg", entities).save("anonymized.png")
deid = redactor.deidentify("page.jpg", entities)   # ImageDeidResult
deid.image.save("deidentified.png")      # pseudonyms rendered in place;
                                          # faces/signatures become placeholders
deid.mapping.to_dict()                   # store securely to re-link later

# --- Text: same pattern ---
text = "Mr. John Doe, 45 yrs, DOB 12-03-1979, Indiranagar, Karnataka."
spans = r.detect(text)

r.redact(text, spans, mask="[{category}]")
r.anonymize(text, spans)
# -> "[PERSON], 40–49 yrs, DOB 1979, [LOCATION], Karnataka."  (no mapping exists)
result = r.deidentify(text, spans)
result.text      # -> "Person_1, Age_1 yrs, DOB Date_1, City_1, State_1."
result.mapping   # entity -> pseudonym map; pass mapping=result.mapping on the
                 # next page of the same record to keep numbering consistent
```

(Example outputs are illustrative — exact spans depend on the model's tagging
of the input.) De-identification consistency is per exact surface form
(`"John Doe"` and a later bare `"John"` get different pseudonyms).

## API

### ImagePIIRedactor

```python
ImagePIIRedactor(hf_repo, *, detect_visual=True, device=None,
                  exclude_entities=None, visual_score_threshold=0.25,
                  ocr_lang=None, cache_dir=None)
```

| arg | meaning |
|---|---|
| `hf_repo` | HF repo id **or** a local dir with `text_model/` + `visual_model/best.pt`. |
| `detect_visual` | If `False`, visual entities (QR codes, face photos, signatures, etc.) are **not** downloaded or loaded — text PII only. |
| `device` | `"cuda"` / `"cpu"`. `None` → auto (CUDA if available). |
| `exclude_entities` | Categories to never detect/redact. Default: none excluded (all on). |
| `visual_score_threshold` | Visual-entity confidence cutoff. |
| `ocr_lang` | Tesseract language/script, e.g. `"eng"`, `"eng+Devanagari"`. |

### detect

```python
detect(image, *, exclude_entities=None, ocr_lang=None) -> list[PIIEntity]
```

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

Returns a redacted copy. `mode` ∈ `solid` | `blur` | `pixelate`. All
transforms (`redact` / `anonymize` / `deidentify`, both modalities) take a
`detect()` result as their second argument — `entities` for images, `spans`
for text. Detection is the only step that runs the models; per-call
`exclude_entities` / `ocr_lang` therefore live on `detect()`.

### list_entities

```python
ImagePIIRedactor.list_entities() -> {"text": [...], "visual": [...]}
```

All selectable categories.

## Categories

- **Text (47):** person (name, age, gender, occupation, …), location (address,
  city, state, postcode, …), date_time, contact (phone, email, web_url, fax),
  uid (aadhaar, pan, passport, mrn/uhid, abha, insurance policy, bank/iban/upi, …),
  device_net, credential, and `brandname`.
- **Visual (6):** `signature`, `seal_stamp`, `qr_barcode`, `face_photo`,
  `fingerprint_thumb_impression`, `logo`.

`ImagePIIRedactor.list_entities()` returns the exact set.

## Run as a container

The image bundles torch+CUDA, Tesseract, the FastAPI service, and the React
demo UI (`web/`, built at image-build time and served by the same process at
`/`). Same image runs on GPU or CPU. This is also what runs on the
[`ekacare/pii-redactor-demo`](https://huggingface.co/spaces/ekacare/pii-redactor-demo)
HF Space.

```bash
docker build -t eka-pii-redaction .

# GPU
docker run --gpus all -p 7860:7860 \
  -e EKA_PII_HF_REPO=ekacare/pii-redactors \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  eka-pii-redaction

# CPU
docker run -e EKA_PII_DEVICE=cpu -p 7860:7860 eka-pii-redaction
```

Open `http://localhost:7860` for the UI. API endpoints: `GET /health`,
`GET /entities`, `GET /entities-text`,
`POST /detect` / `POST /redact` / `POST /anonymize` (multipart `file`,
optional `exclude` query param, `mode`/`color` form fields for `/redact`),
`POST /deidentify` (multipart `file` → JSON `{"image": <base64 png>,
"mapping": ...}`), and `POST /detect-text` / `POST /redact-text` /
`POST /deidentify-text` / `POST /anonymize-text`
(JSON `{"text": ..., "exclude": [...]}`; `/deidentify-text` also accepts
`"mapping"` from a prior call and returns the updated one).
Env: `EKA_PII_HF_REPO`, `EKA_PII_DETECT_VISUAL`, `EKA_PII_DEVICE`, `EKA_PII_EXCLUDE`.

```bash
curl -F file=@page.jpg http://localhost:7860/detect
curl -F file=@page.jpg -F mode=blur http://localhost:7860/redact -o redacted.png
```

## Structure (by modality)

The library is organized by **modality**, so additional models slot in cleanly:

```
eka_pii_redaction/
  taxonomy.py, entities.py          # shared
  image/                            # IMAGE modality (implemented)
    redactor.py  -> ImagePIIRedactor
    layoutlmv3.py -> text-PII-in-image detector
    yolo11m.py    -> visual-entity detector
  text/                             # TEXT modality (implemented)
    redactor.py  -> TextPIIRedactor  (PII inside plain-text strings, no image)
    minilm.py    -> text-PII token classifier (char-span detector)
```

The single model repo mirrors this:

```
<hf_repo>/
  image/ layoutlmv3/   yolo/best.pt
  text/  minilm/                    # multilingual text-PII model
```

Both modalities share the category taxonomy (`eka_pii_redaction.taxonomy`); the
text model also detects `mac_address` (device_net).

## Deploying the demo to HF Spaces

The Space ([`ekacare/pii-redactor-demo`](https://huggingface.co/spaces/ekacare/pii-redactor-demo))
runs this same repo's Docker image — see "Run as a container" above. To
(re)deploy, push the repo to the Space's git remote with HF's Space front
matter (`title` / `sdk: docker` / `app_port: 7860` / ...) prepended to
`README.md` for that push — the Space needs it to render its card, but it is
kept out of this tracked README because GitHub and PyPI would render it as
literal text. The Space reads the model via an `HF_TOKEN` repository secret
(Space → Settings → Repository secrets).

## How it works (image modality)

1. **Text-in-image:** Tesseract OCR (via the processor) → words + boxes →
   token classifier → per-word BIO labels → merged spans.
2. **Visual:** a detector over the page → boxes + categories.
3. **Redact:** fill / blur / pixelate every selected entity's box.
