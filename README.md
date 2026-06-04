# Eka-PII-redaction

Detect and redact **PII in document images** — both **text** PII (names, addresses,
IDs, dates, phone/email, …) and **visual** entities (signatures, stamps/seals,
QR/barcodes, face photos, fingerprints, logos) — behind a single class.

- **One install, one Hugging Face repo.** The package internally pulls two model
  weights (a LayoutLMv3 text classifier + a YOLO visual detector) from one HF
  repo; you only ever reference one repo id.
- **Text + visual** in one call, or text-only if you don't need the detector.
- **CPU or GPU** — auto-selects CUDA if available, else CPU.
- **Choose what to redact** — exclude any categories; everything is on by default.
- Returns a structured **entity list** (`text`, `bbox`, `category`, …) and/or a
  **redacted image**.

## Install

```bash
pip install eka-pii-redaction          # core library
pip install "eka-pii-redaction[server]"  # + FastAPI service
```

System dependency: **Tesseract OCR** (used for the text model).

```bash
# Debian/Ubuntu
sudo apt-get install -y tesseract-ocr
# macOS
brew install tesseract
```

(The provided Docker image installs everything for you — see below.)

## Quickstart

```python
from eka_pii_redaction import ImagePIIRedactor

# Loads both models from one HF repo; GPU if available, else CPU.
redactor = ImagePIIRedactor(
    "ekacare/pii-redactors",
    detect_visual=True,          # set False to skip YOLO download + visual detection
    # device="cpu",              # force CPU (default: auto)
    # exclude_entities=["logo", "brandname"],  # never redact these
)

# 1) Get the structured entity list
for e in redactor.detect("page.jpg"):
    print(e.kind, e.category, e.bbox, e.text, e.score)

# 2) Get a redacted image (PIL.Image)
redactor.redact("page.jpg").save("redacted.png")
redactor.redact("page.jpg", mode="blur").save("blurred.png")  # solid|blur|pixelate
```

`detect()` / `redact()` accept a file path, raw `bytes`, or a `PIL.Image`.

## API

### `ImagePIIRedactor(hf_repo, *, detect_visual=True, device=None, exclude_entities=None, visual_score_threshold=0.25, ocr_lang=None, cache_dir=None)`

| arg | meaning |
|---|---|
| `hf_repo` | HF repo id **or** a local dir with `text_model/` + `visual_model/best.pt`. |
| `detect_visual` | If `False`, YOLO weights are **not** downloaded or loaded — text PII only. |
| `device` | `"cuda"` / `"cpu"`. `None` → auto (CUDA if available). |
| `exclude_entities` | Categories to never detect/redact. Default: none excluded (all on). |
| `visual_score_threshold` | YOLO confidence cutoff. |
| `ocr_lang` | Tesseract language/script, e.g. `"eng"`, `"eng+Devanagari"`. |

### `detect(image, *, exclude_entities=None, ocr_lang=None) -> list[PIIEntity]`

Each `PIIEntity` has:

| field | description |
|---|---|
| `category` | fine category, e.g. `primary_subject_name`, `signature` |
| `kind` | `"text"` or `"visual"` |
| `bbox` | `[x0, y0, x1, y1]` in original-image **pixels** |
| `l1` | coarse group: `person`/`location`/`contact`/`uid`/… or `biometric_visual` |
| `text` | OCR text (text entities) or `None` (visual) |
| `score` | confidence in `[0,1]` |

### `redact(image, *, mode="solid", color=(0,0,0), exclude_entities=None, ocr_lang=None, pad=2) -> PIL.Image`

Returns a redacted copy. `mode` ∈ `solid` | `blur` | `pixelate`.

### `ImagePIIRedactor.list_entities() -> {"text": [...], "visual": [...]}`

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

The image bundles torch+CUDA, Tesseract, and the FastAPI service. Same image runs
on GPU or CPU.

```bash
docker build -t eka-pii-redaction .

# GPU
docker run --gpus all -p 8080:8080 \
  -e EKA_PII_HF_REPO=ekacare/pii-redactors \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  eka-pii-redaction

# CPU
docker run -e EKA_PII_DEVICE=cpu -p 8080:8080 eka-pii-redaction
```

Endpoints: `GET /health`, `GET /entities`, `POST /detect` (multipart `file`),
`POST /redact` (multipart `file`, `mode`). Env: `EKA_PII_HF_REPO`,
`EKA_PII_DETECT_VISUAL`, `EKA_PII_DEVICE`, `EKA_PII_EXCLUDE`.

```bash
curl -F file=@page.jpg http://localhost:8080/detect
curl -F file=@page.jpg -F mode=blur http://localhost:8080/redact -o redacted.png
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
  text/                             # TEXT modality (planned)
    redactor.py  -> TextPIIRedactor  (redact PII inside plain-text blobs, no image)
```

The single model repo mirrors this:

```
<hf_repo>/
  image/ layoutlmv3/   yolo/best.pt
  text/  …                          # reserved for the text-only model
```

> **Roadmap:** `eka_pii_redaction.text.TextPIIRedactor` will redact PII inside raw
> strings (character spans, no OCR), reusing the shared category taxonomy.

## Publishing the model weights

The trained checkpoints are assembled into the single HF repo with:

```bash
python scripts/build_hf_repo.py \
  --layoutlmv3   .../checkpoints/base_v3_combined_4ep/final \
  --yolo-weights .../checkpoints/visual/visual_yolo11m/weights/best.pt \
  --out /tmp/eka-pii-hf --push --repo-id ekacare/pii-redactors
```

## How it works (image modality)

1. **Text-in-image (LayoutLMv3):** Tesseract OCR (via the processor) → words +
   boxes → LayoutLMv3 token classifier → per-word BIO labels → merged spans.
2. **Visual (YOLO):** detector over the page → boxes + categories.
3. **Redact:** fill / blur / pixelate every selected entity's box.
