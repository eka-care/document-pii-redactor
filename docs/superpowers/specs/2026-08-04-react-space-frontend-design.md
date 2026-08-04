# React Frontend for HF Space Demo — Design

Date: 2026-08-04

## Goal

Host `ekacare/pii-redactors` on a Hugging Face Space (`ekacare/pii-redactor-demo`)
behind a React frontend, covering both modalities (image redaction and text
redaction), matching the feature depth of the existing Streamlit tester
(`streamlit_app.py`, branch `add-streamlit-tester`). Model repo is currently
private; will go public later.

## Architecture

One HF Space, Docker SDK, container listens on **7860** (HF's conventional
Spaces port). The existing `Dockerfile` becomes multi-stage:

1. **Node stage** — `npm ci && npm run build` inside `web/`, producing a static
   build.
2. **Python stage** — unchanged base (`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`,
   tesseract, libgl), `pip install ".[server]"`, then
   `COPY --from=builder web/dist ./web/dist`.

`eka_pii_redaction/server.py` gets one addition: after the existing API routes,
`app.mount("/", StaticFiles(directory="web/dist", html=True))`, so one FastAPI
process serves both the React app and the API — no nginx, no second process, no
CORS (same origin).

Space config: `EKA_PII_HF_REPO=ekacare/pii-redactors`, `EKA_PII_DEVICE=cpu`
(pins the free CPU tier), `HF_TOKEN` as a Space repository secret for the
private model pull. HF Space metadata (YAML front matter prepended to the top
of the existing root `README.md`, which HF reads for the Space card while the
rest of the file still serves as the library README) sets `sdk: docker`,
`app_port: 7860`.

## Components

`web/` — a Vite + React (TS) SPA:

- `App.tsx` — two-tab layout (Image / Text).
- `lib/api.ts` — fetch wrapper: `getEntities()`, `getTextEntities()`,
  `detect(file)`, `redact(file, mode, color)`, `detectText(text)` — one per
  backend endpoint used.
- `lib/colors.ts` — `L1_COLORS` mapping ported from `streamlit_app.py`, shared
  by both tabs so the two UIs (Streamlit tester, React demo) stay visually
  consistent.
- `components/ImageTab.tsx` — upload dropzone + preset sample thumbnails
  (samples in `web/public/samples/`, provided by the user — no real PII),
  redact-mode select (solid/blur/pixelate), color picker, exclude-categories
  multiselect (from `GET /entities`), results: canvas box overlay next to the
  redacted image, entity data table, PNG download.
- `components/TextTab.tsx` — textarea (preloaded example text, matching
  Streamlit's), exclude-categories multiselect (from `GET /entities-text`),
  "Detect PII" button, client-side highlighted output + legend, span data
  table. Only calls `/detect-text` — no masked-text view, matching the
  Streamlit tester (which never calls `/redact-text`).
- `components/Legend.tsx` — shared color-coded l1-group legend.

Exclude-category lists come from the backend (not hardcoded), keeping the
taxonomy single-sourced from `taxonomy.py`.

## Data flow

- **On load:** parallel `GET /entities` + `GET /entities-text` populate the
  two multiselects. A backing-off `GET /health` poll gates the
  upload/submit controls, showing a "waking up the model…" state instead of
  letting users submit into a cold-start 503.
- **Image tab:** upload or pick a sample → `POST /detect` (multipart) for the
  entity list + `POST /redact` (multipart, `mode`/`color`) for the redacted
  PNG, both against the same bytes. Canvas overlay draws boxes from `/detect`
  over the original image; `/redact`'s PNG renders as the "after" image.
- **Text tab:** submit → `POST /detect-text` (JSON) → spans → highlighting
  computed client-side from returned char offsets (same approach as
  `highlight_spans()` in `streamlit_app.py`).

## Error handling

- Backend 5xx / network failure → inline error banner scoped to the active
  tab; the other tab stays usable.
- Cold start (models load at FastAPI startup via the existing `_warm()` hook)
  → `/health` poll keeps controls disabled with a "waking up" message.
- Client-side validation: file type restricted to
  `png/jpg/jpeg/bmp/tiff/webp` (matches Streamlit); submit disabled on empty
  text (matches Streamlit's `disabled=not text_in.strip()`).

## Testing

No automated test suite for this demo UI (YAGNI — its correctness is "does it
look right and work in a browser"). Verification before calling it done:
`docker build` locally, run the container, `curl` each existing endpoint to
confirm the static mount didn't break API routing, then click through both
tabs in a real browser with the sample images and example text, including
empty/error states (bad file type, empty text, backend briefly down).

## Deployment

Space creation and secret configuration require the user's HF account — not
done by the agent. After the code is ready: create the Space
(`ekacare/pii-redactor-demo`, Docker SDK) via the HF UI or `huggingface-cli`,
add `HF_TOKEN` under Settings → Repository secrets, add the Space as a git
remote and push this repo (or the relevant state of it) to it.
