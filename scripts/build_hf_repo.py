"""Assemble the single Hugging Face repo from local training checkpoints, and
optionally push it to the Hub. Weights are organized BY MODALITY.

Layout produced:
    <out>/
        image/
            layoutlmv3/      # contents of the LayoutLMv3 'final' dir
            yolo/best.pt     # YOLO visual detector
        README.md
        # text/  -> reserved for the future text-only model

Usage:
    python scripts/build_hf_repo.py \
        --layoutlmv3   /path/to/checkpoints/base_v3_combined_4ep/final \
        --yolo-weights /path/to/checkpoints/visual/visual_yolo11m/weights/best.pt \
        --out /tmp/eka-pii-hf \
        [--push --repo-id ekacare/pii-redactors]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

README = """---
license: apache-2.0
tags: [pii, redaction, layoutlmv3, yolo, document-ai]
---

# eka-pii-redaction model weights

Single repo holding the models used by the
[`eka-pii-redaction`](https://github.com/eka-care/Eka-PII-redactors) library,
organized by modality:

- `image/layoutlmv3/` — LayoutLMv3 token classifier for **text PII in images**
  (47 L2 classes), run on Tesseract OCR words.
- `image/yolo/best.pt` — YOLO detector for **visual entities** (signature,
  seal/stamp, QR/barcode, face photo, fingerprint, logo).
- `text/` — reserved for the future text-only PII model.

## Usage

```python
from eka_pii_redaction import ImagePIIRedactor
r = ImagePIIRedactor("ekacare/pii-redactors", detect_visual=True)
entities = r.detect("page.jpg")
redacted = r.redact("page.jpg")
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layoutlmv3", required=True, help="LayoutLMv3 'final' dir")
    ap.add_argument("--yolo-weights", required=True, help="YOLO best.pt")
    ap.add_argument("--out", required=True, help="output dir for the HF repo layout")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true", help="create the HF repo as private")
    ap.add_argument("--repo-id", default="ekacare/pii-redactors")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "image" / "layoutlmv3").mkdir(parents=True, exist_ok=True)
    (out / "image" / "yolo").mkdir(parents=True, exist_ok=True)

    # layoutlmv3: copy the whole 'final' dir contents
    for f in Path(args.layoutlmv3).iterdir():
        if f.is_file():
            shutil.copy2(f, out / "image" / "layoutlmv3" / f.name)
    # yolo weights
    shutil.copy2(args.yolo_weights, out / "image" / "yolo" / "best.pt")
    (out / "README.md").write_text(README)

    print(f"assembled HF repo layout at {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({p.stat().st_size/1e6:.1f} MB)")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True,
                        private=args.private)
        api.upload_folder(folder_path=str(out), repo_id=args.repo_id, repo_type="model")
        vis = "private" if args.private else "public"
        print(f"\npushed ({vis}) to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
