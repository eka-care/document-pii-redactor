"""YOLO detector — finds visual entities *within document images*.

Detects the 6 image-region categories (signature, seal_stamp, qr_barcode,
face_photo, fingerprint_thumb_impression, logo). Loaded lazily so that
`detect_visual=False` users never pay the ultralytics import / weight download.

Default backbone is YOLO11m; the loader is weight-agnostic, so a smaller/larger
YOLO can be swapped in by replacing `image/yolo/best.pt` in the model repo.
"""
from __future__ import annotations

from PIL import Image

from ..entities import PIIEntity
from ..taxonomy import l1_group


def require_ultralytics():
    """Return the ultralytics YOLO class, or raise a helpful ImportError.

    Called both here and at `ImagePIIRedactor` construction so a missing
    extra fails fast — before any weights are downloaded."""
    try:
        from ultralytics import YOLO  # lazy import — AGPL-3.0, optional extra
    except ImportError as exc:
        raise ImportError(
            "Visual-entity detection needs the 'visual' extra: "
            "pip install 'document-pii-redactor[visual]'. Note that the "
            "ultralytics dependency and the visual detector weights are "
            "AGPL-3.0 licensed (see the README's license section); use "
            "detect_visual=False for the permissively-licensed text-only "
            "pipeline."
        ) from exc
    return YOLO


class YOLODetector:
    def __init__(self, weights_path: str, device: str, score_threshold: float = 0.25):
        YOLO = require_ultralytics()

        self.model = YOLO(weights_path)
        # ultralytics device: 0 for the first CUDA GPU, else "cpu".
        self.device = 0 if str(device).startswith("cuda") else "cpu"
        self.score_threshold = score_threshold
        self.names = self.model.names  # {class_id: category_name}

    def detect(self, image: Image.Image, score_threshold: float | None = None
               ) -> list[PIIEntity]:
        conf = self.score_threshold if score_threshold is None else score_threshold
        res = self.model.predict(image, conf=conf, device=self.device, verbose=False)[0]
        out: list[PIIEntity] = []
        for b in res.boxes:
            x0, y0, x1, y1 = (int(v) for v in b.xyxy[0].tolist())
            cat = self.names[int(b.cls[0])]
            out.append(PIIEntity(
                category=cat, kind="visual", bbox=(x0, y0, x1, y1),
                l1=l1_group(cat), text=None, score=round(float(b.conf[0]), 4),
            ))
        return out
