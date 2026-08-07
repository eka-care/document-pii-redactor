import os

import pytest

from eka_pii_redaction.image.layoutlmv3 import normalize_boxes_to_1000
from eka_pii_redaction.image.redactor import ImagePIIRedactor


# ---------------------------------------------------- pure logic (no model) --- #
def test_normalize_boxes_scales_to_1000_space():
    # A box spanning the full image maps to the full 0..1000 range.
    assert normalize_boxes_to_1000([[0, 0, 200, 100]], 200, 100) == [[0, 0, 1000, 1000]]
    # Half-way points land at 500.
    assert normalize_boxes_to_1000([[50, 25, 100, 50]], 200, 100) == [[250, 250, 500, 500]]


def test_normalize_boxes_clamps_overshoot():
    # OCR engines sometimes emit boxes a pixel past the edge; clamp, don't crash.
    assert normalize_boxes_to_1000([[-3, 0, 205, 101]], 200, 100) == [[0, 0, 1000, 1000]]


def test_validation_requires_words_and_boxes_together():
    with pytest.raises(ValueError, match="BOTH words and boxes"):
        ImagePIIRedactor._validate_ocr_input(["John"], None)
    with pytest.raises(ValueError, match="BOTH words and boxes"):
        ImagePIIRedactor._validate_ocr_input(None, [[0, 0, 1, 1]])
    ImagePIIRedactor._validate_ocr_input(None, None)  # neither is fine


def test_validation_rejects_length_mismatch_and_bad_boxes():
    with pytest.raises(ValueError, match="same length"):
        ImagePIIRedactor._validate_ocr_input(["a", "b"], [[0, 0, 1, 1]])
    with pytest.raises(ValueError, match=r"boxes\[0\]"):
        ImagePIIRedactor._validate_ocr_input(["a"], [[0, 0, 1]])
    with pytest.raises(ValueError, match="inverted"):
        ImagePIIRedactor._validate_ocr_input(["a"], [[10, 0, 5, 1]])


# ------------------------------------------------- real-model integration --- #
pytestmark_integration = pytest.mark.skipif(
    os.environ.get("RUN_IMAGE_INTEGRATION") != "1",
    reason="set RUN_IMAGE_INTEGRATION=1 to run the real-model BYO-OCR test",
)


@pytest.fixture(scope="module")
def redactor():
    return ImagePIIRedactor("ekacare/pii-redactors", detect_visual=False,
                            device="cpu")


@pytestmark_integration
def test_byo_ocr_skips_tesseract_and_passes_boxes_through(redactor):
    from PIL import Image

    img = Image.new("RGB", (600, 200), "white")  # deliberately BLANK: if
    # Tesseract ran, it would find no words and detection would be empty —
    # results can only come from the supplied OCR.
    words = ["Patient:", "John", "Doe", "Phone:", "+91", "98765", "43210"]
    boxes = [[20, 20, 90, 40], [100, 20, 140, 40], [145, 20, 180, 40],
             [20, 60, 80, 80], [90, 60, 120, 80], [125, 60, 170, 80],
             [175, 60, 220, 80]]

    entities = redactor.detect(img, words=words, boxes=boxes)

    assert entities, "expected the supplied words to yield detections"
    by_text = {e.text: e for e in entities}
    assert "John" in by_text and by_text["John"].l1 == "person"
    # Supplied pixel boxes come back exactly — no normalization round trip.
    assert by_text["John"].bbox == (100, 20, 140, 40)
    phone_hits = [e for e in entities if e.category == "phone_mobile"]
    assert phone_hits, "expected the phone number words to be tagged"


@pytestmark_integration
def test_byo_ocr_empty_words_returns_empty(redactor):
    from PIL import Image

    img = Image.new("RGB", (100, 100), "white")
    assert redactor.detect(img, words=[], boxes=[]) == []
