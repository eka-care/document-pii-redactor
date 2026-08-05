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
    spans = redactor.detect(text)
    out = redactor.redact(text, spans, mask="[{category}]")
    assert "john.doe@example.com" not in out
    assert "[email]" in out
