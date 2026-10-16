"""EKA_PII_TEXT_ONLY=1: the server loads and serves the text model only.
No model is downloaded: _get_text / _get are monkeypatched."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from document_pii_redactor import server  # noqa: E402


class FakeSpan:
    def to_dict(self):
        return {"start": 0, "end": 4, "text": "Ravi", "category": "PERSON"}


class FakeText:
    device = "cpu"

    def detect(self, text, categories=None):
        return [FakeSpan()]


class FakeImage:
    device = "cpu"
    detect_visual = True


@pytest.fixture
def calls(monkeypatch):
    calls = {"text": 0, "image": 0}

    def get_text():
        calls["text"] += 1
        return FakeText()

    def get_image():
        calls["image"] += 1
        return FakeImage()

    monkeypatch.setattr(server, "_get_text", get_text)
    monkeypatch.setattr(server, "_get", get_image)
    return calls


PNG = ("a.png", b"\x89PNG", "image/png")


def test_text_only_never_loads_the_image_model(monkeypatch, calls):
    monkeypatch.setenv("EKA_PII_TEXT_ONLY", "1")
    with TestClient(server.app) as c:
        assert calls["text"] == 1 and calls["image"] == 0          # startup warm-up
        assert c.get("/health").json() == {"status": "ok", "device": "cpu", "mode": "text"}
        r = c.post("/detect-text", json={"text": "Ravi"})
        assert r.status_code == 200 and r.json() == {"spans": [FakeSpan().to_dict()]}
        for method, path, kw in [("post", "/detect", {"files": {"file": PNG}}),
                                 ("post", "/redact", {"files": {"file": PNG}}),
                                 ("post", "/deidentify", {"files": {"file": PNG}}),
                                 ("post", "/anonymize", {"files": {"file": PNG}}),
                                 ("get", "/entities", {})]:
            r = getattr(c, method)(path, **kw)
            assert r.status_code == 503 and r.json() == {"error": "text-only server"}, path
        assert calls["image"] == 0


def test_default_is_unchanged(monkeypatch, calls):
    monkeypatch.delenv("EKA_PII_TEXT_ONLY", raising=False)
    with TestClient(server.app) as c:
        assert calls["image"] >= 1                                  # startup loads the image model
        assert c.get("/health").json() == {"status": "ok", "device": "cpu", "detect_visual": True}
        assert c.get("/entities").status_code == 200
        assert c.post("/detect-text", json={"text": "Ravi"}).status_code == 200
