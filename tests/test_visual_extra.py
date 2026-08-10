"""Missing-[visual]-extra error: raised early, with the install command."""
import builtins
import sys

import pytest

from document_pii_redactor import ImagePIIRedactor


def test_missing_ultralytics_fails_fast_with_install_hint(monkeypatch):
    # Simulate an environment without the extra, even if it's installed here.
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    real_import = builtins.__import__

    def no_ultralytics(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("No module named 'ultralytics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ultralytics)

    downloads = []
    monkeypatch.setattr(
        "document_pii_redactor.image.redactor._resolve_sources",
        lambda *a, **k: downloads.append(a) or ("unused", None),
    )

    with pytest.raises(ImportError) as exc:
        ImagePIIRedactor("ekacare/document-pii-redactor")  # detect_visual defaults True

    assert "pip install 'document-pii-redactor[visual]'" in str(exc.value)
    assert "detect_visual=False" in str(exc.value)
    assert downloads == []  # failed before any weight resolution/download
