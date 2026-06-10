def test_text_package_exports():
    from eka_pii_redaction.text import (
        TextPIIRedactor, TextPIISpan, DEFAULT_HF_REPO,
    )
    assert DEFAULT_HF_REPO == "ekacare/pii-redactors"
    assert TextPIIRedactor.list_entities()  # non-empty list


def test_top_level_exports():
    import eka_pii_redaction as pkg
    assert hasattr(pkg, "TextPIIRedactor")
    assert hasattr(pkg, "TextPIISpan")
    assert hasattr(pkg, "ImagePIIRedactor")
    for name in ("TextPIIRedactor", "TextPIISpan", "ImagePIIRedactor"):
        assert name in pkg.__all__
