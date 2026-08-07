from document_pii_redactor.text.redactor import (
    DEFAULT_HF_REPO, TextPIISpan, apply_mask,
)


def _span(cat, start, end, l1, txt):
    return TextPIISpan(category=cat, start=start, end=end, l1=l1, text=txt, score=0.9)


def test_default_hf_repo_constant():
    assert DEFAULT_HF_REPO == "ekacare/document-pii-redactor"


def test_apply_mask_default_token():
    text = "Name Asha here"
    spans = [_span("primary_subject_name", 5, 9, "person", "Asha")]
    assert apply_mask(text, spans) == "Name [REDACTED] here"


def test_apply_mask_category_placeholder():
    text = "Name Asha here"
    spans = [_span("primary_subject_name", 5, 9, "person", "Asha")]
    assert apply_mask(text, spans, mask="[{category}]") == "Name [primary_subject_name] here"


def test_apply_mask_l1_placeholder():
    text = "Name Asha here"
    spans = [_span("primary_subject_name", 5, 9, "person", "Asha")]
    assert apply_mask(text, spans, mask="<{l1}>") == "Name <person> here"


def test_apply_mask_multiple_spans_in_order():
    text = "a@b.com and Asha"
    spans = [
        _span("email", 0, 7, "contact", "a@b.com"),
        _span("primary_subject_name", 12, 16, "person", "Asha"),
    ]
    assert apply_mask(text, spans) == "[REDACTED] and [REDACTED]"


def test_apply_mask_unsorted_spans_handled():
    text = "a@b.com and Asha"
    spans = [
        _span("primary_subject_name", 12, 16, "person", "Asha"),
        _span("email", 0, 7, "contact", "a@b.com"),
    ]
    assert apply_mask(text, spans) == "[REDACTED] and [REDACTED]"
