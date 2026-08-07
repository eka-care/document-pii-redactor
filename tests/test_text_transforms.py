from document_pii_redactor.pseudonym import PseudonymMapping
from document_pii_redactor.text.redactor import TextPIISpan
from document_pii_redactor.text.transforms import (
    anonymize_value, apply_anonymization, apply_pseudonyms, merge_adjacent_spans,
)


def _span(cat, start, end, l1, txt):
    return TextPIISpan(category=cat, start=start, end=end, l1=l1, text=txt, score=0.9)


# ------------------------------------------------------------- deidentify --- #
def test_pseudonyms_are_consistent_within_text():
    text = "John met Asha. John left."
    spans = [
        _span("primary_subject_name", 0, 4, "person", "John"),
        _span("other_person_name", 9, 13, "person", "Asha"),
        _span("primary_subject_name", 15, 19, "person", "John"),
    ]
    m = PseudonymMapping()
    assert apply_pseudonyms(text, spans, m) == "Person_1 met Person_2. Person_1 left."


def test_pseudonyms_continue_across_documents_with_shared_mapping():
    m = PseudonymMapping()
    page1 = apply_pseudonyms(
        "John", [_span("primary_subject_name", 0, 4, "person", "John")], m)
    page2 = apply_pseudonyms(
        "Asha and John",
        [_span("other_person_name", 0, 4, "person", "Asha"),
         _span("primary_subject_name", 9, 13, "person", "John")], m)
    assert page1 == "Person_1"
    assert page2 == "Person_2 and Person_1"


def test_overlapping_spans_skip_consumed_region():
    text = "abcdef"
    spans = [
        _span("email", 0, 4, "contact", "abcd"),
        _span("email", 2, 6, "contact", "cdef"),  # overlaps -> skipped
    ]
    out = apply_pseudonyms(text, spans, PseudonymMapping())
    assert out == "Email_1ef"


def test_split_name_spans_merge_into_one_pseudonym():
    # BIO decoding sometimes restarts mid-entity ("Mr. John" + "Doe");
    # whitespace-separated same-category spans must get ONE pseudonym.
    text = "Patient: Mr. John Doe here"
    spans = [
        _span("primary_subject_name", 9, 17, "person", "Mr. John"),
        _span("primary_subject_name", 18, 21, "person", "Doe"),
    ]
    out = apply_pseudonyms(text, spans, PseudonymMapping())
    assert out == "Patient: Person_1 here"


def test_comma_separated_names_stay_separate_people():
    text = "John, Asha"
    spans = [
        _span("primary_subject_name", 0, 4, "person", "John"),
        _span("primary_subject_name", 6, 10, "person", "Asha"),
    ]
    out = apply_pseudonyms(text, spans, PseudonymMapping())
    assert out == "Person_1, Person_2"


def test_adjacent_different_categories_do_not_merge():
    text = "Indiranagar Karnataka"
    spans = [
        _span("city_district", 0, 11, "location", "Indiranagar"),
        _span("state_province", 12, 21, "location", "Karnataka"),
    ]
    merged = merge_adjacent_spans(text, spans)
    assert len(merged) == 2


# -------------------------------------------------------------- anonymize --- #
def test_age_buckets():
    assert anonymize_value("age", "45 yrs") == "40–49"
    assert anonymize_value("age", "7") == "0–9"
    assert anonymize_value("age", "elderly") == "[AGE]"


def test_dates_keep_year_only():
    assert anonymize_value("date_of_birth", "12-03-1979") == "1979"
    assert anonymize_value("other_date_time", "02/06/2026") == "2026"
    assert anonymize_value("death_date", "12th March") == "[DATE]"


def test_coarse_geography_kept_fine_geography_collapsed():
    assert anonymize_value("state_province", "Karnataka") == "Karnataka"
    assert anonymize_value("country", "India") == "India"
    assert anonymize_value("city_district", "Indiranagar") == "[LOCATION]"
    assert anonymize_value("street_address", "14 MG Road") == "[LOCATION]"
    assert anonymize_value("postal_zip_pin_code", "560038") == "[LOCATION]"


def test_direct_identifiers_collapse_without_numbering():
    # Two different names both become the same unnumbered token — nothing to
    # link back through, unlike de-identification.
    assert anonymize_value("primary_subject_name", "John") == "[PERSON]"
    assert anonymize_value("other_person_name", "Asha") == "[PERSON]"
    assert anonymize_value("phone_mobile", "+91 98765 43210") == "[PHONE]"
    assert anonymize_value("aadhaar_12_digit", "1234 5678 9012") == "[AADHAAR]"


def test_anonymize_collapses_adjacent_same_category_spans():
    # Without merging this would read "[PERSON] [PERSON]".
    text = "Mr. John Doe visited"
    spans = [
        _span("primary_subject_name", 0, 8, "person", "Mr. John"),
        _span("primary_subject_name", 9, 12, "person", "Doe"),
    ]
    assert apply_anonymization(text, spans) == "[PERSON] visited"


def test_apply_anonymization_end_to_end():
    text = "John, 45 yrs, DOB 12-03-1979, Indiranagar, Karnataka"
    spans = [
        _span("primary_subject_name", 0, 4, "person", "John"),
        _span("age", 6, 12, "person", "45 yrs"),
        _span("date_of_birth", 18, 28, "date_time", "12-03-1979"),
        _span("city_district", 30, 41, "location", "Indiranagar"),
        _span("state_province", 43, 52, "location", "Karnataka"),
    ]
    assert apply_anonymization(text, spans) == \
        "[PERSON], 40–49, DOB 1979, [LOCATION], Karnataka"


# ---------------------------------------------------------- hash strategy --- #
def test_hash_strategy_is_consistent_across_documents():
    from document_pii_redactor.pseudonym import token_for

    span = [_span("primary_subject_name", 0, 8, "person", "John Doe")]
    # Two separate calls with FRESH mappings — counters would restart at
    # Person_1 both times; hash tokens must agree because they derive from
    # the value itself.
    out1 = apply_pseudonyms("John Doe", span, PseudonymMapping(), strategy="hash")
    out2 = apply_pseudonyms("John Doe", span, PseudonymMapping(), strategy="hash")
    assert out1 == out2 == token_for("primary_subject_name", "John Doe")


def test_hash_strategy_records_mapping_and_merges_spans():
    text = "Patient: Mr. John Doe here"
    spans = [
        _span("primary_subject_name", 9, 17, "person", "Mr. John"),
        _span("primary_subject_name", 18, 21, "person", "Doe"),
    ]
    m = PseudonymMapping()
    out = apply_pseudonyms(text, spans, m, strategy="hash")
    from document_pii_redactor.pseudonym import token_for

    tok = token_for("primary_subject_name", "Mr. John Doe")
    assert out == f"Patient: {tok} here"          # merged -> ONE token
    assert m.entries["Person"] == {tok: "Mr. John Doe"}


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError, match="strategy"):
        apply_pseudonyms("x", [], PseudonymMapping(), strategy="vault")
