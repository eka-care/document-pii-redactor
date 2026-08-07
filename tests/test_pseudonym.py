from document_pii_redactor.pseudonym import PseudonymMapping, label_for


def test_label_for_friendly_and_fallback():
    assert label_for("primary_subject_name") == "Person"
    assert label_for("city_district") == "City"
    assert label_for("not_a_real_category") == "NotARealCategory"


def test_assigns_sequential_pseudonyms_per_label():
    m = PseudonymMapping()
    assert m.pseudonym_for("primary_subject_name", "John") == "Person_1"
    assert m.pseudonym_for("other_person_name", "Asha") == "Person_2"
    assert m.pseudonym_for("city_district", "Bangalore") == "City_1"


def test_same_entity_reuses_pseudonym():
    m = PseudonymMapping()
    first = m.pseudonym_for("primary_subject_name", "John")
    assert m.pseudonym_for("primary_subject_name", "John") == first
    # Normalization: whitespace + case don't split an entity.
    assert m.pseudonym_for("primary_subject_name", "  JOHN ") == first


def test_shared_person_label_across_name_categories():
    # The same name tagged under either person-name class is one entity.
    m = PseudonymMapping()
    a = m.pseudonym_for("primary_subject_name", "John")
    b = m.pseudonym_for("other_person_name", "John")
    assert a == b == "Person_1"


def test_mapping_stores_first_seen_original():
    m = PseudonymMapping()
    m.pseudonym_for("primary_subject_name", "John")
    m.pseudonym_for("primary_subject_name", "JOHN")
    assert m.entries["Person"] == {"Person_1": "John"}


def test_round_trip_serialization_continues_numbering():
    m = PseudonymMapping()
    m.pseudonym_for("primary_subject_name", "John")
    revived = PseudonymMapping.from_dict(m.to_dict())
    assert revived.pseudonym_for("primary_subject_name", "John") == "Person_1"
    assert revived.pseudonym_for("primary_subject_name", "Asha") == "Person_2"


def test_from_dict_none_gives_empty_mapping():
    m = PseudonymMapping.from_dict(None)
    assert m.pseudonym_for("email", "a@b.com") == "Email_1"


# ------------------------------------------------------------ hash tokens --- #
def test_token_for_is_deterministic_and_normalized():
    import re
    from document_pii_redactor.pseudonym import token_for

    t = token_for("primary_subject_name", "John Doe")
    assert re.fullmatch(r"Person_[0-9a-f]{6}", t)
    # Global determinism: same value -> same token, no shared state needed.
    assert token_for("primary_subject_name", "John Doe") == t
    # Normalization: whitespace/case variants agree.
    assert token_for("primary_subject_name", "  JOHN   doe ") == t
    # Different values -> different tokens.
    assert token_for("primary_subject_name", "Asha Menon") != t


def test_token_for_secret_salts_the_hash():
    from document_pii_redactor.pseudonym import token_for

    plain = token_for("phone_mobile", "+91 98765 43210")
    salted = token_for("phone_mobile", "+91 98765 43210", secret="s3cr3t")
    assert plain != salted
    # Same secret reproduces the same token — cross-system consistency.
    assert token_for("phone_mobile", "+91 98765 43210", secret="s3cr3t") == salted


def test_record_token_captures_reverse_mapping_once():
    m = PseudonymMapping()
    from document_pii_redactor.pseudonym import token_for

    tok = token_for("primary_subject_name", "John Doe")
    assert m.record_token("primary_subject_name", "John Doe", tok) == tok
    m.record_token("primary_subject_name", "JOHN DOE", tok)  # normalized dup
    assert m.entries["Person"] == {tok: "John Doe"}
