from eka_pii_redaction.pseudonym import PseudonymMapping, label_for


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
