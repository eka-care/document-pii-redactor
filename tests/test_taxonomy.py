from eka_pii_redaction.taxonomy import (
    ALL_ENTITIES, TEXT_ENTITIES, TEXT_REDACTABLE, VISUAL_ENTITIES,
    TEXT_L2_TO_L1, l1_group,
)


def test_mac_address_added_to_device_net():
    assert "mac_address" in TEXT_L2_TO_L1
    assert l1_group("mac_address") == "device_net"
    assert "mac_address" in TEXT_ENTITIES
    assert "mac_address" in ALL_ENTITIES


def test_text_redactable_is_text_only():
    # Text model predicts the text categories, never the visual ones.
    assert set(TEXT_REDACTABLE) == set(TEXT_ENTITIES)
    for v in VISUAL_ENTITIES:
        assert v not in TEXT_REDACTABLE


def test_text_redactable_matches_text_model_label_set():
    # The 48 lowercased entity types emitted by text/minilm/labels.json.
    expected = {
        "aadhaar_12_digit", "abha_address", "abha_number_14_digit", "age",
        "api_key_token", "bank_account_number", "blood_type", "brandname",
        "city_district", "country", "date_of_birth", "death_date",
        "device_serial_identifier", "driving_licence_no", "email", "fax",
        "gender", "geocode_coordinates", "health_plan_beneficiary_no",
        "high_entropy_secret", "iban", "insurance_tpa_policy_no", "ip_address",
        "mac_address", "mrn_uhid", "national_id",
        "occupation_designation_education_level", "other_date_time", "other_id",
        "other_person_name", "pan", "passport_no", "password", "phone_mobile",
        "pmjay_ayushman_id", "postal_zip_pin_code", "practitioner_reg_no_npi_nmc",
        "primary_subject_name", "religion_ethnicity_cast", "sexual_orientation",
        "ssn", "state_province", "street_address", "tax_id", "upi_id",
        "vehicle_id_licence_plate", "voter_id", "web_url",
    }
    assert set(TEXT_REDACTABLE) == expected
