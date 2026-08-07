"""PII entity taxonomy for document-pii-redactor.

Two kinds of entities:
  - TEXT entities: 47 fine-grained L2 PII classes (grouped under 8 L1 groups),
    predicted per OCR word by the LayoutLMv3 token classifier.
  - VISUAL entities: 6 image-region categories predicted by the YOLO detector.

These names are the canonical category strings used everywhere in the public API
(`detect()` results, `categories=[...]`, `list_entities()`).
"""
from __future__ import annotations

# L2 (fine) text class -> L1 (coarse) group. 47 classes.
TEXT_L2_TO_L1: dict[str, str] = {
    "primary_subject_name": "person", "other_person_name": "person", "gender": "person",
    "age": "person", "religion_ethnicity_cast": "person", "sexual_orientation": "person",
    "blood_type": "person", "occupation_designation_education_level": "person",
    "country": "location", "state_province": "location", "city_district": "location",
    "postal_zip_pin_code": "location", "street_address": "location",
    "geocode_coordinates": "location",
    "date_of_birth": "date_time", "death_date": "date_time", "other_date_time": "date_time",
    "phone_mobile": "contact", "fax": "contact", "email": "contact", "web_url": "contact",
    "ssn": "uid", "aadhaar_12_digit": "uid", "pan": "uid", "voter_id": "uid",
    "passport_no": "uid", "driving_licence_no": "uid", "national_id": "uid", "tax_id": "uid",
    "mrn_uhid": "uid", "health_plan_beneficiary_no": "uid", "abha_number_14_digit": "uid",
    "abha_address": "uid", "pmjay_ayushman_id": "uid", "practitioner_reg_no_npi_nmc": "uid",
    "bank_account_number": "uid", "iban": "uid", "upi_id": "uid",
    "insurance_tpa_policy_no": "uid", "other_id": "uid", "vehicle_id_licence_plate": "uid",
    "ip_address": "device_net", "device_serial_identifier": "device_net",
    "mac_address": "device_net",
    "password": "credential", "api_key_token": "credential", "high_entropy_secret": "credential",
    "brandname": "entity",
}

TEXT_ENTITIES: list[str] = list(TEXT_L2_TO_L1.keys())

# Categories the TEXT modality (plain-string redactor) can emit. The text model
# predicts every text category but none of the visual ones, so this is just the
# text entity list — named separately so the text API can advertise its own set.
TEXT_REDACTABLE: list[str] = list(TEXT_ENTITIES)

# Visual entity categories (image regions, no associated OCR text). All share the
# L1 group "biometric_visual".
VISUAL_ENTITIES: list[str] = [
    "face_photo", "signature", "seal_stamp",
    "fingerprint_thumb_impression", "qr_barcode", "logo",
]

ALL_ENTITIES: list[str] = TEXT_ENTITIES + VISUAL_ENTITIES


def l1_group(category: str) -> str:
    """Return the L1 group for a category ('biometric_visual' for visual entities)."""
    if category in VISUAL_ENTITIES:
        return "biometric_visual"
    return TEXT_L2_TO_L1.get(category, "unknown")


def validate_entities(names) -> list[str]:
    """Validate a list of category names; raise on unknown ones."""
    names = list(names or [])
    unknown = [n for n in names if n not in ALL_ENTITIES]
    if unknown:
        raise ValueError(
            f"Unknown entity categories: {unknown}. "
            f"Valid categories: {ALL_ENTITIES}"
        )
    return names
