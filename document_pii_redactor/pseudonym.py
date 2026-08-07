"""Shared pseudonym mapping for de-identification (both modalities).

De-identification replaces each detected entity with a consistent numbered
pseudonym ("Person_1"): the same entity always gets the same pseudonym, a
different entity gets a different one, and the mapping is returned to the
caller so an authorized party can store it securely and re-link later. The
library itself never persists a mapping.

Anonymization deliberately does NOT use this module — no mapping may exist.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# L2 category -> human-friendly pseudonym label. Categories that share a label
# (both person-name classes -> "Person") also share a counter, so one real-world
# entity tagged inconsistently by the model still gets one pseudonym.
FRIENDLY_LABELS: dict[str, str] = {
    "primary_subject_name": "Person", "other_person_name": "Person",
    "gender": "Gender", "age": "Age", "religion_ethnicity_cast": "Community",
    "sexual_orientation": "Orientation", "blood_type": "BloodType",
    "occupation_designation_education_level": "Occupation",
    "country": "Country", "state_province": "State", "city_district": "City",
    "postal_zip_pin_code": "Postcode", "street_address": "Address",
    "geocode_coordinates": "Geocode",
    "date_of_birth": "Date", "death_date": "Date", "other_date_time": "Date",
    "phone_mobile": "Phone", "fax": "Fax", "email": "Email", "web_url": "URL",
    "ssn": "SSN", "aadhaar_12_digit": "Aadhaar", "pan": "PAN",
    "voter_id": "VoterID", "passport_no": "Passport",
    "driving_licence_no": "DrivingLicence", "national_id": "NationalID",
    "tax_id": "TaxID", "mrn_uhid": "MRN", "health_plan_beneficiary_no": "HealthPlanID",
    "abha_number_14_digit": "ABHA", "abha_address": "ABHAAddress",
    "pmjay_ayushman_id": "AyushmanID", "practitioner_reg_no_npi_nmc": "PractitionerReg",
    "bank_account_number": "BankAccount", "iban": "IBAN", "upi_id": "UPI",
    "insurance_tpa_policy_no": "PolicyNo", "other_id": "ID",
    "vehicle_id_licence_plate": "VehicleID",
    "ip_address": "IPAddress", "device_serial_identifier": "DeviceID",
    "mac_address": "MACAddress",
    "password": "Password", "api_key_token": "APIKey",
    "high_entropy_secret": "Secret",
    "brandname": "Brand",
}


def label_for(category: str) -> str:
    """Pseudonym label for a category; TitleCased category name as fallback."""
    got = FRIENDLY_LABELS.get(category)
    if got:
        return got
    return "".join(part.capitalize() for part in category.split("_"))


def _normalize(text: str) -> str:
    """Consistency key for one surface form: collapse whitespace + casefold."""
    return " ".join(text.split()).casefold()


def token_for(category: str, text: str, secret: str | None = None) -> str:
    """Globally deterministic hash pseudonym: {Label}_{md5[:6]}.

    Same value -> same token across documents, machines, and time — no
    mapping needed for consistency (the de-identify "hash" strategy).
    Hashed on the normalized form, so "John Doe" and "john  doe" agree.

    An optional `secret` salts the hash: without it, tokens for guessable
    values (phone numbers, dates) can be reversed by dictionary attack;
    with it, only holders of the same secret can reproduce the mapping.
    """
    payload = (secret or "") + _normalize(text)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"{label_for(category)}_{digest[:6]}"


@dataclass
class PseudonymMapping:
    """Entity -> pseudonym assignments for one record (possibly many pages).

    `entries` is serialized pseudonym-first (`label -> {pseudonym: original}`)
    because that is the re-identification direction the key-holder needs; a
    runtime reverse index provides the assignment direction.
    """
    entries: dict[str, dict[str, str]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index: dict[tuple[str, str], str] = {}
        for label, by_pseudonym in self.entries.items():
            for pseudonym, original in by_pseudonym.items():
                self._index[(label, _normalize(original))] = pseudonym

    def pseudonym_for(self, category: str, original: str) -> str:
        """Return this entity's pseudonym, assigning the next number if new."""
        label = label_for(category)
        key = (label, _normalize(original))
        existing = self._index.get(key)
        if existing is not None:
            return existing
        n = self.counters.get(label, 0) + 1
        self.counters[label] = n
        pseudonym = f"{label}_{n}"
        self.entries.setdefault(label, {})[pseudonym] = original
        self._index[key] = pseudonym
        return pseudonym

    def record_token(self, category: str, original: str, token: str) -> str:
        """Record an externally minted pseudonym (the hash strategy's tokens)
        so the mapping still captures token -> original for re-linking —
        hash tokens are one-way, so an uncaptured mapping is unrecoverable."""
        label = label_for(category)
        key = (label, _normalize(original))
        if key not in self._index:
            self.entries.setdefault(label, {})[token] = original
            self._index[key] = token
        return token

    def to_dict(self) -> dict:
        return {"entries": self.entries, "counters": self.counters}

    @classmethod
    def from_dict(cls, d: dict | None) -> "PseudonymMapping":
        if not d:
            return cls()
        return cls(entries=d.get("entries", {}), counters=d.get("counters", {}))
