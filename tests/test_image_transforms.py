from PIL import Image

from eka_pii_redaction.entities import PIIEntity
from eka_pii_redaction.image.transforms import (
    apply_anonymize, apply_deidentify, group_text_entities,
)
from eka_pii_redaction.pseudonym import PseudonymMapping


def _word(cat, bbox, txt, kind="text"):
    return PIIEntity(category=cat, kind=kind, bbox=bbox,
                     l1="person", text=txt, score=0.9)


# --------------------------------------------------------------- grouping --- #
def test_adjacent_same_category_words_merge():
    entities = [
        _word("primary_subject_name", (10, 10, 50, 30), "John"),
        _word("primary_subject_name", (55, 10, 90, 30), "Doe"),
    ]
    groups = group_text_entities(entities)
    assert len(groups) == 1
    assert groups[0].text == "John Doe"
    assert groups[0].bbox == (10, 10, 90, 30)


def test_different_lines_do_not_merge():
    entities = [
        _word("primary_subject_name", (10, 10, 50, 30), "John"),
        _word("primary_subject_name", (10, 50, 50, 70), "Asha"),
    ]
    assert len(group_text_entities(entities)) == 2


def test_large_horizontal_gap_does_not_merge():
    entities = [
        _word("primary_subject_name", (10, 10, 50, 30), "John"),
        _word("primary_subject_name", (200, 10, 240, 30), "Asha"),
    ]
    assert len(group_text_entities(entities)) == 2


def test_different_categories_do_not_merge():
    entities = [
        _word("primary_subject_name", (10, 10, 50, 30), "John"),
        _word("city_district", (55, 10, 90, 30), "Pune"),
    ]
    assert len(group_text_entities(entities)) == 2


def test_visual_entities_are_not_grouped():
    entities = [_word("signature", (10, 10, 80, 40), None, kind="visual")]
    assert group_text_entities(entities) == []


def test_mixed_box_heights_on_one_printed_line_still_merge():
    # Real OCR gives differing box heights for words on the same line; that
    # must not split "John Doe" into Person_1 + Person_2.
    entities = [
        _word("primary_subject_name", (10, 10, 50, 30), "John"),
        _word("primary_subject_name", (55, 16, 90, 28), "Doe"),
    ]
    groups = group_text_entities(entities)
    assert len(groups) == 1
    assert groups[0].text == "John Doe"


def test_groups_come_back_in_reading_order():
    # Input deliberately lists the lower line first; numbering follows the
    # order groups are returned, so reading order matters.
    entities = [
        _word("primary_subject_name", (10, 50, 50, 70), "Asha"),
        _word("primary_subject_name", (10, 10, 50, 30), "John"),
    ]
    groups = group_text_entities(entities)
    assert [g.text for g in groups] == ["John", "Asha"]


def test_chain_survives_interleaved_other_category_word():
    # "14 MG Road, <Bangalore,> Karnataka" — the address run continues across
    # the city word instead of restarting a new numbered group.
    entities = [
        _word("street_address", (0, 0, 40, 20), "14"),
        _word("city_district", (44, 0, 58, 20), "Blr"),
        _word("street_address", (62, 0, 100, 20), "Road"),
    ]
    groups = group_text_entities(entities)
    assert len(groups) == 2
    street = next(g for g in groups if g.category == "street_address")
    assert street.text == "14 Road"


# --------------------------------------------------------------- appliers --- #
def test_deidentify_assigns_one_pseudonym_per_merged_entity():
    img = Image.new("RGB", (200, 60), "white")
    entities = [
        _word("primary_subject_name", (10, 10, 50, 30), "John"),
        _word("primary_subject_name", (55, 10, 90, 30), "Doe"),
    ]
    mapping = PseudonymMapping()
    out = apply_deidentify(img, entities, mapping)
    assert out.size == img.size
    assert mapping.entries["Person"] == {"Person_1": "John Doe"}


def test_anonymize_blacks_out_visual_entities_and_returns_copy():
    img = Image.new("RGB", (100, 100), "white")
    entities = [_word("signature", (20, 20, 80, 80), None, kind="visual")]
    out = apply_anonymize(img, entities)
    assert out.getpixel((50, 50)) == (0, 0, 0)
    assert img.getpixel((50, 50)) == (255, 255, 255)  # input untouched
