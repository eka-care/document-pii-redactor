"""text_score_threshold: low-confidence text detections are dropped."""
from document_pii_redactor.entities import PIIEntity
from document_pii_redactor.image.redactor import filter_text_entities


def _e(kind, score, cat="primary_subject_name"):
    return PIIEntity(category=cat, kind=kind, bbox=(0, 0, 10, 10),
                     l1="person", text="x" if kind == "text" else None,
                     score=score)


def test_low_confidence_text_dropped_visual_untouched():
    ents = [_e("text", 0.98), _e("text", 0.60), _e("text", 0.75),
            _e("visual", 0.30, cat="signature"), _e("text", None)]
    kept = filter_text_entities(ents, 0.75)
    assert [e.score for e in kept] == [0.98, 0.75, 0.30, None]


def test_zero_threshold_disables_cutoff():
    ents = [_e("text", 0.01), _e("text", 0.99)]
    assert filter_text_entities(ents, 0.0) == ents
