from document_pii_redactor.text.minilm import _trim, _flatten_chunks, merge_bio_spans


def test_trim_strips_surrounding_whitespace():
    text = "  Asha  "
    assert _trim(text, 0, len(text)) == (2, 6)


def test_merge_simple_bi_span_trims_leading_space():
    text = "Email john@x.com now"
    offsets = [(0, 5), (5, 10), (10, 12), (12, 16), (16, 20)]
    labels = ["O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "O"]
    scores = [0.9, 0.8, 0.7, 0.6, 0.95]
    spans = merge_bio_spans(text, offsets, labels, scores)
    assert len(spans) == 1
    s = spans[0]
    assert s.category == "email"
    assert s.l1 == "contact"
    assert (s.start, s.end) == (6, 16)          # leading space (5->6) trimmed
    assert s.text == "john@x.com"
    assert s.score == round((0.8 + 0.7 + 0.6) / 3, 4)


def test_b_tag_starts_new_span_even_when_adjacent():
    text = "Asha Bob"
    offsets = [(0, 4), (4, 8)]
    labels = ["B-PRIMARY_SUBJECT_NAME", "B-OTHER_PERSON_NAME"]
    scores = [0.9, 0.9]
    spans = merge_bio_spans(text, offsets, labels, scores)
    assert [s.category for s in spans] == [
        "primary_subject_name", "other_person_name"]


def test_specials_and_O_tokens_skipped():
    text = "hi name"
    offsets = [(0, 0), (0, 2), (2, 7), (0, 0)]   # <s>, "hi"=O, " name", </s>
    labels = ["O", "O", "B-OTHER_PERSON_NAME", "O"]
    scores = [None, 0.9, 0.8, None]
    spans = merge_bio_spans(text, offsets, labels, scores)
    assert len(spans) == 1
    assert (spans[0].start, spans[0].end) == (3, 7)
    assert spans[0].text == "name"


def test_flatten_dedups_overlap_first_chunk_wins():
    chunk_offsets = [[(0, 3), (3, 6)], [(3, 6), (6, 9)]]
    chunk_labels = [["B-AGE", "I-AGE"], ["B-GENDER", "B-COUNTRY"]]
    chunk_scores = [[0.9, 0.8], [0.5, 0.7]]
    offs, labs, scs = _flatten_chunks(chunk_offsets, chunk_labels, chunk_scores)
    assert offs == [(0, 3), (3, 6), (6, 9)]
    assert labs == ["B-AGE", "I-AGE", "B-COUNTRY"]   # (3,6): first chunk wins
    assert scs == [0.9, 0.8, 0.7]


def test_flatten_skips_zero_width_tokens():
    chunk_offsets = [[(0, 0), (0, 3), (0, 0)]]
    chunk_labels = [["O", "B-AGE", "O"]]
    chunk_scores = [[None, 0.9, None]]
    offs, labs, scs = _flatten_chunks(chunk_offsets, chunk_labels, chunk_scores)
    assert offs == [(0, 3)]
    assert labs == ["B-AGE"]
    assert scs == [0.9]
