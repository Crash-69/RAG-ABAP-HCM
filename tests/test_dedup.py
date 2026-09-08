import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_abap_hcm.dedup import ContentDeduplicatorV4, content_hash, hamming_distance, simhash


def test_exact_duplicates_are_removed():
    dedup = ContentDeduplicatorV4()
    texts = ["FORM foo.\nWRITE 'a'.\nENDFORM.", "FORM foo.\nWRITE 'a'.\nENDFORM."]
    result = dedup.deduplicate(texts)
    assert result.kept_indices == [0]
    assert result.duplicate_of == {1: 0}


def test_near_duplicates_with_minor_variation_are_detected():
    dedup = ContentDeduplicatorV4(similarity_threshold=0.85)
    base = "FORM get_employee_name USING p_pernr.\n" + "\n".join(
        f"  DATA lv_{i} TYPE i." for i in range(30)
    ) + "\nENDFORM."
    variant = base.replace("get_employee_name", "get_employee_name_v2")
    result = dedup.deduplicate([base, variant])
    assert result.kept_indices == [0]
    assert 1 in result.duplicate_of


def test_distinct_content_is_kept():
    dedup = ContentDeduplicatorV4()
    texts = ["FORM foo.\nWRITE 'a'.\nENDFORM.", "FORM completely_different.\nCALL FUNCTION 'Z_XYZ'.\nENDFORM."]
    result = dedup.deduplicate(texts)
    assert result.kept_indices == [0, 1]
    assert result.duplicate_of == {}


def test_content_hash_is_case_and_whitespace_insensitive():
    a = content_hash("FORM foo.\n  WRITE 'a'.\nENDFORM.")
    b = content_hash("form foo.\nwrite 'a'.\nendform.")
    assert a == b


def test_simhash_similarity_threshold_helper():
    dedup = ContentDeduplicatorV4(similarity_threshold=0.99)
    assert dedup.is_near_duplicate("hello world", "hello world")
    assert not dedup.is_near_duplicate("hello world", "completely unrelated text about payroll clusters")


def test_similarity_threshold_validation():
    import pytest

    with pytest.raises(ValueError):
        ContentDeduplicatorV4(similarity_threshold=0)
    with pytest.raises(ValueError):
        ContentDeduplicatorV4(similarity_threshold=1.5)


def test_hamming_distance_zero_for_identical_fingerprints():
    fp = simhash("some abap source code")
    assert hamming_distance(fp, fp) == 0
