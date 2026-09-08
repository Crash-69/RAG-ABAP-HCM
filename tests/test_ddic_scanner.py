import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_abap_hcm.ddic_scanner import DDICScanner


def test_scan_detects_known_hcm_table():
    scanner = DDICScanner()
    text = "TABLES: pa0002.\nSELECT SINGLE * FROM pa0002 INTO ls_pa0002."
    hits = scanner.scan(text)
    assert "PA0002" in hits
    assert hits["PA0002"].is_known_hcm_table


def test_scan_detects_hrp_table():
    scanner = DDICScanner()
    text = "DATA: ls_hrp1001 TYPE hrp1001.\nSELECT SINGLE * FROM hrp1001 INTO ls_hrp1001."
    hits = scanner.scan(text)
    assert "HRP1001" in hits


def test_scan_ignores_generic_lines_without_ddic_keywords():
    scanner = DDICScanner()
    text = "WRITE 'just some text without ddic terms'."
    hits = scanner.scan(text)
    assert hits == {}


def test_score_is_higher_for_known_tables_than_generic_text():
    scanner = DDICScanner()
    hcm_score = scanner.score("TABLES: pa0002, pa0001.\nSELECT * FROM pa0002 INTO TABLE lt_pa0002.")
    generic_score = scanner.score("DATA: lv_foo TYPE string.\nlv_foo = 'bar'.")
    assert hcm_score > generic_score


def test_score_bounded_between_zero_and_one():
    scanner = DDICScanner()
    text = "\n".join([f"SELECT * FROM pa000{i%2} INTO TABLE lt_{i}." for i in range(50)])
    score = scanner.score(text)
    assert 0.0 <= score <= 1.0
