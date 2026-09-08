import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_abap_hcm.structural_scorer import StructuralScorer

COMPLETE_FORM = """FORM get_employee_name USING p_pernr TYPE persno.
  DATA: ls_pa0002 TYPE pa0002.
  SELECT SINGLE * FROM pa0002 INTO ls_pa0002 WHERE pernr = p_pernr.
  IF sy-subrc = 0.
    PERFORM format_name.
  ELSE.
    CALL FUNCTION 'Z_LOG_ERROR'.
  ENDIF.
ENDFORM.
"""

STUB_FORM = "FORM stub.\nENDFORM.\n"

FLAT_STATEMENTS = "\n".join(f"WRITE 'line {i}'." for i in range(10))


def test_complete_hcm_form_scores_higher_than_stub():
    scorer = StructuralScorer()
    complete_score = scorer.score(COMPLETE_FORM).total
    stub_score = scorer.score(STUB_FORM).total
    assert complete_score > stub_score


def test_score_is_bounded():
    scorer = StructuralScorer()
    breakdown = scorer.score(COMPLETE_FORM)
    assert 0.0 <= breakdown.total <= 1.0


def test_flat_code_has_lower_complexity_than_nested_code():
    scorer = StructuralScorer()
    flat_breakdown = scorer.score(FLAT_STATEMENTS)
    nested_breakdown = scorer.score(COMPLETE_FORM)
    assert nested_breakdown.complexity_score > flat_breakdown.complexity_score


def test_empty_text_scores_zero():
    scorer = StructuralScorer()
    breakdown = scorer.score("")
    assert breakdown.total == 0.0
