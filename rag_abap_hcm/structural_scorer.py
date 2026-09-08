"""Proprietary structural scoring for ABAP source chunks.

Produces a normalized [0, 1] "structural quality" score for a chunk of ABAP
code, used as one of the three signals (vector, lexical, structural) in the
hybrid retriever's blended ranking. The heuristic favors chunks that:

* Are complete, well-formed logical units (matched FORM/ENDFORM etc.)
* Contain meaningful HCM-relevant operations (infotype access, PERFORM/CALL
  FUNCTION, exception handling)
* Are not pathologically short (stub) or long (dumping-ground routines)
* Exhibit reasonable nesting complexity rather than being a flat, low value
  sequence of statements
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INDENT_UNIT = 2

_HCM_OPERATION_RE = re.compile(
    r"\b(PROVIDE|INFOTYPE|RP_PROVIDE_FROM_LAST|RP_READ_INFOTYPE|"
    r"PNP-PERNR|CALL\s+FUNCTION|PERFORM|EXCEPTIONS)\b",
    re.IGNORECASE,
)

_COMPLETE_BLOCK_RE = re.compile(
    r"^\s*(FORM|METHOD|FUNCTION|CLASS|MODULE)\b.*", re.IGNORECASE
)
_END_BLOCK_RE = re.compile(
    r"^\s*(ENDFORM|ENDMETHOD|ENDFUNCTION|ENDCLASS|ENDMODULE)\b", re.IGNORECASE
)

_CONTROL_FLOW_RE = re.compile(
    r"^\s*(IF|LOOP|DO|WHILE|CASE|TRY)\b", re.IGNORECASE
)
_END_CONTROL_FLOW_RE = re.compile(
    r"^\s*(ENDIF|ENDLOOP|ENDDO|ENDWHILE|ENDCASE|ENDTRY)\b", re.IGNORECASE
)


@dataclass
class StructuralScoreBreakdown:
    completeness: float
    size_score: float
    complexity_score: float
    hcm_relevance: float
    total: float


class StructuralScorer:
    """Computes a proprietary structural quality score for ABAP chunks."""

    def __init__(
        self,
        ideal_min_lines: int = 5,
        ideal_max_lines: int = 80,
        weight_completeness: float = 0.35,
        weight_size: float = 0.20,
        weight_complexity: float = 0.20,
        weight_hcm_relevance: float = 0.25,
    ):
        self.ideal_min_lines = ideal_min_lines
        self.ideal_max_lines = ideal_max_lines
        self.weight_completeness = weight_completeness
        self.weight_size = weight_size
        self.weight_complexity = weight_complexity
        self.weight_hcm_relevance = weight_hcm_relevance

    def score(self, text: str) -> StructuralScoreBreakdown:
        lines = [l for l in text.splitlines() if l.strip()]
        completeness = self._completeness_score(lines)
        size_score = self._size_score(len(lines))
        complexity_score = self._complexity_score(lines)
        hcm_relevance = self._hcm_relevance_score(text)

        total = (
            completeness * self.weight_completeness
            + size_score * self.weight_size
            + complexity_score * self.weight_complexity
            + hcm_relevance * self.weight_hcm_relevance
        )
        return StructuralScoreBreakdown(
            completeness=completeness,
            size_score=size_score,
            complexity_score=complexity_score,
            hcm_relevance=hcm_relevance,
            total=min(max(total, 0.0), 1.0),
        )

    def _completeness_score(self, lines) -> float:
        if not lines:
            return 0.0
        opens_block = bool(_COMPLETE_BLOCK_RE.match(lines[0]))
        closes_block = bool(_END_BLOCK_RE.match(lines[-1]))
        if opens_block and closes_block:
            return 1.0
        if opens_block or closes_block:
            return 0.6
        return 0.5

    def _size_score(self, n_lines: int) -> float:
        if n_lines == 0:
            return 0.0
        if self.ideal_min_lines <= n_lines <= self.ideal_max_lines:
            return 1.0
        if n_lines < self.ideal_min_lines:
            return max(n_lines / self.ideal_min_lines, 0.1)
        # Decay smoothly for oversized chunks rather than hard-cutting to 0.
        overflow = n_lines - self.ideal_max_lines
        return max(1.0 - overflow / (self.ideal_max_lines * 3), 0.1)

    def _complexity_score(self, lines) -> float:
        if not lines:
            return 0.0
        depth = 0
        max_depth = 0
        control_statements = 0
        for line in lines:
            if _CONTROL_FLOW_RE.match(line):
                depth += 1
                max_depth = max(max_depth, depth)
                control_statements += 1
            elif _END_CONTROL_FLOW_RE.match(line):
                depth = max(depth - 1, 0)
        if control_statements == 0:
            return 0.3  # flat code: some value, but limited structure
        # Sweet spot around 1-4 levels of nesting; deeper is penalized.
        if max_depth <= 4:
            return min(0.5 + max_depth * 0.125, 1.0)
        return max(1.0 - (max_depth - 4) * 0.15, 0.2)

    def _hcm_relevance_score(self, text: str) -> float:
        matches = len(_HCM_OPERATION_RE.findall(text))
        if matches == 0:
            return 0.0
        return min(matches / 5.0, 1.0)
