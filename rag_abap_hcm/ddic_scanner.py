"""Lexical scanning for SAP DDIC (Data Dictionary) terms.

This module performs a fast, regex based scan of ABAP/HCM source text to
surface references to DDIC objects (tables, structures, domains, data
elements) as well as well known HR/HCM infrastructure tables (PA/PB/PD
tables, cluster tables, HRP* tables, etc). The resulting hits are used by
the hybrid retriever to boost lexical relevance alongside vector similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

# Well known SAP HCM DDIC objects. This is not exhaustive but covers the
# core Personnel Administration (PA), Organizational Management (PD/HRP) and
# Payroll cluster tables that dominate HCM custom development.
KNOWN_HCM_TABLES: Set[str] = {
    "PA0000", "PA0001", "PA0002", "PA0006", "PA0008", "PA0009", "PA0014",
    "PA0015", "PA0041", "PA0105", "PA0185", "PA0302", "PA0041",
    "T500P", "T501", "T503", "T510", "T512W", "T528B", "T549Q",
    "HRP1000", "HRP1001", "HRP1002", "HRP1008", "HRP1013",
    "HRPAD03", "HRPADYT",
    "PCL1", "PCL2", "PCL3", "PCL4",
    "PERNR", "INFTY", "MOLGA", "PLVAR", "OTYPE", "OBJID",
}

# ABAP DDIC-related statement keywords that indicate a Data Dictionary
# reference is being declared/used nearby.
_DDIC_KEYWORDS = re.compile(
    r"\b(TABLES|DATA|TYPES|SELECT|INTO\s+TABLE|FROM|INCLUDE\s+STRUCTURE|"
    r"REF\s+TO|LIKE|TYPE)\b",
    re.IGNORECASE,
)

# Matches identifiers that look like ABAP DDIC object names: a HCM infotype
# table (PA/PB/PD followed by 4 digits), an HRP* org-management table, a
# generic customizing table starting with T followed by digits, or any
# all-caps identifier of at least 3 characters that could be a structure
# or data element name.
_CANDIDATE_RE = re.compile(
    r"\b("
    r"P[AB]\d{4}|"
    r"HRP\w{2,10}|"
    r"T\d{3}\w{0,6}|"
    r"[A-Za-z][A-Za-z0-9_]{2,29}"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class DDICHit:
    term: str
    count: int
    is_known_hcm_table: bool
    lines: List[int] = field(default_factory=list)


class DDICScanner:
    """Scans ABAP source text for DDIC term references."""

    def __init__(self, known_tables: Set[str] = None):
        self.known_tables = known_tables if known_tables is not None else KNOWN_HCM_TABLES

    def scan(self, text: str) -> Dict[str, DDICHit]:
        hits: Dict[str, DDICHit] = {}
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not _DDIC_KEYWORDS.search(line) and not any(
                table in line.upper() for table in self.known_tables
            ):
                continue
            for match in _CANDIDATE_RE.finditer(line):
                term = match.group(1).upper()
                if not self._is_plausible_ddic_term(term):
                    continue
                hit = hits.get(term)
                if hit is None:
                    hit = DDICHit(
                        term=term,
                        count=0,
                        is_known_hcm_table=term in self.known_tables,
                    )
                    hits[term] = hit
                hit.count += 1
                hit.lines.append(line_no)
        return hits

    def score(self, text: str) -> float:
        """Return a normalized lexical relevance score in [0, 1]."""
        hits = self.scan(text)
        if not hits:
            return 0.0
        known = sum(1 for h in hits.values() if h.is_known_hcm_table)
        total_mentions = sum(h.count for h in hits.values())
        # Weight known HCM tables heavily, generic DDIC-looking identifiers
        # more lightly, and saturate so a handful of hits already scores well.
        raw = known * 2.0 + min(total_mentions, 20) * 0.1
        return min(raw / 5.0, 1.0)

    def _is_plausible_ddic_term(self, term: str) -> bool:
        if term in self.known_tables:
            return True
        if re.fullmatch(r"P[AB]\d{4}", term):
            return True
        if re.fullmatch(r"HRP\w{2,10}", term):
            return True
        if re.fullmatch(r"T\d{3}\w{0,6}", term):
            return True
        # Reject common ABAP language keywords that would otherwise look
        # like plausible DDIC object names.
        if term in _ABAP_KEYWORDS:
            return False
        return (len(term) >= 5 and "_" in term) or len(term) >= 6


_ABAP_KEYWORDS = {
    "TABLES", "DATA", "TYPES", "SELECT", "FROM", "INTO", "WHERE", "ENDIF",
    "ENDFORM", "ENDLOOP", "ENDMETHOD", "ENDFUNCTION", "ENDCLASS", "PERFORM",
    "MOVE", "WRITE", "APPEND", "READ", "LOOP", "IF", "ELSE", "ELSEIF",
    "CALL", "FUNCTION", "METHOD", "CLASS", "PUBLIC", "PRIVATE", "REF",
    "STRUCTURE", "INCLUDE", "LIKE", "TYPE", "VALUE", "OPTIONAL",
}
