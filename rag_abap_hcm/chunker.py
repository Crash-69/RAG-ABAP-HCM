"""ABAP-aware source chunking.

SAP ABAP source files mix multiple logical units (FORM routines, METHODs,
FUNCTION modules, module pool screens, CLASS definitions/implementations).
Naively splitting on line count or characters destroys semantic boundaries
and hurts retrieval quality. This chunker splits ABAP source on well known
structural boundaries, falling back to a sliding window for source that has
no recognizable structure (e.g. plain includes with only statements).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Pairs of (opening keyword pattern, closing keyword pattern) that delimit a
# logical ABAP unit. Matching is case-insensitive and anchored at the start
# of a (stripped) line, which is how ABAP conventionally formats these
# statements.
_BLOCK_PATTERNS = [
    (r"^FORM\s+(\w+)", r"^ENDFORM\b"),
    (r"^METHOD\s+(\S+)", r"^ENDMETHOD\b"),
    (r"^FUNCTION\s+(\S+)", r"^ENDFUNCTION\b"),
    (r"^CLASS\s+(\S+)\s+(?:DEFINITION|IMPLEMENTATION)", r"^ENDCLASS\b"),
    (r"^MODULE\s+(\S+)", r"^ENDMODULE\b"),
]

_HEADER_KEYWORDS = re.compile(
    r"^(REPORT|PROGRAM|INCLUDE|FUNCTION-POOL)\b", re.IGNORECASE
)

_OPEN_RE = [
    (re.compile(open_pat, re.IGNORECASE), re.compile(close_pat, re.IGNORECASE))
    for open_pat, close_pat in _BLOCK_PATTERNS
]


@dataclass
class Chunk:
    """A single logical unit of ABAP source extracted from a file."""

    content: str
    unit_type: str
    name: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    source_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "unit_type": self.unit_type,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "source_path": self.source_path,
            **self.metadata,
        }


class ABAPChunker:
    """Splits ABAP source code into logical, retrieval-friendly chunks."""

    def __init__(self, fallback_window: int = 60, fallback_overlap: int = 10):
        if fallback_overlap >= fallback_window:
            raise ValueError("fallback_overlap must be smaller than fallback_window")
        self.fallback_window = fallback_window
        self.fallback_overlap = fallback_overlap

    def chunk_text(self, text: str, source_path: Optional[str] = None) -> List[Chunk]:
        lines = text.splitlines()
        chunks: List[Chunk] = []
        consumed = [False] * len(lines)

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            matched = False
            for open_re, close_re in _OPEN_RE:
                m = open_re.match(stripped)
                if not m:
                    continue
                end_idx = self._find_matching_end(lines, i, close_re)
                unit_type = self._unit_type_for(open_re.pattern)
                name = m.group(1).rstrip(".") if m.groups() else None
                block_lines = lines[i : end_idx + 1]
                chunks.append(
                    Chunk(
                        content="\n".join(block_lines),
                        unit_type=unit_type,
                        name=name,
                        start_line=i + 1,
                        end_line=end_idx + 1,
                        source_path=source_path,
                    )
                )
                for j in range(i, end_idx + 1):
                    consumed[j] = True
                i = end_idx + 1
                matched = True
                break
            if not matched:
                i += 1

        # Any remaining, un-consumed lines are treated as "header" or
        # "loose statement" material and are chunked with a sliding window
        # so that nothing is silently dropped from retrieval.
        remainder_lines = [
            (idx, line) for idx, line in enumerate(lines) if not consumed[idx]
        ]
        if remainder_lines:
            chunks.extend(
                self._window_chunks(remainder_lines, source_path=source_path)
            )

        chunks.sort(key=lambda c: c.start_line)
        return chunks

    def _unit_type_for(self, pattern: str) -> str:
        if pattern.startswith("^FORM"):
            return "FORM"
        if pattern.startswith("^METHOD"):
            return "METHOD"
        if pattern.startswith("^FUNCTION"):
            return "FUNCTION"
        if pattern.startswith("^CLASS"):
            return "CLASS"
        if pattern.startswith("^MODULE"):
            return "MODULE"
        return "BLOCK"

    def _find_matching_end(self, lines: List[str], start: int, close_re) -> int:
        for idx in range(start + 1, len(lines)):
            if close_re.match(lines[idx].strip()):
                return idx
        # No terminator found (malformed/truncated source): treat rest of
        # file as part of this block rather than losing content.
        return len(lines) - 1

    def _window_chunks(self, remainder_lines, source_path=None) -> List[Chunk]:
        chunks: List[Chunk] = []
        step = self.fallback_window - self.fallback_overlap
        for start in range(0, len(remainder_lines), step):
            window = remainder_lines[start : start + self.fallback_window]
            if not window:
                continue
            content = "\n".join(line for _, line in window)
            if not content.strip():
                continue
            first_idx = window[0][0]
            last_idx = window[-1][0]
            unit_type = "HEADER" if _HEADER_KEYWORDS.match(content.strip()) else "STATEMENTS"
            chunks.append(
                Chunk(
                    content=content,
                    unit_type=unit_type,
                    name=None,
                    start_line=first_idx + 1,
                    end_line=last_idx + 1,
                    source_path=source_path,
                )
            )
            if start + self.fallback_window >= len(remainder_lines):
                break
        return chunks
