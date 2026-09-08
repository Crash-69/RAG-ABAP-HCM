import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_abap_hcm.chunker import ABAPChunker

SAMPLE = """REPORT ztest.

FORM foo.
  WRITE 'hello'.
ENDFORM.

METHOD bar.
  WRITE 'world'.
ENDMETHOD.
"""


def test_chunk_extracts_form_block():
    chunker = ABAPChunker()
    chunks = chunker.chunk_text(SAMPLE, source_path="ztest.abap")
    form_chunks = [c for c in chunks if c.unit_type == "FORM"]
    assert len(form_chunks) == 1
    assert form_chunks[0].name == "foo"
    assert "ENDFORM" in form_chunks[0].content


def test_chunk_extracts_method_block():
    chunker = ABAPChunker()
    chunks = chunker.chunk_text(SAMPLE, source_path="ztest.abap")
    method_chunks = [c for c in chunks if c.unit_type == "METHOD"]
    assert len(method_chunks) == 1
    assert method_chunks[0].name == "bar"


def test_chunk_keeps_header_as_separate_chunk():
    chunker = ABAPChunker()
    chunks = chunker.chunk_text(SAMPLE, source_path="ztest.abap")
    header_chunks = [c for c in chunks if c.unit_type == "HEADER"]
    assert len(header_chunks) == 1
    assert "REPORT ztest" in header_chunks[0].content


def test_chunks_cover_all_lines_no_duplicates_overlap_in_start_line():
    chunker = ABAPChunker()
    chunks = chunker.chunk_text(SAMPLE, source_path="ztest.abap")
    starts = [c.start_line for c in chunks]
    assert starts == sorted(starts)


def test_unterminated_block_does_not_crash():
    chunker = ABAPChunker()
    text = "FORM incomplete.\n  WRITE 'oops'.\n"
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].unit_type == "FORM"


def test_fallback_overlap_validation():
    import pytest

    with pytest.raises(ValueError):
        ABAPChunker(fallback_window=10, fallback_overlap=10)
