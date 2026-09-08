import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_abap_hcm.config import RAGConfig
from rag_abap_hcm.retriever import Candidate, HybridRetriever


def make_candidate(id_, content, vector_distance):
    return Candidate(id=id_, content=content, metadata={"source_path": id_}, vector_distance=vector_distance)


def test_rank_prefers_higher_vector_similarity_when_lexical_and_structural_equal():
    retriever = HybridRetriever(RAGConfig())
    candidates = [
        make_candidate("a", "WRITE 'hello'.", vector_distance=1.8),
        make_candidate("b", "WRITE 'hello there'.", vector_distance=0.2),
    ]
    ranked = retriever.rank(candidates, top_k=2)
    assert ranked[0].id == "b"


def test_rank_deduplicates_near_identical_content():
    retriever = HybridRetriever(RAGConfig())
    duplicate_text = "FORM foo.\nWRITE 'dup'.\nENDFORM."
    candidates = [
        make_candidate("a", duplicate_text, vector_distance=0.5),
        make_candidate("b", duplicate_text, vector_distance=0.4),
        make_candidate("c", "FORM totally_different.\nCALL FUNCTION 'Z_XYZ'.\nENDFORM.", vector_distance=0.9),
    ]
    ranked = retriever.rank(candidates, top_k=5)
    ids = {c.id for c in ranked}
    assert len(ids) == 2  # one of a/b removed as duplicate, c kept
    assert "c" in ids


def test_rank_boosts_ddic_heavy_chunks_lexically():
    retriever = HybridRetriever(RAGConfig())
    hcm_chunk = make_candidate(
        "hcm",
        "TABLES: pa0002.\nSELECT SINGLE * FROM pa0002 INTO ls_pa0002 WHERE pernr = p_pernr.",
        vector_distance=1.0,
    )
    generic_chunk = make_candidate(
        "generic", "DATA: lv_x TYPE i.\nlv_x = 1.", vector_distance=1.0
    )
    ranked = retriever.rank([hcm_chunk, generic_chunk], top_k=2)
    by_id = {c.id: c for c in ranked}
    assert by_id["hcm"].lexical_score > by_id["generic"].lexical_score
    assert by_id["hcm"].combined_score > by_id["generic"].combined_score


def test_empty_candidates_returns_empty_list():
    retriever = HybridRetriever(RAGConfig())
    assert retriever.rank([]) == []


def test_top_k_limits_results():
    retriever = HybridRetriever(RAGConfig())
    candidates = [make_candidate(str(i), f"FORM f{i}.\nWRITE '{i}'.\nENDFORM.", vector_distance=1.0) for i in range(10)]
    ranked = retriever.rank(candidates, top_k=3)
    assert len(ranked) == 3


def test_distance_to_similarity_bounds():
    assert HybridRetriever.distance_to_similarity(0.0) == 1.0
    assert HybridRetriever.distance_to_similarity(2.0) == 0.0
    assert HybridRetriever.distance_to_similarity(None) == 0.0
