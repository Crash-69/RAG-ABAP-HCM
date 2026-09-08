"""Hybrid retriever combining vector search, lexical DDIC scan, content
deduplication and structural scoring for SAP ABAP/HCM source retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import RAGConfig
from .ddic_scanner import DDICScanner
from .dedup import ContentDeduplicatorV4
from .structural_scorer import StructuralScorer


@dataclass
class RetrievedChunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    vector_score: float
    lexical_score: float
    structural_score: float
    combined_score: float = 0.0


@dataclass
class Candidate:
    """A candidate document to be scored by the hybrid retriever."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_distance: Optional[float] = None
    vector_score: Optional[float] = None


class HybridRetriever:
    """Blends vector similarity, DDIC lexical relevance and structural
    quality into a single ranked list of chunks, after removing
    near-duplicate content (V4 deduplication).
    """

    def __init__(
        self,
        config: RAGConfig = None,
        ddic_scanner: DDICScanner = None,
        deduplicator: ContentDeduplicatorV4 = None,
        structural_scorer: StructuralScorer = None,
    ):
        self.config = config or RAGConfig()
        self.ddic_scanner = ddic_scanner or DDICScanner()
        self.deduplicator = deduplicator or ContentDeduplicatorV4(
            similarity_threshold=self.config.dedup_similarity_threshold
        )
        self.structural_scorer = structural_scorer or StructuralScorer()

    @staticmethod
    def distance_to_similarity(distance: Optional[float]) -> float:
        """Convert a ChromaDB distance (lower = more similar) into a
        normalized similarity score in [0, 1]. Assumes cosine distance in
        [0, 2]; clamps defensively for other metrics."""
        if distance is None:
            return 0.0
        similarity = 1.0 - (distance / 2.0)
        return min(max(similarity, 0.0), 1.0)

    def rank(self, candidates: Sequence[Candidate], top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if not candidates:
            return []

        top_k = top_k or self.config.top_k

        # 1. Deduplicate near-identical content before scoring/ranking so
        #    duplicated boilerplate doesn't crowd out the result set.
        contents = [c.content for c in candidates]
        dedup_result = self.deduplicator.deduplicate(contents)
        unique_candidates = [candidates[i] for i in dedup_result.kept_indices]

        # 2. Score each surviving candidate on vector, lexical and
        #    structural axes, then blend them per the configured weights.
        scored: List[RetrievedChunk] = []
        for candidate in unique_candidates:
            if candidate.vector_score is not None:
                vector_score = candidate.vector_score
            else:
                vector_score = self.distance_to_similarity(candidate.vector_distance)
            lexical_score = self.ddic_scanner.score(candidate.content)
            structural_score = self.structural_scorer.score(candidate.content).total

            combined = (
                vector_score * self.config.vector_weight
                + lexical_score * self.config.lexical_weight
                + structural_score * self.config.structural_weight
            )
            scored.append(
                RetrievedChunk(
                    id=candidate.id,
                    content=candidate.content,
                    metadata=candidate.metadata,
                    vector_score=vector_score,
                    lexical_score=lexical_score,
                    structural_score=structural_score,
                    combined_score=combined,
                )
            )

        scored.sort(key=lambda c: c.combined_score, reverse=True)
        return scored[:top_k]
