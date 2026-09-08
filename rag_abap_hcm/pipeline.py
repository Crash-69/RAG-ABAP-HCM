"""End-to-end ingestion and query pipeline wiring together chunking, DDIC
scanning, deduplication, structural scoring, embeddings, the vector store
and the Qwen generator.
"""

from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional

from .chunker import ABAPChunker, Chunk
from .config import RAGConfig
from .dedup import ContentDeduplicatorV4
from .embeddings import OllamaEmbedder
from .llm import QwenGenerator
from .retriever import Candidate, HybridRetriever, RetrievedChunk
from .vector_store import VectorStore

DEFAULT_ABAP_GLOBS = ("**/*.abap", "**/*.txt")


def _chunk_id(source_path: str, chunk: Chunk) -> str:
    digest = hashlib.sha1(f"{source_path}:{chunk.start_line}:{chunk.end_line}".encode("utf-8"))
    return digest.hexdigest()


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: List[RetrievedChunk]


class ABAPHCMPipeline:
    """High level facade used by the CLI and by callers embedding the
    pipeline into other tools."""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.chunker = ABAPChunker()
        self.embedder = OllamaEmbedder(self.config)
        self.vector_store = VectorStore(self.config)
        self.retriever = HybridRetriever(self.config)
        self.generator = QwenGenerator(self.config)
        self.deduplicator = ContentDeduplicatorV4(
            similarity_threshold=self.config.dedup_similarity_threshold
        )

    def ingest_directory(self, root_dir: str, patterns=DEFAULT_ABAP_GLOBS) -> int:
        """Ingest all ABAP source files under `root_dir` into the vector
        store. Returns the number of chunks stored (after deduplication)."""
        file_paths: List[str] = []
        for pattern in patterns:
            file_paths.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
        file_paths = sorted(set(file_paths))

        all_chunks: List[Chunk] = []
        for path in file_paths:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            all_chunks.extend(self.chunker.chunk_text(text, source_path=path))

        if not all_chunks:
            return 0

        dedup_result = self.deduplicator.deduplicate([c.content for c in all_chunks])
        unique_chunks = [all_chunks[i] for i in dedup_result.kept_indices]

        embeddings = self.embedder.embed_batch([c.content for c in unique_chunks])
        ids = [_chunk_id(c.source_path or "", c) for c in unique_chunks]
        metadatas = [
            {
                "unit_type": c.unit_type,
                "name": c.name or "",
                "start_line": c.start_line,
                "end_line": c.end_line,
                "source_path": c.source_path or "",
            }
            for c in unique_chunks
        ]

        self.vector_store.add(
            ids=ids,
            documents=[c.content for c in unique_chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(unique_chunks)

    def query(self, question: str, top_k: Optional[int] = None) -> QueryResult:
        query_embedding = self.embedder.embed(question)
        raw_results = self.vector_store.query(
            query_embedding, top_k=(top_k or self.config.top_k) * 3
        )

        candidates = [
            Candidate(
                id=r["id"],
                content=r["document"] or "",
                metadata=r["metadata"] or {},
                vector_distance=r["distance"],
            )
            for r in raw_results
        ]

        ranked = self.retriever.rank(candidates, top_k=top_k)
        answer = self.generator.generate(question, [c.content for c in ranked])
        return QueryResult(question=question, answer=answer, sources=ranked)
