"""Central configuration for the RAG ABAP/HCM pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """Runtime configuration for the hybrid RAG pipeline.

    All values default to sensible local development settings and can be
    overridden via environment variables so that the pipeline can be wired
    into different deployments (docker-compose, CI, production) without code
    changes.
    """

    # ChromaDB
    chroma_persist_dir: str = field(
        default_factory=lambda: os.environ.get("RAG_CHROMA_DIR", ".chromadb")
    )
    chroma_collection: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_CHROMA_COLLECTION", "abap_hcm_chunks"
        )
    )

    # Ollama
    ollama_host: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_OLLAMA_HOST", "http://localhost:11434"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_EMBEDDING_MODEL", "nomic-embed-text"
        )
    )
    generation_model: str = field(
        default_factory=lambda: os.environ.get("RAG_GENERATION_MODEL", "qwen2.5:7b")
    )

    # Retrieval tuning
    top_k: int = field(default_factory=lambda: int(os.environ.get("RAG_TOP_K", "5")))
    vector_weight: float = field(
        default_factory=lambda: float(os.environ.get("RAG_VECTOR_WEIGHT", "0.55"))
    )
    lexical_weight: float = field(
        default_factory=lambda: float(os.environ.get("RAG_LEXICAL_WEIGHT", "0.25"))
    )
    structural_weight: float = field(
        default_factory=lambda: float(os.environ.get("RAG_STRUCTURAL_WEIGHT", "0.20"))
    )

    # Deduplication
    dedup_similarity_threshold: float = field(
        default_factory=lambda: float(os.environ.get("RAG_DEDUP_THRESHOLD", "0.90"))
    )

    def request_timeout(self) -> float:
        return float(os.environ.get("RAG_REQUEST_TIMEOUT", "30"))
