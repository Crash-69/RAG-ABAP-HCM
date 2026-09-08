"""ChromaDB-backed vector store wrapper.

`chromadb` is imported lazily so the rest of the pipeline (chunking, DDIC
scanning, deduplication, structural scoring, and hybrid ranking) can be
unit tested without the dependency installed or a persisted database
present.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .config import RAGConfig


class VectorStore:
    """Thin wrapper around a ChromaDB persistent collection."""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self._client = None
        self._collection = None

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb  # lazy import: optional runtime dependency

        self._client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.config.chroma_collection
        )
        return self._collection

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        collection = self._ensure_collection()
        collection.add(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(e) for e in embeddings],
            metadatas=list(metadatas) if metadatas is not None else None,
        )

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        collection = self._ensure_collection()
        result = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where=where,
        )
        return self._format_results(result)

    @staticmethod
    def _format_results(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        formatted = []
        for idx in range(len(ids)):
            formatted.append(
                {
                    "id": ids[idx],
                    "document": documents[idx] if idx < len(documents) else None,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "distance": distances[idx] if idx < len(distances) else None,
                }
            )
        return formatted

    def count(self) -> int:
        collection = self._ensure_collection()
        return collection.count()
