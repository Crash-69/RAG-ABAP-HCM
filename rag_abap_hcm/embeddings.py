"""Ollama embeddings client.

Wraps the Ollama HTTP API for generating text embeddings. The `requests`
dependency is imported lazily so that unit tests exercising the rest of the
pipeline do not require network access or the Ollama server to be running.
"""

from __future__ import annotations

from typing import List, Sequence

from .config import RAGConfig


class EmbeddingError(RuntimeError):
    """Raised when the Ollama embeddings API call fails."""


class OllamaEmbedder:
    """Generates embeddings for text using an Ollama-served model."""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        import requests  # lazy import: optional runtime dependency

        embeddings: List[List[float]] = []
        url = f"{self.config.ollama_host.rstrip('/')}/api/embeddings"
        for text in texts:
            try:
                response = requests.post(
                    url,
                    json={"model": self.config.embedding_model, "prompt": text},
                    timeout=self.config.request_timeout(),
                )
                response.raise_for_status()
            except requests.RequestException as exc:  # pragma: no cover - network
                raise EmbeddingError(f"Failed to embed text via Ollama: {exc}") from exc

            payload = response.json()
            embedding = payload.get("embedding")
            if embedding is None:
                raise EmbeddingError(
                    f"Ollama embeddings response missing 'embedding' field: {payload}"
                )
            embeddings.append(embedding)
        return embeddings
