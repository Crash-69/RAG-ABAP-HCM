"""Ollama-served Qwen generation client."""

from __future__ import annotations

from typing import List, Optional

from .config import RAGConfig

SYSTEM_PROMPT = (
    "Sei un assistente esperto di sviluppo SAP ABAP e SAP HCM. Rispondi "
    "basandoti esclusivamente sul contesto di codice fornito. Se il "
    "contesto non contiene informazioni sufficienti, dichiaralo "
    "esplicitamente invece di inventare dettagli."
)


class GenerationError(RuntimeError):
    """Raised when the Ollama chat/generate API call fails."""


class QwenGenerator:
    """Generates answers grounded in retrieved ABAP/HCM context via Qwen."""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        context_block = "\n\n---\n\n".join(contexts) if contexts else "(nessun contesto trovato)"
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Contesto recuperato:\n{context_block}\n\n"
            f"Domanda: {question}\n"
            f"Risposta:"
        )

    def generate(self, question: str, contexts: List[str]) -> str:
        import requests  # lazy import: optional runtime dependency

        prompt = self.build_prompt(question, contexts)
        url = f"{self.config.ollama_host.rstrip('/')}/api/generate"
        try:
            response = requests.post(
                url,
                json={
                    "model": self.config.generation_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=self.config.request_timeout(),
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise GenerationError(f"Failed to generate response via Ollama: {exc}") from exc

        payload = response.json()
        text = payload.get("response")
        if text is None:
            raise GenerationError(
                f"Ollama generate response missing 'response' field: {payload}"
            )
        return text
