"""RAG Ibrido per SAP ABAP/HCM.

Hybrid retrieval-augmented generation pipeline specialized for mission
critical SAP ABAP/HCM repositories. Combines:

* Semantic vector search (ChromaDB + Ollama embeddings)
* Lexical scanning for DDIC terms (tables, domains, data elements)
* Content deduplication (V4 near-duplicate detection)
* Proprietary structural scoring of ABAP source code
* Generation through a local Qwen model served by Ollama
"""

from .config import RAGConfig

__version__ = "0.1.0"

__all__ = ["RAGConfig", "__version__"]
