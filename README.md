# RAG-ABAP-HCM

RAG Ibrido per SAP ABAP/HCM con ChromaDB, Ollama e Qwen

Pipeline di retrieval specializzata per repository SAP ABAP/HCM
mission-critical. Integra bidirezionalmente:

* **Ricerca semantica vettoriale** — chunking strutturale del codice ABAP
  (per `FORM`/`METHOD`/`FUNCTION`/`CLASS`), embedding tramite un modello
  servito da [Ollama](https://ollama.ai) e indicizzazione/ricerca su
  [ChromaDB](https://www.trychroma.com/).
* **Lexical scan per termini DDIC** — riconoscimento di tabelle infotipo
  (`PAxxxx`), tabelle di Organizational Management (`HRPxxxx`), tabelle di
  customizing (`Txxx`) e altri riferimenti al Dizionario Dati SAP.
* **Content deduplication V4** — rimozione di duplicati esatti e
  quasi-duplicati (SimHash) per evitare che routine clonate inquinino i
  risultati del retrieval.
* **Scoring strutturale proprietario** — punteggio di qualità basato su
  completezza del blocco ABAP, dimensione, complessità di nesting e
  rilevanza per operazioni HCM (accesso infotipi, `CALL FUNCTION`, ecc.).
* **Generazione con Qwen** — le risposte sono generate da un modello Qwen
  servito localmente via Ollama, basandosi esclusivamente sul contesto
  recuperato.

## Installazione

```bash
pip install -r requirements.txt
```

## Utilizzo

Ingestione di un repository di sorgenti ABAP in ChromaDB:

```bash
python -m rag_abap_hcm.cli ingest /percorso/ai/sorgenti/abap
```

Interrogazione della pipeline:

```bash
python -m rag_abap_hcm.cli query "Come viene letta l'infotipo 0002?"
```

Configurazione tramite variabili d'ambiente (vedi `rag_abap_hcm/config.py`):
`RAG_CHROMA_DIR`, `RAG_CHROMA_COLLECTION`, `RAG_OLLAMA_HOST`,
`RAG_EMBEDDING_MODEL`, `RAG_GENERATION_MODEL`, `RAG_TOP_K`,
`RAG_VECTOR_WEIGHT`, `RAG_LEXICAL_WEIGHT`, `RAG_STRUCTURAL_WEIGHT`,
`RAG_DEDUP_THRESHOLD`.

## Test

```bash
pip install -r requirements-dev.txt
pytest
```
