from pathlib import Path
import re

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(r"C:\Progetto_AI")
DB_DIR = BASE_DIR / "Abap_VectorDB"

COLLECTION_NAME = "abap_hcm"

EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Numero massimo di risultati semantici iniziali
SEMANTIC_K = 30

# Numero massimo di risultati finali
FINAL_K = 15

# Batch utilizzato dalla ricerca esatta Chroma
EXACT_BATCH_SIZE = 500


# ============================================================
# TERMINI SAP
# ============================================================

SAP_TERMS = {

    "PA0001": [
        "PA0001",
        "P0001",
        "PERNR",
        "BUKRS",
        "WERKS",
        "PERSG",
        "PERSK",
        "BTRTL",
        "GSBER",
        "KOSTL",
        "ORGEH",
        "PLANS",
        "STELL",
        "SACHZ",
        "BEGDA",
        "ENDDA",
        "STAT2",
    ],

    "PA0002": [
        "PA0002",
        "P0002",
        "PERNR",
        "VORNA",
        "NACHN",
        "GESCH",
        "GBDAT",
        "GBLND",
        "FAMST",
        "SPRSL",
    ],

    "PA0007": [
        "PA0007",
        "P0007",
        "PERNR",
        "BEGDA",
        "ENDDA",
        "SCHKZ",
        "WOSTD",
        "MOSTD",
        "ZEITY",
    ],

    "PA0008": [
        "PA0008",
        "P0008",
        "PERNR",
        "BET01",
        "BET02",
        "BET03",
        "TRFAR",
        "TRFGB",
        "TRFGR",
        "TRFST",
    ],
}


# ============================================================
# NORMALIZZAZIONE
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = str(text).upper()

    # normalizzazione spazi
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# INIZIALIZZAZIONE VECTOR DB
# ============================================================

def get_vector_db():

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )

    return vector_db


# ============================================================
# TERMINI SAP DELLA QUERY
# ============================================================

def get_query_terms(query):

    query_upper = normalize_text(query)

    found_terms = []

    for table_name, terms in SAP_TERMS.items():

        for term in terms:

            pattern = r"\b" + re.escape(term.upper()) + r"\b"

            if re.search(pattern, query_upper):
                found_terms.append(term.upper())

    return list(dict.fromkeys(found_terms))


# ============================================================
# RICERCA SEMANTICA
# ============================================================

def ricerca_semantica(vector_db, query):

    results = vector_db.similarity_search_with_relevance_scores(
        query,
        k=SEMANTIC_K
    )

    return results


# ============================================================
# RICERCA ESATTA
# ============================================================

def ricerca_esatta(vector_db, query):

    collection = vector_db._collection

    query_upper = normalize_text(query)

    query_terms = get_query_terms(query)

    # Se non ci sono termini SAP riconosciuti,
    # utilizziamo comunque le parole tecniche della query.
    if not query_terms:
        query_terms = [
            word.upper()
            for word in re.findall(r"\b[A-Z0-9_]{3,}\b", query_upper)
        ]

    total = collection.count()

    risultati = []

    for offset in range(0, total, EXACT_BATCH_SIZE):

        batch_limit = min(
            EXACT_BATCH_SIZE,
            total - offset
        )

        data = collection.get(
            limit=batch_limit,
            offset=offset,
            include=["documents", "metadatas"]
        )

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        for document, metadata in zip(documents, metadatas):

            if not document:
                continue

            doc_upper = normalize_text(document)

            found_terms = []

            for term in query_terms:

                pattern = r"\b" + re.escape(term) + r"\b"

                if re.search(pattern, doc_upper):
                    found_terms.append(term)

            if not found_terms:
                continue

            exact_score = 0

            # ------------------------------------------------
            # Query esatta
            # ------------------------------------------------

            if query_upper in doc_upper:
                exact_score += 10

            # ------------------------------------------------
            # Presenza tabella PAxxxx
            # ------------------------------------------------

            for table_name in SAP_TERMS:

                if table_name in query_upper:

                    if table_name in doc_upper:
                        exact_score += 5

            # ------------------------------------------------
            # Termini SAP
            # ------------------------------------------------

            technical_terms = [
                term for term in found_terms
                if term not in SAP_TERMS
            ]

            exact_score += len(technical_terms)

            # massimo +5 per i termini aggiuntivi
            exact_score += min(
                len(found_terms) * 0.5,
                5
            )

            risultati.append({
                "document": document,
                "metadata": metadata or {},
                "exact_score": exact_score,
                "found_terms": found_terms
            })

    return risultati


# ============================================================
# DEDUPLICAZIONE
# ============================================================

def deduplica(risultati):

    unique = {}

    for result in risultati:

        metadata = result.get("metadata", {})

        chunk_hash = metadata.get("chunk_hash")

        if chunk_hash:
            key = chunk_hash

        else:
            key = (
                metadata.get("source_file"),
                metadata.get("chunk_index"),
                result.get("document", "")
            )

        if key not in unique:
            unique[key] = result

    return list(unique.values())


# ============================================================
# RANKING
# ============================================================

def ranking_ibrido(
    query,
    semantic_results,
    exact_results
):

    query_upper = normalize_text(query)

    ranked = {}

    # --------------------------------------------------------
    # RISULTATI SEMANTICI
    # --------------------------------------------------------

    for document, semantic_score in semantic_results:

        metadata = {}

        # similarity_search_with_relevance_scores
        # restituisce normalmente il Document
        # e lo score.

        try:
            metadata = document.metadata or {}
            content = document.page_content or ""
        except Exception:
            continue

        key = metadata.get("chunk_hash")

        if not key:
            key = (
                metadata.get("source_file"),
                metadata.get("chunk_index"),
                content
            )

        score = semantic_score * 10

        ranked[key] = {
            "document": content,
            "metadata": metadata,
            "semantic_score": semantic_score,
            "exact_score": 0,
            "found_terms": [],
            "final_score": score
        }

    # --------------------------------------------------------
    # RISULTATI ESATTI
    # --------------------------------------------------------

    for result in exact_results:

        metadata = result.get("metadata", {})

        key = metadata.get("chunk_hash")

        if not key:
            key = (
                metadata.get("source_file"),
                metadata.get("chunk_index"),
                result.get("document", "")
            )

        if key not in ranked:

            ranked[key] = {
                "document": result.get("document", ""),
                "metadata": metadata,
                "semantic_score": 0,
                "exact_score": 0,
                "found_terms": [],
                "final_score": 0
            }

        item = ranked[key]

        exact_score = result.get("exact_score", 0)

        item["exact_score"] = max(
            item["exact_score"],
            exact_score
        )

        item["found_terms"] = list(
            set(item["found_terms"])
            | set(result.get("found_terms", []))
        )

    # --------------------------------------------------------
    # CALCOLO SCORE FINALE
    # --------------------------------------------------------

    for item in ranked.values():

        metadata = item["metadata"]

        source_file = normalize_text(
            metadata.get("source_file", "")
        )

        object_name = normalize_text(
            metadata.get("object_name", "")
        )

        language = normalize_text(
            metadata.get("language", "")
        )

        final_score = (
            item["semantic_score"] * 10
            + item["exact_score"] * 3
        )

        # ----------------------------------------------------
        # Query presente tra i termini trovati
        # ----------------------------------------------------

        if query_upper in item["found_terms"]:
            final_score += 30

        # ----------------------------------------------------
        # Query nel nome file
        # ----------------------------------------------------

        if query_upper and query_upper in source_file:
            final_score += 20

        # ----------------------------------------------------
        # Object name esatto
        # ----------------------------------------------------

        if query_upper == object_name:
            final_score += 20

        # ----------------------------------------------------
        # Linguaggio ABAP
        # ----------------------------------------------------

        if language == "ABAP":
            final_score += 2

        # ----------------------------------------------------
        # Penalizzazione DDIC
        # ----------------------------------------------------

        if language == "SAP_DDIC":
            final_score -= 1

        item["final_score"] = final_score

    # --------------------------------------------------------
    # ORDINAMENTO
    # --------------------------------------------------------

    risultati = list(ranked.values())

    risultati.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # DEDUPLICAZIONE FINALE
    # --------------------------------------------------------

    risultati = deduplica(risultati)

    return risultati[:FINAL_K]


# ============================================================
# FUNZIONE PRINCIPALE
# ============================================================

def retrieve(query):

    vector_db = get_vector_db()

    semantic_results = ricerca_semantica(
        vector_db,
        query
    )

    exact_results = ricerca_esatta(
        vector_db,
        query
    )

    risultati = ranking_ibrido(
        query,
        semantic_results,
        exact_results
    )

    return risultati

