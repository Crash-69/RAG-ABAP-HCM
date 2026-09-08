import re
from pathlib import Path
from collections import defaultdict

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURAZIONE
# ============================================================

DB_DIR = Path(r"C:\Progetto_AI\Abap_VectorDB")

COLLECTION_NAME = "abap_hcm"

EMBEDDING_MODEL = "nomic-embed-text:latest"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Numero risultati della ricerca semantica
SEMANTIC_K = 30

# Numero massimo risultati finali mostrati
FINAL_K = 15

# Quanti documenti Chroma leggere per la ricerca esatta
# None = tutti
EXACT_SCAN_LIMIT = None


# ============================================================
# TERMINI SAP CONOSCIUTI
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
        "GBDAT",
        "GESCH",
        "FAMST",
    ],

    "PA0007": [
        "PA0007",
        "P0007",
        "PERNR",
        "WOSTD",
        "SCHKZ",
        "ZTERF",
        "ARBPL",
    ],

    "PA0008": [
        "PA0008",
        "P0008",
        "PERNR",
        "BET01",
        "BET02",
        "WAERS",
        "TRFAR",
        "TRFGB",
        "TRFGR",
        "TRFST",
    ],
}


# ============================================================
# UTILITY
# ============================================================

def normalizza_testo(text):
    """
    Normalizzazione utilizzata esclusivamente per il confronto
    lessicale.
    """
    if not text:
        return ""

    return re.sub(
        r"[^A-Z0-9_/~]",
        " ",
        text.upper()
    )


def trova_termini_esatti(text, termini):
    """
    Restituisce i termini SAP effettivamente presenti nel testo.
    """

    normalized = normalizza_testo(text)

    trovati = []

    for termine in termini:

        termine_norm = normalizza_testo(termine)

        if not termine_norm:
            continue

        # Boundary permissivo ma evita falsi positivi grossolani.
        pattern = rf"(?<![A-Z0-9_]){re.escape(termine_norm)}(?![A-Z0-9_])"

        if re.search(pattern, normalized):
            trovati.append(termine)

    return trovati


def estrai_query_terms(query):
    """
    Estrae automaticamente i token significativi dalla query.
    """

    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_/~]*",
        query.upper()
    )

    return list(dict.fromkeys(tokens))


def termini_sap_per_query(query):
    """
    Restituisce i termini SAP associati alla query.
    """

    query_upper = query.upper()

    termini = []

    # Match diretto nella knowledge base SAP.
    for key, values in SAP_TERMS.items():

        if key in query_upper:
            termini.extend(values)

    # Aggiungiamo comunque i token della query.
    termini.extend(estrai_query_terms(query))

    return list(dict.fromkeys(termini))


# ============================================================
# CHROMA
# ============================================================

def crea_vectordb():

    print()
    print("=" * 70)
    print("INIZIALIZZAZIONE CHROMADB")
    print("=" * 70)

    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"VectorDB non trovato: {DB_DIR}"
        )

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
# RICERCA SEMANTICA
# ============================================================

def ricerca_semantica(vector_db, query):

    print()
    print("=" * 70)
    print("RICERCA SEMANTICA")
    print("=" * 70)

    print(f"Query: {query}")
    print(f"K    : {SEMANTIC_K}")

    risultati = vector_db.similarity_search_with_relevance_scores(
        query,
        k=SEMANTIC_K
    )

    risultati_semantici = []

    for doc, score in risultati:

        risultati_semantici.append({
            "document": doc,
            "semantic_score": float(score),
            "exact_score": 0.0,
            "final_score": float(score),
            "exact_terms": [],
            "key": None
        })

    print(
        f"Risultati semantici ottenuti: "
        f"{len(risultati_semantici)}"
    )

    return risultati_semantici


# ============================================================
# RICERCA ESATTA
# ============================================================

def ricerca_esatta(vector_db, query):

    print()
    print("=" * 70)
    print("RICERCA ESATTA TERMINI SAP")
    print("=" * 70)

    termini = termini_sap_per_query(query)

    print("Termini ricercati:")

    for termine in termini:
        print(f"  - {termine}")

    print()

    collection = vector_db._collection

    total = collection.count()

    print(f"Chunk presenti nel VectorDB: {total}")

    # ========================================================
    # IMPORTANTE:
    # Non chiediamo a Chroma tutti i 46.346 record in una sola
    # chiamata. Vengono letti a piccoli batch per evitare:
    #
    # chromadb.errors.InternalError:
    # too many SQL variables
    # ========================================================

    BATCH_SIZE = 500

    risultati = []

    elaborati = 0

    for offset in range(0, total, BATCH_SIZE):

        batch_limit = min(
            BATCH_SIZE,
            total - offset
        )

        data = collection.get(
            limit=batch_limit,
            offset=offset,
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        for index, content in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            if not content:
                continue

            trovati = trova_termini_esatti(
                content,
                termini
            )

            if not trovati:
                continue

            # ------------------------------------------------
            # Score lessicale
            # ------------------------------------------------

            score = 0.0

            for termine in trovati:

                termine_upper = termine.upper()

                # Match esatto della query.
                if termine_upper == query.upper():
                    score += 10.0

                # Tabelle PAxxxx.
                elif termine_upper.startswith("PA"):
                    score += 5.0

                # Altri termini tecnici SAP.
                else:
                    score += 1.0

            # Bonus per quantità di termini trovati.
            score += min(
                len(trovati) * 0.5,
                5.0
            )

            risultati.append({
                "document": None,
                "semantic_score": 0.0,
                "exact_score": score,
                "final_score": score,
                "exact_terms": trovati,
                "metadata": metadata,
                "content": content,
                "key": None
            })

        elaborati += len(documents)

        # ----------------------------------------------------
        # Progressivo
        # ----------------------------------------------------

        if (
            elaborati % 5000 == 0
            or elaborati >= total
        ):
            print(
                f"  Scansionati: "
                f"{elaborati}/{total} "
                f"| Match: {len(risultati)}"
            )

    print()
    print(
        f"Chunk contenenti termini esatti: "
        f"{len(risultati)}"
    )

    return risultati

# ============================================================
# CHIAVE DEDUPLICAZIONE
# ============================================================

def chiave_documento(item):

    doc = item.get("document")

    if doc is not None:
        metadata = doc.metadata

        return (
            metadata.get("source_file"),
            metadata.get("chunk_index"),
            metadata.get("chunk_hash")
        )

    metadata = item.get("metadata") or {}

    return (
        metadata.get("source_file"),
        metadata.get("chunk_index"),
        metadata.get("chunk_hash")
    )


# ============================================================
# FUSIONE DEI RISULTATI
# ============================================================

def fondi_risultati(
    risultati_semantici,
    risultati_esatti
):

    print()
    print("=" * 70)
    print("FUSIONE + DEDUPLICAZIONE")
    print("=" * 70)

    merged = {}

    # --------------------------------------------------------
    # Prima inseriamo i risultati semantici.
    # --------------------------------------------------------

    for item in risultati_semantici:

        key = chiave_documento(item)

        item["key"] = key

        merged[key] = item

    # --------------------------------------------------------
    # Poi fondiamo quelli esatti.
    # --------------------------------------------------------

    for exact in risultati_esatti:

        key = chiave_documento(exact)

        exact["key"] = key

        if key in merged:

            existing = merged[key]

            existing["exact_score"] = (
                exact["exact_score"]
            )

            existing["exact_terms"] = (
                exact["exact_terms"]
            )

            # Il documento esiste già come risultato semantico.
            # Manteniamo il Document originale.

        else:

            # Risultato trovato esclusivamente
            # dalla ricerca lessicale.
            from langchain_core.documents import Document

            doc = Document(
                page_content=exact["content"],
                metadata=exact["metadata"]
            )

            exact["document"] = doc

            merged[key] = exact

    print(
        f"Risultati semantici : "
        f"{len(risultati_semantici)}"
    )

    print(
        f"Risultati esatti    : "
        f"{len(risultati_esatti)}"
    )

    print(
        f"Risultati unici     : "
        f"{len(merged)}"
    )

    return list(merged.values())


# ============================================================
# RANKING
# ============================================================

def calcola_final_score(item, query):

    semantic = item.get(
        "semantic_score",
        0.0
    )

    exact = item.get(
        "exact_score",
        0.0
    )

    terms = item.get(
        "exact_terms",
        []
    )

    metadata = (
        item["document"].metadata
        if item.get("document")
        else {}
    )

    score = 0.0

    # --------------------------------------------------------
    # Componente semantica
    # --------------------------------------------------------

    score += semantic * 10.0

    # --------------------------------------------------------
    # Componente exact match
    # --------------------------------------------------------

    score += exact * 3.0

    # --------------------------------------------------------
    # Bonus molto forte se compare esattamente la query.
    # --------------------------------------------------------

    query_upper = query.upper()

    if query_upper in [
        term.upper()
        for term in terms
    ]:
        score += 30.0

    # --------------------------------------------------------
    # Bonus se il source file riguarda direttamente
    # l'oggetto cercato.
    # --------------------------------------------------------

    source_file = (
        metadata.get(
            "source_file",
            ""
        ).upper()
    )

    object_name = (
        metadata.get(
            "object_name",
            ""
        ).upper()
    )

    if query_upper in source_file:
        score += 20.0

    if query_upper == object_name:
        score += 20.0

    # --------------------------------------------------------
    # Bonus ABAP
    # --------------------------------------------------------

    if metadata.get("language") == "ABAP":
        score += 2.0

    # --------------------------------------------------------
    # Penalizzazione leggera dei risultati DDIC HTML
    # quando abbiamo codice ABAP disponibile.
    # --------------------------------------------------------

    if metadata.get("language") == "SAP_DDIC":
        score -= 1.0

    return score


def applica_ranking(risultati, query):

    for item in risultati:

        item["final_score"] = calcola_final_score(
            item,
            query
        )

    risultati.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return risultati[:FINAL_K]


# ============================================================
# STAMPA RISULTATI
# ============================================================

def stampa_risultati(risultati, query):

    print()
    print("=" * 70)
    print("RISULTATI FINALI - HYBRID RANKING")
    print("=" * 70)

    print(
        f"Query: {query}"
    )

    print(
        f"Risultati mostrati: {len(risultati)}"
    )

    print()

    for posizione, item in enumerate(
        risultati,
        start=1
    ):

        doc = item["document"]

        metadata = doc.metadata

        content = doc.page_content

        print("-" * 70)

        print(
            f"RANK #{posizione}"
        )

        print(
            f"Final score    : "
            f"{item['final_score']:.4f}"
        )

        print(
            f"Semantic score : "
            f"{item['semantic_score']:.4f}"
        )

        print(
            f"Exact score    : "
            f"{item['exact_score']:.4f}"
        )

        print(
            f"Exact terms    : "
            f"{', '.join(item['exact_terms']) or '-'}"
        )

        print(
            f"Source file    : "
            f"{metadata.get('source_file', '-')}"
        )

        print(
            f"Source path    : "
            f"{metadata.get('source_path', '-')}"
        )

        print(
            f"Object type    : "
            f"{metadata.get('object_type', '-')}"
        )

        print(
            f"Object name    : "
            f"{metadata.get('object_name', '-')}"
        )

        print(
            f"Language       : "
            f"{metadata.get('language', '-')}"
        )

        print(
            f"Chunk index    : "
            f"{metadata.get('chunk_index', '-')}"
        )

        print()

        # Mostriamo una quantità sufficiente per verificare
        # manualmente il codice recuperato.
        print("CODICE / CONTENUTO:")

        print("-" * 70)

        print(
            content[:2500]
        )

        if len(content) > 2500:
            print()
            print(
                "... [contenuto troncato]"
            )


# ============================================================
# DIAGNOSTICA
# ============================================================

def diagnostica(risultati, query):

    print()
    print("=" * 70)
    print("DIAGNOSTICA RETRIEVAL")
    print("=" * 70)

    query_upper = query.upper()

    exact_query_count = 0
    abap_count = 0

    for item in risultati:

        doc = item["document"]

        metadata = doc.metadata

        if query_upper in [
            term.upper()
            for term in item["exact_terms"]
        ]:
            exact_query_count += 1

        if metadata.get("language") == "ABAP":
            abap_count += 1

    print(
        f"Risultati finali             : "
        f"{len(risultati)}"
    )

    print(
        f"Risultati ABAP               : "
        f"{abap_count}"
    )

    print(
        f"Risultati con match esatto   : "
        f"{exact_query_count}"
    )

    print()

    if exact_query_count > 0 and abap_count > 0:

        print(
            "ESITO: RETRIEVAL PROMETTENTE"
        )

        print(
            "La query è stata trovata "
            "all'interno di contenuto ABAP."
        )

    elif exact_query_count > 0:

        print(
            "ESITO: MATCH ESATTO MA "
            "CODICE ABAP DA VERIFICARE"
        )

    elif abap_count > 0:

        print(
            "ESITO: SOLO MATCH SEMANTICO"
        )

        print(
            "Non è stata trovata la stringa "
            "esatta della query nei risultati finali."
        )

    else:

        print(
            "ESITO: RETRIEVAL DA MIGLIORARE"
        )

        print(
            "Nessun risultato ABAP pertinente "
            "nei primi risultati."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    query = input(
        "Inserisci la query SAP/ABAP "
        "(es. PA0001): "
    ).strip()

    if not query:
        print(
            "[ERRORE] Query vuota."
        )
        return

    vector_db = crea_vectordb()

    # --------------------------------------------------------
    # 1. SEMANTIC SEARCH
    # --------------------------------------------------------

    risultati_semantici = ricerca_semantica(
        vector_db,
        query
    )

    # --------------------------------------------------------
    # 2. EXACT SAP SEARCH
    # --------------------------------------------------------

    risultati_esatti = ricerca_esatta(
        vector_db,
        query
    )

    # --------------------------------------------------------
    # 3. MERGE + DEDUP
    # --------------------------------------------------------

    risultati = fondi_risultati(
        risultati_semantici,
        risultati_esatti
    )

    # --------------------------------------------------------
    # 4. RANKING
    # --------------------------------------------------------

    risultati = applica_ranking(
        risultati,
        query
    )

    # --------------------------------------------------------
    # 5. OUTPUT
    # --------------------------------------------------------

    stampa_risultati(
        risultati,
        query
    )

    # --------------------------------------------------------
    # 6. DIAGNOSTICA
    # --------------------------------------------------------

    diagnostica(
        risultati,
        query
    )


if __name__ == "__main__":
    main()
