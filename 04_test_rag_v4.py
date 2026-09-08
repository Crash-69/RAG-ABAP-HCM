import re
import time
import hashlib

from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ============================================================
# CONFIGURAZIONE
# ============================================================

DB_DIR = Path(r"C:\Progetto_AI\Abap_VectorDB")

COLLECTION_NAME = "abap_hcm"

EMBEDDING_MODEL = "nomic-embed-text:latest"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Ricerca semantica
SEMANTIC_K = 30

# Risultati finali
FINAL_K = 15

# Batch utilizzato per la ricerca esatta
EXACT_BATCH_SIZE = 500

# Numero massimo di risultati provenienti dallo stesso source_file
MAX_RESULTS_PER_SOURCE = 3

# Mostra questo numero di caratteri per risultato
DISPLAY_CHARS = 2500


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
# PATTERN STRUTTURALI ABAP
# ============================================================

STRUCTURAL_PATTERNS = {

    # Accesso diretto alla tabella
    "from_pa_table": (
        r"\bFROM\s+PA\d{4}\b",
        8.0
    ),

    # SELECT ... FROM PAxxxx
    "select_from_pa": (
        r"\bSELECT\b[\s\S]{0,500}\bFROM\s+PA\d{4}\b",
        10.0
    ),

    # SELECT SINGLE ... FROM PAxxxx
    "select_single_from_pa": (
        r"\bSELECT\s+SINGLE\b[\s\S]{0,500}"
        r"\bFROM\s+PA\d{4}\b",
        14.0
    ),

    # Campo PAxxxx-FIELD
    "pa_field": (
        r"\bPA\d{4}-[A-Z0-9_]+\b",
        3.0
    ),

    # WHERE con PERNR
    "where_pernr": (
        r"\bWHERE\b[\s\S]{0,300}\bPERNR\b",
        8.0
    ),

    # WHERE con KOSTL
    "where_kostl": (
        r"\bWHERE\b[\s\S]{0,300}\bKOSTL\b",
        7.0
    ),

    # LOOP AT PAxxxx
    "loop_pa": (
        r"\bLOOP\s+AT\s+PA\d{4}\b",
        6.0
    ),

    # READ TABLE ... PAxxxx
    "read_pa": (
        r"\bREAD\s+TABLE\b[\s\S]{0,300}\bPA\d{4}\b",
        5.0
    ),

    # MODIFY PAxxxx
    "modify_pa": (
        r"\bMODIFY\s+PA\d{4}\b",
        4.0
    ),

    # UPDATE PAxxxx
    "update_pa": (
        r"\bUPDATE\s+PA\d{4}\b",
        4.0
    ),

    # INSERT PAxxxx
    "insert_pa": (
        r"\bINSERT\s+INTO\s+PA\d{4}\b",
        4.0
    ),

    # DELETE FROM PAxxxx
    "delete_pa": (
        r"\bDELETE\s+FROM\s+PA\d{4}\b",
        4.0
    ),
}


# ============================================================
# UTILITY
# ============================================================

def normalizza_testo(text):
    """
    Normalizzazione utilizzata esclusivamente
    per confronti lessicali.
    """

    if not text:
        return ""

    return re.sub(
        r"[^A-Z0-9_/\~]",
        " ",
        text.upper()
    )


def trova_termini_esatti(text, termini):
    """
    Restituisce i termini SAP effettivamente
    presenti nel testo.
    """

    normalized = normalizza_testo(text)

    trovati = []

    for termine in termini:

        termine_norm = normalizza_testo(termine)

        if not termine_norm:
            continue

        pattern = (
            rf"(?<![A-Z0-9_])"
            rf"{re.escape(termine_norm)}"
            rf"(?![A-Z0-9_])"
        )

        if re.search(pattern, normalized):
            trovati.append(termine)

    return trovati


def estrai_query_terms(query):
    """
    Estrae token tecnici dalla query.
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

    for key, values in SAP_TERMS.items():

        if key in query_upper:
            termini.extend(values)

    termini.extend(
        estrai_query_terms(query)
    )

    return list(
        dict.fromkeys(termini)
    )


# ============================================================
# NORMALIZZAZIONE CONTENUTO
# ============================================================

def normalizza_codice_per_hash(content):
    """
    Normalizza il codice per individuare chunk
    sostanzialmente identici provenienti da file diversi.

    Non modifica il contenuto originale.
    """

    if not content:
        return ""

    text = content.upper()

    # Rimuove commenti ABAP semplici
    lines = []

    for line in text.splitlines():

        stripped = line.strip()

        if stripped.startswith("*"):
            continue

        if stripped.startswith('"'):
            continue

        lines.append(stripped)

    text = "\n".join(lines)

    # Riduce spazi multipli
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def hash_contenuto(content):
    """
    SHA256 del contenuto normalizzato.
    """

    normalized = normalizza_codice_per_hash(
        content
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()


# ============================================================
# CHROMA
# ============================================================

def crea_vectordb():

    print()
    print("=" * 70)
    print("INIZIALIZZAZIONE CHROMADB V4")
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
    print("RICERCA SEMANTICA V4")
    print("=" * 70)

    print(f"Query: {query}")
    print(f"K    : {SEMANTIC_K}")

    risultati = (
        vector_db
        .similarity_search_with_relevance_scores(
            query,
            k=SEMANTIC_K
        )
    )

    risultati_semantici = []

    for doc, score in risultati:

        risultati_semantici.append({

            "document": doc,

            "semantic_score": float(score),

            "exact_score": 0.0,

            "final_score": float(score),

            "exact_terms": [],

            "structural_matches": [],

            "structural_score": 0.0,

            "penalty": 0.0,

            "key": None,

            "content_hash": None,
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
    print("RICERCA ESATTA TERMINI SAP V4")
    print("=" * 70)

    termini = termini_sap_per_query(query)

    print("Termini ricercati:")

    for termine in termini:
        print(f"  - {termine}")

    print()

    collection = vector_db._collection

    total = collection.count()

    print(
        f"Chunk presenti nel VectorDB: {total}"
    )

    risultati = []

    elaborati = 0

    for offset in range(
        0,
        total,
        EXACT_BATCH_SIZE
    ):

        batch_limit = min(
            EXACT_BATCH_SIZE,
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

        documents = (
            data.get("documents")
            or []
        )

        metadatas = (
            data.get("metadatas")
            or []
        )

        for index, content in enumerate(
            documents
        ):

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
            # SCORE LESSICALE
            # ------------------------------------------------

            score = 0.0

            for termine in trovati:

                termine_upper = termine.upper()

                if termine_upper == query.upper():

                    score += 10.0

                elif termine_upper.startswith("PA"):

                    score += 5.0

                else:

                    score += 1.0

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

                "structural_matches": [],

                "structural_score": 0.0,

                "penalty": 0.0,

                "metadata": metadata,

                "content": content,

                "key": None,

                "content_hash": hash_contenuto(
                    content
                ),
            })

        elaborati += len(documents)

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
# CHIAVE DOCUMENTO
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

    metadata = (
        item.get("metadata")
        or {}
    )

    return (
        metadata.get("source_file"),
        metadata.get("chunk_index"),
        metadata.get("chunk_hash")
    )


# ============================================================
# FUSIONE
# ============================================================

def fondi_risultati(
    risultati_semantici,
    risultati_esatti
):

    print()
    print("=" * 70)
    print("FUSIONE + DEDUPLICAZIONE V4")
    print("=" * 70)

    merged = {}

    # --------------------------------------------------------
    # Risultati semantici
    # --------------------------------------------------------

    for item in risultati_semantici:

        key = chiave_documento(item)

        item["key"] = key

        if not item.get("content_hash"):

            item["content_hash"] = hash_contenuto(
                item["document"].page_content
            )

        merged[key] = item

    # --------------------------------------------------------
    # Risultati esatti
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

            existing["content_hash"] = (
                exact["content_hash"]
            )

        else:

            doc = Document(
                page_content=exact["content"],
                metadata=exact["metadata"]
            )

            exact["document"] = doc

            merged[key] = exact

    risultati = list(
        merged.values()
    )

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
        f"{len(risultati)}"
    )

    # --------------------------------------------------------
    # DEDUPLICAZIONE CONTENUTO
    # --------------------------------------------------------

    by_content = {}

    duplicati = 0

    for item in risultati:

        content_hash = item.get(
            "content_hash"
        )

        if not content_hash:

            content_hash = hash_contenuto(
                item["document"].page_content
            )

            item["content_hash"] = (
                content_hash
            )

        if content_hash not in by_content:

            by_content[
                content_hash
            ] = item

        else:

            duplicati += 1

            existing = by_content[
                content_hash
            ]

            # Manteniamo quello con
            # migliore semantic score.
            if (
                item["semantic_score"]
                > existing["semantic_score"]
            ):

                by_content[
                    content_hash
                ] = item

            # Se semantic score uguale,
            # manteniamo quello con exact score maggiore.
            elif (
                item["semantic_score"]
                == existing["semantic_score"]
                and
                item["exact_score"]
                > existing["exact_score"]
            ):

                by_content[
                    content_hash
                ] = item

    risultati = list(
        by_content.values()
    )

    print(
        f"Duplicati per contenuto rimossi: "
        f"{duplicati}"
    )

    print(
        f"Risultati dopo content dedup   : "
        f"{len(risultati)}"
    )

    return risultati


# ============================================================
# ANALISI STRUTTURALE ABAP
# ============================================================

def analizza_struttura_abap(
    content,
    query
):

    if not content:

        return {
            "score": 0.0,
            "matches": [],
            "has_select": False,
            "has_from": False,
            "has_where": False,
            "has_direct_read": False,
            "declaration_only": False,
        }

    text = content.upper()

    score = 0.0

    matches = []

    has_select = bool(
        re.search(
            r"\bSELECT\b",
            text
        )
    )

    has_from = bool(
        re.search(
            r"\bFROM\s+PA\d{4}\b",
            text
        )
    )

    has_where = bool(
        re.search(
            r"\bWHERE\b",
            text
        )
    )

    has_direct_read = False

    # --------------------------------------------------------
    # Pattern strutturali
    # --------------------------------------------------------

    for name, (
        pattern,
        points
    ) in STRUCTURAL_PATTERNS.items():

        found = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if found:

            count = len(found)

            # Evitiamo che una moltitudine di
            # PA0001-FIELD faccia esplodere lo score.
            if name == "pa_field":

                count = min(
                    count,
                    3
                )

            added = points * count

            score += added

            matches.append(
                (
                    name,
                    count,
                    added
                )
            )

    # --------------------------------------------------------
    # Accesso diretto molto forte
    # --------------------------------------------------------

    query_upper = query.upper()

    table_pattern = (
        rf"\bFROM\s+{re.escape(query_upper)}\b"
    )

    if re.search(
        table_pattern,
        text
    ):

        has_direct_read = True

        score += 12.0

        matches.append(
            (
                "FROM QUERY TABLE",
                1,
                12.0
            )
        )

    # --------------------------------------------------------
    # SELECT con PERNR nella WHERE
    # --------------------------------------------------------

    if (
        re.search(
            rf"\bFROM\s+{re.escape(query_upper)}\b",
            text
        )
        and
        re.search(
            r"\bWHERE\b[\s\S]{0,400}\bPERNR\b",
            text
        )
    ):

        score += 10.0

        matches.append(
            (
                "SELECT TABLE + WHERE PERNR",
                1,
                10.0
            )
        )

    # --------------------------------------------------------
    # SELECT con KOSTL
    # --------------------------------------------------------

    if (
        query_upper == "PA0001"
        and
        re.search(
            r"\bSELECT\b[\s\S]{0,500}"
            r"\bKOSTL\b[\s\S]{0,500}"
            r"\bFROM\s+PA0001\b",
            text
        )
    ):

        score += 8.0

        matches.append(
            (
                "SELECT KOSTL FROM PA0001",
                1,
                8.0
            )
        )

    # --------------------------------------------------------
    # SELECT SINGLE PA0001 + PERNR
    # --------------------------------------------------------

    if (
        query_upper == "PA0001"
        and
        re.search(
            r"\bSELECT\s+SINGLE\b"
            r"[\s\S]{0,500}"
            r"\bFROM\s+PA0001\b"
            r"[\s\S]{0,500}"
            r"\bPERNR\b",
            text
        )
    ):

        score += 12.0

        matches.append(
            (
                "SELECT SINGLE PA0001 + PERNR",
                1,
                12.0
            )
        )

    # --------------------------------------------------------
    # Dichiarazione senza lettura
    # --------------------------------------------------------

    declaration_patterns = [

        r"\bDATA\b[\s\S]{0,100}"
        r"\bLIKE\s+PA\d{4}-[A-Z0-9_]+",

        r"\bTYPE\s+PA\d{4}-[A-Z0-9_]+",

        r"\bLIKE\s+PA\d{4}\b",
    ]

    declaration_count = 0

    for pattern in declaration_patterns:

        declaration_count += len(
            re.findall(
                pattern,
                text
            )
        )

    declaration_only = (
        declaration_count > 0
        and not has_select
        and not has_direct_read
    )

    penalty = 0.0

    if declaration_only:

        penalty = min(
            declaration_count * 2.0,
            8.0
        )

        score -= penalty

        matches.append(
            (
                "DECLARATION ONLY PENALTY",
                declaration_count,
                -penalty
            )
        )

    return {
        "score": max(score, 0.0),
        "matches": matches,
        "has_select": has_select,
        "has_from": has_from,
        "has_where": has_where,
        "has_direct_read": has_direct_read,
        "declaration_only": declaration_only,
    }


# ============================================================
# CALCOLO FINAL SCORE V4
# ============================================================

def calcola_final_score(
    item,
    query
):

    doc = item["document"]

    metadata = doc.metadata

    content = doc.page_content

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

    # --------------------------------------------------------
    # Semantic component
    # --------------------------------------------------------

    semantic_component = (
        semantic * 10.0
    )

    # --------------------------------------------------------
    # Exact component
    # --------------------------------------------------------

    exact_component = (
        exact * 3.0
    )

    # --------------------------------------------------------
    # Structural analysis
    # --------------------------------------------------------

    structural = analizza_struttura_abap(
        content,
        query
    )

    structural_score = structural[
        "score"
    ]

    # --------------------------------------------------------
    # Query exact bonus
    # --------------------------------------------------------

    query_upper = query.upper()

    query_exact_bonus = 0.0

    if query_upper in [
        term.upper()
        for term in terms
    ]:

        query_exact_bonus = 15.0

    # --------------------------------------------------------
    # Source bonus
    # --------------------------------------------------------

    source_file = (
        metadata.get(
            "source_file",
            ""
        )
        or ""
    ).upper()

    object_name = (
        metadata.get(
            "object_name",
            ""
        )
        or ""
    ).upper()

    source_bonus = 0.0

    if query_upper in source_file:

        source_bonus += 10.0

    if query_upper == object_name:

        source_bonus += 10.0

    # --------------------------------------------------------
    # ABAP bonus
    # --------------------------------------------------------

    abap_bonus = 0.0

    if metadata.get(
        "language"
    ) == "ABAP":

        abap_bonus = 2.0

    # --------------------------------------------------------
    # DDIC penalty
    # --------------------------------------------------------

    ddic_penalty = 0.0

    if metadata.get(
        "language"
    ) == "SAP_DDIC":

        ddic_penalty = 3.0

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    final_score = (

        semantic_component

        + exact_component

        + structural_score

        + query_exact_bonus

        + source_bonus

        + abap_bonus

        - ddic_penalty
    )

    item["final_score"] = final_score

    item["semantic_component"] = (
        semantic_component
    )

    item["exact_component"] = (
        exact_component
    )

    item["structural_score"] = (
        structural_score
    )

    item["structural_matches"] = (
        structural["matches"]
    )

    item["penalty"] = (
        ddic_penalty
        + (
            min(
                len(
                    structural["matches"]
                ),
                0
            )
        )
    )

    item["query_exact_bonus"] = (
        query_exact_bonus
    )

    item["source_bonus"] = (
        source_bonus
    )

    item["abap_bonus"] = (
        abap_bonus
    )

    item["declaration_only"] = (
        structural["declaration_only"]
    )

    item["has_select"] = (
        structural["has_select"]
    )

    item["has_from"] = (
        structural["has_from"]
    )

    item["has_where"] = (
        structural["has_where"]
    )

    item["has_direct_read"] = (
        structural["has_direct_read"]
    )

    return item


# ============================================================
# RANKING
# ============================================================

def applica_ranking(
    risultati,
    query
):

    print()
    print("=" * 70)
    print("CALCOLO SAP STRUCTURAL RANKING V4")
    print("=" * 70)

    for item in risultati:

        calcola_final_score(
            item,
            query
        )

    # --------------------------------------------------------
    # Ordinamento iniziale
    # --------------------------------------------------------

    risultati.sort(
        key=lambda x: (
            x["final_score"],
            x["structural_score"],
            x["exact_score"],
            x["semantic_score"]
        ),
        reverse=True
    )

    # --------------------------------------------------------
    # Diversificazione per source_file
    # --------------------------------------------------------

    selezionati = []

    source_count = {}

    differiti = []

    for item in risultati:

        source = (
            item["document"]
            .metadata
            .get(
                "source_file",
                "<UNKNOWN>"
            )
        )

        count = source_count.get(
            source,
            0
        )

        if (
            count
            < MAX_RESULTS_PER_SOURCE
        ):

            selezionati.append(item)

            source_count[source] = (
                count + 1
            )

        else:

            differiti.append(item)

        if len(selezionati) >= FINAL_K:
            break

    # --------------------------------------------------------
    # Se non abbiamo abbastanza risultati,
    # riempiamo con i differiti migliori.
    # --------------------------------------------------------

    if len(selezionati) < FINAL_K:

        gia_presenti = {
            id(item)
            for item in selezionati
        }

        for item in differiti:

            if id(item) in gia_presenti:
                continue

            selezionati.append(item)

            if len(selezionati) >= FINAL_K:
                break

    return selezionati


# ============================================================
# STAMPA RISULTATI
# ============================================================

def stampa_risultati(
    risultati,
    query
):

    print()
    print("=" * 70)
    print(
        "RISULTATI FINALI - "
        "SAP STRUCTURAL HYBRID RANKING V4"
    )
    print("=" * 70)

    print(
        f"Query: {query}"
    )

    print(
        f"Risultati mostrati: "
        f"{len(risultati)}"
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
            f"Final score       : "
            f"{item['final_score']:.4f}"
        )

        print(
            f"Semantic score    : "
            f"{item['semantic_score']:.4f}"
        )

        print(
            f"Semantic component: "
            f"{item['semantic_component']:.4f}"
        )

        print(
            f"Exact score       : "
            f"{item['exact_score']:.4f}"
        )

        print(
            f"Exact component   : "
            f"{item['exact_component']:.4f}"
        )

        print(
            f"Structural score  : "
            f"{item['structural_score']:.4f}"
        )

        print(
            f"Query exact bonus : "
            f"{item['query_exact_bonus']:.4f}"
        )

        print(
            f"Source bonus      : "
            f"{item['source_bonus']:.4f}"
        )

        print(
            f"Exact terms       : "
            f"{', '.join(item['exact_terms']) or '-'}"
        )

        print()

        print(
            "Structural matches:"
        )

        if item["structural_matches"]:

            for (
                name,
                count,
                points
            ) in item["structural_matches"]:

                print(
                    f"  + {name} "
                    f"({count}) "
                    f"[{points:+.1f}]"
                )

        else:

            print(
                "  - Nessun pattern strutturale"
            )

        print()

        print(
            f"Source file       : "
            f"{metadata.get('source_file', '-')}"
        )

        print(
            f"Source path       : "
            f"{metadata.get('source_path', '-')}"
        )

        print(
            f"Object type       : "
            f"{metadata.get('object_type', '-')}"
        )

        print(
            f"Object name       : "
            f"{metadata.get('object_name', '-')}"
        )

        print(
            f"Language          : "
            f"{metadata.get('language', '-')}"
        )

        print(
            f"Chunk index       : "
            f"{metadata.get('chunk_index', '-')}"
        )

        print(
            f"Content hash      : "
            f"{item.get('content_hash', '-')[:16]}"
        )

        print()

        print(
            "CODICE / CONTENUTO:"
        )

        print("-" * 70)

        print(
            content[:DISPLAY_CHARS]
        )

        if len(content) > DISPLAY_CHARS:

            print()

            print(
                "... [contenuto troncato]"
            )


# ============================================================
# DIAGNOSTICA V4
# ============================================================

def diagnostica(
    risultati,
    query
):

    print()
    print("=" * 70)
    print("DIAGNOSTICA RETRIEVAL V4")
    print("=" * 70)

    query_upper = query.upper()

    exact_query_count = 0

    abap_count = 0

    structural_count = 0

    select_count = 0

    from_count = 0

    where_pernr_count = 0

    direct_read_count = 0

    declaration_only_count = 0

    source_files = set()

    for item in risultati:

        doc = item["document"]

        metadata = doc.metadata

        source_files.add(
            metadata.get(
                "source_file",
                ""
            )
        )

        if query_upper in [
            term.upper()
            for term in item["exact_terms"]
        ]:

            exact_query_count += 1

        if metadata.get(
            "language"
        ) == "ABAP":

            abap_count += 1

        if item.get(
            "structural_score",
            0
        ) > 0:

            structural_count += 1

        if item.get(
            "has_select"
        ):

            select_count += 1

        if item.get(
            "has_from"
        ):

            from_count += 1

        content_upper = (
            doc.page_content.upper()
        )

        if re.search(
            r"\bWHERE\b[\s\S]{0,300}\bPERNR\b",
            content_upper
        ):

            where_pernr_count += 1

        if item.get(
            "has_direct_read"
        ):

            direct_read_count += 1

        if item.get(
            "declaration_only"
        ):

            declaration_only_count += 1

    print(
        f"Risultati finali             : "
        f"{len(risultati)}"
    )

    print(
        f"Source file distinti         : "
        f"{len(source_files)}"
    )

    print(
        f"Risultati ABAP               : "
        f"{abap_count}"
    )

    print(
        f"Match query esatto           : "
        f"{exact_query_count}"
    )

    print(
        f"Con score strutturale        : "
        f"{structural_count}"
    )

    print(
        f"Con SELECT                   : "
        f"{select_count}"
    )

    print(
        f"Con FROM PAxxxx              : "
        f"{from_count}"
    )

    print(
        f"Con WHERE ... PERNR          : "
        f"{where_pernr_count}"
    )

    print(
        f"Con accesso diretto tabella : "
        f"{direct_read_count}"
    )

    print(
        f"Solo dichiarativi            : "
        f"{declaration_only_count}"
    )

    print()

    # --------------------------------------------------------
    # Valutazione
    # --------------------------------------------------------

    if (
        abap_count > 0
        and
        structural_count > 0
        and
        from_count > 0
    ):

        print(
            "ESITO: RETRIEVAL ABAP MOLTO PROMETTENTE"
        )

        print(
            "Il ranking V4 privilegia "
            "codice ABAP operativo rispetto "
            "alle sole dichiarazioni."
        )

    elif (
        abap_count > 0
        and
        exact_query_count > 0
    ):

        print(
            "ESITO: RETRIEVAL ABAP PROMETTENTE"
        )

    else:

        print(
            "ESITO: RETRIEVAL DA MIGLIORARE"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RAG TEST V4 - SAP ABAP/HCM")
    print("=" * 70)

    query = input(
        "Inserisci la query SAP/ABAP "
        "(es. PA0001): "
    ).strip()

    if not query:

        print(
            "[ERRORE] Query vuota."
        )

        return

    tempo_inizio = time.time()

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    vector_db = crea_vectordb()

    # --------------------------------------------------------
    # 1. SEMANTIC SEARCH
    # --------------------------------------------------------

    risultati_semantici = (
        ricerca_semantica(
            vector_db,
            query
        )
    )

    # --------------------------------------------------------
    # 2. EXACT SAP SEARCH
    # --------------------------------------------------------

    risultati_esatti = (
        ricerca_esatta(
            vector_db,
            query
        )
    )

    # --------------------------------------------------------
    # 3. MERGE + CONTENT DEDUP
    # --------------------------------------------------------

    risultati = fondi_risultati(
        risultati_semantici,
        risultati_esatti
    )

    # --------------------------------------------------------
    # 4. STRUCTURAL RANKING
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

    # --------------------------------------------------------
    # TEMPI
    # --------------------------------------------------------

    tempo_totale = (
        time.time()
        - tempo_inizio
    )

    print()

    print("=" * 70)
    print("TEMPI")
    print("=" * 70)

    print(
        f"Tempo totale retrieval : "
        f"{tempo_totale:.2f} sec"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
