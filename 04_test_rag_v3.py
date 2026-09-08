import re
from pathlib import Path

from langchain_core.documents import Document
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

# Batch utilizzato durante la scansione Chroma
EXACT_BATCH_SIZE = 500

# Numero massimo caratteri mostrati per chunk
DISPLAY_CONTENT_LENGTH = 2500


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
        r"[^A-Z0-9_/\~]",
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
        r"[A-Za-z_][A-Za-z0-9_/\~]*",
        query.upper()
    )

    return list(dict.fromkeys(tokens))


def termini_sap_per_query(query):
    """
    Restituisce i termini SAP associati alla query.
    """

    query_upper = query.upper()

    termini = []

    # Match diretto nella knowledge base SAP
    for key, values in SAP_TERMS.items():

        if key in query_upper:
            termini.extend(values)

    # Aggiungiamo comunque i token della query
    termini.extend(
        estrai_query_terms(query)
    )

    return list(
        dict.fromkeys(termini)
    )


def metadata_da_item(item):
    """
    Restituisce sempre i metadata indipendentemente dal fatto
    che il risultato provenga dalla ricerca semantica o esatta.
    """

    doc = item.get("document")

    if doc is not None:
        return doc.metadata or {}

    return item.get("metadata") or {}


def contenuto_da_item(item):
    """
    Restituisce sempre il contenuto del chunk.
    """

    doc = item.get("document")

    if doc is not None:
        return doc.page_content or ""

    return item.get("content") or ""


# ============================================================
# ANALISI DELL'INTENTO DELLA QUERY
# ============================================================

def analizza_intento_query(query):
    """
    Identifica in modo semplice l'intento tecnico della query.

    Non utilizziamo un LLM:
    l'obiettivo è avere un ranking deterministico,
    riproducibile e facilmente diagnosticabile.
    """

    query_upper = query.upper()

    intent = {
        "read": False,
        "write": False,
        "select": False,
        "update": False,
        "delete": False,
        "join": False,
        "table_reference": False,
    }

    parole_lettura = [
        "LEGGERE",
        "LEGGO",
        "LEGGE",
        "LEGGI",
        "OTTENERE",
        "OTTIENI",
        "RECUPERARE",
        "RECUPERA",
        "RECUPERO",
        "PRENDERE",
        "PRELEVA",
        "LETTURA",
        "READ",
        "GET",
        "RETRIEVE",
        "FETCH",
    ]

    parole_select = [
        "SELECT",
        "ESTRARRE",
        "ESTRAI",
        "ESTRAZIONE",
    ]

    parole_scrittura = [
        "SCRIVERE",
        "INSERIRE",
        "INSERT",
        "MODIFICARE",
        "MODIFICA",
        "UPDATE",
        "AGGIORNARE",
    ]

    parole_delete = [
        "CANCELLARE",
        "DELETE",
        "ELIMINARE",
    ]

    parole_join = [
        "JOIN",
        "COLLEGARE",
        "COLLEGA",
        "UNIRE",
    ]

    if any(
        parola in query_upper
        for parola in parole_lettura
    ):
        intent["read"] = True

    if any(
        parola in query_upper
        for parola in parole_select
    ):
        intent["select"] = True

    if any(
        parola in query_upper
        for parola in parole_scrittura
    ):
        intent["write"] = True

    if "UPDATE" in query_upper:
        intent["update"] = True

    if any(
        parola in query_upper
        for parola in parole_delete
    ):
        intent["delete"] = True

    if any(
        parola in query_upper
        for parola in parole_join
    ):
        intent["join"] = True

    # Query costituita principalmente da un nome tabella
    if re.search(
        r"\bPA\d{4}\b",
        query_upper
    ):
        intent["table_reference"] = True

    return intent


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

            "structural_score": 0.0,

            "penalty_score": 0.0,

            "final_score": float(score),

            "exact_terms": [],

            "structural_matches": [],

            "key": None,
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
            # SCORE LESSICALE
            # ------------------------------------------------

            score = 0.0

            query_upper = query.upper()

            for termine in trovati:

                termine_upper = termine.upper()

                # Match esatto della query
                if termine_upper == query_upper:

                    score += 10.0

                # Tabelle PAxxxx
                elif re.fullmatch(
                    r"PA\d{4}",
                    termine_upper
                ):

                    score += 5.0

                # Altri termini SAP
                else:

                    score += 1.0

            # Bonus per quantità di termini trovati

            score += min(
                len(trovati) * 0.5,
                5.0
            )

            risultati.append({

                "document": None,

                "semantic_score": 0.0,

                "exact_score": score,

                "structural_score": 0.0,

                "penalty_score": 0.0,

                "final_score": score,

                "exact_terms": trovati,

                "structural_matches": [],

                "metadata": metadata,

                "content": content,

                "key": None
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
# ANALISI STRUTTURALE ABAP
# ============================================================

def analizza_struttura_abap(
    content,
    query,
    intent
):
    """
    Analizza il contenuto ABAP cercando strutture realmente
    indicative dell'utilizzo dell'oggetto SAP.

    Restituisce:

        structural_score
        penalty_score
        matches
    """

    text = content or ""

    upper = text.upper()

    score = 0.0

    penalty = 0.0

    matches = []

    # --------------------------------------------------------
    # Tabelle SAP presenti nella query
    # --------------------------------------------------------

    query_tables = re.findall(
        r"\bPA\d{4}\b",
        query.upper()
    )

    query_tables = list(
        dict.fromkeys(query_tables)
    )

    # --------------------------------------------------------
    # Per ogni tabella SAP cercata
    # --------------------------------------------------------

    for table in query_tables:

        # ----------------------------------------------------
        # FROM PA0001
        # ----------------------------------------------------

        if re.search(
            rf"\bFROM\s+{re.escape(table)}\b",
            upper
        ):

            score += 12.0

            matches.append(
                f"FROM {table}"
            )

        # ----------------------------------------------------
        # JOIN PA0001
        # ----------------------------------------------------

        if re.search(
            rf"\b(?:INNER|LEFT|RIGHT|FULL)?\s*JOIN\s+"
            rf"{re.escape(table)}\b",
            upper
        ):

            score += 10.0

            matches.append(
                f"JOIN {table}"
            )

        # ----------------------------------------------------
        # SELECT ... FROM PA0001
        # ----------------------------------------------------

        if re.search(
            rf"\bSELECT\b[\s\S]{{0,1200}}?"
            rf"\bFROM\s+{re.escape(table)}\b",
            upper
        ):

            score += 10.0

            matches.append(
                f"SELECT ... FROM {table}"
            )

        # ----------------------------------------------------
        # SELECT SINGLE ... FROM PA0001
        # ----------------------------------------------------

        if re.search(
            rf"\bSELECT\s+SINGLE\b[\s\S]{{0,1200}}?"
            rf"\bFROM\s+{re.escape(table)}\b",
            upper
        ):

            score += 6.0

            matches.append(
                f"SELECT SINGLE ... FROM {table}"
            )

        # ----------------------------------------------------
        # Accesso strutturato PA0001-CAMPO
        # ----------------------------------------------------

        field_matches = re.findall(
            rf"\b{re.escape(table)}-"
            rf"[A-Z0-9_]+\b",
            upper
        )

        if field_matches:

            unique_fields = list(
                dict.fromkeys(field_matches)
            )

            field_bonus = min(
                len(unique_fields) * 2.0,
                8.0
            )

            score += field_bonus

            matches.append(
                f"{table}-FIELD "
                f"({len(unique_fields)})"
            )

        # ----------------------------------------------------
        # WHERE ... PERNR
        # ----------------------------------------------------

        if re.search(
            r"\bWHERE\b[\s\S]{0,800}\bPERNR\b",
            upper
        ):

            score += 6.0

            matches.append(
                "WHERE ... PERNR"
            )

        # ----------------------------------------------------
        # WHERE ... KOSTL
        # ----------------------------------------------------

        if re.search(
            r"\bWHERE\b[\s\S]{0,800}\bKOSTL\b",
            upper
        ):

            score += 4.0

            matches.append(
                "WHERE ... KOSTL"
            )

    # ========================================================
    # PATTERN SAP P0001
    # ========================================================

    if re.search(
        r"\bP0001-"
        r"[A-Z0-9_]+\b",
        upper
    ):

        score += 5.0

        matches.append(
            "P0001-FIELD"
        )

    # ========================================================
    # READ TABLE
    # ========================================================

    if re.search(
        r"\bREAD\s+TABLE\b",
        upper
    ):

        score += 4.0

        matches.append(
            "READ TABLE"
        )

    # ========================================================
    # READ TABLE P0001
    # ========================================================

    if re.search(
        r"\bREAD\s+TABLE\b[\s\S]{0,500}\bP0001\b",
        upper
    ):

        score += 6.0

        matches.append(
            "READ TABLE ... P0001"
        )

    # ========================================================
    # LOOP AT PA0001 / P0001
    # ========================================================

    if re.search(
        r"\bLOOP\s+AT\b[\s\S]{0,100}\bPA\d{4}\b",
        upper
    ):

        score += 6.0

        matches.append(
            "LOOP AT PAxxxx"
        )

    # ========================================================
    # MODULO LETTURA
    # ========================================================

    if intent["read"] or intent["select"]:

        if re.search(
            r"\bSELECT\s+SINGLE\b",
            upper
        ):

            score += 8.0

            matches.append(
                "SELECT SINGLE"
            )

        elif re.search(
            r"\bSELECT\b",
            upper
        ):

            score += 4.0

            matches.append(
                "SELECT"
            )

    # ========================================================
    # QUERY DI JOIN
    # ========================================================

    if intent["join"]:

        if re.search(
            r"\bJOIN\b",
            upper
        ):

            score += 5.0

            matches.append(
                "JOIN"
            )

    # ========================================================
    # RELAZIONE CAMPO + QUERY TERM
    # ========================================================

    query_terms = set(
        estrai_query_terms(query)
    )

    important_query_fields = {
        "PERNR",
        "KOSTL",
        "BUKRS",
        "WERKS",
        "PERSG",
        "PERSK",
        "BTRTL",
        "GSBER",
        "ORGEH",
        "PLANS",
        "STELL",
        "BEGDA",
        "ENDDA",
        "STAT2",
    }

    relevant_fields = (
        query_terms
        & important_query_fields
    )

    for field in relevant_fields:

        # Campo come componente PAxxxx-CAMPO
        if re.search(
            rf"\bPA\d{{4}}-{re.escape(field)}\b",
            upper
        ):

            score += 3.0

            matches.append(
                f"PAxxxx-{field}"
            )

        # Campo isolato in codice
        elif re.search(
            rf"\b{re.escape(field)}\b",
            upper
        ):

            score += 1.0

            matches.append(
                field
            )

    # ========================================================
    # PENALIZZAZIONE DYNPRO / UI
    # ========================================================

    dynpro_patterns = [
        r"\bFIELD\s+PA\d{4}-",
        r"\bMODULE\s+\w+",
        r"\bCHAIN\b",
        r"\bENDCHAIN\b",
    ]

    dynpro_hits = 0

    for pattern in dynpro_patterns:

        if re.search(
            pattern,
            upper
        ):
            dynpro_hits += 1

    # Se il chunk è chiaramente Dynpro e non contiene SELECT,
    # lo penalizziamo.

    has_select = bool(
        re.search(
            r"\bSELECT\b",
            upper
        )
    )

    if dynpro_hits >= 2 and not has_select:

        penalty += 6.0

        matches.append(
            "PENALTY: DYNPRO"
        )

    # FIELD PAxxxx è molto meno utile di una SELECT
    if (
        re.search(
            r"\bFIELD\s+PA\d{4}-",
            upper
        )
        and not has_select
    ):

        penalty += 3.0

        matches.append(
            "PENALTY: FIELD"
        )

    # ========================================================
    # PENALIZZAZIONE COMMENTI
    # ========================================================

    comment_lines = re.findall(
        r"(?m)^\s*\*.*$",
        text
    )

    code_lines = re.findall(
        r"(?m)^\s*[A-Z].*$",
        text
    )

    if (
        len(comment_lines) > 10
        and len(comment_lines) > len(code_lines)
        and score < 10
    ):

        penalty += 2.0

        matches.append(
            "PENALTY: COMMENT HEAVY"
        )

    return (
        score,
        penalty,
        list(dict.fromkeys(matches))
    )


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
    # Risultati semantici
    # --------------------------------------------------------

    for item in risultati_semantici:

        key = chiave_documento(item)

        item["key"] = key

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

        else:

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

    return list(
        merged.values()
    )


# ============================================================
# RANKING V3
# ============================================================

def calcola_final_score(
    item,
    query,
    intent
):

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

    metadata = metadata_da_item(item)

    content = contenuto_da_item(item)

    # ========================================================
    # 1. SEMANTIC SCORE
    # ========================================================

    semantic_component = (
        semantic * 10.0
    )

    # ========================================================
    # 2. EXACT SCORE
    # ========================================================

    exact_component = (
        exact * 2.5
    )

    # ========================================================
    # 3. STRUCTURAL SCORE
    # ========================================================

    (
        structural_score,
        penalty_score,
        structural_matches
    ) = analizza_struttura_abap(
        content,
        query,
        intent
    )

    # ========================================================
    # 4. BONUS QUERY ESATTA
    # ========================================================

    query_upper = query.upper()

    exact_query_bonus = 0.0

    if query_upper in [
        term.upper()
        for term in terms
    ]:

        exact_query_bonus = 15.0

    # ========================================================
    # 5. SOURCE FILE
    # ========================================================

    source_file = (
        metadata
        .get(
            "source_file",
            ""
        )
        .upper()
    )

    object_name = (
        metadata
        .get(
            "object_name",
            ""
        )
        .upper()
    )

    source_bonus = 0.0

    if query_upper in source_file:

        source_bonus += 8.0

    if query_upper == object_name:

        source_bonus += 12.0

    # ========================================================
    # 6. LINGUAGGIO ABAP
    # ========================================================

    language_bonus = 0.0

    if metadata.get("language") == "ABAP":

        language_bonus = 2.0

    elif metadata.get("language") == "SAP_DDIC":

        language_bonus = -1.0

    # ========================================================
    # SCORE FINALE
    # ========================================================

    final_score = (

        semantic_component
        + exact_component
        + structural_score
        + exact_query_bonus
        + source_bonus
        + language_bonus
        - penalty_score
    )

    # Salviamo tutto per la diagnostica

    item["semantic_component"] = (
        semantic_component
    )

    item["exact_component"] = (
        exact_component
    )

    item["structural_score"] = (
        structural_score
    )

    item["penalty_score"] = (
        penalty_score
    )

    item["exact_query_bonus"] = (
        exact_query_bonus
    )

    item["source_bonus"] = (
        source_bonus
    )

    item["language_bonus"] = (
        language_bonus
    )

    item["structural_matches"] = (
        structural_matches
    )

    item["final_score"] = (
        final_score
    )

    return final_score


def applica_ranking(
    risultati,
    query
):

    print()
    print("=" * 70)
    print("CALCOLO SAP STRUCTURAL RANKING")
    print("=" * 70)

    intent = analizza_intento_query(
        query
    )

    print("Intent query:")

    for key, value in intent.items():

        if value:
            print(
                f"  - {key}"
            )

    print()

    for item in risultati:

        calcola_final_score(
            item,
            query,
            intent
        )

    risultati.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return risultati[:FINAL_K]


# ============================================================
# STAMPA RISULTATI
# ============================================================

def stampa_risultati(
    risultati,
    query
):

    print()
    print("=" * 70)
    print("RISULTATI FINALI - SAP STRUCTURAL HYBRID RANKING V3")
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
            f"{item.get('semantic_component', 0.0):.4f}"
        )

        print(
            f"Exact score       : "
            f"{item['exact_score']:.4f}"
        )

        print(
            f"Exact component   : "
            f"{item.get('exact_component', 0.0):.4f}"
        )

        print(
            f"Structural score  : "
            f"{item.get('structural_score', 0.0):.4f}"
        )

        print(
            f"Penalty           : "
            f"{item.get('penalty_score', 0.0):.4f}"
        )

        print(
            f"Query exact bonus : "
            f"{item.get('exact_query_bonus', 0.0):.4f}"
        )

        print(
            f"Source bonus      : "
            f"{item.get('source_bonus', 0.0):.4f}"
        )

        print(
            f"Exact terms       : "
            f"{', '.join(item['exact_terms']) or '-'}"
        )

        print()

        print(
            "Structural matches:"
        )

        structural_matches = (
            item.get(
                "structural_matches",
                []
            )
        )

        if structural_matches:

            for match in structural_matches:

                print(
                    f"  + {match}"
                )

        else:

            print(
                "  - nessun pattern strutturale"
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

        print()

        print(
            "CODICE / CONTENUTO:"
        )

        print("-" * 70)

        print(
            content[:DISPLAY_CONTENT_LENGTH]
        )

        if len(content) > DISPLAY_CONTENT_LENGTH:

            print()

            print(
                "... [contenuto troncato]"
            )


# ============================================================
# DIAGNOSTICA
# ============================================================

def diagnostica(
    risultati,
    query
):

    print()
    print("=" * 70)
    print("DIAGNOSTICA RETRIEVAL V3")
    print("=" * 70)

    query_upper = query.upper()

    exact_query_count = 0

    abap_count = 0

    structural_count = 0

    select_count = 0

    from_table_count = 0

    dynpro_count = 0

    for item in risultati:

        metadata = metadata_da_item(item)

        structural_matches = (
            item.get(
                "structural_matches",
                []
            )
        )

        if query_upper in [
            term.upper()
            for term in item.get(
                "exact_terms",
                []
            )
        ]:

            exact_query_count += 1

        if metadata.get(
            "language"
        ) == "ABAP":

            abap_count += 1

        if item.get(
            "structural_score",
            0.0
        ) > 0:

            structural_count += 1

        if any(
            "SELECT" in match
            for match in structural_matches
        ):

            select_count += 1

        if any(
            "FROM PA" in match
            for match in structural_matches
        ):

            from_table_count += 1

        if any(
            "DYNPRO" in match
            or "FIELD" in match
            for match in structural_matches
        ):

            dynpro_count += 1

    print(
        f"Risultati finali             : "
        f"{len(risultati)}"
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
        f"{from_table_count}"
    )

    print(
        f"Con pattern Dynpro           : "
        f"{dynpro_count}"
    )

    print()

    # ========================================================
    # VALUTAZIONE
    # ========================================================

    if (
        select_count > 0
        and from_table_count > 0
        and abap_count > 0
    ):

        print(
            "ESITO: RETRIEVAL ABAP MOLTO PROMETTENTE"
        )

        print(
            "Sono stati trovati risultati ABAP "
            "contenenti strutture SELECT/FROM "
            "pertinenti alla query."
        )

    elif (
        structural_count > 0
        and abap_count > 0
    ):

        print(
            "ESITO: RETRIEVAL STRUTTURALMENTE PROMETTENTE"
        )

        print(
            "Sono presenti pattern ABAP pertinenti, "
            "ma non è stata trovata una SELECT "
            "diretta nei risultati finali."
        )

    elif (
        exact_query_count > 0
        and abap_count > 0
    ):

        print(
            "ESITO: MATCH ESATTO ABAP"
        )

        print(
            "La query è stata trovata in contenuto ABAP, "
            "ma la pertinenza strutturale deve essere verificata."
        )

    elif abap_count > 0:

        print(
            "ESITO: SOLO MATCH SEMANTICO/LESSICALE"
        )

        print(
            "Sono presenti risultati ABAP, "
            "ma manca una forte evidenza strutturale."
        )

    else:

        print(
            "ESITO: RETRIEVAL DA MIGLIORARE"
        )

        print(
            "Nessun risultato ABAP sufficientemente pertinente."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RAG ABAP/HCM - RETRIEVAL TEST V3")
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

    # --------------------------------------------------------
    # VectorDB
    # --------------------------------------------------------

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
    # 4. SAP STRUCTURAL RANKING
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()