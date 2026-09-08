# ================================================================
# 04_test_rag_v5.py
# RAG TEST - SAP STRUCTURAL QUERY-AWARE HYBRID RANKING V5
#
# NON MODIFICA IL VECTORDB
#
# Pipeline:
#   1. Ricerca semantica Chroma
#   2. Ricerca esatta termini SAP
#   3. Fusione
#   4. Deduplicazione contenuto
#   5. Query profiling
#   6. SAP structural query-aware ranking
#   7. Penalizzazione Dynpro / dichiarazioni
#   8. Deduplicazione near-duplicate
#   9. Diversificazione risultati
#  10. Diagnostica
# ================================================================

from pathlib import Path
import re
import hashlib
from difflib import SequenceMatcher

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ================================================================
# CONFIGURAZIONE
# ================================================================

DB_DIR = Path(r"C:\Progetto_AI\Abap_VectorDB")

COLLECTION_NAME = "abap_hcm"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

EMBEDDING_MODEL = "nomic-embed-text:latest"

SEMANTIC_K = 30

FINAL_K = 15

# Numero massimo di risultati candidati analizzati per la
# deduplicazione near-duplicate.
MAX_CANDIDATES_FOR_DIVERSITY = 150

# Similarità oltre la quale due chunk vengono considerati
# praticamente lo stesso codice.
NEAR_DUPLICATE_THRESHOLD = 0.90

# Batch usato per evitare:
# sqlite3.OperationalError: too many SQL variables
EXACT_SCAN_BATCH = 500


# ================================================================
# TERMINI SAP
# ================================================================

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


ALL_SAP_TERMS = sorted(
    set(
        term
        for terms in SAP_TERMS.values()
        for term in terms
    ),
    key=len,
    reverse=True,
)


# ================================================================
# PAROLE INTENTO
# ================================================================

DATA_READ_WORDS = {
    "leggere",
    "leggi",
    "lettura",
    "recuperare",
    "recupera",
    "recupero",
    "ottenere",
    "ottieni",
    "estrarre",
    "estrai",
    "selezionare",
    "seleziona",
    "seleziono",
    "select",
    "read",
    "retrieve",
    "get",
    "fetch",
    "lookup",
}

CODE_WORDS = {
    "come",
    "codice",
    "esempio",
    "implementare",
    "implementazione",
    "programma",
    "abap",
    "istruzione",
    "istruzioni",
}

FIELD_WORDS = {
    "campo",
    "field",
    "valore",
    "valori",
    "campo",
    "colonna",
}

# Termini funzionali che possono accompagnare il campo SAP.
BUSINESS_TO_TECH = {
    "centro di costo": "KOSTL",
    "centro costo": "KOSTL",
    "cost center": "KOSTL",
    "dipendente": "PERNR",
    "employee": "PERNR",
    "personale": "PERNR",
    "numero dipendente": "PERNR",
    "company code": "BUKRS",
    "società": "BUKRS",
    "azienda": "BUKRS",
    "area del personale": "WERKS",
    "personnel area": "WERKS",
    "gruppo dipendenti": "PERSG",
    "employee group": "PERSG",
}


# ================================================================
# REGEX SAP
# ================================================================

RE_TABLE = re.compile(
    r"\b(PA\d{4})\b",
    re.IGNORECASE,
)

RE_P_FIELD = re.compile(
    r"\b(P\d{4})-([A-Z0-9_]+)\b",
    re.IGNORECASE,
)

# ================================================================
# SOLO CAMPI SAP
#
# Le tabelle PAxxxx sono escluse esplicitamente.
# ================================================================

SAP_FIELD_TERMS = sorted(
    {
        term
        for term in ALL_SAP_TERMS
        if not re.fullmatch(
            r"PA\d{4}",
            term,
            re.IGNORECASE,
        )
        and not re.fullmatch(
            r"P\d{4}",
            term,
            re.IGNORECASE,
        )
    },
    key=len,
    reverse=True,
)

RE_SAP_FIELD = re.compile(
    r"\b("
    + "|".join(
        re.escape(x)
        for x in SAP_FIELD_TERMS
    )
    + r")\b",
    re.IGNORECASE,
)


# ================================================================
# NORMALIZZAZIONE
# ================================================================

def normalizza_testo(text):
    if not text:
        return ""

    text = str(text).upper()

    # Uniforma CR/LF
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Rimuove commenti ABAP classici.
    lines = []

    for line in text.splitlines():
        stripped = line.lstrip()

        if stripped.startswith("*"):
            continue

        # Commento inline ABAP.
        if '"' in line:
            line = line.split('"', 1)[0]

        lines.append(line)

    text = "\n".join(lines)

    # Spaziature.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalizza_per_hash(text):
    text = normalizza_testo(text)

    # Per la similarità eliminiamo differenze minori.
    text = re.sub(r"\s+", "", text)

    return text


# ================================================================
# HASH
# ================================================================

def content_hash(text):
    normalized = normalizza_per_hash(text)

    return hashlib.sha256(
        normalized.encode("utf-8", errors="ignore")
    ).hexdigest()


# ================================================================
# TERM SEARCH
# ================================================================

def contiene_termine(text, term):
    pattern = rf"\b{re.escape(term)}\b"

    return re.search(
        pattern,
        text,
        re.IGNORECASE,
    ) is not None


def trova_termini_sap(text):
    found = []

    for term in ALL_SAP_TERMS:

        if contiene_termine(text, term):
            found.append(term)

    return found


# ================================================================
# QUERY PROFILING
# ================================================================

def estrai_query_profile(query):

    q = query.upper().strip()

    # ============================================================
    # 1. ESTRAZIONE TABELLE SAP
    # ============================================================

    tables = sorted(
        set(
            match.upper()
            for match in RE_TABLE.findall(q)
        )
    )

    # ============================================================
    # 2. ESTRAZIONE CAMPI SAP
    #
    # IMPORTANTE:
    # le tabelle PAxxxx NON devono essere considerate campi.
    # ============================================================

    fields = sorted(
        set(
            match.upper()
            for match in RE_SAP_FIELD.findall(q)
        )
    )

    # Rimuove tutte le tabelle dalla lista dei campi.
    fields = [
        field
        for field in fields
        if field not in tables
    ]

    # ============================================================
    # 3. CAMPI QUALIFICATI
    #
    # Esempio:
    # PA0001-KOSTL
    # PA0001-PERNR
    # ============================================================

    qualified_fields = []

    for table_prefix, field in RE_P_FIELD.findall(q):

        table_prefix = table_prefix.upper()
        field = field.upper()

        qualified_fields.append(
            f"{table_prefix}-{field}"
        )

        # Se il prefisso Pxxxx corrisponde alla tabella
        # PAxxxx, aggiungiamo il campo.
        #
        # Esempio:
        # P0001-KOSTL -> KOSTL
        #

        if table_prefix.startswith("P"):

            field_upper = field

            # MAI aggiungere Pxxxx come campo.
            if field_upper not in tables:

                if field_upper not in fields:

                    fields.append(
                        field_upper
                    )

    # ============================================================
    # 4. BUSINESS TERM -> CAMPO TECNICO SAP
    # ============================================================

    for business_term, technical_field in BUSINESS_TO_TECH.items():

        if business_term.upper() in q:

            technical_field = (
                technical_field.upper()
            )

            # Non aggiungere mai una tabella
            # nella lista dei campi.
            if technical_field not in tables:

                if technical_field not in fields:

                    fields.append(
                        technical_field
                    )

    # ============================================================
    # 5. RIMOZIONE FINALE DI EVENTUALI TABELLE
    #
    # Difesa ulteriore contro errori di classificazione.
    # ============================================================

    fields = [
        field
        for field in fields
        if field not in tables
    ]

    fields = sorted(
        set(fields)
    )

    # ============================================================
    # 6. TOKEN DELLA QUERY
    # ============================================================

    q_words = set(
        re.findall(
            r"[A-Z0-9_]+",
            q
        )
    )

    # ============================================================
    # 7. INTENT DATA READ
    # ============================================================

    has_data_read = any(
        word.upper() in q_words
        for word in DATA_READ_WORDS
    )

    # ============================================================
    # 8. INTENT CODE
    # ============================================================

    has_code = any(
        word.upper() in q_words
        for word in CODE_WORDS
    )

    # ============================================================
    # 9. INTENT FIELD
    # ============================================================

    has_field_word = any(
        word.upper() in q_words
        for word in FIELD_WORDS
    )

    # ============================================================
    # 10. INTENT FINALE
    # ============================================================

    if has_data_read:

        intent = "data_read"

    elif has_code:

        intent = "code_generation"

    elif fields or has_field_word:

        intent = "field_lookup"

    elif tables:

        intent = "table_reference"

    else:

        intent = "general"

    # ============================================================
    # 11. KEY FIELDS
    # ============================================================

    key_fields = []

    if "PERNR" in fields:

        key_fields.append(
            "PERNR"
        )

    if "BUKRS" in fields:

        key_fields.append(
            "BUKRS"
        )

    if "WERKS" in fields:

        key_fields.append(
            "WERKS"
        )

    # ============================================================
    # 12. REQUESTED FIELDS
    # ============================================================

    requested_fields = [
        field
        for field in fields
        if field not in key_fields
    ]

    return {
        "tables": tables,
        "fields": sorted(set(fields)),
        "qualified_fields": qualified_fields,
        "key_fields": sorted(set(key_fields)),
        "requested_fields": sorted(set(requested_fields)),
        "intent": intent,
    }


# ================================================================
# EXACT SEARCH SCORE
# ================================================================

def calcola_exact_score(
    text,
    query_profile,
    query,
):

    text_upper = text.upper()

    score = 0.0

    found_terms = trova_termini_sap(text_upper)

    query_upper = query.upper()

    # ------------------------------------------------------------
    # Query exact term
    # ------------------------------------------------------------

    for term in query_profile["tables"]:

        if contiene_termine(text_upper, term):

            score += 10.0

    for field in query_profile["fields"]:

        if contiene_termine(text_upper, field):

            score += 4.0

    # ------------------------------------------------------------
    # SAP technical terms generici
    # ------------------------------------------------------------

    for term in found_terms:

        if term not in query_profile["tables"]:

            if term not in query_profile["fields"]:

                score += 0.5

    # ------------------------------------------------------------
    # Cap
    # ------------------------------------------------------------

    score = min(score, 30.0)

    return score, found_terms


# ================================================================
# CONTESTO SELECT
# ================================================================

def estrai_select_blocchi(text):

    """
    Estrae blocchi approssimativi SELECT ... .
    Non è un parser ABAP completo.
    Serve esclusivamente al ranking.
    """

    text_upper = text.upper()

    patterns = [
        r"\bSELECT\b.*?(?:\.)",
        r"\bSELECT\s+SINGLE\b.*?(?:\.)",
    ]

    blocks = []

    for pattern in patterns:

        matches = re.finditer(
            pattern,
            text_upper,
            flags=re.IGNORECASE | re.DOTALL,
        )

        for match in matches:

            block = match.group(0)

            if len(block) > 4000:
                block = block[:4000]

            blocks.append(block)

    return blocks


# ================================================================
# STRUCTURAL RANKING V5
# ================================================================

def calcola_structural_score(
    text,
    query_profile,
):

    t = normalizza_testo(text)

    tables = query_profile["tables"]

    fields = query_profile["fields"]

    requested_fields = query_profile["requested_fields"]

    key_fields = query_profile["key_fields"]

    intent = query_profile["intent"]

    score = 0.0

    matches = []

    upper = t.upper()

    # ------------------------------------------------------------
    # Se non abbiamo una tabella target, ranking generico.
    # ------------------------------------------------------------

    if not tables:

        return score, matches

    # ------------------------------------------------------------
    # Per ogni tabella richiesta
    # ------------------------------------------------------------

    for table in tables:

        # ========================================================
        # 1. FROM PAxxxx
        # ========================================================

        from_matches = re.findall(
            rf"\bFROM\s+{re.escape(table)}\b",
            upper,
            flags=re.IGNORECASE,
        )

        count = len(from_matches)

        if count:

            # CAP: la prima occorrenza conta molto, le successive
            # molto meno.
            contribution = min(count, 2) * 20.0

            if count > 2:
                contribution += min(count - 2, 3) * 3.0

            matches.append(
                f"from_{table.lower()} "
                f"({count}) [+{contribution:.1f}]"
            )

            score += contribution

        # ========================================================
        # 2. SELECT ... FROM PAxxxx
        # ========================================================

        select_from_matches = re.findall(
            rf"\bSELECT\b.*?\bFROM\s+{re.escape(table)}\b",
            upper,
            flags=re.IGNORECASE | re.DOTALL,
        )

        count = len(select_from_matches)

        if count:

            contribution = min(count, 2) * 28.0

            if count > 2:
                contribution += min(count - 2, 2) * 4.0

            matches.append(
                f"select_from_{table.lower()} "
                f"({count}) [+{contribution:.1f}]"
            )

            score += contribution

        # ========================================================
        # 3. SELECT SINGLE ... FROM PAxxxx
        # ========================================================

        select_single_matches = re.findall(
            rf"\bSELECT\s+SINGLE\b.*?\bFROM\s+{re.escape(table)}\b",
            upper,
            flags=re.IGNORECASE | re.DOTALL,
        )

        count = len(select_single_matches)

        if count:

            contribution = min(count, 2) * 30.0

            if count > 2:
                contribution += min(count - 2, 2) * 4.0

            matches.append(
                f"select_single_from_{table.lower()} "
                f"({count}) [+{contribution:.1f}]"
            )

            score += contribution

        # ========================================================
        # 4. Campi richiesti
        # ========================================================

        for field in requested_fields:

            # PA0001-KOSTL
            qualified_pattern = (
                rf"\b{re.escape(table)}-{re.escape(field)}\b"
            )

            qualified_count = len(
                re.findall(
                    qualified_pattern,
                    upper,
                    flags=re.IGNORECASE,
                )
            )

            if qualified_count:

                contribution = min(
                    qualified_count,
                    2,
                ) * 12.0

                matches.append(
                    f"{table}-{field} "
                    f"({qualified_count}) [+{contribution:.1f}]"
                )

                score += contribution

            # Campo dentro SELECT.
            select_field_from_table = re.findall(
                rf"\bSELECT\b.*?"
                rf"\b{re.escape(field)}\b.*?"
                rf"\bFROM\s+{re.escape(table)}\b",
                upper,
                flags=re.IGNORECASE | re.DOTALL,
            )

            count = len(select_field_from_table)

            if count:

                contribution = min(
                    count,
                    2,
                ) * 45.0

                matches.append(
                    f"SELECT {field} FROM {table} "
                    f"({count}) [+{contribution:.1f}]"
                )

                score += contribution

            # SELECT SINGLE campo FROM tabella
            select_single_field = re.findall(
                rf"\bSELECT\s+SINGLE\b.*?"
                rf"\b{re.escape(field)}\b.*?"
                rf"\bFROM\s+{re.escape(table)}\b",
                upper,
                flags=re.IGNORECASE | re.DOTALL,
            )

            count = len(select_single_field)

            if count:

                contribution = min(
                    count,
                    2,
                ) * 60.0

                matches.append(
                    f"SELECT SINGLE {field} FROM {table} "
                    f"({count}) [+{contribution:.1f}]"
                )

                score += contribution

        # ========================================================
        # 5. Key field WHERE
        # ========================================================

        for key_field in key_fields:

            where_key = re.findall(
                rf"\bWHERE\b.*?"
                rf"\b{re.escape(key_field)}\b",
                upper,
                flags=re.IGNORECASE | re.DOTALL,
            )

            count = len(where_key)

            if count:

                contribution = min(
                    count,
                    2,
                ) * 25.0

                if count > 2:
                    contribution += min(
                        count - 2,
                        2,
                    ) * 3.0

                matches.append(
                    f"WHERE {key_field} "
                    f"({count}) [+{contribution:.1f}]"
                )

                score += contribution

        # ========================================================
        # 6. Relazione richiesta:
        #
        # SELECT ... campo ... FROM PAxxxx ... WHERE PERNR
        # ========================================================

        if requested_fields and key_fields:

            for requested_field in requested_fields:

                for key_field in key_fields:

                    relation_pattern = (
                        rf"\bSELECT\b"
                        rf".*?\b{re.escape(requested_field)}\b"
                        rf".*?\bFROM\s+{re.escape(table)}\b"
                        rf".*?\bWHERE\b"
                        rf".*?\b{re.escape(key_field)}\b"
                    )

                    relation_matches = re.findall(
                        relation_pattern,
                        upper,
                        flags=re.IGNORECASE | re.DOTALL,
                    )

                    count = len(relation_matches)

                    if count:

                        contribution = min(
                            count,
                            2,
                        ) * 100.0

                        matches.append(
                            f"SELECT {requested_field} "
                            f"FROM {table} "
                            f"WHERE {key_field} "
                            f"({count}) "
                            f"[+{contribution:.1f}]"
                        )

                        score += contribution

                    # Versione SELECT SINGLE
                    single_relation_pattern = (
                        rf"\bSELECT\s+SINGLE\b"
                        rf".*?\b{re.escape(requested_field)}\b"
                        rf".*?\bFROM\s+{re.escape(table)}\b"
                        rf".*?\bWHERE\b"
                        rf".*?\b{re.escape(key_field)}\b"
                    )

                    single_relation_matches = re.findall(
                        single_relation_pattern,
                        upper,
                        flags=re.IGNORECASE | re.DOTALL,
                    )

                    count = len(single_relation_matches)

                    if count:

                        contribution = min(
                            count,
                            2,
                        ) * 130.0

                        matches.append(
                            f"SELECT SINGLE "
                            f"{requested_field} FROM {table} "
                            f"WHERE {key_field} "
                            f"({count}) "
                            f"[+{contribution:.1f}]"
                        )

                        score += contribution

        # ========================================================
        # 7. LOOP AT PAxxxx
        # ========================================================

        loop_matches = re.findall(
            rf"\bLOOP\s+AT\b.*?\b{re.escape(table)}\b",
            upper,
            flags=re.IGNORECASE | re.DOTALL,
        )

        count = len(loop_matches)

        if count:

            contribution = min(count, 2) * 12.0

            matches.append(
                f"LOOP AT {table} "
                f"({count}) [+{contribution:.1f}]"
            )

            score += contribution

    # ============================================================
    # INTENT-SPECIFIC BOOST
    # ============================================================

    if intent in {
        "data_read",
        "field_lookup",
        "code_generation",
    }:

        # Se c'è SELECT vera.
        if re.search(
            r"\bSELECT\b",
            upper,
        ):

            matches.append(
                "INTENT DATA ACCESS [+20.0]"
            )

            score += 20.0

        # Se esiste una WHERE.
        if re.search(
            r"\bWHERE\b",
            upper,
        ):

            matches.append(
                "INTENT WHERE [+10.0]"
            )

            score += 10.0

    # ============================================================
    # PENALIZZAZIONI
    # ============================================================

    penalty = 0.0

    # ------------------------------------------------------------
    # Solo Dynpro / FIELD usage
    # ------------------------------------------------------------

    dynpro_patterns = [
        r"\bFIELD\s+PA\d{4}-",
        r"\bCHAIN\b",
        r"\bMODULE\b.*?\bINPUT\b",
        r"\bMODULE\b.*?\bOUTPUT\b",
    ]

    dynpro_hits = 0

    for pattern in dynpro_patterns:

        dynpro_hits += len(
            re.findall(
                pattern,
                upper,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )

    has_select = bool(
        re.search(
            r"\bSELECT\b",
            upper,
        )
    )

    if dynpro_hits and not has_select:

        penalty += 45.0

        matches.append(
            f"PENALTY DYNPROMODEL [+{-45.0:.1f}]"
        )

    elif dynpro_hits and has_select:

        penalty += min(
            dynpro_hits * 5.0,
            20.0,
        )

        matches.append(
            f"PENALTY MIXED DYNPRO "
            f"[+{-min(dynpro_hits * 5.0, 20.0):.1f}]"
        )

    # ------------------------------------------------------------
    # Semplice campo PAxxxx-FIELD senza accesso dati
    # ------------------------------------------------------------

    qualified_count = len(
        re.findall(
            r"\bPA\d{4}-[A-Z0-9_]+\b",
            upper,
        )
    )

    if (
        qualified_count
        and not has_select
        and intent in {
            "data_read",
            "code_generation",
            "field_lookup",
        }
    ):

        penalty += 30.0

        matches.append(
            "PENALTY FIELD-ONLY [+-30.0]"
        )

    # ------------------------------------------------------------
    # Commenti/documentazione residua
    # ------------------------------------------------------------

    comment_like = 0

    for line in text.splitlines():

        stripped = line.strip()

        if stripped.startswith("*"):
            comment_like += 1

        if stripped.startswith('"'):
            comment_like += 1

    if comment_like > 10 and not has_select:

        penalty += 15.0

        matches.append(
            "PENALTY COMMENT-HEAVY [+-15.0]"
        )

    score -= penalty

    return score, matches


# ================================================================
# QUERY EXACT BONUS
# ================================================================

def calcola_query_exact_bonus(
    text,
    query_profile,
):

    upper = text.upper()

    bonus = 0.0

    # Query con una sola tabella.
    if len(query_profile["tables"]) == 1:

        table = query_profile["tables"][0]

        if contiene_termine(upper, table):

            bonus += 15.0

    # Query specifica con campi.
    for field in query_profile["fields"]:

        if contiene_termine(upper, field):

            bonus += 3.0

    return min(
        bonus,
        25.0,
    )


# ================================================================
# SOURCE BONUS
# ================================================================

def calcola_source_bonus(metadata):

    bonus = 0.0

    object_type = str(
        metadata.get(
            "object_type",
            "",
        )
    ).upper()

    language = str(
        metadata.get(
            "language",
            "",
        )
    ).upper()

    if language == "ABAP":
        bonus += 5.0

    if object_type in {
        "REPORT",
        "FUNCTION",
        "FUNCTION_MODULE",
        "CLASS",
        "METHOD",
        "PROGRAM",
    }:

        bonus += 3.0

    return bonus


# ================================================================
# CONTENT SIMILARITY
# ================================================================

def similarity(a, b):

    na = normalizza_per_hash(a)
    nb = normalizza_per_hash(b)

    if not na or not nb:
        return 0.0

    # Limitiamo il confronto per evitare costi eccessivi.
    na = na[:5000]
    nb = nb[:5000]

    return SequenceMatcher(
        None,
        na,
        nb,
    ).ratio()


# ================================================================
# NORMALIZZA SOURCE FAMILY
# ================================================================

def normalizza_source_family(source_file):

    if not source_file:
        return ""

    stem = Path(
        str(source_file)
    ).stem.upper()

    # Elimina suffissi tipici di copie/varianti.
    stem = re.sub(
        r"(_\d+)$",
        "",
        stem,
    )

    stem = re.sub(
        r"(_ELE\d*)$",
        "",
        stem,
    )

    stem = re.sub(
        r"(_OLD|_TEST|_COPY|_BACKUP)$",
        "",
        stem,
    )

    return stem


# ================================================================
# DEDUPLICAZIONE
# ================================================================

def dedup_content(results):

    unique = []

    hashes = set()

    removed = 0

    for item in results:

        h = item["content_hash"]

        if h in hashes:

            removed += 1

            continue

        hashes.add(h)

        unique.append(item)

    return unique, removed


# ================================================================
# DIVERSIFICAZIONE RISULTATI
# ================================================================

def diversify_results(
    ranked_results,
    final_k,
):

    selected = []

    source_family_count = {}

    duplicate_suppressed = 0

    candidates = ranked_results[
        :MAX_CANDIDATES_FOR_DIVERSITY
    ]

    for item in candidates:

        source_family = normalizza_source_family(
            item["metadata"].get(
                "source_file",
                "",
            )
        )

        # Non permettiamo che una stessa famiglia occupi
        # tutto il ranking.
        family_count = source_family_count.get(
            source_family,
            0,
        )

        if family_count >= 2:

            duplicate_suppressed += 1

            continue

        is_near_duplicate = False

        for selected_item in selected:

            sim = similarity(
                item["content"],
                selected_item["content"],
            )

            if sim >= NEAR_DUPLICATE_THRESHOLD:

                is_near_duplicate = True

                duplicate_suppressed += 1

                break

        if is_near_duplicate:
            continue

        selected.append(item)

        source_family_count[source_family] = (
            family_count + 1
        )

        if len(selected) >= final_k:
            break

    return selected, duplicate_suppressed


# ================================================================
# EXACT SCAN BATCH
# ================================================================

def exact_scan(
    collection,
    query_profile,
    query,
):

    print()
    print("=" * 70)
    print("RICERCA ESATTA SAP V5")
    print("=" * 70)

    total = collection.count()

    print(
        f"Chunk VectorDB totali : {total}"
    )

    # Termini da cercare.
    terms = set(
        query_profile["tables"]
        + query_profile["fields"]
    )

    # Aggiungiamo i termini SAP della tabella target.
    for table in query_profile["tables"]:

        for term in SAP_TERMS.get(
            table,
            [],
        ):

            terms.add(term)

    terms = sorted(
        terms,
        key=len,
        reverse=True,
    )

    print(
        "Termini exact:",
        ", ".join(terms),
    )

    results = []

    scanned = 0

    for offset in range(
        0,
        total,
        EXACT_SCAN_BATCH,
    ):

        batch = collection.get(
            limit=EXACT_SCAN_BATCH,
            offset=offset,
            include=[
                "documents",
                "metadatas",
            ],
        )

        documents = batch.get(
            "documents",
            [],
        )

        metadatas = batch.get(
            "metadatas",
            [],
        )

        if not documents:
            break

        for content, metadata in zip(
            documents,
            metadatas,
        ):

            if not content:
                continue

            upper = content.upper()

            found = []

            for term in terms:

                if contiene_termine(
                    upper,
                    term,
                ):

                    found.append(term)

            if not found:
                continue

            exact_score = 0.0

            for term in found:

                # Tabella target.
                if term in query_profile["tables"]:

                    exact_score += 8.0

                # Campo richiesto.
                elif term in query_profile["fields"]:

                    exact_score += 5.0

                # Key field.
                elif term in query_profile["key_fields"]:

                    exact_score += 3.0

                else:

                    exact_score += 0.5

            exact_score = min(
                exact_score,
                30.0,
            )

            item = {
                "content": content,
                "metadata": metadata or {},
                "exact_score": exact_score,
                "semantic_score": 0.0,
                "content_hash": content_hash(
                    content
                ),
            }

            results.append(item)

        scanned += len(documents)

        if scanned % 5000 == 0 or scanned == total:

            print(
                f"Scansione: {scanned}/{total}"
            )

    return results


# ================================================================
# SEMANTIC SEARCH
# ================================================================

def semantic_search(
    vector_db,
    query,
):

    print()
    print("=" * 70)
    print("RICERCA SEMANTICA V5")
    print("=" * 70)

    docs_with_scores = (
        vector_db.similarity_search_with_score(
            query,
            k=SEMANTIC_K,
        )
    )

    results = []

    for doc, distance in docs_with_scores:

        # Chroma restituisce distanza.
        # La trasformiamo in uno score qualitativo.
        semantic_score = 1.0 / (
            1.0 + float(distance)
        )

        results.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata or {},
                "semantic_score": semantic_score,
                "exact_score": 0.0,
                "content_hash": content_hash(
                    doc.page_content
                ),
            }
        )

    return results


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 70)
    print("SAP ABAP RAG TEST - V5")
    print("=" * 70)

    print(
        f"VectorDB : {DB_DIR}"
    )

    print(
        f"Collection : {COLLECTION_NAME}"
    )

    print(
        f"Embedding : {EMBEDDING_MODEL}"
    )

    # ------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------

    query = input(
        "\nInserisci la query: "
    ).strip()

    if not query:

        print(
            "Query vuota."
        )

        return

    # ------------------------------------------------------------
    # QUERY PROFILE
    # ------------------------------------------------------------

    query_profile = estrai_query_profile(
        query
    )

    print()
    print("=" * 70)
    print("QUERY PROFILE V5")
    print("=" * 70)

    print(
        f"Query           : {query}"
    )

    print(
        f"Tabelle         : "
        f"{', '.join(query_profile['tables']) or '-'}"
    )

    print(
        f"Campi           : "
        f"{', '.join(query_profile['fields']) or '-'}"
    )

    print(
        f"Campi richiesti : "
        f"{', '.join(query_profile['requested_fields']) or '-'}"
    )

    print(
        f"Campi chiave    : "
        f"{', '.join(query_profile['key_fields']) or '-'}"
    )

    print(
        f"Intent           : "
        f"{query_profile['intent']}"
    )

    # ------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # ------------------------------------------------------------
    # CHROMA
    # ------------------------------------------------------------

    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )

    collection = vector_db._collection

    # ------------------------------------------------------------
    # SEMANTIC
    # ------------------------------------------------------------

    semantic_results = semantic_search(
        vector_db,
        query,
    )

    # ------------------------------------------------------------
    # EXACT
    # ------------------------------------------------------------

    exact_results = exact_scan(
        collection,
        query_profile,
        query,
    )

    # ------------------------------------------------------------
    # FUSIONE
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FUSIONE + DEDUPLICAZIONE V5")
    print("=" * 70)

    print(
        f"Risultati semantici : "
        f"{len(semantic_results)}"
    )

    print(
        f"Risultati esatti    : "
        f"{len(exact_results)}"
    )

    merged = {}

    # ------------------------------------------------------------
    # Semantic
    # ------------------------------------------------------------

    for item in semantic_results:

        h = item["content_hash"]

        merged[h] = item

    # ------------------------------------------------------------
    # Exact
    # ------------------------------------------------------------

    for item in exact_results:

        h = item["content_hash"]

        if h in merged:

            # Manteniamo il semantic score.
            merged[h]["exact_score"] = max(
                merged[h]["exact_score"],
                item["exact_score"],
            )

        else:

            merged[h] = item

    merged_results = list(
        merged.values()
    )

    print(
        f"Risultati unici     : "
        f"{len(merged_results)}"
    )

    # ------------------------------------------------------------
    # Content dedup
    # ------------------------------------------------------------

    deduped, removed = dedup_content(
        merged_results
    )

    print(
        f"Duplicati per contenuto rimossi: "
        f"{removed}"
    )

    print(
        f"Risultati dopo content dedup   : "
        f"{len(deduped)}"
    )

    # ------------------------------------------------------------
    # STRUCTURAL RANKING
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CALCOLO SAP STRUCTURAL QUERY-AWARE RANKING V5")
    print("=" * 70)

    for item in deduped:

        structural_score, structural_matches = (
            calcola_structural_score(
                item["content"],
                query_profile,
            )
        )

        query_exact_bonus = (
            calcola_query_exact_bonus(
                item["content"],
                query_profile,
            )
        )

        source_bonus = (
            calcola_source_bonus(
                item["metadata"]
            )
        )

        # Semantic component.
        semantic_component = (
            item["semantic_score"] * 25.0
        )

        # Exact component.
        exact_component = (
            item["exact_score"] * 2.0
        )

        final_score = (
            semantic_component
            + exact_component
            + structural_score
            + query_exact_bonus
            + source_bonus
        )

        item["semantic_component"] = (
            semantic_component
        )

        item["exact_component"] = (
            exact_component
        )

        item["structural_score"] = (
            structural_score
        )

        item["query_exact_bonus"] = (
            query_exact_bonus
        )

        item["source_bonus"] = (
            source_bonus
        )

        item["final_score"] = (
            final_score
        )

        item["structural_matches"] = (
            structural_matches
        )

    # ------------------------------------------------------------
    # SORT
    # ------------------------------------------------------------

    ranked = sorted(
        deduped,
        key=lambda x: x["final_score"],
        reverse=True,
    )

    # ------------------------------------------------------------
    # DIVERSIFICATION
    # ------------------------------------------------------------

    final_results, diversity_removed = (
        diversify_results(
            ranked,
            FINAL_K,
        )
    )

    # ------------------------------------------------------------
    # RISULTATI
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RISULTATI FINALI - "
        "SAP STRUCTURAL QUERY-AWARE HYBRID RANKING V5"
    )
    print("=" * 70)

    print(
        f"Query: {query}"
    )

    print(
        f"Intent: {query_profile['intent']}"
    )

    print(
        f"Risultati mostrati: "
        f"{len(final_results)}"
    )

    print(
        f"Risultati soppressi per diversità: "
        f"{diversity_removed}"
    )

    # ------------------------------------------------------------
    # COUNTERS
    # ------------------------------------------------------------

    diagnostic = {
        "ABAP": 0,
        "table_access": 0,
        "query_fields": 0,
        "key_fields": 0,
        "select_from_table": 0,
        "select_single": 0,
        "select_field_where_key": 0,
        "dynpro_penalty": 0,
    }

    # ------------------------------------------------------------
    # PRINT TOP RESULTS
    # ------------------------------------------------------------

    for rank, item in enumerate(
        final_results,
        start=1,
    ):

        metadata = item["metadata"]

        content = item["content"]

        upper = content.upper()

        source_file = metadata.get(
            "source_file",
            "",
        )

        source_path = metadata.get(
            "source_path",
            "",
        )

        object_type = metadata.get(
            "object_type",
            "",
        )

        object_name = metadata.get(
            "object_name",
            "",
        )

        language = metadata.get(
            "language",
            "",
        )

        chunk_index = metadata.get(
            "chunk_index",
            "",
        )

        print()
        print("-" * 70)
        print(
            f"RANK #{rank}"
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

        # Exact terms.
        found_terms = trova_termini_sap(
            upper
        )

        print(
            "Exact terms       : "
            + (
                ", ".join(found_terms)
                if found_terms
                else "-"
            )
        )

        # Structural matches.
        print()
        print(
            "Structural matches:"
        )

        if item["structural_matches"]:

            for match in item[
                "structural_matches"
            ]:

                print(
                    f"  + {match}"
                )

        else:

            print(
                "  nessuna"
            )

        # Metadata.
        print()
        print(
            f"Source file       : {source_file}"
        )

        print(
            f"Source path       : {source_path}"
        )

        print(
            f"Object type       : {object_type}"
        )

        print(
            f"Object name       : {object_name}"
        )

        print(
            f"Language          : {language}"
        )

        print(
            f"Chunk index       : {chunk_index}"
        )

        print(
            f"Content hash      : "
            f"{item['content_hash'][:16]}"
        )

        # --------------------------------------------------------
        # Diagnostic counters
        # --------------------------------------------------------

        if language.upper() == "ABAP":

            diagnostic["ABAP"] += 1

        if query_profile["tables"]:

            if any(
                re.search(
                    rf"\bFROM\s+{re.escape(table)}\b",
                    upper,
                    re.IGNORECASE,
                )
                for table in query_profile["tables"]
            ):

                diagnostic["table_access"] += 1

        if query_profile["fields"]:

            if any(
                contiene_termine(
                    upper,
                    field,
                )
                for field in query_profile["fields"]
            ):

                diagnostic["query_fields"] += 1

        if query_profile["key_fields"]:

            if any(
                contiene_termine(
                    upper,
                    field,
                )
                for field in query_profile["key_fields"]
            ):

                diagnostic["key_fields"] += 1

        if re.search(
            r"\bSELECT\b.*?\bFROM\s+PA\d{4}\b",
            upper,
            re.IGNORECASE | re.DOTALL,
        ):

            diagnostic[
                "select_from_table"
            ] += 1

        if re.search(
            r"\bSELECT\s+SINGLE\b",
            upper,
            re.IGNORECASE,
        ):

            diagnostic[
                "select_single"
            ] += 1

        # Relazione richiesta.
        relation_found = False

        for table in query_profile["tables"]:

            for requested_field in query_profile[
                "requested_fields"
            ]:

                for key_field in query_profile[
                    "key_fields"
                ]:

                    pattern = (
                        rf"\bSELECT\b.*?"
                        rf"\b{re.escape(requested_field)}\b"
                        rf".*?"
                        rf"\bFROM\s+{re.escape(table)}\b"
                        rf".*?"
                        rf"\bWHERE\b.*?"
                        rf"\b{re.escape(key_field)}\b"
                    )

                    if re.search(
                        pattern,
                        upper,
                        re.IGNORECASE | re.DOTALL,
                    ):

                        relation_found = True

        if relation_found:

            diagnostic[
                "select_field_where_key"
            ] += 1

        if any(
            "DYNPRO" in match
            or "FIELD-ONLY" in match
            for match in item[
                "structural_matches"
            ]
        ):

            diagnostic[
                "dynpro_penalty"
            ] += 1

        # --------------------------------------------------------
        # CONTENT
        # --------------------------------------------------------

        print()
        print(
            "Content:"
        )

        print(
            content
        )

    # ------------------------------------------------------------
    # DIAGNOSTICA
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("DIAGNOSTICA RANKING V5")
    print("=" * 70)

    print(
        f"ABAP                         : "
        f"{diagnostic['ABAP']}"
    )

    print(
        f"Accesso diretto tabella     : "
        f"{diagnostic['table_access']}"
    )

    print(
        f"Query fields presenti       : "
        f"{diagnostic['query_fields']}"
    )

    print(
        f"Key fields presenti         : "
        f"{diagnostic['key_fields']}"
    )

    print(
        f"SELECT FROM tabella         : "
        f"{diagnostic['select_from_table']}"
    )

    print(
        f"SELECT SINGLE               : "
        f"{diagnostic['select_single']}"
    )

    print(
        f"SELECT campo + WHERE key    : "
        f"{diagnostic['select_field_where_key']}"
    )

    print(
        f"Risultati penalizzati Dynpro: "
        f"{diagnostic['dynpro_penalty']}"
    )

    # ------------------------------------------------------------
    # VALUTAZIONE AUTOMATICA
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("VALUTAZIONE AUTOMATICA V5")
    print("=" * 70)

    if query_profile["intent"] == "data_read":

        if diagnostic[
            "select_field_where_key"
        ] > 0:

            print(
                "✓ RETRIEVAL MOLTO PROMETTENTE"
            )

            print(
                "  Trovata la relazione diretta:"
            )

            print(
                "  SELECT + campo richiesto + "
                "tabella + WHERE chiave"
            )

        elif diagnostic[
            "select_from_table"
        ] > 0:

            print(
                "△ RETRIEVAL PROMETTENTE MA "
                "NON ANCORA OTTIMALE"
            )

            print(
                "  Sono presenti SELECT sulla "
                "tabella, ma non è stata trovata "
                "sufficientemente spesso la "
                "relazione campo/chiave."
            )

        else:

            print(
                "✗ RETRIEVAL DEBOLE"
            )

            print(
                "  Non sono state trovate SELECT "
                "sufficientemente pertinenti."
            )

    elif query_profile["intent"] == "table_reference":

        if diagnostic[
            "select_from_table"
        ] > 0:

            print(
                "✓ RIFERIMENTI ALLA TABELLA "
                "TROVATI IN CODICE ABAP"
            )

        else:

            print(
                "△ RIFERIMENTO ALLA TABELLA "
                "SENZA ACCESSI SELECT SIGNIFICATIVI"
            )

    else:

        if diagnostic[
            "select_from_table"
        ] > 0:

            print(
                "✓ CODICE ABAP RILEVANTE TROVATO"
            )

        else:

            print(
                "△ RANKING DA ANALIZZARE"
            )

    print()
    print("=" * 70)
    print("FINE TEST V5")
    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()