# SAP ABAP RAG TEST - V5.6
# SQL Relationship-Aware Ranking
# VectorDB is READ-ONLY. No embeddings or Chroma data are modified.

import re, csv, time, pickle
from pathlib import Path
from datetime import datetime
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DB_DIR = Path(r"C:\Progetto_AI\Abap_VectorDB")
COLLECTION_NAME = "abap_hcm"
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
SEMANTIC_K = 30
FINAL_K = 15
BATCH_SIZE = 500
OUTPUT_DIR = Path(r"C:\Progetto_AI\RAG_V5_6_RESULTS")
CACHE_FILE = OUTPUT_DIR / "exact_cache_v5_6.pkl"
CACHE_VERSION = "5.6"

SAP_TERMS = {
    "PA0001": ["PA0001","P0001","PERNR","BUKRS","WERKS","PERSG","PERSK","BTRTL","GSBER","KOSTL","ORGEH","PLANS","STELL","SACHZ","BEGDA","ENDDA","STAT2"],
    "PA0002": ["PA0002","P0002","PERNR","VORNA","NACHN","GBDAT","GESCH","FAMST"],
    "PA0007": ["PA0007","P0007","PERNR","WOSTD","SCHKZ","ZTERF","ARBPL"],
    "PA0008": ["PA0008","P0008","PERNR","BET01","BET02","WAERS","TRFAR","TRFGB","TRFGR","TRFST"],
}

FUNCTIONAL_CONCEPTS = {
    "centro di costo": {"terms":["KOSTL","PA0001"],"related":["PERNR","BUKRS","WERKS","ORGEH"]},
    "centro costo": {"terms":["KOSTL","PA0001"],"related":["PERNR","BUKRS","WERKS","ORGEH"]},
    "costo del dipendente": {"terms":["KOSTL","PERNR","PA0001"],"related":["BUKRS","WERKS","ORGEH"]},
    "dati organizzativi": {"terms":["PA0001","PERNR","BUKRS","WERKS","BTRTL","PERSG","PERSK","KOSTL","ORGEH","PLANS","STELL"],"related":["BEGDA","ENDDA","STAT2"]},
    "dati organizzativi dipendente": {"terms":["PA0001","PERNR","BUKRS","WERKS","BTRTL","PERSG","PERSK","KOSTL","ORGEH","PLANS","STELL"],"related":["BEGDA","ENDDA","STAT2"]},
    "dipendente": {"terms":["PERNR","PA0001"],"related":["BUKRS","WERKS","PERSG","PERSK"]},
}

# SAP HCM structural priors. These are deliberately only tie-breakers;
# they do NOT create a strong relationship score by themselves.
FIELD_TABLE_PRIORS = {
    "PERNR": ["PA0001","PA0002","PA0007","PA0008"],
    "KOSTL": ["PA0001"],
    "WERKS": ["PA0001"],
    "BTRTL": ["PA0001"],
    "PERSG": ["PA0001"],
    "PERSK": ["PA0001"],
    "ORGEH": ["PA0001"],
    "PLANS": ["PA0001"],
    "STELL": ["PA0001"],
    "BUKRS": ["PA0001"],
    "GSBER": ["PA0001"],
}

TEST_CASES = [
    ("T01","PA0001","table_reference",["PA0001"],["PA0001"]),
    ("T02","PA0002","table_reference",["PA0002"],["PA0002"]),
    ("T03","PA0007","table_reference",["PA0007"],["PA0007"]),
    ("T04","PA0008","table_reference",["PA0008"],["PA0008"]),
    ("T05","PERNR","field_lookup",["PERNR","PA0001"],["PA0001"]),
    ("T06","KOSTL","field_lookup",["KOSTL","PA0001"],["PA0001"]),
    ("T07","WERKS","field_lookup",["WERKS","PA0001"],["PA0001"]),
    ("T08","ORGEH","field_lookup",["ORGEH","PA0001"],["PA0001"]),
    ("T09","centro di costo del dipendente","functional_lookup",["KOSTL","PERNR","PA0001"],["PA0001"]),
    ("T10","leggere PA0001 per PERNR","code_lookup",["PA0001","PERNR"],["PA0001"]),
    ("T11","SELECT PA0001 WHERE PERNR","code_pattern",["PA0001","PERNR"],["PA0001"]),
    ("T12","dati organizzativi dipendente SAP HCM","functional_lookup",["PA0001","PERNR","WERKS","PERSG","PERSK","BTRTL","ORGEH","PLANS","STELL"],["PA0001"]),
]


def norm(s):
    return re.sub(r"[^A-Z0-9_/~]", " ", (s or "").upper())


def present(s, t):
    return bool(re.search(rf"(?<![A-Z0-9_]){re.escape(norm(t))}(?![A-Z0-9_])", norm(s)))


def count_term(s, t):
    return len(re.findall(rf"(?<![A-Z0-9_]){re.escape(norm(t))}(?![A-Z0-9_])", norm(s)))


def exact_terms(s, terms):
    return list(dict.fromkeys([t.upper() for t in terms if present(s, t)]))


def detect_tables(q):
    return sorted(t for t in SAP_TERMS if re.search(rf"\b{t}\b", q.upper()))


def detect_fields(q):
    out = []
    for terms in SAP_TERMS.values():
        for t in terms:
            if t.startswith("PA") and len(t) == 6:
                continue
            if t.startswith("P") and len(t) == 5:
                continue
            if re.search(rf"\b{re.escape(t)}\b", q.upper()):
                out.append(t)
    return list(dict.fromkeys(out))


def detect_concepts(q):
    q = q.lower()
    return [p for p in FUNCTIONAL_CONCEPTS if p in q]


def infer_intent(q):
    u = q.upper().strip()
    if re.search(r"\bSELECT\b|\bWHERE\b", u):
        return "code_pattern"
    if re.search(r"\b(LEGGERE|LEGGI|RECUPERARE|RECUPERA|ESTRARRE|OTTENERE|SELEZIONARE|READ|RETRIEVE|GET)\b", u):
        return "code_lookup"
    if detect_concepts(q):
        return "functional_lookup"
    if detect_tables(q) and not detect_fields(q):
        return "table_reference"
    if detect_fields(q):
        return "field_lookup"
    return "semantic_lookup"


def profile(q):
    tables = detect_tables(q)
    fields = detect_fields(q)
    concepts = detect_concepts(q)
    functional, related = [], []
    for c in concepts:
        functional += FUNCTIONAL_CONCEPTS[c]["terms"]
        related += FUNCTIONAL_CONCEPTS[c]["related"]
    requested = list(dict.fromkeys(fields + functional))
    terms = list(dict.fromkeys(tables + requested + related + functional))
    return {
        "query": q,
        "intent": infer_intent(q),
        "tables": tables,
        "explicit_fields": fields,
        "concepts": concepts,
        "functional_terms": list(dict.fromkeys(functional)),
        "related_terms": list(dict.fromkeys(related)),
        "requested_fields": requested,
        "terms": terms,
    }


def create_db():
    e = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=e, persist_directory=str(DB_DIR))


def load_cache(db):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = db._collection.count()
    if CACHE_FILE.exists():
        try:
            p = pickle.loads(CACHE_FILE.read_bytes())
            if p.get("version") == CACHE_VERSION and p.get("total") == total:
                print(f"Cache V5.5 caricata: {len(p['items'])} chunk")
                return p["items"], 0
        except Exception:
            pass

    start = time.perf_counter()
    items = []
    for off in range(0, total, BATCH_SIZE):
        lim = min(BATCH_SIZE, total - off)
        d = db._collection.get(limit=lim, offset=off, include=["documents", "metadatas"])
        docs = d.get("documents") or []
        metas = d.get("metadatas") or []
        for i, c in enumerate(docs):
            if c:
                items.append({"content": c, "metadata": metas[i] if i < len(metas) else {}})
        done = min(off + len(docs), total)
        if done % 5000 == 0 or done >= total:
            print(f"Scansionati: {done}/{total} | cache: {len(items)}")

    CACHE_FILE.write_bytes(pickle.dumps({
        "version": CACHE_VERSION,
        "total": total,
        "created_at": datetime.now().isoformat(),
        "items": items,
    }, protocol=pickle.HIGHEST_PROTOCOL))
    return items, time.perf_counter() - start


def semantic(db, q):
    st = time.perf_counter()
    try:
        rr = db.similarity_search_with_relevance_scores(q, k=SEMANTIC_K)
    except Exception:
        rr = [(d, 0.0) for d in db.similarity_search(q, k=SEMANTIC_K)]
    out = []
    for d, s in rr:
        out.append({
            "document": d,
            "semantic_score": float(s),
            "exact_score": 0.0,
            "exact_terms": [],
            "structural_score": 0.0,
            "field_score": 0.0,
            "functional_score": 0.0,
            "context_score": 0.0,
            "relationship_score": 0.0,
            "relationship_penalty": 0.0,
            "relationship_type": "NONE",
            "relationship_reason": "",
            "relationship_hit": False,
        })
    return out, time.perf_counter() - st


def exact(cache, p):
    out = []
    for x in cache:
        found = exact_terms(x["content"], p["terms"])
        if not found:
            continue
        s = 0
        for t in found:
            if t in p["tables"]:
                s += 8
            elif t in p["explicit_fields"]:
                s += 3
            elif t in p["functional_terms"]:
                s += 2.5
            elif t in p["related_terms"]:
                s += 1
            elif t.startswith("PA"):
                s += 3
            else:
                s += 0.5
        s += min(len(found) * 0.5, 5)
        out.append({
            "document": None,
            "content": x["content"],
            "metadata": x["metadata"],
            "semantic_score": 0.0,
            "exact_score": s,
            "exact_terms": found,
            "structural_score": 0.0,
            "field_score": 0.0,
            "functional_score": 0.0,
            "context_score": 0.0,
            "relationship_score": 0.0,
            "relationship_penalty": 0.0,
            "relationship_type": "NONE",
            "relationship_reason": "",
            "relationship_hit": False,
        })
    return out


def skey(x):
    d = x.get("document")
    m = d.metadata if d is not None else x.get("metadata", {})
    return (m.get("source_file"), m.get("chunk_index"), m.get("chunk_hash"))


def ckey(x):
    d = x.get("document")
    c = d.page_content if d is not None else x.get("content", "")
    c = re.sub(r"<[^>]+>", " ", c)
    c = re.sub(r"&(?:nbsp|amp|lt|gt);", " ", c, flags=re.I)
    return re.sub(r"[^A-Z0-9]", "", c.upper())


def merge(a, b):
    m = {skey(x): x for x in a}
    for x in b:
        k = skey(x)
        if k in m:
            m[k]["exact_score"] = x["exact_score"]
            m[k]["exact_terms"] = x["exact_terms"]
        else:
            x["document"] = Document(page_content=x["content"], metadata=x["metadata"])
            m[k] = x

    u = {}
    dup = 0
    for x in m.values():
        k = ckey(x)
        if k in u:
            dup += 1
            if x["exact_score"] > u[k]["exact_score"]:
                u[k] = x
        else:
            u[k] = x
    return list(u.values()), dup


# ---------------------------------------------------------------------------
# V5.5 SQL RELATIONSHIP PARSER
# ---------------------------------------------------------------------------

def clean_sql_fragment(s):
    s = re.sub(r"\*[^\n]*", " ", s)  # basic ABAP comment removal
    s = re.sub(r"\"[^\n]*", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def split_abap_statements(text):
    """Return conservative ABAP statement fragments.

    V5.6 no longer requires a period followed by whitespace. Chroma chunks can
    cut an ABAP statement at an arbitrary character, and HTML/text conversion
    can also remove newlines. We therefore split at periods, SELECT starts,
    and chunk boundaries while preserving SELECT blocks.
    """
    text = text or ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?i)(?=\bSELECT(?:\s+SINGLE)?\b)", text)
    out = []
    for part in parts:
        part = clean_sql_fragment(part)
        if not part:
            continue
        # A SELECT fragment ends at its first statement period.
        m = re.search(r"\.", part)
        if m:
            first = part[:m.end()]
            rest = part[m.end():].strip()
            if first.strip():
                out.append(first.strip())
            if rest and not re.match(r"(?i)^SELECT(?:\s+SINGLE)?\b", rest):
                out.append(rest)
        else:
            out.append(part)
    return out

def _fields_from_projection(projection):
    p = clean_sql_fragment(projection).upper()
    # Remove common ABAP SQL additions that are not field names.
    p = re.sub(r"\bINTO(?:\s+CORRESPONDING\s+FIELDS\s+OF)?\b.*", " ", p, flags=re.I)
    p = re.sub(r"\bUP\s+TO\s+\d+\s+ROWS\b", " ", p, flags=re.I)
    p = re.sub(r"\bDISTINCT\b", " ", p, flags=re.I)
    p = re.sub(r"\bSINGLE\b", " ", p, flags=re.I)
    candidates = re.findall(r"(?<![A-Z0-9_])([A-Z][A-Z0-9_]{1,29})(?![A-Z0-9_])", p)
    ignored = {
        "INTO","TABLE","FROM","WHERE","ORDER","GROUP","BY","HAVING","ENDSELECT",
        "AS","INNER","LEFT","RIGHT","OUTER","JOIN","ON","UP","TO","ROWS","CORRESPONDING",
        "FIELDS","OF","CLIENT","SPECIFIED","DISTINCT","SINGLE","FOR","ALL","ENTRIES",
    }
    return list(dict.fromkeys(x for x in candidates if x not in ignored))


def _fields_from_where(where):
    w = clean_sql_fragment(where).upper()
    # We only regard identifiers immediately participating in conditions as WHERE fields.
    candidates = []
    for m in re.finditer(r"(?:^|\bAND\b|\bOR\b|\()\s*(?:[A-Z0-9_]+-)?([A-Z][A-Z0-9_]*)\s*(?:=|<>|<=|>=|<|>|IN\b|BETWEEN\b|LIKE\b|IS\b)", w):
        candidates.append(m.group(1))
    # Also catch qualified fields if syntax is slightly different.
    for m in re.finditer(r"\b(?:PA\d{4})-([A-Z][A-Z0-9_]*)\b", w):
        candidates.append(m.group(1))
    return list(dict.fromkeys(candidates))


def extract_sql_relations(text):
    """Extract SELECT/FROM/WHERE relations from one SQL statement at a time.

    Handles ABAP formatting variants, chunk boundaries, and qualified fields
    such as PA0001~PERNR and PA0001-PERNR.
    """
    relations = []
    for stmt in split_abap_statements(text):
        if not re.search(r"\bSELECT\b", stmt, re.I) or not re.search(r"\bFROM\b", stmt, re.I):
            continue
        fm = re.search(r"\bFROM\s+([A-Z0-9_/~]+)", stmt, re.I)
        if not fm:
            continue
        table = fm.group(1).upper().replace("~", "")

        sm = re.search(r"\bSELECT\s+(SINGLE\s+)?(.*?)(?=\bINTO\b|\bFROM\b)", stmt, re.I)
        if not sm:
            continue
        single = bool(sm.group(1))
        projection = sm.group(2)
        fields_select = _fields_from_projection(projection)

        # WHERE belongs only to this extracted SELECT fragment.
        wm = re.search(
            r"\bWHERE\s+(.*?)(?=\bORDER\s+BY\b|\bGROUP\s+BY\b|\bHAVING\b|\bUP\s+TO\b|\bENDSELECT\b|\.|$)",
            stmt, re.I
        )
        where = wm.group(1) if wm else ""
        fields_where = _fields_from_where(where) if where else []

        # Qualified fields are technical proof even when the projection parser
        # cannot classify them as plain identifiers.
        qualified = []
        for m in re.finditer(rf"\b{re.escape(table)}[-~]([A-Z][A-Z0-9_]*)\b", stmt, re.I):
            qualified.append(m.group(1).upper())
        fields_select = list(dict.fromkeys(fields_select + qualified))
        fields_where = list(dict.fromkeys(fields_where + [
            m.group(1).upper() for m in re.finditer(rf"\b{re.escape(table)}[-~]([A-Z][A-Z0-9_]*)\b", where, re.I)
        ]))

        relations.append({
            "table": table,
            "fields_select": fields_select,
            "fields_where": fields_where,
            "single": single,
            "statement": stmt,
        })
    return relations

def relationship_score(c, p):
    """V5.6 relationship scoring: specificity beats term co-presence."""
    text = c.upper()
    relations = extract_sql_relations(text)
    requested = list(dict.fromkeys(p["requested_fields"] + p["explicit_fields"] + p["functional_terms"]))
    target_tables = p["tables"][:]
    score = 0.0
    best_type, best_reason, best_rank = "NONE", "", -1

    def consider(s, typ, reason):
        nonlocal score, best_type, best_reason, best_rank
        rank = {
            "NONE":0, "TABLE_FROM":1, "SELECT_FROM":2, "SELECT_FIELD_FROM":3,
            "SELECT_FIELDS_FROM":4, "SELECT_FIELD_WHERE":5,
            "SELECT_FIELDS_WHERE":6, "SELECT_KOSTL_FROM_PA0001_WHERE_PERNR":8,
            "PA0001_ORG_FIELDS":5, "PA0001_ORG_FIELDS_PERNR":7
        }.get(typ, 1)
        if s > score:
            score = s
        if rank > best_rank or (rank == best_rank and s > 0 and not best_reason):
            best_rank, best_type, best_reason = rank, typ, reason

    org = {"BUKRS","WERKS","BTRTL","PERSG","PERSK","KOSTL","ORGEH","PLANS","STELL"}

    for r in relations:
        table = r["table"]
        sel = set(r["fields_select"])
        wh = set(r["fields_where"])

        # Explicit table queries.
        if table in target_tables:
            relevant_sel = set(requested) & sel
            relevant_where = set(requested) & wh

            if not relevant_sel and not relevant_where:
                consider(18, "TABLE_FROM", f"FROM {table} nello stesso SELECT")
                continue

            s = 45
            typ = "SELECT_FROM"
            if relevant_sel:
                s += 25
                typ = "SELECT_FIELD_FROM" if len(relevant_sel) == 1 else "SELECT_FIELDS_FROM"
            if relevant_where:
                s += 45
                typ = "SELECT_FIELD_WHERE" if len(relevant_sel | relevant_where) == 1 else "SELECT_FIELDS_WHERE"
            if r["single"]:
                s += 15

            if table == "PA0001" and "PERNR" in wh:
                s += 65
            if table == "PA0001" and "KOSTL" in sel and "PERNR" in wh:
                s += 85
                typ = "SELECT_KOSTL_FROM_PA0001_WHERE_PERNR"

            org_sel = sorted(org & sel)
            if table == "PA0001" and len(org_sel) >= 2:
                s += min(len(org_sel) * 15, 90)
                typ = "PA0001_ORG_FIELDS_PERNR" if ("PERNR" in sel or "PERNR" in wh) else "PA0001_ORG_FIELDS"
                if "PERNR" in sel or "PERNR" in wh:
                    s += 30

            consider(s, typ, self_reason(table, sel, wh, r["single"]))

        # Field-only query: SAP table prior is a modest boost only if the field
        # is actually selected/filtered from the prior table.
        if not target_tables:
            for field in requested:
                for tab in FIELD_TABLE_PRIORS.get(field, []):
                    if table != tab:
                        continue
                    if field in sel and field in wh:
                        consider(82, "SELECT_FIELD_WHERE", f"SELECT {field} FROM {tab} + WHERE {field}")
                    elif field in sel:
                        consider(58, "SELECT_FIELD_FROM", f"SELECT {field} FROM {tab}")
                    elif field in wh:
                        consider(42, "SELECT_FIELD_WHERE", f"WHERE {field} su {tab}")

        # Qualified ABAP usage is useful, but never outranks proven SQL.
        for field in requested:
            for tab in target_tables:
                if re.search(rf"\b{re.escape(tab)}[-~]{re.escape(field)}\b", text):
                    consider(22, "FIELD_FROM", f"{tab}-{field}")

    return score, best_type, best_reason, relations


def relationship_penalty_for_intent(x, p):
    typ = x.get("relationship_type", "NONE")
    rel = x.get("relationship_score", 0.0)
    if p["intent"] in {"code_lookup", "code_pattern"} and "PA0001" in p["tables"]:
        if typ == "TABLE_FROM":
            return 75.0
        if typ in {"SELECT_FROM", "SELECT_FIELD_FROM", "SELECT_FIELDS_FROM"} and rel < 100:
            return 35.0
    if p["intent"] == "functional_lookup" and "KOSTL" in p["functional_terms"]:
        if typ in {"SELECT_FIELD_FROM", "SELECT_FIELD_WHERE"} and "KOSTL" not in " ".join(x.get("relationship_reason", "")).upper():
            return 30.0
    if p["intent"] == "functional_lookup" and any(z in p["concepts"] for z in ("dati organizzativi", "dati organizzativi dipendente")):
        if typ in {"SELECT_FROM", "TABLE_FROM"}:
            return 45.0
    return 0.0

def relationship_noise_penalty(c, p):
    text = c.upper()
    relations = extract_sql_relations(text)
    penalty = 0.0
    hits = []

    # Explicit target table queries: mere co-presence is not a relationship.
    if p["tables"]:
        for tab in p["tables"]:
            has_from_target = any(r["table"] == tab for r in relations)
            if present(text, tab) and not has_from_target:
                penalty += 30
                hits.append(f"NO FROM {tab}")

        if p["intent"] in {"code_lookup","code_pattern"} and "PA0001" in p["tables"]:
            strong = any(r["table"] == "PA0001" and "PERNR" in r["fields_where"] for r in relations)
            if present(text, "PA0001") and present(text, "PERNR") and not strong:
                penalty += 42
                hits.append("PA0001/PERNR without same-SQL WHERE PERNR")

        if p["intent"] == "functional_lookup" and "PA0001" in p["functional_terms"] and "KOSTL" in p["functional_terms"]:
            strong = any(r["table"] == "PA0001" and "KOSTL" in r["fields_select"] and "PERNR" in r["fields_where"] for r in relations)
            medium = any(r["table"] == "PA0001" and "KOSTL" in r["fields_select"] for r in relations)
            if present(text,"PA0001") and present(text,"KOSTL") and present(text,"PERNR") and not strong and not medium:
                penalty += 45
                hits.append("KOSTL/PERNR/PA0001 co-presence only")

    # Functional organizational query: many terms without actual PA0001 SQL
    # should not outrank a smaller but technically valid SQL example.
    if p["intent"] == "functional_lookup" and any(x in p["concepts"] for x in ("dati organizzativi", "dati organizzativi dipendente")):
        if present(text,"PA0001") and not any(r["table"] == "PA0001" for r in relations):
            penalty += 35
            hits.append("PA0001 presente ma nessun SELECT FROM PA0001")

    return penalty, list(dict.fromkeys(hits))


def structural(c, p):
    t = c.upper(); score = 0; hits = []
    for tab in p["tables"]:
        n = len(re.findall(rf"\bFROM\s+{re.escape(tab)}\b", t))
        if n:
            score += min(n,5)*8; hits.append(f"FROM {tab} x{n}")
        n = len(re.findall(rf"\bSELECT\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(tab)}\b", t))
        if n:
            score += min(n,4)*10; hits.append(f"SELECT ... FROM {tab} x{n}")
        n = len(re.findall(rf"\bSELECT\s+SINGLE\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(tab)}\b", t))
        if n:
            score += min(n,4)*14; hits.append(f"SELECT SINGLE ... FROM {tab} x{n}")
        for f in p["requested_fields"]:
            if re.search(rf"\b{re.escape(tab)}-{re.escape(f)}\b", t):
                score += 5; hits.append(f"{tab}-{f}")
    if "PERNR" in p["requested_fields"]:
        n = len(re.findall(r"\bWHERE\b[\s\S]{0,500}?\bPERNR\b", t))
        if n:
            score += min(n,4)*8; hits.append(f"WHERE ... PERNR x{n}")
    for tab in p["tables"]:
        for f in p["requested_fields"]:
            if re.search(rf"\bSELECT\b[\s\S]{{0,500}}?\b{re.escape(f)}\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(tab)}\b", t):
                score += 18; hits.append(f"SELECT {f} FROM {tab}")
    if p["tables"] and "PERNR" in p["requested_fields"] and re.search(rf"\bSELECT\b[\s\S]{{0,900}}?\bFROM\s+(?:{'|'.join(map(re.escape,p['tables']))})\b[\s\S]{{0,900}}?\bWHERE\b[\s\S]{{0,500}}?\bPERNR\b", t):
        score += 22; hits.append("SELECT target table + WHERE PERNR")
    if re.search(r"\bLOOP\s+AT\b[\s\S]{0,250}?\bPA\d{4}\b", t):
        score += 5; hits.append("LOOP AT PAxxxx")
    if re.search(r"\bFIELD\b[\s\S]{0,200}?\bPA\d{4}-[A-Z0-9_]+\b", t):
        if p["intent"] == "field_lookup": score += 4; hits.append("DYNPRO FIELD")
        else: score -= 4; hits.append("DYNPRO PENALTY")
    return score, list(dict.fromkeys(hits))


def field_score(c, p):
    t = c.upper(); s = 0; hits = []
    for f in p["requested_fields"]:
        n = count_term(t, f)
        if n: s += min(n,5)*1.5
        for tab in p["tables"]:
            if re.search(rf"\b{re.escape(tab)}-{re.escape(f)}\b", t):
                s += 10; hits.append(f"{tab}-{f}")
            if re.search(rf"\bSELECT\b[\s\S]{{0,500}}?\b{re.escape(f)}\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(tab)}\b", t):
                s += 14; hits.append(f"SELECT usage {f}")
    return s, list(dict.fromkeys(hits))


def functional_score(c, p):
    t = c.upper(); s = 0; hits = []
    for x in p["functional_terms"]:
        if present(t, x): s += 3; hits.append(f"functional:{x}")
    for x in p["related_terms"]:
        if present(t, x): s += 1; hits.append(f"related:{x}")
    if "KOSTL" in p["functional_terms"] and "PERNR" in p["functional_terms"]:
        if re.search(r"\bKOSTL\b[\s\S]{0,450}?\bPERNR\b|\bPERNR\b[\s\S]{0,450}?\bKOSTL\b", t):
            s += 10; hits.append("KOSTL <-> PERNR proximity")
    if any(x in p["concepts"] for x in ("dati organizzativi", "dati organizzativi dipendente")):
        org = ["BUKRS","WERKS","BTRTL","PERSG","PERSK","KOSTL","ORGEH","PLANS","STELL"]
        got = [x for x in org if present(t,x)]
        s += min(len(got),8)*3
        if got: hits.append("organizational fields: "+",".join(got))
        if present(t,"PA0001") and present(t,"PERNR") and len(got)>=3:
            s += 10; hits.append("PA0001 + PERNR + organizational fields")
    return s, list(dict.fromkeys(hits))


def context_score(c, p):
    t = c.upper(); important = list(dict.fromkeys(p["tables"] + p["requested_fields"]))
    if len(important) < 2: return 0, []
    s = 0; hits = []
    for start in range(0, max(len(t),1), 400):
        w = t[start:start+800]; got = [x for x in important if present(w,x)]
        if len(got) >= 2: s += min(len(got)*2,12)
        if "PA0001" in got and "KOSTL" in got and "PERNR" in got:
            s += 8; hits.append("LOCAL PA0001 + KOSTL + PERNR")
    return min(s,30), list(dict.fromkeys(hits))


def source_family(source_file):
    sf = str(source_file or "").lower()
    stem = re.sub(r"\.(txt|html?)$", "", sf)
    # Handles suffix chains and common copy/backup naming conventions.
    stem = re.sub(r"(?:[_-](?:bk|backup|copia|copy|old|new|p|sp\d+|v\d+|\d+))+$", "", stem)
    stem = re.sub(r"(?:[_-](?:bk|backup|copia|copy|old|new|p|sp\d+|v\d+|\d+))(?=\D|$)", "", stem)
    return stem


def relation_pass(x, p):
    typ = x.get("relationship_type", "NONE")
    rel = x.get("relationship_score", 0)
    if p["intent"] in {"code_lookup", "code_pattern"} and "PA0001" in p["tables"]:
        return typ in {"SELECT_KOSTL_FROM_PA0001_WHERE_PERNR", "SELECT_FIELDS_WHERE", "SELECT_FIELD_WHERE", "PA0001_ORG_FIELDS_PERNR"} and rel >= 110
    if p["intent"] == "functional_lookup" and "KOSTL" in p["functional_terms"]:
        return typ == "SELECT_KOSTL_FROM_PA0001_WHERE_PERNR" or (typ in {"SELECT_FIELD_FROM","SELECT_FIELDS_FROM","SELECT_FIELD_WHERE","SELECT_FIELDS_WHERE"} and "KOSTL" in x.get("relationship_reason", "").upper() and rel >= 80)
    if p["intent"] == "functional_lookup" and any(z in p["concepts"] for z in ("dati organizzativi", "dati organizzativi dipendente")):
        return typ == "PA0001_ORG_FIELDS_PERNR" and rel >= 130
    if p["intent"] == "field_lookup":
        return typ in {"SELECT_FIELD_FROM","SELECT_FIELD_WHERE","FIELD_FROM"} and rel >= 35
    if p["intent"] == "table_reference":
        return typ in {"TABLE_FROM","SELECT_FIELD_FROM","SELECT_FIELDS_FROM","SELECT_FIELDS_WHERE"}
    return rel > 0

def rank(items, p):
    for x in items:
        d = x["document"]
        c = d.page_content if d is not None else x["content"]
        m = d.metadata if d is not None else x["metadata"]
        x["structural_score"], x["structural_matches"] = structural(c,p)
        x["field_score"], x["field_matches"] = field_score(c,p)
        x["functional_score"], x["functional_matches"] = functional_score(c,p)
        x["context_score"], x["context_matches"] = context_score(c,p)
        rel, rtype, reason, relations = relationship_score(c,p)
        x["relationship_score"] = rel
        x["relationship_type"] = rtype
        x["relationship_reason"] = reason
        x["sql_relations"] = relations
        x["relationship_hit"] = relation_pass(x,p)
        x["relationship_penalty"], x["relationship_penalty_matches"] = relationship_noise_penalty(c,p)
        x["relationship_penalty"] += relationship_penalty_for_intent(x,p)

        se=x["semantic_score"]; ex=x["exact_score"]; st=x["structural_score"]
        fs=x["field_score"]; fu=x["functional_score"]; cs=x["context_score"]
        rel=x["relationship_score"]; pen=x["relationship_penalty"]

        # V5.6: relation score dominates for technical/code queries.
        if p["intent"] == "functional_lookup":
            score = se*18 + ex*1.2 + st*.22 + fs*.45 + fu*.45 + cs*.18 + rel*1.75 - pen
        elif p["intent"] == "field_lookup":
            score = se*8 + ex*1.5 + st*.25 + fs*.7 + cs*.15 + rel*1.45 - pen*.7
        elif p["intent"] == "code_pattern":
            score = se*6 + ex*1.5 + st*.35 + fs*.25 + cs*.12 + rel*2.10 - pen*1.15
        elif p["intent"] == "code_lookup":
            score = se*6 + ex*1.5 + st*.30 + fs*.25 + cs*.12 + rel*2.10 - pen*1.15
        elif p["intent"] == "table_reference":
            score = se*10 + ex*3 + st*.65 + fs*.25 + cs*.15 + rel*.75 - pen*.5
        else:
            score = se*25 + ex*2 + st*.3 + fs*.7 + fu*.5 + rel*1.0 - pen*.5

        if p["query"].upper() in [z.upper() for z in x["exact_terms"]]:
            score += 10
        if str(m.get("language","")).upper() == "ABAP":
            score += 6
        sf = str(m.get("source_file","")).lower()
        if sf.endswith(".txt"): score += 2
        elif sf.endswith(".html"): score -= 5
        x["source_family"] = source_family(sf)
        x["final_score"] = score

    ordered = sorted(items, key=lambda x: (x["final_score"], x.get("relationship_hit",False)), reverse=True)
    selected = []
    family_counts = {}
    for x in ordered:
        fam = x.get("source_family", "")
        if family_counts.get(fam,0) >= 2:
            continue
        selected.append(x)
        family_counts[fam] = family_counts.get(fam,0) + 1
        if len(selected) >= FINAL_K:
            break
    return selected


def evaluate(res, case):
    _,_,_,expected,tables = case
    es=set(x.upper() for x in expected); hits=set(); th=ab=st=sel=0
    for x in res:
        d=x["document"]; m=d.metadata; c=d.page_content; u=c.upper()
        hits.update(z.upper() for z in x["exact_terms"])
        if any(present(u,t) for t in tables): th += 1
        if str(m.get("language","")).upper() == "ABAP": ab += 1
        if x["structural_matches"]: st += 1
        if re.search(r"\bSELECT\b|\bFROM\b",u): sel += 1
    cov = len(hits & es) / len(es) if es else 0
    relation_hits = sum(1 for x in res if x.get("relationship_hit"))
    top_relation = res[0].get("relationship_score",0) if res else 0
    top_type = res[0].get("relationship_type","NONE") if res else "NONE"
    top_pass = bool(res and res[0].get("relationship_hit"))
    return cov,th,ab,st,sel,relation_hits,top_relation,top_type,top_pass


def main():
    print("="*70)
    print("SAP ABAP RAG TEST - V5.6")
    print("SQL RELATIONSHIP-AWARE RANKING")
    print("="*70)
    print(f"VectorDB : {DB_DIR}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Embedding : {EMBEDDING_MODEL}")
    print(f"Test cases : {len(TEST_CASES)}")
    print("VectorDB mode : READ-ONLY")

    db = create_db()
    cache, cache_time = load_cache(db)
    rows=[]; semtot=0

    for i, case in enumerate(TEST_CASES,1):
        tid,q,expected_intent,expected,tables=case
        p=profile(q)
        print("\n"+"="*70)
        print(f"TEST {i}/{len(TEST_CASES)} - {tid}")
        print(f"Query: {q}")
        print(f"Intent: {p['intent']}")
        print(f"Tabelle: {', '.join(p['tables']) or '-'}")
        print(f"Campi: {', '.join(p['explicit_fields']) or '-'}")
        print(f"Concetti: {', '.join(p['concepts']) or '-'}")
        print(f"Campi richiesti: {', '.join(p['requested_fields']) or '-'}")

        a,dt=semantic(db,q); semtot += dt
        b=exact(cache,p)
        merged,dup=merge(a,b)
        res=rank(merged,p)
        cov,th,ab,st,sel,rhits,toprel,toptype,toppass=evaluate(res,case)

        for n,x in enumerate(res[:5],1):
            d=x["document"];m=d.metadata
            print(
                f"  #{n} score={x['final_score']:.2f} sem={x['semantic_score']:.3f} "
                f"exact={x['exact_score']:.1f} struct={x['structural_score']:.1f} "
                f"field={x['field_score']:.1f} func={x['functional_score']:.1f} "
                f"ctx={x['context_score']:.1f} rel={x['relationship_score']:.1f} "
                f"pen={x['relationship_penalty']:.1f} hit={'YES' if x['relationship_hit'] else 'NO'} "
                f"type={x['relationship_type']} | {m.get('source_file','-')} | chunk={m.get('chunk_index','-')}"
            )
            if n == 1:
                print(f"       reason: {x.get('relationship_reason','-')}")
                if x.get('relationship_penalty_matches'):
                    print(f"       penalty: {', '.join(x['relationship_penalty_matches'])}")

        print(
            f"Coverage: {cov:.2%} | Table hit: {th}/15 | ABAP: {ab}/15 | "
            f"Structural: {st}/15 | SELECT/FROM: {sel}/15 | "
            f"Relation hit: {rhits}/15 | Top relation: {toprel:.1f} | "
            f"Top type: {toptype} | Top relation PASS: {'YES' if toppass else 'NO'} | "
            f"duplicates: {dup} | semantic: {dt:.2f}s"
        )

        rows.append({
            "id":tid,"query":q,"intent":p["intent"],"coverage":cov,
            "table_hit":th,"abap":ab,"structural":st,"select":sel,
            "relation_hit_count":rhits,"top_relation":toprel,
            "top_relationship_type":toptype,"top_relation_pass":toppass,
            "duplicates":dup,"semantic_seconds":dt
        })

    avg=sum(x["coverage"] for x in rows)/len(rows)
    relation_passes=sum(1 for x in rows if x["top_relation_pass"])
    # Promising now requires both useful coverage and a technically meaningful top relation.
    promising=sum(1 for x in rows if x["coverage"]>=.5 and x["abap"]>0 and x["top_relation_pass"])

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    with (OUTPUT_DIR/"rag_v5_6_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

    md=[
        "# SAP ABAP RAG V5.6 - Report","",
        f"Data: {datetime.now().isoformat(timespec='seconds')}","",
        "VectorDB: READ-ONLY","",
        f"Test eseguiti: {len(rows)}",
        f"Promettenti: {promising}/{len(rows)}",
        f"Top relation PASS: {relation_passes}/{len(rows)}",
        f"Coverage media: {avg:.2%}",
        f"Exact cache: {cache_time:.2f}s",
        f"Semantic totale: {semtot:.2f}s","",
        "| ID | Query | Intent | Coverage | Table | ABAP | Structural | SELECT | Relation hit | Top Rel | Relation Type | PASS | Duplicati |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|:---:|---:|"
    ]
    md += [
        f"| {x['id']} | {x['query']} | {x['intent']} | {x['coverage']:.2%} | {x['table_hit']} | {x['abap']} | {x['structural']} | {x['select']} | {x['relation_hit_count']} | {x['top_relation']:.1f} | {x['top_relationship_type']} | {'YES' if x['top_relation_pass'] else 'NO'} | {x['duplicates']} |"
        for x in rows
    ]
    (OUTPUT_DIR/"rag_v5_6_report.md").write_text("\n".join(md),encoding="utf-8")

    print("\n"+"="*70)
    print("SUMMARY V5.6")
    print("="*70)
    print(f"Test eseguiti: {len(rows)}")
    print(f"Promettenti: {promising}/{len(rows)}")
    print(f"Top relation PASS: {relation_passes}/{len(rows)}")
    print(f"Coverage media: {avg:.2%}")
    print(f"Exact cache: {cache_time:.2f}s")
    print(f"Semantic totale: {semtot:.2f}s")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
