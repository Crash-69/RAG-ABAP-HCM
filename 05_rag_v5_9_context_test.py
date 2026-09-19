# SAP ABAP RAG - V5.9 ANTI CONTEXT INSPECTION
# Diagnostic tool: does NOT modify V5.9 ANTI, VectorDB, Chroma data or embeddings.
import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(r"C:\Progetto_AI")
RAG_SCRIPT = BASE_DIR / "05_rag_v5_9_anti.py"
DEFAULT_TOP = 5
DEFAULT_MAX_CHARS = 12000


def load_rag():
    if not RAG_SCRIPT.exists():
        raise FileNotFoundError(f"Script V5.9 ANTI non trovato: {RAG_SCRIPT}")
    spec = importlib.util.spec_from_file_location("rag_v5_9_anti", RAG_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossibile caricare: {RAG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rag_v5_9_anti"] = module
    spec.loader.exec_module(module)
    return module


def get_content(item):
    doc = item.get("document")
    return (doc.page_content if doc is not None else item.get("content", "")) or ""


def get_metadata(item):
    doc = item.get("document")
    return (doc.metadata if doc is not None else item.get("metadata", {})) or {}


def print_list(label, values):
    print(f"{label}: {', '.join(map(str, values)) if values else '-'}")


def sql_relation_diagnostics(rag, content, profile):
    relations = rag.extract_sql_relations(content)
    requested = list(dict.fromkeys(profile["requested_fields"] + profile["explicit_fields"] + profile["functional_terms"]))
    print("      SQL RELATIONS:")
    if not relations:
        print("        - nessuna relazione SELECT/FROM riconosciuta")
    else:
        for i, r in enumerate(relations, 1):
            print(f"        [{i}] table={r['table']} | single={r['single']}")
            print(f"            select={', '.join(r['fields_select']) or '-'}")
            print(f"            where={', '.join(r['fields_where']) or '-'}")
            print(f"            statement={r['statement']}")

    abap_signal = bool(re.search(r"\b(SELECT|ENDSELECT|LOOP|READ|CALL FUNCTION|FORM|METHOD|CLASS|FUNCTION|REPORT)\b", content, re.I))
    print("      CONTEXT QUALITY (diagnostico, NON usato per ranking):")
    print(f"        ABAP/code signal              : {'YES' if abap_signal else 'NO'}")
    print(f"        SELECT                        : {'YES' if re.search(r'\bSELECT\b', content, re.I) else 'NO'}")
    print(f"        FROM                          : {'YES' if re.search(r'\bFROM\b', content, re.I) else 'NO'}")
    print(f"        WHERE                         : {'YES' if re.search(r'\bWHERE\b', content, re.I) else 'NO'}")

    present = [f for f in requested if rag.present(content, f)]
    print(f"        requested fields presenti    : {', '.join(present) if present else '-'}")

    pa_pernr = False
    pa_kostl_pernr = False
    best_org = 0
    same_sql_hits = []
    org = {"BUKRS","WERKS","BTRTL","PERSG","PERSK","KOSTL","ORGEH","PLANS","STELL"}

    for r in relations:
        all_fields = set(r["fields_select"]) | set(r["fields_where"])
        target = r["table"] in profile["tables"] if profile["tables"] else r["table"] in {"PA0001","PA0002","PA0007","PA0008"}
        if target:
            same = sorted(set(requested) & all_fields)
            if same:
                same_sql_hits.append(f"{r['table']} -> {', '.join(same)}")
        if r["table"] == "PA0001":
            if "PERNR" in r["fields_where"]:
                pa_pernr = True
            if "KOSTL" in r["fields_select"] and "PERNR" in r["fields_where"]:
                pa_kostl_pernr = True
            best_org = max(best_org, len(org & all_fields))

    print(f"        requested fields same SQL  : {'; '.join(same_sql_hits) if same_sql_hits else '-'}")
    print(f"        PA0001 + PERNR same SQL     : {'YES' if pa_pernr else 'NO'}")
    print(f"        PA0001 KOSTL SELECT + PERNR WHERE: {'YES' if pa_kostl_pernr else 'NO'}")
    print(f"        best PA0001 org field count : {best_org}")


def inspect_one(rag, db, cache, query, top_n, max_chars, case=None):
    p = rag.profile(query)
    print("\n" + "=" * 90)
    print("CONTEXT INSPECTION - V5.9 ANTI")
    print("=" * 90)
    print(f"Query: {query}")
    print(f"Intent: {p['intent']}")
    print_list("Tabelle", p["tables"])
    print_list("Campi espliciti", p["explicit_fields"])
    print_list("Concetti", p["concepts"])
    print_list("Campi richiesti", p["requested_fields"])
    print_list("Retrieval terms", p["terms"])

    semantic, semantic_time = rag.semantic(db, query)
    exact_start = time.perf_counter()
    exact = rag.exact(cache, p)
    exact_time = time.perf_counter() - exact_start
    merged, duplicates = rag.merge(semantic, exact)
    ranked = rag.rank(merged, p)

    print("\n" + "-" * 90)
    print("RETRIEVAL")
    print("-" * 90)
    print(f"Semantic results : {len(semantic)}")
    print(f"Semantic time    : {semantic_time:.3f}s")
    print(f"Exact results    : {len(exact)}")
    print(f"Exact time       : {exact_time:.3f}s")
    print(f"Merged results   : {len(merged)}")
    print(f"Duplicates       : {duplicates}")
    print(f"Final ranked     : {len(ranked)}")
    print(f"TOP-K inspected  : {min(top_n, len(ranked))}")

    if case is not None:
        tid, _, expected_intent, expected, tables = case
        print(f"Test case        : {tid}")
        print(f"Expected intent  : {expected_intent}")
        print_list("Expected terms", expected)
        print_list("Expected tables", tables)

    for n, x in enumerate(ranked[:top_n], 1):
        m = get_metadata(x)
        content = get_content(x)
        print("\n" + "=" * 90)
        print(f"TOP {n}")
        print("=" * 90)
        print(f"final={x.get('final_score',0):.2f} | sem={x.get('semantic_score',0):.3f} | exact={x.get('exact_score',0):.1f} | struct={x.get('structural_score',0):.1f} | field={x.get('field_score',0):.1f} | func={x.get('functional_score',0):.1f} | ctx={x.get('context_score',0):.1f} | rel={x.get('relationship_score',0):.1f} | pen={x.get('relationship_penalty',0):.1f}")
        print(f"relation_type={x.get('relationship_type','NONE')} | relation_hit={'YES' if x.get('relationship_hit') else 'NO'}")
        print("\nMETADATA:")
        for key in ("source_file","chunk_index","chunk_hash","language","object_type","name","file_type"):
            print(f"  {key:12}: {m.get(key,'-')}")
        print(f"  {'source_family':12}: {x.get('source_family','-')}")
        print("\nRELATIONSHIP:")
        print(f"  type          : {x.get('relationship_type','NONE')}")
        print(f"  score         : {x.get('relationship_score',0):.1f}")
        print(f"  hit           : {'YES' if x.get('relationship_hit') else 'NO'}")
        print(f"  reason        : {x.get('relationship_reason') or '-'}")
        penalties = x.get("relationship_penalty_matches") or []
        print(f"  penalty match : {', '.join(penalties) if penalties else '-'}")
        print("\nMATCHES:")
        for key in ("exact_terms","structural_matches","field_matches","functional_matches","context_matches"):
            vals = x.get(key) or []
            print(f"  {key:20}: {', '.join(map(str,vals)) if vals else '-'}")
        sql_relation_diagnostics(rag, content, p)
        print("\nCHUNK CONTENT:")
        print("-" * 90)
        shown = content if max_chars == 0 else content[:max_chars]
        print(shown)
        if max_chars and len(content) > max_chars:
            print(f"\n[TRUNCATED: mostrati {len(shown)} di {len(content)} caratteri]")


def main():
    parser = argparse.ArgumentParser(description="Context Inspection per SAP ABAP RAG V5.9 ANTI")
    parser.add_argument("--query", help="Query singola da ispezionare")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help=f"Numero TOP-K da mostrare (default: {DEFAULT_TOP})")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help=f"Massimo caratteri per chunk; 0 = tutto (default: {DEFAULT_MAX_CHARS})")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top deve essere >= 1")
    if args.max_chars < 0:
        parser.error("--max-chars deve essere >= 0")

    rag = load_rag()
    print("=" * 90)
    print("SAP ABAP RAG - V5.9 ANTI CONTEXT INSPECTION")
    print("=" * 90)
    print(f"V5.9 script      : {RAG_SCRIPT}")
    print(f"VectorDB         : {rag.DB_DIR}")
    print(f"Collection       : {rag.COLLECTION_NAME}")
    print(f"Embedding        : {rag.EMBEDDING_MODEL}")
    print("VectorDB mode    : READ-ONLY")
    print("Ranking source   : 05_rag_v5_9_anti.py")
    print("Context score    : diagnostico, NON modificato\n")

    db = rag.create_db()
    cache, cache_time = rag.load_cache(db)
    print(f"Cache load time   : {cache_time:.3f}s")
    print(f"Cache chunks      : {len(cache)}")

    if args.query:
        inspect_one(rag, db, cache, args.query, args.top, args.max_chars)
    else:
        print("\nNessuna --query: esecuzione dei 12 smoke test V5.9 ANTI.")
        for case in rag.TEST_CASES:
            tid, query, *_ = case
            print(f"\n>>> {tid}: {query}")
            inspect_one(rag, db, cache, query, args.top, args.max_chars, case=case)

    print("\n" + "=" * 90)
    print("INTEGRITY CHECK")
    print("=" * 90)
    print("VectorDB modificato : NO")
    print("Nuovi embedding     : NO")
    print("Ranking modificato  : NO")
    print("=" * 90)


if __name__ == "__main__":
    main()
