# SAP ABAP RAG - V5.9 CONTEXT BUILDER V2
# Candidate pool = V3 semantic + exact + merge + V3 ranking
# Context selection = coverage-aware, relation-aware, source-family diverse
# VectorDB is READ-ONLY. No Chroma data or embeddings are modified.

import json
import re
import sys
import importlib.util
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"C:\Progetto_AI")
V3_CANDIDATE = BASE_DIR / "05_rag_v5_9_anti_v3.py"
V3_FALLBACKS = [
    BASE_DIR / "05_rag_v5_9_anti_v2.py",
    BASE_DIR / "05_rag_v5_9.py",
]
OUTPUT_DIR = BASE_DIR / "RAG_V5_9_RESULTS"
CONTEXT_K = 8

TEST_CASES = [
    ("T09", "centro di costo del dipendente"),
    ("T10", "leggere PA0001 per PERNR"),
    ("T11", "SELECT PA0001 WHERE PERNR"),
    ("T12", "dati organizzativi dipendente SAP HCM"),
]

ORG_FIELDS = ["WERKS", "PERSG", "PERSK", "BTRTL", "ORGEH", "PLANS", "STELL", "BUKRS", "KOSTL"]


def load_v3():
    candidates = [V3_CANDIDATE] + V3_FALLBACKS
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("rag_v3_runtime", str(path))
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        required = ["create_db", "load_cache", "profile", "semantic", "exact", "merge", "rank"]
        missing = [x for x in required if not hasattr(mod, x)]
        if not missing:
            print(f"V3 engine: {path}")
            return mod
        print(f"Ignorato {path.name}: mancano funzioni {missing}")
    raise FileNotFoundError(
        "Non trovo una versione V3 compatibile. Atteso: "
        f"{V3_CANDIDATE}"
    )


def source_family(name):
    s = str(name or "").lower()
    stem = re.sub(r"\.(txt|html?)$", "", s)
    stem = re.sub(r"(?:[_-](?:bk|backup|copia|copy|old|new|p|sp\d+|v\d+|\d+))+$", "", stem)
    return stem


def content(x):
    d = x.get("document")
    return d.page_content if d is not None else x.get("content", "")


def metadata(x):
    d = x.get("document")
    return d.metadata if d is not None else x.get("metadata", {})


def norm_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def present(text, term):
    return bool(re.search(rf"(?<![A-Z0-9_]){re.escape(term.upper())}(?![A-Z0-9_])", text.upper()))


def infer_requested(p):
    requested = list(dict.fromkeys(p.get("requested_fields", []) + p.get("tables", [])))
    concepts = set(p.get("concepts", []))
    q = p.get("query", "").lower()
    if "centro di costo" in q or "centro costo" in q:
        requested = list(dict.fromkeys(requested + ["KOSTL", "PERNR", "PA0001"]))
    if "dati organizzativi" in q:
        requested = list(dict.fromkeys(requested + ["PA0001", "PERNR"] + ORG_FIELDS))
    # Preserve V3's profile as the source of truth; the additions above only
    # make the context contract explicit for natural-language functional tests.
    return requested


def relation_signature(x):
    rels = x.get("sql_relations", []) or []
    sig = []
    for r in rels:
        sig.append((
            r.get("table"),
            tuple(sorted(set(r.get("fields_select", [])))),
            tuple(sorted(set(r.get("fields_where", [])))),
            bool(r.get("single")),
        ))
    return tuple(sig)


def relation_types(x):
    out = []
    typ = x.get("relationship_type", "NONE")
    if typ and typ != "NONE":
        out.append(typ)
    return out


def candidate_features(x, requested):
    c = content(x)
    u = c.upper()
    m = metadata(x)
    covered = {t for t in requested if present(u, t)}
    rels = x.get("sql_relations", []) or []
    reltypes = set(relation_types(x))
    if not reltypes and x.get("relationship_type"):
        reltypes.add(x.get("relationship_type"))
    families = {source_family(m.get("source_file", ""))}

    # Extra coverage from fields actually extracted by the V3 SQL parser.
    sql_fields = set()
    for r in rels:
        sql_fields |= set(r.get("fields_select", []))
        sql_fields |= set(r.get("fields_where", []))
    covered_sql = {t for t in requested if t in sql_fields}

    return covered, covered_sql, reltypes, families


def v3_retrieve(v3, db, cache, query):
    p = v3.profile(query)
    sem, sem_time = v3.semantic(db, query)
    exact = v3.exact(cache, p)
    merged, duplicates = v3.merge(sem, exact)
    ranked = v3.rank(merged, p)
    return p, ranked, duplicates, sem_time, len(sem), len(exact), len(merged)


def select_context(candidates, p, context_k=CONTEXT_K):
    requested = infer_requested(p)
    selected = []
    covered = set()
    relations = set()
    families = set()

    # Preserve V3's top technical evidence. This prevents the T09 failure seen
    # in the previous builder, where a fresh semantic retrieval discarded the
    # SELECT_KOSTL_FROM_PA0001_WHERE_PERNR result found by V3.
    if candidates:
        top = candidates[0]
        top_cov, top_sql, top_rel, top_fam = candidate_features(top, requested)
        selected.append(top)
        covered |= top_cov
        relations |= top_rel
        families |= top_fam

    remaining = [x for x in candidates if x not in selected]

    while remaining and len(selected) < context_k:
        best = None
        best_key = None
        for x in remaining:
            cov, sql_cov, rel, fam = candidate_features(x, requested)
            new_cov = cov - covered
            new_sql_cov = sql_cov - covered
            new_rel = rel - relations
            new_fam = fam - families

            # Marginal utility: coverage dominates. SQL relation and diversity
            # are secondary, while repeated source families are discouraged.
            gain = 45.0 * len(new_cov)
            gain += 12.0 * len(new_sql_cov)
            gain += 30.0 * len(new_rel)
            gain += 14.0 * len(new_fam)
            gain += 5.0 if x.get("exact_terms") else 0.0
            gain += min(float(x.get("relationship_score", 0.0)), 130.0) * 0.08
            gain += min(float(x.get("final_score", 0.0)), 300.0) * 0.02

            if not new_cov and not new_rel and not new_fam:
                gain -= 28.0
            if source_family(metadata(x).get("source_file", "")) in families:
                gain -= 18.0

            # Query-aware guards retained from V3.
            typ = x.get("relationship_type", "NONE")
            if p.get("intent") in {"code_lookup", "code_pattern"} and "PA0001" in p.get("tables", []):
                if "KOSTL" not in p.get("requested_fields", []) and typ == "SELECT_KOSTL_FROM_PA0001_WHERE_PERNR":
                    gain -= 25.0
                if typ == "SELECT_PA0001_WHERE_PERNR":
                    gain += 20.0

            key = (
                gain,
                len(new_cov),
                len(new_rel),
                len(new_fam),
                float(x.get("relationship_score", 0.0)),
                float(x.get("final_score", 0.0)),
            )
            if best is None or key > best_key:
                best, best_key = x, key

        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        cov, sql_cov, rel, fam = candidate_features(best, requested)
        covered |= cov
        relations |= rel
        families |= fam

    missing = [t for t in requested if t not in covered]
    return selected, requested, sorted(covered), missing, sorted(relations), len(families)


def print_context(tid, query, p, candidates, selected, requested, covered, missing, relations, family_count):
    print("\n" + "=" * 78)
    print(f"CONTEXT BUILDER V2 - {tid}")
    print(f"Query: {query}")
    print(f"Intent: {p.get('intent')}")
    print(f"V3 candidate pool: {len(candidates)}")
    print(f"Context TOP-K: {len(selected)}")
    print(f"Coverage: {len(covered)}/{len(requested)} = {len(covered)/len(requested):.2%}")
    print(f"Missing: {', '.join(missing) if missing else '-'}")
    print(f"Relations: {', '.join(relations) if relations else '-'}")
    print(f"Source families: {family_count}")

    for i, x in enumerate(selected, 1):
        m = metadata(x)
        print("-" * 78)
        print(
            f"#{i} V3={x.get('final_score', 0):.2f} rel={x.get('relationship_score', 0):.1f} "
            f"hit={'YES' if x.get('relationship_hit') else 'NO'} "
            f"type={x.get('relationship_type', 'NONE')} | "
            f"{m.get('source_file', '-')} | chunk={m.get('chunk_index', '-') }"
        )
        print(f"   family: {source_family(m.get('source_file', ''))}")
        print(f"   exact: {', '.join(x.get('exact_terms', [])) or '-'}")
        text = norm_text(content(x))
        print(f"   context: {text[:900]}")


def main():
    v3 = load_v3()
    db = v3.create_db()
    cache, cache_time = v3.load_cache(db)

    print("=" * 78)
    print("SAP ABAP RAG - V5.9 CONTEXT BUILDER V2")
    print("V3 CANDIDATE POOL + COVERAGE-AWARE CONTEXT SELECTION")
    print("VectorDB mode: READ-ONLY")
    print(f"VectorDB: {getattr(v3, 'DB_DIR', 'configured by V3')}")
    print(f"Collection: {getattr(v3, 'COLLECTION_NAME', 'configured by V3')}")
    print(f"Exact cache load: {cache_time:.2f}s")

    report = {
        "version": "V5.9 Context Builder V2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "vector_db_read_only": True,
        "cases": [],
    }

    for tid, query in TEST_CASES:
        p, candidates, duplicates, sem_time, sem_n, exact_n, merged_n = v3_retrieve(v3, db, cache, query)
        selected, requested, covered, missing, relations, family_count = select_context(candidates, p)
        print_context(tid, query, p, candidates, selected, requested, covered, missing, relations, family_count)

        report["cases"].append({
            "id": tid,
            "query": query,
            "intent": p.get("intent"),
            "requested_terms": requested,
            "covered_terms": covered,
            "missing_terms": missing,
            "coverage": round(len(covered) / len(requested), 6) if requested else 0.0,
            "relations": relations,
            "source_family_count": family_count,
            "v3_candidates": len(candidates),
            "semantic_candidates": sem_n,
            "exact_candidates": exact_n,
            "merged_candidates": merged_n,
            "duplicates_removed": duplicates,
            "semantic_seconds": round(sem_time, 4),
            "context_k": len(selected),
            "selected": [
                {
                    "rank": i,
                    "source_file": metadata(x).get("source_file", ""),
                    "chunk_index": metadata(x).get("chunk_index", ""),
                    "source_family": source_family(metadata(x).get("source_file", "")),
                    "v3_score": x.get("final_score", 0.0),
                    "relationship_score": x.get("relationship_score", 0.0),
                    "relationship_type": x.get("relationship_type", "NONE"),
                    "relationship_hit": bool(x.get("relationship_hit")),
                    "exact_terms": x.get("exact_terms", []),
                    "content": content(x),
                }
                for i, x in enumerate(selected, 1)
            ],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "context_v2_results.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("SUMMARY CONTEXT BUILDER V2")
    print("=" * 78)
    for c in report["cases"]:
        print(
            f"{c['id']}: coverage={c['coverage']:.2%} "
            f"missing={len(c['missing_terms'])} families={c['source_family_count']} "
            f"context={c['context_k']}"
        )
    print(f"Report JSON: {json_path}")
    print("Nessuna scrittura sul VectorDB.")


if __name__ == "__main__":
    main()
