# -*- coding: utf-8 -*-
"""
SAP ABAP RAG - V5.9 CONTEXT BUILDER V3.1

V3.1 Context Builder only.
- VectorDB READ-ONLY
- Uses the V3 retrieval/ranking candidate pool unchanged
- Preserves V3 rank #1
- Coverage first
- Stronger relationship_hit preference
- Stronger identical SQL/relation redundancy penalty
- Text redundancy penalty
- Source-family diversity
- Structural / SQL-shape diversity
- Classifies selected chunks as PRIMARY / COMPLEMENTARY / REDUNDANT / FALLBACK
- Does NOT modify Chroma, embeddings or V3 retrieval
"""

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

# T09/T10/T11: compact technical context.
# T12: broader functional/organizational context.
DEFAULT_CONTEXT_K = 3
T12_CONTEXT_K = 5

TEST_CASES = [
    ("T09", "centro di costo del dipendente"),
    ("T10", "leggere PA0001 per PERNR"),
    ("T11", "SELECT PA0001 WHERE PERNR"),
    ("T12", "dati organizzativi dipendente SAP HCM"),
]

ORG_FIELDS = [
    "WERKS", "PERSG", "PERSK", "BTRTL",
    "ORGEH", "PLANS", "STELL", "BUKRS", "KOSTL"
]


def load_v3():
    for path in [V3_CANDIDATE] + V3_FALLBACKS:
        if not path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            "rag_v3_runtime", str(path)
        )
        if spec is None or spec.loader is None:
            continue

        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        required = [
            "create_db", "load_cache", "profile",
            "semantic", "exact", "merge", "rank"
        ]
        missing = [x for x in required if not hasattr(mod, x)]

        if not missing:
            print(f"V3 engine: {path}")
            return mod

        print(f"Ignorato {path.name}: mancano funzioni {missing}")

    raise FileNotFoundError(
        "Non trovo una versione V3 compatibile. "
        f"Atteso: {V3_CANDIDATE}"
    )


def source_family(name):
    stem = re.sub(
        r"\.(txt|html?)$",
        "",
        str(name or "").lower()
    )
    return re.sub(
        r"(?:[_-](?:bk|backup|copia|copy|old|new|p|sp\d+|v\d+|\d+))+$",
        "",
        stem
    )


def content(x):
    d = x.get("document")
    return d.page_content if d is not None else x.get("content", "")


def metadata(x):
    d = x.get("document")
    return d.metadata if d is not None else x.get("metadata", {})


def norm_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def present(text, term):
    return bool(
        re.search(
            rf"(?<![A-Z0-9_]){re.escape(str(term).upper())}(?![A-Z0-9_])",
            text.upper()
        )
    )


def text_tokens(x):
    return set(
        re.findall(
            r"[A-Z][A-Z0-9_]{1,30}",
            content(x).upper()
        )
    )


def text_overlap(a, b):
    ta = text_tokens(a)
    tb = text_tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def infer_requested(p):
    requested = list(dict.fromkeys(
        p.get("requested_fields", []) + p.get("tables", [])
    ))

    q = p.get("query", "").lower()

    if "centro di costo" in q or "centro costo" in q:
        requested = list(dict.fromkeys(
            requested + ["KOSTL", "PERNR", "PA0001"]
        ))

    if "dati organizzativi" in q:
        requested = list(dict.fromkeys(
            requested + ["PA0001", "PERNR"] + ORG_FIELDS
        ))

    return requested


def relation_types(x):
    typ = x.get("relationship_type", "NONE")
    return {typ} if typ and typ != "NONE" else set()


def relation_signature(x):
    sig = []

    for r in x.get("sql_relations", []) or []:
        sig.append((
            str(r.get("table", "")).upper(),
            tuple(sorted({
                str(v).upper()
                for v in r.get("fields_select", [])
            })),
            tuple(sorted({
                str(v).upper()
                for v in r.get("fields_where", [])
            })),
            bool(r.get("single")),
        ))

    return tuple(sorted(sig))


def sql_shape(x):
    tables = set()
    selected = set()
    where = set()
    single = False

    for r in x.get("sql_relations", []) or []:
        if r.get("table"):
            tables.add(str(r["table"]).upper())

        selected |= {
            str(v).upper()
            for v in r.get("fields_select", [])
        }

        where |= {
            str(v).upper()
            for v in r.get("fields_where", [])
        }

        single = single or bool(r.get("single"))

    return (
        tuple(sorted(tables)),
        tuple(sorted(selected)),
        tuple(sorted(where)),
        single,
    )


def features(x, requested):
    u = content(x).upper()
    rels = x.get("sql_relations", []) or []

    sql_fields = set()
    for r in rels:
        sql_fields |= {
            str(v).upper()
            for v in r.get("fields_select", [])
        }
        sql_fields |= {
            str(v).upper()
            for v in r.get("fields_where", [])
        }

    return {
        "covered": {
            t for t in requested if present(u, t)
        },
        "covered_sql": {
            t for t in requested if str(t).upper() in sql_fields
        },
        "relations": relation_types(x),
        "family": source_family(
            metadata(x).get("source_file", "")
        ),
        "relation_signature": relation_signature(x),
        "sql_shape": sql_shape(x),
    }


def v3_retrieve(v3, db, cache, query):
    p = v3.profile(query)
    sem, sem_time = v3.semantic(db, query)
    exact = v3.exact(cache, p)
    merged, duplicates = v3.merge(sem, exact)
    ranked = v3.rank(merged, p)

    return (
        p,
        ranked,
        duplicates,
        sem_time,
        len(sem),
        len(exact),
        len(merged),
    )


def context_limit(tid):
    return T12_CONTEXT_K if tid == "T12" else DEFAULT_CONTEXT_K


def gain(
    x,
    requested,
    covered,
    relations,
    families,
    sigs,
    shapes,
    selected,
    p,
):
    f = features(x, requested)

    new_cov = f["covered"] - covered
    new_sql = f["covered_sql"] - covered
    new_rel = f["relations"] - relations

    score = 100 * len(new_cov)
    score += 22 * len(new_sql)
    score += 35 * len(new_rel)

    # Source-family diversity.
    score += 12 if f["family"] not in families else -14

    # Stronger SQL/relation diversity than V3.
    if f["relation_signature"]:
        if f["relation_signature"] not in sigs:
            score += 24
        else:
            score -= 42

    if f["sql_shape"]:
        if f["sql_shape"] not in shapes:
            score += 18
        else:
            score -= 26

    # Exact SAP evidence remains useful.
    score += 5 if x.get("exact_terms") else 0

    # Preserve useful relationship strength, but don't let it dominate
    # coverage/diversity.
    score += min(
        float(x.get("relationship_score", 0)),
        320
    ) * 0.06

    score += min(
        float(x.get("final_score", 0)),
        1200
    ) * 0.015

    # Stronger preference for a parser-confirmed relationship.
    if x.get("relationship_hit"):
        score += 18
    else:
        score -= 20

    # Text-level redundancy.
    max_ov = max(
        (text_overlap(x, y) for y in selected),
        default=0.0
    )

    if max_ov >= 0.90:
        score -= 120
    elif max_ov >= 0.80:
        score -= 75
    elif max_ov >= 0.70:
        score -= 45
    elif max_ov >= 0.60:
        score -= 20

    # Preserve V3's query-specific PA0001/PERNR protection.
    if (
        p.get("intent") in {"code_lookup", "code_pattern"}
        and "PA0001" in p.get("tables", [])
    ):
        requested_fields = {
            str(v).upper()
            for v in p.get("requested_fields", [])
        }

        typ = x.get("relationship_type", "NONE")

        if (
            "KOSTL" not in requested_fields
            and typ == "SELECT_KOSTL_FROM_PA0001_WHERE_PERNR"
        ):
            score -= 45

        if typ == "SELECT_PA0001_WHERE_PERNR":
            score += 25

    # Candidate that adds no new information and repeats a family:
    # strongly discourage it.
    if (
        not new_cov
        and not new_sql
        and not new_rel
        and f["family"] in families
    ):
        score -= 50

    return (
        score,
        len(new_cov),
        len(new_sql),
        len(new_rel),
        -max_ov,
        float(x.get("relationship_score", 0)),
        float(x.get("final_score", 0)),
    )


def classify_selected(
    idx,
    x,
    requested,
    selected,
    covered_before,
):
    f = features(x, requested)

    if idx == 1:
        return "PRIMARY"

    new_cov = f["covered"] - covered_before

    previous = selected[:-1]

    same_signature = any(
        f["relation_signature"]
        and f["relation_signature"] == features(y, requested)["relation_signature"]
        for y in previous
    )

    max_ov = max(
        (text_overlap(x, y) for y in previous),
        default=0.0
    )

    if not new_cov and (same_signature or max_ov >= 0.80):
        return "REDUNDANT"

    if (
        not x.get("relationship_hit")
        and not new_cov
        and max_ov >= 0.60
    ):
        return "REDUNDANT"

    if not x.get("relationship_hit") and not new_cov:
        return "FALLBACK"

    return "COMPLEMENTARY"


def select_context(tid, candidates, p):
    requested = infer_requested(p)

    if not candidates:
        return [], requested, [], requested, [], 0, []

    limit = context_limit(tid)

    # NON-NEGOTIABLE: V3 rank #1 is always preserved.
    selected = [candidates[0]]

    f = features(selected[0], requested)

    covered = set(f["covered"])
    relations = set(f["relations"])
    families = {f["family"]}

    sigs = (
        {f["relation_signature"]}
        if f["relation_signature"]
        else set()
    )

    shapes = (
        {f["sql_shape"]}
        if f["sql_shape"]
        else set()
    )

    remaining = list(candidates[1:])

    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda x: gain(
                x,
                requested,
                covered,
                relations,
                families,
                sigs,
                shapes,
                selected,
                p,
            )
        )

        selected.append(best)
        remaining.remove(best)

        f = features(best, requested)

        covered |= f["covered"]
        relations |= f["relations"]
        families.add(f["family"])

        if f["relation_signature"]:
            sigs.add(f["relation_signature"])

        if f["sql_shape"]:
            shapes.add(f["sql_shape"])

        # For compact technical queries, stop once coverage is complete
        # and at least two useful examples exist.
        if (
            len(covered) == len(requested)
            and len(selected) >= 2
            and tid != "T12"
        ):
            break

    # Safety net: never lose requested coverage if another candidate
    # in the existing V3 pool can supply the missing term.
    for term in [t for t in requested if t not in covered]:
        if len(selected) >= limit:
            break

        eligible = [
            x for x in remaining
            if term in features(x, requested)["covered"]
        ]

        if not eligible:
            continue

        best = max(
            eligible,
            key=lambda x: (
                term in features(x, requested)["covered_sql"],
                len(features(x, requested)["covered"]),
                bool(x.get("relationship_hit")),
                float(x.get("relationship_score", 0)),
                float(x.get("final_score", 0)),
            )
        )

        selected.append(best)
        remaining.remove(best)

        f = features(best, requested)

        covered |= f["covered"]
        relations |= f["relations"]
        families.add(f["family"])

        if f["relation_signature"]:
            sigs.add(f["relation_signature"])

        if f["sql_shape"]:
            shapes.add(f["sql_shape"])

    missing = [
        t for t in requested
        if t not in covered
    ]

    diagnostics = []
    covered_before = set()

    for i, x in enumerate(selected, 1):
        f = features(x, requested)
        role = classify_selected(
            i,
            x,
            requested,
            selected[:i],
            covered_before,
        )

        diagnostics.append({
            "rank": i,
            "role": role,
            "covered": sorted(f["covered"]),
            "covered_sql": sorted(f["covered_sql"]),
            "relationship_type": x.get(
                "relationship_type", "NONE"
            ),
            "relationship_hit": bool(
                x.get("relationship_hit")
            ),
            "source_family": f["family"],
            "relation_signature": repr(
                f["relation_signature"]
            ),
            "sql_shape": repr(f["sql_shape"]),
        })

        covered_before |= f["covered"]

    return (
        selected,
        requested,
        sorted(covered),
        missing,
        sorted(relations),
        len(families),
        diagnostics,
    )


def print_context(
    tid,
    query,
    p,
    candidates,
    selected,
    requested,
    covered,
    missing,
    relations,
    family_count,
    diagnostics,
):
    print("\n" + "=" * 78)
    print(f"CONTEXT BUILDER V3.1 - {tid}")
    print(f"Query: {query}")
    print(f"Intent: {p.get('intent')}")
    print(f"V3 candidate pool: {len(candidates)}")
    print(f"Context selected: {len(selected)}")
    print(
        f"Coverage: {len(covered)}/{len(requested)} = "
        f"{(len(covered) / len(requested) if requested else 0):.2%}"
    )
    print(
        f"Missing: {', '.join(missing) if missing else '-'}"
    )
    print(
        f"Relations: {', '.join(relations) if relations else '-'}"
    )
    print(f"Source families: {family_count}")

    for i, x in enumerate(selected, 1):
        m = metadata(x)
        diag = diagnostics[i - 1]

        print("-" * 78)
        print(
            f"#{i} [{diag['role']}] "
            f"V3={x.get('final_score', 0):.2f} "
            f"rel={x.get('relationship_score', 0):.1f} "
            f"hit={'YES' if x.get('relationship_hit') else 'NO'} "
            f"type={x.get('relationship_type', 'NONE')} | "
            f"{m.get('source_file', '-')} | "
            f"chunk={m.get('chunk_index', '-')}"
        )

        print(
            f"   family: "
            f"{source_family(m.get('source_file', ''))}"
        )

        print(
            f"   exact: "
            f"{', '.join(x.get('exact_terms', [])) or '-'}"
        )

        print(
            f"   context: "
            f"{norm_text(content(x))[:900]}"
        )


def main():
    v3 = load_v3()
    db = v3.create_db()
    cache, cache_time = v3.load_cache(db)

    print("=" * 78)
    print("SAP ABAP RAG - V5.9 CONTEXT BUILDER V3.1")
    print(
        "V3 CANDIDATE POOL + COVERAGE + RELATION/SQL DIVERSITY "
        "+ REDUNDANCY CONTROL"
    )
    print("VectorDB mode: READ-ONLY")
    print(
        f"VectorDB: "
        f"{getattr(v3, 'DB_DIR', 'configured by V3')}"
    )
    print(
        f"Collection: "
        f"{getattr(v3, 'COLLECTION_NAME', 'configured by V3')}"
    )
    print(f"Exact cache load: {cache_time:.2f}s")

    report = {
        "version": "V5.9 Context Builder V3.1",
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "vector_db_read_only": True,
        "selection_policy": {
            "default_context_k": DEFAULT_CONTEXT_K,
            "t12_context_k": T12_CONTEXT_K,
            "preserve_v3_rank_1": True,
            "marginal_coverage": True,
            "relation_hit_preference": True,
            "relation_diversity": True,
            "sql_signature_diversity": True,
            "structural_diversity": True,
            "source_family_diversity": True,
            "text_redundancy_penalty": True,
            "strong_sql_redundancy_penalty": True,
            "coverage_safety_net": True,
            "role_classification": [
                "PRIMARY",
                "COMPLEMENTARY",
                "REDUNDANT",
                "FALLBACK",
            ],
        },
        "cases": [],
    }

    for tid, query in TEST_CASES:
        (
            p,
            candidates,
            duplicates,
            sem_time,
            sem_n,
            exact_n,
            merged_n,
        ) = v3_retrieve(
            v3,
            db,
            cache,
            query,
        )

        (
            selected,
            requested,
            covered,
            missing,
            relations,
            family_count,
            diagnostics,
        ) = select_context(
            tid,
            candidates,
            p,
        )

        print_context(
            tid,
            query,
            p,
            candidates,
            selected,
            requested,
            covered,
            missing,
            relations,
            family_count,
            diagnostics,
        )

        report["cases"].append({
            "id": tid,
            "query": query,
            "intent": p.get("intent"),
            "requested_terms": requested,
            "covered_terms": covered,
            "missing_terms": missing,
            "coverage": round(
                len(covered) / len(requested),
                6
            ) if requested else 0.0,
            "relations": relations,
            "source_family_count": family_count,
            "v3_candidates": len(candidates),
            "semantic_candidates": sem_n,
            "exact_candidates": exact_n,
            "merged_candidates": merged_n,
            "duplicates_removed": duplicates,
            "semantic_seconds": round(
                sem_time,
                4
            ),
            "context_k": len(selected),
            "selected": [
                {
                    "rank": i,
                    "role": diagnostics[i - 1]["role"],
                    "source_file": metadata(x).get(
                        "source_file", ""
                    ),
                    "chunk_index": metadata(x).get(
                        "chunk_index", ""
                    ),
                    "source_family": source_family(
                        metadata(x).get(
                            "source_file", ""
                        )
                    ),
                    "v3_score": x.get(
                        "final_score", 0.0
                    ),
                    "relationship_score": x.get(
                        "relationship_score", 0.0
                    ),
                    "relationship_type": x.get(
                        "relationship_type", "NONE"
                    ),
                    "relationship_hit": bool(
                        x.get("relationship_hit")
                    ),
                    "exact_terms": x.get(
                        "exact_terms", []
                    ),
                    "content": content(x),
                }
                for i, x in enumerate(
                    selected,
                    1
                )
            ],
            "selection_diagnostics": diagnostics,
        })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    json_path = (
        OUTPUT_DIR /
        "context_v3_1_results.json"
    )

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("SUMMARY CONTEXT BUILDER V3.1")
    print("=" * 78)

    for c in report["cases"]:
        print(
            f"{c['id']}: "
            f"coverage={c['coverage']:.2%} "
            f"missing={len(c['missing_terms'])} "
            f"families={c['source_family_count']} "
            f"context={c['context_k']}"
        )

        roles = [
            item["role"]
            for item in c["selected"]
        ]
        print(
            f"       roles={roles}"
        )

    print(f"Report JSON: {json_path}")
    print("Nessuna scrittura sul VectorDB.")


if __name__ == "__main__":
    main()
