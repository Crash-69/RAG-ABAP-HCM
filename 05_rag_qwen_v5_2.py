import re, csv, time
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
OUTPUT_DIR = Path(r"C:\Progetto_AI\RAG_V5_2_RESULTS")

SAP_TERMS = {
 "PA0001":["PA0001","P0001","PERNR","BUKRS","WERKS","PERSG","PERSK","BTRTL","GSBER","KOSTL","ORGEH","PLANS","STELL","SACHZ","BEGDA","ENDDA","STAT2"],
 "PA0002":["PA0002","P0002","PERNR","VORNA","NACHN","GBDAT","GESCH","FAMST"],
 "PA0007":["PA0007","P0007","PERNR","WOSTD","SCHKZ","ZTERF","ARBPL"],
 "PA0008":["PA0008","P0008","PERNR","BET01","BET02","WAERS","TRFAR","TRFGB","TRFGR","TRFST"]}

FUNCTIONAL_MAP = {
    "centro di costo": ["KOSTL", "PA0001", "PERNR"],
    "centro costo": ["KOSTL", "PA0001", "PERNR"],
    "dati organizzativi": ["PA0001", "PERNR", "BUKRS", "WERKS", "BTRTL", "PERSG", "PERSK", "ORGEH", "PLANS", "STELL"],
    "dati organizzativi dipendente": ["PA0001", "PERNR", "BUKRS", "WERKS", "BTRTL", "PERSG", "PERSK", "ORGEH", "PLANS", "STELL"],
}

def infer_intent(q):
    u=q.upper().strip()
    if re.search(r"\bSELECT\b|\bWHERE\b",u): return "code_pattern"
    if re.search(r"\bLEGGERE\b|\bLEGGI\b|\bRECUPERARE\b",u): return "code_lookup"
    if any(k in u for k in SAP_TERMS): return "table_reference"
    if any(re.search(rf"\b{re.escape(t)}\b",u) for v in SAP_TERMS.values() for t in v): return "field_lookup"
    if any(k in q.lower() for k in FUNCTIONAL_MAP): return "functional_lookup"
    return "semantic_lookup"

def query_profile(q):
    tables=[k for k in SAP_TERMS if re.search(rf"\b{k}\b",q.upper())]
    fields=[]
    for table in tables:
        fields += [x for x in SAP_TERMS[table] if x not in (table,"P"+table[2:])]
    fields += [x for v in SAP_TERMS.values() for x in v if re.search(rf"\b{re.escape(x)}\b",q.upper()) and x not in tables]
    functional=[]
    for k,v in FUNCTIONAL_MAP.items():
        if k in q.lower(): functional += v
    terms=list(dict.fromkeys(tables+fields+functional+re.findall(r"[A-Za-z_][A-Za-z0-9_/~]*",q.upper())))
    return {"query":q,"intent":infer_intent(q),"tables":tables,"fields":list(dict.fromkeys(fields)),"functional":list(dict.fromkeys(functional)),"terms":terms}

TEST_CASES = [
 ("T01","PA0001","table_reference",["PA0001","PERNR","KOSTL","WERKS","PERSG","PERSK","BTRTL","ORGEH","BEGDA","ENDDA"],["PA0001"]),
 ("T02","PA0002","table_reference",["PA0002","PERNR","VORNA","NACHN","GBDAT","GESCH","FAMST"],["PA0002"]),
 ("T03","PA0007","table_reference",["PA0007","PERNR","WOSTD","SCHKZ","ZTERF","ARBPL"],["PA0007"]),
 ("T04","PA0008","table_reference",["PA0008","PERNR","BET01","BET02","WAERS","TRFAR","TRFGB","TRFGR","TRFST"],["PA0008"]),
 ("T05","PERNR","field_lookup",["PERNR","PA0001"],["PA0001"]),
 ("T06","KOSTL","field_lookup",["KOSTL","PA0001"],["PA0001"]),
 ("T07","WERKS","field_lookup",["WERKS","PA0001"],["PA0001"]),
 ("T08","ORGEH","field_lookup",["ORGEH","PA0001"],["PA0001"]),
 ("T09","centro di costo del dipendente","functional_lookup",["KOSTL","PERNR","PA0001"],["PA0001"]),
 ("T10","leggere PA0001 per PERNR","code_lookup",["PA0001","PERNR"],["PA0001"]),
 ("T11","SELECT PA0001 WHERE PERNR","code_pattern",["PA0001","PERNR"],["PA0001"]),
 ("T12","dati organizzativi dipendente SAP HCM","functional_lookup",["PA0001","PERNR","WERKS","PERSG","PERSK","BTRTL","ORGEH","PLANS","STELL"],["PA0001"]),
]

def norm(s): return re.sub(r"[^A-Z0-9_/~]"," ",(s or "").upper())
def exact_terms(text, terms):
    t=norm(text); out=[]
    for x in terms:
        x=norm(x)
        if x and re.search(rf"(?<![A-Z0-9_]){re.escape(x)}(?![A-Z0-9_])",t): out.append(x)
    return list(dict.fromkeys(out))
def query_terms(q):
    return query_profile(q)["terms"]

def key(item):
    m=(item.get("document").metadata if item.get("document") else item.get("metadata") or {})
    return (m.get("source_file"),m.get("chunk_index"),m.get("chunk_hash"))
def ckey(item):
    c=item.get("document").page_content if item.get("document") else item.get("content","")
    return norm(c)
def db():
    if not DB_DIR.exists(): raise FileNotFoundError(f"VectorDB non trovato: {DB_DIR}")
    e=OllamaEmbeddings(model=EMBEDDING_MODEL,base_url=OLLAMA_BASE_URL)
    return Chroma(collection_name=COLLECTION_NAME,embedding_function=e,persist_directory=str(DB_DIR))

def exact_cache(vdb):
    col=vdb._collection; total=col.count(); cache=[]; start=time.perf_counter()
    print(f"Chunk VectorDB totali : {total}")
    for off in range(0,total,BATCH_SIZE):
        n=min(BATCH_SIZE,total-off); d=col.get(limit=n,offset=off,include=["documents","metadatas"])
        docs=d.get("documents") or []; metas=d.get("metadatas") or []
        for i,c in enumerate(docs):
            if c: cache.append({"content":c,"metadata":metas[i] if i<len(metas) else {}})
        done=off+len(docs)
        if done%5000==0 or done>=total: print(f"  Scansionati: {done}/{total} | cache: {len(cache)}")
    return cache,time.perf_counter()-start

def semantic(vdb,q):
    st=time.perf_counter(); rows=[]
    for doc,score in vdb.similarity_search_with_relevance_scores(q,k=SEMANTIC_K):
        rows.append({"document":doc,"semantic_score":float(score),"exact_score":0.,"exact_terms":[],"structural_score":0.,"structural_matches":[]})
    return rows,time.perf_counter()-st

def exact_search(cache,q):
    terms=query_terms(q); rows=[]
    for x in cache:
        found=exact_terms(x["content"],terms)
        if not found: continue
        score=sum(10 if t==q.upper() else 5 if t.startswith("PA") else 1 for t in found)+min(len(found)*.5,5)
        rows.append({"document":None,"content":x["content"],"metadata":x["metadata"],"semantic_score":0.,"exact_score":score,"exact_terms":found,"structural_score":0.,"structural_matches":[]})
    return rows

def structural(content, profile):
    t=content.upper(); tables=profile["tables"]; fields=profile["fields"]; intent=profile["intent"]; score=0.; matches=[]
    for table in tables:
        n=len(re.findall(rf"\bFROM\s+{re.escape(table)}\b",t))
        if n: matches.append(f"FROM {table} ({n})"); score+=min(n,6)*10
        n=len(re.findall(rf"\bSELECT\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(table)}\b",t))
        if n: matches.append(f"SELECT ... FROM {table} ({n})"); score+=min(n,4)*12
        n=len(re.findall(rf"\bSELECT\s+SINGLE\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(table)}\b",t))
        if n: matches.append(f"SELECT SINGLE ... FROM {table} ({n})"); score+=min(n,4)*15
        for f in fields:
            n=len(re.findall(rf"\b{re.escape(table)}-{re.escape(f)}\b",t))
            if n: matches.append(f"{table}-{f} ({n})"); score+=min(n,6)*4
    for f in fields:
        if re.search(rf"\bWHERE\b[\s\S]{{0,700}}?\b{re.escape(f)}\b",t): matches.append(f"WHERE ... {f}"); score+=10
    if re.search(r"\bLOOP\s+AT\b[\s\S]{0,250}?\bPA\d{4}\b",t): matches.append("LOOP AT PAxxxx"); score+=8
    if intent=="code_pattern" and re.search(r"\bSELECT\b[\s\S]{0,900}?\bFROM\s+PA\d{4}\b[\s\S]{0,900}?\bWHERE\b[\s\S]{0,500}?\bPERNR\b",t): matches.append("SELECT+FROM+WHERE+PERNR"); score+=30
    if intent=="functional_lookup" and "KOSTL" in profile["functional"] and "PERNR" in profile["functional"] and re.search(r"\bKOSTL\b[\s\S]{0,500}?\bPERNR\b|\bPERNR\b[\s\S]{0,500}?\bKOSTL\b",t): matches.append("KOSTL <-> PERNR"); score+=20
    return score,list(dict.fromkeys(matches))

def rank(items,q,expected_tables):
    profile=query_profile(q)
    for x in items:
        d=x.get("document"); m=d.metadata if d else x.get("metadata") or {}; c=d.page_content if d else x.get("content","")
        ss,sm=structural(c,profile); x["structural_score"]=ss; x["structural_matches"]=sm
        intent=profile["intent"]
        semantic_weight=35 if intent in ("functional_lookup","semantic_lookup") else 12
        score=x["semantic_score"]*semantic_weight+x["exact_score"]*3+ss
        if q.upper() in [z.upper() for z in x["exact_terms"]]: score+=20
        u=c.upper()
        for table in profile["tables"]:
            if re.search(rf"\b{re.escape(table)}\b",u): score+=8
        for field in profile["fields"]:
            if re.search(rf"\b{re.escape(field)}\b",u): score+=3
        if str(m.get("language","")).upper()=="ABAP": score+=8
        if str(m.get("language","")).upper()=="SAP_DDIC": score-=4
        src=str(m.get("source_file","")).lower()
        if src.endswith('.html'): score-=5
        elif src.endswith('.txt'): score+=2
        if intent=="field_lookup" and profile["fields"]:
            f=profile["fields"][0]
            if re.search(rf"\bPA\d{{4}}-{re.escape(f)}\b",u): score+=15
            if re.search(rf"\bSELECT\b[\s\S]{{0,700}}?\b{re.escape(f)}\b",u): score+=10
        if intent=="code_pattern" and re.search(r"\bSELECT\b[\s\S]{0,900}?\bFROM\s+PA\d{4}\b[\s\S]{0,900}?\bWHERE\b[\s\S]{0,500}?\bPERNR\b",u): score+=30
        x["final_score"]=score
    return sorted(items,key=lambda x:x["final_score"],reverse=True)[:FINAL_K]

def merge(a,b):
    d={x.setdefault("key",key(x)):x for x in a}
    for x in b:
        k=key(x); x["key"]=k
        if k in d: d[k]["exact_score"]=x["exact_score"]; d[k]["exact_terms"]=x["exact_terms"]
        else: x["document"]=Document(page_content=x["content"],metadata=x["metadata"]); d[k]=x
    unique={}; dup=0
    for x in d.values():
        k=ckey(x)
        if k in unique:
            dup+=1
            if x.get("exact_score",0)>unique[k].get("exact_score",0): unique[k]=x
        else: unique[k]=x
    return list(unique.values()),dup

def evaluate(results,case):
    _,q,intent,expected,expected_tables=case; expected={x.upper() for x in expected}; hits=set(); table_hit=abap=struct=selects=0
    for x in results:
        d=x["document"]; m=d.metadata or {}; c=d.page_content or ""; u=c.upper(); hits.update(z.upper() for z in x["exact_terms"])
        if any(re.search(rf"\b{re.escape(t)}\b",u) for t in expected_tables): table_hit+=1
        if m.get("language")=="ABAP": abap+=1
        if x.get("structural_matches"): struct+=1
        if re.search(r"\bSELECT\b|\bFROM\b",u): selects+=1
    return len(hits&expected)/len(expected) if expected else 0,table_hit,abap,struct,selects

def main():
    print("="*70); print("SAP ABAP RAG TEST - V5.2"); print("="*70)
    print(f"VectorDB : {DB_DIR}\nCollection : {COLLECTION_NAME}\nEmbedding : {EMBEDDING_MODEL}\nTest cases : {len(TEST_CASES)}")
    vdb=db(); cache,exact_time=exact_cache(vdb); rows=[]; sem_total=0
    for no,case in enumerate(TEST_CASES,1):
        tid,q,intent,expected,tables=case; profile=query_profile(q); print("\n"+"="*70); print(f"TEST {no}/{len(TEST_CASES)} - {tid}\nQuery: {q}\nIntent: {intent}"); print(f"Profile: intent={profile["intent"]} tables={profile["tables"]} fields={profile["fields"]}")
        s,st=semantic(vdb,q); sem_total+=st; e=exact_search(cache,q); merged,dups=merge(s,e); final=rank(merged,q,tables); cov,th,ab,ss,se=evaluate(final,case)
        for i,x in enumerate(final[:5],1):
            d=x["document"]; m=d.metadata or {}; print(f"  #{i} score={x['final_score']:.2f} exact={x['exact_score']:.1f} struct={x['structural_score']:.1f} | {m.get('source_file','-')} | chunk={m.get('chunk_index','-')}")
        print(f"Coverage: {cov:.2%} | Table hit: {th}/{len(final)} | ABAP: {ab}/{len(final)} | Structural: {ss}/{len(final)} | SELECT/FROM: {se}/{len(final)} | duplicates: {dups} | semantic: {st:.2f}s")
        rows.append(dict(id=tid,query=q,intent=intent,coverage_terms=f"{cov:.6f}",table_hit=th,abap_count=ab,structural_count=ss,select_count=se,semantic_seconds=f"{st:.4f}",duplicates_removed=dups,exact_results=len(e),final_results=len(final)))
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    csvp=OUTPUT_DIR/"rag_v5_2_results.csv"; mdp=OUTPUT_DIR/"rag_v5_2_report.md"
    with csvp.open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    avg=sum(float(x["coverage_terms"]) for x in rows)/len(rows); passed=sum(float(x["coverage_terms"])>=.5 and int(x["abap_count"])>0 for x in rows)
    md=["# SAP ABAP RAG V5.1 - Report",f"\nData: {datetime.now().isoformat(timespec='seconds')}",f"\nTest: {len(rows)}",f"\nPromettenti: {passed}/{len(rows)}",f"\nCoverage media: {avg:.2%}",f"\nExact cache: {exact_time:.2f}s",f"\nSemantic totale: {sem_total:.2f}s","\n| ID | Query | Intent | Coverage | Table hit | ABAP | Structural | SELECT |","|---|---|---|---:|---:|---:|---:|---:|"]
    md += [f"| {x['id']} | {x['query']} | {x['intent']} | {float(x['coverage_terms']):.2%} | {x['table_hit']} | {x['abap_count']} | {x['structural_count']} | {x['select_count']} |" for x in rows]
    mdp.write_text("\n".join(md),encoding="utf-8")
    print("\n"+"="*70); print("SUMMARY V5.1"); print("="*70); print(f"Test eseguiti: {len(rows)}\nPromettenti: {passed}/{len(rows)}\nCoverage media: {avg:.2%}\nExact cache: {exact_time:.2f}s\nSemantic totale: {sem_total:.2f}s")
    print(f"\nCSV: {csvp}\nMD : {mdp}")

if __name__=="__main__": main()
