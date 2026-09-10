# SAP ABAP RAG TEST - V5.3
# Functional Query Expansion + Field Usage + Contextual Proximity
# VectorDB is read-only.

import re, csv, time, pickle
from pathlib import Path
from datetime import datetime
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DB_DIR=Path(r"C:\Progetto_AI\Abap_VectorDB")
COLLECTION_NAME="abap_hcm"
EMBEDDING_MODEL="nomic-embed-text:latest"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
SEMANTIC_K=30
FINAL_K=15
BATCH_SIZE=500
OUTPUT_DIR=Path(r"C:\Progetto_AI\RAG_V5_3_RESULTS")
CACHE_FILE=OUTPUT_DIR/"exact_cache_v5_3.pkl"
CACHE_VERSION="5.3"

SAP_TERMS={
"PA0001":["PA0001","P0001","PERNR","BUKRS","WERKS","PERSG","PERSK","BTRTL","GSBER","KOSTL","ORGEH","PLANS","STELL","SACHZ","BEGDA","ENDDA","STAT2"],
"PA0002":["PA0002","P0002","PERNR","VORNA","NACHN","GBDAT","GESCH","FAMST"],
"PA0007":["PA0007","P0007","PERNR","WOSTD","SCHKZ","ZTERF","ARBPL"],
"PA0008":["PA0008","P0008","PERNR","BET01","BET02","WAERS","TRFAR","TRFGB","TRFGR","TRFST"]}

FUNCTIONAL_CONCEPTS={
"centro di costo":{"terms":["KOSTL","PA0001"],"related":["PERNR","BUKRS","WERKS","ORGEH"]},
"centro costo":{"terms":["KOSTL","PA0001"],"related":["PERNR","BUKRS","WERKS","ORGEH"]},
"costo del dipendente":{"terms":["KOSTL","PERNR","PA0001"],"related":["BUKRS","WERKS","ORGEH"]},
"dati organizzativi":{"terms":["PA0001","PERNR","BUKRS","WERKS","BTRTL","PERSG","PERSK","KOSTL","ORGEH","PLANS","STELL"],"related":["BEGDA","ENDDA","STAT2"]},
"dati organizzativi dipendente":{"terms":["PA0001","PERNR","BUKRS","WERKS","BTRTL","PERSG","PERSK","KOSTL","ORGEH","PLANS","STELL"],"related":["BEGDA","ENDDA","STAT2"]},
"dipendente":{"terms":["PERNR","PA0001"],"related":["BUKRS","WERKS","PERSG","PERSK"]}}

TEST_CASES=[
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
("T12","dati organizzativi dipendente SAP HCM","functional_lookup",["PA0001","PERNR","WERKS","PERSG","PERSK","BTRTL","ORGEH","PLANS","STELL"],["PA0001"])]

def norm(s): return re.sub(r"[^A-Z0-9_/~]"," ",(s or "").upper())
def present(s,t): return bool(re.search(rf"(?<![A-Z0-9_]){re.escape(norm(t))}(?![A-Z0-9_])",norm(s)))
def count_term(s,t): return len(re.findall(rf"(?<![A-Z0-9_]){re.escape(norm(t))}(?![A-Z0-9_])",norm(s)))
def exact_terms(s,terms): return list(dict.fromkeys([t.upper() for t in terms if present(s,t)]))

def detect_tables(q):
    return sorted(t for t in SAP_TERMS if re.search(rf"\b{t}\b",q.upper()))
def detect_fields(q):
    out=[]
    for terms in SAP_TERMS.values():
        for t in terms:
            if t.startswith("PA") and len(t)==6: continue
            if t.startswith("P") and len(t)==5: continue
            if re.search(rf"\b{re.escape(t)}\b",q.upper()): out.append(t)
    return list(dict.fromkeys(out))
def detect_concepts(q):
    q=q.lower()
    return [p for p in FUNCTIONAL_CONCEPTS if p in q]
def infer_intent(q):
    u=q.upper().strip()
    if re.search(r"\bSELECT\b|\bWHERE\b",u): return "code_pattern"
    if re.search(r"\b(LEGGERE|LEGGI|RECUPERARE|RECUPERA|ESTRARRE|OTTENERE|SELEZIONARE|READ|RETRIEVE|GET)\b",u): return "code_lookup"
    if detect_concepts(q): return "functional_lookup"
    if detect_tables(q) and not detect_fields(q): return "table_reference"
    if detect_fields(q): return "field_lookup"
    return "semantic_lookup"

def profile(q):
    tables=detect_tables(q); fields=detect_fields(q); concepts=detect_concepts(q)
    functional=[]; related=[]
    for c in concepts:
        functional += FUNCTIONAL_CONCEPTS[c]["terms"]; related += FUNCTIONAL_CONCEPTS[c]["related"]
    requested=list(dict.fromkeys(fields+functional))
    terms=list(dict.fromkeys(tables+requested+related+functional))
    return {"query":q,"intent":infer_intent(q),"tables":tables,"explicit_fields":fields,"concepts":concepts,"functional_terms":list(dict.fromkeys(functional)),"related_terms":list(dict.fromkeys(related)),"requested_fields":requested,"terms":terms}

def create_db():
    e=OllamaEmbeddings(model=EMBEDDING_MODEL,base_url=OLLAMA_BASE_URL)
    return Chroma(collection_name=COLLECTION_NAME,embedding_function=e,persist_directory=str(DB_DIR))

def load_cache(db):
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True); total=db._collection.count()
    if CACHE_FILE.exists():
        try:
            p=pickle.loads(CACHE_FILE.read_bytes())
            if p.get("version")==CACHE_VERSION and p.get("total")==total:
                print(f"Cache V5.3 caricata: {len(p['items'])} chunk"); return p["items"],0
        except Exception: pass
    start=time.perf_counter(); items=[]
    for off in range(0,total,BATCH_SIZE):
        lim=min(BATCH_SIZE,total-off)
        d=db._collection.get(limit=lim,offset=off,include=["documents","metadatas"])
        docs=d.get("documents") or []; metas=d.get("metadatas") or []
        for i,c in enumerate(docs):
            if c: items.append({"content":c,"metadata":metas[i] if i<len(metas) else {}})
        done=min(off+len(docs),total)
        if done%5000==0 or done>=total: print(f"Scansionati: {done}/{total} | cache: {len(items)}")
    CACHE_FILE.write_bytes(pickle.dumps({"version":CACHE_VERSION,"total":total,"created_at":datetime.now().isoformat(),"items":items},protocol=pickle.HIGHEST_PROTOCOL))
    return items,time.perf_counter()-start

def semantic(db,q):
    st=time.perf_counter()
    try: rr=db.similarity_search_with_relevance_scores(q,k=SEMANTIC_K)
    except Exception: rr=[(d,0.0) for d in db.similarity_search(q,k=SEMANTIC_K)]
    return [{"document":d,"semantic_score":float(s),"exact_score":0.0,"exact_terms":[],"structural_score":0.0,"field_score":0.0,"functional_score":0.0,"context_score":0.0} for d,s in rr],time.perf_counter()-st

def exact(cache,p):
    out=[]
    for x in cache:
        found=exact_terms(x["content"],p["terms"])
        if not found: continue
        s=0
        for t in found:
            if t in p["tables"]: s+=8
            elif t in p["explicit_fields"]: s+=3
            elif t in p["functional_terms"]: s+=2.5
            elif t in p["related_terms"]: s+=1
            elif t.startswith("PA"): s+=3
            else: s+=.5
        s+=min(len(found)*.5,5)
        out.append({"document":None,"content":x["content"],"metadata":x["metadata"],"semantic_score":0.0,"exact_score":s,"exact_terms":found,"structural_score":0.0,"field_score":0.0,"functional_score":0.0,"context_score":0.0})
    return out

def skey(x):
    d=x.get("document")
    m=d.metadata if d is not None else x.get("metadata",{})
    return (m.get("source_file"),m.get("chunk_index"),m.get("chunk_hash"))
def ckey(x):
    d=x.get("document"); c=d.page_content if d is not None else x.get("content","")
    c=re.sub(r"<[^>]+>"," ",c); c=re.sub(r"&(?:nbsp|amp|lt|gt);"," ",c,flags=re.I)
    return re.sub(r"[^A-Z0-9]","",c.upper())

def merge(a,b):
    m={skey(x):x for x in a}
    for x in b:
        k=skey(x)
        if k in m: m[k]["exact_score"]=x["exact_score"];m[k]["exact_terms"]=x["exact_terms"]
        else: x["document"]=Document(page_content=x["content"],metadata=x["metadata"]);m[k]=x
    u={}; dup=0
    for x in m.values():
        k=ckey(x)
        if k in u:
            dup+=1
            if x["exact_score"]>u[k]["exact_score"]: u[k]=x
        else: u[k]=x
    return list(u.values()),dup

def structural(c,p):
    t=c.upper(); score=0; hits=[]
    for tab in p["tables"]:
        n=len(re.findall(rf"\bFROM\s+{re.escape(tab)}\b",t))
        if n: score+=min(n,5)*8;hits.append(f"FROM {tab} x{n}")
        n=len(re.findall(rf"\bSELECT\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(tab)}\b",t))
        if n: score+=min(n,4)*10;hits.append(f"SELECT ... FROM {tab} x{n}")
        n=len(re.findall(rf"\bSELECT\s+SINGLE\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(tab)}\b",t))
        if n: score+=min(n,4)*14;hits.append(f"SELECT SINGLE ... FROM {tab} x{n}")
        for f in p["requested_fields"]:
            if re.search(rf"\b{re.escape(tab)}-{re.escape(f)}\b",t): score+=5;hits.append(f"{tab}-{f}")
    if "PERNR" in p["requested_fields"]:
        n=len(re.findall(r"\bWHERE\b[\s\S]{0,500}?\bPERNR\b",t))
        if n: score+=min(n,4)*8;hits.append(f"WHERE ... PERNR x{n}")
    for tab in p["tables"]:
        for f in p["requested_fields"]:
            if re.search(rf"\bSELECT\b[\s\S]{{0,500}}?\b{re.escape(f)}\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(tab)}\b",t):
                score+=18;hits.append(f"SELECT {f} FROM {tab}")
    if p["tables"] and "PERNR" in p["requested_fields"] and re.search(rf"\bSELECT\b[\s\S]{{0,900}}?\bFROM\s+(?:{'|'.join(map(re.escape,p['tables']))})\b[\s\S]{{0,900}}?\bWHERE\b[\s\S]{{0,500}}?\bPERNR\b",t):
        score+=22;hits.append("SELECT target table + WHERE PERNR")
    if re.search(r"\bLOOP\s+AT\b[\s\S]{0,250}?\bPA\d{4}\b",t): score+=5;hits.append("LOOP AT PAxxxx")
    if re.search(r"\bFIELD\b[\s\S]{0,200}?\bPA\d{4}-[A-Z0-9_]+\b",t):
        if p["intent"]=="field_lookup": score+=4;hits.append("DYNPRO FIELD")
        else: score-=4;hits.append("DYNPRO PENALTY")
    return score,list(dict.fromkeys(hits))

def field_score(c,p):
    t=c.upper(); s=0; hits=[]
    for f in p["requested_fields"]:
        n=count_term(t,f)
        if n: s+=min(n,5)*1.5
        for tab in p["tables"]:
            if re.search(rf"\b{re.escape(tab)}-{re.escape(f)}\b",t): s+=10;hits.append(f"{tab}-{f}")
            if re.search(rf"\bSELECT\b[\s\S]{{0,500}}?\b{re.escape(f)}\b[\s\S]{{0,700}}?\bFROM\s+{re.escape(tab)}\b",t): s+=14;hits.append(f"SELECT usage {f}")
    return s,list(dict.fromkeys(hits))

def functional_score(c,p):
    t=c.upper(); s=0; hits=[]
    for x in p["functional_terms"]:
        if present(t,x): s+=3;hits.append(f"functional:{x}")
    for x in p["related_terms"]:
        if present(t,x): s+=1;hits.append(f"related:{x}")
    if "KOSTL" in p["functional_terms"] and "PERNR" in p["functional_terms"]:
        if re.search(r"\bKOSTL\b[\s\S]{0,450}?\bPERNR\b|\bPERNR\b[\s\S]{0,450}?\bKOSTL\b",t):
            s+=25;hits.append("KOSTL <-> PERNR proximity")
        if re.search(r"\bSELECT\b[\s\S]{0,600}?\bKOSTL\b[\s\S]{0,700}?\bFROM\s+PA0001\b[\s\S]{0,800}?\bPERNR\b",t):
            s+=35;hits.append("SELECT KOSTL FROM PA0001 ... PERNR")
        if re.search(r"\bSELECT\s+SINGLE\b[\s\S]{0,600}?\bKOSTL\b[\s\S]{0,700}?\bFROM\s+PA0001\b[\s\S]{0,800}?\bPERNR\b",t):
            s+=20;hits.append("SELECT SINGLE KOSTL FROM PA0001 ... PERNR")
    if any(x in p["concepts"] for x in ("dati organizzativi","dati organizzativi dipendente")):
        org=["BUKRS","WERKS","BTRTL","PERSG","PERSK","KOSTL","ORGEH","PLANS","STELL"]
        got=[x for x in org if present(t,x)]
        s+=min(len(got),8)*4
        if got: hits.append("organizational fields: "+",".join(got))
        if present(t,"PA0001") and present(t,"PERNR") and len(got)>=3:
            s+=25;hits.append("PA0001 + PERNR + organizational fields")
    return s,list(dict.fromkeys(hits))

def context_score(c,p):
    t=c.upper(); important=list(dict.fromkeys(p["tables"]+p["requested_fields"]))
    if len(important)<2:return 0,[]
    s=0;hits=[]
    for start in range(0,max(len(t),1),400):
        w=t[start:start+800]; got=[x for x in important if present(w,x)]
        if len(got)>=2:s+=min(len(got)*2,12)
        if "PA0001" in got and "KOSTL" in got and "PERNR" in got:
            s+=20;hits.append("LOCAL PA0001 + KOSTL + PERNR")
    return min(s,50),list(dict.fromkeys(hits))

def rank(items,p):
    for x in items:
        d=x["document"]; c=d.page_content if d is not None else x["content"]; m=d.metadata if d is not None else x["metadata"]
        x["structural_score"],x["structural_matches"]=structural(c,p)
        x["field_score"],x["field_matches"]=field_score(c,p)
        x["functional_score"],x["functional_matches"]=functional_score(c,p)
        x["context_score"],x["context_matches"]=context_score(c,p)
        se=x["semantic_score"]; ex=x["exact_score"]; st=x["structural_score"]; fs=x["field_score"]; fu=x["functional_score"]; cs=x["context_score"]
        if p["intent"]=="functional_lookup": score=se*30+ex*2+st*.55+fs+fu+cs*.8
        elif p["intent"]=="field_lookup": score=se*12+ex*2.5+st*.55+fs*1.25+cs*.5
        elif p["intent"]=="code_pattern": score=se*10+ex*3+st*.9+fs*.9+cs*.5
        elif p["intent"]=="code_lookup": score=se*12+ex*3+st*.8+fs*.8+cs*.5
        elif p["intent"]=="table_reference": score=se*10+ex*3+st*.8+fs*.4+cs*.25
        else: score=se*35+ex*2+st*.5+fs+fu
        if p["query"].upper() in [z.upper() for z in x["exact_terms"]]: score+=15
        if str(m.get("language","")).upper()=="ABAP":score+=8
        sf=str(m.get("source_file","")).lower()
        if sf.endswith(".txt"):score+=2
        elif sf.endswith(".html"):score-=5
        if p["intent"]=="functional_lookup" and {"KOSTL","PERNR"}.issubset(set(p["functional_terms"])) and re.search(r"\bSELECT\b[\s\S]{0,700}?\bKOSTL\b[\s\S]{0,700}?\bFROM\s+PA0001\b[\s\S]{0,800}?\bPERNR\b",c.upper()): score+=60
        x["final_score"]=score
    return sorted(items,key=lambda x:x["final_score"],reverse=True)[:FINAL_K]

def evaluate(res,case):
    _,_,_,expected,tables=case; es=set(x.upper() for x in expected); hits=set();th=ab=st=sel=0
    for x in res:
        d=x["document"];m=d.metadata;c=d.page_content;u=c.upper();hits.update(z.upper() for z in x["exact_terms"])
        if any(present(u,t) for t in tables):th+=1
        if str(m.get("language","")).upper()=="ABAP":ab+=1
        if x["structural_matches"]:st+=1
        if re.search(r"\bSELECT\b|\bFROM\b",u):sel+=1
    return (len(hits&es)/len(es) if es else 0),th,ab,st,sel

def main():
    print("="*70);print("SAP ABAP RAG TEST - V5.3");print("="*70)
    print(f"VectorDB : {DB_DIR}\nCollection : {COLLECTION_NAME}\nEmbedding : {EMBEDDING_MODEL}\nTest cases : {len(TEST_CASES)}")
    db=create_db(); cache,cache_time=load_cache(db); rows=[]; semtot=0
    for i,case in enumerate(TEST_CASES,1):
        tid,q,expected_intent,expected,tables=case;p=profile(q)
        print("\n"+"="*70);print(f"TEST {i}/{len(TEST_CASES)} - {tid}");print(f"Query: {q}");print(f"Intent: {p['intent']}")
        print(f"Tabelle: {', '.join(p['tables']) or '-'}");print(f"Campi: {', '.join(p['explicit_fields']) or '-'}");print(f"Concetti: {', '.join(p['concepts']) or '-'}");print(f"Campi richiesti: {', '.join(p['requested_fields']) or '-'}")
        a,dt=semantic(db,q);semtot+=dt;b=exact(cache,p);merged,dup=merge(a,b);res=rank(merged,p)
        cov,th,ab,st,sel=evaluate(res,case)
        for n,x in enumerate(res[:5],1):
            d=x["document"];m=d.metadata
            print(f"  #{n} score={x['final_score']:.2f} sem={x['semantic_score']:.3f} exact={x['exact_score']:.1f} struct={x['structural_score']:.1f} field={x['field_score']:.1f} func={x['functional_score']:.1f} ctx={x['context_score']:.1f} | {m.get('source_file','-')} | chunk={m.get('chunk_index','-')}")
        print(f"Coverage: {cov:.2%} | Table hit: {th}/15 | ABAP: {ab}/15 | Structural: {st}/15 | SELECT/FROM: {sel}/15 | duplicates: {dup} | semantic: {dt:.2f}s")
        rows.append({"id":tid,"query":q,"intent":p["intent"],"coverage":cov,"table_hit":th,"abap":ab,"structural":st,"select":sel,"duplicates":dup,"semantic_seconds":dt})
    avg=sum(x["coverage"] for x in rows)/len(rows);prom=sum(1 for x in rows if x["coverage"]>=.5 and x["abap"]>0)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    with (OUTPUT_DIR/"rag_v5_3_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    md=["# SAP ABAP RAG V5.3 - Report","",f"Data: {datetime.now().isoformat(timespec='seconds')}","",f"Test eseguiti: {len(rows)}",f"Promettenti: {prom}/{len(rows)}",f"Coverage media: {avg:.2%}",f"Exact cache: {cache_time:.2f}s",f"Semantic totale: {semtot:.2f}s","", "| ID | Query | Intent | Coverage | Table | ABAP | Structural | SELECT | Duplicati |","|---|---|---|---:|---:|---:|---:|---:|---:|"]
    md += [f"| {x['id']} | {x['query']} | {x['intent']} | {x['coverage']:.2%} | {x['table_hit']} | {x['abap']} | {x['structural']} | {x['select']} | {x['duplicates']} |" for x in rows]
    (OUTPUT_DIR/"rag_v5_3_report.md").write_text("\n".join(md),encoding="utf-8")
    print("\n"+"="*70);print("SUMMARY V5.3");print("="*70);print(f"Test eseguiti: {len(rows)}");print(f"Promettenti: {prom}/{len(rows)}");print(f"Coverage media: {avg:.2%}");print(f"Exact cache: {cache_time:.2f}s");print(f"Semantic totale: {semtot:.2f}s")

if __name__=="__main__": main()
