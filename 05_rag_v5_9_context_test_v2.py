# -*- coding: utf-8 -*-
"""V5.9 Context Builder quality test. READ-ONLY VectorDB."""
import argparse, hashlib, os, re
from collections import Counter
import chromadb, requests

DB_DIR=r'C:\Progetto_AI\Abap_VectorDB'
COLLECTION_NAME='abap_hcm'
OLLAMA_URL='http://127.0.0.1:11434'
EMBEDDING_MODEL='nomic-embed-text:latest'
RAW=60
TESTS=[
('T01','PA0001',['PA0001']),('T02','PA0002',['PA0002']),('T03','PA0007',['PA0007']),('T04','PA0008',['PA0008']),
('T05','PERNR',['PERNR']),('T06','KOSTL',['KOSTL']),('T07','WERKS',['WERKS']),('T08','ORGEH',['ORGEH']),
('T09','centro di costo del dipendente',['KOSTL','PERNR','PA0001']),
('T10','leggere PA0001 per PERNR',['PA0001','PERNR']),('T11','SELECT PA0001 WHERE PERNR',['PA0001','PERNR']),
('T12','dati organizzativi dipendente SAP HCM',['PA0001','PERNR','WERKS','PERSG','PERSK','BTRTL','ORGEH','PLANS','STELL'])]

def U(x): return str(x or '').upper()
def clean(x): return re.sub(r'\s+',' ',x or '').strip()
def sh(x): return hashlib.sha1(clean(x).encode('utf8','ignore')).hexdigest()
def family(p):
    n=os.path.basename(str(p or '')).lower(); n=re.sub(r'\.(txt|html?|abap)$','',n)
    n=re.sub(r'(_bk|_backup|_copia|_copy|_old|_new|_p|_sp\d+|_v\d+)$','',n)
    return re.sub(r'([_-]\d+)$','',n)
def emb(q):
    r=requests.post(f'{OLLAMA_URL}/api/embeddings',json={'model':EMBEDDING_MODEL,'prompt':q},timeout=120); r.raise_for_status()
    e=r.json().get('embedding');
    if not e: raise RuntimeError('Ollama non ha restituito embedding')
    return e

def info(doc):
    u=U(doc); fields=set(re.findall(r'\b(PERNR|KOSTL|WERKS|PERSG|PERSK|BTRTL|ORGEH|PLANS|STELL|BUKRS)\b',u)); tables=set(re.findall(r'\bPA\d{4}\b',u))
    same=bool(re.search(r'\bSELECT\b[\s\S]{0,1200}\bFROM\s+PA0001\b[\s\S]{0,1200}\bWHERE\b',u))
    pa='PA0001' in tables or 'FROM PA0001' in u; per='PERNR' in fields; ko='KOSTL' in fields
    if pa and per and ko and same: rel='SELECT_KOSTL_FROM_PA0001_WHERE_PERNR'
    elif pa and per and same: rel='SELECT_PA0001_WHERE_PERNR'
    elif pa and per and fields & {'WERKS','PERSG','PERSK','BTRTL','ORGEH','PLANS','STELL'}: rel='PA0001_ORG_FIELDS_PERNR'
    elif pa and 'SELECT' in u and 'FROM PA0001' in u: rel='TABLE_FROM'
    elif 'SELECT' in u and fields: rel='SELECT_FIELD'
    else: rel='NONE'
    return fields|tables,rel,same

def retrieve(c,q):
    r=c.query(query_embeddings=[emb(q)],n_results=RAW,include=['documents','metadatas','distances'])
    out=[]
    for i,d in enumerate(r['documents'][0]):
        m=r['metadatas'][0][i] or {}; src=m.get('source_file') or m.get('source') or ''
        out.append({'id':r['ids'][0][i],'doc':d,'m':m,'dist':r['distances'][0][i],'fam':family(src)})
    seen=set(); seenfc=set(); z=[]
    for x in out:
        fc=(x['fam'],str(x['m'].get('chunk_index',x['m'].get('index','')))); h=sh(x['doc'])
        if h in seen or fc in seenfc: continue
        seen.add(h); seenfc.add(fc); x['terms'],x['rel'],x['same']=info(x['doc']); z.append(x)
    return z,len(out)-len(z)

def build(rows,q,req,k):
    covered=set(); rels=set(); fams=Counter(); selected=[]
    for _ in range(k):
        best=None; bs=-10**9; br=[]
        for x in rows:
            s=20/(1+max(0,x['dist'])); reasons=[]
            for t in req:
                if U(t) in x['terms']:
                    s += 28 if U(t) not in covered else 3; reasons.append(('NEW:' if U(t) not in covered else 'HIT:')+U(t))
            if x['rel']!='NONE':
                s += 24 if x['rel'] not in rels else 3
                if x['rel'] not in rels: reasons.append('NEW_REL:'+x['rel'])
            if x['same']: s+=12; reasons.append('SAME_SELECT')
            if U(x['m'].get('language'))=='ABAP': s+=5
            s += sum(5 for t in req if re.search(rf'\b{re.escape(U(t))}\b',U(x['doc'])))
            if fams[x['fam']]: s-=22*fams[x['fam']]
            uq=U(q)
            if 'PA0001' in uq and 'PERNR' in uq:
                if x['rel']=='SELECT_PA0001_WHERE_PERNR': s+=18
                elif x['rel']=='SELECT_KOSTL_FROM_PA0001_WHERE_PERNR' and 'KOSTL' not in uq: s-=8
            if s>bs: best,bs,br=x,s,reasons
        if best is None: break
        best=dict(best); best['score']=round(bs,2); best['why']=br; selected.append(best)
        covered|=best['terms']; rels.add(best['rel']); fams[best['fam']]+=1; rows.remove(next(y for y in rows if y['id']==best['id']))
    return selected

def run(q,req,k):
    c=chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION_NAME)
    rows,dups=retrieve(c,q); ctx=build(rows,q,req,k); covered=set(); rels=set()
    print('\n'+'='*78); print('SAP ABAP RAG - V5.9 CONTEXT BUILDER TEST | VectorDB READ-ONLY'); print('='*78)
    print('Query:',q); print('Richiesti:',', '.join(req)); print(f'Raw={len(rows)+dups}  Duplicati eliminati={dups}  Unici={len(rows)}  Context TOP={k}')
    print('\n'+'-'*78+'\nCONTEXT FINALE\n'+'-'*78)
    for i,x in enumerate(ctx,1):
        src=x['m'].get('source_file') or x['m'].get('source') or '?'; ch=x['m'].get('chunk_index',x['m'].get('index','?'))
        print(f'\n[{i}] score={x["score"]} | relation={x["rel"]} | family={x["fam"]} | chunk={ch}')
        print('source:',src); print('why:',', '.join(x['why']) or '-'); print('TEXT:',clean(x['doc'])[:1800])
        covered|=x['terms']; rels.add(x['rel'])
    missing=[t for t in req if U(t) not in covered]
    print('\n'+'-'*78+'\nCOVERAGE\n'+'-'*78)
    print('Coperti:',', '.join(t for t in req if U(t) in covered) or '-')
    print('Mancanti:',', '.join(missing) if missing else '-')
    print('Relazioni:',', '.join(sorted(rels-{"NONE"})) or '-')
    print('Famiglie:',len(set(x['fam'] for x in ctx)),'/',len(ctx))
    print('STATO:', 'ATTENZIONE - servono chunk complementari' if missing else 'OK - contesto completo sui termini richiesti')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--query'); p.add_argument('--top',type=int,default=8); p.add_argument('--test-all',action='store_true'); a=p.parse_args()
    if not a.query and not a.test_all: p.error('specificare --query oppure --test-all')
    if a.test_all:
        for tid,q,req in TESTS: print(f'\n\n######## {tid} ########'); run(q,req,a.top)
    else:
        q=U(a.query); req=[]
        for t in ['PA0001','PA0002','PA0007','PA0008','PERNR','KOSTL','WERKS','ORGEH','PERSG','PERSK','BTRTL','PLANS','STELL','BUKRS']:
            if re.search(rf'\b{t}\b',q): req.append(t)
        if 'CENTRO DI COSTO' in q: req += ['KOSTL','PERNR','PA0001']
        if 'DATI ORGANIZZATIVI' in q: req += ['PA0001','PERNR','WERKS','PERSG','PERSK','BTRTL','ORGEH','PLANS','STELL']
        run(a.query,list(dict.fromkeys(req)),a.top)
if __name__=='__main__': main()
