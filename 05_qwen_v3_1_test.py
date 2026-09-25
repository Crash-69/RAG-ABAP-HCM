# -*- coding: utf-8 -*-
import argparse,json,sys,time
from datetime import datetime
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

DEFAULT_CONTEXT=Path(r'C:\Progetto_AI\context_v3_1_results.json')
DEFAULT_OUTPUT=Path(r'C:\Progetto_AI\qwen_v3_1_results.json')
DEFAULT_OLLAMA='http://127.0.0.1:11434'
DEFAULT_MODEL='oisee/qwen-coder-abap:v7'
TESTS={
'T09':('centro di costo del dipendente','Spiega come leggere il centro di costo KOSTL di un dipendente identificato da PERNR tramite PA0001. Fornisci un esempio ABAP minimo, usando esclusivamente il contesto fornito, e considera la validità temporale del record.',['KOSTL','PA0001','PERNR']),
'T10':('leggere PA0001 per PERNR','Mostra come leggere PA0001 per un PERNR. Indica i campi utili e la condizione WHERE. Fornisci un esempio ABAP minimo basato esclusivamente sul contesto fornito.',['PA0001','PERNR']),
'T11':('SELECT PA0001 WHERE PERNR','Mostra il pattern ABAP/Open SQL per eseguire una SELECT su PA0001 filtrando per PERNR. Considera anche la validità temporale se presente nel contesto. Usa esclusivamente il contesto fornito.',['PA0001','PERNR']),
'T12':('dati organizzativi dipendente SAP HCM','Spiega quali dati organizzativi del dipendente possono essere letti da PA0001 sulla base del contesto fornito. Riporta i principali campi supportati dal contesto e un esempio ABAP coerente con esso. Non inventare campi o oggetti non presenti nel contesto.',['PA0001','PERNR'])}
SYSTEM='''Sei un Expert ABAP Enterprise Developer su SAP S/4HANA HCM.\n\nRegole vincolanti:\n1. Usa esclusivamente il contesto SAP ABAP fornito nel prompt.\n2. Non eseguire retrieval e non assumere accesso al VectorDB.\n3. Non inventare tabelle, campi, classi, function module o altri oggetti SAP.\n4. Se il contesto non è sufficiente, dichiaralo esplicitamente.\n5. Mantieni esattamente i nomi tecnici SAP presenti nel contesto.\n6. Preferisci ABAP Open SQL moderno e leggibile quando il contesto lo consente.\n7. Non usare SELECT *.\n8. Per PA infotype, considera begda/endda e gli eventuali filtri di validità presenti negli esempi forniti.\n9. Rispondi in italiano.\n10. La risposta deve essere tecnica e concreta, senza spiegazioni generiche non supportate dal contesto.'''
def get(url,payload=None,timeout=120):
 r=Request(url,method='GET' if payload is None else 'POST',headers={'Content-Type':'application/json'} if payload is not None else {})
 if payload is not None:r.data=json.dumps(payload,ensure_ascii=False).encode()
 try:
  with urlopen(r,timeout=timeout) as x:return json.loads(x.read().decode())
 except HTTPError as e:raise RuntimeError(f'HTTP {e.code} da Ollama: {e.read().decode(errors="replace")}')
 except URLError as e:raise RuntimeError(f'Ollama non raggiungibile: {e}')
def validate(d):
 if 'V3.1' not in str(d.get('version','')):raise ValueError('Context JSON non riconosciuto come V3.1')
 if d.get('vector_db_read_only') is not True:raise ValueError('Blocco sicurezza: vector_db_read_only non è true')
def cases(d):
 for k in ('tests','results'):
  v=d.get(k)
  if isinstance(v,dict):return v
  if isinstance(v,list):return {(x.get('test_id') or x.get('id')):x for x in v if isinstance(x,dict) and (x.get('test_id') or x.get('id'))}
 out={}
 def walk(x):
  if isinstance(x,dict):
   tid=x.get('test_id') or x.get('id')
   if tid in TESTS:out[tid]=x
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(d);return out
def selected(c):
 raw=next((c.get(k) for k in ('selected_context','selected_chunks','context','selected','items') if c.get(k) is not None),[])
 if isinstance(raw,dict):raw=raw.get('chunks') or raw.get('items') or [raw]
 out=[]
 for i,it in enumerate(raw if isinstance(raw,list) else [],1):
  if not isinstance(it,dict):continue
  m=dict(it.get('metadata') or {})
  for k in ('source_file','chunk_index','chunk','role','rank','v3_rank','v3_score','relationship_type','hit'):
   if k in it and k not in m:m[k]=it[k]
  out.append({'position':i,'metadata':m,'content':str(it.get('content') or it.get('document') or it.get('text') or it.get('chunk_text') or '')})
 return out
def prompt(tid,spec,c):
 blocks=[]
 for i,it in enumerate(selected(c),1):
  m=it['metadata'];blocks.append(f"--- CONTEXT {i} ---\nsource_file: {m.get('source_file','N/D')}\nchunk: {m.get('chunk_index',m.get('chunk','N/D'))}\nrole: {m.get('role','N/D')}\nrelationship_type: {m.get('relationship_type','N/D')}\n\n{it['content']}\n--- END CONTEXT {i} ---")
 return f"TEST {tid}\nQuery: {spec[0]}\n\nTASK:\n{spec[1]}\n\nCONTEXT SELEZIONATO DAL CONTEXT BUILDER V3.1:\n"+'\n\n'.join(blocks)+'\n\nProduci ora la risposta tecnica richiesta. Usa solo il contesto sopra.'
def main():
 p=argparse.ArgumentParser();p.add_argument('--context',type=Path,default=DEFAULT_CONTEXT);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--tests',default='T09,T10,T11,T12');p.add_argument('--ollama',default=DEFAULT_OLLAMA);p.add_argument('--model',default=DEFAULT_MODEL);p.add_argument('--temperature',type=float,default=0.0);p.add_argument('--timeout',type=int,default=300);a=p.parse_args()
 print('QWEN DIAGNOSTIC HARNESS V3.1 | READ-ONLY VectorDB | retrieval=NO')
 if not a.context.exists():print(f'ERRORE: non trovato {a.context}');sys.exit(2)
 d=json.loads(a.context.read_text(encoding='utf-8'));validate(d);cs=cases(d);req=[x.strip().upper() for x in a.tests.split(',') if x.strip()]
 miss=[x for x in req if x not in cs]
 if miss:raise ValueError('Test mancanti nel JSON: '+','.join(miss))
 tags=get(a.ollama.rstrip('/')+'/api/tags',timeout=30);models=[m.get('name','') for m in tags.get('models',[]) if isinstance(m,dict)]
 out={'version':'Qwen Diagnostic Harness V1','generated_at':datetime.now().isoformat(timespec='seconds'),'source_context_version':d.get('version'),'vector_db_read_only':True,'retrieval_executed':False,'ollama':{'base_url':a.ollama,'model':a.model,'temperature':a.temperature,'model_exactly_listed':a.model in models},'tests':{}}
 for tid in req:
  spec=TESTS[tid];sel=selected(cs[tid]);up=prompt(tid,spec,cs[tid]);payload={'model':a.model,'system':SYSTEM,'prompt':up,'stream':False,'options':{'temperature':a.temperature}}
  print(f'\n--- {tid} | {spec[0]} | chunks={len(sel)} ---');t=time.perf_counter()
  try:
   r=get(a.ollama.rstrip('/')+'/api/generate',payload,a.timeout);elapsed=time.perf_counter()-t;ans=str(r.get('response',''));diag={x:x.upper() in ans.upper() for x in spec[2]}
   out['tests'][tid]={'query':spec[0],'instruction':spec[1],'selected_context':sel,'system_prompt':SYSTEM,'user_prompt':up,'response':ans,'latency_seconds':round(elapsed,3),'anchor_diagnostics':diag,'anchors_all_present':all(diag.values()),'ollama_response_metadata':{k:r.get(k) for k in ('model','created_at','done','done_reason','total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration') if k in r}}
   print(f'latency={elapsed:.2f}s anchors={diag}\n{ans}')
  except Exception as e:out['tests'][tid]={'query':spec[0],'instruction':spec[1],'selected_context':sel,'system_prompt':SYSTEM,'user_prompt':up,'error':repr(e)};print('ERRORE:',e)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(f'\nOutput: {a.output}\nVectorDB modificato: NO\nRetrieval eseguito: NO')
if __name__=='__main__':main()
