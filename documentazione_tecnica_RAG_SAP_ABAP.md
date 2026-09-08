# Documentazione tecnica — RAG ibrido per SAP ABAP/HCM con ChromaDB, Ollama e Qwen

## 1. Obiettivo del progetto

Il progetto nasce con l'obiettivo di realizzare un sistema RAG (Retrieval-Augmented Generation) specializzato sul patrimonio di codice SAP ABAP/HCM.

L'architettura deve permettere di:

1. interrogare una base documentale composta da codice ABAP e informazioni tecniche SAP;
2. recuperare il codice più pertinente rispetto a una query;
3. combinare ricerca semantica e ricerca lessicale/esatta;
4. riconoscere automaticamente tabelle e campi SAP;
5. eliminare duplicati derivanti da esportazioni o versioni duplicate del codice;
6. applicare un ranking strutturale specifico per ABAP;
7. inviare a un modello locale Qwen soltanto il contesto più utile;
8. generare codice ABAP coerente con il patrimonio esistente.

Il caso di test principale utilizzato durante lo sviluppo è stato:

```text
PA0001
```

con particolare attenzione alla ricerca di utilizzi reali della tabella PA0001 e dei relativi campi, soprattutto PERNR, KOSTL, WERKS, PERSG, PERSK, BTRTL, ORGEH, BEGDA ed ENDDA.

---

# 2. Architettura generale

La pipeline sviluppata è concettualmente:

```text
                         ┌──────────────────────┐
                         │      Query utente    │
                         │       PA0001         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Query profiling    │
                         │ tabelle / campi /    │
                         │ intent / chiavi      │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
          ┌──────────────────┐             ┌──────────────────┐
          │ Ricerca          │             │ Ricerca esatta   │
          │ semantica        │             │ termini SAP      │
          │ Embedding        │             │ lexical scan     │
          └────────┬─────────┘             └────────┬─────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ Merge                │
                         │ + deduplicazione     │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ SAP Structural       │
                         │ Ranking              │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Top K contesto       │
                         │ ABAP                 │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Qwen Coder ABAP      │
                         │ tramite Ollama       │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Risposta ABAP        │
                         └──────────────────────┘
```

L'idea fondamentale è evitare di affidarsi esclusivamente agli embedding.

Nel dominio SAP una query come `PA0001` è un identificatore tecnico molto preciso. Una ricerca puramente semantica può infatti recuperare codice concettualmente correlato ma non necessariamente contenente la tabella richiesta.

Per questo motivo è stata introdotta una pipeline ibrida.

---

# 3. Ambiente tecnico

L'implementazione utilizza:

- Python;
- ChromaDB;
- LangChain;
- Ollama;
- `nomic-embed-text:latest` per gli embedding;
- `oisee/qwen-coder-abap:v7` come modello generativo;
- VectorDB persistente locale;
- codice SAP ABAP/HCM come corpus.

Configurazione utilizzata:

```python
DB_DIR = Path(r"C:\Progetto_AI\Abap_VectorDB")
COLLECTION_NAME = "abap_hcm"

EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
```

La base contiene circa:

```text
46.346 chunk
```

---

# 4. Prima fase — verifica del VectorDB e degli embedding

Prima di intervenire sul ranking è stata verificata la disponibilità del VectorDB e del modello di embedding.

La configurazione prevede:

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(DB_DIR)
)
```

Questa fase è importante perché un problema nel modello di embedding o nella connessione a Ollama renderebbe inutili le successive ottimizzazioni del retrieval.

---

# 5. Prima implementazione — ricerca semantica

La prima componente del retrieval utilizza la ricerca semantica di ChromaDB:

```python
vector_db.similarity_search_with_relevance_scores(
    query,
    k=30
)
```

Il parametro iniziale utilizzato è:

```python
SEMANTIC_K = 30
```

Il risultato contiene:

- `Document`;
- score semantico;
- metadata;
- contenuto del chunk.

La ricerca semantica è utile per comprendere il significato della richiesta, ma non è sufficiente per identificatori SAP.

---

# 6. Problema emerso: la ricerca semantica da sola

Con la query:

```text
PA0001
```

la ricerca semantica produceva 30 risultati.

Tuttavia, non era garantito che i primi risultati rappresentassero realmente gli utilizzi più importanti della tabella.

Nel codice ABAP possono infatti esistere moltissimi riferimenti indiretti o concettualmente correlati.

Esempio:

```abap
SELECT SINGLE kostl
  FROM pa0001
  WHERE pernr = lv_pernr.
```

deve essere considerato molto più pertinente rispetto a un chunk che contiene soltanto un riferimento generico a `PERNR`.

Da questa osservazione nasce la ricerca esatta dei termini SAP.

---

# 7. Seconda fase — SAP Terms Knowledge Base

È stata introdotta una knowledge base locale dei termini SAP.

Esempio:

```python
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
}
```

La stessa impostazione è stata applicata anche ad altre infotype:

```text
PA0002
PA0007
PA0008
```

L'approccio è riutilizzabile per qualsiasi dominio SAP:

```text
PAxxxx
Pxxxx
Txxxx
HRPxxxx
V_Txxxx
strutture Pxxxx
campi DDIC
transaction code
function module
class
method
BAdI
enhancement
```

---

# 8. Normalizzazione lessicale

Per evitare problemi derivanti da maiuscole, caratteri speciali e formattazioni differenti è stata introdotta una funzione di normalizzazione:

```python
def normalizza_testo(text):
    if not text:
        return ""

    return re.sub(
        r"[^A-Z0-9_/~]",
        " ",
        text.upper()
    )
```

La normalizzazione viene utilizzata esclusivamente per il confronto lessicale.

Non modifica il contenuto originale del chunk.

---

# 9. Exact Match con regex

I termini vengono cercati utilizzando boundary regex.

L'obiettivo è evitare falsi positivi grossolani.

Per esempio:

```text
PERNR
```

deve essere distinto da una semplice sottostringa casuale contenuta in un identificatore più lungo.

La funzione concettuale è:

```python
def trova_termini_esatti(text, termini):
    ...
```

e restituisce esclusivamente i termini effettivamente presenti.

---

# 10. Problema ChromaDB — "too many SQL variables"

Durante la prima implementazione della ricerca esatta si è verificato:

```text
chromadb.errors.InternalError:
Error executing plan:
Internal error:
error returned from database:
(code: 1) too many SQL variables
```

Il problema era causato dal tentativo di leggere troppi record da ChromaDB in una singola chiamata.

La soluzione è stata fondamentale e generalizzabile:

```python
BATCH_SIZE = 500
```

e lettura tramite:

```python
for offset in range(0, total, BATCH_SIZE):

    data = collection.get(
        limit=batch_limit,
        offset=offset,
        include=[
            "documents",
            "metadatas"
        ]
    )
```

In questo modo 46.346 chunk vengono elaborati a blocchi.

Il VectorDB non viene modificato.

Questa soluzione deve essere mantenuta anche nelle implementazioni successive.

---

# 11. Risultato della ricerca esatta

Con:

```text
PA0001
```

sono stati individuati:

```text
46.346 chunk totali
22.471 chunk contenenti almeno un termine SAP rilevante
```

Questo dato è stato importante.

Dimostra che il retrieval lessicale è molto più selettivo rispetto alla sola ricerca semantica, ma genera anche un numero elevato di candidati.

Per questo è necessario un ranking più sofisticato.

---

# 12. Terza fase — fusione semantica + exact search

I risultati delle due ricerche vengono combinati:

```text
Semantic results
       +
Exact results
       ↓
Merge
       ↓
Deduplication
```

La chiave iniziale di deduplicazione utilizza metadata come:

```text
source_file
chunk_index
chunk_hash
```

In questo modo un chunk presente sia nella ricerca semantica sia in quella esatta non viene duplicato.

---

# 13. Quarta fase — primo ranking ibrido

È stato introdotto un primo ranking che combina:

- semantic score;
- exact score;
- presenza esatta della query;
- source file;
- object name;
- language;
- struttura ABAP.

Concettualmente:

```text
Final Score =
    Semantic Component
  + Exact Component
  + Query Exact Bonus
  + Source Bonus
  + Structural Component
  - Penalty
```

Questa rappresentazione è più utile di un semplice score semantico.

---

# 14. Perché il ranking semantico non era sufficiente

Durante i test V3 è emerso che alcuni chunk raggiungevano score elevati semplicemente perché contenevano molti termini SAP.

Questo non significa necessariamente che siano i migliori esempi di codice per rispondere alla domanda.

Per esempio, per:

```text
PA0001
```

è molto più interessante:

```abap
SELECT SINGLE *
  FROM pa0001
  WHERE pernr = ...
```

rispetto a:

```abap
DATA lv_kostl LIKE pa0001-kostl.
```

Il secondo contiene PA0001, ma non dimostra un accesso alla tabella.

Da qui nasce il SAP Structural Ranking.

---

# 15. Quinta fase — SAP Structural Ranking

Sono stati introdotti pattern specifici per ABAP.

Tra i pattern utilizzati:

```text
FROM PAxxxx
SELECT ... FROM PAxxxx
SELECT SINGLE ... FROM PAxxxx
PAxxxx-FIELD
WHERE ... PERNR
WHERE ... KOSTL
LOOP AT PAxxxx
READ TABLE
```

Esempio di struttura di scoring:

```text
from_pa_table
select_from_pa
select_single_from_pa
pa_field
where_pernr
```

Sono stati inoltre introdotti pattern composti, come:

```text
FROM QUERY TABLE
SELECT TABLE + WHERE PERNR
SELECT SINGLE PA0001 + PERNR
```

L'obiettivo è riconoscere non solo la presenza di una stringa, ma il suo ruolo all'interno del codice.

---

# 16. Risultati V3

La V3 ha prodotto risultati molto più coerenti.

Esempio:

```text
Final score       : 125.0000
Exact score       : 24.0000
Structural score  : 48.0000
Query exact bonus : 15.0000
```

I primi risultati contenevano effettivamente:

```abap
SELECT SINGLE * FROM pa0001
  WHERE pernr = ...
```

oppure:

```abap
SELECT pernr kostl sname
  INTO ...
  FROM pa0001
  WHERE pernr IN ...
```

La diagnostica indicava:

```text
Risultati ABAP               : 15
Match query esatto           : 15
Con score strutturale        : 15
Con SELECT                   : 15
Con FROM PAxxxx              : 15
Con pattern Dynpro           : 15
```

Questo ha confermato che la direzione tecnica era corretta.

---

# 17. Sesta fase — Content Deduplication V4

Un problema ulteriore è emerso analizzando i risultati.

Il corpus contiene file con nomi differenti ma contenuto identico o quasi identico.

Esempi concettuali:

```text
ztm_simulazione_di_spesa.txt
ztm_simulazione_di_spesa_ele.txt
ztm_simulazione_di_spesa_ele_1.txt
```

possono contenere lo stesso blocco ABAP.

La deduplicazione basata soltanto su metadata non è sufficiente.

È stata quindi introdotta una deduplicazione per contenuto.

Risultato del test:

```text
Risultati unici iniziali       : 22.482
Duplicati per contenuto rimossi: 7.933
Risultati dopo content dedup   : 14.549
```

Questa fase riduce drasticamente il rumore del corpus.

---

# 18. Perché il content deduplication è importante

In un sistema RAG, duplicati del codice producono almeno tre problemi:

1. lo stesso esempio può occupare molte posizioni del ranking;
2. il contesto inviato al modello può contenere codice ripetuto;
3. il modello può interpretare la frequenza del codice come maggiore rilevanza.

La deduplicazione rende il retrieval più rappresentativo del patrimonio reale.

È quindi una componente consigliata in qualsiasi RAG costruito su esportazioni di codice.

---

# 19. Settima fase — problema del context window di Qwen

Dopo aver ottenuto un retrieval corretto, la fase generativa ha evidenziato un problema indipendente.

Qwen restituiva:

```text
request (4489 tokens) exceeds
the available context size (4096 tokens)
```

Quindi:

```text
Prompt = 4489 token
Context disponibile = 4096 token
```

Il problema non era il retrieval.

Era la quantità di testo inviata al modello.

Questo è un punto architetturale fondamentale:

```text
Retrieval quality
        ≠
Generation context management
```

Sono due problemi distinti.

---

# 20. Riduzione del contesto

È stata quindi introdotta una gestione più controllata del contesto.

L'obiettivo è:

```text
46.346 chunk
     ↓
retrieval
     ↓
ranking
     ↓
pochi chunk ad alta qualità
     ↓
prompt compatto
     ↓
Qwen
```

Non bisogna inviare al modello tutto ciò che il retriever trova.

Il retriever deve essere aggressivo nella selezione.

Il generatore deve ricevere soltanto evidenze pertinenti.

---

# 21. Risultato del primo test Qwen

In una fase intermedia Qwen ha prodotto:

```abap
SELECT SINGLE kostl
  FROM pa0001
  INTO @DATA(lv_kostl)
  WHERE pernr = lv_pernr.
```

La risposta era tecnicamente plausibile, ma il retrieval mostrava ancora margini di miglioramento.

In un altro test, invece, Qwen ha prodotto:

```abap
DATA(lv_kostl) = pa0001-kostl.
```

Questa risposta è stata considerata non corretta rispetto alla domanda di lettura da PA0001, perché presumeva che il record PA0001 fosse già disponibile.

Questo ha dimostrato che la qualità della risposta generativa dipende fortemente dal tipo di contesto recuperato.

---

# 22. Ottava fase — Query Profiling V5

Per migliorare ulteriormente il sistema è stato introdotto il Query Profile.

La query viene analizzata prima del retrieval.

Il profilo cerca di distinguere:

```text
Tabelle
Campi
Campi richiesti
Campi chiave
Intent
```

Esempio desiderato per una domanda:

```text
Come posso leggere da PA0001 il centro di costo
partendo dal PERNR?
```

Profilo:

```text
Tabelle         : PA0001
Campi           : KOSTL, PERNR
Campi richiesti : KOSTL
Campi chiave    : PERNR
Intent          : data_read
```

Questo permette al ranking di sapere che non basta trovare PA0001: deve trovare un accesso alla tabella coerente con PERNR e KOSTL.

---

# 23. Bug rilevato nel Query Profile V5

Durante il test:

```text
PA0001
```

il profilo ha prodotto:

```text
Tabelle         : PA0001
Campi           : PA0001
Campi richiesti : PA0001
Intent          : field_lookup
```

Questo è errato.

Il motivo è che `PA0001` era contemporaneamente presente:

- nella lista delle tabelle;
- nella lista generale dei termini SAP.

Il parser lo classificava quindi anche come campo.

---

# 24. Correzione concettuale del Query Profile

È stata definita una separazione esplicita:

```text
PA0001 → TABLE
P0001  → TABLE ALIAS
PERNR  → FIELD
KOSTL  → FIELD
BUKRS  → FIELD
WERKS  → FIELD
...
```

Le tabelle non devono mai essere trattate come campi.

La regola generale è:

```text
TABLE_TERMS
FIELD_TERMS
```

devono essere insiemi distinti.

Eventuali tabelle individuate nel set dei termini generali devono essere filtrate prima della classificazione dei campi.

---

# 25. Intent classification

La classificazione dell'intento permette di distinguere almeno:

```text
table_reference
field_lookup
data_read
code_generation
general
```

Esempi:

### Query

```text
PA0001
```

Intent:

```text
table_reference
```

### Query

```text
Qual è il campo del centro di costo in PA0001?
```

Intent:

```text
field_lookup
```

### Query

```text
Leggi KOSTL da PA0001 usando PERNR
```

Intent:

```text
data_read
```

### Query

```text
Scrivi una SELECT ABAP per leggere KOSTL
```

Intent:

```text
code_generation
```

Questo permette di usare pesi diversi nel ranking.

---

# 26. Architettura finale prevista

La pipeline evoluta può essere rappresentata come:

```text
                         QUERY
                           │
                           ▼
                  ┌─────────────────┐
                  │ Query Profile   │
                  ├─────────────────┤
                  │ Tables          │
                  │ Fields          │
                  │ Key fields      │
                  │ Requested       │
                  │ Intent          │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Semantic Search            Exact SAP Search
              │                         │
              └────────────┬────────────┘
                           ▼
                    Candidate Pool
                           │
                           ▼
                  Metadata Dedup
                           │
                           ▼
                  Content Dedup
                           │
                           ▼
                Structural Analysis
                           │
                           ▼
                   Hybrid Ranking
                           │
                           ▼
                    Top N Context
                           │
                           ▼
                 Context Compression
                           │
                           ▼
                       Qwen
                           │
                           ▼
                    ABAP Response
```

---

# 27. Principi di progettazione riutilizzabili

## 27.1 Non affidarsi solo agli embedding

Gli embedding sono ottimi per il significato.

Non sono sufficienti per identificatori tecnici.

Per domini come SAP è necessario un layer lexical/exact.

---

## 27.2 Separare retrieval e ranking

Il retrieval deve raccogliere candidati.

Il ranking deve stabilire quali candidati sono migliori.

Non bisogna usare il primo score disponibile come risultato finale.

---

## 27.3 Il codice ha una struttura

Nel codice sorgente la presenza di una parola non equivale alla sua rilevanza.

È necessario riconoscere pattern:

```text
SELECT
FROM
WHERE
READ TABLE
LOOP AT
CALL FUNCTION
CALL METHOD
PERFORM
```

e le relazioni tra gli elementi.

---

## 27.4 Deduplicare prima della generazione

I duplicati devono essere eliminati prima della costruzione del prompt.

Questo riduce:

- token;
- latenza;
- rumore;
- rischio di bias del modello.

---

## 27.5 Il contesto deve essere limitato

Un RAG non deve passare al LLM tutto ciò che trova.

La qualità del contesto è più importante della quantità.

---

# 28. Metriche da monitorare

Per ogni test è consigliabile registrare almeno:

```text
Retrieval time
Semantic results
Exact results
Merged results
Metadata duplicates
Content duplicates
Final candidates
Top K
Prompt characters
Estimated prompt tokens
LLM input tokens
LLM output tokens
Generation time
Total time
```

Questo permette di distinguere problemi di:

```text
retrieval
ranking
prompt construction
model inference
```

---

# 29. Test case consigliati

Il sistema non deve essere validato soltanto con `PA0001`.

È consigliabile creare una suite di test.

## Test 1 — Tabella

```text
PA0001
```

Atteso:

```text
Intent = table_reference
Table = PA0001
Fields = none
```

## Test 2 — Campo

```text
KOSTL in PA0001
```

Atteso:

```text
Table = PA0001
Field = KOSTL
Intent = field_lookup
```

## Test 3 — Lettura

```text
Leggere KOSTL da PA0001 tramite PERNR
```

Atteso:

```text
Table = PA0001
Fields = KOSTL, PERNR
Intent = data_read
```

## Test 4 — SELECT

```text
SELECT ABAP per leggere KOSTL da PA0001
```

Atteso:

```text
Intent = code_generation
```

## Test 5 — altra infotype

```text
PA0002
```

## Test 6 — più tabelle

```text
PA0001 PA0002
```

## Test 7 — struttura P

```text
P0001-KOSTL
```

## Test 8 — concetto funzionale

```text
Come recuperare il centro di costo di un dipendente?
```

In questo caso il parser deve inferire i termini tecnici senza richiedere necessariamente il nome esplicito della tabella.

---

# 30. Validazione ABAP

Per un sistema destinato alla generazione di codice SAP ABAP è importante introdurre una fase di validazione.

Il codice generato dovrebbe essere controllato almeno per:

1. sintassi ABAP plausibile;
2. esistenza della tabella;
3. esistenza del campo;
4. coerenza tra tabella e campo;
5. uso corretto delle variabili host;
6. gestione delle date di validità;
7. presenza di `PERNR` quando richiesto;
8. rispetto delle convenzioni ABAP moderne.

Per esempio, una risposta:

```abap
SELECT SINGLE kostl
  FROM pa0001
  INTO @DATA(lv_kostl)
  WHERE pernr = @lv_pernr.
```

è concettualmente diversa da:

```abap
DATA(lv_kostl) = pa0001-kostl.
```

La seconda presuppone che il work area `PA0001` sia già popolato.

Questa distinzione deve essere incorporata nel futuro validator.

---

# 31. Miglioramenti futuri

## 31.1 Metadata filtering

Prima del ranking potrebbe essere applicato un filtro per:

```text
language = ABAP
```

quando la query richiede codice.

---

## 31.2 Source diversity

Il ranking dovrebbe evitare che i primi 15 risultati provengano tutti dallo stesso programma.

Si può introdurre:

```text
source diversity bonus
```

o una penalizzazione per troppi chunk dello stesso source.

---

## 31.3 Chunk adjacency

Se il chunk precedente o successivo contiene la continuazione di una SELECT, il sistema dovrebbe poter recuperare anche i chunk adiacenti.

Questo è particolarmente importante per codice ABAP spezzato durante il chunking.

---

## 31.4 Object-level ranking

Invece di classificare soltanto i chunk, si può aggregare il punteggio a livello di:

```text
programma
classe
function module
method
include
dynpro
```

Esempio:

```text
Chunk score
     ↓
Object score
     ↓
Object ranking
     ↓
Best chunks
```

---

## 31.5 Query expansion SAP

Una query:

```text
centro di costo dipendente
```

può essere espansa automaticamente in:

```text
KOSTL
PA0001
PERNR
```

Questo aumenterebbe notevolmente la capacità del sistema di rispondere a domande funzionali.

---

## 31.6 Hybrid reranker

In una fase successiva si può introdurre un reranker dedicato:

```text
Embedding retrieval
        ↓
Exact retrieval
        ↓
Structural ranking
        ↓
Cross-encoder / LLM reranker
```

Il reranker dovrebbe lavorare soltanto su un numero limitato di candidati per non aumentare eccessivamente la latenza.

---

# 32. Lezioni apprese

Le principali conclusioni tecniche raggiunte sono:

### 1. Il VectorDB funziona

Il problema non era il VectorDB.

---

### 2. Gli embedding funzionano

`nomic-embed-text` consente di ottenere una prima selezione semantica utile.

---

### 3. La ricerca esatta è indispensabile

Per SAP gli identificatori tecnici devono essere trattati come token ad alta precisione.

---

### 4. Il batch scan è necessario

Su decine di migliaia di chunk non è opportuno utilizzare una singola `collection.get()`.

---

### 5. La deduplicazione è fondamentale

Il corpus reale contiene duplicazioni dovute a versioni, esportazioni e copie dei programmi.

---

### 6. Il ranking deve conoscere ABAP

Un ranking generico non comprende la differenza tra:

```abap
DATA x LIKE pa0001-kostl.
```

e:

```abap
SELECT SINGLE kostl
  FROM pa0001
  WHERE pernr = ...
```

---

### 7. Il Query Profile è un livello superiore

Il sistema deve capire cosa vuole l'utente prima di decidere quali chunk sono pertinenti.

---

### 8. Retrieval e generation devono essere ottimizzati separatamente

Il retrieval può essere eccellente e la generazione comunque fallire per un context window troppo piccolo.

---

# 33. Stato del progetto

Al momento della documentazione il progetto ha raggiunto queste componenti:

```text
[OK] VectorDB ChromaDB
[OK] Embedding Ollama
[OK] Semantic retrieval
[OK] Exact SAP retrieval
[OK] Batch scan Chroma
[OK] Metadata deduplication
[OK] Content deduplication
[OK] SAP structural ranking
[OK] Query profiling — in evoluzione
[OK] Intent detection — in evoluzione
[OK] Context size monitoring
[OK] Qwen via Ollama
```

La prossima fase deve concentrarsi sulla stabilizzazione del Query Profile e sul miglioramento della selezione del contesto prima dell'invio a Qwen.

---

# 34. Versionamento consigliato

Per mantenere tracciabile l'evoluzione:

```text
04_test_rag.py
    │
    ├── V2
    │    └── semantic + exact
    │
    ├── V3
    │    └── structural ranking
    │
    ├── V4
    │    └── content deduplication
    │
    └── V5
         └── query profiling
              └── V5.1
                   └── table/field classification fix
```

È consigliabile non modificare il VectorDB durante queste fasi sperimentali.

Il retrieval layer, il ranking layer e il generation layer devono essere evoluti indipendentemente.

---

# 35. Conclusione

Il progetto è passato da un semplice:

```text
Query → Embedding → Chroma → LLM
```

a un'architettura RAG specializzata:

```text
Query
  ↓
Query Profile
  ↓
Semantic Retrieval
  +
Exact SAP Retrieval
  ↓
Merge
  ↓
Metadata Deduplication
  ↓
Content Deduplication
  ↓
SAP Structural Analysis
  ↓
Hybrid Ranking
  ↓
Context Selection
  ↓
Qwen Coder ABAP
  ↓
ABAP Response
```

Questo approccio è generalizzabile non solo a SAP ABAP/HCM, ma a qualunque corpus di codice tecnico in cui gli identificatori hanno un significato preciso.

La lezione architetturale principale è che un RAG specializzato non deve limitarsi a recuperare testo semanticamente simile: deve comprendere la struttura del dominio, gli identificatori tecnici, la sintassi del linguaggio e il contesto necessario alla generazione.

Per SAP ABAP, il livello di conoscenza del dominio — tabelle, campi, infotype, strutture Pxxxx, oggetti Repository e pattern sintattici ABAP — è quindi parte integrante del motore di retrieval e ranking.
