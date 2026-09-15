# Soluzione RAG locale per SAP ABAP/HCM con Retrieval Relationship-Aware e Loop Engineering

## 1. Obiettivo della soluzione

Questa soluzione realizza un sistema locale di **Retrieval-Augmented Generation (RAG)** specializzato in **SAP ABAP/HCM**, con l'obiettivo di fornire a un LLM codice e contesto tecnico SAP pertinenti prima della generazione.
Tutti i progetti qui presenti sono nati per scopo di studio e di ricerca personale sull'Intelligenza Artificiale e non sono a scopo di lucro .

Il principio fondamentale è:

> **Il modello generativo non deve essere interrogato direttamente sul problema ABAP: deve prima ricevere contesto recuperato da una base documentale SAP verificata.**

La soluzione separa quindi chiaramente:

1. acquisizione e indicizzazione della conoscenza ABAP;
2. retrieval dei contenuti pertinenti;
3. valutazione della qualità del retrieval;
4. costruzione del contesto per il modello;
5. generazione del codice;
6. validazione del codice;
7. eventuale iterazione del processo.

L'architettura è pensata per funzionare **localmente**, senza rendere necessario inviare il patrimonio documentale ABAP/HCM a servizi cloud esterni.

---

## 2. Architettura generale

L'architettura logica prevista è:

```text
                         ┌─────────────────────┐
                         │       UTENTE        │
                         │ richiesta ABAP/HCM  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Query / Intent      │
                         │ analisi tecnica     │
                         │ termini SAP        │
                         └──────────┬──────────┘
                                    │
                                    ▼
              ┌────────────────────────────────────────┐
              │          RETRIEVAL V5.9 ANTI            │
              │                                        │
              │ • ricerca semantica                    │
              │ • ricerca esatta termini SAP           │
              │ • ranking strutturale                  │
              │ • relazione SQL                        │
              │ • ranking per intent                   │
              │ • deduplicazione/source-family         │
              └────────────────────┬───────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │     TOP-K CONTEXT   │
                         │ risultati rilevanti │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ CONTEXT BUILDER     │
                         │ contesto ABAP       │
                         │ verificato          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ QWEN-CODER-ABAP     │
                         │ oisee/...:v7        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ ABAP VALIDATOR      │
                         │ controlli tecnici   │
                         └──────────┬──────────┘
                                    │
                              PASS / FAIL
                               ┌────┴────┐
                               │         │
                              PASS      FAIL
                               │         │
                               ▼         ▼
                           RISPOSTA    LOOP
                                      ENGINEERING
```

La componente di retrieval deve rimanere indipendente dal modello generativo. Questo permette di determinare se un eventuale errore deriva dal **contesto recuperato** oppure dal **modello LLM**.

---

# 3. Componenti della soluzione

## 3.1 Repository sorgente ABAP/HCM

Il patrimonio documentale utilizzato dal sistema è costituito da sorgenti e documentazione ABAP/HCM.

Directory attuale:

```text
C:\Progetto_AI\Abap
```

I formati attualmente considerati dall'ingestion sono:

```text
.txt
.html
.htm
```

I contenuti possono rappresentare, tra gli altri:

- programmi ABAP;
- function group;
- function module;
- classi;
- report;
- riferimenti DDIC;
- esempi di utilizzo delle tabelle SAP;
- codice contenente SELECT e relazioni tra tabelle e campi.

---

# 4. Ingestion e VectorDB

La fase di ingestion è stata completata e non deve essere ripetuta durante la fase corrente di tuning del retrieval.

La configurazione utilizzata comprende:

```text
CHUNK_SIZE = 1800
OVERLAP    = 250
```

La suddivisione tiene conto anche di terminatori strutturali ABAP come:

```text
ENDMETHOD.
ENDCLASS.
ENDFUNCTION.
ENDFORM.
ENDMODULE.
```

Questo consente di evitare, per quanto possibile, chunk che interrompano arbitrariamente una struttura ABAP.

## 4.1 Identificazione deterministica dei chunk

Gli identificativi dei chunk sono costruiti in maniera deterministica utilizzando informazioni come:

- hash del file;
- indice del chunk;
- hash del contenuto del chunk.

Questo permette di riconoscere in modo affidabile contenuti già indicizzati.

## 4.2 Manifest

Il manifest contiene informazioni utili a evitare elaborazioni inutili dei file già trattati.

Percorso:

```text
C:\Progetto_AI\Abap_RAG\manifest.json
```

## 4.3 Metadati

I chunk vengono associati a metadati quali:

```text
source path
file hash
chunk index
file type
language
object type
object name
```

Gli object type comprendono categorie come:

```text
FUNCTION_GROUP
FUNCTION
REPORT
CLASS
DDIC
```

---

# 5. Stato attuale del VectorDB

Il VectorDB attuale contiene:

```text
46.346 chunk
5.099 file unici indicizzati
6.819 candidati iniziali
1.719 duplicati ignorati
```

La fase di embedding è stata completata.

Il sistema utilizza:

```text
Chroma
collection: abap_hcm
```

VectorDB:

```text
C:\Progetto_AI\Abap_VectorDB
```

Modalità richiesta durante il tuning:

```text
READ-ONLY
```

## 5.1 Motivazione della modalità READ-ONLY

Durante il miglioramento del retrieval non è necessario modificare il VectorDB.

Questa separazione è importante perché permette di confrontare diverse versioni dell'algoritmo di ranking utilizzando esattamente lo stesso patrimonio indicizzato.

In questo modo:

```text
stesso VectorDB
        +
retrieval differente
        =
confronto attendibile
```

e non:

```text
VectorDB modificato
        +
retrieval modificato
        =
risultato non facilmente attribuibile
```

---

# 6. Embedding

Il modello utilizzato per l'embedding è:

```text
nomic-embed-text:latest
```

tramite Ollama.

Endpoint locale:

```text
http://127.0.0.1:11434
```

La configurazione locale è importante perché il sistema è stato progettato per mantenere il processo di retrieval nell'ambiente locale.

Il modello embedding è distinto dal modello generativo.

Questa separazione è architetturalmente corretta:

```text
Embedding model
      │
      ▼
Vector retrieval

Generative model
      │
      ▼
ABAP generation
```

---

# 7. Perché il semplice semantic search non è sufficiente

Le interrogazioni SAP ABAP/HCM contengono frequentemente identificativi tecnici molto specifici:

```text
PA0001
PA0002
PA0007
PA0008
PERNR
KOSTL
WERKS
ORGEH
BUKRS
BTRTL
PERSG
PERSK
PLANS
STELL
```

Una ricerca esclusivamente semantica può fallire perché termini tecnici come `PA0001` non hanno necessariamente una rappresentazione semantica sufficiente nello spazio vettoriale.

Per questo motivo il retrieval utilizza una strategia ibrida.

---

# 8. Retrieval V5.9 ANTI

La versione attuale validata è:

```text
05_rag_v5_9_anti.py
```

Il sistema è denominato:

```text
SAP ABAP RAG TEST - V5.9 ANTI
```

e utilizza:

```text
SQL RELATIONSHIP-AWARE RANKING
WITH TARGETED T12 & T10/T11 FIXES
```

Il retrieval combina più segnali.

---

# 9. Ricerca semantica

La ricerca semantica utilizza gli embedding per individuare contenuti concettualmente vicini alla domanda.

È particolarmente utile quando la richiesta non contiene direttamente il nome tecnico dell'oggetto.

Esempio:

```text
Come viene determinato il centro di costo del dipendente?
```

La query potrebbe non essere una semplice ricerca testuale di:

```text
KOSTL
PA0001
PERNR
```

Il semantic retrieval permette quindi di recuperare anche contenuti descrittivamente simili.

Tuttavia, per query fortemente tecniche, il semantic score può essere anche:

```text
0.000
```

senza che questo rappresenti necessariamente un errore.

Nel sistema V5.9 il semantic score è solo uno dei segnali utilizzati.

---

# 10. Ricerca esatta dei termini SAP

Accanto al semantic retrieval viene effettuata una ricerca esatta dei termini tecnici.

Esempio:

```text
PA0001
PERNR
KOSTL
WERKS
```

La presenza effettiva del termine nel chunk costituisce un segnale molto forte.

Questo è particolarmente importante per SAP perché nomi di tabelle e campi sono identificativi tecnici e non semplici parole naturali.

---

# 11. Structural ranking

Il ranking considera la struttura del codice.

Non è sufficiente trovare:

```text
PA0001
```

all'interno del documento.

È preferibile un risultato che mostri realmente un utilizzo ABAP della tabella, ad esempio:

```abap
SELECT ...
  FROM pa0001
  INTO ...
 WHERE pernr = ...
```

rispetto a un semplice testo:

```text
PA0001 è la tabella dei dati organizzativi...
```

Il ranking considera quindi segnali come:

- presenza di `SELECT`;
- presenza di `FROM`;
- presenza di `WHERE`;
- presenza della tabella;
- presenza del campo richiesto;
- relazione tra tabella e campo;
- presenza di codice ABAP.

---

# 12. SQL Relationship-Aware Ranking

Questa è una delle caratteristiche principali della soluzione.

Il sistema non valuta soltanto se un chunk contiene:

```text
PA0001
```

e:

```text
PERNR
```

ma cerca di determinare se i due elementi sono realmente collegati nella stessa operazione SQL.

La relazione desiderata può essere rappresentata concettualmente come:

```text
SELECT
   ...
FROM PA0001
WHERE PERNR ...
```

Questo è molto più informativo di due semplici occorrenze indipendenti.

## 12.1 Esempio

Una fonte contenente:

```abap
SELECT SINGLE *
  FROM pa0001
 WHERE pernr = @lv_pernr.
```

è molto più rilevante per:

```text
leggere PA0001 per PERNR
```

rispetto a una fonte che contiene semplicemente le parole:

```text
PA0001
PERNR
```

in sezioni differenti.

---

# 13. Relation types

Il ranking utilizza relazioni semantico-strutturali tra oggetti SAP.

Tra le relazioni utilizzate nei test sono presenti categorie come:

```text
SELECT_PA0001_WHERE_PERNR
SELECT_KOSTL_FROM_PA0001_WHERE_PERNR
SELECT_FIELDS_WHERE
SELECT_FIELD_WHERE
PA0001_ORG_FIELDS_PERNR
PA0001_ORG_FIELDS
TABLE_FROM
```

Queste relazioni permettono di distinguere:

```text
presenza del termine
```

da:

```text
utilizzo tecnico del termine
```

---

# 14. Intent-aware ranking

La query viene classificata in base all'intento.

Esempi di intent utilizzati nei test:

```text
table_reference
field_lookup
code_lookup
code_pattern
functional_lookup
```

Il ranking cambia in funzione dell'intento.

## 14.1 table_reference

Se l'utente chiede informazioni su una tabella, viene privilegiato un contenuto che dimostri un utilizzo effettivo della tabella.

## 14.2 field_lookup

Se viene richiesto un campo, viene privilegiata la relazione:

```text
FIELD → TABLE
```

e, quando disponibile, l'utilizzo del campo nel codice.

## 14.3 code_lookup / code_pattern

Per richieste come:

```text
SELECT PA0001 WHERE PERNR
```

la priorità è data a codice ABAP reale che implementa la relazione richiesta.

## 14.4 functional_lookup

Per richieste funzionali, come:

```text
centro di costo del dipendente
```

il ranking deve collegare concetti funzionali e oggetti tecnici:

```text
dipendente
   │
   ▼
PERNR
   │
   ▼
PA0001
   │
   ▼
KOSTL
```

---

# 15. Ranking specifico PA0001 + PERNR

Uno dei casi più importanti è:

```text
PA0001 + PERNR
```

Il ranking V5.9 assegna un bonus elevato quando trova una relazione SQL coerente.

In particolare viene premiato:

```text
PA0001
+
WHERE PERNR
```

mentre viene penalizzato un risultato che contiene PA0001 senza una relazione con PERNR.

Questo evita che una generica occorrenza di PA0001 domini il risultato quando l'utente ha chiesto esplicitamente il legame con PERNR.

---

# 16. Ranking per richieste organizzative

Per query come:

```text
dati organizzativi dipendente SAP HCM
```

il sistema considera i principali campi organizzativi di PA0001:

```text
BUKRS
WERKS
BTRTL
PERSG
PERSK
KOSTL
ORGEH
PLANS
STELL
```

oltre a:

```text
PERNR
PA0001
```

Il ranking premia quindi i chunk che mostrano un numero elevato di campi organizzativi realmente utilizzati.

Questo ha permesso di risolvere il problema precedentemente osservato su T12.

---

# 17. Deduplicazione

Il VectorDB contiene un numero significativo di contenuti duplicati o quasi duplicati.

Per evitare che lo stesso programma, presente con suffissi differenti, occupi molti risultati finali, V5.9 utilizza il concetto di:

```text
source_family
```

La normalizzazione riconosce suffissi come:

```text
_bk
_backup
_copia
_copy
_old
_new
_p
_spN
_vN
```

e alcune varianti numeriche.

L'obiettivo è consentire al ranking di rappresentare più sorgenti differenti invece di restituire molte copie della stessa sorgente.

---

# 18. Importanza della source diversity

La deduplicazione non significa eliminare fisicamente i duplicati dal VectorDB.

Significa invece impedire che, nella lista finale:

```text
TOP-K
```

siano presenti molte varianti della stessa famiglia sorgente.

Questo è particolarmente importante per il modello generativo.

Un contesto costituito da:

```text
5 copie quasi identiche
```

è meno utile di:

```text
5 esempi tecnicamente differenti
```

che mostrano la stessa relazione SAP in contesti diversi.

---

# 19. Test V5.9 ANTI

La suite attuale contiene:

```text
12 test
```

Risultati principali:

```text
Promising:       12/12
Top relation:    12/12
Mean coverage:   83,33%
```

Questo rappresenta un risultato positivo per lo smoke test.

È però importante distinguere:

```text
test superato
```

da:

```text
retrieval pronto per produzione
```

I 12 test costituiscono una baseline, non una prova definitiva di robustezza.

---

# 20. T12: caso particolarmente significativo

La query:

```text
dati organizzativi dipendente SAP HCM
```

richiede:

```text
PA0001
PERNR
BUKRS
WERKS
BTRTL
PERSG
PERSK
KOSTL
ORGEH
PLANS
STELL
```

Il miglior risultato ottenuto è:

```text
zse_read_occupational_data.html
chunk 0
```

con relazione:

```text
PA0001_ORG_FIELDS_PERNR
```

e una SELECT che recupera numerosi campi organizzativi da PA0001 con filtro PERNR.

Il risultato ha ottenuto:

```text
coverage        100%
table           15/15
ABAP            13/15
structural      15/15
SELECT/FROM     15/15
relation hit    15/15
top relation    PASS
```

Questo dimostra che il ranking riesce ora a privilegiare un esempio realmente pertinente alla richiesta funzionale.

---

# 21. Problemi ancora presenti

La soluzione V5.9 è positiva, ma non deve essere considerata definitiva.

## 21.1 T10/T11

Le query generiche:

```text
leggere PA0001 per PERNR
SELECT PA0001 WHERE PERNR
```

possono restituire al primo posto un esempio più specifico, ad esempio:

```text
SELECT_KOSTL_FROM_PA0001_WHERE_PERNR
```

Il risultato è tecnicamente pertinente, ma potrebbe non essere il miglior esempio minimale per una richiesta generica.

Il problema è quindi di:

```text
specificità del ranking
```

non necessariamente di:

```text
correttezza del retrieval
```

Non è opportuno correggerlo con un'altra modifica aggressiva al ranking senza prima verificare gli effetti sugli altri test.

---

# 22. Duplicati ancora elevati

Nonostante la source-family diversity, i contatori di duplicati rimangono elevati.

Esempi:

```text
T12 ≈ 7.303
T09 ≈ 5.830
```

Questo non implica che il VectorDB debba essere ricostruito.

Durante la fase attuale è preferibile lasciare il VectorDB invariato e intervenire sull'algoritmo di selezione finale.

Una eventuale pulizia fisica del VectorDB può essere valutata successivamente, come attività separata.

---

# 23. Perché non collegare subito Qwen

Un principio fondamentale della soluzione è:

> **Prima validare il retrieval, poi valutare il modello generativo.**

Se Qwen produce codice errato dopo aver ricevuto un contesto errato, non è possibile stabilire quale componente sia responsabile del problema.

La sequenza corretta è:

```text
Query
  ↓
Retrieval
  ↓
Context validation
  ↓
Qwen
  ↓
ABAP validation
```

Non:

```text
Query
  ↓
Qwen
  ↓
tentativo di capire perché il risultato è errato
```

---

# 24. Context Builder

Il prossimo componente logico è il:

```text
Context Builder
```

Il Context Builder deve trasformare i risultati del retrieval in un contesto controllato per il modello.

Non dovrebbe limitarsi a concatenare indiscriminatamente i chunk.

Dovrebbe includere, per ogni risultato:

```text
source
chunk
object type
language
relation type
relation score
retrieval score
reason
content
```

Il contesto dovrebbe essere ordinato in base alla rilevanza.

---

# 25. Context Quality Test

Prima di collegare Qwen è necessario introdurre un test specifico del contesto.

Il test dovrebbe verificare almeno:

```text
PA0001 presente?
PERNR presente?
SELECT presente?
FROM presente?
WHERE PERNR presente?
campo richiesto presente?
relazione SQL presente?
codice ABAP presente?
```

Per richieste multi-field:

```text
numero campi richiesti
numero campi trovati
numero campi trovati nella stessa SELECT
```

Questa distinzione è fondamentale.

Non è sufficiente sapere che un campo esiste da qualche parte nei TOP-K.

È molto più importante sapere se il campo richiesto compare nella stessa istruzione SQL del contesto selezionato.

---

# 26. Esempio di Context Quality

Per:

```text
leggere PA0001 per PERNR
```

un contesto di qualità elevata dovrebbe contenere qualcosa come:

```abap
SELECT ...
  FROM pa0001
  WHERE pernr = ...
```

Un contesto di qualità bassa potrebbe invece contenere:

```text
PA0001
```

in una descrizione DDIC e:

```text
PERNR
```

in un'altra sorgente non correlata.

Il secondo caso può ottenere un buon punteggio semantico/lessicale ma non è sufficiente per generare codice affidabile.

---

# 27. Test adversarial

Dopo i 12 smoke test è necessario creare una suite più ampia.

Obiettivo iniziale:

```text
30–50 query
```

Le query dovrebbero comprendere:

### Positive

Richieste con oggetti SAP espliciti:

```text
PA0001
PA0002
PA0007
PA0008
PERNR
KOSTL
WERKS
ORGEH
```

### Negative

Richieste per cui il sistema non dovrebbe inventare una relazione.

### Sinonimi

Esempi:

```text
numero dipendente
matricola
personnel number
centro di costo
unità organizzativa
stabilimento
società
```

### ABAP

Esempi:

```text
SELECT PA0001 WHERE PERNR
leggere PA0001
recuperare KOSTL da PA0001
```

### Funzionali

Esempi:

```text
dove trovo il centro di costo del dipendente?
quali sono i dati organizzativi del dipendente?
```

### Ambigue

Query volutamente poco specifiche per verificare la robustezza del ranking.

### Multi-field

Richieste contenenti molti campi contemporaneamente.

### Multi-table

Richieste che richiedono la relazione tra più tabelle SAP.

---

# 28. Separazione Retrieval / Evaluation

L'architettura dovrebbe essere ulteriormente separata in due livelli.

## Retrieval

Responsabile esclusivamente di:

```text
query
→ risultati ordinati
```

## Evaluation

Responsabile di:

```text
risultati
→ metriche
→ PASS/FAIL
```

Concettualmente:

```python
results = retrieve(query, top_k=5)
```

e separatamente:

```python
metrics = evaluate(query, results)
```

Questa separazione rende il sistema più facile da mantenere e testare.

---

# 29. Pipeline definitiva prevista

La pipeline completa diventa:

```text
USER
 │
 ▼
QUERY ANALYSIS
 │
 ├── intent
 ├── SAP terms
 ├── tables
 ├── fields
 └── functional concepts
 │
 ▼
V5.9 ANTI RETRIEVAL
 │
 ├── semantic search
 ├── exact search
 ├── structural analysis
 ├── SQL relationship
 ├── intent-aware ranking
 └── source-family diversity
 │
 ▼
TOP-K
 │
 ▼
CONTEXT QUALITY
 │
 ├── table check
 ├── field check
 ├── SELECT/FROM check
 ├── WHERE check
 └── relationship check
 │
 ▼
CONTEXT BUILDER
 │
 ▼
QWEN-CODER-ABAP
 │
 ▼
ABAP VALIDATOR
 │
 ├── syntax/structure
 ├── DDIC verification
 ├── authorization considerations
 ├── Clean ABAP
 └── requested behavior
 │
 ├──────── PASS ────────► FINAL ANSWER
 │
 └──────── FAIL
              │
              ▼
       LOOP ENGINEERING
              │
              ▼
          RETRIEVAL /
        PROMPT REVISION
              │
              ▼
             QWEN
```

---

# 30. Loop Engineering

Il Loop Engineering rappresenta il livello successivo della soluzione.

Il concetto è:

```text
Analisi
   ↓
Generazione
   ↓
Verifica
   ↓
Ottimizzazione
   ↓
Nuova verifica
```

Per ABAP il ciclo può diventare:

```text
Functional Requirement
        ↓
Technical Analysis
        ↓
RAG Retrieval
        ↓
ABAP Generation
        ↓
ABAP Validation
        ↓
PASS / FAIL
        ↓
Correction
        ↓
Re-validation
```

L'obiettivo non è chiedere al modello di "provare" semplicemente a scrivere codice, ma costruire un processo verificabile.

---

# 31. Validazione ABAP

Il validator futuro dovrà verificare, per quanto possibile:

- esistenza degli oggetti DDIC;
- esistenza delle tabelle;
- esistenza dei campi;
- correttezza delle relazioni;
- utilizzo coerente di `SELECT`;
- presenza dei filtri richiesti;
- assenza di `SELECT *` quando non necessario;
- rispetto delle convenzioni Clean ABAP;
- gestione corretta di `PERNR`;
- eventuali aspetti autorizzativi HCM;
- compatibilità con SAP S/4HANA;
- correttezza della soluzione rispetto alla richiesta funzionale.

Il validator deve utilizzare esclusivamente conoscenza SAP verificata quando il requisito richiede una verifica oggettiva.

---

# 32. RAG come sistema di grounding

La funzione del RAG non è semplicemente "dare informazioni al modello".

Il RAG deve fornire un **grounding tecnico**.

In un sistema SAP questo è particolarmente importante perché il modello potrebbe altrimenti generare:

```text
tabelle inesistenti
campi inesistenti
API inesistenti
funzioni non disponibili
relazioni HCM errate
```

Il retrieval deve quindi ridurre la probabilità di allucinazione tecnica.

---

# 33. Principio "verified knowledge"

Per il progetto ABAP/HCM è consigliabile adottare una regola forte:

> Se un oggetto SAP non è presente nella conoscenza verificata, il sistema non deve inventarlo.

Concettualmente:

```text
Oggetto presente nel contesto verificato
            │
           YES
            │
            ▼
       utilizzabile

            NO
            │
            ▼
       NON ASSUMERE
```

Questo principio diventa particolarmente importante nel validator e nel futuro RAG con RAG-as-constraint.

---

# 34. Perché V5.9 ANTI costituisce una buona baseline

La versione V5.9 ANTI ha raggiunto un livello sufficiente per essere congelata come baseline perché:

- il VectorDB è stabile;
- il retrieval esatto funziona;
- il semantic retrieval rimane disponibile;
- le relazioni SQL vengono riconosciute;
- il ranking è consapevole dell'intento;
- T10/T11 sono stati corretti;
- T12 è stato significativamente migliorato;
- la source-family diversity limita la ridondanza;
- 12/12 test risultano promettenti;
- 12/12 hanno una top relation PASS.

A questo punto ulteriori modifiche al ranking dovrebbero essere guidate da casi di test concreti, non da ottimizzazioni teoriche.

---

# 35. Strategia di sviluppo consigliata

## Fase 1 — Freeze

Congelare:

```text
05_rag_v5_9_anti.py
```

come baseline.

Creare una copia identificata, ad esempio:

```text
05_rag_v5_9_baseline.py
```

## Fase 2 — Context inspection

Creare:

```text
05_rag_v5_9_context_test.py
```

per visualizzare il contenuto reale dei TOP-K.

## Fase 3 — Adversarial testing

Creare una suite:

```text
30–50 query
```

con categorie differenti.

## Fase 4 — Context Builder

Implementare la trasformazione:

```text
retrieval results
        ↓
verified context
```

## Fase 5 — Qwen

Collegare:

```text
oisee/qwen-coder-abap:v7
```

solo dopo aver verificato la qualità del contesto.

## Fase 6 — End-to-end

Eseguire:

```text
5–10 richieste ABAP reali
```

## Fase 7 — Validator

Introdurre:

```text
ABAP Validator
```

## Fase 8 — Loop Engineering

Implementare:

```text
Generate
→ Validate
→ Fail
→ Correct
→ Validate
```

## Fase 9 — V6

Solo dopo queste fasi valutare una versione V6 del retrieval.

---

# 36. Architettura software futura

Una struttura possibile è:

```text
C:\Progetto_AI
│
├── Abap\
│
├── Abap_VectorDB\
│
├── Abap_RAG\
│   └── manifest.json
│
├── retrieval\
│   ├── v5_9_anti.py
│   ├── context_builder.py
│   ├── context_quality.py
│   └── evaluator.py
│
├── tests\
│   ├── smoke_tests.py
│   ├── adversarial_tests.py
│   └── context_tests.py
│
├── generation\
│   └── qwen_abap.py
│
├── validation\
│   └── abap_validator.py
│
└── loop\
    └── engineering_loop.py
```

La struttura è concettuale e non implica modifiche immediate all'attuale progetto.

---

# 37. Criterio di successo

Il progetto non deve essere valutato esclusivamente sulla base della metrica:

```text
retrieval score
```

Un retrieval realmente utile deve soddisfare contemporaneamente:

```text
1. rilevanza semantica
2. corrispondenza esatta SAP
3. relazione strutturale
4. relazione SQL
5. qualità del codice
6. diversità delle fonti
7. completezza del contesto
8. verificabilità degli oggetti
```

Il vero KPI finale è:

> **Il modello riceve abbastanza contesto SAP corretto e specifico da poter generare codice ABAP verificabile senza inventare oggetti tecnici?**

---

# 38. Stato del progetto

| Componente | Stato |
|---|---|
| Sorgenti ABAP/HCM | ✅ disponibili |
| Ingestion | ✅ completata |
| Embedding | ✅ completato |
| VectorDB | ✅ stabile |
| VectorDB READ-ONLY | ✅ |
| Exact SAP search | ✅ |
| Semantic search | ✅ |
| Structural ranking | ✅ |
| SQL relationship-aware | ✅ |
| Intent-aware ranking | ✅ |
| Source-family diversity | ✅ |
| Smoke test 12 query | ✅ 12/12 |
| T10/T11 | 🟡 migliorati, da verificare qualitativamente |
| T12 | ✅ risolto nello smoke test |
| Context inspection | 🔜 prossimo step |
| Adversarial test 30–50 query | 🔜 |
| Context Builder | 🔜 |
| Qwen integration | 🔜 |
| ABAP Validator | 🔜 |
| Loop Engineering | 🔜 |
| V6 retrieval | ⏸ dopo validazione |

---

# 39. Conclusione

La soluzione attuale rappresenta un'architettura RAG specializzata per SAP ABAP/HCM nella quale il retrieval non viene trattato come una semplice ricerca vettoriale.

Il punto qualificante è la combinazione di:

```text
Semantic Search
       +
Exact SAP Terms
       +
Structural Analysis
       +
SQL Relationship
       +
Intent-Aware Ranking
       +
Source Diversity
```

La V5.9 ANTI costituisce una buona baseline perché i test attuali mostrano una capacità consistente di recuperare esempi ABAP realmente correlati alle richieste.

Il prossimo obiettivo non dovrebbe essere una nuova modifica del VectorDB né un'ulteriore modifica del ranking a priori.

Il passo corretto è verificare **il contenuto effettivamente consegnato al modello**.

La sequenza di sviluppo consigliata è quindi:

```text
V5.9 ANTI
   ↓
Context Inspection
   ↓
Adversarial Testing
   ↓
Context Builder
   ↓
Qwen-Coder-ABAP
   ↓
ABAP Validator
   ↓
Loop Engineering
   ↓
V6
```

In questo modo il progetto mantiene una separazione netta tra:

```text
RETRIEVAL
GENERAZIONE
VALIDAZIONE
```

e permette di identificare con precisione la causa degli eventuali errori.

Il risultato finale atteso è un **agente locale specializzato in SAP ABAP/HCM**, capace di utilizzare conoscenza tecnica verificata, generare codice contestualizzato e sottoporlo a un ciclo iterativo di verifica e correzione.
