# RAG-ABAP-HCM
RAG ABAP HCM
 Documentazione tecnica — RAG ibrido per SAP ABAP/HCM con ChromaDB, Ollama e Qwen

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
