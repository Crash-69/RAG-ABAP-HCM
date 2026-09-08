from pathlib import Path
import time

from langchain_ollama import ChatOllama

from rag_retriever import retrieve


# ============================================================
# CONFIGURAZIONE
# ============================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

QWEN_MODEL = "oisee/qwen-coder-abap:v7"

# Numero massimo di chunk che verranno effettivamente
# inviati al modello.
RAG_TOP_K = 4

# Limite massimo del contesto inviato a Qwen.
MAX_CONTEXT_CHARS = 7000


# ============================================================
# COSTRUZIONE CONTESTO
# ============================================================

def build_context(results):

    chunks = []

    total_chars = 0

    for index, result in enumerate(results[:RAG_TOP_K], start=1):

        metadata = result.get("metadata", {})

        source_file = metadata.get(
            "source_file",
            "N/D"
        )

        object_type = metadata.get(
            "object_type",
            "N/D"
        )

        object_name = metadata.get(
            "object_name",
            "N/D"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "N/D"
        )

        final_score = result.get(
            "final_score",
            0
        )

        semantic_score = result.get(
            "semantic_score",
            0
        )

        exact_score = result.get(
            "exact_score",
            0
        )

        found_terms = result.get(
            "found_terms",
            []
        )

        document = result.get(
            "document",
            ""
        )

        chunk = f"""
[CHUNK {index}]

source_file: {source_file}
object_type: {object_type}
object_name: {object_name}
chunk_index: {chunk_index}

semantic_score: {semantic_score:.4f}
exact_score: {exact_score:.4f}
final_score: {final_score:.4f}

found_terms: {", ".join(found_terms)}

CONTENT:
{document}

--------------------------------------------------
"""

        # ----------------------------------------------------
        # Controllo dimensione contesto
        # ----------------------------------------------------

        if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
            break

        chunks.append(chunk)

        total_chars += len(chunk)

    return "\n".join(chunks)


# ============================================================
# PROMPT QWEN
# ============================================================

def build_prompt(query, context):

    prompt = f"""
SEI UN ESPERTO SAP ABAP/HCM.

Devi rispondere alla domanda dell'utente utilizzando
prioritariamente il contesto recuperato dal sistema RAG.

============================================================
DOMANDA UTENTE
============================================================

{query}

============================================================
CONTESTO RECUPERATO DAL RAG
============================================================

{context}

============================================================
ISTRUZIONI
============================================================

1. Usa prioritariamente il contesto recuperato dal RAG.

2. Non inventare:
   - tabelle SAP
   - strutture
   - campi
   - classi
   - function module
   - API
   - metodi
   - transazioni
   - sintassi ABAP

3. Se una informazione non è presente nel contesto,
   dichiaralo esplicitamente.

4. Quando proponi codice ABAP:
   - utilizza sintassi ABAP valida;
   - mantieni il codice coerente con SAP HCM;
   - non introdurre costrutti non necessari.

5. Se il contesto contiene codice ABAP pertinente,
   usalo come riferimento tecnico.

6. Se esistono più possibilità, indica quella
   più coerente con il contesto recuperato.

7. La risposta deve essere tecnica ma concisa.

8. Quando utile, struttura la risposta in:
   - spiegazione
   - codice ABAP
   - note tecniche

============================================================
RISPOSTA
============================================================
"""

    return prompt


# ============================================================
# QWEN
# ============================================================

def ask_qwen(prompt):

    llm = ChatOllama(
        model=QWEN_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# VISUALIZZAZIONE RISULTATI RAG
# ============================================================

def print_rag_results(results):

    print()
    print("=" * 70)
    print("RISULTATI RAG INVIATI A QWEN")
    print("=" * 70)

    for index, result in enumerate(
        results[:RAG_TOP_K],
        start=1
    ):

        metadata = result.get("metadata", {})

        print()
        print("-" * 70)
        print(f"CHUNK {index}")
        print("-" * 70)

        print(
            "SOURCE      :",
            metadata.get("source_file", "N/D")
        )

        print(
            "OBJECT TYPE :",
            metadata.get("object_type", "N/D")
        )

        print(
            "OBJECT NAME :",
            metadata.get("object_name", "N/D")
        )

        print(
            "CHUNK INDEX :",
            metadata.get("chunk_index", "N/D")
        )

        print(
            "SEMANTIC    :",
            f"{result.get('semantic_score', 0):.4f}"
        )

        print(
            "EXACT       :",
            f"{result.get('exact_score', 0):.4f}"
        )

        print(
            "FINAL       :",
            f"{result.get('final_score', 0):.4f}"
        )

        print(
            "TERMS       :",
            ", ".join(result.get("found_terms", []))
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAG + QWEN ABAP/HCM")
    print("=" * 70)

    query = input(
        "\nInserisci la domanda ABAP/HCM:\n> "
    ).strip()

    if not query:
        print("Query vuota.")
        return

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("1. RETRIEVAL")
    print("=" * 70)

    start_retrieval = time.perf_counter()

    results = retrieve(query)

    retrieval_time = (
        time.perf_counter()
        - start_retrieval
    )

    print(
        f"\nRisultati recuperati: {len(results)}"
    )

    print(
        f"Tempo retrieval: {retrieval_time:.2f} secondi"
    )

    # --------------------------------------------------------
    # DIAGNOSTICA
    # --------------------------------------------------------

    print_rag_results(results)

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = build_context(results)

    print()
    print("=" * 70)
    print("2. CONTEXT INVIATO A QWEN")
    print("=" * 70)

    print(context)

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = build_prompt(
        query,
        context
    )

    print()
    print("=" * 70)
    print("3. PROMPT")
    print("=" * 70)

    print(prompt)

    # --------------------------------------------------------
    # QWEN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("4. QWEN")
    print("=" * 70)

    print(
        f"Modello: {QWEN_MODEL}"
    )

    print(
        "\nElaborazione in corso...\n"
    )

    start_qwen = time.perf_counter()

    try:

        answer = ask_qwen(prompt)

    except Exception as e:

        print()
        print("=" * 70)
        print("ERRORE QWEN")
        print("=" * 70)

        print(type(e).__name__)
        print(str(e))

        return

    qwen_time = (
        time.perf_counter()
        - start_qwen
    )

    # --------------------------------------------------------
    # RISPOSTA
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("5. RISPOSTA QWEN")
    print("=" * 70)

    print(answer)

    print()
    print("=" * 70)
    print("TEMPI")
    print("=" * 70)

    print(
        f"Retrieval : {retrieval_time:.2f} sec"
    )

    print(
        f"Qwen      : {qwen_time:.2f} sec"
    )

    print(
        f"Totale    : "
        f"{retrieval_time + qwen_time:.2f} sec"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

