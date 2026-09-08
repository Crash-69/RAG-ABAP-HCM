import time
from pathlib import Path

from langchain_ollama import ChatOllama

from rag_retriever import retrieve


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(r"C:\Progetto_AI")

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

QWEN_MODEL = "oisee/qwen-coder-abap:v7"

# Numero di chunk recuperati dal RAG e passati a Qwen
RAG_TOP_K = 4

# Context massimo in caratteri.
#
# Il modello segnala un context window di 4096 token.
# Manteniamo un margine per prompt + domanda + risposta.
MAX_CONTEXT_CHARS = 7000

# Stima prudenziale:
# 1 token ~= 4 caratteri in media.
TOKEN_ESTIMATE_DIVISOR = 4


# ============================================================
# STIMA TOKEN
# ============================================================

def estimate_tokens(text):
    """
    Stima approssimativa del numero di token.

    Non sostituisce il tokenizer reale del modello.
    Serve esclusivamente per diagnostica e per evitare
    di avvicinarci troppo al limite di 4096 token.
    """

    if not text:
        return 0

    return max(
        1,
        len(text) // TOKEN_ESTIMATE_DIVISOR
    )


# ============================================================
# COSTRUZIONE DEL CONTEXTO
# ============================================================

def build_context(results):

    chunks = []

    total_chars = 0

    for index, result in enumerate(
        results[:RAG_TOP_K],
        start=1
    ):

        metadata = result.get(
            "metadata",
            {}
        )

        source_file = metadata.get(
            "source_file",
            "N/D"
        )

        source_path = metadata.get(
            "source_path",
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

        semantic_score = result.get(
            "semantic_score",
            0.0
        )

        exact_score = result.get(
            "exact_score",
            0.0
        )

        final_score = result.get(
            "final_score",
            0.0
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
source_file={source_file}
source_path={source_path}
object_type={object_type}
object_name={object_name}
chunk_index={chunk_index}
semantic_score={semantic_score:.4f}
exact_score={exact_score:.4f}
final_score={final_score:.4f}
terms={",".join(found_terms)}

{document}

[END CHUNK {index}]
"""

        # ----------------------------------------------------
        # Controllo limite contesto
        # ----------------------------------------------------

        if (
            total_chars + len(chunk)
            > MAX_CONTEXT_CHARS
        ):
            break

        chunks.append(chunk)

        total_chars += len(chunk)

    return "\n".join(chunks)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(query, context):

    return f"""Sei un esperto SAP ABAP/HCM.

Rispondi alla domanda usando prioritariamente il contesto RAG.

DOMANDA:
{query}

CONTESTO RAG:
{context}

REGOLE:
- Usa il contesto come principale fonte tecnica.
- Non inventare tabelle, campi, API, classi o Function Module.
- Fornisci codice ABAP valido e coerente con SAP HCM.
- Considera la validità temporale degli infotype quando pertinente.
- Se il contesto non è sufficiente, dichiaralo.
- Spiega brevemente la soluzione.
- Se proponi codice, mostra prima il codice e poi le note tecniche.

RISPOSTA:
"""


# ============================================================
# CHIAMATA QWEN
# ============================================================

def ask_qwen(prompt):

    llm = ChatOllama(
        model=QWEN_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )

    start = time.perf_counter()

    response = llm.invoke(prompt)

    elapsed = time.perf_counter() - start

    return response, elapsed


# ============================================================
# DIAGNOSTICA RAG
# ============================================================

def print_retrieval_results(results):

    print()
    print("=" * 70)
    print("CHUNK RECUPERATI DAL RAG")
    print("=" * 70)

    for index, result in enumerate(
        results[:RAG_TOP_K],
        start=1
    ):

        metadata = result.get(
            "metadata",
            {}
        )

        print()
        print("-" * 70)
        print(f"CHUNK #{index}")
        print("-" * 70)

        print(
            "Source      :",
            metadata.get(
                "source_file",
                "N/D"
            )
        )

        print(
            "Object type :",
            metadata.get(
                "object_type",
                "N/D"
            )
        )

        print(
            "Object name :",
            metadata.get(
                "object_name",
                "N/D"
            )
        )

        print(
            "Chunk index :",
            metadata.get(
                "chunk_index",
                "N/D"
            )
        )

        print(
            "Semantic    :",
            f"{result.get('semantic_score', 0):.4f}"
        )

        print(
            "Exact       :",
            f"{result.get('exact_score', 0):.4f}"
        )

        print(
            "Final       :",
            f"{result.get('final_score', 0):.4f}"
        )

        print(
            "SAP terms   :",
            ", ".join(
                result.get(
                    "found_terms",
                    []
                )
            )
        )


# ============================================================
# METRICHE QWEN
# ============================================================

def print_qwen_metrics(response, elapsed):

    print()
    print("=" * 70)
    print("METRICHE QWEN")
    print("=" * 70)

    print(
        f"Tempo generazione : {elapsed:.2f} sec"
    )

    # ChatOllama normalmente espone usage_metadata
    # nelle versioni recenti di LangChain.
    usage = getattr(
        response,
        "usage_metadata",
        None
    )

    if usage:

        print(
            "Input tokens      :",
            usage.get(
                "input_tokens",
                "N/D"
            )
        )

        print(
            "Output tokens     :",
            usage.get(
                "output_tokens",
                "N/D"
            )
        )

        print(
            "Total tokens      :",
            usage.get(
                "total_tokens",
                "N/D"
            )
        )

    else:

        print(
            "Token usage       : non disponibile"
        )

        # Alcune versioni possono esporre metadata
        # alternativamente.
        response_metadata = getattr(
            response,
            "response_metadata",
            {}
        )

        if response_metadata:

            print(
                "Response metadata :"
            )

            for key in [
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration"
            ]:

                if key in response_metadata:

                    print(
                        f"  {key}: "
                        f"{response_metadata[key]}"
                    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("05 - RAG + QWEN ABAP/HCM V2")
    print("=" * 70)

    print()
    print(
        f"Modello Qwen : {QWEN_MODEL}"
    )

    print(
        f"RAG TOP K    : {RAG_TOP_K}"
    )

    print(
        f"Max context  : {MAX_CONTEXT_CHARS} caratteri"
    )

    query = input(
        "\nInserisci la domanda ABAP/HCM:\n> "
    ).strip()

    if not query:

        print(
            "[ERRORE] Query vuota."
        )

        return

    # ========================================================
    # 1. RETRIEVAL
    # ========================================================

    print()
    print("=" * 70)
    print("1. RETRIEVAL")
    print("=" * 70)

    retrieval_start = time.perf_counter()

    try:

        results = retrieve(query)

    except Exception as e:

        print()
        print(
            "[ERRORE RETRIEVAL]"
        )

        print(
            type(e).__name__,
            str(e)
        )

        return

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    print(
        f"Risultati recuperati : {len(results)}"
    )

    print(
        f"Tempo retrieval      : "
        f"{retrieval_time:.2f} sec"
    )

    if not results:

        print()
        print(
            "[ATTENZIONE] "
            "Nessun risultato RAG."
        )

        return

    # ========================================================
    # 2. DIAGNOSTICA CHUNK
    # ========================================================

    print_retrieval_results(
        results
    )

    # ========================================================
    # 3. CONTEXT
    # ========================================================

    context = build_context(
        results
    )

    context_chars = len(context)

    context_tokens = estimate_tokens(
        context
    )

    print()
    print("=" * 70)
    print("2. CONTEXT")
    print("=" * 70)

    print(
        f"Chunk utilizzati       : "
        f"{min(len(results), RAG_TOP_K)}"
    )

    print(
        f"Caratteri context      : "
        f"{context_chars}"
    )

    print(
        f"Token stimati context : "
        f"{context_tokens}"
    )

    # ========================================================
    # 4. PROMPT
    # ========================================================

    prompt = build_prompt(
        query,
        context
    )

    prompt_chars = len(prompt)

    prompt_tokens = estimate_tokens(
        prompt
    )

    print()
    print("=" * 70)
    print("3. PROMPT")
    print("=" * 70)

    print(
        f"Caratteri prompt       : "
        f"{prompt_chars}"
    )

    print(
        f"Token stimati prompt  : "
        f"{prompt_tokens}"
    )

    # --------------------------------------------------------
    # Avviso preventivo
    # --------------------------------------------------------

    if prompt_tokens >= 3700:

        print()
        print(
            "[ATTENZIONE]"
        )

        print(
            "Il prompt stimato è vicino "
            "al limite di 4096 token."
        )

        print(
            "Considerare una riduzione del context."
        )

    # ========================================================
    # 5. PROMPT COMPLETO
    # ========================================================

    print()
    print("=" * 70)
    print("CONTESTO INVIATO A QWEN")
    print("=" * 70)

    print(context)

    # ========================================================
    # 6. QWEN
    # ========================================================

    print()
    print("=" * 70)
    print("4. QWEN")
    print("=" * 70)

    print(
        f"Modello: {QWEN_MODEL}"
    )

    print(
        "\nElaborazione in corso..."
    )

    try:

        response, qwen_time = ask_qwen(
            prompt
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("ERRORE QWEN")
        print("=" * 70)

        print(
            "Tipo:",
            type(e).__name__
        )

        print(
            "Messaggio:",
            str(e)
        )

        return

    # ========================================================
    # 7. RISPOSTA
    # ========================================================

    print()
    print("=" * 70)
    print("5. RISPOSTA QWEN")
    print("=" * 70)

    print(
        response.content
    )

    # ========================================================
    # 8. METRICHE
    # ========================================================

    print_qwen_metrics(
        response,
        qwen_time
    )

    # ========================================================
    # 9. RIEPILOGO
    # ========================================================

    print()
    print("=" * 70)
    print("RIEPILOGO")
    print("=" * 70)

    print(
        f"Retrieval : "
        f"{retrieval_time:.2f} sec"
    )

    print(
        f"Qwen      : "
        f"{qwen_time:.2f} sec"
    )

    print(
        f"Totale    : "
        f"{retrieval_time + qwen_time:.2f} sec"
    )

    print(
        f"Chunk RAG : "
        f"{min(len(results), RAG_TOP_K)}"
    )

    print(
        f"Context   : "
        f"{context_chars} caratteri"
    )

    print(
        f"Prompt ~  : "
        f"{prompt_tokens} token"
    )


if __name__ == "__main__":
    main()

