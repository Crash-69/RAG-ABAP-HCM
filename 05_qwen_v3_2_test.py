# ================================================================
# SAP ABAP RAG - QWEN DIAGNOSTIC HARNESS V2
# ================================================================
#
# Obiettivo:
#   Validare Qwen separatamente da Retrieval e Context Builder.
#
# Pipeline:
#   VectorDB
#       ↓
#   Retrieval V3.1
#       ↓
#   Context Builder V3.1
#       ↓
#   EVIDENZA TECNICA VERIFICATA
#       ↓
#   Qwen
#
# IMPORTANTE:
#   - VectorDB READ-ONLY
#   - Nessuna retrieval
#   - Nessuna modifica al VectorDB
#   - Nessuna invenzione di oggetti/campi SAP
#
# V2:
#   - field extraction test
#   - evidence-first prompting
#   - controllo esplicito dei campi
#   - distinzione evidenza / interpretazione / codice
#
# ================================================================

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


DEFAULT_CONTEXT = r"C:\Progetto_AI\context_v3_1_results.json"
DEFAULT_OUTPUT = r"C:\Progetto_AI\qwen_v3_2_results.json"

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "oisee/qwen-coder-abap:v7"


# ================================================================
# ARGOMENTI
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context",
        default=DEFAULT_CONTEXT,
        help="JSON prodotto dal Context Builder V3.1"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="File JSON risultati"
    )

    parser.add_argument(
        "--tests",
        default="T09,T10,T11,T12,T12E",
        help="Test da eseguire separati da virgola"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=900
    )

    return parser.parse_args()


# ================================================================
# HTTP OLLAMA
# ================================================================

def http_json(url, payload=None, timeout=120):

    if payload is None:
        req = urllib.request.Request(url)
    else:
        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")

    return json.loads(raw)


# ================================================================
# CHECK OLLAMA
# ================================================================

def check_ollama():

    print("\n" + "=" * 70)
    print("CHECK OLLAMA")
    print("=" * 70)

    try:
        data = http_json(
            OLLAMA_URL + "/api/tags",
            timeout=30
        )

    except Exception as exc:
        print("ERRORE Ollama:", exc)
        return False, []

    models = []

    for item in data.get("models", []):
        name = item.get("name", "")
        if name:
            models.append(name)

    print("Ollama:", OLLAMA_URL)
    print("Modello richiesto:", MODEL)

    found = MODEL in models

    print(
        "Modello disponibile:",
        "SI" if found else "NO"
    )

    if not found:
        print("\nModelli disponibili:")
        for name in models:
            print("  -", name)

    return found, models


# ================================================================
# VALIDAZIONE CONTEXT
# ================================================================

def load_context(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Context file non trovato: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return data


def validate_context(data):

    print("\n" + "=" * 70)
    print("VALIDAZIONE CONTEXT BUILDER")
    print("=" * 70)

    version = str(data.get("version", ""))

    print("Version:", version)
    print(
        "VectorDB read-only:",
        data.get("vector_db_read_only")
    )

    if "V3.1" not in version:
        raise ValueError(
            "Il context non risulta prodotto dal Context Builder V3.1"
        )

    if data.get("vector_db_read_only") is not True:
        raise ValueError(
            "vector_db_read_only != true"
        )

    return True


# ================================================================
# ESTRAZIONE CONTEXT
# ================================================================

def extract_context(test):

    candidates = []

    for key in (
        "tests",
        "results",
        "test_results",
        "cases"
    ):
        value = test_context.get(key)

        if isinstance(value, list):
            candidates.extend(value)

        elif isinstance(value, dict):
            candidates.extend(value.values())

    # ------------------------------------------------------------
    # T12E è un test diagnostico LOCALE.
    # Usa esattamente il contesto di T12.
    # Non deve esistere come test nel Context Builder.
    # ------------------------------------------------------------

    lookup_test = "T12" if test == "T12E" else test

    target = None

    for item in candidates:

        if not isinstance(item, dict):
            continue

        tid = str(
            item.get("test_id")
            or item.get("id")
            or item.get("test")
            or ""
        )

        if tid == lookup_test:
            target = item
            break

    if target is None:
        raise ValueError(
            f"Test {lookup_test} non trovato nel context "
            f"(richiesto da {test})"
        )

    return target


def normalize_selected(item):

    """
    Recupera i chunk selezionati indipendentemente
    dalla forma del JSON.
    """

    for key in (
        "selected",
        "selected_chunks",
        "context",
        "chunks"
    ):
        value = item.get(key)

        if isinstance(value, list):
            return value

    return []


# ================================================================
# COSTRUZIONE EVIDENZA
# ================================================================

def build_evidence(selected):

    """
    Costruisce una sezione compatta di evidenza.

    NON interpreta semanticamente i campi.
    Riporta esclusivamente ciò che compare
    nei chunk forniti dal Context Builder.
    """

    evidence = []

    for idx, chunk in enumerate(selected, start=1):

        if not isinstance(chunk, dict):
            continue

        source = (
            chunk.get("source_file")
            or chunk.get("source")
            or chunk.get("file")
            or "UNKNOWN"
        )

        chunk_id = (
            chunk.get("chunk_id")
            or chunk.get("chunk")
            or idx
        )

        relation = (
            chunk.get("relationship_type")
            or chunk.get("relation_type")
            or "NONE"
        )

        content = (
            chunk.get("content")
            or chunk.get("text")
            or chunk.get("document")
            or ""
        )

        evidence.append({
            "evidence_id": idx,
            "source_file": source,
            "chunk": chunk_id,
            "relationship_type": relation,
            "content": content
        })

    return evidence


def evidence_text(evidence):

    blocks = []

    for e in evidence:

        blocks.append(
            f"""
[EVIDENZA {e['evidence_id']}]
SOURCE: {e['source_file']}
CHUNK: {e['chunk']}
RELATION: {e['relationship_type']}

{e['content']}
"""
        )

    return "\n".join(blocks)


# ================================================================
# PROMPT SYSTEM
# ================================================================

SYSTEM_PROMPT = r"""
Sei un Expert ABAP Enterprise Developer S/4HANA HCM.

Stai eseguendo una verifica diagnostica di un sistema RAG SAP ABAP.

REGOLA PRINCIPALE:

Devi utilizzare ESCLUSIVAMENTE le informazioni contenute
nell'EVIDENZA TECNICA VERIFICATA fornita nel prompt.

Non utilizzare conoscenza esterna per introdurre:

- tabelle
- campi
- strutture
- classi
- function module
- metodi
- tipi
- relazioni
- significati semantici

che non siano supportati dall'evidenza.

============================================================
DISTINZIONE OBBLIGATORIA
============================================================

1. EVIDENZA

Riporta ciò che è esplicitamente presente nel contesto.

2. INTERPRETAZIONE

Puoi interpretare soltanto ciò che è documentato
o direttamente deducibile dal codice mostrato.

Se il significato di un campo non è documentato,
NON inventarlo.

3. CODICE

Il codice generato deve utilizzare esclusivamente:

- tabelle presenti nell'evidenza
- campi presenti nell'evidenza
- relazioni presenti nell'evidenza

============================================================
REGOLE SUI CAMPI
============================================================

Quando la richiesta chiede:

"quali campi"
"quali dati"
"dati organizzativi"

devi considerare TUTTI i campi SAP esplicitamente presenti
nell'evidenza rilevante.

Non devi omettere campi verificati.

Se un campo è presente nel codice ma il suo significato
non è documentato, riportalo comunque come campo verificato
e specifica che il significato non è documentato.

NON attribuire autonomamente un significato SAP
ad un campo.

============================================================
REGOLE ABAP
============================================================

- Non usare SELECT *
- Non inventare campi
- Non inventare tabelle
- Preferisci Open SQL moderno quando appropriato
- Mantieni le date di validità quando presenti nell'evidenza
- Mantieni le condizioni tecniche rilevanti quando presenti
- Se generalizzi variabili concrete del codice sorgente
  in parametri come iv_pernr, dichiaralo esplicitamente
  come generalizzazione del pattern sorgente

============================================================
SE IL CONTESTO NON BASTA
============================================================

Dichiara:

"CONTESTO INSUFFICIENTE"

e specifica esattamente quale informazione manca.

Non completare il risultato utilizzando conoscenza esterna.
"""


# ================================================================
# PROMPT TEST
# ================================================================

TEST_PROMPTS = {

    "T09": """
Richiesta:

Spiega come leggere il centro di costo KOSTL
del dipendente tramite PERNR e PA0001.

Fornisci un esempio ABAP minimo.

Mantieni le condizioni di validità presenti nell'evidenza.
""",

    "T10": """
Richiesta:

Mostra come leggere PA0001 per un PERNR.

Indica:
- tabella
- campi utilizzati
- condizione WHERE
- validità temporale

Poi mostra un esempio ABAP.
""",

    "T11": """
Richiesta:

Mostra il pattern ABAP per:

SELECT PA0001 WHERE PERNR

utilizzando esclusivamente il contesto.

Mantieni le condizioni di validità presenti.
""",

    "T12": """
Richiesta:

Spiega quali dati organizzativi del dipendente
sono esplicitamente presenti in PA0001 nel contesto.

1. Elenca TUTTI i campi PA0001 esplicitamente presenti.
2. Non omettere nessun campo verificato.
3. Per ogni campo indica il significato SOLO se documentato.
4. Se il significato non è documentato, scrivi:
   "significato non documentato nel contesto".
5. Mostra infine un esempio ABAP basato esclusivamente
   sui campi verificati.
""",

    "T12E": """
Richiesta di sola estrazione dell'evidenza.

Elenca TUTTI i campi di PA0001 esplicitamente presenti
nel contesto fornito.

NON generare codice.

NON attribuire significati ai campi se non sono documentati.

Per ogni campo restituisci:

CAMPO: <nome>
SIGNIFICATO: <significato documentato oppure
"non documentato nel contesto">

Alla fine indica:

CAMPI TOTALI: <numero>
"""
}


# ================================================================
# CHIAMATA QWEN
# ================================================================

def call_qwen(
    prompt,
    temperature=0.0,
    timeout=900
):

    payload = {
        "model": MODEL,

        "system": SYSTEM_PROMPT,

        "prompt": prompt,

        "stream": False,

        "options": {
            "temperature": temperature
        }
    }

    start = time.perf_counter()

    response = http_json(
        OLLAMA_URL + "/api/generate",
        payload=payload,
        timeout=timeout
    )

    elapsed = time.perf_counter() - start

    return response, elapsed


# ================================================================
# DIAGNOSTICA FIELD EXTRACTION
# ================================================================

EXPECTED_T12_FIELDS = [
    "PA0001",
    "PERNR",
    "BUKRS",
    "WERKS",
    "BTRTL",
    "PERSG",
    "PERSK",
    "KOSTL",
    "ORGEH",
    "PLANS",
    "STELL",
]


def field_presence_check(text, expected):

    upper = text.upper()

    result = {}

    for field in expected:
        result[field] = field in upper

    return result


# ================================================================
# ESECUZIONE TEST
# ================================================================

def run_test(
    test_id,
    context_item,
    temperature,
    timeout
):

    selected = normalize_selected(context_item)

    evidence = build_evidence(selected)

    prompt = f"""
==============================
TEST {test_id}
==============================

EVIDENZA TECNICA VERIFICATA

{evidence_text(evidence)}

==============================
RICHIESTA
==============================

{TEST_PROMPTS[test_id]}
"""

    print("\n" + "=" * 70)
    print(f"TEST {test_id}")
    print("=" * 70)

    print(
        f"Context chunks: {len(selected)}"
    )

    print(
        f"Evidence blocks: {len(evidence)}"
    )

    response, elapsed = call_qwen(
        prompt,
        temperature=temperature,
        timeout=timeout
    )

    answer = response.get("response", "")

    print("\n--- QWEN RESPONSE ---\n")
    print(answer)

    print("\n--- DIAGNOSTICA ---")

    print(
        "Latency: {:.3f} sec".format(elapsed)
    )

    metadata = {
        key: response.get(key)
        for key in (
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
            "total_duration",
            "load_duration"
        )
        if key in response
    }

    for key, value in metadata.items():
        print(f"{key}: {value}")

    result = {
        "test_id": test_id,
        "context_chunks": len(selected),
        "evidence_blocks": len(evidence),
        "response": answer,
        "latency_seconds": elapsed,
        "ollama_metadata": metadata
    }

    if test_id in ("T12", "T12E"):

        check = field_presence_check(
            answer,
            EXPECTED_T12_FIELDS
        )

        result["expected_t12_fields"] = EXPECTED_T12_FIELDS
        result["field_presence"] = check

        print("\n--- T12 FIELD CHECK ---")

        for field, found in check.items():

            print(
                f"{field:8s}: "
                f"{'FOUND' if found else 'MISSING'}"
            )

    return result


# ================================================================
# MAIN
# ================================================================

def main():

    global test_context

    args = parse_args()

    requested_tests = [
        x.strip().upper()
        for x in args.tests.split(",")
        if x.strip()
    ]

    print("=" * 70)
    print("SAP ABAP RAG - QWEN DIAGNOSTIC HARNESS V2")
    print("=" * 70)

    print("Context:", args.context)
    print("Output:", args.output)
    print("Model:", MODEL)
    print("Temperature:", args.temperature)
    print("Tests:", ", ".join(requested_tests))

    # ------------------------------------------------------------
    # OLLAMA
    # ------------------------------------------------------------

    model_ok, models = check_ollama()

    if not model_ok:
        print(
            "\nERRORE: modello Qwen non disponibile."
        )
        sys.exit(1)

    # ------------------------------------------------------------
    # CONTEXT
    # ------------------------------------------------------------

    try:

        test_context = load_context(
            args.context
        )

        validate_context(
            test_context
        )

    except Exception as exc:

        print(
            "\nERRORE context:",
            exc
        )

        sys.exit(1)

    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------

    results = []

    for test_id in requested_tests:

        if test_id not in TEST_PROMPTS:

            print(
                f"\nWARNING: test sconosciuto: {test_id}"
            )

            continue

        try:

            context_item = extract_context(
                test_id
            )

            result = run_test(
                test_id,
                context_item,
                args.temperature,
                args.timeout
            )

            results.append(result)

        except Exception as exc:

            print(
                f"\nERRORE TEST {test_id}:",
                exc
            )

            results.append({
                "test_id": test_id,
                "error": str(exc)
            })

    # ------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------

    output = {
        "version": "Qwen Diagnostic Harness V2",
        "generated": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "source_context": args.context,
        "vector_db_read_only": True,
        "retrieval_executed": False,
        "ollama_url": OLLAMA_URL,
        "model": MODEL,
        "temperature": args.temperature,
        "tests": results
    }

    output_path = Path(args.output)

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("COMPLETATO")
    print("=" * 70)

    print(
        "Risultati salvati in:",
        output_path
    )


if __name__ == "__main__":
    main()