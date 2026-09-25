# ================================================================
# SAP ABAP RAG - QWEN DIAGNOSTIC HARNESS V3.3
# ================================================================
#
# VectorDB: READ-ONLY
# Retrieval: NON eseguita
# Input: context_v3_1_results.json
#
# V3.3:
#   - estrazione deterministica dei campi PA0001
#   - VERIFIED FIELD INVENTORY
#   - quality gate EXPECTED / VERIFIED / MISSING / UNEXPECTED
#   - T12E = extraction-only
#   - T12  = explanation + ABAP
#   - contesto compatto e verificato per Qwen
#
# ================================================================

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path


DEFAULT_CONTEXT = r"C:\Progetto_AI\context_v3_1_results.json"
DEFAULT_OUTPUT = r"C:\Progetto_AI\qwen_v3_3_results.json"

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "oisee/qwen-coder-abap:v7"
TARGET_TABLE = "PA0001"

EXPECTED_T12_FIELDS = [
    "PERNR", "BUKRS", "WERKS", "BTRTL", "PERSG",
    "PERSK", "KOSTL", "ORGEH", "PLANS", "STELL"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--context", default=DEFAULT_CONTEXT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--tests", default="T12E,T12")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=900)
    return p.parse_args()


def http_json(url, payload=None, timeout=120):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check_ollama():
    print("\n" + "=" * 70)
    print("CHECK OLLAMA")
    print("=" * 70)
    try:
        data = http_json(OLLAMA_URL + "/api/tags", timeout=30)
    except Exception as exc:
        print("ERRORE Ollama:", exc)
        return False

    names = [m.get("name", "") for m in data.get("models", [])]
    found = MODEL in names
    print("Ollama:", OLLAMA_URL)
    print("Modello richiesto:", MODEL)
    print("Modello disponibile:", "SI" if found else "NO")
    if not found:
        print("\nModelli disponibili:")
        for n in names:
            if n:
                print("  -", n)
    return found


def load_context(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Context file non trovato: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_context(data):
    version = str(data.get("version", ""))
    print("\n" + "=" * 70)
    print("VALIDAZIONE CONTEXT BUILDER")
    print("=" * 70)
    print("Version:", version)
    print("VectorDB read-only:", data.get("vector_db_read_only"))

    if "V3.1" not in version:
        raise ValueError("Il context non risulta prodotto dal Context Builder V3.1")
    if data.get("vector_db_read_only") is not True:
        raise ValueError("vector_db_read_only != true")


def collect_items(data):
    out = []
    for key in ("tests", "results", "test_results", "cases"):
        value = data.get(key)
        if isinstance(value, list):
            out.extend(value)
        elif isinstance(value, dict):
            out.extend(value.values())
    return out


def extract_test(data, test_id):
    for item in collect_items(data):
        if not isinstance(item, dict):
            continue
        tid = str(
            item.get("test_id")
            or item.get("id")
            or item.get("test")
            or ""
        )
        if tid == test_id:
            return item
    raise ValueError(f"Test {test_id} non trovato nel context")


def normalize_selected(item):
    for key in ("selected", "selected_chunks", "context", "chunks"):
        value = item.get(key)
        if isinstance(value, list):
            return value
    return []


def chunk_source(chunk):
    return str(
        chunk.get("source_file")
        or chunk.get("source")
        or chunk.get("file")
        or "UNKNOWN"
    )


def chunk_id(chunk, default):
    return chunk.get("chunk_id") or chunk.get("chunk") or default


def chunk_content(chunk):
    return str(
        chunk.get("content")
        or chunk.get("text")
        or chunk.get("document")
        or ""
    )


def extract_pa0001_fields(text):
    """
    Estrazione conservativa e deterministica.

    Fonti riconosciute:
      1) PA0001-FIELD
      2) SELECT ... FROM PA0001

    Non interpreta il significato dei campi.
    Non usa Qwen.
    """
    upper = text.upper()
    fields = set()

    # PA0001-FIELD
    for m in re.finditer(
        r"\bPA0001-([A-Z][A-Z0-9_]*)\b",
        upper,
    ):
        fields.add(m.group(1))

    # SELECT ... FROM PA0001
    for m in re.finditer(
        r"\bSELECT\b(.*?)\bFROM\s+PA0001\b",
        upper,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        part = m.group(1)

        # Elimina una porzione INTO per evitare di interpretare
        # variabili ABAP come campi.
        part = re.split(
            r"\bINTO\b",
            part,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        ignored = {
            "SINGLE", "DISTINCT", "UP", "TO", "N", "ROWS",
            "BYPASSING", "BUFFER", "CORRESPONDING", "FIELDS",
            "OF", "TABLE", "APPENDING"
        }

        for token in re.findall(r"\b[A-Z][A-Z0-9_]*\b", part):
            if token in ignored or token == "PA0001":
                continue
            # Evita keyword o variabili tipiche.
            if token.startswith(
                ("LV_", "LS_", "LT_", "IV_", "EV_", "WA_", "P_")
            ):
                continue
            fields.add(token)

    return fields


def build_inventory(selected):
    inventory = {
        TARGET_TABLE: {
            "fields": set(),
            "sources": []
        }
    }

    evidence = []

    for idx, chunk in enumerate(selected, 1):
        content = chunk_content(chunk)
        fields = extract_pa0001_fields(content)

        if fields:
            inventory[TARGET_TABLE]["fields"].update(fields)
            inventory[TARGET_TABLE]["sources"].append({
                "source_file": chunk_source(chunk),
                "chunk": chunk_id(chunk, idx),
                "fields": sorted(fields),
            })

        evidence.append({
            "evidence_id": idx,
            "source_file": chunk_source(chunk),
            "chunk": chunk_id(chunk, idx),
            "relationship_type": (
                chunk.get("relationship_type")
                or chunk.get("relation_type")
                or "NONE"
            ),
            "content": content,
            "extracted_fields": sorted(fields),
        })

    inventory[TARGET_TABLE]["fields"] = sorted(
        inventory[TARGET_TABLE]["fields"]
    )
    return inventory, evidence


def quality_gate(inventory):
    found = set(inventory[TARGET_TABLE]["fields"])
    expected = set(EXPECTED_T12_FIELDS)
    verified = sorted(found & expected)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)

    return {
        "expected": sorted(expected),
        "verified": verified,
        "missing": missing,
        "unexpected": unexpected,
        "expected_count": len(expected),
        "verified_count": len(verified),
        "coverage": len(verified) / len(expected),
    }


SYSTEM_PROMPT = r"""
Sei un Expert ABAP Enterprise Developer S/4HANA HCM.

La VERIFIED FIELD INVENTORY è costruita deterministicamente
da Python a partire dal codice ABAP recuperato dal RAG.

REGOLE:

1. La VERIFIED FIELD INVENTORY è autorevole per i campi.
2. Non aggiungere campi non presenti nell'inventory.
3. Non omettere campi dell'inventory quando viene richiesto
   l'elenco completo.
4. Non attribuire significati semantici non documentati.
5. Se il significato non è documentato, scrivi:
   "significato non documentato nel contesto".
6. Non usare conoscenza SAP esterna per completare l'evidenza.
7. Il codice può usare solo tabelle, campi e condizioni presenti
   nell'evidenza.
8. Non usare SELECT *.
9. Non inventare strutture, tipi, classi o function module.
10. Se generalizzi variabili concrete del sorgente, dichiaralo.

Se un'informazione richiesta non è verificata:
CONTESTO INSUFFICIENTE
"""


def build_prompt(test_id, inventory, gate, evidence):
    fields = "\n".join(
        f"- {x}" for x in inventory[TARGET_TABLE]["fields"]
    )
    expected = "\n".join(
        f"- {x}" for x in gate["expected"]
    )

    blocks = []
    for e in evidence:
        blocks.append(
            f"""
[EVIDENCE {e['evidence_id']}]
SOURCE: {e['source_file']}
CHUNK: {e['chunk']}
RELATION: {e['relationship_type']}
EXTRACTED_FIELDS: {", ".join(e['extracted_fields']) or "NONE"}

{e['content']}
"""
        )

    if test_id == "T12E":
        request = """
TEST DI SOLA ESTRAZIONE.

Elenca TUTTI i campi PA0001 presenti nella
VERIFIED FIELD INVENTORY.

Per ogni campo:
CAMPO: <nome>
SIGNIFICATO: <solo se esplicitamente documentato;
altrimenti "significato non documentato nel contesto">

Non generare codice.
Non aggiungere campi.
Non omettere campi.

Alla fine:
CAMPI TOTALI: <numero>
"""
    elif test_id == "T12":
        request = """
Spiega quali dati organizzativi del dipendente sono
esplicitamente verificati.

1. Elenca tutti i campi della VERIFIED FIELD INVENTORY.
2. Non omettere nessun campo.
3. Indica il significato solo se documentato.
4. Per significati non documentati usa:
   "significato non documentato nel contesto".
5. Genera un esempio ABAP usando i campi verificati.
6. Mantieni le condizioni di validità presenti nell'evidenza.
7. Non introdurre campi esterni all'inventory.
"""
    else:
        raise ValueError(f"Test non supportato: {test_id}")

    return f"""
============================================================
TEST {test_id}
============================================================

VERIFIED FIELD INVENTORY
TABLE: {TARGET_TABLE}

{fields}

EXPECTED FIELDS
{expected}

FIELD QUALITY GATE
EXPECTED COUNT: {gate['expected_count']}
VERIFIED COUNT: {gate['verified_count']}
COVERAGE: {gate['coverage']:.3f}
MISSING: {", ".join(gate['missing']) or "NONE"}
UNEXPECTED: {", ".join(gate['unexpected']) or "NONE"}

============================================================
SOURCE EVIDENCE
============================================================

{"".join(blocks)}

============================================================
REQUEST
============================================================

{request}
"""


def call_qwen(prompt, temperature, timeout):
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    start = time.perf_counter()
    response = http_json(
        OLLAMA_URL + "/api/generate",
        payload=payload,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    return response, elapsed


def response_field_check(text, expected):
    upper = text.upper()
    presence = {
        field: bool(re.search(rf"\b{re.escape(field)}\b", upper))
        for field in expected
    }
    missing = sorted(
        field for field, ok in presence.items() if not ok
    )
    return {
        "field_presence": presence,
        "missing": missing,
        "missing_count": len(missing),
        "all_expected_present": not missing,
    }


def response_unexpected_check(text, inventory_fields):
    # Controllo mirato su identificatori plausibili già emersi
    # nel test, per intercettare soprattutto SNAME.
    candidates = set(inventory_fields) | {
        "SNAME", "ENAME", "VORNA", "NACHN",
        "GESCH", "GBDAT", "ANSVH", "STAT2"
    }
    upper = text.upper()
    found = {
        f for f in candidates
        if re.search(rf"\b{re.escape(f)}\b", upper)
    }
    return sorted(found - set(inventory_fields))


def run_test(test_id, context_item, temperature, timeout):
    selected = normalize_selected(context_item)
    inventory, evidence = build_inventory(selected)
    gate = quality_gate(inventory)

    print("\n" + "=" * 70)
    print(f"TEST {test_id}")
    print("=" * 70)
    print("Context chunks:", len(selected))
    print("Verified fields:", ", ".join(inventory[TARGET_TABLE]["fields"]))
    print("Expected fields:", ", ".join(gate["expected"]))
    print("Missing:", ", ".join(gate["missing"]) or "NONE")
    print("Unexpected in evidence:", ", ".join(gate["unexpected"]) or "NONE")
    print("Coverage: {:.1%}".format(gate["coverage"]))

    prompt = build_prompt(
        test_id, inventory, gate, evidence
    )
    response, elapsed = call_qwen(
        prompt, temperature, timeout
    )
    answer = response.get("response", "")

    print("\n--- QWEN RESPONSE ---\n")
    print(answer)

    metadata = {
        k: response.get(k)
        for k in (
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
            "total_duration",
            "load_duration",
        )
        if k in response
    }

    print("\n--- DIAGNOSTICA ---")
    print("Latency: {:.3f} sec".format(elapsed))
    for k, v in metadata.items():
        print(f"{k}: {v}")

    result = {
        "test_id": test_id,
        "context_chunks": len(selected),
        "verified_field_inventory": inventory,
        "field_quality_gate": gate,
        "response": answer,
        "latency_seconds": elapsed,
        "ollama_metadata": metadata,
    }

    check = response_field_check(answer, gate["expected"])
    unexpected = response_unexpected_check(
        answer, inventory[TARGET_TABLE]["fields"]
    )

    result["response_field_check"] = check
    result["unexpected_response_fields"] = unexpected

    print("\n--- RESPONSE FIELD CHECK ---")
    for field, present in check["field_presence"].items():
        print(
            f"{field:8s}: "
            f"{'FOUND' if present else 'MISSING'}"
        )
    print(
        "Unexpected response fields:",
        ", ".join(unexpected) or "NONE"
    )

    return result


def main():
    args = parse_args()

    tests = [
        x.strip().upper()
        for x in args.tests.split(",")
        if x.strip()
    ]

    print("=" * 70)
    print("SAP ABAP RAG - QWEN DIAGNOSTIC HARNESS V3.3")
    print("=" * 70)
    print("Context:", args.context)
    print("Output:", args.output)
    print("Model:", MODEL)
    print("Temperature:", args.temperature)
    print("Tests:", ", ".join(tests))

    if not check_ollama():
        sys.exit(1)

    try:
        data = load_context(args.context)
        validate_context(data)
    except Exception as exc:
        print("\nERRORE context:", exc)
        sys.exit(1)

    results = []

    for test_id in tests:
        # T12E è un test diagnostico locale e usa il T12
        # del Context Builder come sorgente.
        context_test_id = (
            "T12" if test_id in ("T12E", "T12")
            else test_id
        )

        try:
            context_item = extract_test(
                data, context_test_id
            )
            results.append(
                run_test(
                    test_id,
                    context_item,
                    args.temperature,
                    args.timeout,
                )
            )
        except Exception as exc:
            print(
                f"\nERRORE TEST {test_id}:",
                exc
            )
            results.append({
                "test_id": test_id,
                "error": str(exc),
            })

    output = {
        "version": "Qwen Diagnostic Harness V3.3",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_context": args.context,
        "vector_db_read_only": True,
        "retrieval_executed": False,
        "verified_evidence_builder": "deterministic_v1",
        "ollama_url": OLLAMA_URL,
        "model": MODEL,
        "temperature": args.temperature,
        "tests": results,
    }

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("COMPLETATO")
    print("=" * 70)
    print("Risultati salvati in:", output_path)


if __name__ == "__main__":
    main()
