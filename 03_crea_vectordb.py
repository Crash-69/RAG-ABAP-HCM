import os
import re
import json
import hashlib
from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURAZIONE
# ============================================================

SOURCE_DIR = Path(r"C:\Progetto_AI\Abap")
DB_DIR = Path(r"C:\Progetto_AI\Abap_VectorDB")
RAG_DIR = Path(r"C:\Progetto_AI\Abap_RAG")

MANIFEST_FILE = RAG_DIR / "manifest.json"

EMBEDDING_MODEL = "nomic-embed-text:latest"

# Chunk iniziale prudente per codice ABAP
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250

# Estensioni gestite
TEXT_EXTENSIONS = {".txt"}
HTML_EXTENSIONS = {".html", ".htm"}


# ============================================================
# UTILITY
# ============================================================

def sha256_file(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


def carica_manifest():
    if not MANIFEST_FILE.exists():
        return {}

    try:
        with MANIFEST_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ATTENZIONE] Manifest non leggibile: {e}")
        return {}


def salva_manifest(manifest):
    RAG_DIR.mkdir(parents=True, exist_ok=True)

    temp_file = MANIFEST_FILE.with_suffix(".tmp")

    with temp_file.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    temp_file.replace(MANIFEST_FILE)


# ============================================================
# CLASSIFICAZIONE ABAP
# ============================================================

def classifica_abap(text: str, file_name: str) -> str:

    upper = text.upper()

    # Ordine importante: controlliamo prima gli oggetti più specifici

    if re.search(r"\bTYPE-POOL\s+\w+", upper):
        return "TYPE_POOL"

    if re.search(r"\bCLASS-POOL\b", upper):
        return "CLASS_POOL"

    if re.search(r"\bINTERFACE\s+\w+", upper):
        return "INTERFACE"

    if re.search(r"\bCLASS\s+\w+", upper):
        return "CLASS"

    if re.search(r"\bFUNCTION-POOL\s+\w+", upper):
        return "FUNCTION_GROUP"

    if re.search(r"\bFUNCTION\s+\w+", upper):
        return "FUNCTION"

    if re.search(r"\bREPORT\s+\w+", upper):
        return "REPORT"

    if re.search(r"\bPROGRAM\s+\w+", upper):
        return "PROGRAM"

    if re.search(r"\bFORM\s+\w+", upper):
        return "FORM"

    if re.search(r"\bMETHOD\s+\w+", upper):
        return "METHOD"

    if re.search(r"\bMODULE\s+\w+", upper):
        return "MODULE"

    if re.search(r"\bDEFINE\s+\w+", upper):
        return "MACRO"

    if re.search(r"\bINCLUDE\b", upper):
        return "INCLUDE"

    # Euristica per file che sembrano DDIC/documentazione
    if re.search(r"\b(TABLE|STRUCTURE|DATA ELEMENT|DOMAIN)\b", upper):
        return "DDIC"

    return "ABAP_SOURCE"


def estrai_nome_oggetto(text: str, object_type: str, file_name: str) -> str:

    patterns = {
        "TYPE_POOL": r"\bTYPE-POOL\s+([A-Z0-9_/]+)",
        "CLASS": r"\bCLASS\s+([A-Z0-9_/]+)",
        "INTERFACE": r"\bINTERFACE\s+([A-Z0-9_/]+)",
        "FUNCTION_GROUP": r"\bFUNCTION-POOL\s+([A-Z0-9_/]+)",
        "FUNCTION": r"\bFUNCTION\s+([A-Z0-9_/]+)",
        "REPORT": r"\bREPORT\s+([A-Z0-9_/]+)",
        "PROGRAM": r"\bPROGRAM\s+([A-Z0-9_/]+)",
        "METHOD": r"\bMETHOD\s+([A-Z0-9_/~]+)",
        "FORM": r"\bFORM\s+([A-Z0-9_/]+)",
        "MODULE": r"\bMODULE\s+([A-Z0-9_/]+)",
    }

    pattern = patterns.get(object_type)

    if pattern:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            return match.group(1)

    return Path(file_name).stem


# ============================================================
# CARICAMENTO DOCUMENTI
# ============================================================

def carica_txt(file_path: Path) -> str:

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1"
    ]

    last_error = None

    for encoding in encodings:
        try:
            return file_path.read_text(
                encoding=encoding,
                errors="strict"
            )
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Impossibile leggere {file_path}: {last_error}"
    )


def carica_html(file_path: Path) -> str:

    try:
        from bs4 import BeautifulSoup

        raw = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        soup = BeautifulSoup(
            raw,
            "html.parser"
        )

        return soup.get_text(
            separator="\n"
        )

    except ImportError:
        raise RuntimeError(
            "BeautifulSoup non installato. "
            "Esegui: pip install beautifulsoup4"
        )


# ============================================================
# CHUNKING
# ============================================================

def crea_chunks(text: str):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            "ENDMETHOD.",
            "ENDCLASS.",
            "ENDFUNCTION.",
            "ENDFORM.",
            "ENDMODULE.",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)


# ============================================================
# CREAZIONE DOCUMENTI
# ============================================================

def crea_documenti(file_path: Path, file_hash: str):

    extension = file_path.suffix.lower()

    if extension in TEXT_EXTENSIONS:

        text = carica_txt(file_path)

        file_type = "ABAP_TXT"

    elif extension in HTML_EXTENSIONS:

        text = carica_html(file_path)

        file_type = "HTML_DDIC"

    else:
        return []

    if not text.strip():
        return []

    if file_type == "ABAP_TXT":

        object_type = classifica_abap(
            text,
            file_path.name
        )

        object_name = estrai_nome_oggetto(
            text,
            object_type,
            file_path.name
        )

        language = "ABAP"

    else:

        object_type = "DDIC_HTML"
        object_name = file_path.stem
        language = "SAP_DDIC"

    chunks = crea_chunks(text)

    documents = []

    relative_path = str(
        file_path.relative_to(SOURCE_DIR)
    )

    for index, chunk in enumerate(chunks):

        chunk_hash = sha256_text(chunk)

        # ID deterministico:
        # stesso file + stesso contenuto + stesso indice
        chunk_id_source = (
            f"{file_hash}:"
            f"{index}:"
            f"{chunk_hash}"
        )

        chunk_id = hashlib.sha256(
            chunk_id_source.encode("utf-8")
        ).hexdigest()

        metadata = {
            "source_file": file_path.name,
            "source_path": relative_path,
            "file_hash": file_hash,
            "chunk_hash": chunk_hash,
            "chunk_index": index,
            "file_type": file_type,
            "language": language,
            "object_type": object_type,
            "object_name": object_name,
        }

        documents.append(
            Document(
                page_content=chunk,
                metadata=metadata
            )
        )

    return documents


# ============================================================
# MAIN
# ============================================================

def crea_vectordb():

    print("=" * 70)
    print("RAG ABAP/HCM - INGESTION")
    print("=" * 70)

    if not SOURCE_DIR.exists():
        print(
            f"[ERRORE] Directory sorgente non trovata: "
            f"{SOURCE_DIR}"
        )
        return

    RAG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_data_old = carica_manifest()

    # Il manifest contiene i file sotto la chiave "files".
    manifest = manifest_data_old.get("files", {})

    print(f"Sorgente : {SOURCE_DIR}")
    print(f"VectorDB : {DB_DIR}")
    print(f"Manifest : {MANIFEST_FILE}")
    print()

    # --------------------------------------------------------
    # Scansione sorgenti
    # --------------------------------------------------------

    files = []

    for root, dirs, filenames in os.walk(SOURCE_DIR):

        dirs.sort()
        filenames.sort()

        for filename in filenames:

            path = Path(root) / filename

            if path.suffix.lower() in (
                TEXT_EXTENSIONS |
                HTML_EXTENSIONS
            ):
                files.append(path)

    print(
        f"File candidati trovati: {len(files)}"
    )

    # --------------------------------------------------------
    # Hash e deduplicazione
    # --------------------------------------------------------

    hash_to_file = {}
    file_info = {}

    duplicati = 0

    for file_path in files:

        try:
            file_hash = sha256_file(file_path)

            relative_path = str(
                file_path.relative_to(SOURCE_DIR)
            )

            file_info[relative_path] = {
                "hash": file_hash,
                "size": file_path.stat().st_size
            }

            if file_hash in hash_to_file:

                duplicati += 1

            else:

                hash_to_file[file_hash] = file_path

        except Exception as e:

            print(
                f"[ERRORE HASH] {file_path}: {e}"
            )

    file_unici = list(
        hash_to_file.values()
    )

    print(
        f"File unici per contenuto: {len(file_unici)}"
    )

    print(
        f"Duplicati ignorati       : {duplicati}"
    )

    # --------------------------------------------------------
    # Inizializzazione embeddings
    # --------------------------------------------------------

    print()
    print(
        f"Inizializzazione embedding: "
        f"{EMBEDDING_MODEL}"
    )

    embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url="http://127.0.0.1:11434"
    )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    print("Inizializzazione ChromaDB...")

    vector_db = Chroma(
        collection_name="abap_hcm",
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )

    # --------------------------------------------------------
    # Indicizzazione incrementale
    # --------------------------------------------------------

    elaborati = 0
    saltati = 0
    errori = 0
    chunks_totali = 0

    nuovi_manifest = {}

    for posizione, file_path in enumerate(
        file_unici,
        start=1
    ):

        relative_path = str(
            file_path.relative_to(SOURCE_DIR)
        )

        info = file_info[relative_path]
        file_hash = info["hash"]

        # ----------------------------------------------------
        # Se il file è già presente con lo stesso hash,
        # non viene rielaborato.
        # ----------------------------------------------------

        old_info = manifest.get(
            relative_path
        )

        if (
            old_info
            and old_info.get("file_hash") == file_hash
        ):

            nuovi_manifest[relative_path] = old_info

            saltati += 1

            continue

        try:

            documents = crea_documenti(
                file_path,
                file_hash
            )

            if not documents:
                continue

            # ------------------------------------------------
            # Se il file era precedentemente indicizzato ma
            # modificato, eliminiamo i suoi vecchi chunk.
            # ------------------------------------------------

            if old_info:

                old_ids = old_info.get(
                    "chunk_ids",
                    []
                )

                if old_ids:

                    vector_db.delete(
                        ids=old_ids
                    )

            chunk_ids = []

            for document in documents:

                metadata = document.metadata

                chunk_id_source = (
                    f"{metadata['file_hash']}:"
                    f"{metadata['chunk_index']}:"
                    f"{metadata['chunk_hash']}"
                )

                chunk_id = hashlib.sha256(
                    chunk_id_source.encode("utf-8")
                ).hexdigest()

                chunk_ids.append(chunk_id)

            # ------------------------------------------------
            # Inserimento dei chunk a piccoli batch
            # per evitare problemi con il model runner Ollama.
            # ------------------------------------------------

            BATCH_SIZE = 20

            for batch_start in range(
                0,
                len(documents),
                BATCH_SIZE
            ):

                batch_documents = documents[
                    batch_start:batch_start + BATCH_SIZE
                ]

                batch_ids = chunk_ids[
                    batch_start:batch_start + BATCH_SIZE
                ]

                batch_num = (
                    batch_start // BATCH_SIZE
                ) + 1

                batch_totali = (
                    (len(documents) + BATCH_SIZE - 1)
                    // BATCH_SIZE
                )

                print(
                    f"    Batch {batch_num}/{batch_totali} "
                    f"({len(batch_documents)} chunk)"
                )

                vector_db.add_documents(
                    documents=batch_documents,
                    ids=batch_ids
                )

            nuovi_manifest[relative_path] = {
                "file_hash": file_hash,
                "size": info["size"],
                "chunk_count": len(documents),
                "chunk_ids": chunk_ids,
                "object_type": documents[0].metadata[
                    "object_type"
                ],
                "object_name": documents[0].metadata[
                    "object_name"
                ]
            }

            elaborati += 1
            chunks_totali += len(documents)

            if (
                posizione % 100 == 0
                or posizione == len(file_unici)
            ):

                print(
                    f"[{posizione}/{len(file_unici)}] "
                    f"Elaborati: {elaborati} | "
                    f"Saltati: {saltati} | "
                    f"Chunk: {chunks_totali}"
                )

        except Exception as e:

            errori += 1

            print(
                f"[ERRORE] {relative_path}: {e}"
            )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_data = {
        "version": 1,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "files": nuovi_manifest
    }

    salva_manifest(
        manifest_data
    )

    # --------------------------------------------------------
    # Riepilogo
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INGESTION COMPLETATA")
    print("=" * 70)
    print(f"File trovati          : {len(files)}")
    print(f"File unici            : {len(file_unici)}")
    print(f"Duplicati ignorati    : {duplicati}")
    print(f"File indicizzati      : {elaborati}")
    print(f"File già presenti     : {saltati}")
    print(f"Chunk generati        : {chunks_totali}")
    print(f"Errori                : {errori}")
    print()
    print(f"VectorDB: {DB_DIR}")
    print(f"Manifest: {MANIFEST_FILE}")
    print()
    print(
        "La directory sorgente NON è stata modificata."
    )


if __name__ == "__main__":
    crea_vectordb()
