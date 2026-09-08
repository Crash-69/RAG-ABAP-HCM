import csv
import hashlib
import os
from collections import defaultdict
from pathlib import Path  # <--- RIGA AGGIUNTA PER RISOLVERE L'ERRORE
# ============================================================
# Questo script controlla i file ABAP / HTML nella directory specificata,
# calcola il loro hash SHA-256, conta le righe e genera un report CSV sui duplicati.    

# ============================================================
# CONFIGURAZIONE
# ============================================================

SOURCE_DIR = Path(r"C:\Progetto_AI\Abap")
REPORT_FILE = Path(r"C:\Progetto_AI\Abap_analisi_duplicati.csv")

# Estensioni che vogliamo analizzare
EXTENSIONS = {".txt", ".html", ".htm"}


# ============================================================
# FUNZIONI
# ============================================================

def calcola_sha256(file_path: Path, block_size: int = 1024 * 1024) -> str:
    """Calcola SHA-256 del file senza caricarlo interamente in RAM."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


def conta_righe(file_path: Path) -> int:
    """Conta le righe del file usando una lettura a blocchi di testo."""
    try:
        with file_path.open(
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:
            return sum(1 for _ in f)
    except Exception:
        return -1


def analizza_file():
    if not SOURCE_DIR.exists():
        print(f"[ERRORE] La cartella non esiste: {SOURCE_DIR}")
        return

    if not SOURCE_DIR.is_dir():
        print(f"[ERRORE] Il percorso non è una directory: {SOURCE_DIR}")
        return

    print("=" * 70)
    print("ANALISI FILE ABAP / HTML")
    print("=" * 70)
    print(f"Directory: {SOURCE_DIR}")
    print(f"Report   : {REPORT_FILE}")
    print()

    # hash -> lista file
    gruppi_hash = defaultdict(list)

    # Tutti i record da scrivere nel CSV
    records = []

    totale = 0
    errori = 0

    # --------------------------------------------------------
    # Scansione ricorsiva
    # --------------------------------------------------------

    for root, dirs, files in os.walk(SOURCE_DIR):

        # Ordine stabile
        dirs.sort()
        files.sort()

        for filename in files:

            file_path = Path(root) / filename

            if file_path.suffix.lower() not in EXTENSIONS:
                continue

            totale += 1

            try:
                stat = file_path.stat()
                sha256 = calcola_sha256(file_path)
                righe = conta_righe(file_path)

                record = {
                    "file_name": file_path.name,
                    "relative_path": str(
                        file_path.relative_to(SOURCE_DIR)
                    ),
                    "extension": file_path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "lines": righe,
                    "sha256": sha256,
                }

                records.append(record)
                gruppi_hash[sha256].append(record)

            except Exception as e:
                errori += 1
                print(
                    f"[ERRORE] Impossibile analizzare "
                    f"{file_path}: {e}"
                )

    # --------------------------------------------------------
    # Scrittura report
    # --------------------------------------------------------

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Determina il gruppo duplicati
    for record in records:
        duplicati = gruppi_hash[record["sha256"]]

        if len(duplicati) == 1:
            record["duplicate_group"] = ""
            record["duplicate_count"] = 0
            record["duplicate_type"] = ""
        else:
            # Gruppo deterministico basato sui primi 12 caratteri hash
            record["duplicate_group"] = record["sha256"][:12]
            record["duplicate_count"] = len(duplicati)
            record["duplicate_type"] = "DUPLICATO_ESATTO"

    fieldnames = [
        "file_name",
        "relative_path",
        "extension",
        "size_bytes",
        "size_kb",
        "lines",
        "sha256",
        "duplicate_group",
        "duplicate_count",
        "duplicate_type",
    ]

    with REPORT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            delimiter=";"
        )

        writer.writeheader()
        writer.writerows(records)

    # --------------------------------------------------------
    # Riepilogo
    # --------------------------------------------------------

    gruppi_duplicati = {
        sha: files
        for sha, files in gruppi_hash.items()
        if len(files) > 1
    }

    file_duplicati = sum(
        len(files)
        for files in gruppi_duplicati.values()
    )

    print()
    print("=" * 70)
    print("RISULTATO ANALISI")
    print("=" * 70)
    print(f"File analizzati       : {totale}")
    print(f"Errori                : {errori}")
    print(f"Gruppi di duplicati   : {len(gruppi_duplicati)}")
    print(f"File duplicati esatti : {file_duplicati}")
    print()

    if gruppi_duplicati:

        print("DUPLICATI TROVATI:")
        print("-" * 70)

        for index, (sha256, files) in enumerate(
            gruppi_duplicati.items(),
            start=1
        ):

            print(
                f"\nGruppo {index} "
                f"(SHA256: {sha256[:12]}...)"
            )

            for record in files:
                print(
                    f"  - {record['relative_path']} "
                    f"({record['size_kb']} KB)"
                )

    else:
        print("Nessun duplicato esatto trovato.")

    print()
    print(f"Report CSV salvato in:")
    print(REPORT_FILE)
    print()
    print(
        "ATTENZIONE: questo script NON modifica, NON sposta "
        "e NON cancella alcun file."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    analizza_file()
