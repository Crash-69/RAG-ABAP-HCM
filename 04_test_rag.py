from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# --------------------------------------------------------
# Configurazione
# --------------------------------------------------------

DB_DIR = r"C:\Progetto_AI\Abap_VectorDB"

EMBEDDING_MODEL = "nomic-embed-text:latest"

COLLECTION_NAME = "abap_hcm"


# --------------------------------------------------------
# Embeddings
# --------------------------------------------------------

print("Inizializzazione embeddings...")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url="http://127.0.0.1:11434"
)


# --------------------------------------------------------
# ChromaDB
# --------------------------------------------------------

print("Connessione a ChromaDB...")

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DB_DIR
)


# --------------------------------------------------------
# Test
# --------------------------------------------------------

query = "PA0001"

print()
print("=" * 70)
print("QUERY")
print("=" * 70)
print(query)

results = vector_db.similarity_search(
    query,
    k=5
)


# --------------------------------------------------------
# Risultati
# --------------------------------------------------------

print()
print("=" * 70)
print("RISULTATI RAG")
print("=" * 70)

for index, document in enumerate(results, start=1):

    print()
    print("-" * 70)
    print(f"RISULTATO {index}")
    print("-" * 70)

    print("METADATA:")

    for key, value in document.metadata.items():
        print(f"  {key}: {value}")

    print()
    print("CONTENUTO:")
    print(document.page_content)