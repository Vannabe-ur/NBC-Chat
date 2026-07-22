import ollama
import chromadb
from pathlib import Path
from .data_loader import load_nbc_dataset, QARecord

EMBED_MODEL = "nomic-embed-text"
PERSIST_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
COLLECTION_NAME = "nbc_qa"


def embed_text(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def build_vector_store(records: list[QARecord], force_rebuild: bool = False):
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        if not force_rebuild:
            print("Collection already exists. Skipping rebuild (use force_rebuild=True to redo).")
            return client.get_collection(COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}    
    )

    ids, embeddings, documents, metadatas = [], [], [], []

    for record in records:
        ids.append(str(record.id))
        embeddings.append(embed_text(record.embed_text))
        documents.append(record.question)
        metadatas.append({
            "category": record.category,
            "answer": record.answer,
        })

        if record.id % 100 == 0:
            print(f"Embedded {record.id} / {len(records)}")

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"Vector store built with {len(records)} records at {PERSIST_DIR}")
    return collection


if __name__ == "__main__":
    records = load_nbc_dataset()
    build_vector_store(records)