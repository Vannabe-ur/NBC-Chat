import chromadb
from .vector_store import embed_text, PERSIST_DIR, COLLECTION_NAME

DISTANCE_THRESHOLD = 0.6

def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME)

def retrieve(question: str, top_k: int = 4) -> dict:
    collection = get_collection()
    query_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    matches = []
    for doc, meta, distance, doc_id in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        matches.append({
            "id":doc_id, 
            "question": doc,
            "answer": meta["answer"],
            "category": meta["category"],
            "distance": distance,
        })

    best_distance = matches[0]["distance"] if matches else float("inf")
    is_confident = best_distance <= DISTANCE_THRESHOLD

    return {
        "matches": matches,
        "is_confident": is_confident,
        "best_distance": best_distance,
    }

if __name__=="__main__":
    result = retrieve("What is the mandate of the National Bank of Cambodia?")
    print(f"Confident: {result['is_confident']} (best distance: {result['best_distance']:.3f})")
    for m in result["matches"]:
        print(f"  [{m['distance']:.3f}] {m['question']} -> {m['answer'][:120]}...")
