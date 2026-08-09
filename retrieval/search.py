import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# Resolve the index and chunk map relative to this file, not the process's
# current working directory, so `search()` works the same whether it's run
# as `python retrieval/search.py`, imported from pipeline/spine.py, or
# imported from anywhere else.
_HERE = Path(__file__).resolve().parent
INDEX_PATH = _HERE / "index.faiss"
CHUNKS_PATH = _HERE / "chunks.json"

if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        "Retrieval artifacts are missing.\n\n"
        "Expected:\n"
        f"  - {INDEX_PATH}\n"
        f"  - {CHUNKS_PATH}\n\n"
        "Build them by running:\n"
        "python retrieval/build_index.py"
    )

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("Loading chunk mapping...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    documents = json.load(file)

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Search system ready!\n")



def search(query, k=5):
    
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # Search FAISS
    distances, indices = index.search(query_embedding, k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        results.append({
            "filename": documents[idx]["filename"],
            "text": documents[idx]["text"],
            "distance": float(distance)
        })

    return results



if __name__ == "__main__":

    query = input("Enter your question: ")

    results = search(query)

    print("\nTop Results\n")

    for i, result in enumerate(results, start=1):

        print(f"Result {i}")
        print(f"File      : {result['filename']}")
        print(f"Distance  : {result['distance']:.4f}")
        print("Passage:")
        print(result["text"])
        print("-" * 60)