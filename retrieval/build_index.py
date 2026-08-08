import os
import json

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

CORPUS_DIR = BASE_DIR / "corpus"

INDEX_PATH = BASE_DIR / "index.faiss"

CHUNKS_PATH = BASE_DIR / "chunks.json"

def load_documents():
    documents = []

    for filename in sorted(os.listdir(CORPUS_DIR)):

        filepath = CORPUS_DIR / filename

        if filename.endswith(".md") or filename.endswith(".txt"):

            with open(filepath, "r", encoding="utf-8") as f:

                text = f.read()

                documents.append({
                    "filename": filename,
                    "text": text,
                })

    return documents
def main():
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")


    print(documents[0]["filename"])
    print()
    print(documents[0]["text"][:500])

    print("Loading embedding model...")

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    print("Model loaded successfully!")

    texts = [doc["text"] for doc in documents]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("Embeddings generated!")

    print(type(embeddings))
    print(embeddings.shape)

    print(embeddings[0][:10])


    print("Building FAISS index...")

    dimension = embeddings.shape[1] 

    index = faiss.IndexFlatL2(dimension) # creates empty vector database

    index.add(embeddings)

    print(f"FAISS index contains {index.ntotal} vectors.")

    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS index saved to {INDEX_PATH}")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4, ensure_ascii=False)

    print(f"Chunk mapping saved to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()