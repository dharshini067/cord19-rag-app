import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

base_dir = os.path.dirname(os.path.dirname(__file__))
vector_path = os.path.join(base_dir, "vectorstore")

index = faiss.read_index(os.path.join(vector_path, "faiss_index"))

with open(os.path.join(vector_path, "docs.pkl"), "rb") as f:
    documents = pickle.load(f)

meta_path = os.path.join(vector_path, "meta.pkl")
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
else:
    metadata = [{"title": "", "source": ""} for _ in documents]

def retrieve(query, k=10):
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)
    results = []
    for i in indices[0]:
        if 0 <= i < len(documents):
            results.append(documents[i])
    return results
