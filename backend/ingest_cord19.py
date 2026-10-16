import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import kaggle

# Kaggle setup - user needs to have kaggle.json in ~/.kaggle/kaggle.json
# Download CORD-19 dataset
print("Downloading CORD-19 dataset from Kaggle...")
kaggle.api.dataset_download_files('allen-institute-for-ai/CORD-19-research-challenge', path='data', unzip=True)
print("Download complete!")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
VECTOR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectorstore")
MAX_DOCS = 5000  # Limit to avoid too large index

os.makedirs(VECTOR_DIR, exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

documents = []
metadata = []

# Read metadata
metadata_path = os.path.join(DATA_DIR, "metadata.csv")
print(f"Reading {metadata_path}...")
df = pd.read_csv(metadata_path)
print(f"Total papers: {len(df)}")

count = 0
for idx, row in df.iterrows():
    if count >= MAX_DOCS:
        break

    paper_id = row.get('sha', '')
    title = row.get('title', '')
    abstract = row.get('abstract', '')
    authors = row.get('authors', '')

    # Try to get full text from JSON
    full_text = ""
    json_path = os.path.join(DATA_DIR, "document_parses", "pdf_json", f"{paper_id}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(DATA_DIR, "document_parses", "pmc_json", f"{paper_id}.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)

            # Extract text from body
            body_texts = []
            for section in paper_data.get('body_text', []):
                body_texts.append(section.get('text', ''))

            full_text = ' '.join(body_texts)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    # Combine title, abstract, and body
    doc_text = f"Title: {title}\nAbstract: {abstract}\nAuthors: {authors}\n\n{full_text}"

    if len(doc_text.strip()) > 100:  # Only include if substantial content
        documents.append(doc_text[:3000])  # Truncate long docs
        metadata.append({
            'title': title,
            'authors': authors,
            'paper_id': paper_id
        })
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} papers...")

print(f"Total documents processed: {len(documents)}")

# Create embeddings
print("Creating embeddings...")
embeddings = model.encode(documents, show_progress_bar=True)

# Build FAISS index
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
faiss.normalize_L2(embeddings)  # Normalize for cosine
index.add(embeddings)

# Save index and documents
faiss.write_index(index, os.path.join(VECTOR_DIR, "faiss_index"))
with open(os.path.join(VECTOR_DIR, "docs.pkl"), "wb") as f:
    pickle.dump(documents, f)
with open(os.path.join(VECTOR_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("Ingestion complete! Vectorstore created.")