import os
import json
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")

EMBED_MODEL = SentenceTransformer("models/bge-small")

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, size=120, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk.split()) > 20:
            chunks.append(chunk)
    return chunks

def build_index(lang):
    pdf_dir = os.path.join(DATA_DIR, lang, "pdfs")
    index_path = os.path.join(EMBED_DIR, f"faiss_{lang}.index")

    model = EMBED_MODEL
    texts = []

    if not os.path.exists(pdf_dir):
        print(f"[RAG] No folder found for {lang}, skipping.")
        return

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            content = extract_text_from_pdf(os.path.join(pdf_dir, file))
            chunks = chunk_text(content)
            texts.extend(chunks)

    if not texts:
        print(f"[RAG] No valid text found for {lang}. Skipping index.")
        return

    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, index_path)

    with open(index_path + ".meta", "w") as f:
        json.dump(texts, f)

    print(f"[RAG] Indexed {len(texts)} chunks for {lang}")

def ensure_indexes():
    os.makedirs(EMBED_DIR, exist_ok=True)
    for lang in ["english", "hindi"]:
        idx = os.path.join(EMBED_DIR, f"faiss_{lang}.index")
        if not os.path.exists(idx):
            print(f"[RAG] Building {lang} index...")
            build_index(lang)
