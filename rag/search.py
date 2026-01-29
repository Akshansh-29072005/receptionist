import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/bge-small")

def load_index(lang):
    index = faiss.read_index(f"embeddings/faiss_{lang}.index")
    with open(f"embeddings/faiss_{lang}.index.meta") as f:
        texts = json.load(f)
    return index, texts

INDEX_CACHE = {}

def search(query, lang, k=3):
    if lang not in INDEX_CACHE:
        INDEX_CACHE[lang] = load_index(lang)

    index, texts = INDEX_CACHE[lang]
    emb = model.encode([query]).astype("float32")
    scores, ids = index.search(emb, k)

    return [texts[i] for i in ids[0] if i < len(texts)]
