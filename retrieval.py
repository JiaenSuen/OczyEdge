import os
import pickle
import threading
import numpy as np

from config import EMBEDDING_PATH

_lock = threading.Lock()
_cache = None


def _normalize(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def load_embeddings():
    global _cache
    with _lock:
        if _cache is not None:
            return _cache

        if not os.path.exists(EMBEDDING_PATH):
            _cache = []
            return _cache

        try:
            with open(EMBEDDING_PATH, "rb") as f:
                _cache = pickle.load(f)
        except Exception:
            _cache = []

        return _cache


def save_embeddings(data):
    global _cache
    os.makedirs(os.path.dirname(EMBEDDING_PATH), exist_ok=True)
    with _lock:
        _cache = data
        with open(EMBEDDING_PATH, "wb") as f:
            pickle.dump(data, f)


def add_embedding(product_id, embedding):
    data = load_embeddings()
    vec = _normalize(embedding)
    data = [d for d in data if d[0] != product_id]
    data.append((product_id, vec))
    save_embeddings(data)


def delete_embedding(product_id):
    data = load_embeddings()
    data = [d for d in data if d[0] != product_id]
    save_embeddings(data)


def search_top_k(query_embedding, valid_ids, k=5):
    data = load_embeddings()
    valid_ids = set(valid_ids)

    data = [d for d in data if d[0] in valid_ids]
    if not data:
        return []

    q = _normalize(query_embedding)
    ids = [d[0] for d in data]
    vecs = np.vstack([d[1] for d in data])

    sims = vecs @ q
    order = np.argsort(-sims)[:k]

    results = []
    for idx in order:
        results.append({
            "product_id": ids[idx],
            "score": float(sims[idx]),
        })
    return results


def search(query_embedding, valid_ids):
    topk = search_top_k(query_embedding, valid_ids, k=1)
    if not topk:
        return None, None
    return topk[0]["product_id"], topk[0]["score"]