import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

class Document:
    def __init__(self, page_content, metadata, score: float = 0.0):
        self.page_content = page_content
        self.metadata = metadata
        self.score = score

class SimpleVectorStore:
    def __init__(self, path: str = None):
        self.path = path
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.embeddings = None
        if self.path and os.path.exists(self.path):
            self._load()

    def _load(self):
        if self.path and os.path.exists(self.path):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self.ids = data.get("ids", [])
            self.documents = data.get("documents", [])
            self.metadatas = data.get("metadatas", [])
            self.embeddings = data.get("embeddings", None)

    def _save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({
                "ids": self.ids,
                "documents": self.documents,
                "metadatas": self.metadatas,
                "embeddings": self.embeddings,
            }, f)

    def add(self, ids: list, documents: list, metadatas: list, embeddings: list, persist: bool = False):
        new_embeddings = np.array(embeddings, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        # Only save to disk if explicitly commanded (e.g. build_db.py)
        if persist and self.path:
            self._save()

    def count(self) -> int:
        return len(self.ids)

    def get_indexed_sources(self) -> dict:
        summary = {}
        for meta in self.metadatas:
            src = meta.get("source", "Unknown")
            if src not in summary:
                summary[src] = {
                    "chunks": 0,
                    "company": meta.get("company", "Unknown"),
                    "year": meta.get("year", "Unknown"),
                    "quarter": meta.get("quarter", "Unknown"),
                }
            summary[src]["chunks"] += 1
        return summary

    def query(self, query_embedding: list, top_k: int, active_sources: list = None):
        if self.embeddings is None or len(self.ids) == 0:
            return []

        valid_indices = []
        for i, meta in enumerate(self.metadatas):
            src = meta.get("source")
            if active_sources is not None and src not in active_sources:
                continue
            valid_indices.append(i)
                
        if not valid_indices:
            return []

        valid_indices = np.array(valid_indices)
        valid_embeddings = self.embeddings[valid_indices]

        q = np.array(query_embedding, dtype=np.float32)
        doc_norms = valid_embeddings / (np.linalg.norm(valid_embeddings, axis=1, keepdims=True) + 1e-10)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        similarities = doc_norms @ q_norm
        distances = 1 - similarities

        k = min(top_k, len(valid_indices))
        top_local_indices = np.argsort(distances)[:k]
        
        results = []
        for local_idx in top_local_indices:
            global_idx = valid_indices[local_idx]
            dist = float(distances[local_idx])
            results.append(Document(
                page_content=self.documents[global_idx],
                metadata=self.metadatas[global_idx],
                score=dist
            ))
        return results

def load_existing_vector_store():
    return SimpleVectorStore("vector_db/store.pkl")