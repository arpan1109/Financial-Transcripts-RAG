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
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

class SimpleVectorStore:
    def __init__(self, path: str):
        self.path = path
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.embeddings = None
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self.ids = data["ids"]
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]
            self.embeddings = data["embeddings"]

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({
                "ids": self.ids,
                "documents": self.documents,
                "metadatas": self.metadatas,
                "embeddings": self.embeddings,
            }, f)

    def add(self, ids: list, documents: list, metadatas: list, embeddings: list):
        new_embeddings = np.array(embeddings, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._save()

    def count(self) -> int:
        return len(self.ids)

    def get_indexed_sources(self) -> dict:
        """Returns metadata summary for all unique documents in the store."""
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

    def query(self, query_embedding: list, top_k: int, filters: dict = None, active_sources: list = None):
        if self.embeddings is None or len(self.ids) == 0:
            return []

        filter_dict = {}
        if filters:
            if "$and" in filters:
                for f in filters["$and"]:
                    filter_dict.update(f)
            else:
                filter_dict = filters

        # Filter indices by financial metadata AND active sidebar source selection
        valid_indices = []
        for i, meta in enumerate(self.metadatas):
            src = meta.get("source")
            if active_sources is not None and src not in active_sources:
                continue

            match = True
            for k, v in filter_dict.items():
                if meta.get(k) != v:
                    match = False
                    break
            if match:
                valid_indices.append(i)
                
        if not valid_indices:
            return []

        valid_indices = np.array(valid_indices)
        valid_embeddings = self.embeddings[valid_indices]

        # Cosine distance computation in RAM
        q = np.array(query_embedding, dtype=np.float32)
        doc_norms = valid_embeddings / (np.linalg.norm(valid_embeddings, axis=1, keepdims=True) + 1e-10)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        similarities = doc_norms @ q_norm
        distances = 1 - similarities

        k = min(top_k, len(valid_indices))
        top_local_indices = np.argsort(distances)[:k]
        
        return [
            Document(
                page_content=self.documents[valid_indices[idx]],
                metadata=self.metadatas[valid_indices[idx]]
            )
            for idx in top_local_indices
        ]

    def as_retriever(self, search_kwargs=None):
        class Retriever:
            def __init__(self, store, kwargs):
                self.store = store
                self.kwargs = kwargs or {}
            
            def invoke(self, query: str):
                embedder = get_embedder()
                q_emb = embedder.encode([query]).tolist()[0]
                k = self.kwargs.get("k", 40)
                filters = self.kwargs.get("filter", {})
                active_sources = self.kwargs.get("active_sources", None)
                return self.store.query(q_emb, k, filters, active_sources)
                
        return Retriever(self, search_kwargs)

def load_existing_vector_store():
    return SimpleVectorStore("vector_db/store.pkl")