import os
import streamlit as st
from dotenv import load_dotenv
from vector_store import load_existing_vector_store, get_embedder

load_dotenv()

@st.cache_resource
def get_cached_vectorstore():
    return load_existing_vector_store()

def retrieve_documents(query: str, library_sources: list = None, user_sources: list = None, user_store = None):
    """Executes semantic search across both persistent library files and temporary session files."""
    embedder = get_embedder()
    q_emb = embedder.encode([query]).tolist()[0]
    
    candidates = []
    
    # 1. Search persistent knowledge base
    if library_sources:
        base_store = get_cached_vectorstore()
        candidates.extend(base_store.query(q_emb, top_k=15, active_sources=library_sources))
        
    # 2. Search session-only uploaded files
    if user_sources and user_store and user_store.count() > 0:
        candidates.extend(user_store.query(q_emb, top_k=15, active_sources=user_sources))
        
    # 3. Sort by cosine distance (lower distance = higher semantic similarity)
    candidates.sort(key=lambda doc: doc.score)
    
    return candidates[:15]