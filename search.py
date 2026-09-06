import os
import streamlit as st
from dotenv import load_dotenv
from vector_store import load_existing_vector_store

load_dotenv()

@st.cache_resource
def get_cached_vectorstore():
    return load_existing_vector_store()

def retrieve_documents(query: str, active_sources: list = None):
    """Executes semantic search strictly within the user's selected sidebar files."""
    vectorstore = get_cached_vectorstore()
    
    # K=15 pulls enough chunks to compare multiple files simultaneously
    search_kwargs = {"k": 15, "active_sources": active_sources}
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)