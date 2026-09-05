from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Use a fast, free open-source embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "./vector_db"

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_and_save_vector_store(chunks):
    """Run this once to build the database from your document chunks."""
    embeddings = get_embedding_model()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

def load_existing_vector_store():
    """Run this in your app to quickly load the pre-built database."""
    embeddings = get_embedding_model()
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vectorstore