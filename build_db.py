from data_loader import load_and_chunk_txt_data
from vector_store import build_and_save_vector_store

print("Loading documents and creating chunks...")
chunks = load_and_chunk_txt_data("./data")
print(f"Created {len(chunks)} chunks.")

print("Building vector database...")
vectorstore = build_and_save_vector_store(chunks)
print("Database built and persisted to ./vector_db successfully!")