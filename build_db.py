import os
import re
import uuid
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from vector_store import SimpleVectorStore

DATA_DIR = "data"
STORE_PATH = "vector_db/store.pkl"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
def extract_financial_metadata(filename: str):
    name_lower = filename.lower()
    
    company = "Unknown"
    if "goog" in name_lower or "alphabet" in name_lower:
        company = "Alphabet"
    elif "msft" in name_lower or "microsoft" in name_lower:
        company = "Microsoft"
    elif "nvda" in name_lower or "nvidia" in name_lower:
        company = "Nvidia"
        
    year_match = re.search(r'(20\d{2})', name_lower)
    year = year_match.group(1) if year_match else "Unknown"
    
    q_match = re.search(r'(q[1-4])', name_lower)
    quarter = q_match.group(1).upper() if q_match else "Unknown"
    
    return company, year, quarter
def chunk_text(text: str):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= CHUNK_SIZE:
            current = f"{current}\n{para}".strip()
        else:
            if current: chunks.append(current)
            if len(para) > CHUNK_SIZE:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= CHUNK_SIZE:
                        current = f"{current} {sent}".strip()
                    else:
                        if current: chunks.append(current)
                        current = sent
            else:
                current = para
    if current: chunks.append(current)
    
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0 or CHUNK_OVERLAP == 0:
            overlapped.append(c)
        else:
            tail = chunks[i - 1][-CHUNK_OVERLAP:]
            overlapped.append(f"{tail} {c}")
    return overlapped

def build_database():
    os.makedirs(os.path.dirname(STORE_PATH) or ".", exist_ok=True)
    store = SimpleVectorStore(STORE_PATH)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    total_chunks = 0
    
    # FIX: os.walk recursively searches all subfolders inside data/
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.lower().endswith((".pdf", ".txt")): 
                continue
            
            fpath = os.path.join(root, fname)
            company, year, quarter = extract_financial_metadata(fname)
            
            pages = []
            if fname.lower().endswith(".pdf"):
                doc = fitz.open(fpath)
                for i, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip(): pages.append((i + 1, text))
                doc.close()
            else:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    pages.append((1, f.read()))
                    
            ids, docs, metadatas = [], [], []
            for page_num, page_text in pages:
                for chunk in chunk_text(page_text):
                    if len(chunk.strip()) < 20: continue
                    ids.append(str(uuid.uuid4()))
                    docs.append(chunk)
                    metadatas.append({
                        "source": fname, "page": page_num,
                        "company": company, "year": year, "quarter": quarter
                    })
                    
            if docs:
                embeddings = embedder.encode(docs, show_progress_bar=False).tolist()
                store.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
                total_chunks += len(docs)
                print(f"Processed {fname}: {len(docs)} chunks added.")
            
    print(f"Database built successfully! Total chunks: {total_chunks}")
    
if __name__ == "__main__":
    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH) # Wipes old DB to prevent duplicates
    build_database()