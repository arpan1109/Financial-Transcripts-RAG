import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF

from vector_store import load_existing_vector_store, get_embedder, SimpleVectorStore
from build_db import chunk_text, extract_financial_metadata
from search import retrieve_documents
from llm import reformulate_query, generate_streaming_answer, extract_kpis, parse_stream

load_dotenv()

st.set_page_config(
    page_title="FinSight AI — Financial RAG Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

base_store = load_existing_vector_store()

# --------------------------------------------------
# Styling (Replit UI Theme)
# --------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #090b12; color: #e5e7eb; }
    [data-testid="stSidebar"] { background: #10131d; border-right: 1px solid rgba(255,255,255,.07); }
    [data-testid="stSidebar"] > div:first-child { padding: 1.25rem 1rem; }
    .block-container { max-width: 1100px; padding: 2rem 3rem 5rem; }
    .logo { display: flex; align-items: center; gap: .7rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,.06); }
    .logo-mark { width: 34px; height: 34px; display: grid; place-items: center; border-radius: 12px; background: linear-gradient(135deg, #e879f9, #8b5cf6); color: white; font-weight: 800; }
    .logo-name { color: #fff; font-size: 15px; font-weight: 700; }
    .logo-sub { margin-top: 2px; color: #64748b; font-size: 10px; }
    .section-kicker { color: #64748b; font-size: 10px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .helper { margin: .35rem 0 1rem; color: #94a3b8; font-size: 12px; }
    .source-info { margin: .5rem 0 1rem; padding: .8rem; border: 1px solid rgba(167,139,250,.16); border-radius: 16px; background: rgba(167,139,250,.07); }
    .source-info.private { border-color: rgba(251,191,36,.16); background: rgba(251,191,36,.06); }
    .source-info b { color: #f1f5f9; font-size: 12px; }
    .source-info p { margin: .35rem 0 0; color: #64748b; font-size: 10px; line-height: 1.5; }
    .context { margin: 1rem -1rem 0; padding: 1rem; border-top: 1px solid rgba(255,255,255,.06); }
    [data-testid="stTabs"] button { color: #64748b; font-size: 11px; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #f5f3ff; }
    .stButton > button { border: 1px solid rgba(255,255,255,.08); border-radius: 11px; background: rgba(255,255,255,.04); color: #cbd5e1; }
    .stButton > button:hover { border-color: rgba(167,139,250,.45); color: #fff; }
    .primary .stButton > button { border: 0; background: #a78bfa; color: #171321; font-weight: 700; }
    [data-testid="stChatMessage"] { border: 1px solid rgba(255,255,255,.07); border-radius: 16px; background: rgba(255,255,255,.025); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
def initialize_state():
    if "library_selected" not in st.session_state:
        st.session_state.library_selected = set()
    if "user_files" not in st.session_state:
        st.session_state.user_files = {}
    if "user_selected" not in st.session_state:
        st.session_state.user_selected = set()
    if "user_store" not in st.session_state:
        # PURE IN-MEMORY STORE FOR THIS SESSION ONLY (NEVER WRITES TO DISK)
        st.session_state.user_store = SimpleVectorStore(path=None)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kpi_snapshot" not in st.session_state:
        st.session_state.kpi_snapshot = None

initialize_state()

def selected_source_names() -> list[str]:
    return [*sorted(st.session_state.library_selected), *sorted(st.session_state.user_selected)]

def total_context_chunk_count() -> int:
    lib_sources = base_store.get_indexed_sources()
    lib_chunks = sum(lib_sources[name]["chunks"] for name in st.session_state.library_selected if name in lib_sources)
    
    usr_sources = st.session_state.user_store.get_indexed_sources()
    usr_chunks = sum(usr_sources[name]["chunks"] for name in st.session_state.user_selected if name in usr_sources)
    
    return lib_chunks + usr_chunks

def upload_metadata(uploaded_file) -> str:
    extension = Path(uploaded_file.name).suffix.replace(".", "").upper() or "FILE"
    size_kb = max(1, uploaded_file.size // 1024)
    return f"{extension} · {size_kb} KB"

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="logo">
            <div class="logo-mark">📈</div>
            <div>
                <div class="logo-name">FinSight AI</div>
                <div class="logo-sub">Grounded financial answers, every time</div>
            </div>
        </div>
        <div style="height:1.2rem"></div>
        <div class="section-kicker">Context sources</div>
        <div class="helper">Choose what the assistant can see</div>
        """,
        unsafe_allow_html=True,
    )

    knowledge_tab, uploads_tab = st.tabs(["Knowledge base", "Your files"])

    with knowledge_tab:
        st.markdown(
            '<div class="source-info"><b>Shared knowledge base</b><p>Curated earnings transcripts grouped by company.</p></div>',
            unsafe_allow_html=True,
        )
        search = st.text_input("Search", placeholder="Search documents...", label_visibility="collapsed", key="lib_search")
        indexed_library = base_store.get_indexed_sources()

        companies = {}
        for name, info in indexed_library.items():
            comp = info.get("company", "Unknown")
            companies.setdefault(comp, []).append((name, info))

        for company, files in sorted(companies.items()):
            matching_files = [
                (n, inf) for n, inf in files 
                if search.lower() in n.lower() or search.lower() in company.lower() or search.lower() in inf.get('quarter', '').lower()
            ]
            if not matching_files and search:
                continue

            with st.expander(f"📁 {company} ({len(matching_files)} quarters)", expanded=True):
                col_btn1, col_btn2 = st.columns(2)
                
                if col_btn1.button("Select All", key=f"sel_all_{company}", use_container_width=True):
                    for n, _ in matching_files:
                        st.session_state[f"chk_{n}"] = True
                        st.session_state.library_selected.add(n)
                    st.rerun()
                    
                if col_btn2.button("Deselect All", key=f"desel_all_{company}", use_container_width=True):
                    for n, _ in matching_files:
                        st.session_state[f"chk_{n}"] = False
                        st.session_state.library_selected.discard(n)
                    st.rerun()

                for name, info in sorted(matching_files, key=lambda x: (x[1].get("year", ""), x[1].get("quarter", ""))):
                    display_label = f"{info.get('quarter', 'Q?')} {info.get('year', '20XX')} · {info.get('chunks')} chunks"
                    
                    if f"chk_{name}" not in st.session_state:
                        st.session_state[f"chk_{name}"] = name in st.session_state.library_selected

                    checked = st.checkbox(display_label, key=f"chk_{name}")
                    if checked:
                        st.session_state.library_selected.add(name)
                    else:
                        st.session_state.library_selected.discard(name)

    with uploads_tab:
        st.markdown(
            '<div class="source-info private"><b>Private session files</b><p>Upload files on the fly. Ingested strictly into temporary RAM.</p></div>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader("Add files", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed", key="file_uploader")
        if uploaded_files:
            embedder = get_embedder()
            for uploaded_file in uploaded_files:
                fname = uploaded_file.name
                if fname not in st.session_state.user_files:
                    st.session_state.user_files[fname] = uploaded_file
                    st.session_state.user_selected.add(fname)
                    
                    company, year, quarter = extract_financial_metadata(fname)
                    if fname.lower().endswith(".pdf"):
                        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc) if page.get_text().strip()]
                        doc.close()
                    else:
                        pages = [(1, uploaded_file.read().decode("utf-8", errors="ignore"))]

                    ids, docs, metadatas = [], [], []
                    for page_num, page_text in pages:
                        for chunk in chunk_text(page_text):
                            if len(chunk.strip()) < 20: continue
                            ids.append(str(uuid.uuid4()))
                            docs.append(chunk)
                            metadatas.append({"source": fname, "page": page_num, "company": company, "year": year, "quarter": quarter})
                    
                    # Store ONLY in temporary session RAM (persist=False)
                    if docs:
                        embeddings = embedder.encode(docs, show_progress_bar=False).tolist()
                        st.session_state.user_store.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings, persist=False)

        for name, uploaded_file in st.session_state.user_files.items():
            if f"upload_{name}" not in st.session_state:
                st.session_state[f"upload_{name}"] = name in st.session_state.user_selected
                
            checked = st.checkbox(f"{name}\n{upload_metadata(uploaded_file)}", key=f"upload_{name}")
            if checked:
                st.session_state.user_selected.add(name)
            else:
                st.session_state.user_selected.discard(name)

        st.markdown("<br>", unsafe_allow_html=True)
        # INSTANT ZERO-CONTAMINATION RESET
        if st.button("🗑️ Clear Uploaded Files", use_container_width=True):
            st.session_state.user_files = {}
            st.session_state.user_selected = set()
            st.session_state.user_store = SimpleVectorStore(path=None)
            for k in list(st.session_state.keys()):
                if k.startswith("upload_"):
                    del st.session_state[k]
            st.rerun()

    selected_count = len(selected_source_names())
    total_chunks = total_context_chunk_count()

    st.markdown(
        f'<div class="context"><div class="section-kicker">Current context</div><div class="helper">{total_chunks:,} indexed chunks · {selected_count} sources selected</div></div>',
        unsafe_allow_html=True,
    )

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.kpi_snapshot = None
        st.rerun()

# --------------------------------------------------
# Main area
# --------------------------------------------------
st.title("Financial Intelligence Brief")
st.caption(f"{selected_count} context sources active")

chat_tab, doc_tab = st.tabs(["💬 Copilot Chat", "📊 Document Explorer"])

with chat_tab:
    if not st.session_state.messages:
        st.info("👋 Select your sources in the sidebar and ask an analytical question!")

    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🧠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

with doc_tab:
    if st.session_state.kpi_snapshot:
        snap = st.session_state.kpi_snapshot
        st.subheader(f"📊 {snap['company']} Snapshot ({snap['quarter']} {snap['year']})")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", snap["kpis"].get("revenue", "N/A"), border=True)
        c2.metric("Operating Margin", snap["kpis"].get("margin", "N/A"), border=True)
        c3.metric("Net Income", snap["kpis"].get("net_income", "N/A"), border=True)
        c4.metric("Diluted EPS", snap["kpis"].get("eps", "N/A"), border=True)
        
        st.markdown("---")
        st.subheader("Retrieved Context Chunks")
        for i, text in enumerate(snap["sources"]):
            with st.expander(f"Source Chunk {i+1}", expanded=(i == 0)):
                st.write(text)
    else:
        st.info("Execute a financial query in chat to populate this interactive dashboard.")

# --------------------------------------------------
# RAG Execution Pipeline
# --------------------------------------------------
if prompt := st.chat_input("Ask a question about your sources..."):
    source_names = selected_source_names()
    
    if not source_names:
        st.warning("⚠️ Please select at least one source from the sidebar before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_tab:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(prompt)

            with st.chat_message("assistant", avatar="🧠"):
                try:
                    with st.spinner("Analyzing context..."):
                        final_query = reformulate_query(prompt, st.session_state.messages)
                        
                        # Searches library sources and user sources concurrently without disk interference
                        docs = retrieve_documents(
                            final_query, 
                            library_sources=list(st.session_state.library_selected),
                            user_sources=list(st.session_state.user_selected),
                            user_store=st.session_state.user_store
                        )

                    if docs:
                        raw_stream = generate_streaming_answer(final_query, docs, st.session_state.messages)
                        answer = st.write_stream(parse_stream(raw_stream))
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                        top_doc = docs[0]
                        top_context = "\n".join([d.page_content for d in docs[:4]])
                        kpi_json = extract_kpis(top_context)
                        
                        st.session_state.kpi_snapshot = {
                            "company": top_doc.metadata.get("company", "Selected Sources"),
                            "year": top_doc.metadata.get("year", ""),
                            "quarter": top_doc.metadata.get("quarter", ""),
                            "kpis": kpi_json,
                            "sources": [d.page_content for d in docs[:6]]
                        }
                        st.rerun()
                    else:
                        answer = "I could not locate any relevant information within your currently selected sidebar sources."
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error processing query: {e}")