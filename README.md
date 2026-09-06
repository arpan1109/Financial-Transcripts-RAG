# 📈 FinSight AI — High-Performance Financial RAG Copilot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finsight-007.streamlit.app)
[![Groq API](https://img.shields.io/badge/LLM%20Inference-Groq-f55036.svg)](https://groq.com)
[![Embeddings](https://img.shields.io/badge/Embeddings-Sentence--Transformers-blue.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Demo:** [finsight-007.streamlit.app](https://finsight-007.streamlit.app)

**FinSight AI** is a financial intelligence copilot built for institutional-grade equity research across corporate earnings call transcripts. It delivers sub-millisecond retrieval speeds by utilizing an optimized, in-memory NumPy vector engine and streaming inference powered by Groq.

---

## ⚡ Key Highlights & Architecture

- **Sub-Millisecond Vector Retrieval:** Bypasses heavy disk-based vector databases in favor of an in-memory NumPy cosine similarity engine (`SimpleVectorStore`).
- **Dual-Store Source Scoping:**
  - **Knowledge Base (Persistent):** Pre-indexed earnings transcripts (Alphabet, Microsoft, Nvidia) grouped hierarchically by company and quarter.
  - **Private Session Files (Ephemeral):** Ad-hoc upload support for PDFs and TXTs, ingested directly into RAM for instant querying and one-click purging without base data contamination.
- **Rule-Based Conversational Memory:** Fast, deterministic follow-up query reformulation without LLM latency penalties.
- **Strict Grounding & Inline Citations:** Context-enforced answers tagged with bracketed source markers (`[1]`, `[2]`) linked directly to exact document chunks.
- **Executive KPI Dashboard:** Interactive secondary dashboard providing automated extraction of revenue, margins, net income, and diluted EPS.

---
## Folder Structure
<img width="940" height="534" alt="image" src="https://github.com/user-attachments/assets/8c18b88f-631a-465a-ab43-e556420df8d2" />

## 🏗️ System Architecture

```text
User Query ──► Query Reformulation (Memory Heuristic)
                     │
                     ▼
          Target Vector Embedding (all-MiniLM-L6-v2)
                     │
                     ▼
     In-Memory NumPy Dual-Store Cosine Distance
      ├── Persistent Knowledge Base (store.pkl)
      └── Ephemeral Session Uploads (RAM)
                     │
                     ▼
         Top Relevant Chunks (with Metadata)
                     │
                     ▼
       Groq API Streaming Inference (LLaMA / Mixtral)
                     │
                     ▼
   Grounded Answer Stream with [n] Citations + KPI Extraction

#
