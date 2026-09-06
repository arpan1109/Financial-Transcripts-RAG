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

## 🏗️ System Architecture
