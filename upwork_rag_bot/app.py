"""
app.py
------
Streamlit UI for the Upwork API Technical Support Bot.
Run with: streamlit run app.py
"""

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import setup_pipeline, load_vector_store, retrieve_chunks
from llm_connector import answer_question

load_dotenv()

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="🤖",
    layout="wide",
)

# -------------------------------------------------------
# Custom CSS — clean dark terminal aesthetic
# -------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }

    .main-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }

    .answer-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #e6edf3;
        margin: 1rem 0;
    }

    .source-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #8b949e;
        margin: 0.4rem 0;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .metric-pill {
        display: inline-block;
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #34d399;
        margin-right: 0.5rem;
    }

    .source-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #58a6ff;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #f0883e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.5rem 0 0.5rem 0;
    }

    .stTextInput > div > div > input {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        font-family: 'Inter', sans-serif !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background: #1f6feb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: background 0.2s !important;
    }

    .stButton > button:hover {
        background: #388bfd !important;
    }

    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #3fb950;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown('<div class="main-header">🤖 Upwork API Support Bot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '<span class="status-indicator"></span>'
    'RAG-powered · Meta-LLaMA 3.1 · Upwork API Docs'
    '</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------------
# Sidebar — Setup & Config
# -------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Setup")

    pdf_path = st.text_input(
        "PDF Path",
        value="API_Documentation_Partial.pdf",
        help="Path to the Upwork API documentation PDF"
    )
    db_dir = st.text_input("ChromaDB directory", value="./chroma_db")

    if st.button("🔧 Build / Reload Vector Store"):
        with st.spinner("Building vector store..."):
            try:
                st.session_state["vector_store"] = setup_pipeline(pdf_path, db_dir)
                st.success("✅ Vector store ready!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.markdown("### 💡 Sample Questions")
    samples = [
        "How long is an OAuth access token valid for?",
        "What is the rate limit for the Upwork API?",
        "Can Client Credentials Grant access private contract details?",
        "What grant types does Upwork OAuth2 support?",
        "How do I refresh an expired access token?",
    ]
    for q in samples:
        if st.button(q, key=f"sample_{q[:20]}"):
            st.session_state["prefill_query"] = q

    st.divider()
    st.markdown(
        "<small style='color:#8b949e;'>"
        "Model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo<br>"
        "Embeddings: all-MiniLM-L6-v2<br>"
        "Vector DB: ChromaDB"
        "</small>",
        unsafe_allow_html=True
    )


# -------------------------------------------------------
# Auto-load vector store on first run
# -------------------------------------------------------
if "vector_store" not in st.session_state:
    if Path(db_dir).exists():
        with st.spinner("Loading existing vector store..."):
            try:
                st.session_state["vector_store"] = load_vector_store(db_dir)
            except Exception:
                st.warning("⚠️ No vector store found. Use the sidebar to build one.")
    else:
        st.info("👈 Click **Build / Reload Vector Store** in the sidebar to get started.")


# -------------------------------------------------------
# Main Chat Interface
# -------------------------------------------------------
prefill = st.session_state.pop("prefill_query", "")
query = st.text_input(
    "Ask a question about the Upwork API:",
    value=prefill,
    placeholder="e.g. How long is an OAuth access token valid?",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask →")


if submit and query.strip():
    if "vector_store" not in st.session_state:
        st.error("Please build the vector store first (sidebar).")
    else:
        with st.spinner("Retrieving context & generating answer..."):
            try:
                # B1: Retrieve top 3 chunks
                chunks = retrieve_chunks(query, st.session_state["vector_store"], k=3)

                # B2: Generate answer with LLM
                result = answer_question(query, chunks)

                # ---- Display Answer ----
                st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True
                )

                # ---- Metrics ----
                st.markdown(
                    f'<span class="metric-pill">⏱ {result["latency"]}s</span>'
                    f'<span class="metric-pill">📄 {len(chunks)} sources</span>',
                    unsafe_allow_html=True
                )

                # ---- Sources ----
                st.markdown('<div class="section-title">Sources Used</div>', unsafe_allow_html=True)
                for i, doc in enumerate(result["sources"], 1):
                    page = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source", "Upwork API Docs")
                    st.markdown(
                        f'<div class="source-label">📎 Source {i} — Page {page}</div>'
                        f'<div class="source-box">{doc.page_content}</div>',
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

elif submit and not query.strip():
    st.warning("Please enter a question.")


# -------------------------------------------------------
# Chat history (optional session state)
# -------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []
