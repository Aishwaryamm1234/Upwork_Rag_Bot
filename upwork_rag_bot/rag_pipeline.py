"""
rag_pipeline.py
---------------
Core RAG pipeline for the Upwork API Technical Support Bot.
Handles: PDF loading → chunking → embedding → vector storage → retrieval
"""

import os
import time
from pathlib import Path

# --- Document Loading ---
from langchain_community.document_loaders import PyPDFLoader

# --- Text Splitting ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Embeddings (local, no API needed) ---
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Vector Store ---
from langchain_community.vectorstores import Chroma

# -------------------------------------------------------
# STEP A1: Load the PDF documentation
# -------------------------------------------------------
def load_documents(pdf_path: str) -> list:
    """
    Load a PDF file and return a list of LangChain Document objects.
    Each page becomes its own Document with metadata (page number, source).
    """
    print(f"[A1] Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # --- Sanity Check ---
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"[A1] ✅ Total pages loaded   : {len(documents)}")
    print(f"[A1] ✅ Total character count: {total_chars}")
    print(f"[A1] ✅ Sample text (first 300 chars):\n{documents[0].page_content[:300]}\n")

    return documents


# -------------------------------------------------------
# STEP A2: Chunk the documents
# -------------------------------------------------------
def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    Why overlap matters for technical docs:
    - Code snippets, endpoint URLs, and parameter descriptions often span
      sentence boundaries. A 50-char overlap ensures that the end of one
      chunk and the start of the next share context, so a retrieval hit
      on either chunk still carries the full meaningful unit (e.g., a
      curl command split across a chunk boundary won't lose its endpoint URL).
    """
    print(f"[A2] Chunking documents (size={chunk_size}, overlap={chunk_overlap}) ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[A2] ✅ Total chunks created: {len(chunks)}\n")
    return chunks


# -------------------------------------------------------
# STEP A3: Embed and store in ChromaDB
# -------------------------------------------------------
def build_vector_store(chunks: list, persist_dir: str = "./chroma_db") -> Chroma:
    """
    Convert text chunks to vectors using a local HuggingFace model,
    then persist them in a ChromaDB collection on disk.

    Uses 'sentence-transformers/all-MiniLM-L6-v2':
    - Lightweight (~80MB), runs entirely locally
    - Good semantic similarity for English technical text
    - No API key required for embedding
    """
    print("[A3] Loading local embedding model (all-MiniLM-L6-v2) ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print(f"[A3] Building ChromaDB vector store at '{persist_dir}' ...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"[A3] ✅ Vector store built and persisted.\n")
    return vector_store


# -------------------------------------------------------
# Load existing ChromaDB (skip re-embedding if already built)
# -------------------------------------------------------
def load_vector_store(persist_dir: str = "./chroma_db") -> Chroma:
    """
    Load a previously built ChromaDB from disk.
    Call this instead of build_vector_store() if the DB already exists.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return vector_store


# -------------------------------------------------------
# STEP B1: Semantic Retrieval
# -------------------------------------------------------
def retrieve_chunks(query: str, vector_store: Chroma, k: int = 3) -> list:
    """
    Given a natural language query, return the top-k most semantically
    similar document chunks from the vector store.

    Returns a list of LangChain Document objects.
    Each has .page_content (the text) and .metadata (source, page).
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)
    return results


# -------------------------------------------------------
# Full pipeline runner (for first-time setup)
# -------------------------------------------------------
def setup_pipeline(pdf_path: str, persist_dir: str = "./chroma_db") -> Chroma:
    """
    Run the full A1→A2→A3 pipeline once to build the vector store.
    After this, call load_vector_store() on subsequent runs.
    """
    docs   = load_documents(pdf_path)
    chunks = chunk_documents(docs)
    vs     = build_vector_store(chunks, persist_dir)
    return vs


if __name__ == "__main__":
    # Quick test — change the path to your actual PDF
    PDF_PATH = "API_Documentation_Partial.pdf"
    DB_DIR   = "./chroma_db"

    if Path(DB_DIR).exists():
        print("Vector store already exists — loading from disk.")
        vs = load_vector_store(DB_DIR)
    else:
        vs = setup_pipeline(PDF_PATH, DB_DIR)

    # Test retrieval
    query   = "How long is an OAuth access token valid?"
    results = retrieve_chunks(query, vs)
    print(f"\n[B1] Top {len(results)} chunks for query: '{query}'\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Chunk {i} (page {doc.metadata.get('page', '?')}) ---")
        print(doc.page_content[:200])
        print()
