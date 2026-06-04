import os
import time
import requests
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEEPINFRA_API_KEY = st.secrets.get(
    "DEEPINFRA_API_KEY",
    os.getenv("DEEPINFRA_API_KEY")
)

print("Loaded API Key:", "FOUND" if DEEPINFRA_API_KEY else "NOT FOUND")

DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# -------------------------------------------------------
# STEP B2: System Prompt — forces the AI into character
# -------------------------------------------------------
SYSTEM_PROMPT = """You are a Senior Upwork API Consultant with 10+ years of experience.
You help developers integrate with the Upwork API accurately and efficiently.

STRICT RULES you must follow:
1. Answer ONLY using the provided context chunks. Do not use prior knowledge.
2. If the answer is NOT found in the context, respond with exactly:
   "I'm sorry, but the provided documentation does not contain that information."
3. Be precise and technical. Quote relevant details (endpoints, parameters, token TTLs).
4. Do not make up endpoints, parameters, or behaviours not present in the context.
5. If the context partially answers the question, give the partial answer and note what is missing.
"""

# -------------------------------------------------------
# Build the prompt messages list
# -------------------------------------------------------
def build_messages(user_query: str, retrieved_chunks: list) -> list:
    """
    Construct the messages array for the Chat Completions API.

    Structure:
    - system : persona + hallucination guard rules
    - user   : the retrieved context + the user's question

    We inject the retrieved chunks into the user message so the model
    can only reference them — this is the core of RAG.
    """
    # Format the retrieved chunks into a readable context block
    context_text = ""
    for i, doc in enumerate(retrieved_chunks, 1):
        page = doc.metadata.get("page", "?")
        context_text += f"\n--- Source {i} (Page {page}) ---\n{doc.page_content}\n"

    user_message = f"""Use ONLY the following documentation excerpts to answer the question.

CONTEXT:
{context_text}

QUESTION: {user_query}

ANSWER:"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


# -------------------------------------------------------
# STEP B2: Call the DeepInfra API
# -------------------------------------------------------
def call_llm(messages, temperature=0.1, max_tokens=512):

    if not DEEPINFRA_API_KEY:
        raise ValueError(
            "DEEPINFRA_API_KEY not found. Check your .env file."
        )

    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    start_time = time.time()

    try:
        response = requests.post(
            DEEPINFRA_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        latency = time.time() - start_time

        print("Status Code:", response.status_code)

        if response.status_code != 200:
            print(response.text)

        response.raise_for_status()

        data = response.json()

        answer = (
            data["choices"][0]["message"]["content"]
            .strip()
        )

        return {
            "answer": answer,
            "latency": round(latency, 2),
            "raw": data
        }

    except Exception as e:
        raise Exception(
            f"DeepInfra Error: {str(e)}"
        )
    """
    Send messages to the DeepInfra Meta-LLaMA API and return:
    {
        "answer"   : str,   # the model's text response
        "latency"  : float, # seconds the API call took
        "raw"      : dict,  # full API response (for debugging)
    }

    temperature=0.1 keeps responses focused and deterministic —
    important for a support bot where accuracy > creativity.
    """
    if not DEEPINFRA_API_KEY:
        raise ValueError(
            "DEEPINFRA_API_KEY not found. "
            "Please set it in your .env file."
        )

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
    }

    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    start_time = time.time()
    response   = requests.post(DEEPINFRA_API_URL, headers=headers, json=payload, timeout=60)
    latency    = time.time() - start_time

    response.raise_for_status()  # raise HTTPError for 4xx/5xx responses

    data   = response.json()
    answer = data["choices"][0]["message"]["content"].strip()

    return {
        "answer":  answer,
        "latency": round(latency, 2),
        "raw":     data,
    }


# -------------------------------------------------------
# High-level function used by app.py
# -------------------------------------------------------
def answer_question(user_query: str, retrieved_chunks: list) -> dict:
    """
    Full B2 flow:
    1. Build prompt messages from query + retrieved chunks
    2. Call the LLM
    3. Return answer, latency, and the source chunks for display
    """
    messages = build_messages(user_query, retrieved_chunks)
    result   = call_llm(messages)

    return {
        "answer":  result["answer"],
        "latency": result["latency"],
        "sources": retrieved_chunks,
    }


if __name__ == "__main__":
    # Quick standalone test
    from rag_pipeline import load_vector_store, retrieve_chunks

    vs     = load_vector_store("./chroma_db")
    query  = "How long is an OAuth access token valid for?"
    chunks = retrieve_chunks(query, vs)
    result = answer_question(query, chunks)

    print(f"Answer  : {result['answer']}")
    print(f"Latency : {result['latency']}s")
