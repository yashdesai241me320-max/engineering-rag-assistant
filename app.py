"""
Streamlit UI for the Engineering RAG Assistant.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import chromadb
from embeddings import LocalEmbedder
from query import answer, get_client, DB_DIR, PROVIDER

st.set_page_config(page_title="Engineering RAG Assistant", page_icon="🔧", layout="centered")

st.title("🔧 Engineering RAG Assistant")
st.caption(
    "Ask questions about the Baja NITK rear knuckle SolidWorks build guide. "
    "Answers are grounded strictly in the ingested document — no hallucinated dimensions."
)
st.caption(f"LLM provider: **{PROVIDER}**")

key_env = "GROQ_API_KEY" if PROVIDER == "groq" else "ANTHROPIC_API_KEY"
api_key = os.environ.get(key_env)
if not api_key:
    typed_key = st.text_input(f"{key_env}", type="password", help="Not stored, used only for this session")
    if typed_key:
        os.environ[key_env] = typed_key
        api_key = typed_key

if not api_key:
    st.info(f"Enter your {key_env} to start, or set it as an env var before launching.")
    st.stop()


@st.cache_resource
def load_backend(_api_key):
    provider, client = get_client()
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = chroma_client.get_collection("engineering_docs", embedding_function=None)
        embedder = LocalEmbedder.load()
    except Exception:
        # Knowledge base not built yet (e.g. fresh deploy where chroma_db/
        # isn't committed to git) — build it now from data/*.txt and *.pdf.
        with st.spinner("First run: building knowledge base from source documents..."):
            import ingest
            ingest.main()
        collection = chroma_client.get_collection("engineering_docs", embedding_function=None)
        embedder = LocalEmbedder.load()
    return provider, client, collection, embedder


try:
    provider, client, collection, embedder = load_backend(api_key)
except Exception as e:
    st.error(f"Could not build or load knowledge base. Check that data/*.txt or *.pdf files exist. ({e})")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for q, a, sources in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        st.caption("Sources: " + ", ".join(sources))

question = st.chat_input("e.g. What diameter is the bearing boss?")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            response_text, retrieved = answer(question, collection, provider, client, embedder)
            sources = [f"{m['source']} (chunk {m['chunk_index']})" for _, m in retrieved]
        st.write(response_text)
        st.caption("Sources: " + ", ".join(sources))
    st.session_state.history.append((question, response_text, sources))
