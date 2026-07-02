"""
Query the engineering knowledge base with retrieval-augmented generation.

Supports two LLM backends, selected via the LLM_PROVIDER env var:
  - "groq" (default): free tier, no billing required. Get a key at
    console.groq.com -> API Keys. Set GROQ_API_KEY.
  - "anthropic": requires a funded Anthropic Console account.
    Set ANTHROPIC_API_KEY.

Usage:
    export LLM_PROVIDER=groq
    export GROQ_API_KEY=gsk_...
    python src/query.py "What diameter is the bearing boss?"

Or run with no arguments for an interactive loop.
"""

import os
import sys
import chromadb
from embeddings import LocalEmbedder

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
TOP_K = 4

PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = "llama-3.3-70b-versatile"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an engineering assistant answering questions strictly \
from the provided reference excerpts (a SolidWorks build guide for a Baja \
racing rear knuckle). Rules:
- Only use information present in the excerpts below. Do not use outside knowledge.
- If the excerpts don't contain the answer, say so plainly — do not guess.
- When you state a dimension, tolerance, or coordinate, quote it exactly as written.
- Keep answers short and direct, like you're briefing a teammate mid-build.
"""


def get_client():
    """Return (provider_name, client) based on LLM_PROVIDER."""
    if PROVIDER == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            print("Set GROQ_API_KEY environment variable first (free key at console.groq.com).")
            sys.exit(1)
        from groq import Groq
        return "groq", Groq()
    elif PROVIDER == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Set ANTHROPIC_API_KEY environment variable first.")
            sys.exit(1)
        from anthropic import Anthropic
        return "anthropic", Anthropic()
    else:
        print(f"Unknown LLM_PROVIDER '{PROVIDER}'. Use 'groq' or 'anthropic'.")
        sys.exit(1)


def call_llm(provider, client, user_msg):
    if provider == "groq":
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return response.choices[0].message.content
    else:  # anthropic
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text


def retrieve(collection, question, embedder, k=TOP_K):
    qvec = embedder.transform([question])
    results = collection.query(query_embeddings=qvec, n_results=k)
    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(chunks, metas))


def build_context(retrieved):
    parts = []
    for i, (chunk, meta) in enumerate(retrieved, 1):
        parts.append(f"[Excerpt {i} — {meta['source']}, chunk {meta['chunk_index']}]\n{chunk}")
    return "\n\n".join(parts)


def answer(question, collection, provider, client, embedder):
    retrieved = retrieve(collection, question, embedder)
    context = build_context(retrieved)

    user_msg = f"""Reference excerpts:

{context}

Question: {question}"""

    response_text = call_llm(provider, client, user_msg)
    return response_text, retrieved


def main():
    provider, client = get_client()

    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = chroma_client.get_collection("engineering_docs", embedding_function=None)
    except Exception:
        print("No collection found. Run `python src/ingest.py` first.")
        sys.exit(1)

    try:
        embedder = LocalEmbedder.load()
    except FileNotFoundError:
        print("No saved embedding model found. Run `python src/ingest.py` first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        response_text, retrieved = answer(question, collection, provider, client, embedder)
        print("\n" + response_text)
        print("\n--- sources ---")
        for _, meta in retrieved:
            print(f"  {meta['source']} (chunk {meta['chunk_index']})")
    else:
        print(f"Engineering RAG Assistant (provider: {provider}). Type 'exit' to quit.\n")
        while True:
            question = input("Q: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
            response_text, retrieved = answer(question, collection, provider, client, embedder)
            print("\n" + response_text)
            print("--- sources:", ", ".join(m["source"] for _, m in retrieved), "---\n")


if __name__ == "__main__":
    main()

