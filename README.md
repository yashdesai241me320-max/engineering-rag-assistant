# Engineering RAG Assistant

A retrieval-augmented Q&A tool that answers questions grounded strictly in engineering
build documentation — built on a real SolidWorks fabrication guide for a Baja SAE
rear knuckle (bearing bores, hardpoint coordinates, tolerances, material specs).

**Live demo:** [add your Streamlit Cloud link here after deploying]

## Why this exists

LLMs are unreliable when asked for exact dimensions or tolerances from memory — they
paraphrase, round numbers, or invent plausible-sounding specs. This assistant instead
retrieves the exact source paragraph before answering, so every dimension quoted is
traceable back to the original document, with the source chunk cited.

## How it works

```
docs/*.txt  →  chunk (paragraph sliding window)  →  embed (local TF-IDF+SVD)
            →  store in ChromaDB  →  query  →  retrieve top-k chunks
            →  Claude answers using ONLY retrieved context, with citations
```

- **Chunking:** paragraph-aware sliding window (6 paragraphs/chunk, 2 overlap) —
  splits on structure, not raw character count, so a build step is never cut in half.
- **Embeddings:** local TF-IDF + Truncated SVD, no external model download required.
  This was a deliberate choice: it keeps the pipeline runnable in network-restricted
  environments and performs well on domain-specific technical vocabulary (part names,
  tolerances) — arguably better here than generic sentence embeddings tuned for
  conversational text. Swappable: replace `embeddings.py` with a
  sentence-transformers or OpenAI embeddings call to upgrade retrieval quality on
  larger/more varied corpora.
- **Generation:** Claude (`claude-sonnet-4-6`) answers using only the retrieved
  excerpts, instructed to say "not in the source" rather than guess.

## Project structure

```
rag_project/
├── data/               # source .txt documents (add your own here)
├── src/
│   ├── embeddings.py   # local TF-IDF+SVD embedding backend
│   ├── ingest.py       # chunk + embed + store pipeline
│   └── query.py        # retrieval + grounded answer generation (CLI)
├── app.py              # Streamlit UI
└── requirements.txt
```

## Setup

Uses [Groq](https://console.groq.com) by default — free tier, no billing required.
Anthropic Claude is also supported as a drop-in alternative (see below).

```bash
pip install -r requirements.txt
export LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_...        # free key at console.groq.com -> API Keys

python src/ingest.py          # build the vector store from data/*.txt
python src/query.py "What diameter is the bearing boss?"   # CLI query

streamlit run app.py          # or launch the web UI
```

To use Anthropic Claude instead:
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Example

```
Q: What fillet radius is used on inside corners?
A: 6mm. Every inside corner where sketch lines meet needs a Sketch Fillet
   set to radius 6mm — sharp corners crack under load.
Sources: knuckle_guide.txt (chunk 14)
```

## Adding your own documents

Drop any `.txt` file into `data/` and re-run `python src/ingest.py`. The pipeline
is document-agnostic — this same setup works for any structured technical
documentation (CAM tooling guides, CNC nesting workflows, assembly instructions).

## Notes on scope

This is intentionally a small, focused MVP rather than a general-purpose RAG
framework — the goal was to demonstrate the full retrieval → grounding →
citation pipeline end-to-end and correctly, not to reimplement LangChain.
