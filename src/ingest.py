"""
Ingest engineering documents into a local vector store (ChromaDB).

Usage:
    python src/ingest.py
This will read all .txt files from ./data and build a persistent
Chroma collection at ./chroma_db.

Chunking strategy: paragraph-aware sliding window. Engineering guides
are structured as short instructional paragraphs (one action per line),
so we group N paragraphs per chunk with overlap, instead of splitting
by raw character count, which would cut a step in half.
"""

"""
Ingest engineering documents into a local vector store (ChromaDB).

Usage:
    python src/ingest.py
This will read all .txt and .pdf files from ./data and build a persistent
Chroma collection at ./chroma_db.

Chunking strategy: paragraph-aware sliding window. Engineering guides
are structured as short instructional paragraphs (one action per line),
so we group N paragraphs per chunk with overlap, instead of splitting
by raw character count, which would cut a step in half. PDFs (e.g.
research papers) are extracted page-by-page and split on blank lines
to approximate paragraphs before the same chunker is applied.
"""

import os
import glob
import chromadb
from embeddings import LocalEmbedder

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

CHUNK_SIZE = 6       # paragraphs per chunk
CHUNK_OVERLAP = 2    # paragraphs of overlap between chunks


def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_pdf(path):
    if not HAS_PDFPLUMBER:
        print(f"Skipping {path}: install pdfplumber to ingest PDFs (`pip install pdfplumber`).")
        return []
    paragraphs = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Split on blank lines to approximate paragraphs; fall back to
            # single newlines if the PDF has no blank-line structure.
            blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
            if len(blocks) <= 1:
                blocks = [line.strip() for line in text.split("\n") if line.strip()]
            paragraphs.extend(blocks)
    return paragraphs


def load_documents():
    """Return list of (filename, list_of_paragraphs) for .txt and .pdf files."""
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        docs.append((os.path.basename(path), load_txt(path)))
    for path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        paragraphs = load_pdf(path)
        if paragraphs:
            docs.append((os.path.basename(path), paragraphs))
    return docs


def chunk_paragraphs(paragraphs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Sliding window chunker over a list of paragraphs."""
    chunks = []
    step = size - overlap
    for i in range(0, len(paragraphs), step):
        window = paragraphs[i : i + size]
        if not window:
            continue
        chunks.append(" ".join(window))
        if i + size >= len(paragraphs):
            break
    return chunks


def main():
    docs = load_documents()
    if not docs:
        print(f"No .txt or .pdf files found in {DATA_DIR}. Add source documents and re-run.")
        return

    client = chromadb.PersistentClient(path=DB_DIR)
    # Reset collection each run so re-ingesting is idempotent
    try:
        client.delete_collection("engineering_docs")
    except Exception:
        pass
    # embedding_function=None: we supply our own vectors explicitly below
    collection = client.create_collection("engineering_docs", embedding_function=None)

    all_chunks, all_ids, all_meta = [], [], []
    for fname, paragraphs in docs:
        chunks = chunk_paragraphs(paragraphs)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{fname}::chunk_{idx}")
            all_meta.append({"source": fname, "chunk_index": idx})
        print(f"{fname}: {len(paragraphs)} paragraphs -> {len(chunks)} chunks")

    embedder = LocalEmbedder()
    vectors = embedder.fit_transform(all_chunks)
    embedder.save()

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta, embeddings=vectors)

    print(f"\nIngested {len(all_chunks)} chunks from {len(docs)} document(s) into {DB_DIR}")


if __name__ == "__main__":
    main()
