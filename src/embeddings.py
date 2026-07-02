"""
Lightweight local embedding backend (TF-IDF + Truncated SVD).

Why not sentence-transformers or OpenAI embeddings?
- sentence-transformers needs to download a ~90MB model from HuggingFace on
  first run, which fails in network-restricted environments (CI runners,
  locked-down corp networks, sandboxes). TF-IDF+SVD needs zero downloads
  and is fully deterministic.
- This is a drop-in swap: to upgrade retrieval quality later, replace
  `fit_transform`/`transform` below with a sentence-transformers or
  OpenAI embeddings call and keep the rest of the pipeline unchanged.

For a ~50-page engineering document corpus this is more than adequate —
TF-IDF handles domain-specific technical vocabulary (part names,
tolerances, step numbers) very well, arguably better than generic
sentence embeddings which are tuned for natural conversational text.
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db", "embedding_model.pkl")


class LocalEmbedder:
    def __init__(self, n_components=128):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.fitted = False

    def fit_transform(self, texts):
        tfidf = self.vectorizer.fit_transform(texts)
        # SVD needs n_components < n_samples; clamp for small corpora
        n_comp = min(self.svd.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=max(n_comp, 2), random_state=42)
        embeddings = self.svd.fit_transform(tfidf)
        self.fitted = True
        return self._normalize(embeddings)

    def transform(self, texts):
        if not self.fitted:
            raise RuntimeError("Embedder not fitted. Call fit_transform first or load a saved model.")
        tfidf = self.vectorizer.transform(texts)
        embeddings = self.svd.transform(tfidf)
        return self._normalize(embeddings)

    @staticmethod
    def _normalize(vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (vecs / norms).tolist()

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd, "fitted": self.fitted}, f)

    @classmethod
    def load(cls, path=MODEL_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.svd = data["svd"]
        obj.fitted = data["fitted"]
        return obj
