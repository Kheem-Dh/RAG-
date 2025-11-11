"""Retrieval implementations.

This module provides classes for sparse (BM25) retrieval and dense TF–IDF vector
retrieval.  It rebuilds the indices whenever new documents are ingested.  A
hybrid retrieval method combines both scoring schemes.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class RetrievalIndex:
    """A combined BM25 and TF–IDF retrieval index.

    The index is built over all document chunks.  When new chunks are ingested
    the index must be rebuilt to reflect the updated corpus.  Thread locks are
    used to avoid race conditions during rebuild and query.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        # Data structures
        self.chunk_ids: List[int] = []
        self.chunk_texts: List[str] = []
        # BM25 fields
        self.term_freqs: List[Counter[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        # TF‑IDF fields
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        # Locks
        self._build_lock = threading.Lock()
        self._ready_event = threading.Event()

    def build(self, chunks: Iterable[Tuple[int, str]]) -> None:
        """Build both BM25 and TF–IDF indices from a list of `(id, text)` tuples."""
        with self._build_lock:
            # Reset existing data
            self.chunk_ids = []
            self.chunk_texts = []
            self.term_freqs = []
            self.doc_lens = []
            self.idf = {}
            # Populate raw lists
            for cid, text in chunks:
                self.chunk_ids.append(cid)
                self.chunk_texts.append(text)
            n_docs = len(self.chunk_texts)
            if n_docs == 0:
                # Nothing to build
                self.tfidf_matrix = None
                self.vectorizer = None
                self.avgdl = 0.0
                self._ready_event.set()
                return
            # Precompute term frequencies and document lengths
            doc_freq: Dict[str, int] = defaultdict(int)
            self.term_freqs = []
            self.doc_lens = []
            for text in self.chunk_texts:
                tokens = self._tokenise(text)
                tf = Counter(tokens)
                self.term_freqs.append(tf)
                doc_len = sum(tf.values())
                self.doc_lens.append(doc_len)
                # Update document frequency for each unique term
                for term in tf.keys():
                    doc_freq[term] += 1
            # Average document length
            self.avgdl = sum(self.doc_lens) / n_docs
            # Compute IDF per term
            for term, df in doc_freq.items():
                # BM25 IDF with smoothing to avoid negative values【876197337910835†L64-L112】
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                self.idf[term] = idf
            # Build TF‑IDF matrix
            self.vectorizer = TfidfVectorizer(lowercase=True)
            # The vectorizer returns a sparse matrix; convert to dense array for faster dot‑product
            X = self.vectorizer.fit_transform(self.chunk_texts)
            # L2 normalise row vectors to enable cosine similarity via dot product
            X = normalize(X, norm="l2", copy=False)
            self.tfidf_matrix = X.toarray()
            # Mark as ready
            self._ready_event.set()

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Simple whitespace tokenizer returning lower‑cased tokens."""
        return [t.lower() for t in text.split() if t.strip()]

    def _ensure_ready(self) -> None:
        """Block until the index has finished building."""
        self._ready_event.wait()

    def query_bm25(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Return top_k (chunk_id, score) pairs according to BM25 scoring."""
        self._ensure_ready()
        tokens = self._tokenise(query)
        if not tokens or not self.term_freqs:
            return []
        scores: List[float] = [0.0] * len(self.term_freqs)
        for term in tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for idx, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[idx]
                # BM25 term contribution
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * (numerator / denominator)
        # Pair scores with chunk ids
        pairs = [(self.chunk_ids[i], scores[i]) for i in range(len(scores)) if scores[i] > 0]
        # Sort descending by score
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def query_tfidf(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Return top_k (chunk_id, score) pairs using cosine similarity over TF‑IDF vectors."""
        self._ensure_ready()
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        q_vec = self.vectorizer.transform([query]).toarray()
        # Normalise query vector
        q_vec = normalize(q_vec, norm="l2", copy=False)
        # Cosine similarity via dot product (dense)
        sims = np.dot(self.tfidf_matrix, q_vec.ravel())
        pairs = [(self.chunk_ids[i], float(sims[i])) for i in range(len(sims)) if sims[i] > 0]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def query_hybrid(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Combine BM25 and TF‑IDF scores by averaging normalised scores."""
        self._ensure_ready()
        bm25_results = self.query_bm25(query, top_k=len(self.chunk_ids))
        tfidf_results = self.query_tfidf(query, top_k=len(self.chunk_ids))
        if not bm25_results and not tfidf_results:
            return []
        # Convert to dict for quick lookup
        bm25_dict = {cid: score for cid, score in bm25_results}
        tfidf_dict = {cid: score for cid, score in tfidf_results}
        # Normalise scores to [0,1]
        if bm25_dict:
            max_bm25 = max(bm25_dict.values())
        else:
            max_bm25 = 1.0
        if tfidf_dict:
            max_tfidf = max(tfidf_dict.values())
        else:
            max_tfidf = 1.0
        combined: Dict[int, float] = defaultdict(float)
        for cid in set(list(bm25_dict.keys()) + list(tfidf_dict.keys())):
            bm = bm25_dict.get(cid, 0.0) / max_bm25
            tf = tfidf_dict.get(cid, 0.0) / max_tfidf
            combined[cid] = (bm + tf) / 2
        # Sort by combined score
        results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]