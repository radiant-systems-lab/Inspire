"""
Semantic retriever over the embedding store.

Combines user prompt + planner tags/operations into an augmented query,
then returns the top-scoring example and cheatsheet/index chunks separately.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import (
    RAG_DIR,
    EMBED_CACHE,
    MAX_RETRIEVAL_RESULTS,
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS,
    MAX_INDEX_CHUNKS,
)
from .embed_store import (
    get_or_build_store,
    _bow_embed,
    _SBERT_AVAILABLE,
)


class Retriever:
    """
    Thin wrapper around the embedding store providing a ``retrieve()`` method.

    On first call the store is loaded from the pickle cache (or rebuilt from
    the RAG documents if the cache doesn't exist yet).
    """

    def __init__(self, store: Optional[dict] = None) -> None:
        if store is None:
            store = get_or_build_store(
                RAG_DIR,
                EMBED_CACHE,
                chunk_size=CHUNK_SIZE_CHARS,
                overlap=CHUNK_OVERLAP_CHARS,
                max_index_chunks=MAX_INDEX_CHUNKS,
            )

        self._chunks:     List[dict]          = store["chunks"]
        self._embeddings: np.ndarray          = store["embeddings"]   # (n, d)  L2-normalised
        self._vocab:      Optional[dict]      = store.get("vocab")
        self._idf:        Optional[np.ndarray]= store.get("idf")
        self._method:     str                 = store.get("method", "bow")

        # Load SBERT model once if that's the embedding method
        self._sbert = None
        if self._method == "sbert" and _SBERT_AVAILABLE:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        user_prompt: str,
        plan: Optional[dict] = None,
        n_examples:   int = 3,
        n_cheatsheet: int = 1,
    ) -> List[dict]:
        """
        Return top-n_examples example chunks + top-n_cheatsheet reference chunks.

        Args:
            user_prompt:  The original natural-language CAD request.
            plan:         Output of the planner LLM (optional but improves recall).
            n_examples:   Number of example chunks to return.
            n_cheatsheet: Number of cheatsheet/index chunks to return.

        Returns:
            List of chunk dicts, each with an extra ``"score"`` field.
        """
        query = self._build_query(user_prompt, plan)
        q_vec = self._embed_query(query)

        sims: np.ndarray = self._embeddings @ q_vec   # cosine similarity (n,)

        # Split by source
        ex_mask    = np.array([c["source"] == "examples"   for c in self._chunks])
        ref_mask   = np.array([c["source"] != "examples"   for c in self._chunks])

        results: List[dict] = []

        # Top example chunks
        if ex_mask.any():
            scores = np.where(ex_mask, sims, -np.inf)
            top_idx = np.argsort(-scores)[:n_examples]
            for i in top_idx:
                if sims[i] > -np.inf:
                    results.append({**self._chunks[i], "score": float(sims[i])})

        # Top reference chunks (cheatsheet / index)
        if ref_mask.any():
            scores = np.where(ref_mask, sims, -np.inf)
            top_idx = np.argsort(-scores)[:n_cheatsheet]
            for i in top_idx:
                if sims[i] > -np.inf:
                    results.append({**self._chunks[i], "score": float(sims[i])})

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_query(user_prompt: str, plan: Optional[dict]) -> str:
        """Concatenate prompt + planner tags + operations into one query string."""
        parts = [user_prompt]
        if plan:
            parts.extend(plan.get("tags", []))
            parts.extend(plan.get("operations", []))
            intent = plan.get("intent", "")
            if intent:
                parts.append(intent)
        return " ".join(parts)

    def _embed_query(self, text: str) -> np.ndarray:
        """Return an L2-normalised embedding vector for `text`."""
        d_store = self._embeddings.shape[1]

        if self._sbert is not None:
            vec = self._sbert.encode([text])[0].astype(np.float32)
        else:
            # BOW with the corpus vocabulary + IDF weights
            X, _, _ = _bow_embed([text], vocab=self._vocab, idf=self._idf)
            vec = X[0]

        # Pad / truncate to match stored dimension (safety guard)
        if len(vec) < d_store:
            vec = np.pad(vec, (0, d_store - len(vec)))
        elif len(vec) > d_store:
            vec = vec[:d_store]

        # L2 normalise
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        return vec.astype(np.float32)


# ── Module-level lazy singleton ───────────────────────────────────────────────
_instance: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Return (or create) the module-level Retriever singleton."""
    global _instance
    if _instance is None:
        _instance = Retriever()
    return _instance
