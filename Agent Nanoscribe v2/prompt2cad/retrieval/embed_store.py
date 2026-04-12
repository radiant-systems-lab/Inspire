"""
Build and persist the embedding index for RAG retrieval.

Supports two embedding backends:
  • sentence-transformers (preferred, install: pip install sentence-transformers)
  • numpy BOW TF-IDF (zero-dependency fallback)

The store is saved as a pickle file and reloaded on subsequent runs.
Delete embed_cache.pkl to force a rebuild after adding/changing RAG docs.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Optional: sentence-transformers ──────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    _SentenceTransformer = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# BOW / TF-IDF helpers (pure numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Extract alphanumeric+underscore tokens (suitable for CadQuery API names)."""
    return re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())


def _bow_embed(
    texts: List[str],
    vocab: Optional[Dict[str, int]] = None,
    idf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
    """
    Compute L2-normalised TF-IDF BOW embeddings.

    Args:
        texts:  List of strings to embed.
        vocab:  Pre-built vocabulary dict (word → column index).
                If None, build from `texts`.
        idf:    Pre-computed IDF weights array matching vocab size.
                If None, compute from `texts`.

    Returns:
        (embeddings, vocab, idf)  shapes: (n, d), -, (d,)
    """
    counters: List[Dict[str, int]] = []
    for t in texts:
        c: Dict[str, int] = {}
        for tok in _tokenize(t):
            c[tok] = c.get(tok, 0) + 1
        counters.append(c)

    # Build or reuse vocabulary
    if vocab is None:
        vocab = {}
        for c in counters:
            for w in c:
                if w not in vocab:
                    vocab[w] = len(vocab)

    n, d = len(texts), len(vocab)
    X = np.zeros((n, d), dtype=np.float32)
    for i, c in enumerate(counters):
        for w, cnt in c.items():
            if w in vocab:
                X[i, vocab[w]] = float(cnt)

    # IDF weighting
    if idf is None:
        df = (X > 0).sum(axis=0).astype(np.float32) + 1.0
        idf = np.log((n + 1.0) / df) + 1.0  # smoothed IDF

    # Pad / truncate X if vocab size changed (query-time use)
    if X.shape[1] != len(idf):
        target_d = len(idf)
        if X.shape[1] < target_d:
            X = np.pad(X, ((0, 0), (0, target_d - X.shape[1])))
        else:
            X = X[:, :target_d]

    X = X * idf

    # L2 normalise row-wise
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X /= np.where(norms > 0, norms, 1.0)

    return X, vocab, idf


# ─────────────────────────────────────────────────────────────────────────────
# YAML metadata extraction from cadquery_examples.md
# ─────────────────────────────────────────────────────────────────────────────

def _parse_yaml_metadata(block: str) -> Dict[str, List[str]]:
    """Extract operations / geometry_type / tags from an example's YAML block."""
    meta: Dict[str, List[str]] = {"operations": [], "geometry_type": [], "tags": []}
    for key in meta:
        pattern = rf'{key}:\s*\n((?:\s+- [^\n]+\n?)+)'
        m = re.search(pattern, block)
        if m:
            meta[key] = [s.strip() for s in re.findall(r'-\s+([^\n]+)', m.group(1))]
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Chunking strategies
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_examples_md(path: Path) -> List[dict]:
    """
    Split cadquery_examples.md into one chunk per ``### Example:`` section.
    Each chunk preserves the Python code and YAML metadata.
    """
    text = path.read_text(encoding="utf-8")
    raw_chunks = re.split(r'\n---+\n', text)

    chunks: List[dict] = []
    for raw in raw_chunks:
        raw = raw.strip()
        if not raw or not re.search(r'### Example:', raw):
            continue

        title_m = re.search(r'### Example:\s*(.+)', raw)
        title = title_m.group(1).strip() if title_m else f"Example {len(chunks)+1}"

        yaml_m = re.search(r'```yaml\s*([\s\S]+?)\s*```', raw)
        meta = _parse_yaml_metadata(yaml_m.group(1)) if yaml_m else {}

        chunks.append({
            "text": raw,
            "source": "examples",
            "title": title,
            "metadata": meta,
        })

    return chunks


def _sliding_window_chunks(
    text: str,
    source: str,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
) -> List[dict]:
    """Split `text` into overlapping character-window chunks."""
    chunks: List[dict] = []
    start, n = 0, len(text)
    while start < n and len(chunks) < max_chunks:
        end = min(start + chunk_size, n)
        snippet = text[start:end].strip()
        if snippet:
            chunks.append({
                "text": snippet,
                "source": source,
                "title": f"{source}[{len(chunks)}]",
                "metadata": {},
            })
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_store(
    rag_dir: Path,
    chunk_size: int = 2000,
    overlap: int = 400,
    max_index_chunks: int = 150,
    sbert_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Load, chunk, embed and return the RAG store dict.

    Store schema::

        {
            "chunks":     List[dict],          # text + source + title + metadata
            "embeddings": np.ndarray (n, d),   # L2-normalised
            "vocab":      dict | None,         # BOW vocab (None if SBERT)
            "idf":        np.ndarray | None,   # BOW IDF weights (None if SBERT)
            "method":     "sbert" | "bow",
        }
    """
    all_chunks: List[dict] = []

    # ── 1. cadquery_examples.md (one chunk per example) ───────────────────────
    ex_path = rag_dir / "cadquery_examples.md"
    if ex_path.exists():
        ex_chunks = _chunk_examples_md(ex_path)
        all_chunks.extend(ex_chunks)
        print(f"  examples    : {len(ex_chunks):3d} chunks")
    else:
        print(f"  WARNING: {ex_path} not found")

    # ── 2. cadquery_cheatsheet.txt (sliding window) ───────────────────────────
    cs_path = rag_dir / "cadquery_cheatsheet.txt"
    if cs_path.exists():
        cs_text = cs_path.read_text(encoding="utf-8")
        cs_chunks = _sliding_window_chunks(cs_text, "cheatsheet", chunk_size, overlap, max_chunks=50)
        all_chunks.extend(cs_chunks)
        print(f"  cheatsheet  : {len(cs_chunks):3d} chunks")
    else:
        print(f"  WARNING: {cs_path} not found")

    # ── 3. cadquery_index.txt (large API reference, capped) ───────────────────
    idx_path = rag_dir / "cadquery_index.txt"
    if idx_path.exists():
        idx_text = idx_path.read_text(encoding="utf-8", errors="replace")
        idx_chunks = _sliding_window_chunks(
            idx_text, "index", chunk_size, overlap, max_chunks=max_index_chunks
        )
        all_chunks.extend(idx_chunks)
        print(f"  index       : {len(idx_chunks):3d} chunks  "
              f"(~{len(idx_text)//1024} KB total, capped at {max_index_chunks})")
    else:
        print(f"  WARNING: {idx_path} not found")

    n_total = len(all_chunks)
    print(f"  Total chunks: {n_total}")

    texts = [c["text"] for c in all_chunks]

    # ── Embed ─────────────────────────────────────────────────────────────────
    if _SBERT_AVAILABLE:
        print(f"  Using sentence-transformers ({sbert_model}) …")
        _model = _SentenceTransformer(sbert_model)
        emb_raw = _model.encode(texts, batch_size=64, show_progress_bar=True)
        embeddings = np.array(emb_raw, dtype=np.float32)
        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.where(norms > 0, norms, 1.0)
        vocab, idf = None, None
        method = "sbert"
    else:
        print("  Using BOW TF-IDF (numpy fallback) …")
        embeddings, vocab, idf = _bow_embed(texts)
        method = "bow"

    print(f"  Embeddings  : {embeddings.shape}  method={method}")

    return {
        "chunks":     all_chunks,
        "embeddings": embeddings,
        "vocab":      vocab,
        "idf":        idf,
        "method":     method,
    }


def save_store(store: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(store, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = path.stat().st_size // 1024
    print(f"  Embed store saved → {path}  ({size_kb} KB)")


def load_store(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def get_or_build_store(
    rag_dir: Path,
    cache_path: Path,
    chunk_size: int = 2000,
    overlap: int = 400,
    max_index_chunks: int = 150,
) -> dict:
    """Load from cache if available; otherwise build and cache."""
    store = load_store(cache_path)
    if store is not None:
        n = len(store["chunks"])
        print(f"Loaded embed store from cache: {n} chunks  [{store.get('method','?')}]")
        return store

    print("Building embed store …")
    store = build_store(
        rag_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        max_index_chunks=max_index_chunks,
    )
    save_store(store, cache_path)
    return store
