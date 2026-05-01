"""
Dense-embedding rerank for the corpus retriever.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80 MB, runs CPU-only)
to rerank TF-IDF candidates by semantic similarity. Catches paraphrased
queries that lexical retrieval misses entirely.

Design constraints, in order of priority:
  1. Fail-soft on every error path. If torch / sentence-transformers /
     the model file / the cache file are missing or broken, the agent
     reverts to TF-IDF-only retrieval. The hackathon eval rig must never
     crash because embeddings aren't available.
  2. Pay the cost once. Encoding 6,000+ corpus chunks takes ~30–60 s on
     CPU; we cache the resulting matrix to disk (data/index/embeddings.npz)
     keyed by a content-hash of the corpus so subsequent runs load
     instantly.
  3. Inference is offline. The model is downloaded once on first use and
     cached by huggingface_hub locally; no network call per query.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

# Try to import the heavy deps. If they're missing or broken, the embedder
# stays disabled and the retriever silently falls back to TF-IDF.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _IMPORT_OK = True
    _IMPORT_ERR: str | None = None
except Exception as exc:  # pragma: no cover — depends on install state
    SentenceTransformer = None  # type: ignore
    _IMPORT_OK = False
    _IMPORT_ERR = f"{type(exc).__name__}: {exc}"


_MODEL_NAME    = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
_CACHE_DIR     = Path(__file__).parent.parent / "data" / "index"
_CACHE_FILE    = _CACHE_DIR / "embeddings.npz"
_BATCH_SIZE    = 64


def _hash_chunks(chunks: Sequence[str]) -> str:
    """Stable hash of all corpus chunks. Used to invalidate the disk cache
    when the corpus changes (file added, edited, removed)."""
    h = hashlib.sha256()
    h.update(_MODEL_NAME.encode("utf-8"))
    h.update(str(len(chunks)).encode("utf-8"))
    for c in chunks:
        h.update(c.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


class CorpusEmbedder:
    """
    Lazy-loaded dense embedder for the corpus retriever.

    Public API:
      available           — True if embeddings can actually be used
      fit(chunks)         — embed (or load from cache) the full corpus
      rerank(q, cands, k) — cosine-rerank `cands` (list[dict]) by `q`

    All methods are no-ops when `available` is False.
    """

    def __init__(self) -> None:
        self.available = False
        self.disabled_reason: str = ""
        self._model: SentenceTransformer | None = None
        self._matrix: np.ndarray | None = None
        self._chunk_hash: str = ""

        if not _IMPORT_OK:
            self.disabled_reason = (
                f"sentence-transformers/torch not importable ({_IMPORT_ERR}); "
                "falling back to TF-IDF-only retrieval."
            )
            return

        try:
            self._model = SentenceTransformer(_MODEL_NAME)
            self.available = True
        except Exception as exc:  # network failure, missing model, etc.
            self.disabled_reason = (
                f"could not load embedding model {_MODEL_NAME!r} "
                f"({type(exc).__name__}: {exc}); falling back to TF-IDF."
            )
            self._model = None
            self.available = False

    # ── Corpus indexing ──────────────────────────────────────────────────────

    def fit(self, chunks: Sequence[str]) -> None:
        """Compute or load embeddings for the full corpus.

        Tries the on-disk cache first; only re-encodes when the chunk
        content has changed. Sets `self._matrix` (n_chunks × dim, L2-norm)
        on success; disables the embedder on any failure.
        """
        if not self.available or self._model is None:
            return

        self._chunk_hash = _hash_chunks(chunks)

        # Try disk cache first.
        if _CACHE_FILE.exists():
            try:
                cached = np.load(_CACHE_FILE, allow_pickle=False)
                if str(cached.get("hash", "")) == self._chunk_hash:
                    self._matrix = cached["matrix"]
                    if self._matrix.shape[0] == len(chunks):
                        log.info(
                            "loaded %d corpus embeddings from cache (%s)",
                            self._matrix.shape[0], _CACHE_FILE,
                        )
                        return
                    else:
                        log.warning("cache row count mismatch; recomputing")
            except Exception as exc:
                log.warning("embedding cache unreadable (%s); recomputing", exc)

        # Compute fresh.
        try:
            matrix = self._model.encode(
                list(chunks),
                batch_size=_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # so dot product == cosine
            ).astype(np.float32, copy=False)
        except Exception as exc:
            self.available = False
            self.disabled_reason = (
                f"embedding encode failed ({type(exc).__name__}: {exc}); "
                "falling back to TF-IDF."
            )
            return

        self._matrix = matrix

        # Persist to disk for next run.
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                _CACHE_FILE,
                matrix=matrix,
                hash=np.array(self._chunk_hash),
            )
            log.info(
                "wrote %d corpus embeddings to cache (%s, %.1f MB)",
                matrix.shape[0], _CACHE_FILE, _CACHE_FILE.stat().st_size / 1e6,
            )
        except Exception as exc:
            log.warning("could not write embedding cache (%s)", exc)

    # ── Standalone semantic search ───────────────────────────────────────────

    def search(self, query: str, top_k: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """
        Run a pure semantic search over the entire corpus index.
        Returns (indices, similarities) for the top-k results.

        Returns (empty array, empty array) if the embedder isn't available.
        """
        if not self.available or self._model is None or self._matrix is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        try:
            q_vec = self._model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32, copy=False)
            sims = (self._matrix @ q_vec[0]).astype(np.float32)
            k = min(top_k, sims.shape[0])
            # argpartition is O(n), faster than full argsort for large corpora
            top_idx = np.argpartition(-sims, k - 1)[:k] if k > 0 else np.array([], dtype=np.int64)
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            return top_idx, sims[top_idx]
        except Exception as exc:
            log.warning("semantic search failed (%s)", exc)
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # ── Reranking ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
        global_indices: list[int] | None = None,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Hybrid rerank using Reciprocal Rank Fusion (RRF) — the standard
        sparse+dense fusion algorithm, robust to score-magnitude differences.

        For each candidate at TF-IDF rank `r_t` and semantic rank `r_s`,
        score = 1/(k+r_t) + 1/(k+r_s). Default k=60 (Cormack et al. 2009).
        Missing rank → no contribution from that signal (rank treated as ∞).

        Args:
          query           — the (expanded) user query
          candidates      — list of {domain, title, content, score, ...} dicts
                            (TF-IDF order, possibly merged with semantic-only
                            candidates that have score=0)
          top_k           — how many to return
          global_indices  — corpus embedding row index per candidate
          rrf_k           — RRF dampening constant; 60 is the standard

        Falls back to the original input order on any failure.
        """
        if not self.available or self._model is None or not candidates:
            return candidates[:top_k]

        try:
            q_vec = self._model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32, copy=False)

            if global_indices is not None and self._matrix is not None:
                cand_matrix = self._matrix[global_indices]
            else:
                cand_texts = [c["content"][:1500] for c in candidates]
                cand_matrix = self._model.encode(
                    cand_texts,
                    batch_size=_BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32, copy=False)

            sims = (cand_matrix @ q_vec[0]).astype(np.float32)
            tfidf_scores = np.array([c.get("score", 0.0) for c in candidates], dtype=np.float32)

            # Build per-signal ranks. Higher score == better, so descending sort.
            tfidf_order   = np.argsort(-tfidf_scores)
            sem_order     = np.argsort(-sims)
            tfidf_rank    = np.empty(len(candidates), dtype=np.int32)
            sem_rank      = np.empty(len(candidates), dtype=np.int32)
            tfidf_rank[tfidf_order] = np.arange(len(candidates))
            sem_rank[sem_order]     = np.arange(len(candidates))

            # RRF: sum reciprocal ranks across both rankings. Skip TF-IDF
            # contribution when the candidate has no TF-IDF score (semantic-
            # only candidates) so they're not penalized for being last in TF-IDF.
            rrf_score = np.zeros(len(candidates), dtype=np.float32)
            for i in range(len(candidates)):
                if tfidf_scores[i] > 0:
                    rrf_score[i] += 1.0 / (rrf_k + tfidf_rank[i] + 1)
                rrf_score[i] += 1.0 / (rrf_k + sem_rank[i] + 1)

            ranked = np.argsort(-rrf_score)
            out: list[dict] = []
            for idx in ranked[:top_k]:
                doc = dict(candidates[idx])
                doc["semantic_score"] = float(sims[idx])
                doc["rrf_score"]      = float(rrf_score[idx])
                out.append(doc)
            return out
        except Exception as exc:
            log.warning("embedding rerank failed (%s); using TF-IDF order", exc)
            return candidates[:top_k]

    # ── Convenience for ad-hoc query similarity ──────────────────────────────

    def similarity(self, query: str, text: str) -> float:
        """Cosine similarity for a single (query, text) pair. 0.0 on failure."""
        if not self.available or self._model is None:
            return 0.0
        try:
            v = self._model.encode(
                [query, text[:1500]],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return float(v[0] @ v[1])
        except Exception:
            return 0.0


# Module-level singleton: built once, reused across the process.
_embedder: CorpusEmbedder | None = None


def get_embedder() -> CorpusEmbedder:
    """Return the process-wide embedder. Always returns an instance —
    `embedder.available` tells you whether it's actually usable."""
    global _embedder
    if _embedder is None:
        _embedder = CorpusEmbedder()
    return _embedder
