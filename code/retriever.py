"""
Corpus retriever: indexes all .md files in data/ using TF-IDF,
retrieves top-k relevant documents for a given ticket query.

Index is built once at startup (~3–5s for 774 files) and reused.
"""

import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"

_DOMAIN_MAP = {
    "HackerRank": "hackerrank",
    "Claude":     "claude",
    "Visa":       "visa",
}


class CorpusRetriever:
    def __init__(self):
        self.docs: list[dict] = []
        self._vectorizer = TfidfVectorizer(
            max_features=25000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = None
        self._load_and_index()

    def _load_and_index(self) -> None:
        for domain in ("hackerrank", "claude", "visa"):
            domain_dir = DATA_DIR / domain
            if not domain_dir.exists():
                continue
            for path in domain_dir.rglob("*.md"):
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore").strip()
                    if len(content) < 40:
                        continue
                    self.docs.append({
                        "path":    str(path),
                        "content": content[:4000],
                        "domain":  domain,
                        "title":   path.stem.replace("-", " ").replace("_", " "),
                    })
                except Exception:
                    pass

        if self.docs:
            corpus = [d["content"] for d in self.docs]
            self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, company: str | None = None, top_k: int = 5) -> list[dict]:
        """
        Return top_k most relevant docs.
        If company is provided, docs from that domain are boosted (1.5×).
        Docs from other domains are slightly suppressed (0.7×) unless company is None.
        """
        if not self.docs or self._matrix is None:
            return []

        target_domain = _DOMAIN_MAP.get(company) if company else None
        query_vec = self._vectorizer.transform([query[:1000]])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        if target_domain:
            boost = np.array([
                1.5 if d["domain"] == target_domain else 0.7
                for d in self.docs
            ])
            scores = scores * boost

        ranked = np.argsort(scores)[::-1]

        results, seen = [], set()
        for idx in ranked:
            if scores[idx] < 0.02 or len(results) >= top_k:
                break
            doc = self.docs[idx]
            key = doc["title"]
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "domain":  doc["domain"],
                "title":   doc["title"],
                "content": doc["content"][:900],
                "score":   float(scores[idx]),
            })

        return results
