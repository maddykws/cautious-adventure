"""
Corpus retriever: indexes all .md files in data/ using TF-IDF,
retrieves top-k relevant documents for a given ticket query.

Index is built once at startup (~3–5s for 774 files) and reused.
"""

import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"

_DOMAIN_MAP = {
    "HackerRank": "hackerrank",
    "Claude":     "claude",
    "Visa":       "visa",
}


def _clean_title(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ")


def _iter_markdown_chunks(path: Path, content: str, max_chars: int = 2600):
    """
    Index article sections instead of only the first part of each article.
    Support docs often hide the exact answer under a later heading, so section
    retrieval gives the generator tighter evidence and less irrelevant context.
    """
    article_title = _clean_title(path)
    lines = content.splitlines()
    chunks: list[tuple[str, list[str]]] = []
    current_title = article_title
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if len(text) >= 40:
            chunks.append((current_title, current_lines))
        current_lines = []

    for line in lines:
        heading = re.match(r"^\s{0,3}(#{1,3})\s+(.+?)\s*$", line)
        if heading and current_lines:
            flush()
            section = re.sub(r"\s+", " ", heading.group(2)).strip("# ").strip()
            current_title = f"{article_title} :: {section}" if section else article_title
        current_lines.append(line)

        if sum(len(x) + 1 for x in current_lines) >= max_chars:
            flush()

    flush()

    if not chunks and len(content.strip()) >= 40:
        chunks.append((article_title, [content.strip()]))

    for title, chunk_lines in chunks:
        yield title, "\n".join(chunk_lines).strip()


def _expand_query(query: str, company: str | None) -> str:
    """
    Lightweight domain synonym expansion.
    This gives sparse retrieval a semantic-router feel without adding embeddings
    or a vector DB: common support phrasing is mapped to the terms used in docs.
    """
    lo = query.lower()
    company_lo = (company or "").lower()
    additions: list[str] = []

    if company_lo == "hackerrank" or "hackerrank" in lo:
        if any(w in lo for w in ("employee", "interviewer", "user", "member")) and any(w in lo for w in ("remove", "delete", "left", "leaving")):
            additions.append("teams management user management company admin team member invite user roles")
        if any(w in lo for w in ("reschedule", "missed", "alternative date")):
            additions.append("reschedule interview invite participants interview details candidate recruiter")
        if any(w in lo for w in ("zoom", "connectivity", "compatible", "compatibility")):
            additions.append("zoom system compatibility browser allowlist audio video calls interviews")
        if "apply tab" in lo or "practice" in lo:
            additions.append("apply tab deprecated prepare tab coding challenges notification")

    if company_lo == "visa" or "visa" in lo:
        if any(w in lo for w in ("minimum", "maximum", "spend", "merchant")):
            additions.append("merchant set maximum minimum limit using Visa card rules")
        if any(w in lo for w in ("lost", "stolen", "blocked", "travel", "cash")):
            additions.append("lost Visa card travel support global assistance emergency cash ATM locator")

    if company_lo == "claude" or "claude" in lo:
        if any(w in lo for w in ("lti", "canvas", "students", "professor")):
            additions.append("Claude LTI Canvas Instructure developer key LMS administrator education")
        if "bedrock" in lo:
            additions.append("Amazon Bedrock AWS support account manager Claude models regions")

    return " ".join([query, *additions]).strip()


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
                    for title, chunk in _iter_markdown_chunks(path, content):
                        self.docs.append({
                            "path":    str(path),
                            "content": chunk[:3000],
                            "domain":  domain,
                            "title":   title,
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
        expanded_query = _expand_query(query[:1000], company)
        query_vec = self._vectorizer.transform([expanded_query])
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
            key = (doc["path"], doc["title"])
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "domain":  doc["domain"],
                "title":   doc["title"],
                "content": doc["content"][:1800],
                "score":   float(scores[idx]),
            })

        return results
