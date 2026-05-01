"""
Corpus retriever: indexes all .md files in data/ using TF-IDF,
retrieves top-k relevant documents for a given ticket query.

Index is built once at startup and reused for all tickets.
Section-level chunking (split by ## / ### headings) gives tighter evidence
than whole-article indexing — answers buried in section 3 now surface.
"""

import re
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

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "i", "my", "we", "our",
    "to", "of", "in", "on", "for", "with", "and", "or", "do", "can", "how",
    "what", "this", "that", "it", "be", "have", "has", "had", "not", "will",
})


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
            chunks.append((current_title, list(current_lines)))
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
    Lightweight domain synonym expansion — maps common support phrasing to
    the terms used in the corpus docs, improving sparse retrieval recall.
    """
    lo = query.lower()
    company_lo = (company or "").lower()
    additions: list[str] = []

    if company_lo == "hackerrank" or "hackerrank" in lo:
        if any(w in lo for w in ("employee", "interviewer", "user", "member")) and any(w in lo for w in ("remove", "delete", "left", "leaving")):
            additions.append("teams management user management company admin team member remove from teams transfer ownership three-dot icon")
        if any(w in lo for w in ("reschedule", "missed", "alternative date")):
            additions.append("reschedule interview invite participants interview details candidate recruiter")
        if any(w in lo for w in ("zoom", "connectivity", "compatible", "compatibility")):
            additions.append("zoom system compatibility browser allowlist audio video calls interviews")
        if "apply tab" in lo or ("practice" in lo and "tab" in lo):
            additions.append("apply tab deprecated prepare tab coding challenges notification")
        if any(w in lo for w in ("certificate", "cert", "name on")):
            additions.append("certifications certificate name regenerate update")
        if any(w in lo for w in ("pause", "subscription", "billing", "plan")):
            additions.append("pause subscription billing plan monthly cancel manage")
        if any(w in lo for w in ("inactivity", "inactive", "lobby", "kicked", "screen share", "timeout")):
            additions.append(
                "ending interview leave interview inactivity hour lobby candidate "
                "interviewer no other interviewers present interview ends automatically"
            )
        if any(w in lo for w in ("infosec", "compliance", "vendor", "security review", "gdpr", "soc 2", "audit")):
            additions.append(
                "HackerRank for Work security GDPR account-level security NYC AI law "
                "bias audit results plagiarism detection proctoring impersonation detection "
                "compliance vendor questionnaire data privacy"
            )

    if company_lo == "visa" or "visa" in lo:
        if any(w in lo for w in ("minimum", "maximum", "spend", "merchant", "limit")):
            additions.append("merchant set maximum minimum limit using Visa card rules regulations")
        if any(w in lo for w in ("lost", "stolen", "blocked", "travel", "cash", "atm")):
            additions.append("lost Visa card travel support global assistance emergency cash ATM locator")
        if any(w in lo for w in ("cheque", "check", "traveller")):
            additions.append("travellers cheques lost stolen issuer refund serial numbers")

    if company_lo == "claude" or "claude" in lo:
        if any(w in lo for w in ("lti", "canvas", "students", "professor", "university")):
            additions.append("Claude LTI Canvas Instructure developer key LMS administrator education")
        if "bedrock" in lo:
            additions.append("Amazon Bedrock AWS support account manager Claude models regions")
        if any(w in lo for w in ("crawl", "scrape", "robots", "bot", "website")):
            additions.append("Anthropic crawler robots.txt allowlist data scraping website")
        if any(w in lo for w in ("data", "retention", "retained", "kept", "stored", "training", "improve", "model improvement")):
            additions.append(
                "Claude data retention default 30 days minimum custom retention controls "
                "Enterprise plan Owner organization settings data privacy controls "
                "model improvement training conversation data delete"
            )

    return " ".join([query, *additions]).strip()


def _lexical_score(query_terms: set[str], content: str) -> float:
    """Count normalized query term overlap in doc content (reranker signal)."""
    content_lower = content.lower()
    if not query_terms:
        return 0.0
    hits = sum(1 for t in query_terms if t in content_lower)
    return hits / len(query_terms)


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

    def _collect(
        self,
        scores: np.ndarray,
        top_k: int,
        min_score: float = 0.02,
    ) -> list[dict]:
        """Return top_k unique docs (by path+title) above min_score threshold."""
        ranked = np.argsort(scores)[::-1]
        results, seen = [], set()
        for idx in ranked:
            if scores[idx] < min_score or len(results) >= top_k:
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

    def _rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """
        Reciprocal-rank fusion of TF-IDF score and lexical term overlap.
        No new dependencies — boosts docs that contain the actual query words,
        not just related n-grams. Small alpha keeps TF-IDF as the primary signal.
        """
        if len(docs) <= 1:
            return docs

        query_terms = {
            t for t in re.findall(r"\b\w+\b", query.lower())
            if t not in _STOP_WORDS and len(t) > 2
        }

        alpha = 0.15
        rescored = []
        for doc in docs:
            lex = _lexical_score(query_terms, doc["content"])
            combined = doc["score"] + alpha * lex
            rescored.append((combined, doc))

        rescored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in rescored]

    def retrieve(
        self,
        query: str,
        company: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Return top_k most relevant corpus chunks.

        Domain boosting: target company docs → 1.5×, others → 0.7×.
        Cross-domain fallback: if boosted top result is very weak (< 0.05) and
        off-domain, retry without boost so cross-domain coverage isn't lost.
        Reranker: reciprocal-rank fusion of TF-IDF + lexical overlap as final pass.
        """
        if not self.docs or self._matrix is None:
            return []

        target_domain = _DOMAIN_MAP.get(company) if company else None
        expanded = _expand_query(query[:1000], company)
        query_vec = self._vectorizer.transform([expanded])
        raw_scores = cosine_similarity(query_vec, self._matrix).flatten()

        if target_domain:
            boost = np.array([
                1.5 if d["domain"] == target_domain else 0.7
                for d in self.docs
            ])
            boosted_scores = raw_scores * boost
        else:
            boosted_scores = raw_scores

        results = self._collect(boosted_scores, top_k)

        # Cross-domain fallback: if top result is weak or off-domain, merge in
        # unfiltered results so cross-domain coverage isn't lost entirely.
        if target_domain and (
            not results
            or results[0]["score"] < 0.05
            or results[0]["domain"] != target_domain
        ):
            fallback = self._collect(raw_scores, top_k)
            seen_keys = {(d["domain"], d["title"]) for d in results}
            for doc in fallback:
                key = (doc["domain"], doc["title"])
                if key not in seen_keys and len(results) < top_k:
                    results.append(doc)
                    seen_keys.add(key)

        return self._rerank(expanded, results)
