"""
Response grounding verifier.

Checks whether the claims in Claude's drafted response are actually supported
by the retrieved corpus chunks — without making a second API call.

Approach:
  1. Extract verifiable claim sentences from the response (step instructions,
     bold items, factual statements with product terms).
  2. For each claim, extract key content terms (nouns, product names, settings).
  3. Compute two signals against the retrieved corpus text:
       a. lexical term-overlap ratio (≥ 0.50 → supported)
       b. TF-IDF cosine similarity   (≥ 0.18 → supported)
     A claim passes if EITHER signal supports it. The cosine signal rescues
     paraphrased / synonym-heavy claims that pure term overlap misses.
  4. Return an overall verdict: supported if ≥60% of claims pass.

This borrows from Self-RAG / Corrective-RAG ideas — a drafted answer that cannot
be verified against retrieved evidence gets downgraded to escalation.
"""

from __future__ import annotations

import re

from audit import VerifierClaim

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_OK = True
except Exception:  # pragma: no cover — sklearn is a hard dep, but fail-soft anyway
    _SKLEARN_OK = False

_STOP_WORDS = frozenset({
    "this", "that", "with", "from", "have", "been", "will", "your", "their",
    "please", "also", "note", "following", "below", "above", "here", "some",
    "more", "make", "sure", "then", "next", "step", "first", "second", "third",
    "when", "once", "after", "before", "select", "click", "open", "enter",
    "navigate", "need", "want", "should", "must", "able", "currently",
})


def _extract_key_terms(text: str) -> set[str]:
    """Extract meaningful content terms from a claim sentence."""
    text = re.sub(r"\*+", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"`[^`]+`", lambda m: m.group().replace("-", " "), text)
    words = re.findall(r"\b[A-Za-z][a-z]{3,}\b", text)
    return {w.lower() for w in words if w.lower() not in _STOP_WORDS}


_GAP_STATEMENT_RE = re.compile(
    r"\b("
    r"corpus (does(n'?t| not)|provides? no|contains? no|has(n'?t| not| no))"
    r"|the (documentation|corpus|article|excerpt) (does(n'?t| not)|provides? no|contains? no)"
    r"|(no|not) (information|specific(s|ation)?|coverage|guidance|details?|procedure) (about|on|for|regarding)"
    r"|isn'?t (covered|documented|specified|in the corpus)"
    r"|don'?t (have|cover|specify|address)"
    r"|cannot (find|locate) (this|that|specific)"
    r"|outside (the )?(scope|coverage)"
    r"|for (specific|the specific) .{0,40}(a support agent|please contact|will follow up|will be in touch)"
    r")",
    re.IGNORECASE,
)


def _is_verifiable(line: str) -> bool:
    """Filter out lines that aren't worth grounding-checking.

    Skips:
      • Headings, citation lines, very short fragments
      • Gap statements ("the corpus doesn't specify X", "no information on Y",
        "for the specific timeout, a support agent will follow up") — these
        are meta-statements about corpus coverage, not corpus-grounded claims;
        running term-overlap on them produces false negatives because the
        terms ARE the user's specifics, which by definition aren't in the
        corpus. The partial-reply pattern depends on this filter.
    """
    stripped = line.strip()
    if len(stripped) < 25:
        return False
    if stripped.startswith("Source:") or stripped.startswith("#"):
        return False
    if _GAP_STATEMENT_RE.search(stripped):
        return False
    # Keep: step instructions, bold items, substantive sentences
    return (
        bool(re.match(r"^\d+\.", stripped))
        or "**" in stripped
        or (len(stripped) > 50 and stripped[0].isupper())
    )


def _semantic_max_cosine(claim: str, corpus_chunks: list[dict]) -> float:
    """Max TF-IDF cosine similarity of `claim` against any single corpus chunk.

    Builds a small ad-hoc vectorizer over (claim, chunks…) so synonym-heavy
    answers that fail term-overlap can still be marked supported when the
    underlying chunk is genuinely on-topic. ~5 ms per call on the typical
    5-chunk top-k passed in.
    """
    if not _SKLEARN_OK or not claim or not corpus_chunks:
        return 0.0
    docs = [c.get("content", "") for c in corpus_chunks if c.get("content")]
    if not docs:
        return 0.0
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True,
        )
        matrix = vec.fit_transform([claim, *docs])
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return float(sims.max()) if sims.size else 0.0
    except Exception:
        return 0.0


def check_grounding(response: str, corpus_chunks: list[dict]) -> dict:
    """
    Check whether the response is grounded in the retrieved corpus.

    Returns a dict with:
      overall_supported  bool    — True if ≥60% of claims are supported
      support_ratio      float   — fraction of supported claims
      claims             list    — per-claim breakdown
      unsupported_claims list    — claims that failed the check (for trace output)
    """
    corpus_text = " ".join(d.get("content", "") for d in corpus_chunks).lower()

    verifiable_lines = [ln for ln in response.splitlines() if _is_verifiable(ln)]

    # Cap at 12 claims to avoid runaway checking
    claims: list[VerifierClaim] = []
    for line in verifiable_lines[:12]:
        terms = _extract_key_terms(line)
        if len(terms) < 2:
            continue
        matched = sorted(t for t in terms if t in corpus_text)
        lexical_ratio = len(matched) / len(terms)

        # Cosine rescue: only invoke when lexical signal is borderline-weak
        # to keep verifier fast on the common case.
        if lexical_ratio >= 0.50:
            supported = True
        else:
            cos = _semantic_max_cosine(line, corpus_chunks)
            supported = cos >= 0.18

        claims.append(VerifierClaim(
            text=line.strip()[:120],
            supported=supported,
            matched_terms=matched[:6],
            support_ratio=round(lexical_ratio, 2),
        ))

    if not claims:
        # No verifiable claims found — treat as supported (escalated tickets
        # produce a fixed message that cannot and should not be grounded-checked)
        return {
            "overall_supported": True,
            "support_ratio": 1.0,
            "claims": [],
            "unsupported_claims": [],
        }

    supported_count = sum(1 for c in claims if c.supported)
    overall_ratio = supported_count / len(claims)

    return {
        "overall_supported": overall_ratio >= 0.60,
        "support_ratio": round(overall_ratio, 2),
        "claims": claims,
        "unsupported_claims": [c for c in claims if not c.supported],
    }
