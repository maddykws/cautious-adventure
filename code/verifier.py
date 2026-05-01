"""
Response grounding verifier.

Checks whether the claims in Claude's drafted response are actually supported
by the retrieved corpus chunks — without making a second API call.

Approach:
  1. Extract verifiable claim sentences from the response (step instructions,
     bold items, factual statements with product terms).
  2. For each claim, extract key content terms (nouns, product names, settings).
  3. Check what fraction of those terms appear in the retrieved corpus text.
  4. Mark the claim as "supported" if ≥50% of terms are found.
  5. Return an overall verdict: supported if ≥60% of claims pass.

This borrows from Self-RAG / Corrective-RAG ideas — a drafted answer that cannot
be verified against retrieved evidence gets downgraded to escalation.
"""

from __future__ import annotations

import re

from audit import VerifierClaim

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


def _is_verifiable(line: str) -> bool:
    """Filter out lines that are not worth checking (headers, citations, empty)."""
    stripped = line.strip()
    if len(stripped) < 25:
        return False
    # Skip citation lines and markdown headings
    if stripped.startswith("Source:") or stripped.startswith("#"):
        return False
    # Keep: step instructions, bold items, substantive sentences
    return (
        bool(re.match(r"^\d+\.", stripped))
        or "**" in stripped
        or (len(stripped) > 50 and stripped[0].isupper())
    )


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
        ratio = len(matched) / len(terms)
        claims.append(VerifierClaim(
            text=line.strip()[:120],
            supported=ratio >= 0.50,
            matched_terms=matched[:6],
            support_ratio=round(ratio, 2),
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
