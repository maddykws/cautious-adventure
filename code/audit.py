"""
Evidence & Safety Audit Trail.

Every ticket processed by the triage agent produces an AuditEntry that records:
  - Safety gate decision and reason
  - Corpus evidence chunks (ranked, scored)
  - Answerability pre-check result
  - Verifier grounding check (claims vs corpus)
  - Risk flags detected
  - Confidence band derived from retrieval + grounding

Audit entries are written to support_tickets/evidence_audit.jsonl (one JSON line
per ticket) and printed in human-readable form when --trace is used.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class EvidenceChunk:
    rank: int
    domain: str
    title: str
    score: float


@dataclass
class VerifierClaim:
    text: str           # first 120 chars of the claim sentence
    supported: bool     # True if majority of key terms found in corpus
    matched_terms: list[str]
    support_ratio: float


@dataclass
class ToolCall:
    """One iteration of the agent's retrieve-tool loop."""
    iteration: int
    query: str
    company: str
    n_results: int
    n_new_evidence: int   # new chunks not seen in earlier iterations


@dataclass
class AuditEntry:
    # ── Identity ──────────────────────────────────────────────────────────────
    ticket_id: int
    issue: str
    subject: str
    company: str

    # ── Safety gate ───────────────────────────────────────────────────────────
    safety_triggered: bool
    safety_reason: str          # empty string if not triggered

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_quality: str      # strong | usable | weak | very_weak | no_evidence
    top_score: float
    domain_match: str           # on_domain | off_domain | unknown_company
    evidence: list[EvidenceChunk]

    # ── Answerability ─────────────────────────────────────────────────────────
    answerability_passed: bool
    answerability_reason: str

    # ── Multi-intent ──────────────────────────────────────────────────────────
    multi_intent: bool

    # ── Verifier (grounding check) ────────────────────────────────────────────
    verifier_overall_supported: bool
    verifier_support_ratio: float
    verifier_claims: list[VerifierClaim]

    # ── Agent loop (tool-use iterations) ──────────────────────────────────────
    tool_calls: list[ToolCall] = field(default_factory=list)
    n_iterations: int = 1
    retrieval_mode: str = "n/a"   # "hybrid" | "tfidf_only" | "n/a"

    # ── Decision ──────────────────────────────────────────────────────────────
    status: Literal["replied", "escalated"] = "escalated"
    product_area: str = ""
    request_type: str = ""
    risk_flags: list[str] = field(default_factory=list)
    confidence_band: Literal["high", "medium", "low", "escalated"] = "escalated"
    response: str = ""
    justification: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def _band(
        safety_triggered: bool,
        quality: str,
        verifier_supported: bool,
        status: str,
    ) -> str:
        if status == "escalated":
            return "escalated"
        if quality == "strong" and verifier_supported:
            return "high"
        if quality in ("usable", "strong") and not verifier_supported:
            return "medium"
        if quality == "usable":
            return "medium"
        return "low"
