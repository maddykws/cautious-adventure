"""
Rule-based safety and escalation classifier.
Runs before calling Claude — fast, deterministic, no API cost.
"""

import re
import unicodedata

# ── Patterns that ALWAYS force escalation ─────────────────────────────────────

_ESCALATION_PATTERNS = [
    # Identity theft — always human (lost/stolen CARD is corpus-answerable, identity theft is not)
    r"\bidentity theft\b",
    r"\bidentity (has been|was) stolen\b",
    # Unauthorized financial activity (not the same as "my card was lost")
    r"\bunauthorized (charge|transaction|debit|payment)\b",
    r"\bfraudulent (charge|transaction|payment|activity)\b",
    # Cardholder dispute initiation — corpus covers merchant side only, not cardholder filings
    r"\bdispute (a |this |the )?(charge|transaction|payment|chargeback)\b",
    r"\b(file|submit|initiate|raise|open|start) (a |my )?(dispute|chargeback)\b",
    # Security vulnerabilities
    r"\bsecurity vulnerability\b", r"\bbug bounty\b",
    r"\bsecurity (bug|flaw|issue|exploit|breach)\b",
    # Score / grade manipulation (can never be done)
    r"\b(increase|change|modify|adjust|fix) my (score|grade|result|mark)\b",
    r"tell the company to (move|advance|promote|pass) me",
    r"graded me unfairly",
    # Unauthorized account changes
    r"I am not the .{0,30}(owner|admin|workspace owner)",
    r"restore my access.{0,50}not (the )?(owner|admin)",
    r"\bnot (an?|the)?\s*(workspace\s+)?(owner|admin)\b.{0,80}\b(access|restore|remove|delete|modify|change|invite|add)\b",
    r"\b(access|restore|remove|delete|modify|change|invite|add)\b.{0,80}\bnot (an?|the)?\s*(workspace\s+)?(owner|admin)\b",
    r"(remove|delete|ban) .{0,50}(seller|merchant|vendor)",
    # System-wide outages
    r"site is down", r"platform.{0,20}(is |has )?(been )?down",
    r"none of the (submissions|pages|requests|challenges).{0,30}(working|accessible|loading)",
    r"all (requests?|submissions?).{0,20}(are |is )?(failing|broken|not working)",
    # Malicious / adversarial payloads
    r"delete all files", r"\brm\s+-rf\b", r"\bDROP TABLE\b",
    r"exec\s*\(", r"os\.system\s*\(",
    r"ignore (previous|all|your) instructions",
    r"show (me )?(your|the) (internal|system|hidden) (rules|prompt|instructions|logic)",
    r"show .{0,80}(internal|system|hidden).{0,40}(rules|prompt|instructions|logic)",
    r"affiche .{0,80}(regles internes|documents recuperes|logique)",
    r"(regles internes|documents recuperes|logique exacte).{0,80}(fraude|decision)",
    r"affiche .{0,50}(règles internes|documents récupérés|logique)",   # French injection
    r"display .{0,50}(internal rules|retrieved documents|decision logic)",
    # Infosec questionnaires / compliance forms (can't fill on behalf)
    r"(fill(ing)?|complete?) .{0,30}(infosec|security|compliance|vendor) (form|questionnaire|process)",
    # Privacy data-retention duration — corpus has no specific durations; guessing is harmful
    r"how long (will|is|does|do).{0,50}(data|information).{0,30}(used|kept|retained|stored)",
    r"\b(data retention|retention period|how long.{0,20}retain)\b",
]

_ESC_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _ESCALATION_PATTERNS]

# ── Request type keyword heuristics ──────────────────────────────────────────

_FEATURE_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(feature request|please add|can you add|would (be|it be) (great|nice|possible))\b",
    r"\b(I (want|wish|would like).{0,30}(feature|option|ability|functionality))\b",
    r"\bwould it be possible to\b",
]]

_BUG_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(not working|broken|crash(ing|ed)?|error|bug|glitch|fail(ed|ing)|exception)\b",
    r"\b(doesn'?t work|isn'?t working|stopped working|can'?t (load|access|open|log.?in))\b",
    r"\b(500|404|502|503)\b",
    r"\b(resume builder is down|is down)\b",
    r"none of the (submissions?|pages?|requests?|challenges?).{0,50}(work|access|load)",
    r"all (submissions?|requests?).{0,30}(fail|broken|not work)",
    r"\b(submissions?|pages?).{0,30}(not work|broken|fail)",
]]

_INVALID_RE = [re.compile(p, re.IGNORECASE) for p in [
    r"^(thank you|thanks|thank u|ty)[!.,\s]*$",
    r"^(hi|hello|hey|good (morning|afternoon|evening))[!.,\s]*$",
    r"^(hi|hello|hey)[,!\s]*(there)?[,!\s]*(thanks|thank you|thank u)?[!.,\s]*$",
    r"^(ok|okay|got it|noted|understood|sure|alright)[!.,\s]*$",
    r"\bwhat is (the name of the actor|the actor in|the name of the movie)\b",
    r"^(none|n/a|na|-)\s*$",
    r"^give me (the code|a script|code) to\b",
]]

# ── Multi-intent detection ────────────────────────────────────────────────────

_MULTI_INTENT_RE = re.compile(
    r"\b(and also|additionally|furthermore|another (question|issue|problem|thing)|"
    r"second(ly| question| issue)?|also (i |i'd |i have |i want )|i also (have|want|need)|"
    r"separately|in addition|besides that|on top of (that|this)|"
    r"my (other|second|next) (question|issue|concern|problem))\b",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    """
    Collapse adversarial formatting before regex safety checks.
    Strips Unicode combining characters (e.g. accents) so "règles" matches "regles".
    """
    text = " ".join((text or "").split())
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def check_escalation(issue: str, subject: str = "") -> tuple[bool, str]:
    """
    Returns (should_escalate: bool, reason: str).
    True means the ticket must be escalated without calling Claude.
    """
    raw_text = f"{subject} {issue}"
    text = _normalize_text(raw_text)
    for regex in _ESC_RE:
        m = regex.search(text) or regex.search(raw_text)
        if m:
            return True, f"matched pattern: '{m.group()}'"
    return False, ""


def classify_request_type(issue: str, subject: str = "") -> str:
    """
    Keyword-based pre-classification. Claude may override this in its response.
    Returns one of: product_issue | feature_request | bug | invalid
    """
    text = _normalize_text(f"{subject} {issue}".strip())

    if not text or len(text) < 8:
        return "invalid"

    for regex in _INVALID_RE:
        if regex.search(text):
            return "invalid"

    for regex in _BUG_RE:
        if regex.search(text):
            return "bug"

    for regex in _FEATURE_RE:
        if regex.search(text):
            return "feature_request"

    return "product_issue"


def detect_multi_intent(issue: str) -> bool:
    """
    Returns True if the ticket appears to contain multiple distinct support intents.
    Used to signal Claude to address all intents in one response.
    """
    return bool(_MULTI_INTENT_RE.search(issue))
