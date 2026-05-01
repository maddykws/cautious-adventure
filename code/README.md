# Support Triage Agent

Terminal-based multi-domain support triage agent for HackerRank Orchestrate. Hybrid sparse-and-dense retrieval, multi-turn agentic tool use, and a Self-RAG-style verifier with full evidence-and-safety audit trail.

## Architecture

```
support_tickets.csv
       |
  [classifier.py]        — rule-based safety/escalation gate (no API call)
       |
  [PII redaction]        — mask emails / phones / cards / order IDs / tokens
       |
  [retriever.py]         — hybrid sparse + dense retrieval
       |                   ├─ TF-IDF (sklearn) candidate generation
       |                   ├─ MiniLM-L6-v2 dense embedding (offline, cached)
       |                   ├─ Reciprocal-Rank Fusion (sparse ∪ dense)
       |                   ├─ domain boost + cross-domain fallback
       |                   └─ falls back to TF-IDF-only on any embed failure
       |
  [answerability check]  — escalate early if corpus coverage is too weak
       |
  [agent loop]           — multi-turn tool use with retrieve():
       |                   ├─ seed retrieval in initial user message
       |                   ├─ Claude may call retrieve(query, company)
       |                   │    up to 3 more times with focused queries
       |                   ├─ aggregated evidence across all iterations
       |                   ├─ MAX_ITERATIONS=4, MAX_TOOL_CALLS=6 hard caps
       |                   └─ graceful corpus-only fallback when no API key
       |
  [verifier.py]          — Self-RAG-style grounding check across full
       |                   evidence set: term-overlap OR TF-IDF cosine
       |
  [classifier override]  — enforce deterministic request_type on hard signals
       |                   (bug / feature_request) to prevent LLM downgrades
       |
  output.csv  +  evidence_audit.jsonl  (per-iteration tool calls included)
```

**Key design decisions:**
- **Hybrid sparse+dense retrieval.** TF-IDF generates a wide candidate set (top 50) and is unioned with the top-50 results from a dense semantic search using `sentence-transformers/all-MiniLM-L6-v2`. Both rankings are fused with **Reciprocal Rank Fusion** (k=60, the standard from Cormack et al. 2009), which is robust to the score-magnitude differences between the two signals. Dense embeddings catch paraphrased queries TF-IDF misses entirely; TF-IDF anchors specificity. The MiniLM model is downloaded once via huggingface_hub and runs CPU-only — no inference API call.
- **Persistent embedding cache.** All 6,100+ corpus chunks are embedded once at first run (~30–60 s on CPU) and cached to `data/index/embeddings.npz`, keyed by a SHA-256 of the corpus contents. Subsequent runs load in <2 s.
- **Fail-soft hybrid.** If `sentence-transformers` / `torch` aren't installed, if the model download fails, if the cache is corrupt — every error path silently disables the dense reranker and the agent runs in pure TF-IDF mode with no degradation in correctness, only in paraphrase recall. The hackathon eval rig must never crash because embeddings aren't available.
- **Section-level chunking** of `.md` corpus articles. Articles are split by headings so later sections (Visa minimum-spend rules, HackerRank team-member removal, Claude data-retention controls) are retrieved directly rather than relying only on page beginnings.
- **Multi-turn agentic tool use.** Claude gets a `retrieve(query, company)` tool. The agent decides whether the seed retrieval is sufficient — if so, it answers directly; if not, it issues focused follow-up queries (up to 3 of them, hard cap at 6 total tool calls). The system prompt is explicit about *when not* to retrieve to prevent over-thinking on borderline tickets. All retrieved chunks are aggregated into a single evidence set for the verifier, and every tool call is logged in the audit trail.
- **Domain boosting** — docs from the ticket's company get a 1.5× relevance boost; other domains get 0.7×. Cross-domain fallback kicks in when the primary domain has no strong match.
- **Domain synonym expansion** maps user phrasing to documentation terms ("employee left" → "Teams Management", "minimum spend" → "merchant minimum limit", "inactivity / lobby" → "ending interview / leave interview").
- **Rule-based safety gate** runs before any API call. It normalizes whitespace and Unicode accents (NFKD) so multilingual and newline prompt-injection attacks are caught deterministically.
- **PII redaction** — emails, phone numbers, card numbers, order IDs (`cs_live_…`), SSNs and API tokens are masked before the ticket is sent to the LLM. The local corpus is unchanged; only the outbound API payload is sanitized.
- **Answerability pre-check** — tickets with retrieval score below 0.018, or off-domain matches under 0.04, are escalated without calling Claude. Tightened from earlier 0.025 / 0.06 thresholds after observing over-escalation on `usable` retrievals.
- **Verifier grounding check** (Self-RAG style) — after Claude drafts a reply, verifiable claim sentences are extracted and checked two ways: lexical term-overlap (≥ 0.50) OR TF-IDF cosine similarity against the retrieved chunks (≥ 0.18). The cosine signal rescues paraphrased / synonym-heavy answers that pure term overlap misses. Overall reply is downgraded to escalation only on strong disagreement (< 30 %), tightened from 40 % to reduce false-negative escalations.
- **Classifier-authoritative request_type** — when the rule-based classifier matches a hard pattern (`bug` for "is down" / "not working", `feature_request` for "please add" / "would be nice"), we override the LLM's choice. The LLM still wins on `invalid` (it has full context to judge actionability).
- **Specific-feature-outage rule** — when a ticket reports a single product feature is "down" / "broken" / "not working" AND the corpus only describes the feature without troubleshooting / status-page / incident-response guidance, the ticket is escalated. Describing a feature is not a substitute for fixing it.
- **Justification hygiene** — numeric retrieval scores (e.g. "0.174 confidence", "score=0.06") are stripped from LLM-generated justifications so the user-facing field stays qualitative; precise numbers live in the audit trail.
- **Graceful no-API-key fallback** — if `ANTHROPIC_API_KEY` is missing or the API is unreachable, the agent runs a deterministic classifier + retriever path that quotes the top corpus chunk verbatim (when retrieval is strong on-domain) or escalates. The evaluator still gets a populated CSV row for every ticket.
- **Robust JSON extraction** — the model is asked for JSON only, but if it appends commentary the parser walks the brace structure (string-literal aware) to recover the first balanced `{…}` object.
- **Evidence & Safety Audit Trail** — every ticket produces a full `AuditEntry` written to `evidence_audit.jsonl`. Use `--trace` for human-readable Rich output.
- **Corpus-grounded responses** — Claude is instructed to use only retrieved excerpts and cite the source article. No hallucinated policies.
- **Fallback escalation** — if Claude's response cannot be parsed or an error occurs, the ticket is escalated (safe default).

## Submission deliverable

The authoritative scored artifact is **`support_tickets/output.csv`**, which is **committed to the repository at the same commit as the code that produced it**. It was generated with `temperature=0`, hybrid sparse+dense retrieval (TF-IDF ∪ MiniLM with RRF fusion), and the multi-turn agent loop. The 29 rows are reproducible: re-running `python code/main.py` against the same inputs and the same model IDs produces the same outputs (modulo any provider-side changes to model behavior).

**If the evaluator re-runs the agent without an `ANTHROPIC_API_KEY`,** the agent silently switches to the deterministic-fallback path: it still produces all 29 rows, but only replies when the top corpus chunk is on-domain *and* the TF-IDF score is ≥ 0.18 (a high bar — most tickets escalate). This is intentional: without the LLM there's no safe way to grounded-paraphrase, so a strict threshold prevents hallucination. **The committed `output.csv` is the deliverable** — re-running without a key will produce a more conservative output.csv than the one shipped.

The evaluator should either: (a) score the committed CSV, or (b) re-run with a key. Re-running without a key will not match the committed numbers.

## Setup

```bash
# 1. Install dependencies (from code/ directory)
pip install -r requirements.txt

# 2. Set your API key
cp ../.env.example ../.env
# Edit ../.env and add: ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# From repo root — process all tickets:
python code/main.py

# Single-ticket Evidence & Safety Audit Trail (--trace):
python code/main.py --ticket-id 17 --trace

# Custom paths:
python code/main.py --input path/to/input.csv --output path/to/output.csv
```

Output is written to `support_tickets/output.csv`.
The full audit trail (one JSON line per ticket) is written to `support_tickets/evidence_audit.jsonl`.

## Output schema

| Column          | Values                                              |
|-----------------|-----------------------------------------------------|
| `status`        | `replied` \| `escalated`                           |
| `product_area`  | support category / domain area                      |
| `response`      | user-facing answer grounded in corpus               |
| `justification` | internal routing/decision explanation               |
| `request_type`  | `product_issue` \| `feature_request` \| `bug` \| `invalid` |

## Evidence & Safety Audit Trail

Every processed ticket produces an `AuditEntry` tracking:

- **Safety gate** — rule-based escalation patterns matched (Unicode-normalized)
- **Retrieval** — top-5 corpus chunks with rank, score, and domain
- **Retrieval quality** — `strong` / `usable` / `weak` / `very_weak` / `no_evidence`
- **Answerability check** — whether corpus coverage is sufficient to answer
- **Multi-intent detection** — flag when ticket contains multiple distinct issues
- **Verifier claims** — per-claim grounding result (matched terms vs corpus)
- **Risk flags** — `weak_evidence`, `off_domain_retrieval`, `verifier_warning`, etc.
- **Confidence band** — `high` / `medium` / `low` / `escalated`

Use `--trace` for a rich human-readable panel view:

```
python code/main.py --ticket-id 27 --trace
```

The raw JSONL audit trail is at `support_tickets/evidence_audit.jsonl`.

## Escalation rules

Tickets are **always escalated** (no Claude call) if they match:
- Fraud, identity theft, stolen cards/accounts
- Cardholder requests to dispute or contest a charge (corpus is merchant-side only)
- Score/grade manipulation requests
- Security vulnerability reports
- System-wide outages
- Non-owner/non-admin account access requests
- Adversarial / prompt-injection payloads
- Infosec compliance form requests

## Testing

```bash
# Deterministic guardrail/retrieval/schema/adversarial tests (no API call):
python code/smoke_tests.py

# Quick end-to-end on sample tickets:
python code/main.py --input support_tickets/sample_support_tickets.csv --output /tmp/sample_out.csv
```
