# Support Triage Agent

Terminal-based multi-domain support triage agent for HackerRank Orchestrate.

## Architecture

```
support_tickets.csv
       |
  [classifier.py]        — rule-based safety/escalation gate (no API call)
       |
  [retriever.py]         — section-level TF-IDF search over 774 corpus .md files
       |                   RRF reranker + domain boosting + cross-domain fallback
       |
  [answerability check]  — escalate early if corpus coverage is too weak
       |
  [Claude claude-sonnet-4] — structured JSON response grounded in corpus excerpts
       |
  [verifier.py]          — Self-RAG-style grounding check: claims vs corpus
       |                   downgrades to escalation if < 40% of claims supported
       |
  output.csv  +  evidence_audit.jsonl
```

**Key design decisions:**
- **Section-level TF-IDF retrieval** over the provided `.md` corpus. Articles are split by headings so later sections (Visa minimum-spend rules, HackerRank team-member removal) are retrieved directly rather than relying only on page beginnings.
- **RRF reranker** — combines TF-IDF cosine score with lexical term overlap (α = 0.15) to surface the most relevant chunk, not just the most lexically similar.
- **Domain boosting** — docs from the ticket's company get a 1.5× relevance boost; other domains get 0.7×. Cross-domain fallback kicks in when the primary domain has no strong match.
- **Domain synonym expansion** maps user phrasing to documentation terms ("employee left" → "Teams Management", "minimum spend" → "merchant minimum limit").
- **Rule-based safety gate** runs before any API call. It normalizes whitespace and Unicode accents (NFKD) so multilingual and newline prompt-injection attacks are caught deterministically.
- **Answerability pre-check** — if retrieval score < 0.025 or weak off-domain match, the ticket is escalated immediately without calling Claude. Prevents hallucination without an extra LLM call.
- **Verifier grounding check** (Self-RAG style) — after Claude drafts a reply, verifiable claim sentences are extracted and term-matched against the retrieved corpus. If fewer than 60% of claims are supported, the response is downgraded to escalation.
- **Evidence & Safety Audit Trail** — every ticket produces a full `AuditEntry` written to `evidence_audit.jsonl`. Use `--trace` for human-readable Rich output.
- **Corpus-grounded responses** — Claude is instructed to use only retrieved excerpts and cite the source article. No hallucinated policies.
- **Fallback escalation** — if Claude's response cannot be parsed or an error occurs, the ticket is escalated (safe default).

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
