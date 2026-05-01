# Support Triage Agent

Terminal-based multi-domain support triage agent for HackerRank Orchestrate.

## Architecture

```
support_tickets.csv
       |
  [classifier.py]        — rule-based safety/escalation gate (no API call)
       |
  [retriever.py]         — section-level TF-IDF search over corpus .md files
       |
  [Claude sonnet-4-6]    — structured JSON response grounded in corpus
       |
  output.csv
```

**Key design decisions:**
- **Section-level TF-IDF retrieval** over the provided `.md` corpus. Articles are split by headings so later sections (for example Visa minimum-spend rules or HackerRank team-member removal) can be retrieved directly instead of relying only on the beginning of a page.
- **Domain synonym expansion** maps user phrasing to documentation terms (for example "employee left" -> "Teams Management", "minimum spend" -> "merchant minimum limit").
- **Rule-based safety gate** runs before any API call. It normalizes whitespace and accents so multilingual/newline prompt-injection attacks are caught deterministically.
- **Retrieval quality notes** are passed to Claude so weak or off-domain evidence is treated as a signal to escalate rather than guess.
- **Corpus-grounded responses** — Claude is instructed to use only retrieved excerpts. No hallucinated policies.
- **Domain boosting** — docs from the ticket's company get a 1.5× relevance boost; other domains get 0.7× (cross-domain still searchable for `company=None`).
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
# From repo root:
python code/main.py

# Custom paths:
python code/main.py --input path/to/input.csv --output path/to/output.csv
```

Output is written to `support_tickets/output.csv`.

## Output schema

| Column          | Values                                              |
|-----------------|-----------------------------------------------------|
| `status`        | `replied` \| `escalated`                           |
| `product_area`  | support category / domain area                      |
| `response`      | user-facing answer grounded in corpus               |
| `justification` | internal routing/decision explanation               |
| `request_type`  | `product_issue` \| `feature_request` \| `bug` \| `invalid` |

## Escalation rules

Tickets are **always escalated** (no Claude call) if they match:
- Fraud, identity theft, stolen cards/accounts
- Score/grade manipulation requests
- Security vulnerability reports
- System-wide outages
- Non-owner/non-admin account access requests
- Adversarial / prompt-injection payloads
- Infosec compliance form requests

## Testing

```bash
# Quick smoke test against sample tickets:
python code/main.py --input support_tickets/sample_support_tickets.csv --output /tmp/sample_out.csv

# Deterministic guardrail/retrieval/schema tests (no API call):
python code/smoke_tests.py
```
