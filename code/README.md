# Support Triage Agent

Terminal-based multi-domain support triage agent for HackerRank Orchestrate.

## Architecture

```
support_tickets.csv
       |
  [classifier.py]        — rule-based safety/escalation gate (no API call)
       |
  [retriever.py]         — TF-IDF search over 774 corpus .md files
       |
  [Claude sonnet-4-6]    — structured JSON response grounded in corpus
       |
  output.csv
```

**Key design decisions:**
- **TF-IDF retrieval** over all 774 `.md` files (HackerRank: 438, Claude: 322, Visa: 14). Built once at startup, reused for all tickets.
- **Rule-based safety gate** runs before any API call. Catches fraud, score manipulation, adversarial prompts, system outages, and prompt-injection attacks deterministically.
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
```
