# Decision Policy

The triage agent's decisions are governed by a written policy. The classifier, retriever, prompt, and verifier all enforce it. This document is the canonical reference; the code is the implementation.

The policy has four parts:
1. **Safety gate** — hard rules, deterministic, run before any LLM call.
2. **Reply-vs-escalate** — when the LLM should answer vs hand off.
3. **Partial-reply pattern** — when the corpus addresses *some* of the ticket.
4. **Justification template** — every decision is explained in a fixed format.

---

## 1. Safety gate (deterministic, pre-LLM)

These are forced escalations. They never reach the LLM, so they cannot be argued out of by clever prompting.

| Trigger | Why | Code |
|---|---|---|
| Identity theft | Personal-identity compromise needs verified human triage | `\bidentity theft\b` |
| Unauthorized transactions / fraudulent charges | Cardholder fraud must reach Visa back-office | `\bunauthorized (charge\|transaction\|debit\|payment)\b` |
| Cardholder dispute / chargeback | Corpus covers merchant-side only; cardholder filings need Visa | `\bdispute (a \|this \|the )?(charge\|transaction\|payment\|chargeback)\b` |
| Security vulnerability / bug bounty | Routes to security team via formal disclosure | `\bsecurity vulnerability\b`, `\bbug bounty\b` |
| Score / grade manipulation | Fundamentally impossible — agent must never attempt | `\b(increase\|change\|modify\|adjust\|fix) my (score\|grade\|result\|mark)\b` |
| Non-owner / non-admin account mutation | Unauthorized account changes need owner verification | `\bnot (an?\|the)?\s*(workspace\s+)?(owner\|admin)\b.{0,80}\b(access\|restore\|...)\b` |
| Removing seller / merchant / vendor | Cannot ban third parties on user's behalf | `(remove\|delete\|ban) .{0,50}(seller\|merchant\|vendor)` |
| System-wide outage reports | Routes to engineering, not support | `site is down`, `none of the (submissions\|...) ... working`, `all (requests\|submissions) ... failing` |
| Adversarial / prompt-injection payloads | Includes multilingual variants (NFKD-normalized) | `delete all files`, `\brm\s+-rf\b`, `ignore (previous\|all) instructions`, `affiche .{0,80}(règles internes\|...)` |
| Infosec questionnaire / form completion | Cannot fill compliance forms on user's behalf | `(fill\|complete) .{0,30}(infosec\|security\|compliance) (form\|questionnaire)` |
| Data-retention DURATION questions | Narrowed pattern: only `kept`/`retained`/`stored` (NOT `used`, since the corpus DOES document model-improvement usage and retention defaults) | `how long ... data ... (kept\|retained\|stored)`, `\b(data retention period\|retention period)\b` |

The full pattern list lives in `code/classifier.py::_ESCALATION_PATTERNS`. Patterns are normalized via Unicode NFKD before matching, so `règles` matches `regles` (defends against accent-based injection).

---

## 2. Reply vs Escalate (LLM, prompt-driven)

When the safety gate doesn't fire, the LLM applies these rules in priority order:

### Always escalate (LLM-decided)
- **Refund demands / billing disputes.** Requires human payment authorization.
- **Tasks an agent fundamentally cannot perform** on the user's behalf:
  filling a vendor form, processing a chargeback, modifying account permissions for someone else.
- **Vague distress with no specifics** — "it's not working, help", "broken, please fix" with no product/feature/error context. *Generic support-channel info is not a substitute for understanding the user's problem.*
- **Single-feature outage with no actionable corpus content.** Corpus describing the feature when the user reports it's broken doesn't help — escalate.
- **Product/feature mismatch.** When the user explicitly asks about feature X and the corpus only documents adjacent feature Y. (Example: candidate asks about *assessment* rescheduling; corpus has *interview* rescheduling — Y's procedure is not a substitute for X.)

### Always reply (when corpus directly answers)
- "How do I X" questions where the corpus has a step-by-step procedure for X.
- Lost / stolen card or cheque (Visa corpus has the procedure).
- Account / team management (HackerRank corpus covers it).
- Configuration / setup with documented steps.

### The actionable-content test (decisive)
For everything in between, ask: *Does the corpus give the user (or someone they can forward this to) any concrete next step?*
- **Yes** → reply. Even if the answer isn't perfect, actionable next-step info beats a generic handoff.
- **No** → escalate.

This test is what tells the agent that "Claude has stopped working" gets escalated (no troubleshooting in corpus) but "all my Bedrock requests are failing" gets a reply (corpus has regional-availability check, the most common root cause).

---

## 3. Partial reply pattern

Used when the corpus addresses *some* of the ticket but not all. Format:

1. Open with the relevant corpus content (cite it).
2. Acknowledge gaps in the corpus explicitly.
3. End with: "For [the unanswerable part], a support agent will follow up."

**Use when** the corpus has actionable adjacent info AND the user can act on what's there (or forward it to someone who can).

**Don't use when:**
- The central ask is the unanswerable part (e.g., refund demand → escalate).
- The corpus is purely descriptive and the user reports an outage (escalate).
- The user is in a different role AND a different feature (escalate — see ticket #10 in `TICKET_AUDIT.md`).

### Role-mismatch caveat
When the corpus only documents one role's procedure (admin / owner) and the user is in a different role (candidate / non-owner), open the reply with a one-sentence acknowledgment before quoting the procedure. The user is then equipped to forward it. Examples in the prompt: ticket #28 (professor needs Canvas admin), #14 (company subscription needs admin).

---

## 4. Justification template

Every justification follows this format, regardless of code path (safety, answerability, LLM, fallback):

```
Decision: [replied | escalated]. Why: [one sentence on corpus coverage and the relevant policy]. Next: [concrete next step — what the user does, or which team handles the escalation].
```

Examples (one per code path):

- **Replied (LLM):** "Decision: replied. Why: corpus directly documents the Apply tab deprecation and Prepare tab replacement. Next: navigate to Prepare > Prepare by Topics for coding challenges."
- **Replied (partial):** "Decision: replied. Why: corpus has the 60-minute default interview-end timeout and Leave/End controls but no per-account inactivity knob. Next: try the Leave-Interview / lobby controls; for the specific timeout duration, a support agent will follow up."
- **Escalated (safety):** "Decision: escalated. Why: identity-theft safety rule — personal-identity compromise needs verified human triage. Next: a Visa support agent will contact you to verify identity through the cardholder."
- **Escalated (answerability):** "Decision: escalated. Why: corpus has no on-topic excerpts for this query. Next: a support agent will route the ticket to the right team."
- **Escalated (vague):** "Decision: escalated. Why: ticket lacks any specific product, feature, or error context. Next: a support agent will reach out to identify the issue."
- **Escalated (role+feature mismatch):** "Decision: escalated. Why: candidate-side reschedule procedure is not in corpus and the documented admin-side interview reschedule is not equivalent to assessment reschedule. Next: a support agent will route this to your company's recruiting contact."

The audit trail (`support_tickets/evidence_audit.jsonl`) carries the full reasoning per ticket — retrieval scores, evidence titles, tool calls, verifier per-claim breakdown, risk flags, confidence band — for any decision that needs to be defended in detail.

---

## What this policy buys us

- **Reproducibility.** `temperature=0` plus deterministic safety/classifier/retriever paths means every run on the same corpus produces the same output.
- **Defensibility.** Every call in `support_tickets/output.csv` can be traced to a specific rule above; see `TICKET_AUDIT.md` for the row-by-row mapping.
- **Safety.** No reply can be issued without on-corpus grounding; the verifier downgrades replies whose claims fail both lexical-overlap AND semantic-cosine grounding.
- **No hallucinated policies.** The agent never speaks for HackerRank / Anthropic / Visa beyond what the corpus says. Limits are stated, not papered over.
