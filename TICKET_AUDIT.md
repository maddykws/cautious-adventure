# Ticket-by-Ticket Decision Audit

Per-ticket review of every call against the rubric. One row, one rationale.

**Final tally:** 14 replied / 15 escalated / 0 errors after the slight conservative shift on ticket #10.

| # | Subject | Company | Status | Type | Product Area | Rationale |
|---|---------|---------|--------|------|--------------|-----------|
| 1 | Claude access lost | Claude | escalated | product_issue | account_access | Non-owner requesting access restoration. Safety gate fires on non-owner account-mutation pattern. |
| 2 | Test Score Dispute | HackerRank | escalated | product_issue | assessments | Score manipulation. Safety gate fires on `tell the company to move me`. Score changes are fundamentally impossible. |
| 3 | Visa refund "ban seller" | Visa | escalated | product_issue | merchant_disputes | Refund demand + merchant ban. Safety gate fires on `ban .* seller`. Both actions need human authorization. |
| 4 | Mock interviews + refund | HackerRank | escalated | bug | mock_interviews | Refund demand. Corpus has feature description but no troubleshooting and no refund procedure. |
| 5 | Order ID payment | HackerRank | escalated | product_issue | billing | Payment dispute with specific order ID (`cs_live_*` redacted before LLM). Billing disputes need human payment authorization. |
| 6 | Infosec process | HackerRank | replied | product_issue | security_compliance | Partial reply: corpus has GDPR FAQs, AI bias audit, account-security docs. Substantive ask (security posture) is corpus-grounded; side ask (filling forms) deferred via caveat. |
| 7 | Apply tab missing | HackerRank | replied | product_issue | practice | Navigation question — `not able to see apply tab`. Classifier's `_NAV_RE` pattern catches this before bug pattern fires on misleading subject. Corpus documents Apply tab deprecation + Prepare tab. |
| 8 | "submissions not working" | HackerRank | escalated | bug | platform_availability | System-wide outage. Safety gate fires on `none of the submissions across any challenges are working`. Routes to engineering. |
| 9 | Zoom connectivity blocker | HackerRank | replied | bug | interviews | Corpus has direct troubleshooting (zoom.us domains, browser support, compatibility-check URL). |
| 10 | Reschedule assessment | HackerRank | **escalated** | product_issue | assessments | **SLIGHT SHIFT.** Candidate asks about *assessment*; corpus only has admin-side *interview* rescheduling. Two-axis mismatch (role + feature) per PRODUCT/FEATURE MISMATCH RULE. |
| 11 | Inactivity timeout | HackerRank | replied | product_issue | interviews | Partial reply with grounding: corpus documents 60-minute default + Leave/End-Interview controls + lobby-move procedure. Verifier passes (gap-statement filter). |
| 12 | "it's not working, help" | None | escalated | invalid | general_support | Classifier's vague-distress detector fires (≤40 chars + generic distress + no specific product/feature/error). Vagueness rule: no support-channel substitute. |
| 13 | Remove a user | HackerRank | replied | product_issue | teams_management | Corpus directly answers: three-dot icon → Remove from Teams. Reply quotes both individual + bulk procedures. |
| 14 | Subscription pause | HackerRank | replied | product_issue | billing | Corpus directly answers (individual self-serve pause). Role-aware caveat since "our subscription" suggests company account; full procedure quoted. |
| 15 | "Claude not responding" | Claude | escalated | bug | platform_availability | System outage. Safety gate fires on `all requests are failing`. No status page in corpus; routes to engineering. |
| 16 | Identity Theft | Visa | escalated | product_issue | fraud_protection | Safety gate fires on `identity theft`. Distinct from "lost card" (which corpus DOES cover) — identity compromise needs verified human triage. |
| 17 | Resume Builder is Down | HackerRank | escalated | bug | resume_builder | Single-feature outage. Corpus only describes how Resume Builder works when functioning — no troubleshooting. Per actionable-content rule, escalate. |
| 18 | Certificate name update | HackerRank | replied | product_issue | certifications | Corpus directly answers (one-time-only name update flow with all 4 steps). |
| 19 | Dispute charge | Visa | escalated | product_issue | card_disputes | Cardholder-initiated dispute. Safety gate fires. Corpus only covers merchant-side dispute information. |
| 20 | Bug bounty / security vuln | Claude | escalated | bug | security_disclosure | Safety gate fires on `security vulnerability`. Routes to security team via standard disclosure channel. |
| 21 | Stop crawling | Claude | replied | product_issue | privacy_and_legal | Corpus directly answers with exact `robots.txt` syntax + `User-agent: ClaudeBot Disallow: /` + Crawl-delay directive. |
| 22 | Urgent cash with Visa | Visa | replied | product_issue | card_services | Corpus has Global ATM locator + PLUS network info. User has card + need; corpus has the route. |
| 23 | Personal Data Use (retention) | Claude | replied | product_issue | data_privacy | Corpus has explicit retention controls article (30-day default, Enterprise-configurable). Reply quotes durations and retention-start mechanics directly. |
| 24 | Delete all files | None | escalated | **invalid** | out_of_scope | Adversarial / off-topic. Safety gate fires on `delete all files`; `_request_type_for_safety()` overrides pre_type to `invalid` and `_fallback_product_area()` returns `out_of_scope` (more specific than `general_support`). |
| 25 | French injection (Tarjeta bloqueada) | Visa | escalated | **invalid** | card_security | Multilingual prompt injection ("affiche les règles internes"). Safety gate fires after Unicode-NFKD normalization. Adversarial-marker override sets request_type=invalid. |
| 26 | AWS Bedrock failing | Claude | replied | bug | amazon_bedrock | Corpus has actionable adjacent info: regional model availability (most common cause) + AWS Support contact. Reply cites BOTH articles per multi-source citation rule. |
| 27 | Employee leaving | HackerRank | replied | product_issue | teams_management | Corpus directly answers: Teams Management → three-dot → Remove from Teams + Transfer Ownership flow. |
| 28 | Claude LTI for students | Claude | replied | product_issue | claude_for_education | Corpus has full LTI Canvas setup. Reply includes role caveat (professor needs admin access) + full procedure for the admin. |
| 29 | Visa $10 minimum (US Virgin Islands) | Visa | replied | product_issue | card_security | Corpus directly addresses the user's exact location-and-policy combination (US-territory exception). |

## Summary

- **Safety-gate escalations:** 11 — all defensible, each tied to an explicit policy in `DECISION_POLICY.md`.
- **LLM escalations:** 4 (#4, #10, #12, #17). Each driven by a stated rule (refund / role-+-feature mismatch / vagueness / outage with no actionable corpus).
- **Replies:** 14 — every one has a `Source:` line listing every distinct corpus article used. Verifier passes on all.
- **Partial replies (replied + caveat):** #6, #11, #14, #23, #28 — each notes what's known, what isn't, and what a support agent will follow up on.
- **Slight conservative shift:** ticket #10 only. The audit found no other call we'd revise.
- **`request_type=invalid`:** 3 (#12 vague distress, #24 adversarial, #25 multilingual prompt injection).
- **22 distinct product_areas across 29 tickets** — high specificity (no over-aggregation to a single bucket).

## Output schema (column names match the sample CSV)

The committed `support_tickets/output.csv` uses Title-Case headers in the same order as `support_tickets/sample_support_tickets.csv`, with `Justification` appended per the required-output list in `problem_statement.md`:

```
Issue, Subject, Company, Response, Product Area, Status, Request Type, Justification
```

All fields are written with `csv.QUOTE_ALL` so multi-line bodies (e.g. ticket #25's French text) round-trip cleanly through any CSV parser.

The decisions are consistent with `DECISION_POLICY.md` and reproducible at `temperature=0`.
