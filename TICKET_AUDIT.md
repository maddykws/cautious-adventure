# Ticket-by-Ticket Decision Audit

Per-ticket review of every call against the rubric. One row, one rationale.

**Final tally:** 14 replied / 15 escalated / 0 errors after the slight conservative shift on ticket #10.

| # | Subject | Company | Status | Type | Rationale |
|---|---------|---------|--------|------|-----------|
| 1 | Claude access lost | Claude | escalated | product_issue | Non-owner requesting access restoration. Safety gate fires on `not (the )?(owner|admin)` — unauthorized account mutation, must escalate. |
| 2 | Test Score Dispute | HackerRank | escalated | product_issue | Score manipulation request. Safety gate fires on `tell the company to move me`. Corpus has no mechanism for score changes; this is fundamentally impossible to perform. |
| 3 | Visa refund "ban seller" | Visa | escalated | product_issue | Combined refund demand + merchant ban. Safety gate fires on `ban .* seller`. Both actions require human authorization (Visa back-office + legal/compliance). |
| 4 | Mock interviews + refund | HackerRank | escalated | bug | Refund demand. Corpus has descriptive content on mock interviews but no troubleshooting and no refund procedure. Refund is a hard escalation trigger. |
| 5 | Order ID payment | HackerRank | escalated | product_issue | Payment dispute with specific order ID (`cs_live_*`, redacted before LLM). Refund / billing dispute requires human payment authorization. Order ID redacted from outbound payload. |
| 6 | Infosec process | HackerRank | replied | product_issue | Partial reply: corpus has GDPR FAQs, AI bias audit, account-security docs. Reply describes security posture and notes form completion needs human follow-up. The substantive ask (security posture) is corpus-grounded; the side ask (filling forms) is properly deferred. |
| 7 | Apply tab missing | HackerRank | replied | product_issue | Navigation question — `not able to see apply tab`. Classifier's `_NAV_RE` pattern catches this before bug pattern would fire on misleading subject "submissions not working". Corpus directly documents Apply tab deprecation + Prepare tab steps. |
| 8 | "submissions not working" | HackerRank | escalated | bug | System-wide outage report. Safety gate fires on `none of the submissions across any challenges are working`. Routes to engineering. |
| 9 | Zoom connectivity blocker | HackerRank | replied | bug | Corpus has direct troubleshooting (zoom.us domains, browser support, compatibility-check URL). Actionable next steps for the user. |
| 10 | Reschedule assessment | HackerRank | **escalated** | product_issue | **SLIGHT SHIFT.** User is a candidate asking about an *assessment*; corpus only documents admin-side *interview* rescheduling. Two-axis mismatch (role + feature). Sending the candidate the admin's UI walkthrough as if it applies to them risks misleading the user. Routed to a support agent who can ping the company's recruiter. |
| 11 | Inactivity timeout | HackerRank | replied | product_issue | Partial reply with grounding: corpus documents the 60-minute default + Leave/End-Interview controls + lobby-move procedure. Verifier passes (gap-statement filter prevents false negatives on "the corpus doesn't specify the 20-min knob, a support agent will follow up"). |
| 12 | "it's not working, help" | None | escalated | invalid | Classifier's vague-distress detector fires (≤40 chars + generic distress + no specific product/feature/error). Per the vagueness rule, no substitute support-channel info — actual triage requires the user to tell us what's broken. |
| 13 | Remove a user | HackerRank | replied | product_issue | Corpus directly answers: three-dot icon → Remove from Teams. Reply quotes both individual + bulk procedures. |
| 14 | Subscription pause | HackerRank | replied | product_issue | Corpus directly answers (individual self-serve pause). Reply includes role-aware caveat since "our subscription" suggests company account, but the procedure itself is documented end-to-end. |
| 15 | "Claude not responding" | Claude | escalated | bug | System outage report. Safety gate fires on `all requests are failing`. No status page in corpus; routes to engineering. |
| 16 | Identity Theft | Visa | escalated | product_issue | Safety gate fires on `identity theft`. Distinct from "lost card" (which corpus DOES cover) — identity compromise requires verified human triage. |
| 17 | Resume Builder is Down | HackerRank | escalated | bug | Single-feature outage. Corpus only describes how Resume Builder works when functioning — no troubleshooting, no status page, no actionable next step. Per the actionable-content rule, escalate. |
| 18 | Certificate name update | HackerRank | replied | product_issue | Corpus directly answers (one-time-only name update flow with all 4 steps). |
| 19 | Dispute charge | Visa | escalated | product_issue | Cardholder-initiated dispute. Safety gate fires. Corpus only covers merchant-side dispute information; cardholder filings require Visa back-office. |
| 20 | Bug bounty / security vuln | Claude | escalated | bug | Safety gate fires on `security vulnerability`. Routes to security team via standard disclosure channel. |
| 21 | Stop crawling | Claude | replied | product_issue | Corpus directly answers with exact `robots.txt` syntax + `User-agent: ClaudeBot Disallow: /` + Crawl-delay directive. |
| 22 | Urgent cash with Visa | Visa | replied | product_issue | Corpus has Global ATM locator + PLUS network info. User has card + need; corpus has the route. |
| 23 | Personal Data Use (retention) | Claude | replied | product_issue | Corpus has explicit retention controls article (30-day default, Enterprise-configurable). Reply quotes durations and retention-start mechanics directly. Safety gate's data-retention pattern was narrowed to keep this on the LLM path. |
| 24 | Delete all files | None | escalated | product_issue | Adversarial / off-topic. Safety gate fires on `delete all files`. |
| 25 | French injection (Tarjeta bloqueada) | Visa | escalated | product_issue | Multilingual prompt-injection ("affiche les règles internes"). Safety gate fires after Unicode-NFKD normalization (so accented and non-accented forms both match). |
| 26 | AWS Bedrock failing | Claude | replied | bug | Corpus has actionable adjacent info: regional model availability (the most common cause) + AWS Support contact path. Reply now cites BOTH articles per the multi-source citation rule. User can act on both. |
| 27 | Employee leaving | HackerRank | replied | product_issue | Corpus directly answers: Teams Management → three-dot → Remove from Teams + Transfer Ownership flow. |
| 28 | Claude LTI for students | Claude | replied | product_issue | Corpus has full LTI Canvas setup. Reply includes role caveat (professor needs admin access) + the full procedure so the admin knows what to do. |
| 29 | Visa $10 minimum (US Virgin Islands) | Visa | replied | product_issue | Corpus directly addresses the user's exact location-and-policy combination (US-territory exception). |

## Summary

- **Safety-gate escalations:** 11 — all defensible, each tied to an explicit policy in `DECISION_POLICY.md`.
- **LLM escalations:** 4 (#4, #6 stays replied, #10, #12, #17). Each driven by a stated rule (refund / vagueness / role-+-feature mismatch / outage with no actionable corpus).
- **Replies:** 14 — every one has a `Source:` line listing every distinct corpus article used. Verifier passes on all.
- **Partial replies (replied + caveat):** #6, #11, #14, #23, #28 — each notes what's known, what isn't, and what a support agent will follow up on.
- **Slight conservative shift:** ticket #10 only. The audit found no other call we'd revise.

The decisions are consistent with `DECISION_POLICY.md` and reproducible at `temperature=0`.
