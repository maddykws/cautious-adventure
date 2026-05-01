"""
HackerRank Orchestrate — Support Triage Agent

Usage (from repo root):
    python code/main.py                               # process all tickets
    python code/main.py --ticket-id 17 --trace        # single ticket with audit trail
    python code/main.py --input X.csv --output Y.csv  # custom paths
"""

import csv
import json
import sys
import time
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.rule import Rule
from rich import box

sys.path.insert(0, str(Path(__file__).parent))
from agent import triage_with_audit, get_retriever

console = Console()

REPO_ROOT   = Path(__file__).parent.parent
INPUT_CSV   = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV  = REPO_ROOT / "support_tickets" / "output.csv"
AUDIT_JSONL = REPO_ROOT / "support_tickets" / "evidence_audit.jsonl"

OUTPUT_FIELDS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type",
    "justification",
]


# ── Preflight ─────────────────────────────────────────────────────────────────

def _preflight(retriever) -> None:
    import os
    hard_issues:  list[str] = []
    soft_issues:  list[str] = []

    if not os.getenv("ANTHROPIC_API_KEY"):
        soft_issues.append(
            "ANTHROPIC_API_KEY is not set — falling back to deterministic "
            "(corpus-only) responses. Output CSV will still be produced."
        )
    if not retriever.docs:
        hard_issues.append("Corpus is empty — check that data/ contains .md files")
    elif len(retriever.docs) < 100:
        hard_issues.append(
            f"Corpus looks incomplete ({len(retriever.docs)} chunks, expected 1000+)"
        )
    test_docs = retriever.retrieve("lost visa card stolen", company="Visa", top_k=1)
    if not test_docs:
        hard_issues.append("Retrieval sanity check failed — no docs for 'lost visa card'")

    for msg in hard_issues:
        console.print(f"[red][PREFLIGHT][/red] {msg}")
    for msg in soft_issues:
        console.print(f"[yellow][PREFLIGHT][/yellow] {msg}")
    if hard_issues:
        console.print("[yellow]Proceeding with caution.[/yellow]\n")
    elif soft_issues:
        console.print("[yellow]Proceeding in deterministic-fallback mode.[/yellow]\n")
    else:
        console.print("[green][PREFLIGHT][/green] All checks passed.\n")


# ── Trace / audit print ───────────────────────────────────────────────────────

def _band_color(band: str) -> str:
    return {"high": "green", "medium": "yellow", "low": "red", "escalated": "dim"}.get(band, "white")


def print_audit_trace(entry, ticket_num: int | None = None) -> None:
    """Print a human-readable Evidence & Safety Audit Trail for one ticket."""
    label = f"Ticket #{entry.ticket_id}" if ticket_num is None else f"Ticket #{ticket_num}"
    console.print()
    console.print(Rule(f"[bold cyan]EVIDENCE & SAFETY AUDIT TRAIL — {label}[/bold cyan]"))

    # ── Ticket ────────────────────────────────────────────────────────────────
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("key",   style="cyan",  min_width=10)
    t.add_column("value", style="white", min_width=50)
    t.add_row("Issue",   entry.issue[:120] + ("..." if len(entry.issue) > 120 else ""))
    t.add_row("Subject", entry.subject or "(none)")
    t.add_row("Company", entry.company or "Unknown")
    console.print(Panel(t, title="[bold]Ticket", border_style="blue", padding=(0, 1)))

    # ── Safety gate ───────────────────────────────────────────────────────────
    sg_color = "red" if entry.safety_triggered else "green"
    sg_text = (
        f"[red]TRIGGERED[/red] — {entry.safety_reason}"
        if entry.safety_triggered
        else "[green]PASS[/green] — no escalation patterns matched"
    )
    console.print(Panel(sg_text, title="[bold]Safety Gate", border_style=sg_color, padding=(0, 1)))

    if entry.safety_triggered:
        _print_decision(entry)
        return

    # ── Retrieval ─────────────────────────────────────────────────────────────
    q_color = {"strong": "green", "usable": "yellow", "weak": "red", "very_weak": "red", "no_evidence": "red"}.get(entry.retrieval_quality, "white")
    ev_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    ev_table.add_column("Rank", style="dim",   width=5)
    ev_table.add_column("Score", style="cyan",  width=7)
    ev_table.add_column("Domain", style="yellow", width=12)
    ev_table.add_column("Corpus Section", style="white")
    for ev in entry.evidence:
        ev_table.add_row(str(ev.rank), f"{ev.score:.3f}", ev.domain, ev.title[:80])

    retrieval_text = (
        f"Quality: [{q_color}]{entry.retrieval_quality}[/{q_color}]  "
        f"Top score: {entry.top_score:.3f}  "
        f"Domain match: {entry.domain_match}\n"
    )
    console.print(Panel(retrieval_text + "\n", title="[bold]Retrieval", border_style=q_color, padding=(0, 1)))
    console.print(ev_table)

    # ── Answerability ─────────────────────────────────────────────────────────
    ans_color = "green" if entry.answerability_passed else "red"
    ans_status = "[green]ANSWERABLE[/green]" if entry.answerability_passed else "[red]ESCALATED[/red]"
    console.print(Panel(
        f"{ans_status} — {entry.answerability_reason}",
        title="[bold]Answerability Check", border_style=ans_color, padding=(0, 1),
    ))

    if not entry.answerability_passed:
        _print_decision(entry)
        return

    # ── Multi-intent ──────────────────────────────────────────────────────────
    if entry.multi_intent:
        console.print(Panel(
            "[yellow]DETECTED[/yellow] — Multiple distinct support intents found. "
            "Claude instructed to address all intents.",
            title="[bold]Multi-Intent", border_style="yellow", padding=(0, 1),
        ))

    # ── Verifier ─────────────────────────────────────────────────────────────
    if entry.verifier_claims:
        v_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        v_table.add_column("Supported", width=10)
        v_table.add_column("Claim",     min_width=40)
        v_table.add_column("Matched terms", style="dim")
        for claim in entry.verifier_claims:
            icon = "[green]YES[/green]" if claim.supported else "[red]NO [/red]"
            v_table.add_row(icon, claim.text[:80], ", ".join(claim.matched_terms[:4]))

        v_color = "green" if entry.verifier_overall_supported else "red"
        v_status = "[green]SUPPORTED[/green]" if entry.verifier_overall_supported else "[red]UNSUPPORTED[/red]"
        console.print(Panel(
            f"Verdict: {v_status}  Support ratio: {entry.verifier_support_ratio:.0%}",
            title="[bold]Verifier (Grounding Check)", border_style=v_color, padding=(0, 1),
        ))
        console.print(v_table)
    else:
        console.print(Panel("[dim]No verifiable claims extracted (escalated response).[/dim]",
                            title="[bold]Verifier", border_style="dim", padding=(0, 1)))

    # ── Risk flags ────────────────────────────────────────────────────────────
    if entry.risk_flags:
        flags_str = "  ".join(f"[yellow]{f}[/yellow]" for f in entry.risk_flags)
        console.print(Panel(flags_str, title="[bold]Risk Flags", border_style="yellow", padding=(0, 1)))

    # ── Decision ──────────────────────────────────────────────────────────────
    _print_decision(entry)


def _print_decision(entry) -> None:
    band_color = _band_color(entry.confidence_band)
    status_color = "green" if entry.status == "replied" else "yellow"
    decision_text = (
        f"Status:     [{status_color}]{entry.status.upper()}[/{status_color}]\n"
        f"Intent:     {entry.product_area}\n"
        f"Type:       {entry.request_type}\n"
        f"Confidence: [{band_color}]{entry.confidence_band}[/{band_color}]\n\n"
        f"[bold]Response:[/bold]\n{entry.response[:600]}"
        + ("..." if len(entry.response) > 600 else "")
    )
    console.print(Panel(decision_text, title="[bold]Decision", border_style="cyan", padding=(0, 1)))
    console.print()


# ── Batch run ─────────────────────────────────────────────────────────────────

def run(input_path: Path, output_path: Path, audit_path: Path) -> None:
    console.print(Panel(
        "[bold cyan]HackerRank Orchestrate — Support Triage Agent[/bold cyan]\n"
        "[dim]Multi-domain: HackerRank · Claude · Visa  |  RAG + Reranker + Verifier + Claude sonnet-4[/dim]",
        border_style="cyan",
    ))

    console.print("[yellow]Indexing corpus (one-time)...[/yellow]")
    t0 = time.monotonic()
    retriever = get_retriever()
    elapsed = time.monotonic() - t0
    console.print(f"[green]Indexed[/green] {len(retriever.docs)} corpus chunks in {elapsed:.1f}s\n")

    _preflight(retriever)

    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    console.print(f"[bold]Processing {len(rows)} tickets...[/bold]\n")

    results: list[dict] = []
    audit_entries: list[dict] = []
    stats = {"replied": 0, "escalated": 0, "errors": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Triaging...", total=len(rows))

        for i, row in enumerate(rows):
            issue   = (row.get("Issue")   or row.get("issue")   or "").strip()
            subject = (row.get("Subject") or row.get("subject") or "").strip()
            company = (row.get("Company") or row.get("company") or "").strip()

            try:
                result, entry = triage_with_audit(issue, subject, company, ticket_id=i + 1)
                stats[result.status] += 1
                results.append({
                    "issue":           issue,
                    "subject":         subject,
                    "company":         company,
                    "response":        result.response,
                    "product_area":    result.product_area,
                    "status":          result.status,
                    "request_type":    result.request_type,
                    "justification":   result.justification,
                })
                audit_entries.append(entry.to_dict())
            except Exception as exc:
                stats["errors"] += 1
                console.print(f"\n[red]Row {i + 1} error:[/red] {exc}")
                results.append({
                    "issue":           issue,
                    "subject":         subject,
                    "company":         company,
                    "response":        "This issue requires human review. A support agent will contact you shortly.",
                    "product_area":    "general_support",
                    "status":          "escalated",
                    "request_type":    "product_issue",
                    "justification":   f"Processing error: {str(exc)[:120]}",
                })

            progress.advance(task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    with open(audit_path, "w", encoding="utf-8") as f:
        for entry_dict in audit_entries:
            f.write(json.dumps(entry_dict, ensure_ascii=False) + "\n")

    table = Table(title="Triage Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric",  style="cyan",  min_width=18)
    table.add_column("Value",   style="green", min_width=10)
    table.add_row("Total tickets",  str(len(rows)))
    table.add_row("Replied",        str(stats["replied"]))
    table.add_row("Escalated",      str(stats["escalated"]))
    table.add_row("Errors",         str(stats["errors"]))
    table.add_row("Output CSV",     str(output_path))
    table.add_row("Audit trail",    str(audit_path))
    console.print(table)
    console.print(f"\n[bold green]Done![/bold green]")


# ── Single-ticket trace mode ──────────────────────────────────────────────────

def run_trace(input_path: Path, ticket_id: int) -> None:
    console.print("[yellow]Indexing corpus (one-time)...[/yellow]")
    t0 = time.monotonic()
    retriever = get_retriever()
    elapsed = time.monotonic() - t0
    console.print(f"[green]Indexed[/green] {len(retriever.docs)} corpus chunks in {elapsed:.1f}s\n")

    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if ticket_id < 1 or ticket_id > len(rows):
        console.print(f"[red]Invalid --ticket-id {ticket_id}. File has {len(rows)} tickets (1–{len(rows)}).[/red]")
        sys.exit(1)

    row = rows[ticket_id - 1]
    issue   = (row.get("Issue")   or row.get("issue")   or "").strip()
    subject = (row.get("Subject") or row.get("subject") or "").strip()
    company = (row.get("Company") or row.get("company") or "").strip()

    console.print(f"[bold]Running trace for ticket #{ticket_id} of {len(rows)}...[/bold]")
    result, entry = triage_with_audit(issue, subject, company, ticket_id=ticket_id)
    print_audit_trace(entry, ticket_num=ticket_id)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Support triage agent — HackerRank Orchestrate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python code/main.py                          # process all tickets\n"
            "  python code/main.py --ticket-id 17 --trace  # single ticket audit trail\n"
            "  python code/main.py --input X.csv           # custom input file\n"
        ),
    )
    parser.add_argument("--input",     default=str(INPUT_CSV),   help="Input CSV path")
    parser.add_argument("--output",    default=str(OUTPUT_CSV),  help="Output CSV path")
    parser.add_argument("--audit",     default=str(AUDIT_JSONL), help="Audit trail JSONL path")
    parser.add_argument("--ticket-id", type=int, default=None,   help="Process a single ticket by 1-based row number")
    parser.add_argument("--trace",     action="store_true",      help="Print Evidence & Safety Audit Trail (use with --ticket-id)")
    args = parser.parse_args()

    if args.ticket_id is not None:
        run_trace(Path(args.input), args.ticket_id)
    else:
        run(Path(args.input), Path(args.output), Path(args.audit))


if __name__ == "__main__":
    main()
