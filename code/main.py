"""
HackerRank Orchestrate — Support Triage Agent
Entry point: reads support_tickets/support_tickets.csv, writes support_tickets/output.csv.

Usage (from repo root):
    python code/main.py
    python code/main.py --input path/to/input.csv --output path/to/output.csv
"""

import csv
import sys
import time
import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from agent import triage, get_retriever

console = Console()

REPO_ROOT  = Path(__file__).parent.parent
INPUT_CSV  = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"

OUTPUT_FIELDS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type",
    "retrieval_score", "multi_intent", "justification",
]


def _preflight(retriever) -> None:
    """Validate environment and corpus before processing tickets."""
    import os
    issues = []

    if not os.getenv("ANTHROPIC_API_KEY"):
        issues.append("ANTHROPIC_API_KEY is not set — set it in .env")
    if not retriever.docs:
        issues.append("Corpus is empty — check that data/ directory contains .md files")
    elif len(retriever.docs) < 100:
        issues.append(f"Corpus looks incomplete ({len(retriever.docs)} chunks, expected 1000+)")

    test_docs = retriever.retrieve("lost visa card stolen", company="Visa", top_k=1)
    if not test_docs:
        issues.append("Retrieval sanity check failed — no docs for 'lost visa card'")

    if issues:
        for msg in issues:
            console.print(f"[red][PREFLIGHT][/red] {msg}")
        console.print("[yellow]Proceeding with caution.[/yellow]\n")
    else:
        console.print("[green][PREFLIGHT][/green] All checks passed.\n")


def run(input_path: Path, output_path: Path) -> None:
    console.print(Panel(
        "[bold cyan]HackerRank Orchestrate — Support Triage Agent[/bold cyan]\n"
        "[dim]Multi-domain: HackerRank · Claude · Visa  |  RAG + Reranker + Claude sonnet-4[/dim]",
        border_style="cyan",
    ))

    console.print("[yellow]Indexing corpus (one-time)...[/yellow]")
    t0 = time.monotonic()
    retriever = get_retriever()
    elapsed = time.monotonic() - t0
    console.print(
        f"[green]Indexed[/green] {len(retriever.docs)} corpus chunks "
        f"in {elapsed:.1f}s\n"
    )

    _preflight(retriever)

    with open(input_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    console.print(f"[bold]Processing {len(rows)} tickets...[/bold]\n")

    results: list[dict] = []
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
                result = triage(issue, subject, company)
                stats[result.status] += 1
                results.append({
                    "issue":           issue,
                    "subject":         subject,
                    "company":         company,
                    "response":        result.response,
                    "product_area":    result.product_area,
                    "status":          result.status,
                    "request_type":    result.request_type,
                    "retrieval_score": f"{result.retrieval_score:.3f}",
                    "multi_intent":    str(result.multi_intent).lower(),
                    "justification":   result.justification,
                })
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
                    "retrieval_score": "0.000",
                    "multi_intent":    "false",
                    "justification":   f"Processing error — defaulted to escalation: {str(exc)[:120]}",
                })

            progress.advance(task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    table = Table(title="Triage Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric",  style="cyan",  min_width=18)
    table.add_column("Value",   style="green", min_width=10)
    table.add_row("Total tickets", str(len(rows)))
    table.add_row("Replied",       str(stats["replied"]))
    table.add_row("Escalated",     str(stats["escalated"]))
    table.add_row("Errors",        str(stats["errors"]))
    table.add_row("Output file",   str(output_path))
    console.print(table)
    console.print(f"\n[bold green]Done![/bold green]  Output: [cyan]{output_path}[/cyan]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Support triage agent — HackerRank Orchestrate")
    parser.add_argument("--input",  default=str(INPUT_CSV),  help="Input CSV path")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output CSV path")
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
