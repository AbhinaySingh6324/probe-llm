import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from probe.core.models import SuiteResult, Verdict

console = Console()


def print_summary(suite: SuiteResult):
    status = "ALL PASSED" if suite.failed == 0 and suite.errors == 0 else "FAILURES DETECTED"
    color = "green" if suite.failed == 0 else "red"
    icon = "\u2705" if suite.failed == 0 else "\u274c"

    header = Text()
    header.append("probe", style="bold cyan")
    header.append(f" - {suite.model_name}\n", style="dim")
    header.append(f"{icon} {status}", style=f"bold {color}")

    console.print(Panel(header, title="[bold]Probe Results[/bold]", border_style=color))
    console.print(
        f"  Total: [bold]{suite.total}[/bold]  "
        f"Passed: [green]{suite.passed}[/green]  "
        f"Failed: [red]{suite.failed}[/red]  "
        f"Errors: [yellow]{suite.errors}[/yellow]  "
        f"Pass Rate: [bold]{suite.pass_rate:.1%}[/bold]  "
        f"Time: [dim]{suite.total_elapsed_ms:.0f}ms[/dim]"
    )
    console.print()

    tbl = Table(show_header=True, header_style="bold", show_lines=False)
    tbl.add_column("", width=3)
    tbl.add_column("Input", max_width=50, no_wrap=True)
    tbl.add_column("Property", style="cyan")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Verdict", justify="center")

    for r in suite.results:
        ic = {"pass": "\u2705", "fail": "\u274c", "error": "\u26a0\ufe0f"}[r.verdict.value]
        sc = "green" if r.score >= 0.8 else "yellow" if r.score >= 0.5 else "red"
        tbl.add_row(
            ic,
            r.input[:50] + ("\u2026" if len(r.input) > 50 else ""),
            r.property_name,
            f"[{sc}]{r.score:.2f}[/{sc}]",
            f"[{sc}]{r.verdict.value.upper()}[/{sc}]",
        )
    console.print(tbl)

    failures = suite.failures()
    if failures:
        console.print(f"\n[bold red]-- Failure Details ({len(failures)}) --[/bold red]\n")
        for r in failures:
            console.print(f"  [red]\u2717[/red] [bold]{r.property_name}[/bold] on: {r.input[:60]}")
            console.print(f"    Score: {r.score:.3f} (threshold: {r.details.get('threshold', '?')})")
            if r.original_output:
                console.print(f"    Original: [dim]{r.original_output[:100]}[/dim]")
            for i, v in enumerate(r.variant_outputs[:3]):
                console.print(f"    Variant {i+1}: [dim]{v[:100]}[/dim]")
            console.print()


def export_json(suite: SuiteResult, path: str):
    with open(path, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)
    console.print(f"[dim]Results exported to {path}[/dim]")
