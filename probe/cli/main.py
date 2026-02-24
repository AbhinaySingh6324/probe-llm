import asyncio
import json
import sys
import click
from rich.console import Console

console = Console()


def _load_inputs(inputs, input_text):
    if input_text:
        return [input_text]
    if inputs:
        items = []
        with open(inputs, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                    items.append(obj if isinstance(obj, str) else obj.get("input", obj.get("text", str(obj))))
                except json.JSONDecodeError:
                    items.append(line)
        return items
    console.print("[red]Error: Provide --inputs <file> or --input <text>[/red]")
    sys.exit(1)


@click.group()
@click.version_option(version="0.1.0", prog_name="probe")
def cli():
    """Probe - Behavioral testing framework for LLMs."""
    pass


@cli.command()
@click.option("--model", "-m", required=True, help="e.g. openai:gpt-4o-mini")
@click.option("--inputs", "-i", default=None, help="Path to inputs file (.jsonl/.txt)")
@click.option("--input", "input_text", default=None, help="Single input text")
@click.option("--properties", "-p", default="consistency", help="Comma-separated properties")
@click.option("--threshold", "-t", default=0.8, type=float, help="Pass threshold (0-1)")
@click.option("--concurrency", "-c", default=5, type=int, help="Max concurrent API calls")
@click.option("--output", "-o", default=None, help="Export results to JSON file")
def run(model, inputs, input_text, properties, threshold, concurrency, output):
    """Run behavioral property tests on an LLM."""
    from probe.providers import get_provider
    from probe.properties import get_property
    from probe.core.runner import run_suite_sync
    from probe.core.reporter import print_summary, export_json

    console.print(f"\n[bold cyan]Probe[/bold cyan] [dim]v0.1.0[/dim]")
    console.print(f"  Model: [bold]{model}[/bold]")

    input_list = _load_inputs(inputs, input_text)
    console.print(f"  Inputs: [bold]{len(input_list)}[/bold] test cases")

    prop_names = [p.strip() for p in properties.split(",")]
    props = [get_property(name, threshold=threshold) for name in prop_names]
    console.print(f"  Properties: [bold]{', '.join(prop_names)}[/bold]")
    console.print(f"  Threshold: [bold]{threshold}[/bold]\n")

    provider = get_provider(model)

    with console.status("[bold green]Running behavioral tests..."):
        suite = run_suite_sync(provider, input_list, props, concurrency=concurrency)

    print_summary(suite)
    if output:
        export_json(suite, output)
    if suite.failed > 0 or suite.errors > 0:
        sys.exit(1)


@cli.command()
@click.option("--model-a", required=True)
@click.option("--model-b", required=True)
@click.option("--inputs", "-i", required=True)
@click.option("--properties", "-p", default="consistency,invariance,robustness")
@click.option("--threshold", "-t", default=0.8, type=float)
def compare(model_a, model_b, inputs, properties, threshold):
    """Compare behavioral properties of two models side-by-side."""
    from probe.providers import get_provider
    from probe.properties import get_property
    from probe.core.runner import run_suite_sync
    from rich.table import Table

    console.print(f"\n[bold cyan]Probe Compare[/bold cyan]")
    console.print(f"  Model A: [bold]{model_a}[/bold]")
    console.print(f"  Model B: [bold]{model_b}[/bold]\n")

    input_list = _load_inputs(inputs, None)
    prop_names = [p.strip() for p in properties.split(",")]

    with console.status("[bold green]Testing Model A..."):
        sa = run_suite_sync(get_provider(model_a), input_list,
                            [get_property(n, threshold=threshold) for n in prop_names])
    with console.status("[bold green]Testing Model B..."):
        sb = run_suite_sync(get_provider(model_b), input_list,
                            [get_property(n, threshold=threshold) for n in prop_names])

    tbl = Table(title="Model Comparison", show_header=True, header_style="bold")
    tbl.add_column("Property", style="cyan")
    tbl.add_column(model_a, justify="center")
    tbl.add_column(model_b, justify="center")
    tbl.add_column("Winner", justify="center")

    for pn in prop_names:
        aa = [r.score for r in sa.results if r.property_name == pn]
        bb = [r.score for r in sb.results if r.property_name == pn]
        ma = sum(aa) / len(aa) if aa else 0
        mb = sum(bb) / len(bb) if bb else 0
        ca = "green" if ma >= threshold else "red"
        cb = "green" if mb >= threshold else "red"
        w = "[bold green]<- A[/bold green]" if ma > mb + 0.02 else "[bold green]B ->[/bold green]" if mb > ma + 0.02 else "[dim]Tie[/dim]"
        tbl.add_row(pn, f"[{ca}]{ma:.3f}[/{ca}]", f"[{cb}]{mb:.3f}[/{cb}]", w)

    tbl.add_section()
    tbl.add_row("[bold]Overall[/bold]", f"[bold]{sa.pass_rate:.1%}[/bold]",
                f"[bold]{sb.pass_rate:.1%}[/bold]",
                "[bold green]<- A[/bold green]" if sa.pass_rate > sb.pass_rate else "[bold green]B ->[/bold green]" if sb.pass_rate > sa.pass_rate else "[dim]Tie[/dim]")
    console.print(tbl)


@cli.command("list-properties")
def list_properties():
    """List all available behavioral properties."""
    from probe.properties import PROPERTY_REGISTRY
    console.print("\n[bold cyan]Available Properties:[/bold cyan]\n")
    for name, cls in PROPERTY_REGISTRY.items():
        doc = (cls.__doc__ or "No description").strip().split("\n")[0]
        console.print(f"  [bold]{name}[/bold] - {doc}")
    console.print()


@cli.command("init")
@click.option("--output", "-o", default="probe_tests.txt", help="Output filename")
def init(output):
    """Generate a starter test cases file."""
    import os
    if os.path.exists(output):
        console.print(f"[yellow]File '{output}' already exists. Overwrite? [y/N][/yellow]", end=" ")
        if input().strip().lower() != "y":
            console.print("[dim]Aborted.[/dim]")
            return

    samples = """# probe test cases — one input per line
# Lines starting with # are ignored
# Add your own prompts below!

# === Factual Questions ===
What is the capital of France?
Is 17 a prime number?
What is the boiling point of water in Celsius?
Who wrote Romeo and Juliet?
What is the largest planet in our solar system?

# === Reasoning ===
What is the time complexity of binary search?
What is the derivative of x squared?
If all roses are flowers and all flowers are plants, are all roses plants?

# === Instruction Following ===
Explain photosynthesis in 2 sentences.
Translate "hello" to Spanish.
Write a Python function to reverse a string.
Summarize what an API is in one sentence.

# === Domain-Specific (customize these!) ===
# What is a Kubernetes pod?
# Explain the difference between SQL and NoSQL.
# What does ACID stand for in databases?
"""
    with open(output, "w") as f:
        f.write(samples)

    console.print(f"\n[bold green]✅ Created '{output}'[/bold green]")
    console.print(f"   Edit it to add your own test cases, then run:\n")
    console.print(f"   [bold]probe run --model ollama:llama3.2 --inputs {output} -p consistency[/bold]\n")


@cli.command("list-inputs")
@click.option("--inputs", "-i", required=True, help="Path to inputs file")
def list_inputs(inputs):
    """List all test cases from an inputs file."""
    from rich.table import Table

    items = _load_inputs(inputs, None)
    console.print(f"\n[bold cyan]Test Cases[/bold cyan] [dim]({inputs})[/dim]\n")

    tbl = Table(show_header=True, header_style="bold", show_lines=False)
    tbl.add_column("#", style="dim", width=4, justify="right")
    tbl.add_column("Input", style="white")
    tbl.add_column("Length", style="dim", justify="right")

    for i, item in enumerate(items, 1):
        tbl.add_row(str(i), item, str(len(item)))

    console.print(tbl)
    console.print(f"\n  [bold]{len(items)}[/bold] total test cases\n")


if __name__ == "__main__":
    cli()