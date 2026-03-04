#!/usr/bin/env python3
"""Click CLI for eval v2.

Usage:
    # Run subset (default 20 scenarios, resumes automatically):
    GOOGLE_API_KEY=<key> python -m tests.evaluation.v2.cli run

    # Run all scenarios:
    GOOGLE_API_KEY=<key> python -m tests.evaluation.v2.cli run --all

    # Rerun only failed scenarios:
    GOOGLE_API_KEY=<key> python -m tests.evaluation.v2.cli rerun-failed

    # Show summary of results:
    python -m tests.evaluation.v2.cli summary
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so src.* imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import click

from .storage import AtomicStorage

DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "v2-results")


@click.group()
def cli() -> None:
    """Eval v2 — behavioral scenario evaluation with rate limiting and resume."""


@cli.command()
@click.option(
    "--subset/--all",
    "subset_only",
    default=True,
    help="Run subset (20) or all scenarios.",
)
@click.option(
    "--concurrency",
    default=1,
    type=int,
    help="Max parallel scenarios (default 1; higher values multiply API load).",
)
@click.option(
    "--rpm",
    default=15,
    type=int,
    help="Requests per minute for rate limiter (default 15; Gemini free tier ~10 RPM, "
    "each turn needs ~6 LLM calls).",
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    help="Directory for per-scenario JSON results.",
)
@click.option(
    "--turn-timeout",
    default=90.0,
    type=float,
    help="Timeout per turn in seconds (default 90; allows for rate-limit retries).",
)
def run(
    subset_only: bool,
    concurrency: int,
    rpm: int,
    output_dir: str,
    turn_timeout: float,
) -> None:
    """Run eval scenarios with rate limiting and automatic resume."""
    from .runner import run_eval

    asyncio.run(
        run_eval(
            subset_only=subset_only,
            concurrency=concurrency,
            rpm=rpm,
            output_dir=output_dir,
            turn_timeout=turn_timeout,
        )
    )


@cli.command("rerun-failed")
@click.option(
    "--concurrency",
    default=1,
    type=int,
    help="Max parallel scenarios (default 1; higher values multiply API load).",
)
@click.option(
    "--rpm",
    default=15,
    type=int,
    help="Requests per minute for rate limiter.",
)
@click.option(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    help="Directory for per-scenario JSON results.",
)
@click.option(
    "--turn-timeout",
    default=90.0,
    type=float,
    help="Timeout per turn in seconds.",
)
def rerun_failed(
    concurrency: int,
    rpm: int,
    output_dir: str,
    turn_timeout: float,
) -> None:
    """Rerun only scenarios that previously errored."""
    from .runner import run_eval

    asyncio.run(
        run_eval(
            subset_only=True,
            concurrency=concurrency,
            rpm=rpm,
            output_dir=output_dir,
            turn_timeout=turn_timeout,
            rerun_failed=True,
        )
    )


@cli.command()
@click.option(
    "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory with result JSON files."
)
@click.option("--json-output", is_flag=True, help="Output as JSON instead of table.")
def summary(output_dir: str, json_output: bool) -> None:
    """Show summary of eval results from output directory."""
    storage = AtomicStorage(output_dir)
    results = storage.load_all_results()

    if not results:
        click.echo(f"No results found in {output_dir}")
        return

    completed = [r for r in results if r.get("completed", False)]
    failed = [r for r in results if not r.get("completed", True)]
    total_ms = sum(r.get("elapsed_ms", 0) for r in results)
    total_turns = sum(len(r.get("turns", [])) for r in results)
    error_turns = sum(1 for r in results for t in r.get("turns", []) if t.get("error"))
    real_responses = sum(
        1
        for r in results
        for t in r.get("turns", [])
        if t.get("agent_response")
        and "outside what I can" not in t.get("agent_response", "")
        and len(t.get("agent_response", "")) > 40
    )

    summary_data = {
        "total_scenarios": len(results),
        "completed": len(completed),
        "failed": len(failed),
        "total_turns": total_turns,
        "real_responses": real_responses,
        "error_turns": error_turns,
        "response_rate": f"{real_responses}/{total_turns}" if total_turns else "0/0",
        "avg_ms": total_ms // max(len(results), 1),
    }

    if json_output:
        click.echo(json.dumps(summary_data, indent=2))
        return

    click.echo(f"Eval v2 Results ({output_dir})")
    click.echo("=" * 60)
    click.echo(
        f"Scenarios: {len(completed)} completed, {len(failed)} failed, {len(results)} total"
    )
    click.echo(
        f"Turns:     {real_responses}/{total_turns} real responses ({error_turns} errors)"
    )
    click.echo(f"Avg time:  {summary_data['avg_ms']}ms per scenario")

    if failed:
        click.echo(f"\nFailed scenarios:")
        for r in failed:
            err = r.get("error", "unknown")[:80]
            click.echo(f"  - {r['id']}: {err}")


if __name__ == "__main__":
    cli()
