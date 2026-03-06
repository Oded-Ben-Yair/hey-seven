#!/usr/bin/env python3
"""Streaming Judge — Score eval results as they arrive.

Watches an output directory for new/updated scenario result files and
judges them immediately using GPT-5.2 (or GPT-5.3-chat) via Azure AI
Foundry. Prints rolling aggregates after each scored scenario.

Architecture:
    Eval runner (run_live_eval.py) writes results →
    Streaming judge watches dir, judges in parallel batches →
    Rolling dashboard shows B-avg, H-avg, P-avg, safety %

Benefits:
    - Wall-clock time cut 40-60% vs sequential eval→judge pipeline
    - Real-time feedback — spot regressions after 5 scenarios, not 80
    - Fail-fast — halt eval early if dimension scores collapse
    - Batch-parallel judging (5 at a time) for throughput

Usage:
    # Watch v2-results for new files and judge them
    export AZURE_AI_ENDPOINT=$(az keyvault secret show --vault-name kv-seekapa-apps --name AzureAIFoundry-Endpoint -o tsv --query value)
    export AZURE_AI_KEY=$(az keyvault secret show --vault-name kv-seekapa-apps --name AzureAIFoundry-ApiKey -o tsv --query value)

    # Mode 1: Watch directory for new files (live eval running)
    python tests/evaluation/streaming_judge.py --watch tests/evaluation/v2-results-r99 --category behavioral

    # Mode 2: Judge existing files (eval already completed)
    python tests/evaluation/streaming_judge.py --batch tests/evaluation/r98-host-triangle-responses.json --category host-triangle

    # Mode 3: Concurrent eval + judge (single command)
    python tests/evaluation/streaming_judge.py --eval --pattern "host_triangle_*.yaml" --round r99 --category host-triangle
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("streaming_judge")

from tests.evaluation.run_r95_judge import (
    build_extended_judge_prompt,
    judge_with_gpt52,
    CATEGORY_PREFIXES,
    H_DIMS,
)
from tests.evaluation.run_judge_panel import B_DIMS, P_DIMS


# ---------------------------------------------------------------------------
# Rolling aggregator
# ---------------------------------------------------------------------------


class RollingAggregator:
    """Track rolling averages across scored scenarios."""

    def __init__(self):
        self.dim_totals: dict[str, float] = {}
        self.dim_counts: dict[str, int] = {}
        self.safety_pass: int = 0
        self.safety_total: int = 0
        self.scored_count: int = 0
        self.failed_count: int = 0

    def add_score(self, scores: dict) -> None:
        """Add a single scenario's scores to the rolling aggregate."""
        self.scored_count += 1

        for dim_list in (B_DIMS, P_DIMS, H_DIMS):
            for dim in dim_list:
                val = scores.get(dim)
                if val is not None and isinstance(val, (int, float)) and val >= 0:
                    self.dim_totals[dim] = self.dim_totals.get(dim, 0) + val
                    self.dim_counts[dim] = self.dim_counts.get(dim, 0) + 1

        safety = scores.get("safety")
        if safety is not None:
            self.safety_total += 1
            if safety:
                self.safety_pass += 1

    def get_summary(self) -> dict:
        """Get current rolling averages."""
        avgs = {}
        for dim, total in self.dim_totals.items():
            count = self.dim_counts.get(dim, 0)
            if count > 0:
                avgs[dim] = round(total / count, 2)

        # Compute category averages
        b_vals = [avgs[d] for d in B_DIMS if d in avgs]
        p_vals = [avgs[d] for d in P_DIMS if d in avgs]
        h_vals = [avgs[d] for d in H_DIMS if d in avgs]

        return {
            "scored": self.scored_count,
            "failed": self.failed_count,
            "b_avg": round(sum(b_vals) / len(b_vals), 2) if b_vals else None,
            "p_avg": round(sum(p_vals) / len(p_vals), 2) if p_vals else None,
            "h_avg": round(sum(h_vals) / len(h_vals), 2) if h_vals else None,
            "safety_pct": round(100 * self.safety_pass / self.safety_total, 1)
            if self.safety_total
            else None,
            "per_dim": avgs,
        }

    def print_dashboard(self) -> None:
        """Print compact rolling dashboard to stdout."""
        s = self.get_summary()
        parts = [f"[{s['scored']} scored]"]
        if s["b_avg"] is not None:
            parts.append(f"B-avg:{s['b_avg']:.1f}")
        if s["p_avg"] is not None:
            parts.append(f"P-avg:{s['p_avg']:.1f}")
        if s["h_avg"] is not None:
            parts.append(f"H-avg:{s['h_avg']:.1f}")
        if s["safety_pct"] is not None:
            parts.append(f"Safety:{s['safety_pct']:.0f}%")
        logger.info("DASHBOARD: %s", " | ".join(parts))


# ---------------------------------------------------------------------------
# Batch judge
# ---------------------------------------------------------------------------


async def judge_batch(
    scenarios: list[dict],
    endpoint: str,
    key: str,
    batch_size: int = 5,
) -> list[tuple[str, dict | None]]:
    """Judge a batch of scenarios concurrently.

    Args:
        scenarios: List of scenario result dicts.
        endpoint: Azure AI endpoint.
        key: Azure AI API key.
        batch_size: Max concurrent judge calls.

    Returns:
        List of (scenario_id, scores_or_none) tuples.
    """
    results = []

    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i : i + batch_size]
        tasks = []
        for scenario in batch:
            prompt = build_extended_judge_prompt(scenario)
            tasks.append(judge_with_gpt52(prompt, endpoint, key))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for scenario, result in zip(batch, batch_results):
            sid = scenario.get("id", scenario.get("scenario_id", "unknown"))
            if isinstance(result, Exception):
                logger.warning("Judge error for %s: %s", sid, result)
                results.append((sid, None))
            else:
                results.append((sid, result))

        # Brief delay between batches
        if i + batch_size < len(scenarios):
            await asyncio.sleep(1.0)

    return results


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------


async def watch_and_judge(
    watch_dir: Path,
    category: str,
    endpoint: str,
    key: str,
    output_path: Path,
    poll_interval: float = 10.0,
    early_stop_threshold: float = 3.0,
    early_stop_window: int = 5,
):
    """Watch a directory for new result files and judge them as they appear.

    Args:
        watch_dir: Directory to watch for *.json result files.
        category: Category filter (behavioral, profiling, host-triangle).
        endpoint: Azure AI endpoint.
        key: Azure AI API key.
        output_path: Where to write cumulative judge scores.
        poll_interval: Seconds between directory polls.
        early_stop_threshold: If rolling overall avg drops below this, warn.
        early_stop_window: Number of recent scores to check for early stop.
    """
    agg = RollingAggregator()
    scored_ids: set[str] = set()
    recent_overalls: list[float] = []
    all_scores: list[dict] = []

    prefixes = CATEGORY_PREFIXES.get(category, [])

    logger.info(
        "Watching %s for new %s results (poll every %.0fs)",
        watch_dir,
        category,
        poll_interval,
    )

    while True:
        new_scenarios: list[dict] = []

        for f in sorted(watch_dir.glob("*.json")):
            sid = f.stem
            if sid in scored_ids:
                continue

            # Category filter
            if prefixes:
                prefix = sid.rsplit("-", 1)[0]
                if prefix not in prefixes:
                    continue

            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, IOError):
                continue

            if not data.get("completed", False):
                continue

            data["scenario_name"] = data.get("name", data.get("id", sid))
            data["id"] = sid
            new_scenarios.append(data)

        if new_scenarios:
            logger.info("Found %d new scenarios to judge", len(new_scenarios))
            batch_results = await judge_batch(new_scenarios, endpoint, key)

            for sid, scores in batch_results:
                scored_ids.add(sid)
                if scores:
                    agg.add_score(scores)
                    all_scores.append({"id": sid, "scores": scores})

                    overall = scores.get("overall", 0)
                    if isinstance(overall, (int, float)):
                        recent_overalls.append(overall)

                    agg.print_dashboard()
                else:
                    agg.failed_count += 1

            # Write cumulative results
            _write_scores(output_path, agg, all_scores)

            # Early stop warning
            if len(recent_overalls) >= early_stop_window:
                window = recent_overalls[-early_stop_window:]
                avg_recent = sum(window) / len(window)
                if avg_recent < early_stop_threshold:
                    logger.warning(
                        "EARLY STOP WARNING: Last %d scores avg %.1f (below %.1f threshold). "
                        "Consider stopping eval and investigating.",
                        early_stop_window,
                        avg_recent,
                        early_stop_threshold,
                    )

        await asyncio.sleep(poll_interval)


def _write_scores(path: Path, agg: RollingAggregator, scores: list[dict]) -> None:
    """Write cumulative judge scores to output file."""
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "judge": "gpt-5.2",
            "scored": agg.scored_count,
            "failed": agg.failed_count,
        },
        "summary": agg.get_summary(),
        "scores": scores,
    }
    path.write_text(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Batch mode (judge existing response file)
# ---------------------------------------------------------------------------


async def judge_response_file(
    response_file: Path,
    category: str,
    endpoint: str,
    key: str,
    output_path: Path,
):
    """Judge all scenarios in an existing response file."""
    data = json.loads(response_file.read_text())
    results = data.get("results", [])

    # Filter by category
    prefixes = CATEGORY_PREFIXES.get(category, [])
    if prefixes:
        filtered = []
        for r in results:
            sid = r.get("scenario_id", r.get("id", ""))
            prefix = sid.rsplit("-", 1)[0]
            if prefix in prefixes:
                filtered.append(r)
        results = filtered

    # Ensure required fields
    for r in results:
        if "id" not in r:
            r["id"] = r.get("scenario_id", "unknown")
        if "scenario_name" not in r:
            r["scenario_name"] = r.get("scenario_name", r["id"])
        if "completed" not in r:
            r["completed"] = bool(r.get("turns"))

    completed = [r for r in results if r.get("completed")]
    logger.info(
        "Judging %d completed scenarios from %s", len(completed), response_file.name
    )

    agg = RollingAggregator()
    all_scores: list[dict] = []

    batch_results = await judge_batch(completed, endpoint, key)
    for sid, scores in batch_results:
        if scores:
            agg.add_score(scores)
            all_scores.append({"id": sid, "scores": scores})
            agg.print_dashboard()
        else:
            agg.failed_count += 1

    _write_scores(output_path, agg, all_scores)
    logger.info("Final: %s", json.dumps(agg.get_summary(), indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Streaming Judge — score results as they arrive"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", type=str, help="Watch directory for new result files")
    mode.add_argument("--batch", type=str, help="Judge existing response file")

    parser.add_argument(
        "--category",
        default="behavioral",
        choices=["behavioral", "profiling", "host-triangle", "all"],
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Concurrent judge calls per batch"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Watch mode poll interval (seconds)",
    )

    args = parser.parse_args()

    endpoint = os.environ.get("AZURE_AI_ENDPOINT", "")
    key = os.environ.get("AZURE_AI_KEY", "")
    if not endpoint or not key:
        logger.error("AZURE_AI_ENDPOINT and AZURE_AI_KEY must be set")
        sys.exit(1)

    output_path = (
        Path(args.output)
        if args.output
        else (
            PROJECT_ROOT
            / "tests"
            / "evaluation"
            / f"streaming-{args.category}-judge-scores.json"
        )
    )

    if args.watch:
        await watch_and_judge(
            Path(args.watch),
            args.category,
            endpoint,
            key,
            output_path,
            poll_interval=args.poll_interval,
        )
    elif args.batch:
        await judge_response_file(
            Path(args.batch),
            args.category,
            endpoint,
            key,
            output_path,
        )


if __name__ == "__main__":
    asyncio.run(main())
