"""Async eval runner with rate limiting, retries, and resume support."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so src.* imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from .rate_limiter import TokenBucketRateLimiter
from .schemas import ScenarioResult, TurnResult, load_scenarios
from .storage import AtomicStorage

logger = logging.getLogger(__name__)

# Exponential backoff schedule (seconds) for retries on transient errors.
_BACKOFF_DELAYS = [2, 4, 8]


async def _run_scenario(
    graph: Any,
    chat_fn: Any,
    scenario: dict[str, Any],
    rate_limiter: TokenBucketRateLimiter,
    turn_timeout: float = 60.0,
) -> ScenarioResult:
    """Run a single scenario through the agent, with retries per turn."""
    sid = scenario["id"]
    thread_id = f"eval-v2-{sid}-{int(time.time())}"
    turns: list[TurnResult] = []
    scenario_t0 = time.monotonic()
    scenario_error: str | None = None

    for turn in scenario.get("turns", []):
        if turn["role"] != "human":
            continue

        turn_result = await _run_turn_with_retries(
            graph=graph,
            chat_fn=chat_fn,
            message=turn["content"],
            thread_id=thread_id,
            turn_index=len(turns),
            rate_limiter=rate_limiter,
            turn_timeout=turn_timeout,
        )
        turns.append(turn_result)
        if turn_result.error:
            scenario_error = turn_result.error

    elapsed_ms = int((time.monotonic() - scenario_t0) * 1000)
    completed = scenario_error is None

    return ScenarioResult(
        id=sid,
        name=scenario.get("name", ""),
        source_file=scenario.get("_source_file", ""),
        category=scenario.get("category", ""),
        behavioral_dimension=scenario.get("behavioral_dimension", ""),
        expected_behavior=scenario.get("expected_behavior", ""),
        expected_behavioral_quality=scenario.get("expected_behavioral_quality", ""),
        safety_relevant=scenario.get("safety_relevant", False),
        turns=turns,
        elapsed_ms=elapsed_ms,
        error=scenario_error,
        completed=completed,
    )


async def _run_turn_with_retries(
    graph: Any,
    chat_fn: Any,
    message: str,
    thread_id: str,
    turn_index: int,
    rate_limiter: TokenBucketRateLimiter,
    turn_timeout: float,
) -> TurnResult:
    """Execute a single turn with exponential backoff retries."""
    last_error: str | None = None

    for attempt in range(len(_BACKOFF_DELAYS) + 1):
        await rate_limiter.acquire()
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                chat_fn(graph, message, thread_id=thread_id),
                timeout=turn_timeout,
            )
            ms = int((time.monotonic() - t0) * 1000)
            return TurnResult(
                turn_index=turn_index,
                user_message=message,
                agent_response=result.get("response", ""),
                sources=result.get("sources", []),
                elapsed_ms=ms,
                error=None,
            )
        except Exception as e:
            ms = int((time.monotonic() - t0) * 1000)
            last_error = str(e)[:200]
            if attempt < len(_BACKOFF_DELAYS):
                delay = _BACKOFF_DELAYS[attempt]
                logger.warning(
                    "Turn %d attempt %d failed (%s), retrying in %ds",
                    turn_index,
                    attempt + 1,
                    last_error[:80],
                    delay,
                )
                await asyncio.sleep(delay)

    # All retries exhausted
    return TurnResult(
        turn_index=turn_index,
        user_message=message,
        agent_response="",
        sources=[],
        elapsed_ms=int((time.monotonic() - t0) * 1000),
        error=last_error,
    )


async def run_eval(
    subset_only: bool = True,
    concurrency: int = 4,
    rpm: int = 50,
    output_dir: str | Path | None = None,
    turn_timeout: float = 60.0,
    rerun_failed: bool = False,
) -> dict[str, Any]:
    """Run the eval suite with rate limiting and resume support.

    Args:
        subset_only: Only run the 20-scenario subset.
        concurrency: Max parallel scenario executions.
        rpm: Requests per minute for rate limiter.
        output_dir: Directory for per-scenario JSON results.
        turn_timeout: Timeout per turn in seconds.
        rerun_failed: If True, only rerun scenarios that previously errored.

    Returns:
        Summary dict with total, completed, failed, avg_ms.
    """
    # Set eval env vars
    os.environ.setdefault("SEMANTIC_INJECTION_ENABLED", "false")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "v2-results"
    storage = AtomicStorage(output_dir)

    # Load scenarios
    all_scenarios = load_scenarios(subset_only=subset_only)
    all_ids = {s["id"] for s in all_scenarios}

    if rerun_failed:
        # Only rerun scenarios that previously errored
        completed_ids = storage.get_completed_ids()
        failed_ids: set[str] = set()
        for sid in completed_ids:
            result_data = storage.load_result(sid)
            if result_data and not result_data.get("completed", True):
                failed_ids.add(sid)
        scenarios = [s for s in all_scenarios if s["id"] in failed_ids]
        print(f"Rerunning {len(scenarios)} failed scenarios", flush=True)
    else:
        # Resume: skip already completed scenarios
        pending_ids = storage.get_pending_ids(all_ids)
        scenarios = [s for s in all_scenarios if s["id"] in pending_ids]
        skipped = len(all_scenarios) - len(scenarios)
        if skipped:
            print(
                f"Resuming: {skipped} already completed, {len(scenarios)} pending",
                flush=True,
            )
        else:
            print(f"Running {len(scenarios)} scenarios", flush=True)

    if not scenarios:
        print("Nothing to run.", flush=True)
        return _build_summary(storage, all_ids)

    # Build graph once
    from src.agent.graph import build_graph, chat

    print("Building graph...", flush=True)
    graph = build_graph()

    rate_limiter = TokenBucketRateLimiter(rpm=rpm, burst=min(concurrency, 5))
    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0
    failed_count = 0
    total = len(scenarios)

    async def _run_one(idx: int, scenario: dict[str, Any]) -> None:
        nonlocal completed_count, failed_count
        async with semaphore:
            sid = scenario["id"]
            result = await _run_scenario(
                graph=graph,
                chat_fn=chat,
                scenario=scenario,
                rate_limiter=rate_limiter,
                turn_timeout=turn_timeout,
            )
            storage.save_result(sid, result)

            if result.completed:
                completed_count += 1
                status = "OK"
            else:
                failed_count += 1
                status = f"FAIL ({result.error[:60]})" if result.error else "FAIL"

            print(
                f"[{completed_count + failed_count}/{total}] {sid} ... {status} ({result.elapsed_ms}ms)",
                flush=True,
            )

    # Run scenarios concurrently (bounded by semaphore)
    tasks = [asyncio.create_task(_run_one(i, s)) for i, s in enumerate(scenarios)]
    await asyncio.gather(*tasks, return_exceptions=True)

    summary = _build_summary(storage, all_ids)
    print(
        f"\nDONE: {summary['total']} total, {summary['completed']} completed, "
        f"{summary['failed']} failed, avg {summary['avg_ms']}ms",
        flush=True,
    )
    print(f"Results: {storage.output_dir}", flush=True)
    return summary


def _build_summary(storage: AtomicStorage, all_ids: set[str]) -> dict[str, Any]:
    """Build a summary from all saved results."""
    results = storage.load_all_results()
    completed = sum(1 for r in results if r.get("completed", False))
    failed = sum(1 for r in results if not r.get("completed", True))
    total_ms = sum(r.get("elapsed_ms", 0) for r in results)
    count = len(results) or 1

    return {
        "total": len(all_ids),
        "completed": completed,
        "failed": failed,
        "pending": len(all_ids) - len(results),
        "avg_ms": total_ms // count,
        "output_dir": str(storage.output_dir),
    }
