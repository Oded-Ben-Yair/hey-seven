#!/usr/bin/env python3
"""Live Agent Evaluation — Run scenarios through real Gemini Flash.

This script:
1. Loads YAML scenarios from tests/scenarios/ (configurable glob pattern)
2. Runs each scenario through the full LangGraph pipeline (build_graph + chat)
3. Records all agent responses to tests/evaluation/{round}-responses.json
4. Outputs a summary of results

Usage:
    # All 195 scenarios (behavioral + profiling)
    GOOGLE_API_KEY=<key> python tests/evaluation/run_live_eval.py --round r76-baseline

    # Behavioral only (74 scenarios)
    GOOGLE_API_KEY=<key> python tests/evaluation/run_live_eval.py --pattern "behavioral_*.yaml" --round r76-behavioral

    # Profiling only (56 scenarios)
    GOOGLE_API_KEY=<key> python tests/evaluation/run_live_eval.py --pattern "profiling_*.yaml" --round r76-profiling

Requirements:
    - Real GOOGLE_API_KEY (Gemini Flash)
    - No mocks — this exercises the full production pipeline
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("live_eval")


def load_scenarios(pattern: str = "*.yaml") -> list[dict]:
    """Load scenarios from YAML files matching the given glob pattern.

    Args:
        pattern: Glob pattern for scenario files. Examples:
            "*.yaml" — all scenarios (behavioral + profiling)
            "behavioral_*.yaml" — behavioral only
            "profiling_*.yaml" — profiling only

    Returns:
        List of scenario dicts with _source_file metadata.
    """
    scenarios_dir = PROJECT_ROOT / "tests" / "scenarios"
    all_scenarios = []

    for yaml_file in sorted(scenarios_dir.glob(pattern)):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        for scenario in data.get("scenarios", []):
            scenario["_source_file"] = yaml_file.name
            all_scenarios.append(scenario)

    return all_scenarios


async def run_scenario(graph, scenario: dict, timeout: float = 60.0) -> dict:
    """Run a single multi-turn scenario through the live agent.

    Returns a dict with scenario metadata + per-turn agent responses.
    """
    scenario_id = scenario["id"]
    turns = scenario.get("turns", [])
    thread_id = f"eval-{scenario_id}-{int(time.time())}"

    turn_results = []
    for i, turn in enumerate(turns):
        if turn["role"] != "human":
            continue

        user_msg = turn["content"]
        start_ts = time.monotonic()

        try:
            result = await asyncio.wait_for(
                graph_chat(graph, user_msg, thread_id),
                timeout=timeout,
            )
            elapsed_ms = int((time.monotonic() - start_ts) * 1000)

            turn_results.append(
                {
                    "turn_index": i,
                    "user_message": user_msg,
                    "agent_response": result.get("response", ""),
                    "sources": result.get("sources", []),
                    "elapsed_ms": elapsed_ms,
                    "error": None,
                }
            )
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_ts) * 1000)
            turn_results.append(
                {
                    "turn_index": i,
                    "user_message": user_msg,
                    "agent_response": "",
                    "sources": [],
                    "elapsed_ms": elapsed_ms,
                    "error": f"TIMEOUT ({timeout}s)",
                }
            )
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_ts) * 1000)
            turn_results.append(
                {
                    "turn_index": i,
                    "user_message": user_msg,
                    "agent_response": "",
                    "sources": [],
                    "elapsed_ms": elapsed_ms,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    return {
        "scenario_id": scenario_id,
        "scenario_name": scenario["name"],
        "category": scenario.get("category", ""),
        "behavioral_dimension": scenario.get("behavioral_dimension", ""),
        "expected_behavior": scenario.get("expected_behavior", ""),
        "expected_behavioral_quality": scenario.get("expected_behavioral_quality", ""),
        "safety_relevant": scenario.get("safety_relevant", False),
        "source_file": scenario.get("_source_file", ""),
        "thread_id": thread_id,
        "turns": turn_results,
    }


async def graph_chat(graph, message: str, thread_id: str) -> dict:
    """Wrapper for graph.chat that imports at call time."""
    from src.agent.graph import chat

    return await chat(graph, message, thread_id=thread_id)


async def main():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set. Cannot run live evaluation.")
        sys.exit(1)

    # Accept optional args: --pattern <glob> --round <name>
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="Scenario file glob pattern (default: *.yaml = all)",
    )
    parser.add_argument(
        "--round",
        default="latest",
        help="Round name for output file (e.g., r76, baseline)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-turn timeout in seconds (default: 60, use 120 for Pro)",
    )
    parser.add_argument(
        "--streaming-dir",
        default=None,
        help="Write per-scenario JSON files here for streaming judge (enables real-time scoring)",
    )
    args = parser.parse_args()

    logger.info("Loading scenarios (pattern=%s)...", args.pattern)
    scenarios = load_scenarios(pattern=args.pattern)
    logger.info(
        "Loaded %d scenarios from %d files",
        len(scenarios),
        len(set(s["_source_file"] for s in scenarios)),
    )

    model_name = os.environ.get("MODEL_NAME", "gemini-3-flash-preview")
    logger.info("Building graph with %s (timeout=%.0fs)...", model_name, args.timeout)
    from src.agent.graph import build_graph

    graph = build_graph()

    # Streaming dir for real-time judge scoring
    streaming_dir = None
    if args.streaming_dir:
        streaming_dir = Path(args.streaming_dir)
        streaming_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Streaming per-scenario results to %s", streaming_dir)

    results = []
    errors = 0
    total_turns = 0
    total_elapsed_ms = 0

    for i, scenario in enumerate(scenarios):
        logger.info(
            "[%d/%d] Running scenario %s: %s",
            i + 1,
            len(scenarios),
            scenario["id"],
            scenario["name"],
        )

        result = await run_scenario(graph, scenario, timeout=args.timeout)
        results.append(result)

        turn_errors = sum(1 for t in result["turns"] if t["error"])
        errors += turn_errors
        total_turns += len(result["turns"])
        total_elapsed_ms += sum(t["elapsed_ms"] for t in result["turns"])

        # Brief status per scenario
        status = "OK" if turn_errors == 0 else f"ERRORS: {turn_errors}"
        avg_ms = sum(t["elapsed_ms"] for t in result["turns"]) // max(
            len(result["turns"]), 1
        )
        logger.info(
            "  -> %s | %d turns | avg %dms", status, len(result["turns"]), avg_ms
        )

        # Write per-scenario file for streaming judge
        if streaming_dir:
            scenario_file = streaming_dir / f"{result['scenario_id']}.json"
            scenario_data = {**result, "completed": True}
            scenario_file.write_text(json.dumps(scenario_data, indent=2))

        # Rate limiting: small delay between scenarios to avoid API throttling
        if i < len(scenarios) - 1:
            await asyncio.sleep(0.5)

    # Write results
    output_path = PROJECT_ROOT / "tests" / "evaluation" / f"{args.round}-responses.json"
    output_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": os.environ.get("MODEL_NAME", "gemini-3-flash-preview"),
            "total_scenarios": len(scenarios),
            "total_turns": total_turns,
            "total_errors": errors,
            "total_elapsed_ms": total_elapsed_ms,
            "avg_turn_ms": total_elapsed_ms // max(total_turns, 1),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("  Scenarios: %d", len(scenarios))
    logger.info("  Turns: %d", total_turns)
    logger.info("  Errors: %d", errors)
    logger.info("  Avg turn latency: %dms", total_elapsed_ms // max(total_turns, 1))
    logger.info("  Results: %s", output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
