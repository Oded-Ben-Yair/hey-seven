#!/usr/bin/env python3
"""R111 Targeted Eval — 30 scenarios measuring R110's impact on 7 sub-5.0 dims.

Targets:
  H9(2.35) comp strategy, H10(3.87) lifetime value, H6(4.50) rapport depth,
  P6(3.93) incentive framing, P7(4.70) privacy respect, P8(3.62) profile completeness,
  P9(4.30) host handoff

Plus comparison dims: H3, H5, H7, H8, P4, P5, P10, B3, B4, B6

Usage:
    export GOOGLE_API_KEY=<key>
    export SEMANTIC_INJECTION_ENABLED=false
    python3 tests/evaluation/run_r111_eval.py

    # With streaming for real-time judge scoring:
    python3 tests/evaluation/run_r111_eval.py --streaming-dir tests/evaluation/r111-results
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 30 targeted scenarios across 7 sub-5.0 dims + comparison dims
R111_SCENARIO_IDS = {
    # Primary targets (7 sub-5.0 dims, 18 scenarios)
    "h9-01",
    "h9-02",  # H9 comp strategy (2.35)
    "h10-01",
    "h10-02",  # H10 lifetime value (3.87)
    "h6-01",
    "h6-02",
    "h6-03",  # H6 rapport depth (4.50)
    "p6-01",
    "p6-02",
    "p6-03",  # P6 incentive framing (3.93)
    "p7-01",
    "p7-02",  # P7 privacy respect (4.70)
    "p8-01",
    "p8-02",  # P8 profile completeness (3.62)
    "p9-01",
    "p9-02",
    "p9-03",  # P9 host handoff (4.30)
    # Comparison dims (12 scenarios)
    "h3-01",  # H3 solution synthesis
    "h5-01",
    "h5-02",  # H5 trust building
    "h7-01",  # H7 revenue natural
    "h8-01",  # H8 upsell timing
    "p4-01",  # P4 assumptive bridging
    "p5-01",  # P5 progressive sequencing
    "p10-01",
    "p10-02",  # P10 cross-turn memory
    "engagement-01",  # B3 engagement
    "proactive-01",  # B4 proactivity
    "tone-01",  # B6 tone
}

# R105 baselines for delta comparison
R105_BASELINES = {
    "H9": 2.35,
    "H10": 3.87,
    "H6": 4.50,
    "P6": 3.93,
    "P7": 4.70,
    "P8": 3.62,
    "P9": 4.30,
    "H3": 5.17,
    "H5": 5.30,
    "H7": 5.37,
    "H8": 6.10,
    "P4": 5.34,
    "P5": 5.53,
    "P10": 6.50,
    "B3": 6.64,
    "B4": 6.32,
    "B6": 6.24,
}


def load_targeted_scenarios() -> list[dict]:
    """Load only R111 targeted scenarios from YAML files."""
    scenarios_dir = PROJECT_ROOT / "tests" / "scenarios"
    all_scenarios = []

    for yaml_file in sorted(scenarios_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        for scenario in data.get("scenarios", []):
            if scenario["id"] in R111_SCENARIO_IDS:
                scenario["_source_file"] = yaml_file.name
                all_scenarios.append(scenario)

    return all_scenarios


async def run_scenario(graph, scenario: dict, timeout: float = 120.0) -> dict:
    """Run a single multi-turn scenario through the live agent."""
    from src.agent.graph import chat

    scenario_id = scenario["id"]
    turns = scenario.get("turns", [])
    thread_id = f"r111-{scenario_id}-{int(time.time())}"

    turn_results = []
    for i, turn in enumerate(turns):
        if turn["role"] != "human":
            continue

        user_msg = turn["content"]
        start_ts = time.monotonic()

        try:
            result = await asyncio.wait_for(
                chat(graph, user_msg, thread_id=thread_id),
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
        "id": scenario_id,
        "scenario_id": scenario_id,
        "scenario_name": scenario["name"],
        "name": scenario["name"],
        "category": scenario.get("category", ""),
        "behavioral_dimension": scenario.get("behavioral_dimension", ""),
        "expected_behavior": scenario.get("expected_behavior", ""),
        "expected_behavioral_quality": scenario.get("expected_behavioral_quality", ""),
        "safety_relevant": scenario.get("safety_relevant", False),
        "source_file": scenario.get("_source_file", ""),
        "thread_id": thread_id,
        "turns": turn_results,
        "completed": True,
    }


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="R111 Targeted Eval")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--streaming-dir", default="tests/evaluation/r111-results")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    scenarios = load_targeted_scenarios()
    print(f"Loaded {len(scenarios)} targeted scenarios (expected 30)")
    if len(scenarios) != len(R111_SCENARIO_IDS):
        missing = R111_SCENARIO_IDS - {s["id"] for s in scenarios}
        if missing:
            print(f"WARNING: Missing scenarios: {missing}")

    from src.agent.graph import build_graph

    graph = build_graph()

    streaming_dir = Path(args.streaming_dir)
    streaming_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = 0
    total_turns = 0
    total_ms = 0

    for i, scenario in enumerate(scenarios):
        sid = scenario["id"]
        print(
            f"[{i + 1}/{len(scenarios)}] {sid}: {scenario['name'][:50]}",
            end=" ",
            flush=True,
        )

        result = await run_scenario(graph, scenario, timeout=args.timeout)
        results.append(result)

        turn_errors = sum(1 for t in result["turns"] if t["error"])
        errors += turn_errors
        total_turns += len(result["turns"])
        turn_ms = sum(t["elapsed_ms"] for t in result["turns"])
        total_ms += turn_ms
        avg_ms = turn_ms // max(len(result["turns"]), 1)

        status = "OK" if turn_errors == 0 else f"ERR:{turn_errors}"
        print(f"| {status} | {len(result['turns'])} turns | {avg_ms}ms", flush=True)

        # Write per-scenario JSON for streaming judge
        scenario_file = streaming_dir / f"{sid}.json"
        scenario_file.write_text(json.dumps(result, indent=2))

        if i < len(scenarios) - 1:
            await asyncio.sleep(0.5)

    # Write batch results
    output_path = PROJECT_ROOT / "tests" / "evaluation" / "r111-responses.json"
    output_data = {
        "metadata": {
            "round": "R111",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": os.environ.get("MODEL_NAME", "gemini-3-flash-preview"),
            "total_scenarios": len(scenarios),
            "total_turns": total_turns,
            "total_errors": errors,
            "total_elapsed_ms": total_ms,
            "avg_turn_ms": total_ms // max(total_turns, 1),
            "r105_baselines": R105_BASELINES,
            "targeted_dims": ["H9", "H10", "H6", "P6", "P7", "P8", "P9"],
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"R111 EVAL COMPLETE")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Turns: {total_turns}")
    print(f"  Errors: {errors}")
    print(f"  Avg turn latency: {total_ms // max(total_turns, 1)}ms")
    print(f"  Results: {output_path}")
    print(f"  Streaming dir: {streaming_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
