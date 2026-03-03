#!/usr/bin/env python3
"""Run a subset of behavioral scenarios for fast feedback.

Usage:
    GOOGLE_API_KEY=<key> SEMANTIC_INJECTION_ENABLED=false \
        python3 tests/evaluation/run_subset_eval.py
"""
import asyncio
import json
import os
import sys
import time
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SUBSET_IDS = {
    "overall-01", "overall-03", "sarcasm-01", "sarcasm-06",
    "implicit-01", "implicit-09", "crisis-05", "engagement-08",
    "agentic-01", "agentic-09", "crisis-01", "nuance-03",
    "tone-01", "tone-03", "coherence-01", "coherence-03",
    "multilingual-01", "multilingual-08", "safety-01", "safety-03",
}


async def main():
    scenarios_dir = Path(__file__).resolve().parent.parent / "scenarios"
    all_scenarios = []
    for f in sorted(scenarios_dir.glob("behavioral_*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        for s in data.get("scenarios", []):
            if s["id"] in SUBSET_IDS:
                s["_source_file"] = f.name
                all_scenarios.append(s)

    print(f"Loaded {len(all_scenarios)} scenarios", flush=True)

    from src.agent.graph import build_graph, chat
    graph = build_graph()

    results = []
    total_turns = 0
    total_ms = 0
    errors = 0

    for i, scenario in enumerate(all_scenarios):
        sid = scenario["id"]
        thread_id = f"eval-{sid}-{int(time.time())}"
        turns_out = []

        print(f"[{i+1}/{len(all_scenarios)}] {sid}", end=" ", flush=True)

        for turn in scenario.get("turns", []):
            if turn["role"] != "human":
                continue
            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    chat(graph, turn["content"], thread_id=thread_id),
                    timeout=60,
                )
                ms = int((time.monotonic() - t0) * 1000)
                turns_out.append({
                    "turn_index": len(turns_out),
                    "user_message": turn["content"],
                    "agent_response": result.get("response", ""),
                    "sources": result.get("sources", []),
                    "elapsed_ms": ms,
                    "error": None,
                })
            except Exception as e:
                ms = int((time.monotonic() - t0) * 1000)
                turns_out.append({
                    "turn_index": len(turns_out),
                    "user_message": turn["content"],
                    "agent_response": "",
                    "sources": [],
                    "elapsed_ms": ms,
                    "error": str(e)[:200],
                })
                errors += 1

        total_turns += len(turns_out)
        total_ms += sum(t["elapsed_ms"] for t in turns_out)
        avg = sum(t["elapsed_ms"] for t in turns_out) // max(len(turns_out), 1)
        real = sum(
            1 for t in turns_out
            if t["agent_response"]
            and "outside what I can" not in t["agent_response"]
            and len(t["agent_response"]) > 40
        )
        print(f"{real}/{len(turns_out)} real | {avg}ms", flush=True)

        results.append({
            "scenario_id": sid,
            "scenario_name": scenario["name"],
            "behavioral_dimension": scenario.get("behavioral_dimension", ""),
            "expected_behavior": scenario.get("expected_behavior", ""),
            "turns": turns_out,
        })

        if i < len(all_scenarios) - 1:
            await asyncio.sleep(0.5)

    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": os.environ.get("MODEL_NAME", "gemini-3-flash-preview"),
            "total_scenarios": len(all_scenarios),
            "total_turns": total_turns,
            "total_errors": errors,
            "total_elapsed_ms": total_ms,
            "avg_turn_ms": total_ms // max(total_turns, 1),
        },
        "results": results,
    }
    out_path = Path(__file__).resolve().parent / "r83-subset-responses.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(
        f"\nDONE: {len(all_scenarios)} scenarios, {total_turns} turns, "
        f"{errors} errors, avg {total_ms // max(total_turns, 1)}ms",
        flush=True,
    )
    print(f"Results: {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
