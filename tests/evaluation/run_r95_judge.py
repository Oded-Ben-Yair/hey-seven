#!/usr/bin/env python3
"""R95 Judge — Score v2-results with GPT-5.2 via Azure AI Foundry.

Reads individual scenario JSON files from v2-results/, builds judge prompts,
calls GPT-5.2 for each, and writes aggregated scores.

Usage:
    export AZURE_AI_ENDPOINT="https://brn-azai.cognitiveservices.azure.com/"
    export AZURE_AI_KEY="<key>"
    python tests/evaluation/run_r95_judge.py --results-dir tests/evaluation/v2-results
    python tests/evaluation/run_r95_judge.py --results-dir tests/evaluation/v2-results --category profiling
    python tests/evaluation/run_r95_judge.py --results-dir tests/evaluation/v2-results --category host-triangle
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
logger = logging.getLogger("r95_judge")

# Import judge utilities from existing module
from tests.evaluation.run_judge_panel import (
    B_DIMS,
    P_DIMS,
    ALL_DIMS,
    build_judge_prompt,
    parse_judge_response,
)

# Host Triangle dimensions (H1-H10) — extend the rubric
H_DIMS = [
    "H1_property_knowledge",
    "H2_need_anticipation",
    "H3_solution_synthesis",
    "H4_emotional_attunement",
    "H5_trust_building",
    "H6_rapport_depth",
    "H7_revenue_natural",
    "H8_upsell_timing",
    "H9_comp_strategy",
    "H10_lifetime_value",
]

# Category prefix mapping for filtering results
CATEGORY_PREFIXES = {
    "behavioral": [
        "agentic",
        "crisis",
        "engagement",
        "implicit",
        "extract",
        "multilingual",
        "nuance",
        "overall",
        "proactive",
        "coherence",
        "sarcasm",
        "tone",
        "safety",
        "slang",
    ],
    "profiling": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"],
    "host-triangle": ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"],
}

# Extended judge rubric with H-dimensions
H_RUBRIC_EXTENSION = """
### Host Triangle Dimensions (H1-H10) — Only score if this is a host-triangle scenario

**H1: Property Knowledge Depth (0-10)**
Does the agent demonstrate deep property-specific knowledge? Can it name specific venues, hours, locations, and features without generic filler?

**H2: Need Anticipation (0-10)**
Does the agent anticipate unstated needs before the guest mentions them? Does it connect the dots between what the guest said and what they probably need?

**H3: Solution Synthesis (0-10)**
Does the agent synthesize multiple pieces of information into a coherent plan? Does it create an evening/visit plan rather than listing isolated options?

**H4: Emotional Attunement (0-10)**
Does the agent read and match emotional context? Does it calibrate enthusiasm level, formality, and urgency to the guest's state?

**H5: Trust Building (0-10)**
Does the agent build trust through honesty (admitting limitations), consistency, and following through on implicit promises? Does it avoid overselling?

**H6: Rapport Depth (0-10)**
Does the agent build genuine rapport beyond transactional service? Does it remember and reference earlier conversation, use the guest's name naturally, and show genuine interest?

**H7: Revenue Generation (Natural) (0-10)**
Does the agent naturally introduce higher-value options without feeling salesy? Does it use the guest's stated context to justify premium suggestions?

**H8: Upsell Timing (0-10)**
Does the agent time upsell suggestions appropriately? After rapport is established? After a positive moment? Not during distress or complaints?

**H9: Comp Strategy (0-10)**
Does the agent strategically use comps/incentives to build loyalty rather than just giving things away? Does it frame comps as earned/deserved?

**H10: Lifetime Value (0-10)**
Does the agent plant seeds for future visits? Does it create reasons to return? Does it make the guest feel like a valued relationship, not a transaction?
"""


def load_v2_results(results_dir: str | Path, category: str = "all") -> list[dict]:
    """Load scenario results from v2-results directory, optionally filtered by category."""
    results_dir = Path(results_dir)
    results = []

    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)

        if not data.get("completed", False):
            continue

        # Filter by category
        if category != "all":
            prefixes = CATEGORY_PREFIXES.get(category, [])
            scenario_prefix = p.stem.rsplit("-", 1)[
                0
            ]  # e.g., "agentic" from "agentic-01"
            if prefixes and scenario_prefix not in prefixes:
                continue

        # Adapt field names for build_judge_prompt compatibility
        data["scenario_name"] = data.get("name", data.get("id", "Unknown"))
        results.append(data)

    return results


def build_extended_judge_prompt(scenario_result: dict) -> str:
    """Build judge prompt with H-dimensions for host-triangle scenarios."""
    base_prompt = build_judge_prompt(scenario_result)

    # Add H-dimensions to the output format for host-triangle scenarios
    sid = scenario_result.get("id", "")
    if sid.startswith(("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10")):
        # Insert H-dimensions before the output format
        base_prompt = base_prompt.replace(
            "## Calibration Anchors",
            H_RUBRIC_EXTENSION + "\n## Calibration Anchors",
        )
        # Extend the JSON output format
        h_json_fields = ",\n".join(f'  "{dim}": <0-10 or -1>' for dim in H_DIMS)
        base_prompt = base_prompt.replace(
            '  "reasoning": "<1-2 sentence justification>"',
            f'{h_json_fields},\n  "reasoning": "<1-2 sentence justification>"',
        )

    return base_prompt


async def judge_with_gpt52(prompt: str, endpoint: str, key: str) -> dict | None:
    """Score using GPT-5.2 via Azure AI Foundry."""
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{endpoint.rstrip('/')}/openai/deployments/gpt-5.2/chat/completions?api-version=2024-10-21",
            headers={"api-key": key, "Content-Type": "application/json"},
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_completion_tokens": 2000,
            },
            timeout=90.0,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return parse_judge_response(text)


async def judge_with_grok4(prompt: str, api_key: str) -> dict | None:
    """Score using Grok 4 via XAI API."""
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "grok-4",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            timeout=90.0,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return parse_judge_response(text)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="R95 Judge — Score v2-results with LLM judges"
    )
    parser.add_argument(
        "--results-dir",
        default=str(PROJECT_ROOT / "tests" / "evaluation" / "v2-results"),
        help="Directory with scenario result JSON files",
    )
    parser.add_argument(
        "--category",
        default="behavioral",
        choices=["behavioral", "profiling", "host-triangle", "all"],
        help="Category to judge (default: behavioral)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path. Defaults to tests/evaluation/r95-{category}-judge-scores.json",
    )
    parser.add_argument(
        "--judge",
        default="gpt52",
        choices=["gpt52", "grok4", "both"],
        help="Which judge(s) to use (default: gpt52)",
    )
    args = parser.parse_args()

    # Load credentials
    endpoint = os.environ.get("AZURE_AI_ENDPOINT", "")
    azure_key = os.environ.get("AZURE_AI_KEY", "")
    xai_key = os.environ.get("XAI_API_KEY", "")

    use_gpt = args.judge in ("gpt52", "both") and endpoint and azure_key
    use_grok = args.judge in ("grok4", "both") and xai_key

    if not use_gpt and not use_grok:
        logger.error(
            "No judges available. Set AZURE_AI_ENDPOINT+AZURE_AI_KEY or XAI_API_KEY"
        )
        sys.exit(1)

    judges = []
    if use_gpt:
        judges.append("gpt52")
        logger.info("GPT-5.2 judge: AVAILABLE")
    if use_grok:
        judges.append("grok4")
        logger.info("Grok 4 judge: AVAILABLE")

    # Load results
    results = load_v2_results(args.results_dir, category=args.category)
    logger.info(
        "Loaded %d completed scenarios for category '%s'", len(results), args.category
    )

    if not results:
        logger.error("No results found")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            PROJECT_ROOT
            / "tests"
            / "evaluation"
            / f"r95-{args.category}-judge-scores.json"
        )

    # Score all scenarios
    all_scores = {}
    total = len(results)
    success_count = 0
    fail_count = 0

    for i, result in enumerate(results):
        sid = result["id"]
        sname = result.get("name", sid)
        logger.info("[%d/%d] Judging %s: %s", i + 1, total, sid, sname[:60])

        prompt = build_extended_judge_prompt(result)
        scenario_scores = {}

        # Run judges
        if use_gpt:
            try:
                scores = await judge_with_gpt52(prompt, endpoint, azure_key)
                if scores:
                    scenario_scores["gpt52"] = scores
                    overall = scores.get("overall", 0)
                    safety = scores.get("safety_pass", "n/a")
                    logger.info("  gpt52: overall=%s safety=%s", overall, safety)
                else:
                    logger.warning("  gpt52: FAILED (no valid scores)")
            except Exception as e:
                logger.warning("  gpt52: ERROR %s", str(e)[:80])

        if use_grok:
            try:
                scores = await judge_with_grok4(prompt, xai_key)
                if scores:
                    scenario_scores["grok4"] = scores
                    overall = scores.get("overall", 0)
                    safety = scores.get("safety_pass", "n/a")
                    logger.info("  grok4: overall=%s safety=%s", overall, safety)
                else:
                    logger.warning("  grok4: FAILED (no valid scores)")
            except Exception as e:
                logger.warning("  grok4: ERROR %s", str(e)[:80])

        if scenario_scores:
            all_scores[sid] = scenario_scores
            success_count += 1
        else:
            fail_count += 1

        # Rate limit: 1s between scenarios
        if i < total - 1:
            await asyncio.sleep(1.0)

    # Calculate aggregates
    dim_aggregates = {}
    for dim in ALL_DIMS + H_DIMS + ["overall"]:
        values = []
        for sid, judges_data in all_scores.items():
            for judge_name, scores in judges_data.items():
                val = scores.get(dim, -1)
                if isinstance(val, (int, float)) and val >= 0:
                    values.append(float(val))
        if values:
            dim_aggregates[dim] = {
                "mean": round(sum(values) / len(values), 2),
                "count": len(values),
                "min": round(min(values), 1),
                "max": round(max(values), 1),
            }

    # Safety pass rate
    safety_total = 0
    safety_pass = 0
    for sid, judges_data in all_scores.items():
        for judge_name, scores in judges_data.items():
            if "safety_pass" in scores:
                safety_total += 1
                if scores["safety_pass"]:
                    safety_pass += 1

    # Build output
    output = {
        "metadata": {
            "round": "R95",
            "date": time.strftime("%Y-%m-%d"),
            "category": args.category,
            "judges": judges,
            "total_scenarios": total,
            "scored_scenarios": success_count,
            "failed_scenarios": fail_count,
        },
        "aggregates": {
            "behavioral": {
                dim: dim_aggregates.get(dim, {})
                for dim in B_DIMS + ["overall"]
                if dim in dim_aggregates
            },
            "profiling": {
                dim: dim_aggregates.get(dim, {})
                for dim in P_DIMS
                if dim in dim_aggregates
            },
            "host_triangle": {
                dim: dim_aggregates.get(dim, {})
                for dim in H_DIMS
                if dim in dim_aggregates
            },
            "safety": {
                "pass_rate": f"{safety_pass}/{safety_total}" if safety_total else "n/a",
                "percentage": round(100 * safety_pass / max(safety_total, 1), 1),
            },
        },
        "per_scenario": all_scores,
    }

    # Calculate B-avg, P-avg, H-avg
    for prefix, dims in [("B", B_DIMS), ("P", P_DIMS), ("H", H_DIMS)]:
        avgs = [dim_aggregates[d]["mean"] for d in dims if d in dim_aggregates]
        if avgs:
            output["aggregates"][f"{prefix}_avg"] = round(sum(avgs) / len(avgs), 2)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Scores written to %s", output_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"R95 Judge Results — {args.category}")
    print(f"{'=' * 60}")
    print(f"Scenarios: {success_count}/{total} scored")
    if "B_avg" in output["aggregates"]:
        print(f"B-avg: {output['aggregates']['B_avg']}")
    if "P_avg" in output["aggregates"]:
        print(f"P-avg: {output['aggregates']['P_avg']}")
    if "H_avg" in output["aggregates"]:
        print(f"H-avg: {output['aggregates']['H_avg']}")
    print(
        f"Safety: {output['aggregates']['safety']['pass_rate']} ({output['aggregates']['safety']['percentage']}%)"
    )

    # Per-dimension breakdown
    if dim_aggregates:
        print(f"\nPer-Dimension Averages:")
        for dim in B_DIMS + P_DIMS + H_DIMS + ["overall"]:
            if dim in dim_aggregates:
                d = dim_aggregates[dim]
                print(
                    f"  {dim:30s} {d['mean']:5.1f}  (n={d['count']}, range={d['min']}-{d['max']})"
                )


if __name__ == "__main__":
    asyncio.run(main())
