#!/usr/bin/env python3
"""R106 Judge — Multi-model scoring with GPT-5.4, Grok 4, DeepSeek consensus.

Reads individual scenario JSON files from a results directory, builds judge
prompts, calls 1-3 LLM judges, and writes aggregated scores with optional
consensus (median) scoring.

Judges:
  - GPT-5.4 (via Azure AI Foundry) — primary, structure/completeness
  - Grok 4 (via XAI API) — genuineness/anti-chatbot detection
  - DeepSeek-V3.2-Speciale (via Azure AI Foundry) — outcomes/guest lens
  - GPT-5.2 (via Azure AI Foundry) — legacy fallback

Usage:
    export AZURE_AI_ENDPOINT="https://brn-azai.cognitiveservices.azure.com/"
    export AZURE_AI_KEY="<key>"
    export XAI_API_KEY="<key>"

    # Single judge
    python tests/evaluation/run_r95_judge.py --results-dir <dir> --judge gpt53

    # 3-model consensus (median per dimension)
    python tests/evaluation/run_r95_judge.py --results-dir <dir> --judge consensus

    # All judges separately (no median)
    python tests/evaluation/run_r95_judge.py --results-dir <dir> --judge all
"""

import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from statistics import median

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
        data["scenario_name"] = data.get(
            "scenario_name", data.get("name", data.get("id", "Unknown"))
        )
        # Normalize id field (streaming uses scenario_id, batch uses id)
        if "id" not in data or data["id"] is None:
            data["id"] = data.get("scenario_id", p.stem)
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


# Reasoning models (o1-style) don't support temperature or top_p
_REASONING_DEPLOYMENTS = frozenset({"gpt-5.3-chat", "gpt-5.3-chat-2026-03-03"})


async def _call_azure_judge(
    prompt: str, endpoint: str, key: str, deployment: str, timeout: float = 90.0
) -> dict | None:
    """Call an Azure AI Foundry deployment and parse judge response."""
    import httpx

    body: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 2000,
    }
    # Reasoning models don't support temperature — omit it
    if deployment not in _REASONING_DEPLOYMENTS:
        body["temperature"] = 0.1

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21",
            headers={"api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]

        # Strip reasoning model tags — DeepSeek includes </think>, some models use other tags
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()

        result = parse_judge_response(text)
        if result is None:
            logger.debug(
                "Parse failed for %s. First 300 chars: %s",
                deployment,
                text[:300].replace("\n", " "),
            )
        return result


async def judge_with_gpt54(prompt: str, endpoint: str, key: str) -> dict | None:
    """Score using GPT-5.4 via Azure AI Foundry."""
    return await _call_azure_judge(prompt, endpoint, key, "gpt-5.4")


async def judge_with_gpt52(prompt: str, endpoint: str, key: str) -> dict | None:
    """Score using GPT-5.2 via Azure AI Foundry (legacy fallback)."""
    return await _call_azure_judge(prompt, endpoint, key, "gpt-5.2")


async def judge_with_deepseek(prompt: str, endpoint: str, key: str) -> dict | None:
    """Score using DeepSeek-V3.2-Speciale via Azure AI Foundry.

    WARNING: DeepSeek is very slow (~3-5 min per scenario). Not recommended
    for full panel runs. Use for targeted disagreement mining only.

    Uses a system message to force JSON-only output since DeepSeek tends to
    produce prose analysis instead of structured JSON.
    """
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{endpoint.rstrip('/')}/openai/deployments/DeepSeek-V3.2-Speciale/chat/completions?api-version=2024-10-21",
            headers={"api-key": key, "Content-Type": "application/json"},
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a JSON scoring engine. Output ONLY valid JSON. No prose, no analysis, no markdown. Just the JSON object.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_completion_tokens": 2000,
            },
            timeout=300.0,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]

        # Strip reasoning tags if present
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()

        result = parse_judge_response(text)
        if result is None:
            logger.debug(
                "DeepSeek parse failed. First 300 chars: %s",
                text[:300].replace("\n", " "),
            )
        return result


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


def compute_consensus(judge_scores: dict[str, dict]) -> dict:
    """Compute consensus (median) across multiple judges per dimension.

    Args:
        judge_scores: {judge_name: {dim: score, ...}, ...}

    Returns:
        Consensus dict with median scores, spread, and per-judge breakdown.
    """
    all_dims = ALL_DIMS + H_DIMS + ["overall"]
    consensus = {}

    for dim in all_dims:
        values = []
        for jname, scores in judge_scores.items():
            val = scores.get(dim, -1)
            if isinstance(val, (int, float)) and val >= 0:
                values.append(float(val))
        if values:
            consensus[dim] = round(median(values), 1)

    # Safety: majority vote
    safety_votes = []
    for jname, scores in judge_scores.items():
        sp = scores.get("safety_pass")
        if sp is not None:
            safety_votes.append(bool(sp))
    if safety_votes:
        consensus["safety_pass"] = sum(safety_votes) > len(safety_votes) / 2

    # Reasoning: concatenate
    reasonings = []
    for jname, scores in judge_scores.items():
        r = scores.get("reasoning", "")
        if r:
            reasonings.append(f"[{jname}] {r}")
    consensus["reasoning"] = " | ".join(reasonings)

    # Spread metadata
    consensus["_judge_count"] = len(judge_scores)
    consensus["_judges"] = list(judge_scores.keys())

    return consensus


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="R106 Judge — Multi-model scoring with consensus"
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
        help="Output JSON file path",
    )
    parser.add_argument(
        "--judge",
        default="gpt54",
        choices=["gpt52", "gpt54", "grok4", "deepseek", "consensus", "all"],
        help="Judge mode: single model, 'consensus' (median of 3), or 'all' (default: gpt53)",
    )
    args = parser.parse_args()

    # Load credentials
    endpoint = os.environ.get("AZURE_AI_ENDPOINT", "")
    azure_key = os.environ.get("AZURE_AI_KEY", "")
    xai_key = os.environ.get("XAI_API_KEY", "")

    # Build list of requested judges
    # consensus = GPT-5.4 + Grok 4 (fast, reliable pair)
    # all = GPT-5.4 + Grok 4 + DeepSeek (slow, use for disagreement mining)
    judge_map = {}
    if args.judge in ("gpt54", "consensus", "all"):
        if endpoint and azure_key:
            judge_map["gpt54"] = ("azure", judge_with_gpt54)
            logger.info("GPT-5.4 judge: AVAILABLE")
        else:
            logger.warning("GPT-5.4 judge: UNAVAILABLE (no AZURE_AI_ENDPOINT)")
    if args.judge in ("gpt52",):
        if endpoint and azure_key:
            judge_map["gpt52"] = ("azure", judge_with_gpt52)
            logger.info("GPT-5.2 judge: AVAILABLE")
        else:
            logger.warning("GPT-5.2 judge: UNAVAILABLE")
    if args.judge in ("grok4", "consensus", "all"):
        if xai_key:
            judge_map["grok4"] = ("xai", judge_with_grok4)
            logger.info("Grok 4 judge: AVAILABLE")
        else:
            logger.warning("Grok 4 judge: UNAVAILABLE (no XAI_API_KEY)")
    if args.judge in ("deepseek", "all"):
        # DeepSeek excluded from consensus (too slow: ~3-5 min/scenario)
        if endpoint and azure_key:
            judge_map["deepseek"] = ("azure", judge_with_deepseek)
            logger.info("DeepSeek-V3.2-Speciale judge: AVAILABLE (slow — 300s timeout)")
        else:
            logger.warning("DeepSeek judge: UNAVAILABLE (no AZURE_AI_ENDPOINT)")

    if not judge_map:
        logger.error(
            "No judges available. Set AZURE_AI_ENDPOINT+AZURE_AI_KEY and/or XAI_API_KEY"
        )
        sys.exit(1)

    use_consensus = args.judge == "consensus" and len(judge_map) >= 2
    if args.judge == "consensus" and len(judge_map) < 2:
        logger.warning(
            "Consensus requires 2+ judges, only %d available. Running without consensus.",
            len(judge_map),
        )

    judges = list(judge_map.keys())
    logger.info("Active judges: %s (consensus=%s)", judges, use_consensus)

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
            / f"r106-{args.category}-judge-scores.json"
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

        # Run all judges in parallel
        async def _run_judge(jname, jtype, jfn):
            try:
                if jtype == "azure":
                    return jname, await jfn(prompt, endpoint, azure_key)
                else:  # xai
                    return jname, await jfn(prompt, xai_key)
            except Exception as e:
                logger.warning("  %s: ERROR %s", jname, str(e)[:80])
                return jname, None

        tasks = [
            _run_judge(jname, jtype, jfn) for jname, (jtype, jfn) in judge_map.items()
        ]
        judge_results = await asyncio.gather(*tasks)

        for jname, scores in judge_results:
            if scores:
                scenario_scores[jname] = scores
                overall = scores.get("overall", 0)
                safety = scores.get("safety_pass", "n/a")
                logger.info("  %s: overall=%s safety=%s", jname, overall, safety)
            else:
                logger.warning("  %s: FAILED (no valid scores)", jname)

        # Add consensus if enabled and 2+ judges succeeded
        if use_consensus and len(scenario_scores) >= 2:
            scenario_scores["consensus"] = compute_consensus(scenario_scores)
            c = scenario_scores["consensus"]
            logger.info(
                "  consensus: overall=%s (from %d judges)",
                c.get("overall", "?"),
                c.get("_judge_count", 0),
            )

        if scenario_scores:
            all_scores[sid] = scenario_scores
            success_count += 1
        else:
            fail_count += 1

        # Rate limit: 1s between scenarios
        if i < total - 1:
            await asyncio.sleep(1.0)

    # Determine which judge to aggregate (consensus if available, else all)
    agg_judge_key = "consensus" if use_consensus else None

    # Calculate aggregates — use consensus scores when available, else average all judges
    dim_aggregates = {}
    for dim in ALL_DIMS + H_DIMS + ["overall"]:
        values = []
        for sid, judges_data in all_scores.items():
            if agg_judge_key and agg_judge_key in judges_data:
                val = judges_data[agg_judge_key].get(dim, -1)
                if isinstance(val, (int, float)) and val >= 0:
                    values.append(float(val))
            else:
                for judge_name, scores in judges_data.items():
                    if judge_name.startswith("_"):
                        continue
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
            "round": "R106",
            "date": time.strftime("%Y-%m-%d"),
            "consensus": use_consensus,
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
    print(f"R106 Judge Results — {args.category} — judges: {judges}")
    if use_consensus:
        print(f"Mode: CONSENSUS (median of {len(judge_map)} judges)")
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
