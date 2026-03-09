#!/usr/bin/env python3
"""Export gold traces for Gemini fine-tuning.

Reads eval results from R104/R105/R103 streaming directories and monolithic
response files. Filters conversations where judge scored overall > threshold.
Exports as JSONL in Vertex AI tuning format (contents format, NOT OpenAI messages).

Also exports manually crafted gold trace conversations from the R101 paradigm shift.

Usage:
    python scripts/export_gold_traces.py --min-score 7.0 --output data/training/
    python scripts/export_gold_traces.py --min-score 6.0 --output data/training/  # more permissive
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "tests" / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
GOLD_TRACES_PATH = (
    Path.home() / ".claude" / "teams" / "r101-paradigm-shift" / "gold-traces.md"
)

# Eval response files to scan (monolithic JSON with metadata + results array)
RESPONSE_FILES = [
    "r104-ht-pro-responses.json",
    "r105-ht-pro-responses.json",
    "r105-prof-pro-responses.json",
    "r99-pro-ht-responses.json",
    "r98-host-triangle-responses.json",
]

# Streaming result directories (individual JSON files per scenario)
STREAMING_DIRS = [
    "r104-ht-pro-streaming",
    "r104-prof-pro-streaming",
    "r104-rel-pro-streaming",
    "r105-ht-pro-streaming",
    "r105-prof-pro-streaming",
    "r103-streaming",
    "r108-tools-streaming",
]

# Judge score files (map scenario_id -> dimension scores)
JUDGE_FILES = [
    "r105-ht-pro-judge-scores.json",
    "r105-prof-pro-judge-scores.json",
    "r104-ht-pro-judge-scores.json",
    "r99-pro-ht-judge-scores.json",
    "r98-host-triangle-judge-scores.json",
]

# System prompt placeholder — the actual system prompt from the agent
# In production, this would be read from src/agent/prompts.py
SYSTEM_PROMPT_PLACEHOLDER = (
    "You are a casino host at Mohegan Sun in Uncasville, Connecticut. "
    "You are a relationship builder, not a question-answering kiosk. "
    "Every turn: address the immediate need, ask one natural profiling question, "
    "customize suggestions using gathered info, and offer human host bridge when appropriate."
)


def load_judge_scores(eval_dir: Path) -> dict[str, dict]:
    """Load all judge score files into a unified map: scenario_id -> scores dict."""
    all_scores: dict[str, dict] = {}
    for jf in JUDGE_FILES:
        path = eval_dir / "results" / jf
        if not path.exists():
            path = eval_dir / jf
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        scores_list = data.get("scores", [])
        for entry in scores_list:
            sid = entry.get("id", "")
            scores = entry.get("scores", {})
            if sid and scores:
                # Keep highest-scored version if duplicate
                existing = all_scores.get(sid)
                if existing is None or scores.get("overall", 0) > existing.get(
                    "overall", 0
                ):
                    all_scores[sid] = scores
    return all_scores


def compute_avg_score(scores: dict) -> float:
    """Compute average of all positive dimension scores (skip -1 = N/A)."""
    vals = [
        v
        for k, v in scores.items()
        if isinstance(v, (int, float))
        and v >= 0
        and k not in ("overall", "safety_pass")
    ]
    if not vals:
        return scores.get("overall", 0.0)
    return sum(vals) / len(vals)


def load_streaming_results(results_dir: Path) -> list[dict]:
    """Load individual scenario JSON files from streaming directories."""
    results = []
    for dirname in STREAMING_DIRS:
        d = results_dir / dirname
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if data.get("completed") and data.get("turns"):
                    results.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    return results


def load_monolithic_results(eval_dir: Path) -> list[dict]:
    """Load results from monolithic response JSON files."""
    results = []
    for fname in RESPONSE_FILES:
        path = eval_dir / fname
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for r in data.get("results", []):
            if r.get("turns"):
                results.append(r)
    return results


def dedupe_results(results: list[dict]) -> list[dict]:
    """Deduplicate by scenario_id, keeping the one with most turns."""
    seen: dict[str, dict] = {}
    for r in results:
        sid = r.get("scenario_id", "")
        if not sid:
            continue
        existing = seen.get(sid)
        if existing is None or len(r.get("turns", [])) > len(existing.get("turns", [])):
            seen[sid] = r
    return list(seen.values())


def filter_high_scored(
    results: list[dict],
    judge_scores: dict[str, dict],
    min_score: float,
) -> list[dict]:
    """Filter results where judge overall or avg dimension score >= min_score."""
    filtered = []
    for r in results:
        sid = r.get("scenario_id", "")
        scores = judge_scores.get(sid)
        if scores is None:
            # No judge scores — skip (we only want scored conversations)
            continue
        overall = scores.get("overall", 0)
        avg = compute_avg_score(scores)
        best = max(overall, avg)
        if best >= min_score:
            r["_judge_overall"] = overall
            r["_judge_avg"] = round(avg, 2)
            filtered.append(r)
    return filtered


def conversation_to_vertex_jsonl(result: dict, system_prompt: str) -> dict:
    """Convert an eval result to Vertex AI tuning JSONL format.

    Format:
    {"systemInstruction": {"role": "system", "parts": [{"text": "..."}]},
     "contents": [{"role": "user", "parts": [{"text": "..."}]},
                  {"role": "model", "parts": [{"text": "..."}]}, ...]}
    """
    contents = []
    for turn in result.get("turns", []):
        user_msg = turn.get("user_message", "")
        agent_resp = turn.get("agent_response", "")
        if not user_msg or not agent_resp:
            continue
        # Skip turns with errors
        if turn.get("error"):
            continue
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": agent_resp}]})

    if not contents:
        return {}

    # Last message must be model role
    if contents[-1]["role"] != "model":
        contents = contents[:-1]

    if not contents:
        return {}

    return {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system_prompt}],
        },
        "contents": contents,
    }


def parse_gold_trace_conversation(text: str) -> list[dict]:
    """Parse a gold trace conversation from markdown into turn pairs.

    Looks for patterns like:
    **Turn N - Guest:**
    > message text

    **Turn N - Host:**
    > response text
    """
    turns = []
    # Match turn blocks: "**Turn N - Guest/Host:**" followed by "> text"
    pattern = re.compile(
        r"\*\*Turn\s+\d+\s*-\s*(Guest|Host):\*\*\s*\n>\s*(.+?)(?=\n\n|\n\*\*Turn|\Z)",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    current_user = None
    for role, content in matches:
        content = content.strip()
        if role == "Guest":
            current_user = content
        elif role == "Host" and current_user:
            turns.append({"user_message": current_user, "agent_response": content})
            current_user = None
    return turns


def load_manual_gold_traces(traces_path: Path) -> list[dict]:
    """Load manually crafted gold trace conversations from markdown."""
    if not traces_path.exists():
        return []

    text = traces_path.read_text()
    conversations = []

    # Split by conversation headers "## CONVERSATION N:"
    conv_pattern = re.compile(
        r"## CONVERSATION \d+:(.+?)(?=\n## CONVERSATION|\n## Calibration|\n## Usage|\Z)",
        re.DOTALL,
    )
    conv_blocks = conv_pattern.findall(text)

    for i, block in enumerate(conv_blocks, 1):
        # Extract score from header or "Score: N/10"
        score_match = re.search(r"Score:\s*(\d+)/10", block)
        score = int(score_match.group(1)) if score_match else 0

        # Skip low-scored examples (3/10 chatbot version, etc.)
        if score < 6:
            continue

        turns = parse_gold_trace_conversation(block)
        if turns:
            # Extract scenario description from first line
            first_line = block.strip().split("\n")[0].strip()
            conversations.append(
                {
                    "scenario_id": f"gold-manual-{i:02d}",
                    "scenario_name": first_line[:80],
                    "score": score,
                    "turns": turns,
                }
            )

    return conversations


def manual_trace_to_vertex_jsonl(conv: dict, system_prompt: str) -> dict:
    """Convert a manual gold trace to Vertex AI JSONL format."""
    contents = []
    for turn in conv.get("turns", []):
        contents.append({"role": "user", "parts": [{"text": turn["user_message"]}]})
        contents.append({"role": "model", "parts": [{"text": turn["agent_response"]}]})

    if not contents or contents[-1]["role"] != "model":
        return {}

    return {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system_prompt}],
        },
        "contents": contents,
    }


def export_to_jsonl(records: list[dict], output_path: Path) -> int:
    """Write records as JSONL. Returns count written."""
    count = 0
    with open(output_path, "w") as f:
        for record in records:
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Export gold traces for Gemini fine-tuning"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=8.0,
        help="Minimum judge overall/avg score to include (default: 8.0, raised R107)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "training",
        help="Output directory for JSONL files (default: data/training/)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=SYSTEM_PROMPT_PLACEHOLDER,
        help="System prompt to include in training examples",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading judge scores from {len(JUDGE_FILES)} files...")
    judge_scores = load_judge_scores(EVAL_DIR)
    print(f"  Loaded scores for {len(judge_scores)} scenarios")

    print(f"Loading streaming results from {len(STREAMING_DIRS)} directories...")
    streaming = load_streaming_results(RESULTS_DIR)
    print(f"  Loaded {len(streaming)} streaming results")

    print(f"Loading monolithic results from {len(RESPONSE_FILES)} files...")
    monolithic = load_monolithic_results(EVAL_DIR)
    print(f"  Loaded {len(monolithic)} monolithic results")

    all_results = dedupe_results(streaming + monolithic)
    print(f"  Deduped to {len(all_results)} unique scenarios")

    print(f"\nFiltering with min_score={args.min_score}...")
    high_scored = filter_high_scored(all_results, judge_scores, args.min_score)
    print(f"  {len(high_scored)} conversations passed threshold")

    if high_scored:
        # Score distribution
        overalls = [r["_judge_overall"] for r in high_scored]
        avgs = [r["_judge_avg"] for r in high_scored]
        print(
            f"  Overall scores: min={min(overalls)}, max={max(overalls)}, "
            f"mean={sum(overalls) / len(overalls):.1f}"
        )
        print(
            f"  Avg dim scores: min={min(avgs):.1f}, max={max(avgs):.1f}, "
            f"mean={sum(avgs) / len(avgs):.1f}"
        )

    # Convert to Vertex AI format
    eval_records = []
    for r in high_scored:
        record = conversation_to_vertex_jsonl(r, args.system_prompt)
        if record:
            eval_records.append(record)

    eval_path = args.output / "gold-traces-r106.jsonl"
    eval_count = export_to_jsonl(eval_records, eval_path)
    print(f"\nExported {eval_count} eval conversations to {eval_path}")

    # Manual gold traces
    print(f"\nLoading manual gold traces from {GOLD_TRACES_PATH}...")
    manual_traces = load_manual_gold_traces(GOLD_TRACES_PATH)
    print(f"  Found {len(manual_traces)} gold trace conversations (score >= 6)")

    manual_records = []
    for conv in manual_traces:
        record = manual_trace_to_vertex_jsonl(conv, args.system_prompt)
        if record:
            manual_records.append(record)

    manual_path = args.output / "gold-traces-manual.jsonl"
    manual_count = export_to_jsonl(manual_records, manual_path)
    print(f"  Exported {manual_count} manual conversations to {manual_path}")

    # Write README
    total = eval_count + manual_count
    readme_path = args.output / "README.md"
    readme_path.write_text(
        f"# Hey Seven Fine-Tuning Training Data\n\n"
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"Min score threshold: {args.min_score}\n\n"
        f"## Files\n\n"
        f"| File | Count | Source |\n"
        f"|------|-------|--------|\n"
        f"| `gold-traces-r106.jsonl` | {eval_count} | "
        f"R103-R105 eval results (judge score >= {args.min_score}) |\n"
        f"| `gold-traces-manual.jsonl` | {manual_count} | "
        f"Manually crafted gold traces (score >= 6/10) |\n\n"
        f"## Format\n\n"
        f"Vertex AI supervised fine-tuning JSONL format:\n"
        f'- `systemInstruction.role`: `"system"`\n'
        f'- `contents[].role`: `"user"` or `"model"`\n'
        f"- `contents[].parts[].text`: message text\n"
        f'- Last message is always `"model"` role\n\n'
        f'**NOT OpenAI messages format.** Uses `"model"` not `"assistant"`.\n\n'
        f"## Target Model\n\n"
        f"Gemini 2.5 Flash (via Vertex AI). Gemini 3.x does not support "
        f"fine-tuning as of March 2026.\n\n"
        f"## Total\n\n"
        f"**{total} training conversations**\n\n"
        f"## Usage\n\n"
        f"```bash\n"
        f"# Upload to GCS\n"
        f"gsutil cp data/training/*.jsonl gs://hey-seven-training/\n\n"
        f"# Start tuning job\n"
        f'python -c "\n'
        f"from vertexai.tuning import sft\n"
        f"job = sft.train(\n"
        f"    source_model='gemini-2.5-flash',\n"
        f"    train_dataset='gs://hey-seven-training/gold-traces-r106.jsonl',\n"
        f"    epochs=4, adapter_size=4,\n"
        f"    tuned_model_display_name='hey-seven-host-v1',\n"
        f")\n"
        f'"\n'
        f"```\n"
    )
    print(f"  Wrote {readme_path}")

    print(f"\n{'=' * 50}")
    print(f"TOTAL: {total} training conversations exported")
    print(f"  Eval-sourced: {eval_count}")
    print(f"  Manual gold:  {manual_count}")
    print(f"{'=' * 50}")

    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
