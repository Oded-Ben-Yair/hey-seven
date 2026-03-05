"""Shared constants and scenario loading for eval v2."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

# Same 20 IDs as run_subset_eval.py — representative cross-dimension sample.
SUBSET_IDS: set[str] = {
    "overall-01",
    "overall-03",
    "sarcasm-01",
    "sarcasm-06",
    "implicit-01",
    "implicit-09",
    "crisis-05",
    "engagement-08",
    "agentic-01",
    "agentic-09",
    "crisis-01",
    "nuance-03",
    "tone-01",
    "tone-03",
    "coherence-01",
    "coherence-03",
    "multilingual-01",
    "multilingual-08",
    "safety-01",
    "safety-03",
}

# R90: Profiling subset — 1-2 representative scenarios per P-dimension.
PROFILING_SUBSET_IDS: set[str] = {
    "p1-01",
    "p1-03",
    "p2-01",
    "p2-04",
    "p3-01",
    "p3-04",
    "p4-01",
    "p4-05",
    "p5-01",
    "p5-03",
    "p6-01",
    "p7-01",
    "p8-01",
    "p9-01",
    "p10-01",
    "p10-03",
}

# R90: Host Triangle subset — 1 representative scenario per H-dimension.
HOST_TRIANGLE_SUBSET_IDS: set[str] = {
    "h1-01",
    "h2-01",
    "h3-01",
    "h4-01",
    "h5-01",
    "h6-01",
    "h7-01",
    "h8-01",
    "h9-01",
    "h10-01",
}

# R90: Category → glob pattern for scenario file loading.
_CATEGORY_GLOBS: dict[str, str] = {
    "behavioral": "behavioral_*.yaml",
    "profiling": "profiling_*.yaml",
    "host-triangle": "host_triangle_*.yaml",
}

# R90: Category → subset IDs for --subset mode.
_CATEGORY_SUBSETS: dict[str, set[str]] = {
    "behavioral": SUBSET_IDS,
    "profiling": PROFILING_SUBSET_IDS,
    "host-triangle": HOST_TRIANGLE_SUBSET_IDS,
}

SCENARIOS_DIR = Path(__file__).resolve().parent.parent.parent / "scenarios"


@dataclasses.dataclass
class TurnResult:
    """Result for a single conversation turn."""

    turn_index: int
    user_message: str
    agent_response: str
    sources: list[str]
    elapsed_ms: int
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ScenarioResult:
    """Result for a complete scenario evaluation."""

    id: str
    name: str
    source_file: str
    category: str
    behavioral_dimension: str
    expected_behavior: str
    expected_behavioral_quality: str
    safety_relevant: bool
    turns: list[TurnResult]
    elapsed_ms: int
    error: str | None
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["turns"] = [t.to_dict() for t in self.turns]
        return d


def load_scenarios(
    subset_only: bool = True,
    category: str = "behavioral",
) -> list[dict[str, Any]]:
    """Load scenarios from YAML files filtered by category.

    Args:
        subset_only: If True, only load scenarios in the category's subset IDs.
        category: Scenario category — ``"behavioral"`` (default), ``"profiling"``,
            ``"host-triangle"``, or ``"all"`` (loads every ``*.yaml``).

    Returns:
        List of scenario dicts with ``_source_file`` injected.
    """
    if category == "all":
        glob_pattern = "*.yaml"
        subset_ids: set[str] = set()  # unused — all mode ignores subset
    else:
        glob_pattern = _CATEGORY_GLOBS.get(category, "behavioral_*.yaml")
        subset_ids = _CATEGORY_SUBSETS.get(category, SUBSET_IDS)

    scenarios: list[dict[str, Any]] = []
    for f in sorted(SCENARIOS_DIR.glob(glob_pattern)):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        for s in data.get("scenarios", []):
            if category != "all" and subset_only and s["id"] not in subset_ids:
                continue
            s["_source_file"] = f.name
            scenarios.append(s)
    return scenarios
