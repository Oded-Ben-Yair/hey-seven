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


def load_scenarios(subset_only: bool = True) -> list[dict[str, Any]]:
    """Load behavioral scenarios from YAML files.

    Args:
        subset_only: If True, only load scenarios in SUBSET_IDS.

    Returns:
        List of scenario dicts with ``_source_file`` injected.
    """
    scenarios: list[dict[str, Any]] = []
    for f in sorted(SCENARIOS_DIR.glob("behavioral_*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        for s in data.get("scenarios", []):
            if subset_only and s["id"] not in SUBSET_IDS:
                continue
            s["_source_file"] = f.name
            scenarios.append(s)
    return scenarios
