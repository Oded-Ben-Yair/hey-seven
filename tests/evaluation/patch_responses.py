#!/usr/bin/env python3
"""Enrich r83-subset-responses.json with fields from behavioral YAML sources.

Adds: expected_behavioral_quality, category, safety_relevant
Matches by scenario_id.
"""
import json
import yaml
from pathlib import Path

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"
RESPONSES_FILE = Path(__file__).resolve().parent / "r83-subset-responses.json"


def load_yaml_scenarios() -> dict:
    """Load all behavioral scenarios keyed by ID."""
    scenarios = {}
    for f in sorted(SCENARIOS_DIR.glob("behavioral_*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        for s in data.get("scenarios", []):
            scenarios[s["id"]] = s
    return scenarios


def main():
    yaml_scenarios = load_yaml_scenarios()
    print(f"Loaded {len(yaml_scenarios)} scenarios from YAML files")

    with open(RESPONSES_FILE) as f:
        data = json.load(f)

    patched = 0
    for result in data["results"]:
        sid = result["scenario_id"]
        yaml_s = yaml_scenarios.get(sid)
        if yaml_s:
            result["expected_behavioral_quality"] = yaml_s.get(
                "expected_behavioral_quality", result.get("expected_behavior", "")
            )
            result["category"] = yaml_s.get("category", "")
            result["safety_relevant"] = yaml_s.get("safety_relevant", False)
            patched += 1
        else:
            print(f"  WARNING: {sid} not found in YAML files")
            # Set safe defaults
            result.setdefault("expected_behavioral_quality", result.get("expected_behavior", ""))
            result.setdefault("category", "")
            result.setdefault("safety_relevant", False)

    with open(RESPONSES_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Patched {patched}/{len(data['results'])} scenarios")
    print(f"Written to {RESPONSES_FILE}")


if __name__ == "__main__":
    main()
