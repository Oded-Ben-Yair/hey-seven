"""Per-scenario atomic JSON writes with resume support."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .schemas import ScenarioResult


class AtomicStorage:
    """Write scenario results as individual JSON files with atomic rename.

    Resume is trivial: completed IDs = filenames already present in output_dir.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, scenario_id: str, result: ScenarioResult) -> Path:
        """Atomically write a scenario result to disk.

        Writes to a temp file first, then renames — prevents partial writes
        on crash or interrupt.
        """
        target = self.output_dir / f"{scenario_id}.json"
        # Write to a temp file in the same directory (same filesystem for rename)
        fd, tmp_path = tempfile.mkstemp(
            dir=self.output_dir, suffix=".tmp", prefix=f"{scenario_id}_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            os.replace(tmp_path, target)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return target

    def get_completed_ids(self) -> set[str]:
        """Return set of scenario IDs that have completed result files."""
        completed: set[str] = set()
        for p in self.output_dir.glob("*.json"):
            # Filename is {scenario_id}.json
            completed.add(p.stem)
        return completed

    def get_pending_ids(self, all_ids: set[str]) -> set[str]:
        """Return IDs not yet completed."""
        return all_ids - self.get_completed_ids()

    def load_result(self, scenario_id: str) -> dict[str, Any] | None:
        """Load a saved result by scenario ID, or None if not found."""
        target = self.output_dir / f"{scenario_id}.json"
        if not target.exists():
            return None
        with open(target) as f:
            return json.load(f)

    def load_all_results(self) -> list[dict[str, Any]]:
        """Load all saved results from the output directory."""
        results = []
        for p in sorted(self.output_dir.glob("*.json")):
            with open(p) as f:
                results.append(json.load(f))
        return results
