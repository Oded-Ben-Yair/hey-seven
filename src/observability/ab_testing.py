"""A/B testing framework for Hey Seven prompt and config variants.

Provides deterministic traffic splitting based on thread_id hash,
variant configuration, and trace tagging for metric comparison.

Traffic splitting uses SHA-256 hash of thread_id modulo 100 to assign
variants. This ensures:
- Same thread_id always gets same variant (consistent experience)
- Distribution is uniform across variants
- No external state required (stateless)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ABVariant:
    """A single A/B test variant configuration."""

    name: str  # e.g., "control", "treatment_a"
    description: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    # config_overrides can contain:
    # - "system_prompt_suffix": additional prompt text
    # - "model_temperature": override temperature
    # - "rag_top_k": override retrieval count
    # etc.


@dataclass
class ABExperiment:
    """An A/B test experiment definition."""

    experiment_id: str
    description: str
    variants: list[ABVariant]
    traffic_split: list[int]  # Percentages, must sum to 100
    enabled: bool = True

    def __post_init__(self):
        if len(self.variants) != len(self.traffic_split):
            raise ValueError(
                f"variants ({len(self.variants)}) and traffic_split "
                f"({len(self.traffic_split)}) must have same length"
            )
        total = sum(self.traffic_split)
        if total != 100:
            raise ValueError(f"traffic_split must sum to 100, got {total}")


@dataclass
class ABAssignment:
    """The result of assigning a thread to a variant."""

    experiment_id: str
    variant_name: str
    config_overrides: dict[str, Any]
    bucket: int  # 0-99, the hash bucket


def _hash_to_bucket(thread_id: str, experiment_id: str) -> int:
    """Deterministically map a thread_id to a bucket (0-99).

    Uses SHA-256 of thread_id + experiment_id for uniform distribution.
    Including experiment_id ensures different experiments can assign
    the same thread to different variants.
    """
    h = hashlib.sha256(f"{thread_id}:{experiment_id}".encode()).hexdigest()
    return int(h[:8], 16) % 100


def assign_variant(
    thread_id: str,
    experiment: ABExperiment,
) -> ABAssignment:
    """Assign a thread to a variant based on traffic split.

    Args:
        thread_id: The conversation thread ID.
        experiment: The experiment definition.

    Returns:
        ABAssignment with the assigned variant and config overrides.
    """
    if not experiment.enabled:
        # Disabled experiment: always return first variant (control)
        return ABAssignment(
            experiment_id=experiment.experiment_id,
            variant_name=experiment.variants[0].name,
            config_overrides=experiment.variants[0].config_overrides,
            bucket=0,
        )

    bucket = _hash_to_bucket(thread_id, experiment.experiment_id)

    cumulative = 0
    for variant, split in zip(experiment.variants, experiment.traffic_split):
        cumulative += split
        if bucket < cumulative:
            logger.debug(
                "AB assignment: experiment=%s thread=%s bucket=%d variant=%s",
                experiment.experiment_id,
                thread_id[:8],
                bucket,
                variant.name,
            )
            return ABAssignment(
                experiment_id=experiment.experiment_id,
                variant_name=variant.name,
                config_overrides=variant.config_overrides,
                bucket=bucket,
            )

    # Fallback: last variant (should never reach here with valid split)
    last = experiment.variants[-1]
    return ABAssignment(
        experiment_id=experiment.experiment_id,
        variant_name=last.name,
        config_overrides=last.config_overrides,
        bucket=bucket,
    )


def get_trace_tags(assignment: ABAssignment) -> list[str]:
    """Generate trace tags for an A/B assignment.

    Tags are used to filter and compare metrics in LangFuse.

    Args:
        assignment: The AB assignment.

    Returns:
        List of tag strings.
    """
    return [
        f"ab:{assignment.experiment_id}",
        f"variant:{assignment.variant_name}",
        f"bucket:{assignment.bucket}",
    ]
