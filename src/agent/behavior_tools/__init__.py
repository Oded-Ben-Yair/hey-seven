"""Phase 2 structured behavior tools for specialist agents.

Each tool is pure deterministic business logic — no LLM calls, no I/O.
Tools inject structured prompt sections into specialist agent system prompts
to improve behavioral quality on specific evaluation dimensions.
"""

from src.agent.behavior_tools.comp_strategy import get_comp_prompt_section
from src.agent.behavior_tools.handoff import (
    build_handoff_summary,
    format_handoff_for_prompt,
)
from src.agent.behavior_tools.ltv_nudge import get_ltv_prompt_section
from src.agent.behavior_tools.rapport_ladder import get_rapport_prompt_section

__all__ = [
    "get_comp_prompt_section",
    "build_handoff_summary",
    "format_handoff_for_prompt",
    "get_ltv_prompt_section",
    "get_rapport_prompt_section",
]
