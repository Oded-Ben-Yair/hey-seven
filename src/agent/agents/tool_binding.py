"""Per-agent tool binding for LangGraph tool-use.

Maps specialist agent names to their available tools and provides a helper
to bind tools to an LLM instance. Uses ``llm.bind_tools()`` which returns
a new ``RunnableBinding`` — does NOT mutate the LLM singleton.

R106: Architecture shift — tool-use instead of prompt engineering.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def get_tools_for_agent(agent_name: str) -> list[BaseTool]:
    """Return the tools available for a specific specialist agent.

    Tool assignment is based on which tools are relevant to each agent's domain:
    - **comp**: All 4 tools (comp eligibility is its primary purpose)
    - **host**: All 4 tools (host needs full toolkit for relationship building)
    - **dining**: Events + tier lookup (cross-sell entertainment, check tier perks)
    - **entertainment**: Events + tier lookup (primary domain + loyalty perks)
    - **hotel**: Events + tier lookup (package stays with shows, check tier upgrades)

    Args:
        agent_name: The specialist agent identifier (comp, host, dining,
            entertainment, hotel).

    Returns:
        List of tool instances. Empty list for unknown agents.
    """
    from src.agent.casino_tools import (
        check_comp_eligibility,
        check_incentive_eligibility,
        check_tier_status,
        lookup_upcoming_events,
    )

    # Full toolkit agents
    _FULL_TOOLS: list[BaseTool] = [
        check_comp_eligibility,
        check_tier_status,
        lookup_upcoming_events,
        check_incentive_eligibility,
    ]

    # Partial toolkit agents (cross-sell and tier lookup only)
    _PARTIAL_TOOLS: list[BaseTool] = [
        lookup_upcoming_events,
        check_tier_status,
    ]

    _AGENT_TOOLS: dict[str, list[BaseTool]] = {
        "comp": _FULL_TOOLS,
        "host": _FULL_TOOLS,
        "dining": _PARTIAL_TOOLS,
        "entertainment": _PARTIAL_TOOLS,
        "hotel": _PARTIAL_TOOLS,
    }

    tools = _AGENT_TOOLS.get(agent_name, [])
    if tools:
        logger.debug(
            "R106: %d tools mapped for %s agent: %s",
            len(tools),
            agent_name,
            [t.name for t in tools],
        )
    return tools


def bind_tools_to_llm(
    llm: Any,
    agent_name: str,
    tool_use_enabled: bool = False,
) -> tuple[Any, list[BaseTool], bool]:
    """Bind appropriate tools to an LLM instance.

    Uses ``llm.bind_tools(tools)`` which returns a new ``RunnableBinding``
    instance — the original LLM singleton is NOT mutated.

    Args:
        llm: The LLM instance (ChatGoogleGenerativeAI or similar).
        agent_name: The specialist agent identifier.
        tool_use_enabled: Feature flag gate. If False, returns original LLM.

    Returns:
        Tuple of (llm_instance, tools_list, is_bound):
        - llm_instance: Either the original LLM or a new RunnableBinding with tools
        - tools_list: The tools that were bound (empty if not bound)
        - is_bound: True if tools were successfully bound
    """
    if not tool_use_enabled:
        return llm, [], False

    tools = get_tools_for_agent(agent_name)
    if not tools:
        logger.debug("R106: No tools for %s agent, skipping bind", agent_name)
        return llm, [], False

    try:
        llm_with_tools = llm.bind_tools(tools)
        logger.info(
            "R106: Bound %d tools to %s agent LLM",
            len(tools),
            agent_name,
        )
        return llm_with_tools, tools, True
    except Exception:
        logger.warning(
            "R106: Failed to bind tools for %s agent — falling back to no-tool mode",
            agent_name,
            exc_info=True,
        )
        return llm, [], False
