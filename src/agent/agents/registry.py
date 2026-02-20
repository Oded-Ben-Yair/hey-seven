"""Agent registry — maps agent names to async handler functions.

Provides ``get_agent(name)`` for the v2 graph's dispatch node to
route queries to the appropriate specialist agent.
"""

from collections.abc import Callable

from .comp_agent import comp_agent
from .dining_agent import dining_agent
from .entertainment_agent import entertainment_agent
from .host_agent import host_agent
from .hotel_agent import hotel_agent

_AGENT_REGISTRY: dict[str, Callable] = {
    "host": host_agent,
    "dining": dining_agent,
    "entertainment": entertainment_agent,
    "comp": comp_agent,
    "hotel": hotel_agent,
}


def get_agent(name: str) -> Callable:
    """Return the agent function for the given name.

    Raises:
        KeyError: If the agent name is not registered.
    """
    if name not in _AGENT_REGISTRY:
        raise KeyError(
            f"Unknown agent: {name}. Available: {list(_AGENT_REGISTRY.keys())}"
        )
    return _AGENT_REGISTRY[name]


def list_agents() -> list[str]:
    """Return a sorted list of all registered agent names."""
    return sorted(_AGENT_REGISTRY.keys())
