"""Specialist agents for the v2 swarm architecture."""

from .comp_agent import comp_agent
from .dining_agent import dining_agent
from .entertainment_agent import entertainment_agent
from .host_agent import host_agent
from .registry import get_agent

__all__ = [
    "host_agent",
    "dining_agent",
    "entertainment_agent",
    "comp_agent",
    "get_agent",
]
