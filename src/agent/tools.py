"""Tool definitions for the Property Q&A agent.

Two RAG search tools: general search and schedule-focused lookup.
"""

import logging

from langchain_core.tools import tool

from src.config import get_settings
from src.rag.pipeline import get_retriever

logger = logging.getLogger(__name__)


@tool
def search_property(query: str) -> str:
    """Search the property knowledge base for information about restaurants,
    entertainment, rooms, amenities, gaming, and promotions.

    Args:
        query: Natural language search query about the property.

    Returns:
        Relevant information from the property knowledge base.
    """
    settings = get_settings()
    try:
        retriever = get_retriever()
        results = retriever.retrieve(query, top_k=settings.RAG_TOP_K)
    except Exception:
        logger.exception("Error searching property knowledge base")
        return "I'm having trouble searching the knowledge base right now. Please try again."

    if not results:
        return "No relevant information found in the knowledge base."

    parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("category", "general")
        parts.append(f"[{i}] ({source}) {doc.page_content}")

    return "\n---\n".join(parts)


@tool
def get_property_hours(venue_name: str) -> str:
    """Look up hours, schedules, and operating times for a specific venue
    such as a restaurant, show, or amenity.

    Use this tool when a guest asks about hours, opening times, or schedules
    for a specific venue. For general questions, use search_property instead.

    Args:
        venue_name: Name of the restaurant, show, or venue to look up.

    Returns:
        Hours and schedule information for the venue.
    """
    settings = get_settings()
    try:
        retriever = get_retriever()
        # Search with targeted query combining venue name and schedule keywords
        results = retriever.retrieve(
            f"{venue_name} hours schedule open close",
            top_k=settings.RAG_TOP_K,
        )
    except Exception:
        logger.exception("Error looking up hours for %s", venue_name)
        return "I'm having trouble looking up schedule information right now."

    if not results:
        return f"No schedule information found for {venue_name}."

    parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("category", "general")
        parts.append(f"[{i}] ({source}) {doc.page_content}")

    return "\n---\n".join(parts)
