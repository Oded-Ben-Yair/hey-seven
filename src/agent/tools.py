"""Tool definitions for the Property Q&A agent.

Single RAG search tool that queries the property knowledge base.
"""

import logging

from langchain_core.tools import tool

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
    try:
        retriever = get_retriever()
        results = retriever.retrieve(query, top_k=5)
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
