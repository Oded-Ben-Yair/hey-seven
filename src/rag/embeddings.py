"""Embedding configuration for the RAG pipeline.

Uses Google Generative AI gemini-embedding-001 via API key.
Cached with ``@lru_cache`` consistent with ``_get_llm()`` and
``get_settings()`` singleton patterns.

The ``task_type`` parameter allows optimized embeddings for different
use cases (``RETRIEVAL_QUERY`` vs ``RETRIEVAL_DOCUMENT``).  Since
``@lru_cache`` caches by argument value, each distinct ``task_type``
gets its own cached instance.
"""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import get_settings


@lru_cache(maxsize=4)
def get_embeddings(task_type: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Get or create an embeddings model instance (cached per task_type).

    Uses ``@lru_cache`` consistent with ``_get_llm()`` and ``get_settings()``
    patterns.  Requires GOOGLE_API_KEY environment variable.

    Args:
        task_type: Optional task type for the embedding model (e.g.,
            ``RETRIEVAL_QUERY``, ``RETRIEVAL_DOCUMENT``).  When ``None``,
            uses the model's default behavior.

    Returns:
        A GoogleGenerativeAIEmbeddings instance.
    """
    settings = get_settings()
    kwargs: dict = {"model": settings.EMBEDDING_MODEL}
    if task_type:
        kwargs["task_type"] = task_type
    return GoogleGenerativeAIEmbeddings(**kwargs)
