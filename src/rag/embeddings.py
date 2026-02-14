"""Embedding configuration for the RAG pipeline.

Uses Google Generative AI text-embedding-004 via API key.
Cached with ``@lru_cache`` consistent with ``_get_llm()`` and
``get_settings()`` singleton patterns.
"""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Get or create the shared embeddings model instance (cached singleton).

    Uses ``@lru_cache`` consistent with ``_get_llm()`` and ``get_settings()``
    patterns.  Requires GOOGLE_API_KEY environment variable.

    Returns:
        A GoogleGenerativeAIEmbeddings instance.
    """
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
