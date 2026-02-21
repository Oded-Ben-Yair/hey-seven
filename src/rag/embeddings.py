"""Embedding configuration for the RAG pipeline.

Uses Google Generative AI gemini-embedding-001 via API key.
Cached with TTLCache (1-hour TTL) consistent with ``_get_llm()`` and
other singletons that hold GCP credential references.

The ``task_type`` parameter allows optimized embeddings for different
use cases (``RETRIEVAL_QUERY`` vs ``RETRIEVAL_DOCUMENT``).  Each
distinct ``task_type`` gets its own cached instance.

R14 fix (DeepSeek F-002, Gemini F5, Grok F-001 — 3/3 consensus):
Replaced ``@lru_cache(maxsize=4)`` with ``TTLCache(maxsize=4, ttl=3600)``
for GCP Workload Identity credential rotation.  ``@lru_cache`` never
expires, causing embedding calls to fail after credential rotation
(~1 hour in WIF environments) until process restart.  All other
credential-bearing singletons already use TTLCache.
"""

import threading

from cachetools import TTLCache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import get_settings

# TTLCache with 1-hour TTL for GCP Workload Identity credential rotation.
# maxsize=4 allows separate cached instances per task_type (default,
# RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, etc.).
_embeddings_cache: TTLCache = TTLCache(maxsize=4, ttl=3600)

# R17 fix: DeepSeek F-001 (MEDIUM).  TTLCache is not thread-safe.
# get_embeddings() is called from both async contexts (lifespan startup)
# and sync contexts (inside asyncio.to_thread via retriever).  Concurrent
# TTL expiry could corrupt the cache dict.  Uses threading.Lock (not
# asyncio.Lock) because the primary contention is with to_thread workers —
# consistent with _retriever_lock pattern.
_embeddings_lock = threading.Lock()


def get_embeddings(task_type: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Get or create an embeddings model instance (cached per task_type).

    Uses TTLCache (1-hour TTL) consistent with ``_get_llm()`` and other
    singletons that hold GCP credential references.  Credentials rotate
    under Workload Identity Federation; TTL ensures periodic refresh.

    Protected by ``threading.Lock`` because this function is called from
    both async contexts and thread pool workers (via ``asyncio.to_thread``
    in the retriever path).  R17 fix (DeepSeek F-001).

    Args:
        task_type: Optional task type for the embedding model (e.g.,
            ``RETRIEVAL_QUERY``, ``RETRIEVAL_DOCUMENT``).  When ``None``,
            uses the model's default behavior.

    Returns:
        A GoogleGenerativeAIEmbeddings instance.
    """
    cache_key = task_type or "__default__"
    with _embeddings_lock:
        cached = _embeddings_cache.get(cache_key)
        if cached is not None:
            return cached

        settings = get_settings()
        kwargs: dict = {"model": settings.EMBEDDING_MODEL}
        if task_type:
            kwargs["task_type"] = task_type
        instance = GoogleGenerativeAIEmbeddings(**kwargs)
        _embeddings_cache[cache_key] = instance
        return instance


def clear_embeddings_cache() -> None:
    """Clear the embeddings TTL cache.

    Call from tests or after credential rotation to force fresh
    embedding model creation.
    """
    _embeddings_cache.clear()


# Backward-compatible attribute: callers using ``get_embeddings.cache_clear()``
# are redirected to ``clear_embeddings_cache()``.
get_embeddings.cache_clear = clear_embeddings_cache  # type: ignore[attr-defined]
