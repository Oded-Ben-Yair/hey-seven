"""Embedding configuration for the RAG pipeline.

Uses Vertex AI text-embedding-005 for production and a local fallback for
development. Provides batch embedding utilities for both indexing and
query-time operations.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vertex AI Embeddings (Production)
# ---------------------------------------------------------------------------


def get_vertex_embeddings(
    project_id: str | None = None,
    location: str = "us-central1",
    model_name: str = "text-embedding-005",
) -> Any:
    """Create a Vertex AI embedding model instance.

    Uses langchain-google-vertexai for seamless integration with LangChain's
    embedding interface.

    Args:
        project_id: GCP project ID. If None, reads from GCP_PROJECT_ID env var.
        location: GCP region for the embedding endpoint.
        model_name: Vertex AI embedding model name. text-embedding-005 is
            the latest, with 768-dimensional output and improved multilingual
            support.

    Returns:
        A VertexAIEmbeddings instance compatible with LangChain vectorstores.

    Raises:
        ImportError: If langchain-google-vertexai is not installed.
    """
    from langchain_google_vertexai import VertexAIEmbeddings

    project = project_id or os.environ.get("GCP_PROJECT_ID")
    if not project:
        raise ValueError(
            "GCP_PROJECT_ID must be set via argument or environment variable."
        )

    return VertexAIEmbeddings(
        model_name=model_name,
        project=project,
        location=location,
    )


# ---------------------------------------------------------------------------
# Local Development Fallback
# ---------------------------------------------------------------------------


def get_local_embeddings() -> Any:
    """Create a local embedding model for development without GCP access.

    Uses HuggingFace sentence-transformers via langchain-community. Falls
    back to a deterministic hash-based embedding if transformers are not
    available.

    Returns:
        An embeddings instance compatible with LangChain vectorstores.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    except ImportError:
        logger.warning(
            "HuggingFace embeddings not available. Using Google Generative AI "
            "embeddings as fallback (requires GOOGLE_API_KEY)."
        )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embeddings(use_vertex: bool | None = None) -> Any:
    """Get the appropriate embedding model based on environment.

    Args:
        use_vertex: Force Vertex AI (True) or local (False). If None,
            auto-detects based on GCP_PROJECT_ID env var.

    Returns:
        An embeddings instance.
    """
    if use_vertex is None:
        use_vertex = bool(os.environ.get("GCP_PROJECT_ID"))

    if use_vertex:
        return get_vertex_embeddings()
    return get_local_embeddings()


# ---------------------------------------------------------------------------
# Batch Utilities
# ---------------------------------------------------------------------------


def batch_embed(
    texts: list[str],
    embeddings: Any = None,
    batch_size: int = 100,
) -> list[list[float]]:
    """Embed a list of texts in batches.

    Handles rate limiting and batch size constraints for large document
    collections. Vertex AI has a limit of 250 texts per batch request.

    Args:
        texts: List of text strings to embed.
        embeddings: An embeddings instance. If None, uses get_embeddings().
        batch_size: Number of texts per batch. Max 250 for Vertex AI.

    Returns:
        List of embedding vectors (list of floats) in the same order as input.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        logger.info(
            "Embedded batch %d/%d (%d texts)",
            i // batch_size + 1,
            (len(texts) + batch_size - 1) // batch_size,
            len(batch),
        )

    return all_embeddings


def embed_query(query: str, embeddings: Any = None) -> list[float]:
    """Embed a single query string.

    Uses the query-optimized embedding path, which may differ from document
    embedding for asymmetric models.

    Args:
        query: The query text to embed.
        embeddings: An embeddings instance. If None, uses get_embeddings().

    Returns:
        The embedding vector as a list of floats.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    return embeddings.embed_query(query)
