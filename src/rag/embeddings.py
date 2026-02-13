"""Embedding configuration for the RAG pipeline.

Uses Google Generative AI text-embedding-004 via API key.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Get Google Generative AI embeddings model.

    Requires GOOGLE_API_KEY environment variable.

    Returns:
        A GoogleGenerativeAIEmbeddings instance.
    """
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
