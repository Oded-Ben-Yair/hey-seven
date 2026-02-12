"""Retrieval pipeline for the Casino Host Agent.

Provides semantic search with optional reranking over the casino knowledge
base. Can be used as a standalone retriever or integrated into the agent
graph as a tool or node.
"""

import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retriever Configuration
# ---------------------------------------------------------------------------


class CasinoKnowledgeRetriever:
    """Semantic search retriever for casino domain knowledge.

    Wraps a vector store (ChromaDB or Vertex AI Vector Search) and provides
    retrieval with optional reranking.

    Usage:
        retriever = CasinoKnowledgeRetriever(vectorstore=chroma_instance)
        results = retriever.retrieve("What is theoretical win?", top_k=5)
    """

    def __init__(
        self,
        vectorstore: Any = None,
        reranker: Any = None,
        top_k: int = 5,
        rerank_top_k: int = 3,
    ) -> None:
        """Initialize the retriever.

        Args:
            vectorstore: A LangChain-compatible vectorstore instance.
                If None, attempts to load from default ChromaDB path.
            reranker: Optional reranking model. If None, results are
                returned in raw similarity order.
            top_k: Number of documents to retrieve from vector search.
            rerank_top_k: Number of documents to return after reranking.
        """
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_category: str | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Override the default top_k for this query.
            filter_category: Optional category filter (e.g., "regulations",
                "casino-operations").

        Returns:
            List of LangChain Document objects, ranked by relevance.
        """
        if self.vectorstore is None:
            logger.warning("No vectorstore configured. Returning empty results.")
            return []

        k = top_k or self.top_k

        # Build filter dict if category specified
        search_kwargs: dict[str, Any] = {"k": k}
        if filter_category:
            search_kwargs["filter"] = {"category": filter_category}

        results = self.vectorstore.similarity_search(
            query, **search_kwargs
        )

        # Apply reranking if available
        if self.reranker and len(results) > self.rerank_top_k:
            results = self._rerank(query, results)

        return results

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with similarity scores.

        Args:
            query: The search query.
            top_k: Override the default top_k.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        if self.vectorstore is None:
            return []

        k = top_k or self.top_k
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Rerank retrieved documents using the configured reranker.

        Args:
            query: The original search query.
            documents: List of documents to rerank.

        Returns:
            Reranked list of documents, truncated to rerank_top_k.
        """
        if self.reranker is None:
            return documents[: self.rerank_top_k]

        try:
            # Reranker expects (query, document_text) pairs
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.reranker.predict(pairs)

            # Sort by reranker score (descending)
            scored_docs = sorted(
                zip(documents, scores), key=lambda x: x[1], reverse=True
            )
            return [doc for doc, _ in scored_docs[: self.rerank_top_k]]
        except Exception:
            logger.exception("Reranking failed, returning original order.")
            return documents[: self.rerank_top_k]


# ---------------------------------------------------------------------------
# Global Retriever Instance
# ---------------------------------------------------------------------------

_retriever_instance: CasinoKnowledgeRetriever | None = None


def get_retriever() -> CasinoKnowledgeRetriever:
    """Get or create the global retriever instance.

    Lazy-initializes the retriever with ChromaDB from the default persist
    directory.

    Returns:
        The CasinoKnowledgeRetriever instance.
    """
    global _retriever_instance

    if _retriever_instance is not None:
        return _retriever_instance

    # Attempt to load existing ChromaDB
    persist_dir = os.environ.get(
        "CHROMA_PERSIST_DIR",
        str(
            __import__("pathlib").Path(__file__).resolve().parent.parent
            / "data"
            / "chroma"
        ),
    )

    try:
        from langchain_community.vectorstores import Chroma

        from .embeddings import get_embeddings

        vectorstore = Chroma(
            collection_name="casino_knowledge",
            embedding_function=get_embeddings(use_vertex=False),
            persist_directory=persist_dir,
        )
        _retriever_instance = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        logger.info("Loaded ChromaDB retriever from %s", persist_dir)
    except Exception:
        logger.warning(
            "Could not load ChromaDB from %s. Retriever will return empty results.",
            persist_dir,
        )
        _retriever_instance = CasinoKnowledgeRetriever()

    return _retriever_instance


# ---------------------------------------------------------------------------
# Agent Integration — Tool
# ---------------------------------------------------------------------------


@tool
def search_knowledge_base(query: str, category: str | None = None) -> str:
    """Search the casino knowledge base for relevant information.

    Performs semantic search over indexed casino operations, regulations,
    player psychology, and company context documents.

    Args:
        query: The search query in natural language. Examples:
            "How is theoretical win calculated?"
            "Nevada self-exclusion requirements"
            "VIP player reinvestment best practices"
        category: Optional category filter. One of:
            "casino-operations", "regulations", "player-psychology",
            "company-context". If None, searches all categories.

    Returns:
        A formatted string with the top relevant passages and their sources.
    """
    retriever = get_retriever()
    results = retriever.retrieve(
        query, top_k=5, filter_category=category
    )

    if not results:
        return (
            f"No relevant information found for query: '{query}'. "
            f"The knowledge base may not be indexed yet. "
            f"Run the indexing pipeline first."
        )

    formatted_parts: list[str] = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        formatted_parts.append(
            f"[{i}] Source: {source}\n{doc.page_content}\n"
        )

    return "\n---\n".join(formatted_parts)


# ---------------------------------------------------------------------------
# Agent Integration — Node
# ---------------------------------------------------------------------------


def rag_retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node that retrieves relevant context for the current query.

    Extracts the latest user message, performs retrieval, and injects the
    results into the state for the agent_node to use.

    Args:
        state: The current agent state dict.

    Returns:
        Partial state update with retrieved context in player_context or
        as a system message.
    """
    from langchain_core.messages import SystemMessage

    messages = state.get("messages", [])
    if not messages:
        return {}

    # Get the latest human message
    query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not query:
        return {}

    retriever = get_retriever()
    results = retriever.retrieve(query, top_k=3)

    if not results:
        return {}

    # Format context for the LLM
    context_parts = [doc.page_content for doc in results]
    context_str = "\n\n".join(context_parts)

    return {
        "messages": [
            SystemMessage(
                content=(
                    f"[Retrieved Context from Knowledge Base]\n"
                    f"{context_str}"
                ),
                name="rag_retriever",
            )
        ],
    }
