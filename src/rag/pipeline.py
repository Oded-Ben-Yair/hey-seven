"""RAG pipeline: ingestion and retrieval for property knowledge base.

Loads a property JSON file, chunks the content, embeds with Google GenAI,
and stores in ChromaDB for local vector search.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings

from .embeddings import get_embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON -> Text Conversion
# ---------------------------------------------------------------------------


def _format_restaurant(item: dict[str, Any]) -> str:
    """Convert a restaurant JSON object to a readable text chunk."""
    parts = [f"{item.get('name', 'Unknown')}: {item.get('cuisine', '')} cuisine."]
    if item.get("price_range"):
        parts.append(f"Price range: {item['price_range']}.")
    if item.get("location"):
        parts.append(f"Located in {item['location']}.")
    if item.get("hours"):
        parts.append(f"Hours: {item['hours']}.")
    if item.get("description"):
        parts.append(item["description"])
    if item.get("dress_code"):
        parts.append(f"Dress code: {item['dress_code']}.")
    if item.get("reservations"):
        parts.append(f"Reservations: {item['reservations']}.")
    return " ".join(parts)


def _format_entertainment(item: dict[str, Any]) -> str:
    """Convert an entertainment JSON object to a readable text chunk."""
    parts = [f"{item.get('name', 'Unknown')}:"]
    if item.get("type"):
        parts.append(f"{item['type']}.")
    if item.get("description"):
        parts.append(item["description"])
    if item.get("venue"):
        parts.append(f"Venue: {item['venue']}.")
    if item.get("capacity"):
        parts.append(f"Capacity: {item['capacity']}.")
    if item.get("schedule"):
        parts.append(f"Schedule: {item['schedule']}.")
    return " ".join(parts)


def _format_hotel(item: dict[str, Any]) -> str:
    """Convert a hotel room JSON object to a readable text chunk."""
    parts = [f"{item.get('name', 'Unknown')} room:"]
    if item.get("description"):
        parts.append(item["description"])
    if item.get("size"):
        parts.append(f"Size: {item['size']}.")
    if item.get("rate"):
        parts.append(f"Rate: {item['rate']}.")
    if item.get("amenities"):
        amenities = item["amenities"]
        if isinstance(amenities, list):
            parts.append(f"Amenities: {', '.join(amenities)}.")
        else:
            parts.append(f"Amenities: {amenities}.")
    return " ".join(parts)


def _format_generic(item: dict[str, Any]) -> str:
    """Convert a generic JSON object to a readable text chunk."""
    parts = []
    name = item.get("name", item.get("title", ""))
    if name:
        parts.append(f"{name}:")
    if item.get("description"):
        parts.append(item["description"])
    # Include all string/numeric values that aren't already covered
    skip_keys = {"name", "title", "description"}
    for key, value in item.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float)) and value:
            parts.append(f"{key.replace('_', ' ').title()}: {value}.")
        elif isinstance(value, list) and value:
            parts.append(f"{key.replace('_', ' ').title()}: {', '.join(str(v) for v in value)}.")
    return " ".join(parts)


_FORMATTERS = {
    "restaurants": _format_restaurant,
    "dining": _format_restaurant,
    "entertainment": _format_entertainment,
    "shows": _format_entertainment,
    "hotel": _format_hotel,
    "rooms": _format_hotel,
    "hotel_rooms": _format_hotel,
}


def _flatten_nested_dict(d: dict[str, Any], category: str) -> list[dict[str, Any]]:
    """Flatten a nested dict into a list of document-friendly dicts.

    Handles structures like hotel (with towers, room_types sub-lists)
    and gaming (with poker_room, sportsbook sub-dicts).
    """
    items: list[dict[str, Any]] = []
    scalar_parts = []

    for key, value in d.items():
        if isinstance(value, list):
            # Sub-list: each element becomes its own document
            for sub_item in value:
                if isinstance(sub_item, dict):
                    if "name" not in sub_item:
                        sub_item = {**sub_item, "name": key.replace("_", " ").title()}
                    items.append(sub_item)
                else:
                    scalar_parts.append(f"{key.replace('_', ' ').title()}: {sub_item}")
        elif isinstance(value, dict):
            # Sub-dict: convert to a flat item with the key as the name
            flat = {"name": key.replace("_", " ").title(), **value}
            items.append(flat)
        elif isinstance(value, (str, int, float, bool)):
            scalar_parts.append(f"{key.replace('_', ' ').title()}: {value}")

    # Collect top-level scalars into one overview document
    if scalar_parts:
        items.insert(0, {
            "name": f"{category.replace('_', ' ').title()} Overview",
            "description": ". ".join(str(p) for p in scalar_parts) + ".",
        })

    return items if items else [d]


def _load_property_json(data_path: str | Path) -> list[dict[str, Any]]:
    """Load a property JSON file and convert to document dicts.

    Args:
        data_path: Path to the property JSON file.

    Returns:
        List of dicts with keys: content (str), metadata (dict).
    """
    settings = get_settings()
    path = Path(data_path)
    if not path.exists():
        logger.warning("Property data file not found: %s", path)
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    documents: list[dict[str, Any]] = []

    # Handle both flat dict-of-lists and nested structures
    if isinstance(data, dict):
        for category, items in data.items():
            if isinstance(items, list):
                # List of items (restaurants, entertainment, etc.)
                item_list = items
            elif isinstance(items, dict):
                # Nested dict (hotel, gaming, property) -- flatten sub-keys
                item_list = _flatten_nested_dict(items, category)
            else:
                item_list = [{"description": str(items)}]

            formatter = _FORMATTERS.get(category, _format_generic)
            for item in item_list:
                if isinstance(item, str):
                    text = item
                    item_name = category
                elif isinstance(item, dict):
                    text = formatter(item)
                    item_name = item.get("name", item.get("title", category))
                else:
                    continue

                if text.strip():
                    documents.append({
                        "content": text,
                        "metadata": {
                            "category": category,
                            "item_name": item_name,
                            "source": path.name,
                            "property_id": settings.PROPERTY_NAME.lower().replace(" ", "_"),
                            "last_updated": datetime.now(tz=timezone.utc).isoformat(),
                        },
                    })

    elif isinstance(data, list):
        for item in data:
            text = _format_generic(item) if isinstance(item, dict) else str(item)
            if text.strip():
                documents.append({
                    "content": text,
                    "metadata": {
                        "category": "general",
                        "item_name": item.get("name", "unknown") if isinstance(item, dict) else "unknown",
                        "source": path.name,
                        "property_id": settings.PROPERTY_NAME.lower().replace(" ", "_"),
                        "last_updated": datetime.now(tz=timezone.utc).isoformat(),
                    },
                })

    logger.info("Loaded %d items from %s", len(documents), path.name)
    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """Split documents into chunks for embedding.

    Args:
        documents: List of document dicts from _load_property_json().
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunk dicts with content and enriched metadata.
    """
    settings = get_settings()
    chunk_size = chunk_size if chunk_size is not None else settings.RAG_CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.RAG_CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[dict[str, Any]] = []
    for doc in documents:
        content = doc["content"]
        # Skip splitting for items already below chunk size — per-item chunks
        # from category-specific formatters are typically 200-400 chars and
        # should not be fragmented at arbitrary boundaries.
        if len(content) <= chunk_size:
            chunks.append({
                "content": content,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": 0,
                },
            })
        else:
            text_chunks = splitter.split_text(content)
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                    },
                })

    logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_property(
    data_path: str | None = None,
    persist_dir: str | None = None,
) -> Any:
    """Load property JSON, chunk, embed, and store in ChromaDB.

    Args:
        data_path: Path to the property JSON file.
        persist_dir: Directory to persist ChromaDB data.

    Returns:
        A Chroma vectorstore instance.
    """
    # Lazy import: chromadb is a heavy dependency (~200MB). Importing at module
    # level would slow down test collection and any code that imports src.rag.
    from langchain_community.vectorstores import Chroma

    settings = get_settings()
    data_path = data_path or settings.PROPERTY_DATA_PATH
    persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR

    documents = _load_property_json(data_path)
    if not documents:
        logger.warning("No documents to ingest.")
        return None

    chunks = _chunk_documents(documents)

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    # Deterministic IDs prevent duplicate chunks on re-ingestion.
    # Each ID is a SHA-256 hash of content + source metadata, so
    # the same chunk always maps to the same ID regardless of run.
    ids = [
        hashlib.sha256(
            (text + str(meta.get("source", ""))).encode()
        ).hexdigest()
        for text, meta in zip(texts, metadatas)
    ]

    embeddings = get_embeddings()

    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        collection_name="property_knowledge",
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

    logger.info(
        "Indexed %d chunks into ChromaDB at %s.",
        len(chunks),
        persist_dir,
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class CasinoKnowledgeRetriever:
    """Semantic search retriever for property knowledge.

    Wraps a ChromaDB vectorstore and provides retrieval with optional
    category filtering.
    """

    def __init__(self, vectorstore: Any = None) -> None:
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_category: str | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Number of results to return.
            filter_category: Optional category filter.

        Returns:
            List of LangChain Document objects.
        """
        if self.vectorstore is None:
            logger.warning("No vectorstore configured.")
            return []

        search_kwargs: dict[str, Any] = {"k": top_k}
        if filter_category:
            search_kwargs["filter"] = {"category": filter_category}

        return self.vectorstore.similarity_search(query, **search_kwargs)

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with normalized relevance scores (0-1, higher = more relevant).

        Uses ``similarity_search_with_relevance_scores`` which normalizes the
        underlying distance metric to a [0, 1] relevance range.  The collection
        is configured with ``hnsw:space=cosine`` (set during ingestion), so
        scores represent cosine similarity directly.  This ensures that
        downstream filtering with ``score >= threshold`` is always correct.

        Structural grounding: results are filtered by ``property_id`` metadata
        to ensure only documents from the configured property are returned.
        This prevents cross-property leakage if multiple properties share a
        ChromaDB collection.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of (Document, relevance_score) tuples where 1.0 = exact match.
        """
        if self.vectorstore is None:
            logger.warning("No vectorstore configured.")
            return []

        # Structural grounding: only return documents for the configured property
        property_id = get_settings().PROPERTY_NAME.lower().replace(" ", "_")
        return self.vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k, filter={"property_id": property_id},
        )


# ---------------------------------------------------------------------------
# Global Retriever Instance (lazy singleton via @lru_cache)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_retriever(persist_dir: str | None = None) -> CasinoKnowledgeRetriever:
    """Get or create the global retriever singleton.

    Uses ``@lru_cache`` for consistency with other singletons in the codebase
    (``get_settings``, ``_get_llm``, ``_get_circuit_breaker``, ``get_embeddings``).

    When ``VECTOR_DB=firestore``, returns a ``FirestoreRetriever`` (same
    interface shape).  Otherwise falls back to ChromaDB (local dev default).

    Initialized during FastAPI lifespan (before requests arrive).

    Args:
        persist_dir: ChromaDB persistence directory (ignored for Firestore).

    Returns:
        A retriever with ``retrieve()`` and ``retrieve_with_scores()`` methods.
    """
    settings = get_settings()

    if settings.VECTOR_DB == "firestore":
        try:
            from src.rag.firestore_retriever import FirestoreRetriever

            retriever = FirestoreRetriever(
                project=settings.FIRESTORE_PROJECT,
                collection=settings.FIRESTORE_COLLECTION,
                embeddings=get_embeddings(),
            )
            logger.info("Using Firestore retriever (project=%s)", settings.FIRESTORE_PROJECT)
            return retriever  # type: ignore[return-value]
        except Exception:
            logger.warning(
                "Failed to create Firestore retriever. Falling back to ChromaDB.",
                exc_info=True,
            )

    chroma_dir = persist_dir or settings.CHROMA_PERSIST_DIR

    try:
        # Lazy import: see ingest_property() for rationale.
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma(
            collection_name="property_knowledge",
            embedding_function=get_embeddings(),
            persist_directory=chroma_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        logger.info("Loaded ChromaDB retriever from %s", chroma_dir)
    except Exception:
        logger.warning(
            "Could not load ChromaDB from %s. Retriever returns empty results.",
            chroma_dir,
            exc_info=True,
        )
        retriever = CasinoKnowledgeRetriever()

    return retriever
