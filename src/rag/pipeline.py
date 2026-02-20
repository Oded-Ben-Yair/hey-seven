"""RAG pipeline: ingestion and retrieval for property knowledge base.

Loads a property JSON file and knowledge-base markdown files, chunks the
content, embeds with Google GenAI, and stores in ChromaDB for local
vector search.
"""

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
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
    """Convert a generic JSON object to a readable text chunk.

    Applies human-readable formatting for boolean and numeric values:
    booleans become "Available"/"Not Available", large numbers get
    comma formatting with contextual units.
    """
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
        label = key.replace("_", " ").title()
        if isinstance(value, bool):
            parts.append(f"{label}: {'Available' if value else 'Not Available'}.")
        elif isinstance(value, int) and value > 9999:
            # Large numbers: comma-format with contextual unit
            unit = " sq ft" if "sqft" in key.lower() or "area" in key.lower() else ""
            parts.append(f"{label}: {value:,}{unit}.")
        elif isinstance(value, (str, int, float)) and value:
            parts.append(f"{label}: {value}.")
        elif isinstance(value, list) and value:
            parts.append(f"{label}: {', '.join(str(v) for v in value)}.")
    return " ".join(parts)


def _format_faq(item: dict[str, Any]) -> str:
    """Convert an FAQ JSON object to a Q&A text chunk.

    Leads with the question text for better cosine alignment with user
    queries (which are themselves questions).
    """
    question = item.get("question", "")
    answer = item.get("answer", "")
    if question and answer:
        return f"{question} {answer}"
    # Fall back to generic if missing Q&A structure
    return _format_generic(item)


def _format_gaming(item: dict[str, Any]) -> str:
    """Convert a gaming JSON object to a readable text chunk."""
    parts = []
    name = item.get("name", "")
    if name:
        parts.append(f"{name}:")
    if item.get("description"):
        parts.append(item["description"])
    # Boolean features rendered as available/not available
    for key in ("poker_room", "sportsbook", "high_limit_area"):
        if key in item:
            label = key.replace("_", " ").title()
            val = item[key]
            if isinstance(val, bool):
                parts.append(f"{label}: {'Available' if val else 'Not Available'}.")
            elif isinstance(val, dict):
                # Sub-dict: include details inline
                details = ", ".join(f"{k}: {v}" for k, v in val.items() if v)
                parts.append(f"{label}: {details}.")
    # Numeric stats with comma formatting
    for key in ("slot_machines", "table_games", "casino_size_sqft"):
        if key in item and isinstance(item[key], (int, float)):
            label = key.replace("_", " ").title()
            unit = " sq ft" if "sqft" in key else ""
            parts.append(f"{label}: {item[key]:,}{unit}.")
    if item.get("hours"):
        parts.append(f"Hours: {item['hours']}.")
    # Catch remaining keys not yet covered
    covered = {"name", "description", "poker_room", "sportsbook", "high_limit_area",
               "slot_machines", "table_games", "casino_size_sqft", "hours"}
    for key, value in item.items():
        if key in covered:
            continue
        if isinstance(value, (str, int, float)) and value:
            parts.append(f"{key.replace('_', ' ').title()}: {value}.")
        elif isinstance(value, list) and value:
            parts.append(f"{key.replace('_', ' ').title()}: {', '.join(str(v) for v in value)}.")
    return " ".join(parts)


def _format_amenity(item: dict[str, Any]) -> str:
    """Convert an amenity JSON object to a readable text chunk."""
    parts = []
    name = item.get("name", item.get("title", ""))
    if name:
        parts.append(f"{name}:")
    if item.get("type"):
        parts.append(f"{item['type']}.")
    if item.get("description"):
        parts.append(item["description"])
    if item.get("hours"):
        parts.append(f"Hours: {item['hours']}.")
    if item.get("location"):
        parts.append(f"Location: {item['location']}.")
    # Remaining fields
    covered = {"name", "title", "type", "description", "hours", "location"}
    for key, value in item.items():
        if key in covered:
            continue
        if isinstance(value, (str, int, float)) and value:
            parts.append(f"{key.replace('_', ' ').title()}: {value}.")
        elif isinstance(value, list) and value:
            parts.append(f"{key.replace('_', ' ').title()}: {', '.join(str(v) for v in value)}.")
    return " ".join(parts)


def _format_promotion(item: dict[str, Any]) -> str:
    """Convert a promotion/loyalty program JSON object to a readable text chunk."""
    parts = []
    name = item.get("name", item.get("tier", item.get("title", "")))
    if name:
        parts.append(f"{name}:")
    if item.get("description"):
        parts.append(item["description"])
    if item.get("benefits"):
        benefits = item["benefits"]
        if isinstance(benefits, list):
            parts.append(f"Benefits: {', '.join(str(b) for b in benefits)}.")
        else:
            parts.append(f"Benefits: {benefits}.")
    if item.get("requirements"):
        parts.append(f"Requirements: {item['requirements']}.")
    if item.get("how_to_join"):
        parts.append(f"How to join: {item['how_to_join']}.")
    # Remaining fields
    covered = {"name", "tier", "title", "description", "benefits", "requirements", "how_to_join"}
    for key, value in item.items():
        if key in covered:
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
    "faq": _format_faq,
    "gaming": _format_gaming,
    "amenities": _format_amenity,
    "promotions": _format_promotion,
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
                            "_schema_version": "2.1",
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
                        "ingestion_version": "2.1",
                    },
                })

    logger.info("Loaded %d items from %s", len(documents), path.name)
    return documents


# ---------------------------------------------------------------------------
# Markdown category mapping: directory name -> metadata category
# ---------------------------------------------------------------------------

_MD_CATEGORY_MAP: dict[str, str] = {
    "casino-operations": "casino_operations",
    "regulations": "regulations",
    "player-psychology": "player_psychology",
    "company-context": "company_context",
}


def _load_knowledge_base_markdown(
    kb_dir: str | Path = "knowledge-base",
) -> list[dict[str, Any]]:
    """Load all markdown files from the knowledge-base directory.

    Splits each file by ``## `` headings so each section becomes its own
    chunk (preserving heading-level context).  Falls back to
    ``RecursiveCharacterTextSplitter`` for files without headings.

    Markdown domain documents (regulations, comp formulas, host workflows)
    complement the structured JSON property data.  They are ingested with
    ``doc_type=markdown`` metadata for provenance tracking.

    Args:
        kb_dir: Path to the knowledge-base directory.

    Returns:
        List of document dicts with keys: content, metadata.
    """
    settings = get_settings()
    base_path = Path(kb_dir)
    if not base_path.exists():
        logger.warning("Knowledge base directory not found: %s", base_path)
        return []

    property_id = settings.PROPERTY_NAME.lower().replace(" ", "_")
    documents: list[dict[str, Any]] = []

    for md_file in sorted(base_path.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        # Determine category from parent directory
        parent_name = md_file.parent.name
        category = _MD_CATEGORY_MAP.get(parent_name, parent_name.replace("-", "_"))

        # Split by ## headings to preserve section-level context
        sections = re.split(r"\n(?=## )", text)

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # Extract heading for item_name
            heading_match = re.match(r"^##?\s+(.+?)(?:\n|$)", section)
            item_name = heading_match.group(1).strip() if heading_match else md_file.stem

            documents.append({
                "content": section,
                "metadata": {
                    "category": category,
                    "item_name": item_name,
                    "source": md_file.name,
                    "property_id": property_id,
                    "last_updated": datetime.now(tz=timezone.utc).isoformat(),
                    "_schema_version": "2.1",
                    "doc_type": "markdown",
                },
            })

    logger.info("Loaded %d sections from %d markdown files in %s",
                len(documents), len(list(base_path.rglob("*.md"))), base_path)
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
            logger.warning(
                "Item '%s' (%d chars) exceeds chunk_size=%d — text splitter activated, "
                "structured context may be fragmented",
                doc["metadata"].get("item_name", "unknown"),
                len(content),
                chunk_size,
            )
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
    knowledge_base_dir: str | None = None,
) -> Any:
    """Load property JSON and knowledge-base markdown, chunk, embed, and store in ChromaDB.

    Args:
        data_path: Path to the property JSON file.
        persist_dir: Directory to persist ChromaDB data.
        knowledge_base_dir: Path to the knowledge-base markdown directory.
            Defaults to ``"knowledge-base"`` relative to the working directory.

    Returns:
        A Chroma vectorstore instance.

    Note:
        SHA-256 content hashing prevents duplicate chunks on re-ingestion.
        An ``_ingestion_version`` timestamp is written to each chunk's
        metadata.  After successful upsert, stale chunks from previous
        ingestion versions (same property, different version stamp) are
        purged automatically.  This prevents ghost data accumulation
        when source content is edited.
    """
    # Lazy import: chromadb is a heavy dependency (~200MB). Importing at module
    # level would slow down test collection and any code that imports src.rag.
    from langchain_community.vectorstores import Chroma

    settings = get_settings()
    data_path = data_path or settings.PROPERTY_DATA_PATH
    persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
    kb_dir = knowledge_base_dir or "knowledge-base"

    # Phase 1: Load structured JSON property data
    documents = _load_property_json(data_path)

    # Phase 2: Load knowledge-base markdown files (domain knowledge)
    md_documents = _load_knowledge_base_markdown(kb_dir)
    documents.extend(md_documents)

    if not documents:
        logger.warning("No documents to ingest (JSON + markdown both empty).")
        return None

    logger.info(
        "Total documents to ingest: %d (JSON: %d, markdown: %d)",
        len(documents),
        len(documents) - len(md_documents),
        len(md_documents),
    )

    chunks = _chunk_documents(documents)

    # Version stamp for this ingestion run (ISO timestamp).
    # Used to identify and purge stale chunks from previous ingestions.
    version_stamp = datetime.now(tz=timezone.utc).isoformat()

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Stamp each chunk with the current ingestion version
    for meta in metadatas:
        meta["_ingestion_version"] = version_stamp

    # Deterministic IDs prevent duplicate chunks on re-ingestion.
    # Each ID is a SHA-256 hash of content + source metadata, so
    # the same chunk always maps to the same ID regardless of run.
    ids = [
        hashlib.sha256(
            (text + str(meta.get("source", ""))).encode()
        ).hexdigest()
        for text, meta in zip(texts, metadatas)
    ]

    # Use RETRIEVAL_DOCUMENT task_type for ingestion embeddings.
    # Google's gemini-embedding-001 produces optimized embeddings when told
    # whether the input is a document or a query (asymmetric search).
    embeddings = get_embeddings(task_type="RETRIEVAL_DOCUMENT")
    property_id = settings.PROPERTY_NAME.lower().replace(" ", "_")

    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        collection_name="property_knowledge",
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Purge stale chunks from previous ingestion versions.
    # After successful upsert, any chunks for the same property_id with
    # a different _ingestion_version are leftover from prior runs where
    # content was edited (new ID) but old ID was never deleted.
    try:
        collection = vectorstore._collection
        old_docs = collection.get(
            where={
                "$and": [
                    {"property_id": {"$eq": property_id}},
                    {"_ingestion_version": {"$ne": version_stamp}},
                ]
            }
        )
        if old_docs and old_docs["ids"]:
            collection.delete(ids=old_docs["ids"])
            logger.info(
                "Purged %d stale chunks from previous ingestion versions.",
                len(old_docs["ids"]),
            )
    except Exception:
        # Purge failure is non-critical — stale data persists but
        # does not corrupt new data.  Log and continue.
        logger.warning(
            "Failed to purge stale chunks (non-critical).",
            exc_info=True,
        )

    logger.info(
        "Indexed %d chunks into ChromaDB at %s (version=%s).",
        len(chunks),
        persist_dir,
        version_stamp[:19],
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class AbstractRetriever(ABC):
    """Abstract interface for property knowledge retrieval.

    Concrete implementations:
    - CasinoKnowledgeRetriever: ChromaDB-backed (local development)
    - FirestoreRetriever: Firestore-backed (GCP production)

    Both provide identical retrieve/retrieve_with_scores interfaces,
    ensuring the graph nodes are backend-agnostic.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_category: str | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents for a query."""
        ...

    @abstractmethod
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents with relevance scores."""
        ...


class CasinoKnowledgeRetriever(AbstractRetriever):
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

        # Multi-tenant isolation: always filter by property_id to prevent
        # cross-property leakage (consistent with retrieve_with_scores()).
        property_id = get_settings().PROPERTY_NAME.lower().replace(" ", "_")
        if filter_category:
            # ChromaDB requires $and for multi-key where clauses
            filter_dict: dict[str, Any] = {
                "$and": [
                    {"property_id": property_id},
                    {"category": filter_category},
                ]
            }
        else:
            filter_dict = {"property_id": property_id}

        return self.vectorstore.similarity_search(query, k=top_k, filter=filter_dict)

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
# Global Retriever Instance (TTLCache singleton, 1-hour refresh for
# GCP Workload Identity credential rotation — consistent with _llm_cache
# and _validator_cache in nodes.py)
# ---------------------------------------------------------------------------

_retriever_cache: dict[str, AbstractRetriever] = {}
_retriever_cache_time: dict[str, float] = {}
_RETRIEVER_TTL_SECONDS = 3600  # 1 hour


def _get_retriever_cached() -> AbstractRetriever:
    """Return a cached retriever using default settings.

    Uses a dict-based TTL cache (1 hour) consistent with LLM singletons.
    Credential rotation in GCP Workload Identity requires periodic
    recreation of Firestore clients.

    Internal helper for ``get_retriever()`` — separated so that the
    ``persist_dir`` override path does not pollute the cache key.
    """
    import time

    settings = get_settings()
    cache_key = f"{settings.CASINO_ID}:default"
    now = time.monotonic()
    if cache_key in _retriever_cache:
        if (now - _retriever_cache_time.get(cache_key, 0)) < _RETRIEVER_TTL_SECONDS:
            return _retriever_cache[cache_key]
        logger.info("Retriever TTL expired, recreating (credential rotation safety).")

    if settings.VECTOR_DB == "firestore":
        try:
            from src.rag.firestore_retriever import FirestoreRetriever

            retriever = FirestoreRetriever(
                project=settings.FIRESTORE_PROJECT,
                collection=settings.FIRESTORE_COLLECTION,
                embeddings=get_embeddings(task_type="RETRIEVAL_QUERY"),
            )
            logger.info("Using Firestore retriever (project=%s)", settings.FIRESTORE_PROJECT)
            _retriever_cache[cache_key] = retriever
            _retriever_cache_time[cache_key] = now
            return retriever
        except Exception:
            logger.warning(
                "Failed to create Firestore retriever. Falling back to ChromaDB.",
                exc_info=True,
            )

    chroma_dir = settings.CHROMA_PERSIST_DIR

    # Guard: VECTOR_DB=chroma requires the chromadb package, which is excluded
    # from requirements-prod.txt (~200MB). If running in production without
    # chromadb, fail with a clear message instead of an opaque ImportError.
    if settings.ENVIRONMENT == "production" and settings.VECTOR_DB == "chroma":
        raise RuntimeError(
            "VECTOR_DB=chroma in production environment. chromadb is excluded "
            "from requirements-prod.txt and loses all data on container restart. "
            "Set VECTOR_DB=firestore for production."
        )

    try:
        # Lazy import: see ingest_property() for rationale.
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma(
            collection_name="property_knowledge",
            embedding_function=get_embeddings(task_type="RETRIEVAL_QUERY"),
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

    _retriever_cache[cache_key] = retriever
    _retriever_cache_time[cache_key] = now
    return retriever


def get_retriever(persist_dir: str | None = None) -> AbstractRetriever:
    """Get or create the global retriever singleton.

    Uses a TTL cache internally for consistency with other singletons
    (``get_settings``, ``get_embeddings``).

    When ``VECTOR_DB=firestore``, returns a ``FirestoreRetriever`` (same
    interface shape).  Otherwise falls back to ChromaDB (local dev default).

    Initialized during FastAPI lifespan (before requests arrive).

    Args:
        persist_dir: Override for ChromaDB directory. When None (default),
            uses ``settings.CHROMA_PERSIST_DIR`` and returns a cached instance.
            When specified, creates a new (uncached) instance -- primarily
            used by tests.

    Returns:
        A retriever with ``retrieve()`` and ``retrieve_with_scores()`` methods.
    """
    if persist_dir is None:
        return _get_retriever_cached()

    # Explicit persist_dir: create a new uncached instance (tests, CLI tools).
    try:
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma(
            collection_name="property_knowledge",
            embedding_function=get_embeddings(task_type="RETRIEVAL_QUERY"),
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        retriever = CasinoKnowledgeRetriever(vectorstore=vectorstore)
        logger.info("Loaded ChromaDB retriever from %s (uncached)", persist_dir)
    except Exception:
        logger.warning(
            "Could not load ChromaDB from %s. Retriever returns empty results.",
            persist_dir,
            exc_info=True,
        )
        retriever = CasinoKnowledgeRetriever()

    return retriever


def clear_retriever_cache() -> None:
    """Clear the retriever singleton cache.

    Clears the TTL-based retriever cache dict.
    Call from tests or after re-ingestion to force fresh retriever creation.
    """
    _retriever_cache.clear()
    _retriever_cache_time.clear()


# Backward-compatible attribute: callers using ``get_retriever.cache_clear()``
# are redirected to the standalone ``clear_retriever_cache()`` function.
get_retriever.cache_clear = clear_retriever_cache  # type: ignore[attr-defined]
