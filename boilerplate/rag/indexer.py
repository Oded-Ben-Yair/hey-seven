"""Document indexer for the casino knowledge base.

Loads markdown files from knowledge-base/, chunks them with
RecursiveCharacterTextSplitter, embeds with Vertex AI, and stores in
a vector database (ChromaDB for local dev, Vertex AI Vector Search for
production).
"""

import logging
import os
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------


def load_knowledge_base(
    kb_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load all markdown files from the knowledge base directory.

    Scans the knowledge-base/ directory recursively for .md files and
    extracts content with metadata (source file, section, category).

    Args:
        kb_path: Path to the knowledge base directory. If None, uses
            the default relative to project root.

    Returns:
        List of document dicts with keys:
            - content: The markdown text content.
            - metadata: Dict with source, category, title.
    """
    if kb_path is None:
        # Default: project_root/knowledge-base/
        kb_path = Path(__file__).resolve().parent.parent.parent / "knowledge-base"
    else:
        kb_path = Path(kb_path)

    if not kb_path.exists():
        logger.warning("Knowledge base path does not exist: %s", kb_path)
        return []

    documents: list[dict[str, Any]] = []

    for md_file in sorted(kb_path.rglob("*.md")):
        relative_path = md_file.relative_to(kb_path)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "general"

        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue

        # Extract title from first heading
        title = relative_path.stem
        for line in content.split("\n"):
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": str(relative_path),
                    "category": category,
                    "title": title,
                    "file_path": str(md_file),
                },
            }
        )
        logger.info("Loaded: %s (%d chars)", relative_path, len(content))

    logger.info("Loaded %d documents from knowledge base.", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """Split documents into chunks for embedding.

    Uses RecursiveCharacterTextSplitter which respects markdown structure
    (headers, paragraphs, sentences) when splitting.

    Args:
        documents: List of document dicts from load_knowledge_base().
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunk dicts with content and enriched metadata (includes
        chunk_index and parent document reference).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",   # Markdown H2 headers
            "\n### ",  # Markdown H3 headers
            "\n\n",    # Paragraphs
            "\n",      # Lines
            ". ",      # Sentences
            " ",       # Words
        ],
        length_function=len,
    )

    chunks: list[dict[str, Any]] = []

    for doc in documents:
        text_chunks = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(text_chunks):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    },
                }
            )

    logger.info(
        "Split %d documents into %d chunks (size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Vector Store — ChromaDB (Local Dev)
# ---------------------------------------------------------------------------


def index_to_chroma(
    chunks: list[dict[str, Any]],
    collection_name: str = "casino_knowledge",
    persist_directory: str | None = None,
) -> Any:
    """Index document chunks into a ChromaDB collection.

    ChromaDB is used for local development. For production, use
    index_to_vertex_vector_search().

    Args:
        chunks: List of chunk dicts from chunk_documents().
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory to persist the ChromaDB data.
            If None, uses an in-memory store.

    Returns:
        A Chroma vectorstore instance.
    """
    from langchain_community.vectorstores import Chroma

    embeddings = get_embeddings(use_vertex=False)

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "embedding_function": embeddings,
    }
    if persist_directory:
        kwargs["persist_directory"] = persist_directory

    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        **kwargs,
    )

    logger.info(
        "Indexed %d chunks into ChromaDB collection '%s'.",
        len(chunks),
        collection_name,
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Vector Store — Vertex AI Vector Search (Production)
# ---------------------------------------------------------------------------


def index_to_vertex(
    chunks: list[dict[str, Any]],
    index_endpoint_name: str | None = None,
    deployed_index_id: str | None = None,
) -> None:
    """Index document chunks into Vertex AI Vector Search.

    Uses the ``IndexDatapoint`` proto objects required by
    ``aiplatform >= 1.60.0``. Previous dict-based format is deprecated.

    Requires a pre-created Vector Search index and endpoint. See GCP
    documentation for index creation:
    https://cloud.google.com/vertex-ai/docs/vector-search/overview

    Args:
        chunks: List of chunk dicts from chunk_documents().
        index_endpoint_name: Full resource name of the Vector Search index
            endpoint. Read from VERTEX_INDEX_ENDPOINT env var if None.
        deployed_index_id: The deployed index ID within the endpoint.
            Read from VERTEX_DEPLOYED_INDEX_ID env var if None.
    """
    from google.cloud import aiplatform  # type: ignore[import-untyped]
    from google.cloud.aiplatform.matching_engine import (  # type: ignore[import-untyped]
        matching_engine_index_endpoint,
    )

    from .embeddings import batch_embed

    index_endpoint_name = index_endpoint_name or os.environ.get(
        "VERTEX_INDEX_ENDPOINT"
    )
    deployed_index_id = deployed_index_id or os.environ.get(
        "VERTEX_DEPLOYED_INDEX_ID"
    )

    if not index_endpoint_name or not deployed_index_id:
        raise ValueError(
            "VERTEX_INDEX_ENDPOINT and VERTEX_DEPLOYED_INDEX_ID must be set."
        )

    texts = [c["content"] for c in chunks]
    embeddings_model = get_embeddings(use_vertex=True)
    vectors = batch_embed(texts, embeddings=embeddings_model)

    # Build IndexDatapoint proto objects (aiplatform >= 1.60.0 format, C9 fix)
    datapoints = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        restricts = [
            matching_engine_index_endpoint.Namespace(
                name="category",
                allow_tokens=[chunk["metadata"].get("category", "general")],
            )
        ]
        datapoint = matching_engine_index_endpoint.IndexDatapoint(
            datapoint_id=f"chunk_{i}",
            feature_vector=vector,
            restricts=restricts,
        )
        datapoints.append(datapoint)

    # Upsert to the index
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=index_endpoint_name
    )

    # Batch upsert (Vertex AI handles batching internally)
    logger.info("Upserting %d vectors to Vertex AI Vector Search...", len(datapoints))
    index_endpoint.upsert_datapoints(
        deployed_index_id=deployed_index_id,
        datapoints=datapoints,
    )
    logger.info("Upsert complete.")


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


def run_indexing_pipeline(
    kb_path: str | Path | None = None,
    use_vertex: bool = False,
    chroma_persist_dir: str | None = None,
) -> Any:
    """Run the complete indexing pipeline: load -> chunk -> embed -> store.

    Args:
        kb_path: Path to knowledge base directory.
        use_vertex: If True, index to Vertex AI Vector Search. Otherwise,
            use ChromaDB.
        chroma_persist_dir: ChromaDB persistence directory (local dev only).

    Returns:
        The vectorstore instance (ChromaDB) or None (Vertex AI).
    """
    documents = load_knowledge_base(kb_path)
    if not documents:
        logger.warning("No documents found. Indexing pipeline skipped.")
        return None

    chunks = chunk_documents(documents)

    if use_vertex:
        index_to_vertex(chunks)
        return None
    else:
        return index_to_chroma(
            chunks,
            persist_directory=chroma_persist_dir,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vectorstore = run_indexing_pipeline()
    if vectorstore:
        # Quick test query
        results = vectorstore.similarity_search("What is theoretical win?", k=3)
        for i, doc in enumerate(results):
            print(f"\n--- Result {i + 1} ---")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
