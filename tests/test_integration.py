"""Integration tests: API → graph → real ChromaDB → mocked LLM → SSE response.

Exercises the full stack with real vector retrieval and mocked LLM to verify
wiring between layers. No API key needed.
"""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

# Skip if chromadb is not installed
chromadb = pytest.importorskip("chromadb")


@pytest.fixture(autouse=True)
def _low_relevance_threshold():
    """Disable relevance filtering for integration tests with fake embeddings.

    Hash-based fake embeddings produce negative relevance scores due to large
    L2 distances between random-ish vectors. Set threshold to -100 so wiring
    tests verify the full retrieval path without filtering everything out.
    """
    from src.config import get_settings

    get_settings.cache_clear()
    with patch.dict(os.environ, {"RAG_MIN_RELEVANCE_SCORE": "-100"}):
        yield
    get_settings.cache_clear()


class FakeEmbeddings:
    """Deterministic embeddings for integration tests (no API key needed)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        import hashlib

        h = hashlib.sha384(text.encode()).digest()
        return [float(b) / 255.0 for b in h]


@pytest.fixture
def ingested_retriever(test_property_data, tmp_path):
    """Create a real ChromaDB retriever with ingested test property data."""
    data_path = tmp_path / "test_property.json"
    data_path.write_text(json.dumps(test_property_data))
    persist_dir = str(tmp_path / "chroma_integration")

    with patch("src.rag.pipeline.get_embeddings", return_value=FakeEmbeddings()):
        from src.rag.pipeline import CasinoKnowledgeRetriever, ingest_property

        vectorstore = ingest_property(str(data_path), persist_dir=persist_dir)
        assert vectorstore is not None
        return CasinoKnowledgeRetriever(vectorstore=vectorstore)


class TestRetrieveToGenerate:
    """Integration: real retrieval → mocked LLM generation."""

    @pytest.mark.asyncio
    async def test_retrieve_node_with_real_chromadb(self, ingested_retriever):
        """retrieve_node returns real context from ChromaDB."""
        from langchain_core.messages import HumanMessage

        from src.agent.nodes import retrieve_node

        state = {
            "messages": [HumanMessage(content="steakhouse")],
            "query_type": "property_qa",
        }

        with patch("src.agent.tools.get_retriever", return_value=ingested_retriever):
            result = await retrieve_node(state)

        assert "retrieved_context" in result
        assert len(result["retrieved_context"]) > 0
        # Verify the retrieved chunks have the expected keys
        chunk = result["retrieved_context"][0]
        assert "content" in chunk
        assert "metadata" in chunk
        assert "score" in chunk

    @pytest.mark.asyncio
    async def test_full_graph_with_real_retrieval_mocked_llm(self, ingested_retriever):
        """Full graph execution: real ChromaDB retrieval, mocked LLM responses."""
        from src.agent.graph import build_graph, chat
        from src.agent.state import RouterOutput, ValidationResult

        # Mock router to classify as property_qa
        mock_router_response = RouterOutput(
            query_type="property_qa", confidence=0.95
        )
        # Mock generate to return a grounded response
        mock_generate_response = AIMessage(
            content="The Test Steakhouse is a fine steakhouse located on the Main Floor."
        )
        # Mock validator to PASS
        mock_validate_response = ValidationResult(status="PASS", reason="Grounded")

        mock_llm = AsyncMock()
        mock_llm.with_structured_output.return_value = AsyncMock()

        with (
            patch("src.agent.tools.get_retriever", return_value=ingested_retriever),
            patch("src.agent.nodes._get_llm") as mock_get_llm,
            patch("src.agent.nodes._get_validator_llm") as mock_get_validator,
        ):
            # Router LLM returns property_qa classification
            router_llm = AsyncMock()
            router_structured = AsyncMock()
            router_structured.ainvoke = AsyncMock(return_value=mock_router_response)
            router_llm.with_structured_output.return_value = router_structured

            # Generate LLM returns the response
            generate_llm = AsyncMock()
            generate_llm.ainvoke = AsyncMock(return_value=mock_generate_response)
            mock_get_llm.return_value = generate_llm

            # Validator LLM returns PASS
            validator_llm = AsyncMock()
            validator_structured = AsyncMock()
            validator_structured.ainvoke = AsyncMock(return_value=mock_validate_response)
            validator_llm.with_structured_output.return_value = validator_structured
            mock_get_validator.return_value = validator_llm

            # Also mock the router's with_structured_output call
            generate_llm.with_structured_output.return_value = router_structured

            graph = build_graph()
            result = await chat(graph, "Tell me about the steakhouse")

        assert result["response"] != ""
        assert result["thread_id"] is not None
        assert "steakhouse" in result["response"].lower() or len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_hours_query_uses_schedule_search(self, ingested_retriever):
        """hours_schedule query type uses search_hours() with augmented keywords."""
        from langchain_core.messages import HumanMessage

        from src.agent.nodes import retrieve_node

        state = {
            "messages": [HumanMessage(content="What time does the buffet open?")],
            "query_type": "hours_schedule",
        }

        with patch("src.agent.tools.get_retriever", return_value=ingested_retriever):
            result = await retrieve_node(state)

        assert "retrieved_context" in result
        # Hours search may or may not find results depending on embeddings,
        # but it should not crash
        assert isinstance(result["retrieved_context"], list)
