"""Tests for LLM-augmented field extraction (B2).

Covers: regex-found skips LLM, empty regex + long text triggers LLM,
short message bypass, LLM failure fallback, merge priority, schema validation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.extraction import (
    ExtractionOutput,
    extract_fields,
    extract_fields_augmented,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(
    name: str | None = "Sarah",
    party_size: int | None = 4,
    visit_date: str | None = "next Saturday",
    occasion: str | None = "birthday",
    preferences: str | None = None,
):
    """Create a mock LLM that returns an ExtractionOutput via structured output."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        return_value=ExtractionOutput(
            name=name,
            party_size=party_size,
            visit_date=visit_date,
            occasion=occasion,
            preferences=preferences,
        )
    )
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


# ---------------------------------------------------------------------------
# ExtractionOutput schema validation
# ---------------------------------------------------------------------------


class TestExtractionOutputSchema:
    """Validate the Pydantic model for structured LLM output."""

    def test_all_fields(self):
        out = ExtractionOutput(
            name="John", party_size=6, visit_date="Friday",
            occasion="anniversary", preferences="vegetarian",
        )
        assert out.name == "John"
        assert out.party_size == 6

    def test_all_none(self):
        out = ExtractionOutput()
        assert out.name is None
        assert out.party_size is None
        assert out.visit_date is None
        assert out.occasion is None
        assert out.preferences is None

    def test_partial_fields(self):
        out = ExtractionOutput(name="Alice", occasion="birthday")
        assert out.name == "Alice"
        assert out.occasion == "birthday"
        assert out.party_size is None


# ---------------------------------------------------------------------------
# Regex found -> LLM skipped
# ---------------------------------------------------------------------------


class TestRegexFoundSkipsLLM:
    """When regex already found fields, LLM should NOT be called."""

    @pytest.mark.asyncio
    async def test_regex_found_name_skips_llm(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        regex_result = {"name": "John"}
        result = await extract_fields_augmented(
            "My name is John and I would like to make a reservation for my birthday party next Saturday for four people",
            regex_result,
            get_llm,
        )
        assert result == {"name": "John"}
        get_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_regex_found_party_size_skips_llm(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        regex_result = {"party_size": 4}
        result = await extract_fields_augmented(
            "There are 4 of us coming to dinner",
            regex_result,
            get_llm,
        )
        assert result == {"party_size": 4}
        get_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Short message bypass
# ---------------------------------------------------------------------------


class TestShortMessageBypass:
    """Messages under 15 words skip LLM even if regex found nothing."""

    @pytest.mark.asyncio
    async def test_short_empty_regex_skips_llm(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        result = await extract_fields_augmented(
            "What restaurants are open?",
            {},
            get_llm,
        )
        assert result == {}
        get_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_14_words_skips_llm(self):
        mock_llm = _make_mock_llm()
        get_llm = AsyncMock(return_value=mock_llm)

        # 14 words (< 15 threshold)
        result = await extract_fields_augmented(
            "I was wondering if you could help me find a good place to eat",
            {},
            get_llm,
        )
        assert result == {}
        get_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Empty regex + long message triggers LLM
# ---------------------------------------------------------------------------


class TestEmptyRegexTriggersLLM:
    """Empty regex + 15+ words should invoke LLM."""

    @pytest.mark.asyncio
    async def test_empty_regex_long_message_triggers_llm(self):
        mock_llm = _make_mock_llm(
            name="Sarah", party_size=4, visit_date="next Saturday", occasion="birthday",
        )
        get_llm = AsyncMock(return_value=mock_llm)

        # 19 words, regex finds nothing (conversational paraphrase)
        result = await extract_fields_augmented(
            "Hey so my friend Sarah is turning thirty and we want to celebrate next Saturday there will be four of us coming along",
            {},
            get_llm,
        )
        assert result["name"] == "Sarah"
        assert result["party_size"] == 4
        assert result["visit_date"] == "next Saturday"
        assert result["occasion"] == "birthday"
        get_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_returns_partial(self):
        mock_llm = _make_mock_llm(name="Mike", party_size=None, visit_date=None, occasion=None)
        get_llm = AsyncMock(return_value=mock_llm)

        result = await extract_fields_augmented(
            "Hi there I am Mike and I was wondering if you could recommend a really great steakhouse for me to try tonight",
            {},
            get_llm,
        )
        assert result == {"name": "Mike"}

    @pytest.mark.asyncio
    async def test_llm_returns_all_none(self):
        mock_llm = _make_mock_llm(
            name=None, party_size=None, visit_date=None, occasion=None, preferences=None,
        )
        get_llm = AsyncMock(return_value=mock_llm)

        result = await extract_fields_augmented(
            "I was just wondering about the general atmosphere and what kind of entertainment options are available at the resort this weekend",
            {},
            get_llm,
        )
        assert result == {}


# ---------------------------------------------------------------------------
# Merge priority: regex wins on conflicts
# ---------------------------------------------------------------------------


class TestMergePriority:
    """Regex results take priority over LLM results on conflicts."""

    @pytest.mark.asyncio
    async def test_regex_wins_on_conflict(self):
        """If both regex and LLM return the same field, regex wins.

        Note: This test verifies the DESIGN that regex_result takes priority.
        In practice, if regex_result is non-empty, LLM is never called.
        The merge logic handles the theoretical case where regex_result is
        passed with fields AND the LLM is invoked (defensive programming).
        """
        # The function won't actually call LLM if regex_result is non-empty,
        # so this test verifies the early-return path
        mock_llm = _make_mock_llm(name="LLMName")
        get_llm = AsyncMock(return_value=mock_llm)

        result = await extract_fields_augmented(
            "My name is John and I want a table",
            {"name": "RegexName"},
            get_llm,
        )
        assert result["name"] == "RegexName"
        get_llm.assert_not_called()


# ---------------------------------------------------------------------------
# LLM failure fallback
# ---------------------------------------------------------------------------


class TestLLMFailureFallback:
    """When LLM fails, extract_fields_augmented returns regex result."""

    @pytest.mark.asyncio
    async def test_llm_exception_returns_regex(self):
        get_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        result = await extract_fields_augmented(
            "Hey so my friend Sarah is turning thirty and we want to celebrate next Saturday there will be four of us coming along",
            {},
            get_llm,
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_llm_invoke_error_returns_regex(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=ValueError("Bad response"))
        mock_llm.with_structured_output.return_value = mock_structured
        get_llm = AsyncMock(return_value=mock_llm)

        result = await extract_fields_augmented(
            "Hey so my friend Sarah is turning thirty and we want to celebrate next Saturday there will be four of us coming along",
            {},
            get_llm,
        )
        assert result == {}


# ---------------------------------------------------------------------------
# Regression: existing extract_fields unchanged
# ---------------------------------------------------------------------------


class TestExtractFieldsRegression:
    """Verify existing extract_fields behavior is unchanged."""

    def test_name_extraction(self):
        result = extract_fields("My name is John, I need a reservation")
        assert result.get("name") == "John"

    def test_party_size(self):
        result = extract_fields("Table for 6 people please")
        assert result.get("party_size") == 6

    def test_occasion(self):
        result = extract_fields("We're celebrating our anniversary")
        assert result.get("occasion") == "anniversary"

    def test_empty_string(self):
        assert extract_fields("") == {}

    def test_none_input(self):
        assert extract_fields(None) == {}
