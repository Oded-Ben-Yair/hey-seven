"""Live integration tests for profiling extraction with actual Gemini API.

R85: Verify that ProfileExtractionOutput schema works with Gemini 3.x Flash
and that the extraction prompt actually captures volunteered information.

These tests call the real Gemini API — they are slow (~5s each) and require
GOOGLE_API_KEY. Run with: pytest tests/test_profiling_live.py -m live -v
"""

import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    ),
]


@pytest.mark.asyncio
async def test_extraction_schema_gemini_flash():
    """Verify ProfileExtractionOutput schema is accepted by Gemini Flash."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.agent.profiling import ProfileExtractionOutput

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=1.0,
        max_output_tokens=512,
    )
    extraction_llm = llm.with_structured_output(ProfileExtractionOutput)
    result = await extraction_llm.ainvoke(
        "Extract guest info: Guest says 'Hi, I'm Sarah, party of 4, here for my birthday'"
    )
    assert isinstance(result, ProfileExtractionOutput)
    # Should capture at least name and occasion from this explicit message
    assert result.guest_name is not None, "Should extract guest_name from 'I'm Sarah'"
    assert result.occasion is not None, "Should extract occasion from 'birthday'"


@pytest.mark.asyncio
async def test_extraction_captures_volunteered_info():
    """Verify extraction captures explicitly volunteered information."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.agent.profiling import ProfileExtractionOutput, _EXTRACTION_PROMPT

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=1.0,
        max_output_tokens=512,
    )
    extraction_llm = llm.with_structured_output(ProfileExtractionOutput)

    prompt = _EXTRACTION_PROMPT.format(
        user_message="Hi there! I'm Mike, we're a group of 6 celebrating my wife's birthday. We flew in from Boston this morning and we're staying 3 nights.",
        ai_response="Welcome Mike! Happy birthday to your wife! Let me help make this celebration special.",
        current_profile="{}",
    )
    result: ProfileExtractionOutput = await extraction_llm.ainvoke(prompt)
    assert isinstance(result, ProfileExtractionOutput)

    # These are all explicitly stated — should be captured
    extracted_count = sum(
        1
        for field in ("guest_name", "party_size", "occasion", "home_market", "visit_duration")
        if getattr(result, field, None) is not None
    )
    assert extracted_count >= 3, (
        f"Should extract at least 3/5 explicitly stated fields, got {extracted_count}. "
        f"name={result.guest_name}, size={result.party_size}, "
        f"occasion={result.occasion}, home={result.home_market}, "
        f"duration={result.visit_duration}"
    )
