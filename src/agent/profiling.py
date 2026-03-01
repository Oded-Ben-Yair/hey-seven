"""Guest Profiling Intelligence System — enrichment node for the StateGraph.

Sits between generate and validate nodes. Analyzes the last exchange to:
1. Extract profile fields from the conversation (LLM structured output).
2. Calculate weighted profile completeness.
3. Inject a natural profiling question into the AI response (from whisper plan).

Fail-silent: any error returns empty state update — never crashes the pipeline.

Feature flag: ``profiling_enabled`` (Layer 1 build-time topology).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agent.state import PropertyQAState
from src.config import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "ConfidenceField",
    "ProfileExtractionOutput",
    "profiling_enrichment_node",
    "PROFILING_TECHNIQUE_PROMPTS",
]


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM extraction
# ---------------------------------------------------------------------------


class ConfidenceField(BaseModel):
    """A profile field with confidence and source tracking."""

    value: Any = Field(description="The extracted value")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the extraction (0.0-1.0)",
    )
    source: Literal["explicit", "inferred", "corrected"] = Field(
        default="explicit",
        description="How the value was obtained: explicit (stated), inferred (contextual), corrected (guest corrected a prior value)",
    )


class ProfileExtractionOutput(BaseModel):
    """Structured output from LLM profile extraction."""

    guest_name: ConfidenceField | None = Field(default=None, description="Guest name")
    party_size: ConfidenceField | None = Field(default=None, description="Number in party")
    party_composition: ConfidenceField | None = Field(default=None, description="Adults, kids, ages")
    visit_purpose: ConfidenceField | None = Field(default=None, description="Business, leisure, celebration")
    visit_duration: ConfidenceField | None = Field(default=None, description="Number of nights/days")
    dining_preferences: ConfidenceField | None = Field(default=None, description="Cuisine type, dietary")
    dietary_restrictions: ConfidenceField | None = Field(default=None, description="Allergies, restrictions")
    gaming_preferences: ConfidenceField | None = Field(default=None, description="Game type, stakes")
    entertainment_interests: ConfidenceField | None = Field(default=None, description="Shows, music, events")
    spa_interests: ConfidenceField | None = Field(default=None, description="Treatments, services")
    occasion: ConfidenceField | None = Field(default=None, description="Birthday, anniversary, etc.")
    occasion_details: ConfidenceField | None = Field(default=None, description="Occasion specifics")
    companion_names: ConfidenceField | None = Field(default=None, description="Names of companions")
    companion_ages: ConfidenceField | None = Field(default=None, description="Ages of companions")
    home_market: ConfidenceField | None = Field(default=None, description="Where guest is from")
    budget_signal: ConfidenceField | None = Field(default=None, description="Budget level signals")
    loyalty_tier: ConfidenceField | None = Field(default=None, description="Loyalty program tier")
    visit_frequency: ConfidenceField | None = Field(default=None, description="How often they visit")
    communication_preference: ConfidenceField | None = Field(default=None, description="Text, call, email")


# ---------------------------------------------------------------------------
# Profiling technique prompts (give-to-get, assumptive bridge, etc.)
# ---------------------------------------------------------------------------

PROFILING_TECHNIQUE_PROMPTS: dict[str, str] = {
    "give_to_get": (
        "Share a relevant piece of information about the property first, "
        "then naturally ask a follow-up question. Example: 'Our steakhouse "
        "has been getting great reviews this month — are you more of a steak "
        "or seafood person?'"
    ),
    "assumptive_bridge": (
        "Make a reasonable assumption based on context and bridge to a question. "
        "Example: 'Since you're celebrating, you'll probably want somewhere "
        "special for dinner — how many will be joining you?'"
    ),
    "contextual_inference": (
        "Use context clues to infer information without asking directly. "
        "If the guest mentions kids, infer family composition. If they mention "
        "a late arrival, infer they might need a quick dinner option."
    ),
    "need_payoff": (
        "Frame the question as benefiting the guest. Example: 'To find the "
        "perfect spot for you, would you say you're looking for something "
        "more upscale or casual tonight?'"
    ),
    "incentive_frame": (
        "Tie the question to a tangible benefit. Example: 'We sometimes have "
        "special birthday packages — when is the celebration?'"
    ),
    "reflective_confirm": (
        "Reflect back what you've learned and confirm. Example: 'So it sounds "
        "like a romantic dinner for two on Saturday — is that right?'"
    ),
    "none": "Do not ask any profiling questions this turn.",
}


# ---------------------------------------------------------------------------
# Weighted completeness calculation
# ---------------------------------------------------------------------------

# Weights sum to 1.0. Higher-weight fields are more valuable for personalization.
# Keys use the STORED field names (after _FIELD_NAME_MAP mapping),
# not the ProfileExtractionOutput field names.
_PROFILE_WEIGHTS: dict[str, float] = {
    "name": 0.15,              # mapped from guest_name
    "party_size": 0.10,
    "visit_purpose": 0.08,
    "preferences": 0.08,       # mapped from dining_preferences
    "occasion": 0.08,
    "visit_duration": 0.06,
    "party_composition": 0.06,
    "dietary": 0.06,           # mapped from dietary_restrictions
    "gaming": 0.05,            # mapped from gaming_preferences
    "entertainment": 0.05,     # mapped from entertainment_interests
    "spa": 0.04,               # mapped from spa_interests
    "occasion_details": 0.03,
    "companion_names": 0.03,
    "companion_ages": 0.02,
    "home_market": 0.03,
    "budget_signal": 0.03,
    "loyalty_tier": 0.02,
    "visit_frequency": 0.02,
    "communication_preference": 0.01,
}


def _calculate_profile_completeness_weighted(
    extracted_fields: dict[str, Any],
) -> float:
    """Calculate weighted profile completeness (0.0 to 1.0).

    Fields with higher personalization impact get more weight.
    A profile with name + party_size + occasion scores higher
    than one with companion_ages + communication_preference.

    Args:
        extracted_fields: Accumulated extracted fields dict.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not extracted_fields:
        return 0.0

    score = 0.0
    for field_name, weight in _PROFILE_WEIGHTS.items():
        if extracted_fields.get(field_name):
            score += weight

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Golden path phase determination
# ---------------------------------------------------------------------------


def _determine_profiling_phase(
    extracted_fields: dict[str, Any],
    completeness: float,
) -> str:
    """Determine the current profiling phase based on what we know.

    Golden path: foundation -> preference -> relationship -> behavioral

    Args:
        extracted_fields: Current accumulated profile fields.
        completeness: Current weighted completeness score.

    Returns:
        Phase name string.
    """
    # Foundation: name, party_size, visit_purpose (using stored field names)
    foundation_fields = ("name", "party_size", "visit_purpose")
    has_foundation = sum(1 for f in foundation_fields if extracted_fields.get(f)) >= 2

    if not has_foundation:
        return "foundation"

    # Preference: dining, entertainment, gaming, spa (using stored field names)
    preference_fields = ("preferences", "entertainment", "gaming", "spa")
    has_preferences = sum(1 for f in preference_fields if extracted_fields.get(f)) >= 1

    if not has_preferences:
        return "preference"

    # Relationship: occasion, companion info, visit frequency
    relationship_fields = ("occasion", "companion_names", "visit_frequency", "loyalty_tier")
    has_relationship = sum(1 for f in relationship_fields if extracted_fields.get(f)) >= 1

    if not has_relationship:
        return "relationship"

    return "behavioral"


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a guest profiling assistant for a casino resort concierge.
Analyze the last exchange between the guest and the AI concierge.
Extract any guest profile information that was explicitly stated or
strongly implied.

## Last Guest Message
{user_message}

## Last AI Response
{ai_response}

## Current Profile
{current_profile}

## Instructions
- Only extract information that is explicitly stated or very strongly implied.
- Set confidence to 0.9+ for explicitly stated facts ("My name is Sarah").
- Set confidence to 0.6-0.8 for strongly implied facts ("dinner for our anniversary" implies occasion=anniversary).
- Set confidence below 0.5 for weak inferences (do NOT extract these).
- Use source="corrected" when the guest corrects a prior value.
- Return null for any field where no new information is available.
- Do NOT re-extract information already in the current profile unless the guest corrected it.
"""


# ---------------------------------------------------------------------------
# Profiling enrichment node
# ---------------------------------------------------------------------------


async def profiling_enrichment_node(state: PropertyQAState) -> dict[str, Any]:
    """Profiling enrichment node for the StateGraph.

    Sits between generate and validate. Analyzes the last exchange to:
    1. Extract profile fields via LLM structured output.
    2. Calculate weighted profile completeness.
    3. Inject a natural profiling question into the AI response.

    Fail-silent: any error returns empty state update (never crashes pipeline).

    Args:
        state: Current graph state.

    Returns:
        State update dict with extracted_fields, profiling_phase,
        profile_completeness_score, and optionally modified messages.
    """
    try:
        settings = get_settings()
        min_confidence = settings.PROFILING_MIN_CONFIDENCE
        extracted_fields = dict(state.get("extracted_fields") or {})

        # Get the last user message and last AI response
        messages = state.get("messages", [])
        user_message = ""
        ai_response = ""

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not ai_response:
                ai_response = msg.content if isinstance(msg.content, str) else str(msg.content)
            elif isinstance(msg, HumanMessage) and not user_message:
                user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            if user_message and ai_response:
                break

        if not user_message:
            # No user message to analyze — return current state
            completeness = _calculate_profile_completeness_weighted(extracted_fields)
            phase = _determine_profiling_phase(extracted_fields, completeness)
            return {
                "profiling_phase": phase,
                "profile_completeness_score": completeness,
                "profiling_question_injected": False,
            }

        # Build extraction prompt
        import json

        current_profile_str = json.dumps(extracted_fields, indent=2) if extracted_fields else "{}"
        prompt_text = _EXTRACTION_PROMPT.format(
            user_message=user_message,
            ai_response=ai_response or "(no AI response yet)",
            current_profile=current_profile_str,
        )

        # Call LLM with structured output — reuse whisper planner's singleton
        from src.agent.whisper_planner import _get_whisper_llm

        llm = await _get_whisper_llm()
        extraction_llm = llm.with_structured_output(ProfileExtractionOutput)
        extraction_result: ProfileExtractionOutput = await extraction_llm.ainvoke(prompt_text)

        # Merge high-confidence fields into extracted_fields
        new_fields: dict[str, Any] = {}
        for field_name in ProfileExtractionOutput.model_fields:
            field_value = getattr(extraction_result, field_name, None)
            if field_value is not None and isinstance(field_value, ConfidenceField):
                if field_value.confidence >= min_confidence and field_value.value is not None:
                    # Map profiling field names to extracted_fields keys
                    key = _FIELD_NAME_MAP.get(field_name, field_name)
                    new_fields[key] = field_value.value

        if new_fields:
            logger.info(
                "Profiling extracted %d fields: %s",
                len(new_fields),
                list(new_fields.keys()),
            )

        # Merge new fields (new_fields will be merged by _merge_dicts reducer)
        merged = {**extracted_fields, **new_fields}
        completeness = _calculate_profile_completeness_weighted(merged)
        phase = _determine_profiling_phase(merged, completeness)

        result: dict[str, Any] = {
            "extracted_fields": new_fields,  # reducer merges with existing
            "profiling_phase": phase,
            "profile_completeness_score": completeness,
            "profiling_question_injected": False,
        }

        # Inject profiling question from whisper plan (if available)
        whisper = state.get("whisper_plan")
        if whisper and whisper.get("next_profiling_question"):
            question = whisper["next_profiling_question"]
            technique = whisper.get("question_technique", "none")

            # Only inject if technique is not "none" and we have an AI response to append to
            if technique != "none" and ai_response:
                # Find the last AI message and append the question naturally
                updated_messages = []
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        original = msg.content if isinstance(msg.content, str) else str(msg.content)
                        # Append question with a natural bridge
                        enriched = f"{original}\n\n{question}"
                        updated_messages.append(AIMessage(content=enriched))
                        result["messages"] = updated_messages
                        result["profiling_question_injected"] = True
                        logger.info(
                            "Profiling question injected (technique=%s, phase=%s)",
                            technique,
                            phase,
                        )
                        break

        return result

    except Exception:
        # Fail-silent: any error returns empty state update
        logger.warning("profiling_enrichment_node failed, continuing without profiling", exc_info=True)
        extracted_fields = state.get("extracted_fields") or {}
        completeness = _calculate_profile_completeness_weighted(extracted_fields)
        phase = _determine_profiling_phase(extracted_fields, completeness)
        return {
            "profiling_phase": phase,
            "profile_completeness_score": completeness,
            "profiling_question_injected": False,
        }


# ---------------------------------------------------------------------------
# Field name mapping (profiling model -> extracted_fields keys)
# ---------------------------------------------------------------------------

_FIELD_NAME_MAP: dict[str, str] = {
    "guest_name": "name",
    "dining_preferences": "preferences",
    "dietary_restrictions": "dietary",
    "gaming_preferences": "gaming",
    "entertainment_interests": "entertainment",
    "spa_interests": "spa",
    "visit_purpose": "visit_purpose",
    "visit_duration": "visit_duration",
    "party_composition": "party_composition",
    "occasion_details": "occasion_details",
    "companion_names": "companion_names",
    "companion_ages": "companion_ages",
    "home_market": "home_market",
    "budget_signal": "budget_signal",
    "loyalty_tier": "loyalty_tier",
    "visit_frequency": "visit_frequency",
    "communication_preference": "communication_preference",
    # These map to same name:
    "party_size": "party_size",
    "occasion": "occasion",
}
