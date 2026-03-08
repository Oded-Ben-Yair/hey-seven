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

from src.agent.nodes import _normalize_content
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
    """A profile field with confidence and source tracking.

    NOTE: This model is used for internal processing and tests.
    For LLM structured output, use ProfileExtractionOutput (flat schema)
    to avoid Gemini "too many schema states" errors.
    """

    value: Any = Field(description="The extracted value")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the extraction (0.0-1.0)",
    )
    source: Literal["explicit", "inferred", "corrected"] = Field(
        default="explicit",
        description="How the value was obtained: explicit (stated), inferred (contextual), corrected (guest corrected a prior value)",
    )


class ProfileExtractionOutput(BaseModel):
    """Flat structured output from LLM profile extraction.

    R76 fix: Simplified from 19 nested ConfidenceField objects to flat
    string fields. Gemini Flash rejects schemas with too many nested
    states (19 x {value:Any, confidence:float[0,1], source:Literal} =
    schema constraint overflow → 400 INVALID_ARGUMENT).

    Confidence gating is handled in the extraction prompt: the LLM is
    instructed to only return explicitly stated or strongly implied facts.
    """

    guest_name: str | None = Field(default=None, description="Guest name if stated")
    party_size: str | None = Field(
        default=None, description="Number in party if stated"
    )
    party_composition: str | None = Field(
        default=None, description="Adults, kids, ages if mentioned"
    )
    visit_purpose: str | None = Field(
        default=None, description="Business, leisure, celebration if stated"
    )
    visit_duration: str | None = Field(
        default=None, description="Number of nights/days if stated"
    )
    dining_preferences: str | None = Field(
        default=None, description="Cuisine type, dietary if stated"
    )
    dietary_restrictions: str | None = Field(
        default=None, description="Allergies, restrictions if stated"
    )
    gaming_preferences: str | None = Field(
        default=None, description="Game type, stakes if stated"
    )
    entertainment_interests: str | None = Field(
        default=None, description="Shows, music, events if stated"
    )
    spa_interests: str | None = Field(
        default=None, description="Treatments, services if stated"
    )
    occasion: str | None = Field(
        default=None, description="Birthday, anniversary if stated"
    )
    occasion_details: str | None = Field(
        default=None, description="Occasion specifics if stated"
    )
    home_market: str | None = Field(
        default=None, description="Where guest is from if stated"
    )
    budget_signal: str | None = Field(
        default=None, description="Budget level signals if stated"
    )
    loyalty_tier: str | None = Field(
        default=None, description="Loyalty program tier if stated"
    )
    visit_frequency: str | None = Field(
        default=None, description="How often they visit if stated"
    )


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
    "name": 0.15,  # mapped from guest_name
    "party_size": 0.12,
    "visit_purpose": 0.10,
    "preferences": 0.10,  # mapped from dining_preferences
    "occasion": 0.10,
    "visit_duration": 0.06,
    "party_composition": 0.06,
    "dietary": 0.06,  # mapped from dietary_restrictions
    "gaming": 0.05,  # mapped from gaming_preferences
    "entertainment": 0.05,  # mapped from entertainment_interests
    "spa": 0.04,  # mapped from spa_interests
    "occasion_details": 0.03,
    "home_market": 0.03,
    "budget_signal": 0.02,
    "loyalty_tier": 0.01,
    "visit_frequency": 0.02,
    # R76: companion_names, companion_ages, communication_preference removed
    # to reduce schema complexity for Gemini Flash structured output.
    # Weights redistributed to higher-impact fields.
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
    relationship_fields = (
        "occasion",
        "visit_frequency",
        "loyalty_tier",
    )
    has_relationship = (
        sum(1 for f in relationship_fields if extracted_fields.get(f)) >= 1
    )

    if not has_relationship:
        return "relationship"

    return "behavioral"


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a guest profiling assistant for a casino resort concierge.
Analyze the last exchange between the guest and the AI concierge.
Extract guest profile information from what was SAID in this exchange.

## Last Guest Message
{user_message}

## Last AI Response
{ai_response}

## Current Profile
{current_profile}

## What to Extract

Extract facts that the guest STATED or STRONGLY IMPLIED in THIS exchange.
Capture both direct mentions and indirect signals. If the guest clearly
conveyed information — even indirectly — extract it.

### Worked Examples:

Example 1 — Direct mention:
Guest: "I'm Sarah, here with my husband for our anniversary"
→ guest_name: "Sarah", party_size: "2", party_composition: "couple", occasion: "anniversary"

Example 2 — Indirect signal:
Guest: "The kids have been begging to go swimming"
→ party_composition: "family with children"
(Don't extract party_size — "kids" could mean 1 or 4)

Example 3 — Preference from context:
Guest: "Is there anything gluten-free on the menu?"
→ dietary_restrictions: "gluten-free"
(Also extract dining_preferences: "gluten-free options" since they're clearly food-focused)

Example 4 — Budget signal:
Guest: "Can you get us in somewhere nice? Money's not an issue"
→ budget_signal: "premium/luxury"

Example 5 — Loyalty signal:
Guest: "I've been a Gold member for 3 years"
→ loyalty_tier: "Gold", visit_frequency: "regular (3+ years)"

Example 6 — Multi-fact message:
Guest: "We drove up from Philly yesterday, checking out Sunday. Just the two of us, here to relax and play some blackjack."
→ home_market: "Philadelphia", visit_duration: "weekend (2 nights)", party_size: "2", gaming_preferences: "blackjack", visit_purpose: "leisure/relaxation"

### Do NOT extract:
- "I'm done" / "I'm good" / "I'm fine" / "I'm set" / "I'm out" / "I'm back" / "I'm ready" → guest_name: null (NOT names)
- Information already in the current profile (unless guest CORRECTS it)
- Vague sentiments: "whatever" / "anything" / "I don't care" → null
- AI agent's suggestions as guest preferences (only extract what the GUEST said)

Return null for fields with no clear information in this exchange.
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
        extracted_fields = dict(state.get("extracted_fields") or {})

        # Get the last user message and last AI response
        messages = state.get("messages", [])
        user_message = ""
        ai_response = ""

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not ai_response:
                ai_response = _normalize_content(msg.content)
            elif isinstance(msg, HumanMessage) and not user_message:
                user_message = _normalize_content(msg.content)
            if user_message and ai_response:
                break

        if not user_message:
            # No user message to analyze — return current state
            completeness = _calculate_profile_completeness_weighted(extracted_fields)
            phase = _determine_profiling_phase(extracted_fields, completeness)
            logger.debug(
                "Profiling skip: no user_message, existing_fields=%d, completeness=%.2f, phase=%s",
                len(extracted_fields),
                completeness,
                phase,
            )
            return {
                "profiling_phase": phase,
                "profile_completeness_score": completeness,
                "profiling_question_injected": False,
            }

        # Build extraction prompt
        import json

        current_profile_str = (
            json.dumps(extracted_fields, indent=2) if extracted_fields else "{}"
        )
        prompt_text = _EXTRACTION_PROMPT.format(
            user_message=user_message,
            ai_response=ai_response or "(no AI response yet)",
            current_profile=current_profile_str,
        )

        # Call LLM with structured output — reuse whisper planner's singleton
        from src.agent.whisper_planner import _get_whisper_llm

        llm = await _get_whisper_llm()
        extraction_llm = llm.with_structured_output(ProfileExtractionOutput)
        extraction_result: ProfileExtractionOutput = await extraction_llm.ainvoke(
            prompt_text
        )

        # Merge non-None fields into extracted_fields.
        # R76 fix: ProfileExtractionOutput is now flat (str | None fields)
        # instead of nested ConfidenceField objects. Confidence gating is
        # handled in the extraction prompt — the LLM only returns facts that
        # are explicitly stated or strongly implied.
        total_fields = len(ProfileExtractionOutput.model_fields)
        new_fields: dict[str, Any] = {}
        skipped_fields: list[str] = []
        for field_name in ProfileExtractionOutput.model_fields:
            field_value = getattr(extraction_result, field_name, None)
            if field_value is not None and field_value != "":
                # Map profiling field names to extracted_fields keys
                key = _FIELD_NAME_MAP.get(field_name, field_name)
                new_fields[key] = field_value
            else:
                skipped_fields.append(field_name)

        filled_count = len(new_fields)
        logger.info(
            "Profiling extraction: filled=%d/%d fields, keys=%s, skipped=%s",
            filled_count,
            total_fields,
            list(new_fields.keys()) if new_fields else "[]",
            skipped_fields[:5] if len(skipped_fields) > 5 else skipped_fields,
        )

        # Merge new fields (new_fields will be merged by _merge_dicts reducer)
        merged = {**extracted_fields, **new_fields}
        completeness = _calculate_profile_completeness_weighted(merged)
        phase = _determine_profiling_phase(merged, completeness)

        logger.info(
            "Profiling state: phase=%s, completeness=%.2f, total_known=%d, new_this_turn=%d",
            phase,
            completeness,
            len(merged),
            filled_count,
        )

        result: dict[str, Any] = {
            "extracted_fields": new_fields,  # reducer merges with existing
            "profiling_phase": phase,
            "profile_completeness_score": completeness,
            "profiling_question_injected": False,
        }

        # R93: Profile confirmation — when guest confirms/agrees AND we have
        # enough profile data, prepend a brief profile summary to the AI response.
        # This makes the guest feel heard ("So I've got: anniversary for 2...")
        # and gives them a chance to correct any misunderstandings.
        _CONFIRM_SIGNALS = (
            "sounds good",
            "perfect",
            "great",
            "book it",
            "let's do",
            "that works",
            "yes",
            "yeah",
            "yep",
            "sure",
            "do that",
            "sounds perfect",
            "love it",
            "go with that",
            "awesome",
        )
        _user_lower = (user_message or "").lower()
        if (
            completeness >= 0.3
            and merged
            and any(sig in _user_lower for sig in _CONFIRM_SIGNALS)
        ):
            from src.agent.extraction import get_guest_profile_summary

            profile_summary = get_guest_profile_summary(merged)
            if profile_summary and ai_response:
                # Build a brief guest-facing confirmation line
                _summary_parts: list[str] = []
                if merged.get("name"):
                    _summary_parts.append(merged["name"])
                if merged.get("occasion"):
                    _summary_parts.append(merged["occasion"])
                if merged.get("party_size"):
                    _summary_parts.append(f"party of {merged['party_size']}")
                if merged.get("preferences"):
                    _summary_parts.append(merged["preferences"])
                if _summary_parts:
                    _confirm_text = f"So I've got: {', '.join(_summary_parts)}. "
                    # Prepend confirmation to AI response by replacing last AI message
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            original = _normalize_content(msg.content)
                            enriched = f"{_confirm_text}{original}"
                            replacement = AIMessage(content=enriched, id=msg.id)
                            result["messages"] = [replacement]
                            logger.info(
                                "R93: Profile confirmation injected (completeness=%.2f, fields=%s)",
                                completeness,
                                list(_summary_parts),
                            )
                            break

        # R78 fix: Inject profiling question by REPLACING the last AI message
        # (preserving its id so the add_messages reducer replaces, not appends).
        # The profiling question injection in _base.py is the primary path --
        # this is the fallback for when the LLM ignores the system prompt guidance.
        whisper = state.get("whisper_plan")
        guest_sentiment = state.get("guest_sentiment", "unknown")
        if whisper and whisper.get("next_profiling_question"):
            question = whisper["next_profiling_question"]
            technique = whisper.get("question_technique", "none")

            logger.info(
                "Profiling question: candidate='%s', technique=%s, phase=%s, sentiment=%s",
                question[:60] if question else "none",
                technique,
                phase,
                guest_sentiment,
            )

            # Only inject if technique is not "none" and we have an AI response
            if technique != "none" and ai_response:
                # Check if the question is already in the response (from system prompt injection)
                if question.lower().rstrip("?") not in ai_response.lower():
                    # Find the last AI message and replace it with enriched version
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            original = _normalize_content(msg.content)
                            enriched = f"{original}\n\n{question}"
                            # Use the same id to trigger replacement in add_messages reducer
                            replacement = AIMessage(content=enriched, id=msg.id)
                            result["messages"] = [replacement]
                            result["profiling_question_injected"] = True
                            logger.info(
                                "Profiling question injected via message replacement (technique=%s, phase=%s)",
                                technique,
                                phase,
                            )
                            break
                else:
                    # LLM already included the question from system prompt
                    result["profiling_question_injected"] = True
                    logger.info(
                        "Profiling question already in response from system prompt (technique=%s)",
                        technique,
                    )

        if not result["profiling_question_injected"]:
            _reason = "no_whisper_plan"
            if whisper and whisper.get("next_profiling_question"):
                _t = whisper.get("question_technique", "none")
                if _t == "none":
                    _reason = "technique_none"
                elif not ai_response:
                    _reason = "no_ai_response"
                else:
                    _reason = "already_present"
            elif whisper:
                _reason = "no_question"
            logger.debug(
                "Profiling question: injected=False, reason=%s",
                _reason,
            )

        return result

    except Exception:
        # Fail-silent: any error returns empty state update
        logger.warning(
            "profiling_enrichment_node failed, continuing without profiling",
            exc_info=True,
        )
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
    "home_market": "home_market",
    "budget_signal": "budget_signal",
    "loyalty_tier": "loyalty_tier",
    "visit_frequency": "visit_frequency",
    # These map to same name:
    "party_size": "party_size",
    "occasion": "occasion",
}
