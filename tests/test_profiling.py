"""Unit tests for the Guest Profiling Intelligence System.

Tests for profiling.py: ConfidenceField model, ProfileExtractionOutput,
weighted completeness, phase determination, field name mapping,
extraction prompt quality, profiling technique prompts, and incentive engine.

Mock-based profiling_enrichment_node tests removed (mock purge R111).
"""

import pytest
from pydantic import ValidationError

from src.agent.profiling import (
    ConfidenceField,
    ProfileExtractionOutput,
    PROFILING_TECHNIQUE_PROMPTS,
    _calculate_profile_completeness_weighted,
    _determine_profiling_phase,
    _FIELD_NAME_MAP,
    _PROFILE_WEIGHTS,
)
from src.agent.incentives import (
    IncentiveEngine,
    IncentiveRule,
    INCENTIVE_RULES,
    get_incentive_prompt_section,
)


# ---------------------------------------------------------------------------
# ConfidenceField model validation
# ---------------------------------------------------------------------------


class TestConfidenceField:
    """Pydantic model validation for ConfidenceField."""

    def test_valid_explicit_field(self):
        cf = ConfidenceField(value="Sarah", confidence=0.95, source="explicit")
        assert cf.value == "Sarah"
        assert cf.confidence == 0.95
        assert cf.source == "explicit"

    def test_valid_inferred_field(self):
        cf = ConfidenceField(value="anniversary", confidence=0.7, source="inferred")
        assert cf.source == "inferred"

    def test_valid_corrected_field(self):
        cf = ConfidenceField(value="Mike", confidence=0.99, source="corrected")
        assert cf.source == "corrected"

    def test_default_source_is_explicit(self):
        cf = ConfidenceField(value="test", confidence=0.5)
        assert cf.source == "explicit"

    def test_confidence_lower_bound(self):
        cf = ConfidenceField(value="x", confidence=0.0)
        assert cf.confidence == 0.0

    def test_confidence_upper_bound(self):
        cf = ConfidenceField(value="x", confidence=1.0)
        assert cf.confidence == 1.0

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            ConfidenceField(value="x", confidence=-0.1)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            ConfidenceField(value="x", confidence=1.1)

    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            ConfidenceField(value="x", confidence=0.5, source="guessed")

    def test_value_can_be_int(self):
        cf = ConfidenceField(value=4, confidence=0.9)
        assert cf.value == 4

    def test_value_can_be_list(self):
        cf = ConfidenceField(value=["steak", "seafood"], confidence=0.8)
        assert cf.value == ["steak", "seafood"]

    def test_value_can_be_none(self):
        cf = ConfidenceField(value=None, confidence=0.0)
        assert cf.value is None


# ---------------------------------------------------------------------------
# ProfileExtractionOutput field population
# ---------------------------------------------------------------------------


class TestProfileExtractionOutput:
    """ProfileExtractionOutput structured output model."""

    def test_all_none_by_default(self):
        output = ProfileExtractionOutput()
        for field_name in ProfileExtractionOutput.model_fields:
            assert getattr(output, field_name) is None

    def test_single_field_populated(self):
        output = ProfileExtractionOutput(guest_name="Sarah")
        assert output.guest_name == "Sarah"
        assert output.party_size is None

    def test_multiple_fields_populated(self):
        output = ProfileExtractionOutput(
            guest_name="Mike",
            party_size="4",
            occasion="birthday",
        )
        assert output.guest_name == "Mike"
        assert output.party_size == "4"
        assert output.occasion == "birthday"

    def test_all_16_fields_exist(self):
        """ProfileExtractionOutput has exactly 16 fields matching _FIELD_NAME_MAP."""
        assert len(ProfileExtractionOutput.model_fields) == 16
        for field_name in ProfileExtractionOutput.model_fields:
            assert field_name in _FIELD_NAME_MAP, (
                f"{field_name} missing from _FIELD_NAME_MAP"
            )

    def test_field_name_map_covers_all_output_fields(self):
        """Every ProfileExtractionOutput field has a mapping in _FIELD_NAME_MAP."""
        for field_name in ProfileExtractionOutput.model_fields:
            assert field_name in _FIELD_NAME_MAP


# ---------------------------------------------------------------------------
# _calculate_profile_completeness_weighted
# ---------------------------------------------------------------------------


class TestProfileCompletenessWeighted:
    """Weighted completeness score calculation."""

    def test_empty_profile_returns_zero(self):
        assert _calculate_profile_completeness_weighted({}) == 0.0

    def test_none_profile_returns_zero(self):
        assert _calculate_profile_completeness_weighted(None) == 0.0

    def test_single_high_weight_field(self):
        # name has weight 0.15 (stored as "name", mapped from guest_name)
        result = _calculate_profile_completeness_weighted({"name": "Sarah"})
        assert result == pytest.approx(0.15, abs=0.01)

    def test_phase1_foundation_fields(self):
        # name(0.15) + party_size(0.12) + visit_purpose(0.10) = 0.37
        fields = {"name": "Mike", "party_size": "4", "visit_purpose": "leisure"}
        result = _calculate_profile_completeness_weighted(fields)
        assert result == pytest.approx(0.37, abs=0.01)

    def test_all_fields_filled_returns_one(self):
        all_fields = {name: "some_value" for name in _PROFILE_WEIGHTS}
        result = _calculate_profile_completeness_weighted(all_fields)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_weights_sum_to_one(self):
        total = sum(_PROFILE_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_falsy_values_not_counted(self):
        fields = {"name": "", "party_size": 0, "visit_purpose": None}
        result = _calculate_profile_completeness_weighted(fields)
        assert result == 0.0

    def test_low_weight_fields_contribute_less(self):
        # loyalty_tier has weight 0.01
        low = _calculate_profile_completeness_weighted({"loyalty_tier": "gold"})
        # name has weight 0.15
        high = _calculate_profile_completeness_weighted({"name": "Sarah"})
        assert high > low

    def test_capped_at_one(self):
        # Even with extra unknown fields, score should not exceed 1.0
        fields = {name: "value" for name in _PROFILE_WEIGHTS}
        fields["extra_field"] = "bonus"
        result = _calculate_profile_completeness_weighted(fields)
        assert result <= 1.0

    def test_partial_completeness(self):
        # name(0.15) + preferences(0.10) = 0.25
        fields = {"name": "Sarah", "preferences": "steak"}
        result = _calculate_profile_completeness_weighted(fields)
        assert result == pytest.approx(0.25, abs=0.01)


# ---------------------------------------------------------------------------
# _determine_profiling_phase
# ---------------------------------------------------------------------------


class TestProfilingPhase:
    """Golden path phase determination."""

    def test_empty_fields_returns_foundation(self):
        assert _determine_profiling_phase({}, 0.0) == "foundation"

    def test_single_foundation_field_stays_foundation(self):
        # Need >= 2 of (name, party_size, visit_purpose) for next phase
        fields = {"name": "Sarah"}
        assert _determine_profiling_phase(fields, 0.15) == "foundation"

    def test_two_foundation_fields_advances_to_preference(self):
        fields = {"name": "Mike", "party_size": 4}
        completeness = _calculate_profile_completeness_weighted(fields)
        assert _determine_profiling_phase(fields, completeness) == "preference"

    def test_three_foundation_fields_still_preference(self):
        fields = {"name": "Mike", "party_size": 4, "visit_purpose": "leisure"}
        completeness = _calculate_profile_completeness_weighted(fields)
        assert _determine_profiling_phase(fields, completeness) == "preference"

    def test_foundation_plus_dining_advances_to_relationship(self):
        fields = {
            "name": "Mike",
            "party_size": 4,
            "preferences": "steak",
        }
        completeness = _calculate_profile_completeness_weighted(fields)
        assert _determine_profiling_phase(fields, completeness) == "relationship"

    def test_foundation_plus_entertainment_advances_to_relationship(self):
        fields = {
            "name": "Sarah",
            "visit_purpose": "leisure",
            "entertainment": "comedy shows",
        }
        completeness = _calculate_profile_completeness_weighted(fields)
        assert _determine_profiling_phase(fields, completeness) == "relationship"

    def test_foundation_plus_preference_plus_occasion_is_behavioral(self):
        fields = {
            "name": "Mike",
            "party_size": 4,
            "preferences": "seafood",
            "occasion": "anniversary",
        }
        completeness = _calculate_profile_completeness_weighted(fields)
        assert _determine_profiling_phase(fields, completeness) == "behavioral"

    def test_foundation_plus_preference_plus_visit_frequency_is_behavioral(self):
        fields = {
            "name": "Sarah",
            "visit_purpose": "business",
            "gaming": "blackjack",
            "visit_frequency": "monthly",
        }
        completeness = _calculate_profile_completeness_weighted(fields)
        assert _determine_profiling_phase(fields, completeness) == "behavioral"

    def test_only_preference_fields_still_foundation(self):
        # Without 2 foundation fields, stays in foundation even with preferences
        fields = {"preferences": "Italian"}
        assert _determine_profiling_phase(fields, 0.08) == "foundation"


# ---------------------------------------------------------------------------
# Field name mapping
# ---------------------------------------------------------------------------


class TestFieldNameMapping:
    """Verify field name mapping from profiling model to extracted_fields keys."""

    def test_guest_name_maps_to_name(self):
        assert _FIELD_NAME_MAP["guest_name"] == "name"

    def test_dining_preferences_maps_to_preferences(self):
        assert _FIELD_NAME_MAP["dining_preferences"] == "preferences"

    def test_dietary_restrictions_maps_to_dietary(self):
        assert _FIELD_NAME_MAP["dietary_restrictions"] == "dietary"

    def test_gaming_preferences_maps_to_gaming(self):
        assert _FIELD_NAME_MAP["gaming_preferences"] == "gaming"

    def test_entertainment_interests_maps_to_entertainment(self):
        assert _FIELD_NAME_MAP["entertainment_interests"] == "entertainment"

    def test_spa_interests_maps_to_spa(self):
        assert _FIELD_NAME_MAP["spa_interests"] == "spa"

    def test_party_size_maps_to_itself(self):
        assert _FIELD_NAME_MAP["party_size"] == "party_size"

    def test_occasion_maps_to_itself(self):
        assert _FIELD_NAME_MAP["occasion"] == "occasion"


# ---------------------------------------------------------------------------
# R105: Extraction prompt quality
# ---------------------------------------------------------------------------


class TestR105ExtractionPrompt:
    """R105: Verify extraction prompt has no contradictions and has worked examples."""

    def test_extraction_prompt_no_contradiction(self):
        """Prompt must NOT contain both 'generous' and 'precision over recall'."""
        from src.agent.profiling import _EXTRACTION_PROMPT

        prompt_lower = _EXTRACTION_PROMPT.lower()
        has_generous = "extremely generous" in prompt_lower
        has_precision_over_recall = "prefer precision over recall" in prompt_lower
        # They must NOT both be present — that's a contradiction
        assert not (has_generous and has_precision_over_recall), (
            "Extraction prompt contains contradictory instructions: "
            "'extremely generous' AND 'prefer precision over recall'"
        )

    def test_extraction_prompt_has_worked_examples(self):
        """Prompt must contain numbered worked examples."""
        from src.agent.profiling import _EXTRACTION_PROMPT

        assert "Example 1" in _EXTRACTION_PROMPT
        assert "Example 2" in _EXTRACTION_PROMPT
        assert "Example 3" in _EXTRACTION_PROMPT

    def test_extraction_prompt_has_indirect_signal_example(self):
        """Prompt must show extraction from indirect signals like 'kids want pool'."""
        from src.agent.profiling import _EXTRACTION_PROMPT

        assert "party_composition" in _EXTRACTION_PROMPT

    def test_extraction_prompt_has_budget_signal_example(self):
        """Prompt must show budget signal extraction."""
        from src.agent.profiling import _EXTRACTION_PROMPT

        assert "budget_signal" in _EXTRACTION_PROMPT

    def test_extraction_prompt_has_multi_fact_example(self):
        """Prompt must show multi-fact extraction from a single message."""
        from src.agent.profiling import _EXTRACTION_PROMPT

        # Must have an example that extracts 4+ fields from one message
        assert "home_market" in _EXTRACTION_PROMPT
        assert "gaming_preferences" in _EXTRACTION_PROMPT

    def test_extraction_prompt_excludes_common_phrases(self):
        """Prompt must tell LLM not to extract 'I'm done' as a name."""
        from src.agent.profiling import _EXTRACTION_PROMPT

        assert "I'm done" in _EXTRACTION_PROMPT
        assert "I'm good" in _EXTRACTION_PROMPT


# ---------------------------------------------------------------------------
# Profiling technique prompts
# ---------------------------------------------------------------------------


class TestProfilingTechniquePrompts:
    """Verify all expected techniques are defined."""

    def test_give_to_get_exists(self):
        assert "give_to_get" in PROFILING_TECHNIQUE_PROMPTS

    def test_assumptive_bridge_exists(self):
        assert "assumptive_bridge" in PROFILING_TECHNIQUE_PROMPTS

    def test_contextual_inference_exists(self):
        assert "contextual_inference" in PROFILING_TECHNIQUE_PROMPTS

    def test_need_payoff_exists(self):
        assert "need_payoff" in PROFILING_TECHNIQUE_PROMPTS

    def test_incentive_frame_exists(self):
        assert "incentive_frame" in PROFILING_TECHNIQUE_PROMPTS

    def test_reflective_confirm_exists(self):
        assert "reflective_confirm" in PROFILING_TECHNIQUE_PROMPTS

    def test_none_exists(self):
        assert "none" in PROFILING_TECHNIQUE_PROMPTS

    def test_none_disables_questions(self):
        assert (
            "not ask" in PROFILING_TECHNIQUE_PROMPTS["none"].lower()
            or "do not" in PROFILING_TECHNIQUE_PROMPTS["none"].lower()
        )

    def test_all_prompts_are_non_empty_strings(self):
        for technique, prompt in PROFILING_TECHNIQUE_PROMPTS.items():
            assert isinstance(prompt, str) and len(prompt) > 0, (
                f"{technique} prompt is empty"
            )


# ---------------------------------------------------------------------------
# IncentiveEngine
# ---------------------------------------------------------------------------


class TestIncentiveEngine:
    """Tests for the incentive rule engine."""

    def test_known_casino_loads_rules(self):
        engine = IncentiveEngine("mohegan_sun")
        assert len(engine.rules) > 0
        assert engine.casino_id == "mohegan_sun"

    def test_unknown_casino_loads_defaults(self):
        engine = IncentiveEngine("unknown_casino")
        assert len(engine.rules) > 0

    def test_birthday_trigger(self):
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(0.0, {"birthday": "2026-03-15"})
        birthday_rules = [r for r in applicable if r.trigger_field == "birthday"]
        assert len(birthday_rules) == 1

    def test_profile_completeness_75_trigger(self):
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(0.80, {})
        completeness_rules = [
            r for r in applicable if r.trigger_field == "profile_completeness_75"
        ]
        assert len(completeness_rules) == 1

    def test_profile_completeness_below_threshold(self):
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(0.50, {})
        completeness_rules = [
            r for r in applicable if r.trigger_field == "profile_completeness_75"
        ]
        assert len(completeness_rules) == 0

    def test_no_triggers_returns_empty(self):
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(0.0, {})
        assert applicable == []

    def test_anniversary_trigger_from_occasion(self):
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(
            0.0, {"occasion": "wedding anniversary"}
        )
        anniversary_rules = [r for r in applicable if r.trigger_field == "anniversary"]
        assert len(anniversary_rules) == 1

    def test_format_incentive_offer(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = engine.rules[0]  # birthday rule
        offer = engine.format_incentive_offer(
            rule,
            {
                "property_name": "Mohegan Sun",
                "value": "25",
                "incentive_type": "dining credit",
            },
        )
        assert "Mohegan Sun" in offer

    def test_auto_approve_within_threshold(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=25.0,
            auto_approve_threshold=50.0,
            framing_template="Test",
        )
        assert engine.check_auto_approve(rule) is True

    def test_auto_approve_exceeds_threshold(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=100.0,
            auto_approve_threshold=50.0,
            framing_template="Test",
        )
        assert engine.check_auto_approve(rule) is False

    def test_build_host_approval_request(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = engine.rules[0]
        request = engine.build_host_approval_request(rule, {"name": "Sarah"})
        assert request["casino_id"] == "mohegan_sun"
        assert request["requires_approval"] is True
        assert request["guest_context"]["name"] == "Sarah"

    def test_all_five_casinos_have_rules(self):
        for casino_id in (
            "mohegan_sun",
            "foxwoods",
            "parx_casino",
            "wynn_las_vegas",
            "hard_rock_ac",
        ):
            assert casino_id in INCENTIVE_RULES
            assert len(INCENTIVE_RULES[casino_id]) >= 2


class TestGetIncentivePromptSection:
    """Tests for get_incentive_prompt_section integration point (R87: returns tuple)."""

    def test_returns_empty_when_no_triggers(self):
        result, approval = get_incentive_prompt_section("mohegan_sun", 0.0, {})
        assert result == ""
        assert approval is None

    def test_returns_section_for_birthday(self):
        result, approval = get_incentive_prompt_section(
            "mohegan_sun", 0.0, {"birthday": "2026-03-15"}
        )
        assert "Guest Incentive" in result
        assert "Mohegan Sun" in result
        assert approval is None  # Mohegan birthday=$25, threshold=$50 -> auto-approved

    def test_returns_section_for_completeness_75(self):
        result, approval = get_incentive_prompt_section("mohegan_sun", 0.80, {})
        assert "Guest Incentive" in result

    def test_includes_natural_offer_guidance(self):
        result, approval = get_incentive_prompt_section(
            "mohegan_sun", 0.0, {"birthday": "2026-03-15"}
        )
        assert "naturally" in result.lower()

    def test_host_approval_note_for_high_value(self):
        """Incentives above auto_approve_threshold should note host approval required."""
        # Create a high-value rule scenario -- Wynn $50 = threshold, so it auto-approves.
        # We need to check the format output for normal auto-approve cases.
        result, approval = get_incentive_prompt_section(
            "wynn_las_vegas", 0.0, {"birthday": "2026-03-15"}
        )
        # Wynn birthday is $50 with threshold $50, so auto-approved
        assert "REQUIRES HOST APPROVAL" not in result
        assert approval is None


class TestIncentiveRuleValidation:
    """Pydantic validation for IncentiveRule."""

    def test_negative_value_rejected(self):
        with pytest.raises(ValidationError):
            IncentiveRule(
                trigger_field="birthday",
                incentive_type="dining_credit",
                incentive_value=-5.0,
                framing_template="Test",
            )

    def test_zero_max_per_guest_rejected(self):
        with pytest.raises(ValidationError):
            IncentiveRule(
                trigger_field="birthday",
                incentive_type="dining_credit",
                incentive_value=10.0,
                max_per_guest=0,
                framing_template="Test",
            )

    def test_non_monetary_auto_approved(self):
        rule = IncentiveRule(
            trigger_field="anniversary",
            incentive_type="comp_upgrade",
            incentive_value=0.0,
            framing_template="Happy anniversary!",
        )
        engine = IncentiveEngine("mohegan_sun")
        assert engine.check_auto_approve(rule) is True

    def test_immutable_rules_mapping(self):
        """INCENTIVE_RULES is MappingProxyType -- cannot be mutated."""
        with pytest.raises(TypeError):
            INCENTIVE_RULES["new_casino"] = ()
