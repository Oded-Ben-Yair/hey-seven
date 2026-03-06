"""Tests for CompStrategy tool (H9: Comp Strategy).

Pure deterministic tests — no LLM calls, no mocks needed.
"""

import pytest

from src.agent.behavior_tools.comp_strategy import (
    CompOffer,
    CompStrategyInput,
    CompStrategyOutput,
    CompTier,
    COMP_TIERS,
    get_comp_prompt_section,
    get_comp_strategy,
    _AUTO_APPROVE_THRESHOLD,
    _TIER_ADT_ESTIMATES,
    _detect_occasion,
    _resolve_tier,
)


class TestCompTiers:
    """Test comp tier resolution."""

    def test_all_casinos_have_tiers(self):
        """Every configured casino has comp tiers."""
        for casino_id in (
            "mohegan_sun",
            "foxwoods",
            "parx_casino",
            "wynn_las_vegas",
            "hard_rock_ac",
        ):
            assert casino_id in COMP_TIERS, f"Missing tiers for {casino_id}"

    def test_tiers_are_ordered_by_adt(self):
        """Tiers within each casino are ordered by ascending ADT."""
        for casino_id, tiers in COMP_TIERS.items():
            for i in range(1, len(tiers)):
                assert tiers[i].adt_min >= tiers[i - 1].adt_min, (
                    f"{casino_id}: tier {i} ADT min < tier {i - 1}"
                )

    def test_tier_names_are_distinct(self):
        """Each casino has distinct tier names."""
        for casino_id, tiers in COMP_TIERS.items():
            names = [t.tier_name for t in tiers]
            assert len(names) == len(set(names)), f"Duplicate tier names in {casino_id}"

    def test_exploration_tier_starts_at_zero(self):
        """Lowest tier starts at ADT 0."""
        for casino_id, tiers in COMP_TIERS.items():
            assert tiers[0].adt_min == 0.0, (
                f"{casino_id}: first tier doesn't start at 0"
            )


class TestResolverTier:
    """Test tier resolution from ADT."""

    def test_low_adt_gets_exploration(self):
        tier = _resolve_tier("mohegan_sun", 25.0)
        assert tier.tier_name == "exploration"

    def test_medium_adt_gets_regular(self):
        tier = _resolve_tier("mohegan_sun", 100.0)
        assert tier.tier_name == "regular"

    def test_high_adt_gets_vip(self):
        tier = _resolve_tier("mohegan_sun", 500.0)
        assert tier.tier_name == "vip"

    def test_very_high_adt_gets_high_roller(self):
        tier = _resolve_tier("mohegan_sun", 2000.0)
        assert tier.tier_name == "high_roller"

    def test_unknown_casino_gets_default(self):
        tier = _resolve_tier("unknown_casino", 25.0)
        assert tier.tier_name == "exploration"

    def test_exact_boundary_goes_to_higher_tier(self):
        tier = _resolve_tier("mohegan_sun", 50.0)
        assert tier.tier_name == "regular"


class TestDetectOccasion:
    """Test occasion detection."""

    def test_birthday_detected(self):
        assert _detect_occasion("It's my birthday") == "birthday"

    def test_anniversary_detected(self):
        assert _detect_occasion("celebrating our anniversary") == "anniversary"

    def test_wedding_is_celebration(self):
        assert _detect_occasion("wedding trip") == "celebration"

    def test_no_occasion(self):
        assert _detect_occasion("just visiting") is None

    def test_none_input(self):
        assert _detect_occasion(None) is None

    def test_case_insensitive(self):
        assert _detect_occasion("BIRTHDAY PARTY") == "birthday"


class TestGetCompStrategy:
    """Test comp strategy calculation."""

    def test_new_guest_gets_exploration_comps(self):
        result = get_comp_strategy(
            CompStrategyInput(
                guest_tier="new",
                property_id="mohegan_sun",
            )
        )
        assert result.tier_name == "exploration"
        assert len(result.eligible_comps) > 0
        assert all(c.auto_approve for c in result.eligible_comps)

    def test_vip_gets_high_value_comps(self):
        result = get_comp_strategy(
            CompStrategyInput(
                guest_tier="vip",
                property_id="mohegan_sun",
            )
        )
        assert result.tier_name == "vip"
        assert any(c.estimated_value > 50 for c in result.eligible_comps)

    def test_high_roller_requires_approval(self):
        result = get_comp_strategy(
            CompStrategyInput(
                guest_tier="high_roller",
                property_id="mohegan_sun",
            )
        )
        assert result.approval_required
        assert any(not c.auto_approve for c in result.eligible_comps)

    def test_birthday_occasion_boosts_value(self):
        without_occasion = get_comp_strategy(
            CompStrategyInput(
                guest_tier="regular",
                property_id="mohegan_sun",
            )
        )
        with_occasion = get_comp_strategy(
            CompStrategyInput(
                guest_tier="regular",
                property_id="mohegan_sun",
                current_occasion="birthday dinner",
            )
        )
        # Occasion multiplier should increase some comp values
        max_without = max(c.estimated_value for c in without_occasion.eligible_comps)
        max_with = max(c.estimated_value for c in with_occasion.eligible_comps)
        assert max_with >= max_without

    def test_frustrated_guest_skips_high_value(self):
        result = get_comp_strategy(
            CompStrategyInput(
                guest_tier="vip",
                property_id="mohegan_sun",
                guest_sentiment="frustrated",
            )
        )
        for comp in result.eligible_comps:
            assert comp.estimated_value <= _AUTO_APPROVE_THRESHOLD

    def test_grief_sentiment_skips_high_value(self):
        result = get_comp_strategy(
            CompStrategyInput(
                guest_tier="high_roller",
                property_id="mohegan_sun",
                guest_sentiment="grief",
            )
        )
        for comp in result.eligible_comps:
            assert comp.estimated_value <= _AUTO_APPROVE_THRESHOLD

    def test_talking_points_include_occasion(self):
        result = get_comp_strategy(
            CompStrategyInput(
                guest_tier="regular",
                property_id="mohegan_sun",
                current_occasion="birthday",
            )
        )
        assert any("birthday" in p.lower() for p in result.talking_points)

    def test_talking_points_for_regular_visitors(self):
        result = get_comp_strategy(
            CompStrategyInput(
                guest_tier="regular",
                visit_frequency="regular",
                property_id="mohegan_sun",
            )
        )
        assert any(
            "loyalty" in p.lower() or "regular" in p.lower()
            for p in result.talking_points
        )

    def test_restrictions_always_present(self):
        result = get_comp_strategy(
            CompStrategyInput(
                guest_tier="new",
                property_id="mohegan_sun",
            )
        )
        assert len(result.restrictions) >= 2

    def test_all_casinos_return_results(self):
        for casino_id in (
            "mohegan_sun",
            "foxwoods",
            "parx_casino",
            "wynn_las_vegas",
            "hard_rock_ac",
        ):
            result = get_comp_strategy(
                CompStrategyInput(
                    guest_tier="regular",
                    property_id=casino_id,
                )
            )
            assert len(result.eligible_comps) > 0, f"No comps for {casino_id}"


class TestGetCompPromptSection:
    """Test prompt section generation."""

    def test_empty_state_returns_string(self):
        section = get_comp_prompt_section({}, casino_id="mohegan_sun")
        # New guest with no data should still get exploration comps
        assert isinstance(section, str)

    def test_section_has_comp_details(self):
        state = {
            "extracted_fields": {"loyalty_tier": "Ignite"},
            "guest_sentiment": None,
        }
        section = get_comp_prompt_section(state, casino_id="mohegan_sun")
        assert "Comp Strategy" in section
        assert "Available Comps" in section

    def test_vip_tier_detected_from_loyalty(self):
        state = {
            "extracted_fields": {"loyalty_tier": "Soar"},
            "guest_sentiment": None,
        }
        section = get_comp_prompt_section(state, casino_id="mohegan_sun")
        assert "high_roller" in section

    def test_regular_tier_from_signal(self):
        state = {
            "extracted_fields": {"loyalty_signal": "I come here regularly"},
            "guest_sentiment": None,
        }
        section = get_comp_prompt_section(state, casino_id="mohegan_sun")
        assert "regular" in section

    def test_occasion_injected(self):
        state = {
            "extracted_fields": {"occasion": "birthday"},
            "guest_sentiment": None,
        }
        section = get_comp_prompt_section(state, casino_id="mohegan_sun")
        assert "birthday" in section.lower() or "Talking Points" in section


class TestCompModels:
    """Test Pydantic model validation."""

    def test_comp_offer_validates(self):
        offer = CompOffer(
            comp_type="dining_credit",
            description="Test comp",
            estimated_value=25.0,
            framing="Test framing",
        )
        assert offer.auto_approve is True

    def test_comp_strategy_output_serializable(self):
        """Output must be JSON-serializable (crosses graph boundary)."""
        import json

        result = get_comp_strategy(
            CompStrategyInput(
                guest_tier="regular",
                property_id="mohegan_sun",
            )
        )
        serialized = json.dumps(result.model_dump())
        assert json.loads(serialized)

    def test_tier_adt_estimates_complete(self):
        """All guest tiers have ADT estimates."""
        for tier in ("new", "regular", "vip", "high_roller"):
            assert tier in _TIER_ADT_ESTIMATES
