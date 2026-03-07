"""Tests for the incentive engine.

Covers IncentiveRule validation, trigger evaluation, auto-approve logic,
per-casino config isolation, format rendering, and the system prompt
integration point.
"""

import pytest

from src.agent.incentives import (
    INCENTIVE_RULES,
    IncentiveEngine,
    IncentiveRule,
    get_incentive_prompt_section,
    _DEFAULT_INCENTIVE_RULES,
)


# ---------------------------------------------------------------------------
# IncentiveRule validation
# ---------------------------------------------------------------------------


class TestIncentiveRuleValidation:
    """IncentiveRule Pydantic model validation."""

    def test_valid_rule(self):
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=25.0,
            framing_template="Happy birthday, enjoy $$value at $property_name.",
        )
        assert rule.trigger_field == "birthday"
        assert rule.incentive_value == 25.0
        assert rule.max_per_guest == 1
        assert rule.auto_approve_threshold == 50.0

    def test_negative_value_rejected(self):
        with pytest.raises(Exception):
            IncentiveRule(
                trigger_field="birthday",
                incentive_type="dining_credit",
                incentive_value=-5.0,
                framing_template="test",
            )

    def test_zero_value_allowed(self):
        rule = IncentiveRule(
            trigger_field="anniversary",
            incentive_type="comp_upgrade",
            incentive_value=0.0,
            framing_template="test",
        )
        assert rule.incentive_value == 0.0

    def test_max_per_guest_minimum_one(self):
        with pytest.raises(Exception):
            IncentiveRule(
                trigger_field="birthday",
                incentive_type="dining_credit",
                incentive_value=10.0,
                max_per_guest=0,
                framing_template="test",
            )

    def test_negative_threshold_rejected(self):
        with pytest.raises(Exception):
            IncentiveRule(
                trigger_field="birthday",
                incentive_type="dining_credit",
                incentive_value=10.0,
                auto_approve_threshold=-1.0,
                framing_template="test",
            )


# ---------------------------------------------------------------------------
# IncentiveEngine trigger evaluation
# ---------------------------------------------------------------------------


class TestGetApplicableIncentives:
    """IncentiveEngine.get_applicable_incentives() trigger logic."""

    def test_birthday_trigger(self):
        engine = IncentiveEngine("mohegan_sun")
        result = engine.get_applicable_incentives(
            profile_completeness=0.3,
            extracted_fields={"birthday": "March 15"},
        )
        triggers = [r.trigger_field for r in result]
        assert "birthday" in triggers

    def test_profile_completeness_75_trigger(self):
        engine = IncentiveEngine("mohegan_sun")
        result = engine.get_applicable_incentives(
            profile_completeness=0.80,
            extracted_fields={},
        )
        triggers = [r.trigger_field for r in result]
        assert "profile_completeness_75" in triggers

    def test_profile_completeness_below_75_no_trigger(self):
        engine = IncentiveEngine("mohegan_sun")
        result = engine.get_applicable_incentives(
            profile_completeness=0.74,
            extracted_fields={},
        )
        triggers = [r.trigger_field for r in result]
        assert "profile_completeness_75" not in triggers

    def test_profile_completeness_exactly_75(self):
        engine = IncentiveEngine("mohegan_sun")
        result = engine.get_applicable_incentives(
            profile_completeness=0.75,
            extracted_fields={},
        )
        triggers = [r.trigger_field for r in result]
        assert "profile_completeness_75" in triggers

    def test_anniversary_via_occasion(self):
        engine = IncentiveEngine("mohegan_sun")
        result = engine.get_applicable_incentives(
            profile_completeness=0.3,
            extracted_fields={"occasion": "wedding anniversary"},
        )
        triggers = [r.trigger_field for r in result]
        assert "anniversary" in triggers

    def test_anniversary_via_direct_field(self):
        engine = IncentiveEngine("mohegan_sun")
        result = engine.get_applicable_incentives(
            profile_completeness=0.3,
            extracted_fields={"anniversary": "2026-04-10"},
        )
        triggers = [r.trigger_field for r in result]
        assert "anniversary" in triggers

    def test_no_triggers_empty_fields(self):
        """R88: Lowered gate to 25%, so 0.20 is below threshold."""
        engine = IncentiveEngine("mohegan_sun")
        result = engine.get_applicable_incentives(
            profile_completeness=0.20,
            extracted_fields={},
        )
        assert result == []

    def test_multiple_triggers(self):
        engine = IncentiveEngine("mohegan_sun")
        result = engine.get_applicable_incentives(
            profile_completeness=0.80,
            extracted_fields={"birthday": "March 15"},
        )
        triggers = [r.trigger_field for r in result]
        assert "birthday" in triggers
        assert "profile_completeness_75" in triggers

    def test_gaming_preference_trigger(self):
        engine = IncentiveEngine("hard_rock_ac")
        result = engine.get_applicable_incentives(
            profile_completeness=0.3,
            extracted_fields={"gaming_preference": "slots"},
        )
        triggers = [r.trigger_field for r in result]
        assert "gaming_preference" in triggers

    def test_falsy_field_not_triggered(self):
        engine = IncentiveEngine("mohegan_sun")
        result = engine.get_applicable_incentives(
            profile_completeness=0.3,
            extracted_fields={"birthday": ""},
        )
        triggers = [r.trigger_field for r in result]
        assert "birthday" not in triggers


# ---------------------------------------------------------------------------
# Per-casino config isolation
# ---------------------------------------------------------------------------


class TestPerCasinoIsolation:
    """Each casino loads its own incentive rules, not another casino's."""

    def test_mohegan_sun_rules(self):
        engine = IncentiveEngine("mohegan_sun")
        assert len(engine.rules) == 4  # R78: added profile_completeness_50
        triggers = {r.trigger_field for r in engine.rules}
        assert "birthday" in triggers
        assert "profile_completeness_75" in triggers
        assert "profile_completeness_50" in triggers
        assert "anniversary" in triggers

    def test_foxwoods_rules(self):
        engine = IncentiveEngine("foxwoods")
        assert len(engine.rules) == 3  # R78: added profile_completeness_50
        triggers = {r.trigger_field for r in engine.rules}
        assert "birthday" in triggers
        assert "profile_completeness_75" in triggers
        assert "profile_completeness_50" in triggers

    def test_hard_rock_has_gaming_preference(self):
        engine = IncentiveEngine("hard_rock_ac")
        triggers = {r.trigger_field for r in engine.rules}
        assert "gaming_preference" in triggers

    def test_unknown_casino_gets_defaults(self):
        engine = IncentiveEngine("unknown_casino_xyz")
        assert engine.rules == _DEFAULT_INCENTIVE_RULES
        assert len(engine.rules) >= 1

    def test_wynn_birthday_value(self):
        engine = IncentiveEngine("wynn_las_vegas")
        birthday_rules = [r for r in engine.rules if r.trigger_field == "birthday"]
        assert len(birthday_rules) == 1
        assert birthday_rules[0].incentive_value == 50.0

    def test_mohegan_birthday_value(self):
        engine = IncentiveEngine("mohegan_sun")
        birthday_rules = [r for r in engine.rules if r.trigger_field == "birthday"]
        assert len(birthday_rules) == 1
        assert birthday_rules[0].incentive_value == 25.0


# ---------------------------------------------------------------------------
# Auto-approve logic
# ---------------------------------------------------------------------------


class TestAutoApprove:
    """IncentiveEngine.check_auto_approve() threshold checks."""

    def test_under_threshold_approved(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=25.0,
            auto_approve_threshold=50.0,
            framing_template="test",
        )
        assert engine.check_auto_approve(rule) is True

    def test_over_threshold_not_approved(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=100.0,
            auto_approve_threshold=50.0,
            framing_template="test",
        )
        assert engine.check_auto_approve(rule) is False

    def test_at_threshold_approved(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=50.0,
            auto_approve_threshold=50.0,
            framing_template="test",
        )
        assert engine.check_auto_approve(rule) is True

    def test_zero_value_always_approved(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="anniversary",
            incentive_type="comp_upgrade",
            incentive_value=0.0,
            auto_approve_threshold=50.0,
            framing_template="test",
        )
        assert engine.check_auto_approve(rule) is True


# ---------------------------------------------------------------------------
# Format incentive offer
# ---------------------------------------------------------------------------


class TestFormatIncentiveOffer:
    """IncentiveEngine.format_incentive_offer() template rendering."""

    def test_basic_format(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=25.0,
            framing_template="Enjoy $$$value at $property_name!",
        )
        result = engine.format_incentive_offer(
            rule,
            {
                "property_name": "Mohegan Sun",
                "value": "25",
            },
        )
        assert "Mohegan Sun" in result
        assert "$25" in result

    def test_missing_placeholder_safe(self):
        """safe_substitute does not crash on missing keys."""
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=25.0,
            framing_template="Enjoy at $property_name, $missing_key!",
        )
        result = engine.format_incentive_offer(
            rule,
            {
                "property_name": "Mohegan Sun",
            },
        )
        assert "Mohegan Sun" in result
        assert "$missing_key" in result


# ---------------------------------------------------------------------------
# Host approval request
# ---------------------------------------------------------------------------


class TestBuildHostApprovalRequest:
    """IncentiveEngine.build_host_approval_request() structure."""

    def test_approval_request_structure(self):
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=100.0,
            auto_approve_threshold=50.0,
            framing_template="test",
        )
        request = engine.build_host_approval_request(rule, {"name": "Alice"})
        assert request["casino_id"] == "mohegan_sun"
        assert request["incentive_type"] == "dining_credit"
        assert request["incentive_value"] == 100.0
        assert request["requires_approval"] is True
        assert request["guest_context"]["name"] == "Alice"

    def test_guest_context_is_deepcopy(self):
        """Guest context dict must be a deep copy to prevent mutation."""
        engine = IncentiveEngine("mohegan_sun")
        rule = IncentiveRule(
            trigger_field="birthday",
            incentive_type="dining_credit",
            incentive_value=100.0,
            framing_template="test",
        )
        original = {"name": "Alice", "prefs": {"food": "sushi"}}
        request = engine.build_host_approval_request(rule, original)
        request["guest_context"]["prefs"]["food"] = "steak"
        assert original["prefs"]["food"] == "sushi"


# ---------------------------------------------------------------------------
# get_incentive_prompt_section integration
# ---------------------------------------------------------------------------


class TestGetIncentivePromptSection:
    """Integration point: get_incentive_prompt_section() — R87: returns tuple."""

    def test_returns_empty_when_no_incentives(self, monkeypatch):
        monkeypatch.setenv("PROPERTY_NAME", "Test Casino")
        from src.config import get_settings

        get_settings.cache_clear()
        result, approval = get_incentive_prompt_section(
            casino_id="mohegan_sun",
            profile_completeness=0.20,  # R88: Below 25% threshold
            extracted_fields={},
        )
        assert result == ""
        assert approval is None

    def test_returns_section_with_applicable(self, monkeypatch):
        monkeypatch.setenv("PROPERTY_NAME", "Mohegan Sun")
        from src.config import get_settings

        get_settings.cache_clear()
        result, approval = get_incentive_prompt_section(
            casino_id="mohegan_sun",
            profile_completeness=0.80,
            extracted_fields={"birthday": "March 15"},
        )
        assert "Guest Incentive" in result
        assert "Mohegan Sun" in result

    def test_unknown_casino_uses_defaults(self, monkeypatch):
        monkeypatch.setenv("PROPERTY_NAME", "Unknown Resort")
        from src.config import get_settings

        get_settings.cache_clear()
        result, approval = get_incentive_prompt_section(
            casino_id="unknown_casino",
            profile_completeness=0.3,
            extracted_fields={"birthday": "June 1"},
        )
        assert "Guest Incentive" in result

    def test_high_value_shows_approval_note(self, monkeypatch):
        monkeypatch.setenv("PROPERTY_NAME", "Test Casino")
        from src.config import get_settings

        get_settings.cache_clear()

        engine = IncentiveEngine("mohegan_sun")
        birthday_rules = [r for r in engine.rules if r.trigger_field == "birthday"]
        assert all(
            r.incentive_value <= r.auto_approve_threshold for r in birthday_rules
        )

    def test_weave_naturally_instruction(self, monkeypatch):
        monkeypatch.setenv("PROPERTY_NAME", "Mohegan Sun")
        from src.config import get_settings

        get_settings.cache_clear()
        result, approval = get_incentive_prompt_section(
            casino_id="mohegan_sun",
            profile_completeness=0.80,
            extracted_fields={},
        )
        if result:
            assert "naturally" in result.lower()


# ---------------------------------------------------------------------------
# INCENTIVE_RULES immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    """Module-level INCENTIVE_RULES must be immutable."""

    def test_rules_dict_immutable(self):
        with pytest.raises(TypeError):
            INCENTIVE_RULES["new_casino"] = ()  # type: ignore[index]

    def test_all_casinos_present(self):
        expected = {
            "mohegan_sun",
            "foxwoods",
            "parx_casino",
            "wynn_las_vegas",
            "hard_rock_ac",
        }
        assert set(INCENTIVE_RULES.keys()) == expected


# ---------------------------------------------------------------------------
# R78: profile_completeness_50 and occasion-based birthday triggers
# ---------------------------------------------------------------------------


class TestR78LowerThresholdTriggers:
    """R78 fix: lower threshold and occasion-based incentive triggers."""

    def test_completeness_50_fires_at_threshold(self):
        """R88: Lowered threshold from 50% to 25%. 30% should fire."""
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(
            profile_completeness=0.30,
            extracted_fields={},
        )
        triggers = {r.trigger_field for r in applicable}
        assert "profile_completeness_50" in triggers

    def test_completeness_50_fires_at_old_half(self):
        """Old 55% threshold should still fire (above 25%)."""
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(
            profile_completeness=0.55,
            extracted_fields={},
        )
        triggers = {r.trigger_field for r in applicable}
        assert "profile_completeness_50" in triggers

    def test_completeness_50_does_not_fire_below(self):
        """R88: Below 25% threshold should not fire."""
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(
            profile_completeness=0.20,
            extracted_fields={},
        )
        triggers = {r.trigger_field for r in applicable}
        assert "profile_completeness_50" not in triggers

    def test_birthday_fires_from_occasion_field(self):
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(
            profile_completeness=0.0,
            extracted_fields={"occasion": "birthday"},
        )
        triggers = {r.trigger_field for r in applicable}
        assert "birthday" in triggers

    def test_birthday_fires_from_birthday_field(self):
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(
            profile_completeness=0.0,
            extracted_fields={"birthday": "March 15"},
        )
        triggers = {r.trigger_field for r in applicable}
        assert "birthday" in triggers

    def test_birthday_does_not_fire_without_signal(self):
        engine = IncentiveEngine("mohegan_sun")
        applicable = engine.get_applicable_incentives(
            profile_completeness=0.0,
            extracted_fields={"name": "Mike"},
        )
        triggers = {r.trigger_field for r in applicable}
        assert "birthday" not in triggers

    def test_all_casinos_have_50_threshold(self):
        for casino_id in INCENTIVE_RULES:
            engine = IncentiveEngine(casino_id)
            triggers = {r.trigger_field for r in engine.rules}
            assert "profile_completeness_50" in triggers, (
                f"{casino_id} missing profile_completeness_50"
            )
