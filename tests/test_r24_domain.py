"""R24 Domain & Persona Excellence tests.

Validates:
- CASINO_PROFILES has real multi-property data with required sections
- get_casino_profile returns correct profile or DEFAULT_CONFIG fallback
- Knowledge-base files exist and contain substantive content
- HEART_ESCALATION_LANGUAGE has all 5 HEART steps
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root for knowledge-base file checks
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CASINO_PROFILES tests
# ---------------------------------------------------------------------------


class TestCasinoProfiles:
    """Validate CASINO_PROFILES dict structure and content."""

    def test_casino_profiles_has_at_least_three_entries(self):
        from src.casino.config import CASINO_PROFILES

        assert len(CASINO_PROFILES) >= 3, (
            f"Expected at least 3 casino profiles, got {len(CASINO_PROFILES)}"
        )

    def test_casino_profiles_contains_expected_ids(self):
        from src.casino.config import CASINO_PROFILES

        expected = {"mohegan_sun", "foxwoods", "hard_rock_ac"}
        assert expected.issubset(set(CASINO_PROFILES.keys())), (
            f"Missing profiles: {expected - set(CASINO_PROFILES.keys())}"
        )

    @pytest.mark.parametrize("casino_id", ["mohegan_sun", "foxwoods", "hard_rock_ac"])
    def test_profile_has_branding_section(self, casino_id):
        from src.casino.config import CASINO_PROFILES

        profile = CASINO_PROFILES[casino_id]
        assert "branding" in profile
        branding = profile["branding"]
        assert "persona_name" in branding
        assert isinstance(branding["persona_name"], str)
        assert len(branding["persona_name"]) > 0

    @pytest.mark.parametrize("casino_id", ["mohegan_sun", "foxwoods", "hard_rock_ac"])
    def test_profile_has_regulations_section(self, casino_id):
        from src.casino.config import CASINO_PROFILES

        profile = CASINO_PROFILES[casino_id]
        assert "regulations" in profile
        regs = profile["regulations"]
        assert "state" in regs
        assert "gaming_age_minimum" in regs
        assert regs["gaming_age_minimum"] == 21
        assert "responsible_gaming_helpline" in regs
        assert len(regs["responsible_gaming_helpline"]) > 0

    @pytest.mark.parametrize("casino_id", ["mohegan_sun", "foxwoods", "hard_rock_ac"])
    def test_profile_has_operational_section(self, casino_id):
        from src.casino.config import CASINO_PROFILES

        profile = CASINO_PROFILES[casino_id]
        assert "operational" in profile
        ops = profile["operational"]
        assert "timezone" in ops
        assert ops["timezone"] == "America/New_York"

    def test_mohegan_sun_is_ct(self):
        from src.casino.config import CASINO_PROFILES

        assert CASINO_PROFILES["mohegan_sun"]["regulations"]["state"] == "CT"

    def test_foxwoods_is_ct(self):
        from src.casino.config import CASINO_PROFILES

        assert CASINO_PROFILES["foxwoods"]["regulations"]["state"] == "CT"

    def test_hard_rock_ac_is_nj(self):
        from src.casino.config import CASINO_PROFILES

        assert CASINO_PROFILES["hard_rock_ac"]["regulations"]["state"] == "NJ"

    def test_ct_properties_share_helpline(self):
        from src.casino.config import CASINO_PROFILES

        mohegan = CASINO_PROFILES["mohegan_sun"]["regulations"]
        foxwoods = CASINO_PROFILES["foxwoods"]["regulations"]
        assert mohegan["state_helpline"] == "1-888-789-7777"
        assert foxwoods["state_helpline"] == "1-888-789-7777"

    def test_nj_property_has_gambler_helpline(self):
        from src.casino.config import CASINO_PROFILES

        hard_rock = CASINO_PROFILES["hard_rock_ac"]["regulations"]
        assert "1-800-GAMBLER" in hard_rock["responsible_gaming_helpline"]

    def test_nj_property_has_self_exclusion_options(self):
        from src.casino.config import CASINO_PROFILES

        hard_rock = CASINO_PROFILES["hard_rock_ac"]["regulations"]
        assert "self_exclusion_options" in hard_rock
        opts = hard_rock["self_exclusion_options"]
        assert "1-year" in opts
        assert "5-year" in opts
        assert "lifetime" in opts

    def test_each_profile_has_distinct_persona_name(self):
        from src.casino.config import CASINO_PROFILES

        names = [p["branding"]["persona_name"] for p in CASINO_PROFILES.values()]
        assert len(names) == len(set(names)), (
            f"Persona names must be unique across profiles, got: {names}"
        )

    @pytest.mark.parametrize("casino_id", ["mohegan_sun", "foxwoods", "hard_rock_ac"])
    def test_profile_has_prompts_section(self, casino_id):
        from src.casino.config import CASINO_PROFILES

        profile = CASINO_PROFILES[casino_id]
        assert "prompts" in profile
        assert "greeting_template" in profile["prompts"]
        assert "fallback_message" in profile["prompts"]

    @pytest.mark.parametrize("casino_id", ["mohegan_sun", "foxwoods", "hard_rock_ac"])
    def test_profile_has_rag_section(self, casino_id):
        from src.casino.config import CASINO_PROFILES

        profile = CASINO_PROFILES[casino_id]
        assert "rag" in profile
        assert "min_relevance_score" in profile["rag"]
        assert "top_k" in profile["rag"]


# ---------------------------------------------------------------------------
# get_casino_profile tests
# ---------------------------------------------------------------------------


class TestGetCasinoProfile:
    """Validate get_casino_profile lookup behavior."""

    def test_returns_mohegan_sun_profile(self):
        from src.casino.config import get_casino_profile

        profile = get_casino_profile("mohegan_sun")
        assert profile["_id"] == "mohegan_sun"
        assert profile["branding"]["persona_name"] == "Seven"

    def test_returns_foxwoods_profile(self):
        from src.casino.config import get_casino_profile

        profile = get_casino_profile("foxwoods")
        assert profile["_id"] == "foxwoods"
        assert profile["branding"]["persona_name"] == "Foxy"

    def test_returns_hard_rock_ac_profile(self):
        from src.casino.config import get_casino_profile

        profile = get_casino_profile("hard_rock_ac")
        assert profile["_id"] == "hard_rock_ac"
        assert profile["branding"]["persona_name"] == "Ace"

    def test_returns_default_config_for_unknown_casino(self):
        from src.casino.config import DEFAULT_CONFIG, get_casino_profile

        profile = get_casino_profile("nonexistent_casino")
        assert profile is DEFAULT_CONFIG

    def test_returns_default_config_for_empty_string(self):
        from src.casino.config import DEFAULT_CONFIG, get_casino_profile

        profile = get_casino_profile("")
        assert profile is DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Knowledge-base file existence and content tests
# ---------------------------------------------------------------------------


class TestKnowledgeBaseFiles:
    """Validate knowledge-base files exist and contain substantive content."""

    @pytest.mark.parametrize(
        "filename",
        [
            "casino-operations/loyalty-programs.md",
            "casino-operations/dining-guide.md",
            "casino-operations/hotel-operations.md",
            "casino-operations/comp-system.md",
            "casino-operations/host-workflow.md",
        ],
    )
    def test_knowledge_base_file_exists(self, filename):
        filepath = _PROJECT_ROOT / "knowledge-base" / filename
        assert filepath.exists(), f"Knowledge-base file missing: {filepath}"

    @pytest.mark.parametrize(
        "filename",
        [
            "casino-operations/loyalty-programs.md",
            "casino-operations/dining-guide.md",
            "casino-operations/hotel-operations.md",
        ],
    )
    def test_knowledge_base_file_is_non_empty(self, filename):
        filepath = _PROJECT_ROOT / "knowledge-base" / filename
        content = filepath.read_text()
        # Substantive content should be at least 500 characters
        assert len(content) > 500, (
            f"Knowledge-base file too short ({len(content)} chars): {filepath}"
        )

    def test_loyalty_programs_has_mgm_rewards(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "loyalty-programs.md"
        content = filepath.read_text()
        assert "MGM Rewards" in content

    def test_loyalty_programs_has_caesars_rewards(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "loyalty-programs.md"
        content = filepath.read_text()
        assert "Caesars Rewards" in content

    def test_loyalty_programs_has_mohegan_momentum(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "loyalty-programs.md"
        content = filepath.read_text()
        assert "Mohegan Sun Momentum" in content or "Momentum" in content

    def test_loyalty_programs_has_adt_formula(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "loyalty-programs.md"
        content = filepath.read_text()
        assert "ADT" in content
        assert "Theoretical" in content or "theoretical" in content

    def test_dining_guide_has_mohegan_restaurants(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "dining-guide.md"
        content = filepath.read_text()
        assert "Todd English" in content
        assert "Bobby's Burgers" in content or "Bobby" in content

    def test_dining_guide_has_foxwoods_restaurants(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "dining-guide.md"
        content = filepath.read_text()
        assert "Gordon Ramsay" in content or "Hell's Kitchen" in content
        assert "Momosan" in content or "Morimoto" in content

    def test_dining_guide_has_dietary_accommodations(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "dining-guide.md"
        content = filepath.read_text()
        assert "halal" in content.lower()
        assert "kosher" in content.lower()
        assert "vegan" in content.lower()

    def test_hotel_operations_has_earth_tower(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "hotel-operations.md"
        content = filepath.read_text()
        assert "Earth Tower" in content
        assert "365" in content  # 365+ sq ft standard

    def test_hotel_operations_has_sky_tower(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "hotel-operations.md"
        content = filepath.read_text()
        assert "Sky Tower" in content
        assert "10,000" in content or "10000" in content  # solarium

    def test_hotel_operations_has_grand_pequot(self):
        filepath = _PROJECT_ROOT / "knowledge-base" / "casino-operations" / "hotel-operations.md"
        content = filepath.read_text()
        assert "Grand Pequot" in content
        assert "Four Diamond" in content


# ---------------------------------------------------------------------------
# HEART Escalation Language tests
# ---------------------------------------------------------------------------


class TestHeartEscalationLanguage:
    """Validate HEART framework escalation language constant."""

    def test_heart_has_all_five_steps(self):
        from src.agent.prompts import HEART_ESCALATION_LANGUAGE

        expected_steps = {"hear", "empathize", "apologize", "resolve", "thank"}
        assert set(HEART_ESCALATION_LANGUAGE.keys()) == expected_steps

    def test_heart_values_are_nonempty_strings(self):
        from src.agent.prompts import HEART_ESCALATION_LANGUAGE

        for step, phrase in HEART_ESCALATION_LANGUAGE.items():
            assert isinstance(phrase, str), f"Step '{step}' is not a string"
            assert len(phrase) > 10, f"Step '{step}' phrase too short: {phrase!r}"

    def test_hear_step_invites_guest_to_share(self):
        from src.agent.prompts import HEART_ESCALATION_LANGUAGE

        assert "walk me through" in HEART_ESCALATION_LANGUAGE["hear"].lower()

    def test_empathize_step_acknowledges_feelings(self):
        from src.agent.prompts import HEART_ESCALATION_LANGUAGE

        assert "understand" in HEART_ESCALATION_LANGUAGE["empathize"].lower()

    def test_apologize_step_expresses_regret(self):
        from src.agent.prompts import HEART_ESCALATION_LANGUAGE

        assert "sorry" in HEART_ESCALATION_LANGUAGE["apologize"].lower()

    def test_resolve_step_offers_options(self):
        from src.agent.prompts import HEART_ESCALATION_LANGUAGE

        phrase = HEART_ESCALATION_LANGUAGE["resolve"].lower()
        assert "what i can do" in phrase or "which" in phrase

    def test_thank_step_shows_gratitude(self):
        from src.agent.prompts import HEART_ESCALATION_LANGUAGE

        assert "thank" in HEART_ESCALATION_LANGUAGE["thank"].lower()
