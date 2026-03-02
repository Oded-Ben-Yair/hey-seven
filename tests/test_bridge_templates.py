"""Tests for cross-domain bridges and proactivity gate tuning (R82 Track 2C/1F).

Track 2C: Cross-domain bridge templates — specific transition phrases
between domain pairs, injected into the system prompt when the guest has
discussed domain A and domain B is available.

Track 1F: Proactivity gate instrumentation + threshold tuning — lowered
confidence threshold from 0.8 to 0.6, removed extra gates for neutral
sentiment.
"""

import pytest

from src.agent.agents._base import (
    CROSS_DOMAIN_BRIDGES,
    _build_cross_domain_hint,
    _should_inject_suggestion,
)


# ---------------------------------------------------------------------------
# Track 2C: Cross-domain bridge templates
# ---------------------------------------------------------------------------


class TestCrossDomainBridges:
    """Test bridge template structure and content."""

    def test_bridge_count(self):
        """At least 15 bridge templates defined."""
        assert len(CROSS_DOMAIN_BRIDGES) >= 15

    def test_bridges_are_string_tuples(self):
        """Each key is a 2-tuple of strings, each value is a non-trivial string."""
        for key, value in CROSS_DOMAIN_BRIDGES.items():
            assert isinstance(key, tuple) and len(key) == 2, f"Bad key: {key}"
            assert isinstance(key[0], str) and isinstance(key[1], str)
            assert isinstance(value, str) and len(value) > 10, f"Bridge text too short: {value}"

    def test_dining_to_entertainment_exists(self):
        """Core bridge from dining to entertainment is defined."""
        assert ("dining", "entertainment") in CROSS_DOMAIN_BRIDGES

    def test_entertainment_to_dining_exists(self):
        """Reverse bridge from entertainment to dining is defined."""
        assert ("entertainment", "dining") in CROSS_DOMAIN_BRIDGES

    def test_no_self_bridges(self):
        """No bridge from a domain to itself."""
        for (src, tgt) in CROSS_DOMAIN_BRIDGES:
            assert src != tgt, f"Self-bridge found: {src} -> {tgt}"

    def test_gaming_to_dining_exists(self):
        """Gaming to dining bridge references real venue names."""
        bridge = CROSS_DOMAIN_BRIDGES.get(("gaming", "dining"))
        assert bridge is not None
        assert "Bobby" in bridge or "Tao" in bridge

    def test_comp_bridges_mention_tier(self):
        """Comp-related bridges reference loyalty tiers or rewards."""
        comp_bridges = {k: v for k, v in CROSS_DOMAIN_BRIDGES.items() if k[0] == "comp"}
        assert len(comp_bridges) >= 2
        for (_, _), text in comp_bridges.items():
            assert "tier" in text.lower() or "reward" in text.lower() or "credit" in text.lower()


class TestBuildHintWithBridges:
    """Test _build_cross_domain_hint with bridge template integration."""

    def test_build_hint_with_bridges(self):
        """Dining domain produces bridge-based hint (not generic)."""
        result = _build_cross_domain_hint(["dining"])
        assert "Cross-Domain" in result
        assert len(result) > 50

    def test_build_hint_empty_domains(self):
        """Empty domain list produces empty hint."""
        result = _build_cross_domain_hint([])
        assert result == ""

    def test_build_hint_all_explored(self):
        """All domains explored produces empty hint."""
        all_domains = ["dining", "entertainment", "hotel", "spa",
                       "gaming", "shopping", "promotions", "comp"]
        result = _build_cross_domain_hint(all_domains)
        assert result == ""

    def test_build_hint_uses_specific_bridges_for_dining(self):
        """Dining domain hint contains specific bridge text, not generic."""
        result = _build_cross_domain_hint(["dining"])
        # Should contain specific bridge template text
        assert "transitions" in result.lower() or "inspiration" in result.lower()
        # Should reference at least one target domain
        assert "entertainment" in result.lower() or "spa" in result.lower() or "hotel" in result.lower()

    def test_build_hint_falls_back_to_generic_for_unknown_domain(self):
        """Unknown domain as last_domain falls back to generic hint."""
        result = _build_cross_domain_hint(["valet_parking"])
        assert result != ""
        # Generic fallback uses "you could mention" phrasing
        assert "you could mention" in result

    def test_build_hint_max_three_bridges(self):
        """At most 3 bridge templates suggested."""
        result = _build_cross_domain_hint(["dining"])
        bridge_lines = [line for line in result.split("\n") if line.startswith("- ")]
        assert len(bridge_lines) <= 3

    def test_build_hint_explored_str_sorted(self):
        """Explored domains listed in alphabetical order."""
        result = _build_cross_domain_hint(["entertainment", "dining"])
        assert "dining, entertainment" in result

    def test_build_hint_header_present(self):
        """Hint starts with correct section header."""
        result = _build_cross_domain_hint(["hotel"])
        assert result.startswith("## Cross-Domain Awareness (internal context)")

    def test_build_hint_adapt_instruction(self):
        """Bridge-based hint includes 'adapt' instruction."""
        result = _build_cross_domain_hint(["dining"])
        assert "adapt" in result.lower() or "inspiration" in result.lower()

    def test_build_hint_entertainment_bridges(self):
        """Entertainment as last domain produces entertainment bridges."""
        result = _build_cross_domain_hint(["entertainment"])
        # entertainment has bridges to dining, hotel, spa
        assert "dining" in result.lower() or "hotel" in result.lower() or "spa" in result.lower()

    def test_build_hint_single_explored_domain(self):
        """Single explored domain still generates hint."""
        result = _build_cross_domain_hint(["spa"])
        assert result != ""
        assert "spa" in result


# ---------------------------------------------------------------------------
# Track 1F: Proactivity gate threshold tuning
# ---------------------------------------------------------------------------


class TestProactivityGateThresholds:
    """Test proactivity gate with lowered thresholds (R82 1F)."""

    def test_confidence_06_passes(self):
        """Confidence 0.6 should now pass (was blocked at 0.8)."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "0.65",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "positive", {"turn_count": 3})
        assert offered is True
        assert "spa" in section.lower()

    def test_confidence_exact_06_passes(self):
        """Confidence exactly 0.6 should pass (>= not >)."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "0.6",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "positive", {"turn_count": 1})
        assert offered is True

    def test_confidence_059_blocked(self):
        """Confidence 0.59 should be blocked (below 0.6)."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "0.59",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "positive", {"turn_count": 3})
        assert offered is False

    def test_confidence_05_blocked(self):
        """Confidence 0.5 should still be blocked."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "0.5",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "positive", {"turn_count": 3})
        assert offered is False

    def test_neutral_sentiment_now_passes(self):
        """Neutral sentiment should pass without extra gates (R82 1F)."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Check out the show tonight",
                "suggestion_confidence": "0.7",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "neutral", {"turn_count": 1})
        assert offered is True

    def test_neutral_no_occasion_no_turns_still_passes(self):
        """Neutral sentiment passes even without occasion or 3+ turns (R82 1F relaxation)."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the buffet",
                "suggestion_confidence": "0.65",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
            "extracted_fields": {},  # no occasion
        }
        # turn_count=1, no occasion -- old code would block, new code allows
        section, offered = _should_inject_suggestion(state, "neutral", {"turn_count": 1})
        assert offered is True

    def test_frustrated_still_blocked(self):
        """Frustrated sentiment should still block proactive suggestions."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "0.9",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "frustrated", {"turn_count": 5})
        assert offered is False

    def test_negative_still_blocked(self):
        """Negative sentiment should still block proactive suggestions."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "0.9",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "negative", {"turn_count": 5})
        assert offered is False

    def test_none_sentiment_still_blocked(self):
        """None sentiment should still block proactive suggestions."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "0.9",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, None, {"turn_count": 5})
        assert offered is False

    def test_no_whisper_plan(self):
        """Missing whisper_plan blocks injection."""
        state = {}
        section, offered = _should_inject_suggestion(state, "positive", {"turn_count": 3})
        assert offered is False

    def test_already_offered(self):
        """suggestion_offered=True blocks injection (max 1 per session)."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "0.9",
            },
            "suggestion_offered": True,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "positive", {"turn_count": 3})
        assert offered is False

    def test_no_retrieved_context(self):
        """Empty retrieved_context blocks injection (no grounding)."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "0.9",
            },
            "suggestion_offered": False,
            "retrieved_context": [],
        }
        section, offered = _should_inject_suggestion(state, "positive", {"turn_count": 3})
        assert offered is False

    def test_invalid_confidence_string(self):
        """Non-numeric confidence string defaults to 0.0 (blocked)."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Try the spa",
                "suggestion_confidence": "not-a-number",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "positive", {"turn_count": 3})
        assert offered is False

    def test_suggestion_text_in_output(self):
        """The proactive suggestion text appears in the output section."""
        state = {
            "whisper_plan": {
                "proactive_suggestion": "Visit Elemis Spa for aromatherapy",
                "suggestion_confidence": "0.8",
            },
            "suggestion_offered": False,
            "retrieved_context": [{"content": "test"}],
        }
        section, offered = _should_inject_suggestion(state, "positive", {"turn_count": 3})
        assert offered is True
        assert "Elemis Spa" in section
        assert "Proactive Suggestion" in section
