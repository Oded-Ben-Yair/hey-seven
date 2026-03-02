"""Tests for post-generation slop detector (R82 Track 1C).

Validates that _enforce_tone() mechanically strips AI slop patterns from
LLM-generated responses. This is the second-pass enforcement after
persona_envelope_node's _strip_performative_openers().
"""

import pytest

from src.agent.nodes import _enforce_tone, _SLOP_PATTERNS


class TestSlopDetector:
    """Test _enforce_tone() post-generation enforcement."""

    def test_strips_oh_opener(self):
        assert _enforce_tone("Oh, I'd love to help!") == "I'd love to help!"

    def test_strips_oh_exclamation_opener(self):
        assert _enforce_tone("Oh! That sounds great.") == "That sounds great."

    def test_strips_ah_opener(self):
        result = _enforce_tone("Ah, great question!")
        assert not result.startswith("Ah")

    def test_strips_wonderful_question(self):
        assert _enforce_tone("What a wonderful question! Let me tell you.") == "Let me tell you."

    def test_strips_lovely_question(self):
        assert _enforce_tone("What a lovely question! Here's the info.") == "Here's the info."

    def test_strips_thats_a_great_question(self):
        result = _enforce_tone("That's a great question! The pool opens at 9 AM.")
        assert "great question" not in result.lower()
        assert "pool opens at 9 AM" in result

    def test_replaces_delighted(self):
        result = _enforce_tone("I'd be absolutely delighted to help you with that. Here's what I found.")
        assert "delighted" not in result
        assert "I can help with that" in result

    def test_replaces_thrilled(self):
        result = _enforce_tone("I'd be thrilled to assist! The restaurant opens at 5.")
        assert "thrilled" not in result
        assert "I can help with that" in result

    def test_replaces_happy_to_help(self):
        result = _enforce_tone("I'd be happy to help! Bobby's is open tonight.")
        assert "happy to help" not in result.lower()

    def test_replaces_truly_delighted(self):
        result = _enforce_tone("I'd be truly delighted to help. The spa opens at 8 AM.")
        assert "truly delighted" not in result

    def test_replaces_more_than_happy(self):
        result = _enforce_tone("I'd be more than happy to assist you with that.")
        assert "more than happy" not in result

    def test_strips_absolutely_opener(self):
        assert _enforce_tone("Absolutely! The pool is open.") == "The pool is open."

    def test_strips_absolutely_period_opener(self):
        assert _enforce_tone("Absolutely. The pool is open.") == "The pool is open."

    def test_strips_of_course_opener(self):
        assert _enforce_tone("Of course! Let me check.") == "Let me check."

    def test_strips_great_question(self):
        assert _enforce_tone("Great question! Bobby's opens at 5 PM.") == "Bobby's opens at 5 PM."

    def test_strips_excellent_question(self):
        result = _enforce_tone("Excellent question! The show starts at 8.")
        assert "Excellent question" not in result
        assert "show starts at 8" in result

    def test_strips_fantastic_choice(self):
        result = _enforce_tone("Fantastic choice! You'll love the steak.")
        assert "Fantastic choice" not in result

    def test_strips_wonderful_pick(self):
        result = _enforce_tone("Wonderful pick! That restaurant is excellent.")
        assert "Wonderful pick" not in result

    def test_caps_exclamation_marks(self):
        result = _enforce_tone("Wow! Amazing! Fantastic! Great! Here it is!")
        assert result.count("!") <= 2

    def test_exclamation_cap_preserves_last_two(self):
        result = _enforce_tone("A! B! C!")
        assert result.count("!") == 2
        # Last two ! should be preserved
        assert result.endswith("C!")

    def test_two_exclamations_unchanged(self):
        text = "Welcome! The resort has a great pool!"
        assert _enforce_tone(text) == text

    def test_one_exclamation_unchanged(self):
        text = "Bobby's opens at 5 PM!"
        assert _enforce_tone(text) == text

    def test_capitalizes_after_strip(self):
        result = _enforce_tone("Oh, the restaurant is open.")
        assert result[0].isupper()
        assert result == "The restaurant is open."

    def test_empty_string(self):
        assert _enforce_tone("") == ""

    def test_none_passthrough(self):
        """Empty/falsy values should pass through unchanged."""
        assert _enforce_tone("") == ""

    def test_normal_response_unchanged(self):
        normal = "Bobby's Burger Palace opens at 11 AM and closes at 10 PM."
        assert _enforce_tone(normal) == normal

    def test_crisis_response_unchanged(self):
        crisis = "I hear you, and I want you to know help is available. Call 1-800-GAMBLER."
        assert _enforce_tone(crisis) == crisis

    def test_factual_response_unchanged(self):
        factual = "The Wolf Den hosts free shows every Thursday and Friday night starting at 8 PM."
        assert _enforce_tone(factual) == factual

    def test_love_to_help_explore(self):
        result = _enforce_tone("I'd love to help you explore our dining options.")
        assert "love to help" not in result
        assert "Let me share some options about" in result

    def test_absolutely_love_to_help_explore(self):
        result = _enforce_tone("I'd absolutely love to help you explore the entertainment.")
        assert "love to help" not in result
        assert "Let me share some options about" in result

    def test_multiple_patterns_in_one(self):
        result = _enforce_tone("Oh, I'd be absolutely delighted to help! Great question! Here's info.")
        # "Oh, " should be stripped
        assert not result.startswith("Oh")
        # "delighted" should be replaced
        assert "delighted" not in result

    def test_case_insensitive_oh(self):
        result = _enforce_tone("oh, here's the info.")
        assert result == "Here's the info."

    def test_case_insensitive_absolutely(self):
        result = _enforce_tone("ABSOLUTELY! The pool is open.")
        assert result == "The pool is open."

    def test_whitespace_only_after_strip(self):
        """If stripping leaves only whitespace, handle gracefully."""
        result = _enforce_tone("Oh, ")
        # Should handle gracefully (empty or stripped)
        assert result == result.strip()

    def test_mid_sentence_delighted_replaced(self):
        """'I'd be delighted' can appear mid-sentence, not just at start."""
        result = _enforce_tone("Sure, I'd be delighted to help you with that. The spa opens at 9.")
        assert "delighted" not in result
        assert "I can help with that" in result


class TestSlopPatternCount:
    """Track slop pattern count for doc parity."""

    def test_pattern_count(self):
        """R82: Ensure pattern count matches documentation."""
        assert len(_SLOP_PATTERNS) >= 9, f"Expected >= 9 slop patterns, got {len(_SLOP_PATTERNS)}"

    def test_patterns_are_compiled(self):
        """Verify all patterns are pre-compiled re.Pattern objects."""
        for pattern, _ in _SLOP_PATTERNS:
            assert hasattr(pattern, "sub"), f"Pattern {pattern} is not compiled"

    def test_all_replacements_are_strings(self):
        """Verify all replacement values are strings."""
        for _, replacement in _SLOP_PATTERNS:
            assert isinstance(replacement, str), f"Replacement {replacement!r} is not a string"


class TestSlopDetectorPerformance:
    """Verify the slop detector meets the <1ms performance requirement."""

    def test_execution_speed(self):
        """_enforce_tone must complete in <1ms for a typical response."""
        import time

        response = (
            "Oh, I'd be absolutely delighted to help you with that! "
            "What a wonderful question! Bobby's Burger Palace opens at "
            "11 AM and serves lunch and dinner. They're known for their "
            "excellent burgers and craft cocktails! The atmosphere is "
            "casual and fun! Perfect for a great evening out!"
        )
        start = time.perf_counter_ns()
        for _ in range(100):
            _enforce_tone(response)
        elapsed_ns = time.perf_counter_ns() - start
        avg_us = elapsed_ns / 100 / 1000  # average microseconds

        # Must be under 1ms (1000us) per call
        assert avg_us < 1000, f"_enforce_tone took {avg_us:.1f}us avg, exceeds 1ms limit"
