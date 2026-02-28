"""Tests for B6 tone calibration — performative opener stripping.

Covers: stripping "Absolutely!", "Oh,", "I'd be delighted", preserving
mid-sentence "Oh", exclamation clustering reduction, no-op for clean text.
"""

from src.agent.persona import _strip_performative_openers


# ---------------------------------------------------------------------------
# Performative opener stripping
# ---------------------------------------------------------------------------


class TestStripPerformativeOpeners:
    """Performative openers at response start should be stripped."""

    def test_strip_absolutely(self):
        result = _strip_performative_openers("Absolutely! Here is the info.")
        assert not result.startswith("Absolutely")
        assert "Here is the info." in result

    def test_strip_oh_opener(self):
        result = _strip_performative_openers("Oh, what a wonderful resort this is.")
        assert not result.startswith("Oh,")
        # First letter should be capitalized after stripping
        assert result[0].isupper()

    def test_strip_delighted(self):
        result = _strip_performative_openers("I'd be absolutely delighted to help you with that.")
        assert "delighted" not in result.lower()

    def test_strip_great_question(self):
        result = _strip_performative_openers("Great question! Todd English's Tuscany is excellent.")
        assert not result.startswith("Great question")
        assert "Todd English" in result

    def test_strip_of_course(self):
        result = _strip_performative_openers("Of course! The steakhouse opens at 5 PM.")
        assert not result.startswith("Of course")
        assert "steakhouse" in result

    def test_strip_sure_thing(self):
        result = _strip_performative_openers("Sure thing! Let me look that up for you.")
        assert not result.startswith("Sure thing")

    def test_strip_what_wonderful_question(self):
        result = _strip_performative_openers("What a wonderful question! The spa hours are 9-9.")
        assert not result.startswith("What a wonderful")
        assert "spa" in result.lower()


# ---------------------------------------------------------------------------
# Preserve mid-sentence "Oh"
# ---------------------------------------------------------------------------


class TestPreserveMidSentence:
    """Oh in mid-sentence position should NOT be stripped."""

    def test_preserve_mid_sentence_oh(self):
        text = "The pool area? Oh that's on the second floor."
        result = _strip_performative_openers(text)
        # "Oh" is not at the start, so it should be preserved
        assert result == text

    def test_preserve_regular_sentence(self):
        text = "Todd English's Tuscany is our signature Italian restaurant."
        result = _strip_performative_openers(text)
        assert result == text


# ---------------------------------------------------------------------------
# Exclamation clustering in first paragraph
# ---------------------------------------------------------------------------


class TestExclamationClustering:
    """3+ exclamations in first paragraph should be reduced to 1."""

    def test_three_exclamations_reduced(self):
        text = "Amazing! Wonderful! Great! Here are the dining options."
        result = _strip_performative_openers(text)
        # After stripping openers and reducing exclamations,
        # should have at most 1 exclamation in first paragraph
        first_para_end = result.find('\n\n')
        if first_para_end == -1:
            first_para_end = len(result)
        first_para = result[:first_para_end]
        assert first_para.count('!') <= 1

    def test_two_exclamations_preserved(self):
        # Two exclamations should NOT be reduced (threshold is 3)
        text = "Welcome! The resort has great dining options!"
        result = _strip_performative_openers(text)
        # "Welcome" is not a performative opener, so text should be mostly preserved
        assert result.count('!') == 2


# ---------------------------------------------------------------------------
# No stripping needed
# ---------------------------------------------------------------------------


class TestNoStrippingNeeded:
    """Clean text without performative openers should be unchanged."""

    def test_clean_recommendation(self):
        text = "Todd English's Tuscany is our signature Italian restaurant, open for dinner from 5 PM to 10 PM."
        result = _strip_performative_openers(text)
        assert result == text

    def test_informational_response(self):
        text = "The pool is located on the second floor and is open from 8 AM to 10 PM daily."
        result = _strip_performative_openers(text)
        assert result == text

    def test_empty_string(self):
        result = _strip_performative_openers("")
        assert result == ""

    def test_short_response(self):
        text = "The spa opens at 9 AM."
        result = _strip_performative_openers(text)
        assert result == text


# ---------------------------------------------------------------------------
# Chained opener stripping
# ---------------------------------------------------------------------------


class TestChainedOpeners:
    """Multiple chained openers should all be stripped."""

    def test_oh_absolutely(self):
        result = _strip_performative_openers("Oh, absolutely! The steakhouse is excellent.")
        assert not result.startswith("Oh")
        assert not result.startswith("Absolutely")
        assert "steakhouse" in result.lower()
