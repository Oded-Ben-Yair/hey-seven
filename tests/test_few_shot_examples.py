"""Tests for few-shot example library and comp agent prompt rewrite (R82 Track 2A/2B)."""

import pytest

from src.agent.prompts import FEW_SHOT_EXAMPLES
from src.agent.agents.comp_agent import COMP_SYSTEM_PROMPT


class TestFewShotExamples:
    """Test few-shot example library structure and content."""

    def test_five_specialists(self):
        assert set(FEW_SHOT_EXAMPLES.keys()) == {
            "dining",
            "entertainment",
            "hotel",
            "comp",
            "host",
        }

    def test_min_five_examples_per_specialist(self):
        """R94: host now has 7 examples (5 base + 2 R94 VIP/loss recovery)."""
        for specialist, examples in FEW_SHOT_EXAMPLES.items():
            assert len(examples) >= 5, (
                f"{specialist} has {len(examples)} examples, expected >= 5"
            )

    def test_examples_are_tuples(self):
        for specialist, examples in FEW_SHOT_EXAMPLES.items():
            for i, ex in enumerate(examples):
                assert isinstance(ex, tuple) and len(ex) == 2, (
                    f"{specialist}[{i}] is not a 2-tuple"
                )

    def test_no_slop_in_ideal_responses(self):
        """Ideal responses must not contain the slop patterns we detect."""
        slop_phrases = [
            "I'd be delighted",
            "I'd be happy to",
            "What a wonderful question",
            "Absolutely!",
            "Of course!",
            "I'd love to help you explore",
        ]
        for specialist, examples in FEW_SHOT_EXAMPLES.items():
            for user_msg, ideal in examples:
                for slop in slop_phrases:
                    assert slop not in ideal, (
                        f"Slop '{slop}' found in {specialist} ideal response"
                    )

    def test_no_empty_examples(self):
        for specialist, examples in FEW_SHOT_EXAMPLES.items():
            for user_msg, ideal in examples:
                assert len(user_msg) > 10, f"Empty user message in {specialist}"
                assert len(ideal) > 20, f"Empty ideal response in {specialist}"

    def test_total_example_count(self):
        total = sum(len(examples) for examples in FEW_SHOT_EXAMPLES.values())
        assert total == 27, f"Expected 27 total examples (25 R83 + 2 R94), got {total}"


class TestCompPromptRewrite:
    """Test comp agent prompt no longer uses promotional language."""

    def test_no_promotional_phrases(self):
        prompt_text = COMP_SYSTEM_PROMPT.safe_substitute(
            property_name="Test Casino",
            current_time="2026-03-02 15:00",
            responsible_gaming_helplines="1-800-GAMBLER",
            property_description="A test casino.",
        )
        # Phrases that the OLD prompt used promotionally. The new prompt may
        # reference some of these in a "NEVER use" instruction — that's fine.
        # We check that the OLD promotional context sentences are gone.
        banned_sentences = [
            "trusted rewards insider",
            "benefits really start to shine",
            "make offers sound appealing",
            # Old prompt: "convey exclusive access: 'At the higher tiers...'"
            "convey exclusive access",
            # Old prompt: enthusiastic lead-in style
            "inside track",
            "great perks",
        ]
        for phrase in banned_sentences:
            assert phrase not in prompt_text, (
                f"Banned phrase '{phrase}' still in comp prompt"
            )

    def test_has_factual_guidance(self):
        prompt_text = COMP_SYSTEM_PROMPT.safe_substitute(
            property_name="Test Casino",
            current_time="2026-03-02 15:00",
            responsible_gaming_helplines="1-800-GAMBLER",
            property_description="A test casino.",
        )
        assert "factual" in prompt_text.lower() or "direct" in prompt_text.lower()

    def test_frustrated_guest_guidance(self):
        prompt_text = COMP_SYSTEM_PROMPT.safe_substitute(
            property_name="Test Casino",
            current_time="2026-03-02 15:00",
            responsible_gaming_helplines="1-800-GAMBLER",
            property_description="A test casino.",
        )
        assert "frustrated" in prompt_text.lower()
