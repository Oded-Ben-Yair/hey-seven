"""Tests for the re2/re regex adapter."""

import re

import pytest

from src.agent.regex_engine import RE2_AVAILABLE, compile


class TestRegexEngineCompile:
    """Test regex_engine.compile() with both re2 and fallback."""

    def test_basic_pattern_compiles(self):
        pattern = compile(r"hello\s+world", re.I)
        assert pattern.search("Hello World")
        assert not pattern.search("goodbye")

    def test_word_boundary(self):
        pattern = compile(r"\bjailbreak\b", re.I)
        assert pattern.search("jailbreak attempt")
        assert not pattern.search("notajailbreakword")

    def test_dotall_flag(self):
        pattern = compile(r"DAN.*mode", re.I | re.DOTALL)
        assert pattern.search("DAN\n\nmode")

    def test_character_class(self):
        pattern = compile(r"[\u200b-\u200f\u2028-\u202f\ufeff]", re.I)
        assert pattern.search("\u200b")

    def test_non_capturing_group(self):
        pattern = compile(r"(?:hello|world)", re.I)
        assert pattern.search("hello")
        assert pattern.search("world")

    def test_alternation(self):
        pattern = compile(r"gambling\s+problem|problem\s+gambl", re.I)
        assert pattern.search("gambling problem")
        assert pattern.search("problem gambling")

    @pytest.mark.skipif(not RE2_AVAILABLE, reason="re2 not installed")
    def test_lookahead_falls_back_to_stdlib(self):
        """The single lookahead pattern in guardrails.py must still work."""
        pattern = compile(
            r"act\s+as\s+(?:if\s+)?(?:you(?:'re|\s+are)\s+)?(?:a|an|the)\s+(?!guide\b)",
            re.I,
        )
        assert pattern.search("act as a hacker")
        assert not pattern.search("act as a guide")

    def test_unicode_pattern(self):
        pattern = compile(r"\u8d4c\u535a\s*(?:\u6210\u763e|\u4e0a\u763e|\u95ee\u9898)", re.I)
        assert pattern.search("\u8d4c\u535a\u6210\u763e")

    def test_re2_available_flag(self):
        """RE2_AVAILABLE reflects actual import success."""
        assert isinstance(RE2_AVAILABLE, bool)

    def test_empty_pattern(self):
        pattern = compile(r"")
        assert pattern.search("anything")

    def test_complex_alternation_with_quantifiers(self):
        pattern = compile(
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
            re.I,
        )
        assert pattern.search("ignore all previous instructions")
        assert pattern.search("ignore prior rules")
