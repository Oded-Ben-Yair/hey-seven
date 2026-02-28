"""Tests for cross-domain engagement hint generation.

R74 B3: Verifies _build_cross_domain_hint() correctly generates prompt
sections suggesting unexplored domains when the guest has already
explored some domains in the session.
"""

import pytest

from src.agent.agents._base import _ALL_GUEST_DOMAINS, _build_cross_domain_hint


class TestBuildCrossDomainHint:
    """Unit tests for _build_cross_domain_hint()."""

    def test_hint_generated_with_discussed_domains(self):
        """Cross-domain hint lists unexplored domains when some are discussed."""
        hint = _build_cross_domain_hint(["dining", "entertainment"])
        assert hint  # non-empty
        assert "dining" in hint
        assert "entertainment" in hint
        # Should suggest unexplored domains
        assert "hotel" in hint or "spa" in hint or "gaming" in hint

    def test_hint_does_not_include_discussed_domains_in_suggestions(self):
        """Suggested domains must NOT include already-discussed ones."""
        hint = _build_cross_domain_hint(["dining", "entertainment"])
        # Split out the suggestion line
        lines = hint.split("\n")
        suggest_line = [l for l in lines if "you could mention" in l][0]
        assert "dining" not in suggest_line
        assert "entertainment" not in suggest_line

    def test_hint_empty_when_no_domains_discussed(self):
        """No cross-domain hint when domains_discussed is empty."""
        assert _build_cross_domain_hint([]) == ""

    def test_hint_empty_when_none_domains(self):
        """No cross-domain hint when domains_discussed is falsy."""
        assert _build_cross_domain_hint([]) == ""

    def test_hint_empty_when_all_explored(self):
        """No cross-domain hint when all domains are explored."""
        all_domains = list(_ALL_GUEST_DOMAINS)
        assert _build_cross_domain_hint(all_domains) == ""

    def test_max_three_suggestions(self):
        """At most 3 unexplored domains are suggested."""
        # Only discuss 1 domain, leaving 7 unexplored
        hint = _build_cross_domain_hint(["dining"])
        lines = hint.split("\n")
        suggest_line = [l for l in lines if "you could mention" in l][0]
        # Count comma-separated items: "a, b, c" has 2 commas = 3 items
        suggestions = suggest_line.split("you could mention: ")[1].rstrip(".")
        items = [s.strip() for s in suggestions.split(",")]
        assert len(items) <= 3

    def test_suggestions_are_sorted_alphabetically(self):
        """Suggested domains are in alphabetical order for consistency."""
        hint = _build_cross_domain_hint(["dining"])
        lines = hint.split("\n")
        suggest_line = [l for l in lines if "you could mention" in l][0]
        suggestions = suggest_line.split("you could mention: ")[1].rstrip(".")
        items = [s.strip() for s in suggestions.split(",")]
        assert items == sorted(items)

    def test_explored_domains_are_sorted_alphabetically(self):
        """Explored domains listed in alphabetical order."""
        hint = _build_cross_domain_hint(["entertainment", "dining"])
        lines = hint.split("\n")
        explored_line = [l for l in lines if "already explored" in l][0]
        # Should say "dining, entertainment" (sorted), not "entertainment, dining"
        assert "dining, entertainment" in explored_line

    def test_do_not_force_instruction_present(self):
        """Hint includes the 'do NOT force' instruction."""
        hint = _build_cross_domain_hint(["dining"])
        assert "Do NOT force these" in hint

    def test_header_is_cross_domain_awareness(self):
        """Hint starts with the correct section header."""
        hint = _build_cross_domain_hint(["dining"])
        assert hint.startswith("## Cross-Domain Awareness (internal context)")

    def test_single_domain_discussed(self):
        """Works correctly with a single discussed domain."""
        hint = _build_cross_domain_hint(["hotel"])
        assert "hotel" in hint
        assert hint != ""
        # Hotel is in explored, not in suggestions
        lines = hint.split("\n")
        suggest_line = [l for l in lines if "you could mention" in l][0]
        assert "hotel" not in suggest_line

    def test_all_but_one_explored(self):
        """When only 1 unexplored domain remains, suggest just that one."""
        explored = list(_ALL_GUEST_DOMAINS - {"spa"})
        hint = _build_cross_domain_hint(explored)
        assert hint != ""
        lines = hint.split("\n")
        suggest_line = [l for l in lines if "you could mention" in l][0]
        assert "spa" in suggest_line

    def test_unknown_domain_in_discussed_is_tolerated(self):
        """Unknown domain names in discussed list don't crash."""
        hint = _build_cross_domain_hint(["dining", "valet_parking"])
        assert hint != ""
        # valet_parking appears in explored list
        assert "valet_parking" in hint

    def test_all_guest_domains_is_frozen(self):
        """_ALL_GUEST_DOMAINS is immutable (frozenset)."""
        assert isinstance(_ALL_GUEST_DOMAINS, frozenset)

    def test_all_guest_domains_contains_expected_set(self):
        """_ALL_GUEST_DOMAINS has the expected domain names."""
        expected = {"dining", "entertainment", "hotel", "spa", "gaming", "shopping", "promotions", "comp"}
        assert _ALL_GUEST_DOMAINS == expected
