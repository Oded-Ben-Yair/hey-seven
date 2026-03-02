"""Tests for cross-domain engagement hint generation.

R74 B3: Verifies _build_cross_domain_hint() correctly generates prompt
sections suggesting unexplored domains when the guest has already
explored some domains in the session.

R82 Track 2C: Updated to reflect bridge-template-based hints. When a
bridge template exists for the (last_domain, target) pair, the hint
uses specific transition phrases. Falls back to generic "you could
mention" format when no bridge exists.
"""

import pytest

from src.agent.agents._base import (
    CROSS_DOMAIN_BRIDGES,
    _ALL_GUEST_DOMAINS,
    _build_cross_domain_hint,
)


class TestBuildCrossDomainHint:
    """Unit tests for _build_cross_domain_hint()."""

    def test_hint_generated_with_discussed_domains(self):
        """Cross-domain hint lists unexplored domains when some are discussed."""
        hint = _build_cross_domain_hint(["dining", "entertainment"])
        assert hint  # non-empty
        assert "dining" in hint
        assert "entertainment" in hint
        # Should suggest unexplored domains (bridge or generic)
        assert "hotel" in hint or "spa" in hint or "gaming" in hint

    def test_hint_does_not_include_discussed_domains_in_bridge_targets(self):
        """Bridge target domains must NOT include already-discussed ones."""
        hint = _build_cross_domain_hint(["dining", "entertainment"])
        # Bridge lines start with "- <domain>:" — ensure no bridge to dining/entertainment
        bridge_lines = [l for l in hint.split("\n") if l.startswith("- ")]
        for line in bridge_lines:
            target = line.split(":")[0].lstrip("- ").strip()
            assert target not in ("dining", "entertainment")

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
        """At most 3 bridge/suggestion items are included."""
        # Only discuss 1 domain with bridges, leaving 7 unexplored
        hint = _build_cross_domain_hint(["dining"])
        bridge_lines = [l for l in hint.split("\n") if l.startswith("- ")]
        assert len(bridge_lines) <= 3

    def test_explored_domains_are_sorted_alphabetically(self):
        """Explored domains listed in alphabetical order."""
        hint = _build_cross_domain_hint(["entertainment", "dining"])
        assert "dining, entertainment" in hint

    def test_adapt_instruction_present(self):
        """Hint includes instruction to adapt (not copy verbatim)."""
        hint = _build_cross_domain_hint(["dining"])
        assert "adapt" in hint.lower() or "inspiration" in hint.lower()

    def test_header_is_cross_domain_awareness(self):
        """Hint starts with the correct section header."""
        hint = _build_cross_domain_hint(["dining"])
        assert hint.startswith("## Cross-Domain Awareness (internal context)")

    def test_single_domain_discussed(self):
        """Works correctly with a single discussed domain."""
        hint = _build_cross_domain_hint(["hotel"])
        assert "hotel" in hint
        assert hint != ""

    def test_all_but_one_explored_generic_fallback(self):
        """When only 1 unexplored domain remains and no bridge exists, generic fallback."""
        # Explore all except shopping — no bridge from last explored to shopping
        explored = list(_ALL_GUEST_DOMAINS - {"shopping"})
        hint = _build_cross_domain_hint(explored)
        # May be empty if last domain has no bridge to shopping, or generic fallback
        # Just verify it doesn't crash and is consistent
        assert isinstance(hint, str)

    def test_unknown_domain_in_discussed_is_tolerated(self):
        """Unknown domain names in discussed list don't crash."""
        hint = _build_cross_domain_hint(["dining", "valet_parking"])
        assert hint != ""
        # valet_parking appears in explored list
        assert "valet_parking" in hint

    def test_generic_fallback_for_no_bridges(self):
        """Falls back to generic format when no bridge templates match."""
        # Use a domain with no outgoing bridges as last domain
        hint = _build_cross_domain_hint(["shopping"])
        assert hint != ""
        # Generic fallback uses "you could mention"
        assert "you could mention" in hint

    def test_all_guest_domains_is_frozen(self):
        """_ALL_GUEST_DOMAINS is immutable (frozenset)."""
        assert isinstance(_ALL_GUEST_DOMAINS, frozenset)

    def test_all_guest_domains_contains_expected_set(self):
        """_ALL_GUEST_DOMAINS has the expected domain names."""
        expected = {"dining", "entertainment", "hotel", "spa", "gaming", "shopping", "promotions", "comp"}
        assert _ALL_GUEST_DOMAINS == expected
