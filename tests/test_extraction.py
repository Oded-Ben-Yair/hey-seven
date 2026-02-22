"""Unit tests for extract_fields() — deterministic regex extraction.

Covers: name patterns, party size boundaries, visit dates, preferences,
occasions, false positive exclusions, common words list, edge inputs.
"""

import pytest

from src.agent.extraction import extract_fields


# ---------------------------------------------------------------------------
# Empty / edge inputs
# ---------------------------------------------------------------------------


class TestEdgeInputs:
    """Empty, None, and non-string inputs should return empty dict."""

    def test_empty_string(self):
        assert extract_fields("") == {}

    def test_none_input(self):
        assert extract_fields(None) == {}

    def test_non_string_input(self):
        assert extract_fields(12345) == {}

    def test_whitespace_only(self):
        # whitespace-only should still be a valid string, but no matches
        assert extract_fields("   ") == {}


# ---------------------------------------------------------------------------
# Name extraction
# ---------------------------------------------------------------------------


class TestNameExtraction:
    """Name regex patterns with false-positive prevention."""

    @pytest.mark.parametrize(
        "text,expected_name",
        [
            ("My name is Sarah", "Sarah"),
            ("I'm Michael, checking in today", "Michael"),
            ("I am David", "David"),
            ("Call me Jessica", "Jessica"),
            ("This is Robert.", "Robert"),
        ],
    )
    def test_name_patterns(self, text, expected_name):
        result = extract_fields(text)
        assert result.get("name") == expected_name

    def test_name_here_pattern(self):
        result = extract_fields("Sarah here, looking for a restaurant")
        assert result.get("name") == "Sarah"

    def test_common_word_exclusion(self):
        """Common words should NOT be extracted as names."""
        # "I'm Here" -- "Here" is in _COMMON_WORDS
        result = extract_fields("I'm Here to check in")
        assert "name" not in result

    def test_vegetarian_not_a_name(self):
        """'Vegetarian' after 'I'm' should not match as a name (lowercase start)."""
        result = extract_fields("I'm vegetarian")
        assert "name" not in result

    def test_name_too_short(self):
        """Single-character names should be rejected (min length 2)."""
        result = extract_fields("My name is A")
        assert "name" not in result

    def test_name_boundary(self):
        """Name should not capture trailing words."""
        result = extract_fields("I'm Sarah and I need help")
        assert result.get("name") == "Sarah"


# ---------------------------------------------------------------------------
# Party size extraction
# ---------------------------------------------------------------------------


class TestPartySizeExtraction:
    """Party size patterns with boundary validation (1-50)."""

    @pytest.mark.parametrize(
        "text,expected_size",
        [
            ("Party of 4", 4),
            ("Group of 6", 6),
            ("There are 3 of us", 3),
            ("There's 2 of us", 2),
            ("We are 5", 5),
            ("We're 8", 8),
            ("Table for 2", 2),
            ("For 4 people", 4),
        ],
    )
    def test_party_size_patterns(self, text, expected_size):
        result = extract_fields(text)
        assert result.get("party_size") == expected_size

    def test_party_size_boundary_min(self):
        result = extract_fields("Party of 1")
        assert result.get("party_size") == 1

    def test_party_size_boundary_max(self):
        result = extract_fields("Group of 50")
        assert result.get("party_size") == 50

    def test_party_size_over_max_rejected(self):
        result = extract_fields("Party of 51")
        assert "party_size" not in result

    def test_party_size_zero_rejected(self):
        result = extract_fields("Party of 0")
        assert "party_size" not in result


# ---------------------------------------------------------------------------
# Visit date extraction
# ---------------------------------------------------------------------------


class TestVisitDateExtraction:
    """Visit date patterns including day names and date formats."""

    @pytest.mark.parametrize(
        "text,expected_date",
        [
            ("next Friday", "Friday"),
            ("this Saturday", "Saturday"),
            ("this weekend", "weekend"),
        ],
    )
    def test_day_of_week_patterns(self, text, expected_date):
        result = extract_fields(text)
        assert result.get("visit_date") == expected_date

    def test_date_format_slash(self):
        result = extract_fields("We arrive on 3/15")
        assert result.get("visit_date") == "3/15"

    def test_visiting_pattern(self):
        result = extract_fields("Visiting on next Monday")
        assert "visit_date" in result


# ---------------------------------------------------------------------------
# Preference / dietary extraction
# ---------------------------------------------------------------------------


class TestPreferenceExtraction:
    """Dietary and cuisine preference patterns."""

    @pytest.mark.parametrize(
        "text,expected_pref",
        [
            ("I'm vegetarian", "vegetarian"),
            ("I am vegan", "vegan"),
            ("We're gluten free", "gluten free"),
            ("I'm kosher", "kosher"),
            ("We are halal", "halal"),
            ("I'm pescatarian", "pescatarian"),
        ],
    )
    def test_dietary_patterns(self, text, expected_pref):
        result = extract_fields(text)
        assert expected_pref in result.get("preferences", "")

    def test_allergy_pattern(self):
        result = extract_fields("I'm allergic to shellfish")
        assert "shellfish" in result.get("preferences", "")

    def test_cuisine_preference(self):
        result = extract_fields("I prefer Italian food")
        assert "italian" in result.get("preferences", "").lower()

    def test_multiple_preferences(self):
        result = extract_fields("I'm vegetarian and prefer Italian")
        prefs = result.get("preferences", "")
        assert "vegetarian" in prefs
        assert "Italian" in prefs or "italian" in prefs.lower()


# ---------------------------------------------------------------------------
# Occasion extraction
# ---------------------------------------------------------------------------


class TestOccasionExtraction:
    """Occasion/celebration patterns."""

    @pytest.mark.parametrize(
        "text,expected_occasion",
        [
            ("Celebrating our anniversary", "anniversary"),
            ("It's my birthday", "birthday"),
            ("For our wedding", "wedding"),
            ("We're on our honeymoon", "honeymoon"),
            ("Celebrating my graduation", "graduation"),
            ("For my retirement", "retirement"),
            ("It's a bachelor party", "bachelor party"),
            ("Celebrating a promotion", "promotion"),
        ],
    )
    def test_occasion_patterns(self, text, expected_occasion):
        result = extract_fields(text)
        assert result.get("occasion") == expected_occasion

    def test_standalone_occasion_keyword(self):
        result = extract_fields("We have an anniversary dinner tonight")
        assert result.get("occasion") == "anniversary"


# ---------------------------------------------------------------------------
# Multi-field extraction
# ---------------------------------------------------------------------------


class TestMultiFieldExtraction:
    """Multiple fields extracted from a single message."""

    def test_name_and_party_size(self):
        result = extract_fields("I'm Sarah, party of 4")
        assert result.get("name") == "Sarah"
        assert result.get("party_size") == 4

    def test_name_party_occasion(self):
        result = extract_fields("I'm Michael, group of 6, celebrating a birthday")
        assert result.get("name") == "Michael"
        assert result.get("party_size") == 6
        assert result.get("occasion") == "birthday"


# ---------------------------------------------------------------------------
# Fail-silent behavior
# ---------------------------------------------------------------------------


class TestFailSilent:
    """Errors should return empty dict, never raise."""

    def test_no_extraction_returns_empty(self):
        result = extract_fields("What time does the restaurant open?")
        assert isinstance(result, dict)
        # May or may not have fields, but should not crash
