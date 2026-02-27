"""Tests for gambling slang and drunk-typing normalization.

R72 Phase C3: Verifies normalize_for_search() correctly translates
gambling slang, SMS abbreviations, and common misspellings into
standard English for improved RAG retrieval.
"""

import pytest

from src.agent.slang import normalize_for_search


class TestGamblingSlangNormalization:
    """Multi-word and single-word gambling slang substitution."""

    def test_on_tilt_replaced(self):
        assert "frustrated after losing" in normalize_for_search("I'm on tilt right now")

    def test_whale_replaced(self):
        assert "high-value VIP guest" in normalize_for_search("I'm a whale here")

    def test_degen_replaced(self):
        assert "frequent gambler" in normalize_for_search("yeah I'm a degen")

    def test_run_bad_replaced(self):
        assert "losing streak" in normalize_for_search("I've been run bad all night")

    def test_busted_replaced(self):
        assert "lost all money" in normalize_for_search("I busted at the tables")

    def test_rfb_replaced(self):
        result = normalize_for_search("Do I get RFB?")
        assert "room food and beverage" in result

    def test_comp_me_replaced(self):
        assert "complimentary benefit" in normalize_for_search("Can you comp me dinner?")

    def test_marker_replaced(self):
        assert "casino credit" in normalize_for_search("I need a marker")

    def test_yolo_replaced(self):
        assert "risking everything" in normalize_for_search("I YOLO'd at craps")

    def test_rekt_replaced(self):
        assert "lost everything" in normalize_for_search("got totally rekt")

    def test_hot_streak_replaced(self):
        assert "winning streak" in normalize_for_search("I'm on a hot streak")

    def test_cold_streak_replaced(self):
        assert "losing streak" in normalize_for_search("Been on a cold streak")

    def test_crapped_out_replaced(self):
        assert "lost at craps" in normalize_for_search("I crapped out twice")

    def test_case_insensitive_multi_word(self):
        assert "frustrated after losing" in normalize_for_search("I'm ON TILT")


class TestDrunkTypingNormalization:
    """Common misspellings and SMS abbreviations."""

    def test_resturant_corrected(self):
        assert "restaurant" in normalize_for_search("where is the resturant")

    def test_rm_expanded(self):
        assert "room" in normalize_for_search("cn u get me a rm")

    def test_pls_expanded(self):
        assert "please" in normalize_for_search("help pls")

    def test_thx_expanded(self):
        assert "thanks" in normalize_for_search("thx for the help")

    def test_tonite_expanded(self):
        assert "tonight" in normalize_for_search("what's happening tonite")

    def test_tmrw_expanded(self):
        assert "tomorrow" in normalize_for_search("see you tmrw")

    def test_u_expanded(self):
        assert "you" in normalize_for_search("can u help")

    def test_idk_expanded(self):
        result = normalize_for_search("idk what to eat")
        assert "don't know" in result

    def test_asap_expanded(self):
        result = normalize_for_search("need a room asap")
        assert "as soon as possible" in result

    def test_rn_expanded(self):
        result = normalize_for_search("hungry rn")
        assert "right now" in result

    def test_upgrde_corrected(self):
        assert "upgrade" in normalize_for_search("need an upgrde")


class TestCombinedNormalization:
    """Mixed slang + typos in realistic messages."""

    def test_drunk_vip_message(self):
        result = normalize_for_search("cn u get me a rm upgrde pls im a whale here")
        assert "room" in result
        assert "upgrade" in result
        assert "please" in result
        assert "VIP" in result

    def test_frustrated_gambler(self):
        result = normalize_for_search("on tilt after getting rekt at blackjack need food rn")
        assert "frustrated after losing" in result
        assert "lost everything" in result
        assert "right now" in result

    def test_crypto_whale(self):
        result = normalize_for_search("just YOLO'd at craps, need somewhere to eat tonite")
        assert "risking everything" in result
        assert "tonight" in result


class TestEdgeCases:
    """Edge cases and safety."""

    def test_empty_string(self):
        assert normalize_for_search("") == ""

    def test_none_input(self):
        assert normalize_for_search(None) == ""

    def test_normal_text_unchanged(self):
        text = "Where is the steakhouse?"
        assert normalize_for_search(text) == text

    def test_preserves_punctuation(self):
        result = normalize_for_search("I'm on tilt! Where's food?")
        assert "?" in result

    def test_preserves_non_slang_words(self):
        result = normalize_for_search("The pool is great and the spa is relaxing")
        assert "pool" in result
        assert "great" in result
        assert "spa" in result

    def test_multiple_slang_in_one_message(self):
        result = normalize_for_search("I'm a whale on a hot streak, comp me dinner pls")
        assert "VIP" in result
        assert "winning streak" in result
        assert "complimentary" in result
        assert "please" in result
