"""Tests for crisis detection and escalation protocol.

R72 Phase C5: Verifies detect_crisis_level() correctly classifies
guest messages into none/concern/urgent/immediate crisis levels.
"""

import pytest

from src.agent.crisis import detect_crisis_level


class TestImmediateCrisis:
    """Messages that should trigger IMMEDIATE crisis level."""

    def test_want_to_end_my_life(self):
        assert detect_crisis_level("I don't want to live anymore") == "immediate"

    def test_going_to_kill_myself(self):
        assert detect_crisis_level("I'm going to kill myself") == "immediate"

    def test_rather_be_dead(self):
        assert detect_crisis_level("I'd rather be dead than face my family") == "immediate"

    def test_nothing_left_to_live_for(self):
        assert detect_crisis_level("There's nothing left to live for") == "immediate"

    def test_ending_it(self):
        assert detect_crisis_level("I'm thinking about ending it") == "immediate"

    def test_saying_goodbye(self):
        assert detect_crisis_level("I'm saying goodbye to everyone") == "immediate"

    def test_take_my_own_life(self):
        assert detect_crisis_level("I want to take my own life") == "immediate"

    def test_dont_want_to_exist(self):
        assert detect_crisis_level("I don't want to exist anymore") == "immediate"


class TestUrgentCrisis:
    """Messages that should trigger URGENT crisis level."""

    def test_cant_face_wife(self):
        assert detect_crisis_level("I can't face my wife after this") == "urgent"

    def test_lost_everything(self):
        assert detect_crisis_level("I've lost everything, my life is ruined") == "urgent"

    def test_no_money_to_get_home(self):
        assert detect_crisis_level("I have no money to get home") == "urgent"

    def test_stranded(self):
        assert detect_crisis_level("I'm stranded here with nothing") == "urgent"

    def test_hurt_myself(self):
        assert detect_crisis_level("I feel like hurting myself") == "urgent"

    def test_dont_care_anymore(self):
        assert detect_crisis_level("I don't care anymore about anything") == "urgent"

    def test_too_drunk_to_stand(self):
        assert detect_crisis_level("I'm too drunk to stand straight") == "urgent"

    def test_blacked_out(self):
        assert detect_crisis_level("I think I blacked out earlier") == "urgent"

    def test_cant_afford_to_leave(self):
        assert detect_crisis_level("I can't afford to leave this place") == "urgent"

    def test_ruined_marriage(self):
        assert detect_crisis_level("I've ruined my marriage with gambling") == "urgent"


class TestConcernCrisis:
    """Messages that should trigger CONCERN crisis level."""

    def test_one_more_win(self):
        assert detect_crisis_level("I just need one more win") == "concern"

    def test_win_it_back(self):
        assert detect_crisis_level("I need to win it all back") == "concern"

    def test_need_a_marker(self):
        assert detect_crisis_level("Can I get a marker?") == "concern"

    def test_playing_all_night(self):
        assert detect_crisis_level("I've been playing all night") == "concern"

    def test_cant_stop(self):
        assert detect_crisis_level("I can't stop playing") == "concern"

    def test_gambling_problem(self):
        assert detect_crisis_level("I think I have a gambling problem") == "concern"

    def test_spent_too_much(self):
        assert detect_crisis_level("I spent too much today") == "concern"

    def test_addicted(self):
        assert detect_crisis_level("I think I'm addicted") == "concern"

    def test_double_down(self):
        assert detect_crisis_level("Let me double down, double or nothing") == "concern"

    def test_need_credit(self):
        assert detect_crisis_level("Can I get a credit advance?") == "concern"


class TestNoCrisis:
    """Normal messages that should NOT trigger any crisis level."""

    def test_normal_restaurant_question(self):
        assert detect_crisis_level("What restaurants are open?") == "none"

    def test_normal_gambling_mention(self):
        assert detect_crisis_level("Where are the slot machines?") == "none"

    def test_losing_mention_without_distress(self):
        assert detect_crisis_level("I'm down $50, whatever. Where's the buffet?") == "none"

    def test_empty_string(self):
        assert detect_crisis_level("") == "none"

    def test_none_input(self):
        assert detect_crisis_level(None) == "none"

    def test_normal_fatigue(self):
        assert detect_crisis_level("I'm exhausted from the drive") == "none"

    def test_normal_frustration(self):
        assert detect_crisis_level("The elevator was slow") == "none"

    def test_normal_win_excitement(self):
        assert detect_crisis_level("I just won $500! Where should we celebrate?") == "none"

    def test_normal_comp_request(self):
        assert detect_crisis_level("Can I get a comp for dinner?") == "none"


class TestSeverityOrdering:
    """Verify that higher severity levels take precedence."""

    def test_immediate_over_urgent(self):
        # Contains both immediate and urgent patterns
        msg = "I've lost everything. I don't want to live anymore."
        assert detect_crisis_level(msg) == "immediate"

    def test_urgent_over_concern(self):
        # Contains both urgent and concern patterns
        msg = "I can't face my wife. I need one more win to fix this."
        assert detect_crisis_level(msg) == "urgent"
