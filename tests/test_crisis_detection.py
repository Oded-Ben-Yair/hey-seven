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
        assert (
            detect_crisis_level("I'd rather be dead than face my family") == "immediate"
        )

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


class TestImmediateCrisisSpanish:
    """R77: Spanish messages that should trigger IMMEDIATE crisis level."""

    def test_me_quiero_morir(self):
        assert detect_crisis_level("Me quiero morir") == "immediate"

    def test_no_quiero_vivir(self):
        assert detect_crisis_level("No quiero vivir") == "immediate"

    def test_mejor_muerto(self):
        assert detect_crisis_level("Estaria mejor muerto") == "immediate"

    def test_voy_a_matarme(self):
        assert detect_crisis_level("Me voy a matar") == "immediate"

    def test_acabar_con_todo(self):
        assert detect_crisis_level("Quiero acabar con todo") == "immediate"

    def test_no_quiero_seguir_viviendo(self):
        assert detect_crisis_level("No quiero seguir viviendo") == "immediate"

    def test_quitarme_la_vida(self):
        assert detect_crisis_level("Voy a quitarme la vida") == "immediate"

    def test_mejor_muerta(self):
        assert detect_crisis_level("Estoy mejor muerta") == "immediate"

    def test_voy_a_morir(self):
        assert detect_crisis_level("Me voy a morir") == "immediate"


class TestUrgentCrisis:
    """Messages that should trigger URGENT crisis level."""

    def test_cant_face_wife(self):
        assert detect_crisis_level("I can't face my wife after this") == "urgent"

    def test_lost_everything(self):
        assert (
            detect_crisis_level("I've lost everything, my life is ruined") == "urgent"
        )

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


class TestUrgentCrisisSpanish:
    """R77: Spanish messages that should trigger URGENT crisis level."""

    def test_no_puedo_mas(self):
        assert detect_crisis_level("No puedo mas con esto") == "urgent"

    def test_ya_no_aguanto(self):
        assert detect_crisis_level("Ya no aguanto esta situacion") == "urgent"

    def test_perdi_todo(self):
        assert detect_crisis_level("Perdi todo en el casino") == "urgent"

    def test_arruine_mi_vida(self):
        assert detect_crisis_level("Arruiné mi vida con el juego") == "urgent"

    def test_arruine_mi_matrimonio(self):
        assert detect_crisis_level("Arruiné mi matrimonio") == "urgent"

    def test_no_tengo_dinero_para(self):
        assert detect_crisis_level("No tengo dinero para el hotel") == "urgent"

    def test_hacerme_dano(self):
        assert detect_crisis_level("Tengo ganas de hacerme daño") == "urgent"

    def test_no_veo_sentido(self):
        assert detect_crisis_level("No le veo sentido a nada") == "urgent"

    def test_no_puedo_mas_accent(self):
        assert detect_crisis_level("No puedo más") == "urgent"


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


class TestConcernCrisisSpanish:
    """R77: Spanish messages that should trigger CONCERN crisis level."""

    def test_necesito_recuperar(self):
        assert detect_crisis_level("Necesito recuperar lo que perdi") == "concern"

    def test_recuperar_mi_dinero(self):
        assert detect_crisis_level("Necesito recuperar mi dinero") == "concern"

    def test_llevo_todo_el_dia(self):
        assert detect_crisis_level("Llevo todo el dia jugando") == "concern"

    def test_llevo_horas_aqui(self):
        assert detect_crisis_level("Llevo horas aqui en las maquinas") == "concern"

    def test_gaste_demasiado(self):
        assert detect_crisis_level("Gaste demasiado hoy") == "concern"

    def test_gaste_todo(self):
        assert detect_crisis_level("Gasté todo lo que traje") == "concern"

    def test_no_puedo_parar_de_jugar(self):
        assert detect_crisis_level("No puedo parar de jugar") == "concern"

    def test_no_deberia_dejar_de_jugar(self):
        assert (
            detect_crisis_level("No deberia dejar de jugar pero no puedo") == "concern"
        )


class TestNoCrisis:
    """Normal messages that should NOT trigger any crisis level."""

    def test_normal_restaurant_question(self):
        assert detect_crisis_level("What restaurants are open?") == "none"

    def test_normal_gambling_mention(self):
        assert detect_crisis_level("Where are the slot machines?") == "none"

    def test_losing_mention_without_distress(self):
        assert (
            detect_crisis_level("I'm down $50, whatever. Where's the buffet?") == "none"
        )

    def test_empty_string(self):
        assert detect_crisis_level("") == "none"

    def test_none_input(self):
        assert detect_crisis_level(None) == "none"

    def test_normal_fatigue(self):
        assert detect_crisis_level("I'm exhausted from the drive") == "none"

    def test_normal_frustration(self):
        assert detect_crisis_level("The elevator was slow") == "none"

    def test_normal_win_excitement(self):
        assert (
            detect_crisis_level("I just won $500! Where should we celebrate?") == "none"
        )

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


class TestCrisisPersistence:
    """R73: Tests for crisis_active state field persistence in compliance_gate."""

    @pytest.fixture
    def _base_state(self):
        """Minimal state for compliance_gate_node tests."""
        from langchain_core.messages import HumanMessage

        return {
            "messages": [HumanMessage(content="test")],
            "query_type": None,
            "router_confidence": 0.0,
            "responsible_gaming_count": 0,
            "crisis_active": False,
        }

    @pytest.mark.asyncio
    async def test_crisis_sets_crisis_active(self, _base_state):
        """When crisis is detected, crisis_active should be True in return."""
        from langchain_core.messages import HumanMessage
        from src.agent.compliance_gate import compliance_gate_node

        _base_state["messages"] = [HumanMessage(content="I don't want to live anymore")]
        result = await compliance_gate_node(_base_state)
        assert result.get("query_type") == "self_harm"
        assert result.get("crisis_active") is True

    @pytest.mark.asyncio
    async def test_crisis_persistence_maintains_context(self, _base_state):
        """When crisis_active is True, follow-up non-property messages stay in crisis."""
        from langchain_core.messages import HumanMessage
        from src.agent.compliance_gate import compliance_gate_node

        _base_state["crisis_active"] = True
        _base_state["messages"] = [HumanMessage(content="Nobody can help me")]
        result = await compliance_gate_node(_base_state)
        assert result.get("query_type") == "self_harm"

    @pytest.mark.asyncio
    async def test_crisis_exits_on_property_question_without_distress(
        self, _base_state
    ):
        """R102: Crisis exits when guest asks property question without distress signals.

        A property question without distress language ("What restaurants do you have?")
        indicates the guest has moved on. Keeping them in crisis mode is counterproductive.
        Only persist crisis if distress signals are still present.
        """
        from langchain_core.messages import HumanMessage
        from src.agent.compliance_gate import compliance_gate_node

        _base_state["crisis_active"] = True
        _base_state["messages"] = [
            HumanMessage(content="What restaurants do you have?")
        ]
        result = await compliance_gate_node(_base_state)
        # R102: Property question without distress → allow transition
        assert result.get("query_type") is None

    @pytest.mark.asyncio
    async def test_crisis_persists_on_property_question_with_distress(
        self, _base_state
    ):
        """R102: Crisis persists when property question still contains distress signals."""
        from langchain_core.messages import HumanMessage
        from src.agent.compliance_gate import compliance_gate_node

        _base_state["crisis_active"] = True
        _base_state["messages"] = [
            HumanMessage(content="I want to die but what restaurants do you have?")
        ]
        result = await compliance_gate_node(_base_state)
        assert result.get("query_type") == "self_harm"

    @pytest.mark.asyncio
    async def test_crisis_exits_on_safe_confirmation_plus_property_question(
        self, _base_state
    ):
        """R74: Crisis exits when guest confirms safe AND asks property question."""
        from langchain_core.messages import HumanMessage
        from src.agent.compliance_gate import compliance_gate_node

        _base_state["crisis_active"] = True
        _base_state["messages"] = [
            HumanMessage(content="I'm feeling better now, what restaurants are open?")
        ]
        result = await compliance_gate_node(_base_state)
        # Should pass through to router (query_type=None), not crisis
        assert result.get("query_type") is None

    @pytest.mark.asyncio
    async def test_crisis_persists_on_safe_confirmation_alone(self, _base_state):
        """R74: Crisis persists on safe confirmation without property question.

        A guest saying 'I'm feeling better' may still need support — do not
        exit crisis mode until they also demonstrate intent to move on by
        asking about a property amenity.
        """
        from langchain_core.messages import HumanMessage
        from src.agent.compliance_gate import compliance_gate_node

        _base_state["crisis_active"] = True
        _base_state["messages"] = [HumanMessage(content="I'm feeling better")]
        result = await compliance_gate_node(_base_state)
        assert result.get("query_type") == "self_harm"

    @pytest.mark.asyncio
    async def test_crisis_persists_on_ambiguous_message(self, _base_state):
        """R74: Crisis persists on ambiguous messages like 'whatever'."""
        from langchain_core.messages import HumanMessage
        from src.agent.compliance_gate import compliance_gate_node

        _base_state["crisis_active"] = True
        _base_state["messages"] = [HumanMessage(content="whatever")]
        result = await compliance_gate_node(_base_state)
        assert result.get("query_type") == "self_harm"

    @pytest.mark.asyncio
    async def test_crisis_active_in_state_schema(self):
        """Verify crisis_active field exists in PropertyQAState."""
        from src.agent.state import PropertyQAState

        assert "crisis_active" in PropertyQAState.__annotations__
