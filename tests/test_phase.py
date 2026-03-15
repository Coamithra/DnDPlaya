"""Tests for orchestrator/phase.py — phase state machine."""
from dndplaya.orchestrator.phase import Phase


def test_setup_can_transition_to_exploration():
    assert Phase.SETUP.can_transition_to(Phase.EXPLORATION)


def test_setup_cannot_transition_to_combat():
    assert not Phase.SETUP.can_transition_to(Phase.COMBAT)


def test_setup_cannot_transition_to_completed():
    assert not Phase.SETUP.can_transition_to(Phase.COMPLETED)


def test_exploration_can_transition_to_combat():
    assert Phase.EXPLORATION.can_transition_to(Phase.COMBAT)


def test_exploration_can_transition_to_completed():
    assert Phase.EXPLORATION.can_transition_to(Phase.COMPLETED)


def test_combat_can_transition_to_exploration():
    assert Phase.COMBAT.can_transition_to(Phase.EXPLORATION)


def test_combat_can_transition_to_completed():
    assert Phase.COMBAT.can_transition_to(Phase.COMPLETED)


def test_completed_cannot_transition_anywhere():
    for phase in Phase:
        if phase != Phase.COMPLETED:
            assert not Phase.COMPLETED.can_transition_to(phase)


def test_all_phases_have_transition_rules():
    """Every phase should have an entry in the valid transitions map."""
    for phase in Phase:
        # Should not raise — just return True or False
        phase.can_transition_to(Phase.EXPLORATION)
