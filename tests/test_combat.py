"""Tests for combat resolution."""
from dndplaya.mechanics.dice import DiceRoller
from dndplaya.mechanics.combat import CombatResolver, PressureSignal
from dndplaya.mechanics.characters import create_character, create_default_party
from dndplaya.mechanics.monsters import create_monster


def test_resolve_attack_hit():
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)
    char = create_character("Fighter", "Fighter", 3)
    monster = create_monster("Goblin", 0.25)

    result = combat.resolve_attack(char, monster)
    # Should return a result (may hit or miss depending on seed)
    assert result.attacker_name == "Fighter"
    assert result.target_name == "Goblin"
    assert result.damage_dealt >= 0


def test_resolve_attack_damage_range():
    """Over many attacks, damage should vary but stay in range."""
    dice = DiceRoller(seed=1)
    combat = CombatResolver(dice)
    char = create_character("Fighter", "Fighter", 3)

    damages = []
    for _ in range(50):
        monster = create_monster("Target", 0)  # AC 12
        result = combat.resolve_attack(char, monster)
        if result.damage_dealt > 0:
            damages.append(result.damage_dealt)

    assert len(damages) > 0  # At least some hits
    # avg_damage for L3 fighter is 13, so with 0.6-1.4 variance: ~7.8 to ~18.2
    for d in damages:
        assert 1.0 <= d <= 20.0


def test_resolve_heal():
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)
    cleric = create_character("Cleric", "Cleric", 3)
    target = create_character("Fighter", "Fighter", 3)
    target.current_hp = 10  # Wounded

    result = combat.resolve_heal(cleric, target, 8.0)
    assert result.damage_dealt <= 0  # Negative = healing
    assert result.narrative_hint.startswith("Cleric heals")


def test_resolve_aoe():
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)
    wizard = create_character("Wizard", "Wizard", 3)
    goblins = [create_monster(f"Goblin {i}", 0.25) for i in range(3)]

    results = combat.resolve_aoe(wizard, goblins, 10.0)
    assert len(results) == 3
    assert all(r.damage_dealt > 0 for r in results)


def test_pressure_signal_low_hp():
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)

    party = create_default_party(3)
    party[0].current_hp = 1  # Fighter at 1 HP out of 32
    monsters = [create_monster("Goblin", 0.25)]

    signals = combat.check_pressure_signals(party, monsters)
    assert PressureSignal.CHARACTER_LOW_HP in signals


def test_pressure_signal_bloodied():
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)

    party = create_default_party(3)
    monster = create_monster("Ogre", 2)
    monster.current_hp = 20  # Below 50% of 65

    signals = combat.check_pressure_signals(party, [monster])
    assert PressureSignal.ENEMY_BLOODIED in signals


def test_pressure_signal_tpk_risk():
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)

    party = create_default_party(3)
    for c in party:
        c.current_hp = int(c.max_hp * 0.2)  # All at 20%
    monsters = [create_monster("Ogre", 2)]

    signals = combat.check_pressure_signals(party, monsters)
    assert PressureSignal.TPK_RISK in signals
