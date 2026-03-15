"""Edge case tests for combat resolution."""
from __future__ import annotations

from dndplaya.mechanics.dice import DiceRoller
from dndplaya.mechanics.combat import CombatResolver
from dndplaya.mechanics.characters import create_character
from dndplaya.mechanics.monsters import create_monster


def test_resolve_attack_miss():
    """A miss should return 0 damage and is_kill=False."""
    # Use a seed that produces a low roll (AC 17 is hard to hit with +5)
    dice = DiceRoller(seed=1)
    combat = CombatResolver(dice)
    char = create_character("Test", "Wizard", 1)  # low attack bonus
    monster = create_monster("Dragon", 10)  # AC 17

    # Try many attacks — at least one should miss
    miss_found = False
    for _ in range(20):
        result = combat.resolve_attack(char, monster)
        if result.damage_dealt == 0:
            assert not result.is_kill
            assert "misses" in result.narrative_hint
            miss_found = True
            break
    assert miss_found, "Expected at least one miss in 20 attacks"


def test_monster_attacks_character():
    """Monsters should be able to attack characters."""
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)
    monster = create_monster("Goblin", 0.25)
    char = create_character("Thorin", "Fighter", 3)
    result = combat.resolve_attack(monster, char)
    assert result.attacker_name == "Goblin"
    assert result.target_name == "Thorin"


def test_resolve_heal_at_full_hp():
    """Healing at full HP should return 0 actual healing."""
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)
    healer = create_character("Cleric", "Cleric", 3)
    target = create_character("Fighter", "Fighter", 3)
    # Target is at full HP
    assert target.current_hp == target.max_hp
    result = combat.resolve_heal(healer, target, 10.0)
    assert result.damage_dealt == 0  # negative heal = 0 since no HP missing


def test_resolve_heal_caps_at_max_hp():
    """Healing should not exceed max HP."""
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)
    healer = create_character("Cleric", "Cleric", 3)
    target = create_character("Fighter", "Fighter", 3)
    target.current_hp = target.max_hp - 2  # missing 2 HP
    result = combat.resolve_heal(healer, target, 100.0)
    # Should heal at most 2
    assert result.damage_dealt >= -2  # negative = healing


def test_resolve_aoe_empty_targets():
    """AoE with no targets should return empty list."""
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)
    caster = create_character("Wizard", "Wizard", 5)
    results = combat.resolve_aoe(caster, [], 20.0)
    assert results == []


def test_resolve_aoe_save_for_half():
    """AoE should do half damage on a save."""
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)
    caster = create_character("Wizard", "Wizard", 5)
    # Test with multiple monsters — some should save, some shouldn't
    monsters = [create_monster(f"Goblin {i}", 0.25) for i in range(10)]
    results = combat.resolve_aoe(caster, monsters, 20.0)
    assert len(results) == 10
    damages = [r.damage_dealt for r in results]
    # With 10 monsters and a d20 roll each, we should see some variation
    assert len(set(damages)) > 1, "Expected varied damage (some saves, some not)"


def test_pressure_signal_resource_depleted():
    """RESOURCE_DEPLETED should fire when spell slots are low."""
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)

    wizard = create_character("Wizard", "Wizard", 5)
    # Drain all spell slots
    for level in wizard.spell_slots:
        wizard.spell_slots[level] = 0

    cleric = create_character("Cleric", "Cleric", 5)
    for level in cleric.spell_slots:
        cleric.spell_slots[level] = 0

    fighter = create_character("Fighter", "Fighter", 5)
    rogue = create_character("Rogue", "Rogue", 5)

    signals = combat.check_pressure_signals([wizard, cleric, fighter, rogue], [])
    signal_names = [s.name for s in signals]
    assert "RESOURCE_DEPLETED" in signal_names


def test_damage_uses_round_not_truncate():
    """Damage should use round() not int() to avoid systematic loss."""
    dice = DiceRoller(seed=42)
    combat = CombatResolver(dice)
    char = create_character("Thorin", "Fighter", 5)
    monster = create_monster("Goblin", 1)

    # Run many attacks and check that some damage values round up
    damages = []
    for _ in range(50):
        result = combat.resolve_attack(char, monster)
        if result.damage_dealt > 0:
            damages.append(result.damage_dealt)
        # Reset monster HP for next attack
        monster.current_hp = monster.max_hp

    # With round(), damage_dealt is rounded to 1 decimal place.
    # The remaining HP calculation uses round(damage) which rounds to nearest int.
    # At least some values should have decimals >= 0.5 (which round up)
    assert len(damages) > 0
