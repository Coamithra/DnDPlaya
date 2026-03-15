"""Tests for character creation."""
import pytest

from dndplaya.mechanics.characters import (
    Character,
    create_character,
    create_default_party,
    CLASS_STATS,
)


def test_create_fighter():
    char = create_character("Thorin", "Fighter", 3)
    assert char.name == "Thorin"
    assert char.char_class == "Fighter"
    assert char.level == 3
    assert char.max_hp == 12 + 10 * 2  # base_hp + hp_per_level * (level-1)
    assert char.current_hp == char.max_hp
    assert char.ac == 18
    assert char.avg_damage == 13
    assert char.spell_slots == {}


def test_create_wizard_has_spell_slots():
    char = create_character("Elara", "Wizard", 3)
    assert sum(char.spell_slots.values()) == 4  # level + 1 = 4


def test_create_cleric_has_spell_slots():
    char = create_character("Marcus", "Cleric", 3)
    assert sum(char.spell_slots.values()) == 3  # level = 3


def test_create_rogue_no_spell_slots():
    char = create_character("Shadow", "Rogue", 3)
    assert char.spell_slots == {}


def test_invalid_class():
    with pytest.raises(ValueError, match="Unknown class"):
        create_character("Bad", "Bard", 1)


def test_invalid_level():
    with pytest.raises(ValueError, match="Level must be"):
        create_character("Bad", "Fighter", 0)
    with pytest.raises(ValueError, match="Level must be"):
        create_character("Bad", "Fighter", 12)


def test_default_party():
    party = create_default_party(3)
    assert len(party) == 4
    names = [c.name for c in party]
    assert "Thorin" in names
    assert "Shadow" in names
    assert "Elara" in names
    assert "Brother Marcus" in names
    classes = {c.char_class for c in party}
    assert classes == {"Fighter", "Rogue", "Wizard", "Cleric"}


def test_all_levels():
    """All classes can be created at all valid levels."""
    for cls in CLASS_STATS:
        for level in range(1, 12):
            char = create_character(f"Test_{cls}", cls, level)
            assert char.max_hp > 0
            assert char.avg_damage > 0
