"""Tests for monster creation."""
import pytest

from dndplaya.mechanics.monsters import Monster, create_monster, CR_TABLE


def test_create_goblin():
    m = create_monster("Goblin", 0.25)
    assert m.name == "Goblin"
    assert m.cr == 0.25
    assert m.max_hp == 25
    assert m.current_hp == 25
    assert m.ac == 12
    assert m.damage_per_round == 5


def test_create_with_abilities():
    m = create_monster("Dragon", 10, abilities=["Fire Breath", "Frightful Presence"])
    assert len(m.abilities) == 2
    assert "Fire Breath" in m.abilities


def test_invalid_cr():
    with pytest.raises(ValueError, match="Unknown CR"):
        create_monster("Bad", 99.0)


def test_all_crs():
    """All CR entries in the table produce valid monsters."""
    for cr in CR_TABLE:
        m = create_monster(f"CR_{cr}", cr)
        assert m.max_hp > 0
        assert m.ac > 0
