"""Tests for character creation."""
import pytest

from dndplaya.mechanics.characters import (
    Character,
    create_character,
    create_default_party,
    compute_skills,
    CLASS_STATS,
    INITIATIVE_BONUS,
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


# --- Skills tests ---

class TestComputeSkills:
    def test_fighter_level_1(self):
        skills = compute_skills("Fighter", 1)
        # proficiency = 2 + (1-1)//4 = 2
        assert skills["athletics"] == 5  # 2 + 3
        assert skills["intimidation"] == 5
        assert skills["perception"] == 3  # 2 + 1
        assert skills["strength_save"] == 3  # 2 + 1
        assert skills["constitution_save"] == 3
        assert skills.get("arcana", 0) == 0

    def test_fighter_level_5(self):
        skills = compute_skills("Fighter", 5)
        # proficiency = 2 + (5-1)//4 = 3
        assert skills["athletics"] == 6  # 3 + 3
        assert skills["intimidation"] == 6
        assert skills["perception"] == 4  # 3 + 1
        assert skills["strength_save"] == 4
        assert skills["constitution_save"] == 4

    def test_fighter_level_11(self):
        skills = compute_skills("Fighter", 11)
        # proficiency = 2 + (11-1)//4 = 4
        assert skills["athletics"] == 7  # 4 + 3
        assert skills["intimidation"] == 7
        assert skills["perception"] == 5  # 4 + 1

    def test_rogue_level_1(self):
        skills = compute_skills("Rogue", 1)
        # proficiency = 2
        assert skills["stealth"] == 5
        assert skills["sleight_of_hand"] == 5
        assert skills["acrobatics"] == 5
        assert skills["perception"] == 3
        assert skills["deception"] == 3
        assert skills["dexterity_save"] == 3
        assert skills["intelligence_save"] == 3

    def test_rogue_level_5(self):
        skills = compute_skills("Rogue", 5)
        # proficiency = 3
        assert skills["stealth"] == 6
        assert skills["dexterity_save"] == 4

    def test_rogue_level_11(self):
        skills = compute_skills("Rogue", 11)
        # proficiency = 4
        assert skills["stealth"] == 7

    def test_wizard_level_1(self):
        skills = compute_skills("Wizard", 1)
        assert skills["arcana"] == 5
        assert skills["investigation"] == 5
        assert skills["history"] == 3
        assert skills["perception"] == 3
        assert skills["intelligence_save"] == 3
        assert skills["wisdom_save"] == 3

    def test_wizard_level_5(self):
        skills = compute_skills("Wizard", 5)
        assert skills["arcana"] == 6
        assert skills["intelligence_save"] == 4

    def test_wizard_level_11(self):
        skills = compute_skills("Wizard", 11)
        assert skills["arcana"] == 7

    def test_cleric_level_1(self):
        skills = compute_skills("Cleric", 1)
        assert skills["medicine"] == 5
        assert skills["religion"] == 5
        assert skills["insight"] == 3
        assert skills["persuasion"] == 3
        assert skills["perception"] == 3
        assert skills["wisdom_save"] == 3
        assert skills["charisma_save"] == 3

    def test_cleric_level_5(self):
        skills = compute_skills("Cleric", 5)
        assert skills["medicine"] == 6
        assert skills["wisdom_save"] == 4

    def test_cleric_level_11(self):
        skills = compute_skills("Cleric", 11)
        assert skills["medicine"] == 7

    def test_default_skill_is_zero(self):
        skills = compute_skills("Fighter", 1)
        assert skills.get("arcana", 0) == 0
        assert skills.get("stealth", 0) == 0

    def test_create_character_populates_skills(self):
        char = create_character("Test", "Fighter", 3)
        assert len(char.skills) > 0
        assert "athletics" in char.skills
        assert "intimidation" in char.skills

    def test_create_character_populates_initiative_bonus(self):
        fighter = create_character("F", "Fighter", 1)
        rogue = create_character("R", "Rogue", 1)
        wizard = create_character("W", "Wizard", 1)
        cleric = create_character("C", "Cleric", 1)
        assert fighter.initiative_bonus == 1
        assert rogue.initiative_bonus == 3
        assert wizard.initiative_bonus == 1
        assert cleric.initiative_bonus == 0
