"""Tests for PDF chunker / markdown parsing."""
from dndplaya.pdf.chunker import (
    parse_cr,
    extract_read_aloud,
    extract_monsters,
    extract_traps,
    extract_treasure,
    chunk_markdown,
)


def test_parse_cr_fraction():
    assert parse_cr("1/4") == 0.25
    assert parse_cr("1/2") == 0.5
    assert parse_cr("1/8") == 0.125


def test_parse_cr_integer():
    assert parse_cr("3") == 3.0
    assert parse_cr("10") == 10.0


def test_extract_read_aloud():
    text = "> The room is dark and foreboding.\n> Water drips from the ceiling.\n\nMore text."
    result = extract_read_aloud(text)
    assert "dark and foreboding" in result
    assert "Water drips" in result


def test_extract_monsters():
    text = "In this room are 3 Goblins (CR 1/4) and 1 Hobgoblin (CR 1/2)."
    monsters = extract_monsters(text)
    assert len(monsters) == 2
    assert monsters[0].name == "Goblins"
    assert monsters[0].count == 3
    assert monsters[0].cr == 0.25
    assert monsters[1].cr == 0.5


def test_extract_traps():
    text = "A pressure plate trap is hidden in the floor. DC 14 to detect. Deals 2d6 damage."
    traps = extract_traps(text)
    assert len(traps) >= 1
    assert traps[0].dc == 14
    assert "2d6" in traps[0].damage


def test_extract_treasure():
    text = "The chest contains 50 gold pieces and a potion of healing."
    treasures = extract_treasure(text)
    assert len(treasures) >= 1


def test_chunk_markdown_basic():
    markdown = """# The Lost Caves

## Introduction
Welcome to the Lost Caves adventure.

## Room 1: Entry Hall

> You enter a dimly lit stone hallway.

The hallway stretches before you.

## Room 2: The Guard Room

> Two goblins look up from their dice game.

2 Goblins (CR 1/4) are playing dice here.

## Room 3: The Treasure Room

> Gold glints in the torchlight.

A chest holds 100 gold pieces.
"""
    module = chunk_markdown(markdown)
    assert module.title == "The Lost Caves"
    assert len(module.rooms) == 3
    assert module.rooms[0].name in ("Entry Hall", "Room 1: Entry Hall")
    assert module.rooms[1].encounters  # Guard room has goblins
    # Linear connections since no cross-references
    assert len(module.rooms[0].connections) > 0


def test_chunk_markdown_empty():
    module = chunk_markdown("# Empty Dungeon\n\nNothing here.")
    assert module.title == "Empty Dungeon"
    assert len(module.rooms) == 0
