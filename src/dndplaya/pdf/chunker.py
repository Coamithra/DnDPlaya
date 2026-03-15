from __future__ import annotations

import re
from .models import DungeonModule, Room, Encounter, MonsterRef, Trap, Treasure


# Patterns for identifying room sections
ROOM_PATTERNS = [
    re.compile(
        r"^#{1,3}\s+(?:Room|Area|Chamber|Hall|Cavern|Passage)\s*(\d+|[A-Z])\b[.:]\s*(.*)",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"^#{1,3}\s+(\d+|[A-Z])\.\s+(.*)", re.MULTILINE),
    re.compile(r"^#{1,3}\s+((?:The\s+)?\w[\w\s]+?)$", re.MULTILINE),
]

# Pattern for read-aloud / boxed text (blockquotes in markdown)
READ_ALOUD_PATTERN = re.compile(r"(?:^>\s*.+\n?)+", re.MULTILINE)

# Pattern for monster references like "2 goblins (CR 1/4)" or "a CR 3 owlbear"
# Handles hyphenated names (Half-Dragon), apostrophes (Will-o'-Wisp), multi-word names
MONSTER_PATTERN = re.compile(
    r"(\d+)?\s*(?:x\s+)?([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)*)\s*\(?\s*CR\s+(\d+(?:\s*/\s*\d+)?)\s*\)?",
    re.IGNORECASE,
)

# Pattern for trap DCs
TRAP_DC_PATTERN = re.compile(r"DC\s+(\d+)", re.IGNORECASE)

# Pattern for damage dice
DAMAGE_PATTERN = re.compile(r"\d+d\d+(?:\s*\+\s*\d+)?")


def parse_cr(cr_str: str) -> float:
    """Parse a CR string like '1/4', '1 / 4', or '3' into a float."""
    cr_str = cr_str.strip()
    if "/" in cr_str:
        num, den = cr_str.split("/")
        return int(num.strip()) / int(den.strip())
    return float(cr_str)


def extract_read_aloud(text: str) -> str:
    """Extract blockquoted read-aloud text from a section."""
    matches = READ_ALOUD_PATTERN.findall(text)
    if matches:
        # Clean up blockquote markers
        return "\n".join(
            line.lstrip("> ").strip()
            for match in matches
            for line in match.strip().split("\n")
        )
    return ""


def extract_monsters(text: str) -> list[MonsterRef]:
    """Extract monster references from text."""
    monsters = []
    for match in MONSTER_PATTERN.finditer(text):
        count = int(match.group(1)) if match.group(1) else 1
        name = match.group(2).strip()
        cr = parse_cr(match.group(3))
        monsters.append(MonsterRef(name=name, cr=cr, count=count))
    return monsters


def extract_encounters(text: str) -> list[Encounter]:
    """Extract encounters from a room section."""
    monsters = extract_monsters(text)
    if not monsters:
        return []

    # Try to find encounter description context
    encounter = Encounter(
        description=text[:200].strip() if len(text) > 200 else text.strip(),
        monsters=monsters,
    )
    return [encounter]


def extract_traps(text: str) -> list[Trap]:
    """Extract trap descriptions from text."""
    traps = []
    trap_keywords = ["trap", "hazard", "trigger", "tripwire", "pressure plate", "pit"]

    # Split into paragraphs to scope DC/damage search per trap
    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        lower = para.lower()
        if any(kw in lower for kw in trap_keywords):
            # Search for DC and damage only within this paragraph
            dc_match = TRAP_DC_PATTERN.search(para)
            dc = int(dc_match.group(1)) if dc_match else 15
            damage_match = DAMAGE_PATTERN.search(para)
            damage = damage_match.group(0) if damage_match else ""
            # Use first line as description
            first_line = para.strip().split("\n")[0].strip()
            traps.append(Trap(description=first_line, dc=dc, damage=damage))

    return traps


def extract_treasure(text: str) -> list[Treasure]:
    """Extract treasure/loot from text."""
    treasures = []
    treasure_keywords = [
        "treasure", "gold", "gp", "loot", "reward", "magic item", "potion", "scroll",
    ]

    lines = text.split("\n")
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in treasure_keywords):
            treasures.append(Treasure(description=line.strip()))

    return treasures


def chunk_markdown(markdown: str) -> DungeonModule:
    """Parse markdown text into a structured DungeonModule.

    Strategy:
    1. Try to split on heading patterns that look like room names
    2. Extract read-aloud text, encounters, traps, treasure from each section
    3. Build room connections from cross-references
    """
    # Extract title from first heading
    title_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Unknown Dungeon"

    # Try to find introduction/background before rooms start
    introduction = ""
    background = ""

    # Split into sections by headings
    sections: list[tuple[str, str, int]] = []  # (heading, content, level)
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    headings = list(heading_pattern.finditer(markdown))

    for i, match in enumerate(headings):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        content = markdown[start:end].strip()
        sections.append((heading_text, content, level))

    # Identify which sections are rooms vs intro/background
    rooms: list[Room] = []
    room_id_counter = 0

    # Headings that are clearly NOT rooms
    non_room_keywords = [
        "introduction", "overview", "synopsis", "background", "history",
        "lore", "appendix", "credits", "table of contents", "preface",
        "about", "license", "acknowledgement",
    ]

    for heading, content, level in sections:
        is_room = False
        room_name = heading
        room_id = ""

        lower_heading = heading.lower()

        # Skip the title (level 1) and known non-room sections
        if level == 1:
            pass  # Title headings are never rooms
        elif any(kw in lower_heading for kw in non_room_keywords):
            pass  # Known non-room sections
        else:
            # Check against room patterns (only first two specific patterns)
            for pattern in ROOM_PATTERNS[:2]:
                m = pattern.match(f"{'#' * level} {heading}")
                if m:
                    is_room = True
                    if m.lastindex and m.lastindex >= 2:
                        room_id = f"room_{m.group(1).lower()}"
                        room_name = m.group(2).strip() if m.group(2) else heading
                    elif m.lastindex == 1:
                        room_name = m.group(1).strip()
                    break

            # Heuristic: sections with encounters or read-aloud text are likely rooms
            if not is_room:
                has_encounters = bool(extract_monsters(content))
                has_read_aloud = bool(extract_read_aloud(content))
                if has_encounters or has_read_aloud:
                    is_room = True

        if is_room:
            room_id_counter += 1
            if not room_id:
                room_id = f"room_{room_id_counter}"

            read_aloud = extract_read_aloud(content)
            encounters = extract_encounters(content)
            traps = extract_traps(content)
            treasure = extract_treasure(content)

            rooms.append(Room(
                id=room_id,
                name=room_name or f"Room {room_id_counter}",
                description=content,
                read_aloud=read_aloud,
                encounters=encounters,
                traps=traps,
                treasure=treasure,
                raw_text=content,
            ))
        else:
            # Categorize non-room sections
            if any(kw in lower_heading for kw in ["introduction", "overview", "synopsis"]):
                introduction = content
            elif any(kw in lower_heading for kw in ["background", "history", "lore"]):
                background = content

    # Second pass: find connections between rooms
    for room in rooms:
        for other in rooms:
            if other.id == room.id:
                continue
            # Check if this room's text references other rooms
            lower_text = room.raw_text.lower()
            if other.name.lower() in lower_text or other.id in lower_text:
                if other.id not in room.connections:
                    room.connections.append(other.id)

    # If no connections found, assume linear layout (may not match actual dungeon)
    if rooms and all(len(r.connections) == 0 for r in rooms):
        import warnings
        warnings.warn(
            "No room cross-references found in PDF; assuming linear room layout. "
            "This may not match the actual dungeon structure.",
            stacklevel=2,
        )
        for i in range(len(rooms) - 1):
            rooms[i].connections.append(rooms[i + 1].id)
            rooms[i + 1].connections.append(rooms[i].id)

    return DungeonModule(
        title=title,
        introduction=introduction,
        background=background,
        rooms=rooms,
        raw_markdown=markdown,
    )
