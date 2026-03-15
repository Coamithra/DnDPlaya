from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from ..config import Settings
from ..mechanics.state import GameState, EventType
from ..mechanics.characters import Character, create_default_party
from ..mechanics.dice import DiceRoller
from ..agents.dm import DMAgent
from ..agents.player import PlayerAgent, ARCHETYPES
from ..agents.base import AgentResponse
from ..agents.context import compact_history
from .transcript import SessionTranscript

console = Console()

MAX_TOTAL_TURNS = 100

SESSION_START = """The party arrives at the dungeon entrance. The adventurers are:
{party_description}

Begin the adventure. Describe the entrance and what the party sees. \
Use enter_room to mark the first area, then set the scene."""


class Session:
    """Main game session: DM drives the adventure via tool use."""

    def __init__(
        self,
        module_markdown: str,
        settings: Settings,
        map_images: list[tuple[bytes, str]] | None = None,
        party: list[Character] | None = None,
        seed: int | None = None,
    ):
        self.settings = settings
        self.dice = DiceRoller(seed=seed or settings.seed)

        # Create party
        self.party = party or create_default_party(settings.party_level)

        # Create game state
        self.state = GameState(characters=self.party)

        # Create DM agent
        self.dm = DMAgent(
            module_markdown=module_markdown,
            settings=settings,
            map_images=map_images,
        )

        # Create player agents
        archetype_names = list(ARCHETYPES.keys())
        self.players: list[PlayerAgent] = []
        self._player_map: dict[str, PlayerAgent] = {}
        for i, character in enumerate(self.party):
            archetype = archetype_names[i % len(archetype_names)]
            player = PlayerAgent(
                settings=settings,
                character=character,
                archetype=archetype,
            )
            self.players.append(player)
            self._player_map[character.name.lower()] = player

        self.transcript = SessionTranscript()
        self.total_turns = 0
        self._terminated = False

    def _is_party_dead(self) -> bool:
        """Check if all party members are dead (TPK)."""
        return not self.state.get_alive_characters()

    def run(self) -> SessionResult:
        """Run the complete session via DM conversation loop."""
        console.print("[bold green]Starting session...[/bold green]")
        console.print(f"Party: {', '.join(c.name for c in self.party)}")
        console.print()

        try:
            # Build opening prompt
            party_desc = "\n".join(
                f"- {c.name} ({c.char_class} level {c.level}): "
                f"{c.max_hp} HP, AC {c.ac}"
                for c in self.party
            )
            opening = SESSION_START.format(party_description=party_desc)

            # First DM turn
            compact_history(self.dm)
            response = self.dm.send_with_tools(opening)
            self.total_turns += 1

            # Main conversation loop
            while not self._terminated and self.total_turns < MAX_TOTAL_TURNS:
                if self._is_party_dead():
                    console.print(
                        "[bold red]Total Party Kill — session ending.[/bold red]"
                    )
                    self.transcript.add_system_event(
                        "Session ended: Total Party Kill."
                    )
                    break

                if response.tool_calls:
                    # Process tool calls and submit results back to DM
                    tool_results = self._process_tool_calls(response)

                    if self._terminated or self._is_party_dead():
                        break

                    compact_history(self.dm)
                    response = self.dm.submit_tool_results(tool_results)
                    self.total_turns += 1
                else:
                    # DM just narrated without tools — record and prompt to continue
                    if response.text:
                        self.transcript.add_dm_narration(response.text)
                        console.print("  [dim]DM narrates[/dim]")

                    compact_history(self.dm)
                    response = self.dm.send_with_tools(
                        "Continue the adventure. Use your tools to progress — "
                        "request player input, enter rooms, resolve checks, etc."
                    )
                    self.total_turns += 1

            if self.total_turns >= MAX_TOTAL_TURNS:
                warning = (
                    f"Session truncated: reached {MAX_TOTAL_TURNS} turn limit."
                )
                console.print(f"\n[bold yellow]{warning}[/bold yellow]")
                self.transcript.add_system_event(warning)

            console.print("\n[bold green]Session complete![/bold green]")

        except Exception as e:
            console.print(f"\n[bold red]Session error: {e}[/bold red]")
            self.transcript.add_system_event(
                f"Session ended early due to error: {e}"
            )

        return self._build_result()

    def _process_tool_calls(
        self, response: AgentResponse
    ) -> list[tuple[str, str]]:
        """Process all tool calls from a DM response."""
        results = []

        # Record any narration text before tool calls
        if response.text:
            self.transcript.add_dm_narration(response.text)
            console.print("  [dim]DM narrates[/dim]")

        for tc in response.tool_calls:
            result = self._dispatch_tool(tc.name, tc.arguments)
            results.append((tc.id, result))

        return results

    def _dispatch_tool(self, name: str, args: dict) -> str:
        """Dispatch a single tool call to the appropriate handler."""
        handlers = {
            "roll_check": self._handle_roll_check,
            "roll_dice": self._handle_roll_dice,
            "apply_damage": self._handle_apply_damage,
            "heal": self._handle_heal,
            "get_party_status": self._handle_get_party_status,
            "enter_room": self._handle_enter_room,
            "request_player_input": self._handle_request_player_input,
            "end_session": self._handle_end_session,
        }
        handler = handlers.get(name)
        if not handler:
            return f"Unknown tool: {name}"
        try:
            return handler(args)
        except Exception as e:
            return f"Error: {e}"

    def _handle_roll_check(self, args: dict) -> str:
        modifier = args.get("modifier", 0)
        dc = args.get("dc", 10)
        description = args.get("description", "check")
        success, total = self.dice.check(modifier, dc)
        result = "SUCCESS" if success else "FAILURE"
        text = (
            f"{description}: rolled {total} (d20+{modifier}) "
            f"vs DC {dc} — {result}"
        )
        self.transcript.add_system_event(f"Check: {text}")
        console.print(f"    [dim]{text}[/dim]")
        return text

    def _handle_roll_dice(self, args: dict) -> str:
        expression = args.get("expression", "1d6")
        reason = args.get("reason", "roll")
        result = self.dice.parse_and_roll(expression)
        text = f"{reason}: {expression} = {result}"
        self.transcript.add_system_event(f"Roll: {text}")
        console.print(f"    [dim]{text}[/dim]")
        return text

    def _handle_apply_damage(self, args: dict) -> str:
        name = args.get("character_name", "")
        amount = args.get("amount", 0)
        description = args.get("description", "damage")

        char = self.state.get_character(name)
        if not char:
            available = ", ".join(c.name for c in self.party)
            return f"Character '{name}' not found. Available: {available}"

        char.current_hp = max(0, char.current_hp - amount)
        self.state.add_event(
            EventType.ATTACK,
            f"{description}: {name} takes {amount} damage",
            target=name,
        )
        self.transcript.add_system_event(
            f"Damage: {name} takes {amount} ({description}) "
            f"— now {char.current_hp}/{char.max_hp} HP"
        )
        console.print(
            f"    [red]{name} takes {amount} damage "
            f"({char.current_hp}/{char.max_hp} HP)[/red]"
        )

        status = f"{name}: {char.current_hp}/{char.max_hp} HP"
        if char.current_hp == 0:
            status += " (DOWN!)"
        return status

    def _handle_heal(self, args: dict) -> str:
        name = args.get("character_name", "")
        amount = args.get("amount", 0)
        description = args.get("description", "healing")

        char = self.state.get_character(name)
        if not char:
            available = ", ".join(c.name for c in self.party)
            return f"Character '{name}' not found. Available: {available}"

        old_hp = char.current_hp
        char.current_hp = min(char.max_hp, char.current_hp + amount)
        actual = char.current_hp - old_hp
        self.state.add_event(
            EventType.HEAL,
            f"{description}: {name} healed for {actual}",
            target=name,
        )
        self.transcript.add_system_event(
            f"Heal: {name} healed {actual} ({description}) "
            f"— now {char.current_hp}/{char.max_hp} HP"
        )
        console.print(
            f"    [green]{name} healed {actual} "
            f"({char.current_hp}/{char.max_hp} HP)[/green]"
        )
        return f"{name}: {char.current_hp}/{char.max_hp} HP (healed {actual})"

    def _handle_get_party_status(self, args: dict) -> str:
        lines = []
        for char in self.party:
            slots = ""
            if char.spell_slots:
                slots = f", spell slots: {dict(char.spell_slots)}"
            status = "DOWN" if char.current_hp <= 0 else "OK"
            lines.append(
                f"- {char.name} ({char.char_class} L{char.level}): "
                f"{char.current_hp}/{char.max_hp} HP, AC {char.ac}, "
                f"attack +{char.attack_bonus}, avg damage {char.avg_damage}"
                f"{slots} [{status}]"
            )
        return "Party Status:\n" + "\n".join(lines)

    def _handle_enter_room(self, args: dict) -> str:
        room_name = args.get("room_name", "Unknown")
        self.state.enter_room(room_name)
        self.transcript.set_room(room_name)
        console.print(f"[bold cyan]Entering: {room_name}[/bold cyan]")
        return f"Entered: {room_name}"

    def _handle_request_player_input(self, args: dict) -> str:
        player_names = args.get("player_names", [])
        if not player_names:
            return "No player names specified."

        # Get the DM's narration to send to players
        dm_context = self.transcript.get_recent_dm_narration()

        responses = {}
        for name in player_names:
            player = self._player_map.get(name.lower())
            if not player:
                responses[name] = f"[Unknown player: {name}]"
                continue

            # Only get input from alive characters
            char = self.state.get_character(name)
            if char and char.current_hp <= 0:
                responses[name] = f"[{name} is unconscious and cannot act]"
                self.transcript.add_player_action(name, "(unconscious)")
                continue

            compact_history(player)
            action = player.send(dm_context)
            responses[name] = action
            self.transcript.add_player_action(name, action)
            console.print(f"  [dim]{name} responds[/dim]")

        result_lines = [
            f"{name}: {action}" for name, action in responses.items()
        ]
        return "Player responses:\n" + "\n".join(result_lines)

    def _handle_end_session(self, args: dict) -> str:
        reason = args.get("reason", "Adventure complete")
        self._terminated = True
        self.transcript.add_system_event(f"Session ended: {reason}")
        console.print(f"[bold green]Session ended: {reason}[/bold green]")
        return f"Session ended: {reason}"

    def _build_result(self) -> SessionResult:
        """Build the final session result."""
        token_usage = {"DM": self.dm.get_token_usage()}
        for player in self.players:
            token_usage[player.name] = player.get_token_usage()
        return SessionResult(
            transcript=self.transcript,
            state=self.state,
            dm=self.dm,
            players=self.players,
            token_usage=token_usage,
        )


@dataclass
class SessionResult:
    """Result of a completed session."""

    transcript: SessionTranscript
    state: GameState
    dm: DMAgent
    players: list[PlayerAgent]
    token_usage: dict

    def get_transcript_text(self) -> str:
        return self.transcript.to_text()
