from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from rich.console import Console

from ..config import Settings
from ..pdf.models import DungeonModule, Room, Encounter
from ..mechanics.state import GameState, GameEvent, EventType
from ..mechanics.combat import CombatResolver
from ..mechanics.characters import Character, create_default_party
from ..mechanics.monsters import create_monster
from ..mechanics.dice import DiceRoller
from ..agents.dm import DMAgent
from ..agents.player import PlayerAgent, ARCHETYPES
from ..agents.prompts.combat_narration import (
    COMBAT_START, COMBAT_ROUND_RESULT, COMBAT_END,
    ROOM_ENTRY, EXPLORATION_PROMPT,
)
from .phase import Phase
from .turn_manager import TurnManager
from .transcript import SessionTranscript

console = Console()

MAX_EXPLORATION_TURNS = 3  # Max back-and-forth per room before moving on
MAX_COMBAT_ROUNDS = 10  # Safety limit for combat
MAX_TOTAL_TURNS = 100  # Overall session limit


class Session:
    """Main game session orchestrating DM, players, and mechanics."""

    def __init__(
        self,
        module: DungeonModule,
        settings: Settings,
        party: list[Character] | None = None,
        seed: int | None = None,
    ):
        self.module = module
        self.settings = settings
        self.dice = DiceRoller(seed=seed or settings.seed)
        self.combat = CombatResolver(self.dice)

        # Create party
        self.party = party or create_default_party(settings.party_level)

        # Create game state
        self.state = GameState(characters=self.party)

        # Create agents
        self.dm = DMAgent(settings=settings, module=module)

        archetype_names = list(ARCHETYPES.keys())
        self.players: list[PlayerAgent] = []
        for i, character in enumerate(self.party):
            archetype = archetype_names[i % len(archetype_names)]
            self.players.append(PlayerAgent(
                settings=settings,
                character=character,
                archetype=archetype,
            ))

        self.turn_manager = TurnManager(dm=self.dm, players=self.players)
        self.transcript = SessionTranscript()
        self.phase = Phase.SETUP
        self.total_turns = 0

    def _is_party_dead(self) -> bool:
        """Check if all party members are dead (TPK)."""
        return not self.state.get_alive_characters()

    def run(self) -> SessionResult:
        """Run the complete session and return results."""
        console.print("[bold green]Starting session...[/bold green]")
        console.print(f"Module: {self.module.title}")
        console.print(f"Party: {', '.join(c.name for c in self.party)}")
        console.print(f"Rooms: {len(self.module.rooms)}")
        console.print()

        if not self.module.rooms:
            console.print("[bold red]No rooms found in module! Aborting session.[/bold red]")
            self.transcript.add_system_event("Session aborted: no rooms found in module.")
            return self._build_result()

        try:
            self._transition_phase(Phase.EXPLORATION)

            # Start with the entry room
            entry = self.module.get_entry_room()
            if not entry:
                console.print("[bold red]No rooms found in module![/bold red]")
                return self._build_result()

            # Process each room (BFS traversal)
            rooms_to_visit: deque[str] = deque([entry.id])
            visited: set[str] = set()

            while rooms_to_visit and self.total_turns < MAX_TOTAL_TURNS:
                # Check for TPK before entering next room
                if self._is_party_dead():
                    console.print("[bold red]Total Party Kill — session ending.[/bold red]")
                    self.transcript.add_system_event("Session ended: Total Party Kill.")
                    break

                room_id = rooms_to_visit.popleft()
                if room_id in visited:
                    continue

                visited.add(room_id)
                room = self.module.get_room(room_id)
                if not room:
                    continue

                console.print(f"[bold cyan]Entering: {room.name}[/bold cyan]")
                self._process_room(room)

                # Queue connected rooms not yet visited
                for conn_id in room.connections:
                    if conn_id not in visited:
                        rooms_to_visit.append(conn_id)

            if self.total_turns >= MAX_TOTAL_TURNS:
                warning = (
                    f"Session truncated: reached {MAX_TOTAL_TURNS} turn limit. "
                    f"Some rooms may not have been visited."
                )
                console.print(f"\n[bold yellow]{warning}[/bold yellow]")
                self.transcript.add_system_event(warning)

            self._transition_phase(Phase.COMPLETED)
            console.print("\n[bold green]Session complete![/bold green]")
        except Exception as e:
            console.print(f"\n[bold red]Session error: {e}[/bold red]")
            self.transcript.add_system_event(f"Session ended early due to error: {e}")
            self._transition_phase(Phase.COMPLETED)

        return self._build_result()

    def _process_room(self, room: Room) -> None:
        """Process a single room: enter, explore, encounter, move on."""
        self.state.enter_room(room.id)
        self.dm.enter_room(room.id)
        self.transcript.set_room(room.name)

        # DM describes the room
        entry_prompt = ROOM_ENTRY.format(room_name=room.name)
        dm_description = self.turn_manager.get_dm_description(entry_prompt)
        self.transcript.add_dm_narration(dm_description)
        console.print(f"  [dim]DM describes {room.name}[/dim]")
        self.total_turns += 1

        # Exploration phase
        self._transition_phase(Phase.EXPLORATION)
        for turn in range(MAX_EXPLORATION_TURNS):
            if self.total_turns >= MAX_TOTAL_TURNS:
                break

            # Players respond
            actions = self.turn_manager.get_player_actions(dm_description)
            for name, action in actions.items():
                self.transcript.add_player_action(name, action)
                self.state.add_event(GameEvent(
                    event_type=EventType.PLAYER_ACTION,
                    description=action,
                    actor=name,
                ))

            action_summary = "\n".join(f"{name}: {action}" for name, action in actions.items())

            # DM responds to actions
            dm_response = self.turn_manager.get_dm_description(
                EXPLORATION_PROMPT.format(player_actions=action_summary)
            )
            self.transcript.add_dm_narration(dm_response)
            self.total_turns += 1

            dm_description = dm_response  # For next round of player responses

        # Handle encounters — only fire encounters with no trigger (always-on)
        # or where trigger is set (future: evaluate trigger conditions)
        if room.encounters:
            for encounter in room.encounters:
                if self.total_turns >= MAX_TOTAL_TURNS:
                    break
                if self._is_party_dead():
                    break
                # Only auto-fire encounters with empty trigger (unconditional)
                # Encounters with triggers are skipped (would need DM adjudication)
                if encounter.trigger:
                    self.transcript.add_system_event(
                        f"Skipped conditional encounter (trigger: {encounter.trigger})"
                    )
                    continue
                self._run_combat(encounter, room)

    def _run_combat(self, encounter: Encounter, room: Room) -> None:
        """Run a combat encounter."""
        self._transition_phase(Phase.COMBAT)

        # Create monsters
        monsters = []
        for mref in encounter.monsters:
            try:
                for i in range(mref.count):
                    name = f"{mref.name}" if mref.count == 1 else f"{mref.name} {i+1}"
                    monsters.append(create_monster(name, mref.cr))
            except ValueError as e:
                console.print(f"  [yellow]Skipping monster: {e}[/yellow]")
                self.transcript.add_system_event(f"Skipped monster creation: {e}")

        if not monsters:
            return

        self.state.start_combat(monsters)

        # DM announces combat
        enemies = ", ".join(f"{m.name} (HP:{m.max_hp}, AC:{m.ac})" for m in monsters)
        party_status = ", ".join(
            f"{c.name}: {c.current_hp}/{c.max_hp} HP" for c in self.state.get_alive_characters()
        )

        combat_start = self.turn_manager.get_dm_description(
            COMBAT_START.format(enemies=enemies, party_status=party_status)
        )
        self.transcript.add_dm_narration(combat_start)
        self.total_turns += 1

        # Combat rounds
        dm_narration = combat_start
        for round_num in range(1, MAX_COMBAT_ROUNDS + 1):
            if self.total_turns >= MAX_TOTAL_TURNS:
                self.transcript.add_system_event(
                    "Combat interrupted: session turn limit reached."
                )
                break

            self.state.round_number = round_num
            alive_chars = self.state.get_alive_characters()
            alive_monsters = self.state.get_alive_monsters()

            if not alive_chars or not alive_monsters:
                break

            # Get player actions (for narration flavor, mechanics are resolved separately)
            player_actions = self.turn_manager.get_player_actions(dm_narration)
            for name, action in player_actions.items():
                self.transcript.add_player_action(name, action, round_number=round_num)

            # Resolve mechanics: each character attacks a monster
            action_results = []
            for char in alive_chars:
                alive_monsters = self.state.get_alive_monsters()
                if not alive_monsters:
                    break
                target = alive_monsters[0]  # Simple targeting: focus fire
                result = self.combat.resolve_attack(char, target)
                target.current_hp = max(0, target.current_hp - round(result.damage_dealt))
                action_results.append(
                    f"{char.name} deals {result.damage_dealt:.0f} damage to {target.name}"
                    f" ({target.current_hp}/{target.max_hp} HP)"
                )
                self.state.add_event(GameEvent(
                    event_type=EventType.ATTACK,
                    description=f"{char.name} attacks {target.name} for {result.damage_dealt:.0f} damage",
                    actor=char.name,
                    target=target.name,
                    round_number=round_num,
                ))
                if target.current_hp <= 0:
                    action_results.append(f"  {target.name} is defeated!")

            # Monsters attack
            for monster in self.state.get_alive_monsters():
                alive_chars = self.state.get_alive_characters()
                if not alive_chars:
                    break
                target = min(alive_chars, key=lambda c: c.current_hp)  # Target weakest
                result = self.combat.resolve_attack(monster, target)
                target.current_hp = max(0, target.current_hp - round(result.damage_dealt))
                action_results.append(
                    f"{monster.name} deals {result.damage_dealt:.0f} damage to {target.name}"
                    f" ({target.current_hp}/{target.max_hp} HP)"
                )
                self.state.add_event(GameEvent(
                    event_type=EventType.ATTACK,
                    description=f"{monster.name} attacks {target.name} for {result.damage_dealt:.0f} damage",
                    actor=monster.name,
                    target=target.name,
                    round_number=round_num,
                ))

            # Check pressure signals
            pressure = self.combat.check_pressure_signals(
                self.state.get_alive_characters(),
                self.state.get_alive_monsters(),
            )
            pressure_text = ""
            if pressure:
                signals = [s.name for s in pressure]
                pressure_text = f"\n[PRESSURE: {', '.join(signals)}]"
                for s in pressure:
                    self.state.add_event(GameEvent(
                        event_type=EventType.PRESSURE_SIGNAL,
                        description=s.name,
                        round_number=round_num,
                    ))

            result_text = "\n".join(action_results)
            self.transcript.add_combat_result(result_text, round_num)

            # DM narrates the round
            dm_narration = self.turn_manager.get_dm_description(
                COMBAT_ROUND_RESULT.format(
                    round_number=round_num,
                    action_results=result_text,
                    pressure_signals=pressure_text,
                )
            )
            self.transcript.add_dm_narration(dm_narration, round_number=round_num)
            self.total_turns += 1

            # Update alive lists for next check
            alive_monsters = self.state.get_alive_monsters()
            alive_chars = self.state.get_alive_characters()

        # Combat end
        if self.state.get_alive_monsters():
            outcome = "The party retreats from the remaining enemies."
        elif not self.state.get_alive_characters():
            outcome = "The party has been defeated! (TPK)"
        else:
            outcome = "All enemies have been defeated!"

        party_status = ", ".join(
            f"{c.name}: {c.current_hp}/{c.max_hp} HP" for c in self.party if c.current_hp > 0
        )

        combat_end = self.turn_manager.get_dm_description(
            COMBAT_END.format(outcome=outcome, party_status=party_status)
        )
        self.transcript.add_dm_narration(combat_end)
        self.state.end_combat()
        self._transition_phase(Phase.EXPLORATION)
        self.total_turns += 1

    def _transition_phase(self, new_phase: Phase) -> None:
        """Transition to a new game phase, enforcing valid transitions."""
        if self.phase == new_phase:
            return
        if not self.phase.can_transition_to(new_phase):
            console.print(
                f"[yellow]Warning: skipping invalid phase transition "
                f"{self.phase.name} -> {new_phase.name}[/yellow]"
            )
            return  # Don't force invalid transitions
        self.phase = new_phase
        self.transcript.add_system_event(f"Phase: {new_phase.name}")

    def _build_result(self) -> SessionResult:
        """Build the final session result."""
        return SessionResult(
            transcript=self.transcript,
            state=self.state,
            dm=self.dm,
            players=self.players,
            token_usage=self.turn_manager.get_token_summary(),
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
