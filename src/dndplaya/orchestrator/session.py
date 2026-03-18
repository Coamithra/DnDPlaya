from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..mechanics.state import GameState, EventType
from ..mechanics.characters import Character, create_default_party
from ..mechanics.monsters import Monster, create_monster
from ..mechanics.dice import DiceRoller
from ..agents.dm import DMAgent
from ..agents.player import PlayerAgent, ARCHETYPES
from ..agents.base import AgentResponse
from ..agents.context import compact_history
from ..prompts import load_prompt
from .transcript import SessionTranscript

DEFAULT_MAX_TURNS = 100

DIFFICULTY_DC = {
    "very_easy": 5,
    "easy": 10,
    "medium": 13,
    "hard": 16,
    "very_hard": 20,
    "nearly_impossible": 25,
}


def _char_key(name: str) -> str:
    """Normalise a character name to a UI element key."""
    return name.lower().replace(" ", "_")


class Session:
    """Main game session: DM drives the adventure via tool use."""

    def __init__(
        self,
        module_markdown: str,
        settings: Settings,
        map_images: list[tuple[bytes, str]] | None = None,
        party: list[Character] | None = None,
        seed: int | None = None,
        pages: list[str] | None = None,
        summary: str = "",
        max_turns: int = DEFAULT_MAX_TURNS,
        log_path: Path | None = None,
        ui=None,
        enable_thinking: bool = False,
        music_tracks: list[str] | None = None,
        enable_reviews: bool = True,
    ):
        self.settings = settings
        self.dice = DiceRoller(seed=seed or settings.seed)

        # Module reference state
        self.pages = pages
        self._last_read_page: int | None = None
        self.module_references: list[dict] = []

        # Create party
        self.party = party or create_default_party(settings.party_level)

        # Create game state
        self.state = GameState(characters=self.party)

        # Active monsters (registered via roll_initiative)
        self._active_monsters: dict[str, Monster] = {}

        # Initiative tracking
        self._initiative_order: list[tuple[str, int, str]] = []  # (name, roll, "PC"|"Monster")
        self._initiative_index: int = -1  # -1 = not started

        # Create DM agent — use summary if available, otherwise full markdown
        if summary:
            self.dm = DMAgent(
                summary=summary,
                settings=settings,
                map_images=map_images,
                music_tracks=music_tracks,
                enable_reviews=enable_reviews,
            )
        else:
            self.dm = DMAgent(
                summary=module_markdown,
                settings=settings,
                map_images=map_images,
                music_tracks=music_tracks,
                enable_reviews=enable_reviews,
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
                enable_thinking=enable_thinking,
                enable_reviews=enable_reviews,
            )
            self.players.append(player)
            self._player_map[character.name.lower()] = player

        self.max_turns = max_turns
        self.transcript = SessionTranscript(log_path=log_path)
        self._log_dir = log_path.parent if log_path else None
        self.total_turns = 0
        self._terminated = False

        # Early-out tracking
        self._consecutive_no_tool_turns = 0
        self._narration_count = 0
        self._cost_budget = 3.00  # USD — abort if exceeded
        self._cache_check_turn = 10  # check cache health after this many turns

        # Optional UI emitter (None = headless / console mode)
        self.ui = ui
        self._pending_ui_wait = False  # deferred wait_for_continue

        # Write module summary + party info to transcript header
        if summary:
            self.transcript.add_system_event(f"MODULE SUMMARY:\n\n{summary}")
        party_info = "\n".join(
            f"- {c.name} ({c.char_class} L{c.level}): {c.max_hp} HP, AC {c.ac}"
            for c in self.party
        )
        self.transcript.add_system_event(f"PARTY:\n\n{party_info}")

    def _is_party_dead(self) -> bool:
        """Check if all party members are dead (TPK)."""
        return not self.state.get_alive_characters()

    def _dump_agent_logs(self) -> None:
        """Append all agent conversation histories to disk (live debugging).

        Each call appends a timestamped snapshot so the full history of
        every prompt/response is preserved, even though players replace
        their history via set_cached_context each turn.
        """
        if not self._log_dir:
            return
        logs_dir = self._log_dir / "agent_logs"
        logs_dir.mkdir(exist_ok=True)
        header = f"\n{'='*60}\n[SNAPSHOT turn={self.total_turns}]\n{'='*60}\n"
        for agent in [self.dm] + self.players:
            safe = agent.name.lower().replace(" ", "_")
            with open(logs_dir / f"{safe}.txt", "a", encoding="utf-8") as f:
                f.write(header)
                f.write(agent.dump_history())
                f.write("\n")

    def _flush_ui_wait(self) -> None:
        """If there's a pending speech the user hasn't continued past, wait now."""
        if self.ui and self._pending_ui_wait:
            self.ui.wait_for_continue()
            self._pending_ui_wait = False

    # --- Live ticker output ---

    def _tick(self, speaker: str, detail: str = "") -> None:
        """Print a compact progress tick: Speaker(turn) or Speaker(turn):detail."""
        tag = f"{speaker}({self.total_turns})"
        if detail:
            tag += f":{detail}"
        print(f" {tag}", end="", flush=True)

    def _event(self, text: str) -> None:
        """Print a notable event on its own line."""
        print(f"\n  >> {text}", end="", flush=True)

    def run(self) -> SessionResult:
        """Run the complete session via DM conversation loop."""
        print(f"Session start | Party: {', '.join(c.name for c in self.party)}", flush=True)

        try:
            # Build opening prompt
            party_desc = "\n".join(
                f"- {c.name} ({c.char_class} level {c.level}, {c.pronouns}): "
                f"{c.max_hp} HP, AC {c.ac}"
                for c in self.party
            )
            opening = load_prompt("session_start", party_description=party_desc)

            # First DM turn
            if self.ui:
                self.ui.thinking_start("dm")
            compact_history(self.dm)
            response = self.dm.send_with_tools(opening)
            if self.ui:
                self.ui.thinking_stop("dm")
            self.total_turns += 1
            self._tick("DM")
            self._dump_agent_logs()

            # Main conversation loop
            while not self._terminated and self.total_turns < self.max_turns:
                if self._is_party_dead():
                    self._event("TPK!")
                    self.transcript.add_system_event(
                        "Session ended: Total Party Kill."
                    )
                    break

                # Early-out checks
                abort_reason = self._check_early_outs()
                if abort_reason:
                    self._event(f"ABORT: {abort_reason}")
                    self.transcript.add_system_event(
                        f"Session aborted: {abort_reason}"
                    )
                    self._terminated = True
                    break

                if response.tool_calls:
                    self._consecutive_no_tool_turns = 0
                    # Process tool calls and submit results back to DM
                    tool_results = self._process_tool_calls(response)

                    if self._terminated or self._is_party_dead():
                        break

                    if self.ui:
                        self.ui.thinking_start("dm")
                    compact_history(self.dm)
                    response = self.dm.submit_tool_results(tool_results)
                    if self.ui:
                        self.ui.thinking_stop("dm")
                    self.total_turns += 1
                    self._tick("DM")
                    self._dump_agent_logs()
                else:
                    self._consecutive_no_tool_turns += 1
                    # DM produced text without tools — internal thinking, nudge to use tools
                    if response.text:
                        self.transcript.add_system_event(f"DM internal: {response.text}")

                    if self.ui:
                        self.ui.thinking_start("dm")
                    compact_history(self.dm)
                    response = self.dm.send_with_tools(
                        load_prompt("session_continue")
                    )
                    if self.ui:
                        self.ui.thinking_stop("dm")
                    self.total_turns += 1
                    self._tick("DM")
                    self._dump_agent_logs()

            self._flush_ui_wait()  # flush any pending speech at end of loop

            if self.total_turns >= self.max_turns:
                warning = (
                    f"Session truncated: reached {self.max_turns} turn limit."
                )
                self._event(warning)
                self.transcript.add_system_event(warning)

            print(f"\nSession complete! ({self.total_turns} turns)", flush=True)

        except Exception as e:
            print(f"\nSession error: {e}", flush=True)
            self.transcript.add_system_event(
                f"Session ended early due to error: {e}"
            )

        return self._build_result()

    def _process_tool_calls(
        self, response: AgentResponse
    ) -> list[tuple[str, str]]:
        """Process all tool calls from a DM response."""
        results = []

        # Any free text from the DM is internal thoughts (not narration)
        if response.text:
            self.transcript.add_system_event(f"DM internal: {response.text}")

        for tc in response.tool_calls:
            result = self._dispatch_tool(tc.name, tc.arguments)
            results.append((tc.id, result))

        return results

    def _dispatch_tool(self, name: str, args: dict) -> str:
        """Dispatch a single tool call to the appropriate handler."""
        handlers = {
            "narrate": self._handle_narrate,
            "review_note": self._handle_dm_review_note,
            "ask_skill_check": self._handle_ask_skill_check,
            "attack": self._handle_attack,
            "change_hp": self._handle_change_hp,
            "roll_initiative": self._handle_roll_initiative,
            "next_combat_turn": self._handle_next_combat_turn,
            "request_group_input": self._handle_request_group_input,
            "get_party_status": self._handle_get_party_status,
            "end_session": self._handle_end_session,
            "search_module": self._handle_search_module,
            "read_page": self._handle_read_page,
            "next_page": self._handle_next_page,
            "previous_page": self._handle_previous_page,
            "change_music": self._handle_change_music,
        }
        handler = handlers.get(name)
        if not handler:
            return f"Unknown tool: {name}"
        try:
            return handler(args)
        except Exception as e:
            self.transcript.add_system_event(f"ERROR in {name}: {e}")
            self._event(f"ERROR {name}: {e}")
            self._terminated = True
            return f"Error: {e}"

    # --- Early-out checks ---

    def _check_early_outs(self) -> str | None:
        """Check early-out conditions. Returns abort reason or None."""
        # 1. DM stuck — no tools for 5 consecutive turns
        if self._consecutive_no_tool_turns >= 5:
            return f"DM stuck: {self._consecutive_no_tool_turns} consecutive turns with no tool calls"

        # 2. No narration after 15 turns — DM is doing things but never narrating
        if self.total_turns >= 15 and self._narration_count == 0:
            return f"No narration: DM has not narrated after {self.total_turns} turns"

        # 3. Cost budget exceeded
        cost = self._estimate_current_cost()
        if cost > self._cost_budget:
            return f"Cost budget exceeded: ${cost:.2f} > ${self._cost_budget:.2f}"

        # 4. Cache health — after N turns, if cache reads are 0, caching is broken
        if self.total_turns == self._cache_check_turn:
            total_cache_read = self.dm.total_cache_read_tokens
            for p in self.players:
                total_cache_read += p.total_cache_read_tokens
            if total_cache_read == 0:
                self._event(
                    f"WARNING: 0 cache read tokens after {self.total_turns} turns — "
                    "prompt caching may not be working!"
                )
                # Don't abort, just warn — it's not fatal

        return None

    def _estimate_current_cost(self) -> float:
        """Estimate current session cost from all agents."""
        PRICING = {
            "claude-haiku-4-5-20251001": (0.80, 4.00, 1.00, 0.08),
            "claude-sonnet-4-6-20250514": (3.00, 15.00, 3.75, 0.30),
            "claude-opus-4-6-20250514": (15.00, 75.00, 18.75, 1.50),
        }
        ip, op, cwp, crp = PRICING.get(self.settings.model, (0.80, 4.00, 1.00, 0.08))

        agents = [self.dm] + self.players
        total_in = sum(a.total_input_tokens for a in agents)
        total_out = sum(a.total_output_tokens for a in agents)
        total_cw = sum(a.total_cache_creation_tokens for a in agents)
        total_cr = sum(a.total_cache_read_tokens for a in agents)
        non_cached = total_in - total_cr
        return (non_cached * ip + total_out * op + total_cw * cwp + total_cr * crp) / 1_000_000

    # --- Combat/skill handlers ---

    def _handle_narrate(self, args: dict) -> str:
        """DM narrates to the players."""
        text = args.get("text", "")
        if text:
            self.transcript.add_dm_narration(text)
            self._narration_count += 1
            if self.ui:
                # Flush any previous pending speech first
                self._flush_ui_wait()
                self.ui.speech("dm", "Dungeon Master", text)
                self._pending_ui_wait = True  # deferred — wait later
        return "Narrated."

    def _handle_dm_review_note(self, args: dict) -> str:
        """DM records a private runnability note."""
        text = args.get("text", "")
        if text:
            self.dm.add_runnability_note(text)
            self.transcript.add_system_event(f"DM review note: {text}")
        return "Noted."

    def _handle_change_music(self, args: dict) -> str:
        """Change the background music in the UI."""
        track = args.get("track", "")
        self.transcript.add_system_event(f"Music changed to: {track}")
        self._event(f"Music: {track}")
        if self.ui:
            self.ui.music_change(track)
        return f"Now playing: {track}" if track != "silence" else "Music stopped."

    def _handle_ask_skill_check(self, args: dict) -> str:
        player_name = args.get("player", "")
        skill = args.get("skill", "").lower()
        difficulty = args.get("difficulty", "medium")
        has_advantage = args.get("has_advantage", False)

        char = self.state.get_character(player_name)
        if not char:
            available = ", ".join(c.name for c in self.party)
            return f"Character '{player_name}' not found. Available: {available}"

        dc = DIFFICULTY_DC.get(difficulty, 13)
        bonus = char.skills.get(skill, 0)

        # Roll, with advantage if applicable
        if has_advantage:
            success1, total1 = self.dice.check(bonus, dc)
            success2, total2 = self.dice.check(bonus, dc)
            if total1 >= total2:
                success, total = success1, total1
            else:
                success, total = success2, total2
        else:
            success, total = self.dice.check(bonus, dc)

        result = "SUCCESS" if success else "FAILURE"
        adv_note = " (with advantage)" if has_advantage else ""
        text = (
            f"{result}: {player_name} {'passes' if success else 'fails'} "
            f"the DC {dc} {skill} check "
            f"(rolled {total}, +{bonus} bonus){adv_note}"
        )
        self.state.add_event(
            EventType.CHECK_MADE,
            text,
            actor=player_name,
        )
        self.transcript.add_system_event(f"Skill check: {text}")
        symbol = "+" if success else "-"
        self._event(f"{symbol}{player_name} {skill} DC{dc} ({total})")
        if self.ui:
            self.ui.game_event(text)
        return text

    def _handle_attack(self, args: dict) -> str:
        """Monster attacks a PC."""
        attacker_name = args.get("attacker", "")
        target_name = args.get("target", "")

        monster = self._active_monsters.get(attacker_name.lower())
        if not monster:
            return (
                f"Monster '{attacker_name}' not found in active combat. "
                f"Use roll_initiative first to register monsters."
            )

        char = self.state.get_character(target_name)
        if not char:
            available = ", ".join(c.name for c in self.party)
            return f"Character '{target_name}' not found. Available: {available}"

        # Roll attack
        success, total = self.dice.check(monster.attack_bonus, char.ac)

        if success:
            damage = max(1, round(self.dice.variance_roll(monster.damage_per_round)))
            char.current_hp = max(0, char.current_hp - damage)
            self.state.add_event(
                EventType.ATTACK,
                f"{attacker_name} hits {target_name} for {damage} damage",
                actor=attacker_name,
                target=target_name,
            )
            status = f"{char.current_hp}/{char.max_hp} HP"
            if char.current_hp == 0:
                status += " (DOWN!)"
            text = (
                f"HIT: {attacker_name} hits {target_name} for {damage} damage "
                f"({status})"
            )
            self.transcript.add_system_event(text)
            self._event(f"{attacker_name}->>{target_name} {damage}dmg ({status})")
        else:
            text = (
                f"MISS: {attacker_name} misses {target_name} "
                f"(rolled {total} vs AC {char.ac})"
            )
            self.transcript.add_system_event(text)
            self._event(f"{attacker_name}->>{target_name} miss")

        if self.ui:
            self.ui.game_event(text)
        return text

    def _handle_change_hp(self, args: dict) -> str:
        target_name = args.get("target", "")
        amount = args.get("amount", 0)
        reason = args.get("reason", "unknown")

        char = self.state.get_character(target_name)
        if not char:
            available = ", ".join(c.name for c in self.party)
            return f"Character '{target_name}' not found. Available: {available}"

        if amount < 0:
            # Damage
            char.current_hp = max(0, char.current_hp + amount)
            self.state.add_event(
                EventType.ATTACK,
                f"{reason}: {target_name} takes {abs(amount)} damage",
                target=target_name,
            )
        else:
            # Heal
            char.current_hp = min(char.max_hp, char.current_hp + amount)
            self.state.add_event(
                EventType.HEAL,
                f"{reason}: {target_name} healed for {amount}",
                target=target_name,
            )

        status = f"{target_name}: {char.current_hp}/{char.max_hp} HP ({reason})"
        if char.current_hp == 0:
            status += " (DOWN!)"

        self.transcript.add_system_event(status)
        self._event(status)
        if self.ui:
            self.ui.game_event(status)
        return status

    def _handle_roll_initiative(self, args: dict) -> str:
        monsters_data = args.get("monsters", [])
        if not monsters_data:
            return "No monsters specified."

        # Clear any prior combat state
        self._active_monsters.clear()
        self._initiative_order.clear()
        self._initiative_index = -1

        # Create and register monsters
        created_monsters = []
        for m_data in monsters_data:
            name = m_data.get("name", "Unknown")
            cr = m_data.get("cr", 1)
            try:
                monster = create_monster(name, cr)
                self._active_monsters[name.lower()] = monster
                created_monsters.append(monster)
            except ValueError as e:
                return f"Error creating monster '{name}': {e}"

        # Roll initiative for PCs
        for char in self.state.get_alive_characters():
            roll = self.dice.d20() + char.initiative_bonus
            self._initiative_order.append((char.name, roll, "PC"))

        # Roll initiative for monsters (use attack_bonus // 2 as dex proxy)
        for monster in created_monsters:
            roll = self.dice.d20() + monster.attack_bonus // 2
            self._initiative_order.append((monster.name, roll, "Monster"))

        # Sort descending
        self._initiative_order.sort(key=lambda x: x[1], reverse=True)

        self.state.in_combat = True

        # Format result
        lines = ["Initiative order:"]
        for i, (name, roll, side) in enumerate(self._initiative_order, 1):
            if side == "Monster":
                m = self._active_monsters.get(name.lower())
                extra = f" (CR {m.cr}, {m.max_hp} HP, AC {m.ac})" if m else ""
                lines.append(f"  {i}. {name}{extra} — rolled {roll}")
            else:
                lines.append(f"  {i}. {name} (PC) — rolled {roll}")

        text = "\n".join(lines)
        self.transcript.add_system_event(text)
        names = [name for name, _, side in self._initiative_order]
        self._event(f"Initiative: {' > '.join(names)}")
        if self.ui:
            self.ui.game_event(f"Initiative: {' > '.join(names)}")
        return text

    def _handle_next_combat_turn(self, args: dict) -> str:
        """Advance to the next combatant in initiative order, skipping downed PCs."""
        if not self._initiative_order:
            return "No combat in progress. Use roll_initiative first."

        order_len = len(self._initiative_order)
        # Try each slot once; if everyone is skipped we've gone full circle
        for _ in range(order_len):
            self._initiative_index = (self._initiative_index + 1) % order_len
            name, roll, side = self._initiative_order[self._initiative_index]

            # Auto-skip downed PCs (monster deaths are DM-tracked)
            if side == "PC":
                char = self.state.get_character(name)
                if char and char.current_hp <= 0:
                    continue

            position = self._initiative_index + 1
            if side == "Monster":
                m = self._active_monsters.get(name.lower())
                extra = f" (CR {m.cr}, AC {m.ac})" if m else ""
                text = f"Turn {position}/{order_len}: {name}{extra} (Monster)"
            else:
                char = self.state.get_character(name)
                hp_str = f"{char.current_hp}/{char.max_hp} HP" if char else ""
                text = f"Turn {position}/{order_len}: {name} (PC, {hp_str})"

            self.transcript.add_system_event(f"Combat turn: {text}")
            self._tick(name, side[:1])
            return text

        # All PCs downed and only monsters remain — shouldn't normally reach here
        return "No living combatants to take a turn."

    # --- Player interaction ---

    def _parallel_player_calls(
        self,
        tasks: list[tuple[PlayerAgent, str, str]],
    ) -> list[tuple[PlayerAgent, str, int, list[str]]]:
        """Fire all player API calls + tool resolution in parallel.

        Each task is (player, chat_context, prompt_suffix).
        Thinking bubbles appear/disappear per-player based on actual LLM
        latency — covering both the initial send AND the drain loop
        (submit_tool_results) which is the hidden cost.

        Returns resolved (player, text, urgency, mechanical) tuples.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # Dice and game-state mutations must be serialized
        resolve_lock = threading.Lock()

        def _call(player: PlayerAgent, chat: str, suffix: str):
            pkey = _char_key(player.character.name)
            if self.ui:
                self.ui.thinking_start(pkey)

            player.set_cached_context(chat)

            # Remind the player what they already said (prevents repetition)
            name = player.character.name
            prior = [
                line.split(": ", 1)[1]
                for line in chat.split("\n")
                if line.startswith(f"{name}: ")
            ]
            prompt = f"What does {name} do?\n\n{suffix}"
            if prior:
                prompt += (
                    f"\n\nYou already said: "
                    + " / ".join(f'"{p[:80]}"' for p in prior)
                )

            response = player.send_with_tools(prompt)

            # Resolve tools (includes drain loop with submit_tool_results).
            # The lock protects dice rolls and game-state writes inside
            # attack/heal resolution; the API call itself releases the GIL.
            with resolve_lock:
                text, urgency, mechanical = self._resolve_player_tools(
                    player, response
                )

            if self.ui:
                self.ui.thinking_stop(pkey)
            self._tick(player.character.name)
            return player, text, urgency, mechanical

        if not tasks:
            return []

        results: list[tuple[PlayerAgent, str, int, list[str]]] = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [
                pool.submit(_call, p, chat, suffix)
                for p, chat, suffix in tasks
            ]
            for future in as_completed(futures):
                results.append(future.result())
        self._dump_agent_logs()
        return results

    def _handle_request_group_input(self, args: dict) -> str:
        """Conversational group input with urgency-based turn selection.

        Caching strategy: the "chat" is a single growing text that contains
        all DM narrations, selected player actions, and game events up to now.
        Each player call sends [cached chat] + short prompt. Discarded
        responses never enter the chat. When a response wins, it's appended
        to the chat so subsequent calls see it in the cached prefix.
        """
        # Flush any pending DM speech and stop DM thinking before players think
        self._flush_ui_wait()
        if self.ui:
            self.ui.thinking_stop("dm")
        self.transcript.add_system_event("--- REQUEST GROUP INPUT ---")
        MAX_FOLLOWUP_ROUNDS = 10  # safety cap

        # Build the chat — everything that happened in the session so far.
        # This is the cached prefix, identical for all players.
        chat = self.transcript.get_game_context()
        context_tokens = len(chat) // 4
        self.transcript.add_system_event(
            f"CACHE CONTEXT ({context_tokens}~tok) — identical for all players"
        )

        alive_players = [
            (p, self.state.get_character(p.character.name))
            for p in self.players
            if self.state.get_character(p.character.name)
            and self.state.get_character(p.character.name).current_hp > 0
        ]

        if not alive_players:
            return "All players are unconscious."

        # --- Round 1: All players respond (parallel API calls) ---
        # Each player gets: [cached chat] + "What does [name] do?"
        prompt_suffix = load_prompt("player_followup")
        round1_responses: list[tuple[PlayerAgent, str, int, list[str]]] = []

        round1_responses = self._parallel_player_calls(
            [(p, chat, prompt_suffix) for p, _ in alive_players]
        )

        # Sort by urgency — winner starts the thread (random tiebreaker)
        random.shuffle(round1_responses)
        round1_responses.sort(key=lambda x: x[2], reverse=True)

        # Log all round 1 responses with urgency
        self.transcript.add_system_event(
            "Round 1 urgency: " + ", ".join(
                f"{p.character.name}={u}" for p, _, u, _ in round1_responses
            )
        )

        winner = round1_responses[0]
        thread: list[tuple[str, str, list[str]]] = [
            (winner[0].character.name, winner[1], winner[3])
        ]
        self.transcript.add_player_action(winner[0].character.name, winner[1])

        # Append winner to the chat — subsequent calls see it in the cache
        chat += f"\n{winner[0].character.name}: {winner[1]}"
        for mech in winner[3]:
            chat += f"\n[{mech}]"

        # UI: show winner's speech
        if self.ui and winner[1] and winner[1] != "pass":
            display = winner[1]
            if winner[3]:
                display += "\n" + "\n".join(f"[{m}]" for m in winner[3])
            self.ui.speech(
                _char_key(winner[0].character.name),
                winner[0].character.name,
                display,
            )

        # Log all losing round 1 responses
        for player, text, urgency, mechanical in round1_responses[1:]:
            self.transcript.add_discarded_response(
                player.character.name,
                f"[urgency {urgency}] {text}",
                urgency,
            )

        last_speaker = winner[0].character.name

        # --- Follow-up rounds: everyone except last speaker ---
        # Prefetch: start next round's API calls while user reads speech.
        # Thinking bubbles appear immediately alongside the speech bubble.
        from concurrent.futures import ThreadPoolExecutor
        prefetch_future = None
        prefetch_pool = ThreadPoolExecutor(max_workers=1)

        def _start_prefetch():
            remaining = [
                (p, chat, prompt_suffix)
                for p, _ in alive_players
                if p.character.name != last_speaker
            ]
            if remaining:
                return prefetch_pool.submit(
                    self._parallel_player_calls, remaining
                )
            return None

        # Wait for typewriter to finish, then kick off follow-up while user reads
        if self.ui:
            self.ui.wait_for_typewriter()
            prefetch_future = _start_prefetch()

        for _round in range(MAX_FOLLOWUP_ROUNDS):
            min_urgency = min(_round + 1, 5)  # 1, 2, 3, 4, 5, 5, 5, ...
            # Wait for user to continue past the previous speech
            if self.ui:
                self.ui.wait_for_continue()

            # Get results — from prefetch if available, otherwise call directly
            if prefetch_future is not None:
                all_results = prefetch_future.result()
                prefetch_future = None
            else:
                remaining = [
                    (p, chat, prompt_suffix)
                    for p, _ in alive_players
                    if p.character.name != last_speaker
                ]
                all_results = self._parallel_player_calls(remaining)

            follow_responses: list[tuple[PlayerAgent, str, int, list[str]]] = []
            passed: list[str] = []
            for player, text, urgency, mechanical in all_results:
                if text == "pass" or not text.strip():
                    passed.append(player.character.name)
                    self.transcript.add_discarded_response(
                        player.character.name, "*(pass)*", 0
                    )
                else:
                    follow_responses.append(
                        (player, text, urgency, mechanical)
                    )

            # Everyone passed → done
            if not follow_responses:
                self.transcript.add_system_event(
                    f"All players passed. Thread complete ({len(thread)} messages)."
                )
                break

            # Log urgency summary for this follow-up round
            all_this_round = [(p.character.name, u) for p, _, u, _ in follow_responses]
            all_this_round += [(n, 0) for n in passed]
            self.transcript.add_system_event(
                f"Follow-up round {_round + 1}: "
                + ", ".join(f"{n}={u}" for n, u in all_this_round)
            )

            # Pick highest urgency as winner — must meet rising threshold
            random.shuffle(follow_responses)
            follow_responses.sort(key=lambda x: x[2], reverse=True)
            if follow_responses[0][2] < min_urgency:
                self.transcript.add_system_event(
                    f"No response met urgency threshold {min_urgency}. Thread complete."
                )
                break
            fw = follow_responses[0]
            thread.append((fw[0].character.name, fw[1], fw[3]))
            self.transcript.add_player_action(fw[0].character.name, fw[1])

            # Append winner to the chat
            chat += f"\n{fw[0].character.name}: {fw[1]}"
            for mech in fw[3]:
                chat += f"\n[{mech}]"

            # UI: show follow-up winner's speech
            if self.ui and fw[1] and fw[1] != "pass":
                display = fw[1]
                if fw[3]:
                    display += "\n" + "\n".join(f"[{m}]" for m in fw[3])
                self.ui.speech(
                    _char_key(fw[0].character.name),
                    fw[0].character.name,
                    display,
                )

            # Log losing follow-up responses
            for player, text, urgency, mechanical in follow_responses[1:]:
                self.transcript.add_discarded_response(
                    player.character.name,
                    f"[urgency {urgency}] {text}",
                    urgency,
                )

            last_speaker = fw[0].character.name

            # Wait for typewriter to finish, then prefetch next follow-up
            if self.ui:
                self.ui.wait_for_typewriter()
                prefetch_future = _start_prefetch()
        else:
            self.transcript.add_system_event(
                f"Group input hit {MAX_FOLLOWUP_ROUNDS} round cap."
            )

        prefetch_pool.shutdown(wait=False)

        # --- Bundle the thread for the DM ---
        result_lines = ["Player responses:"]
        for name, text, mechanical in thread:
            result_lines.append(f"{name}: {text}")
            for mech in mechanical:
                result_lines.append(f"[{mech}]")

        return "\n".join(result_lines)

    def _resolve_player_tools(
        self, player: PlayerAgent, response: AgentResponse
    ) -> tuple[str, int, list[str]]:
        """Process player tool calls. Returns (text, urgency, mechanical_results)."""
        mechanical_results: list[str] = []
        say_text = ""
        urgency = 3  # default
        has_pass = False

        if not response.tool_calls:
            # Fallback: player responded with plain text instead of tools
            return response.text, urgency, mechanical_results

        # Process tool calls
        tool_results_for_api: list[tuple[str, str]] = []
        for tc in response.tool_calls:
            if tc.name == "say":
                say_text = re.sub(r'\s*\[URGENCY:\s*\d+\]', '', tc.arguments.get("text", "")).strip()
                urgency = tc.arguments.get("urgency", 3)
                urgency = max(1, min(5, urgency))
                tool_results_for_api.append((tc.id, "Said."))
            elif tc.name == "pass_turn":
                has_pass = True
                tool_results_for_api.append((tc.id, "You passed your turn."))
            elif tc.name == "review_note":
                note = tc.arguments.get("text", "")
                if note:
                    player.add_engagement_note(note)
                    self.transcript.add_system_event(
                        f"{player.character.name} review note: {note}"
                    )
                tool_results_for_api.append((tc.id, "Noted."))
            elif tc.name == "attack":
                result = self._resolve_player_attack(player, tc.arguments)
                mechanical_results.append(result)
                tool_results_for_api.append((tc.id, result))
            elif tc.name == "heal":
                result = self._resolve_player_heal(player, tc.arguments)
                mechanical_results.append(result)
                tool_results_for_api.append((tc.id, result))
            else:
                tool_results_for_api.append((tc.id, f"Unknown tool: {tc.name}"))

        # Submit results back to player — drain any follow-up tool calls
        # This loop ensures ALL tool_use blocks get matching tool_result blocks,
        # preventing "tool_use ids without tool_result" API errors on future calls.
        max_drain = 5  # safety valve
        while tool_results_for_api and max_drain > 0:
            max_drain -= 1
            followup = player.submit_tool_results(tool_results_for_api)
            if not followup.tool_calls:
                break
            # Model responded with more tool calls — process them
            tool_results_for_api = []
            for tc in followup.tool_calls:
                if tc.name == "say" and not say_text:
                    say_text = re.sub(r'\s*\[URGENCY:\s*\d+\]', '', tc.arguments.get("text", "")).strip()
                    urgency = max(1, min(5, tc.arguments.get("urgency", 3)))
                    tool_results_for_api.append((tc.id, "Said."))
                elif tc.name == "pass_turn":
                    has_pass = True
                    tool_results_for_api.append((tc.id, "You passed your turn."))
                elif tc.name == "review_note":
                    note = tc.arguments.get("text", "")
                    if note:
                        player.add_engagement_note(note)
                    tool_results_for_api.append((tc.id, "Noted."))
                elif tc.name == "attack":
                    result = self._resolve_player_attack(player, tc.arguments)
                    mechanical_results.append(result)
                    tool_results_for_api.append((tc.id, result))
                elif tc.name == "heal":
                    result = self._resolve_player_heal(player, tc.arguments)
                    mechanical_results.append(result)
                    tool_results_for_api.append((tc.id, result))
                else:
                    tool_results_for_api.append((tc.id, f"Unknown tool: {tc.name}"))

        if has_pass:
            return "pass", 0, mechanical_results

        return say_text, urgency, mechanical_results

    def _resolve_player_attack(self, player: PlayerAgent, args: dict) -> str:
        """Resolve a player's attack against a monster."""
        target_name = args.get("target", "")
        char = player.character

        monster = self._active_monsters.get(target_name.lower())
        if not monster:
            return f"No monster '{target_name}' in combat."

        success, total = self.dice.check(char.attack_bonus, monster.ac)
        if success:
            damage = max(1, round(self.dice.variance_roll(char.avg_damage)))
            # Do NOT modify monster HP — the DM tracks it
            result = (
                f"{char.name} attacks {target_name}: "
                f"HIT for {damage} damage (rolled {total} vs AC {monster.ac})"
            )
        else:
            result = (
                f"{char.name} attacks {target_name}: "
                f"MISS (rolled {total} vs AC {monster.ac})"
            )
        self.transcript.add_system_event(result)
        if success:
            self._event(f"{char.name}->>{target_name} {damage}dmg")
        else:
            self._event(f"{char.name}->>{target_name} miss")
        if self.ui:
            self.ui.game_event(result)
        return result

    def _resolve_player_heal(self, player: PlayerAgent, args: dict) -> str:
        """Resolve a player's heal action."""
        target_name = args.get("target", "")
        char = player.character

        target_char = self.state.get_character(target_name)
        if not target_char:
            available = ", ".join(c.name for c in self.party)
            return f"Character '{target_name}' not found. Available: {available}"

        # Check and deduct spell slot
        if not char.spell_slots:
            return f"{char.name} has no spell slots to heal with."
        lowest_slot = min(char.spell_slots.keys())
        if char.spell_slots[lowest_slot] <= 0:
            return f"{char.name} has no remaining spell slots."
        char.spell_slots[lowest_slot] -= 1
        if char.spell_slots[lowest_slot] <= 0:
            del char.spell_slots[lowest_slot]

        # Roll healing: use avg_damage as a proxy for healing power
        heal_avg = max(4.0, char.avg_damage * 0.5)
        heal_amount = max(1, round(self.dice.variance_roll(heal_avg)))
        old_hp = target_char.current_hp
        target_char.current_hp = min(target_char.max_hp, target_char.current_hp + heal_amount)
        actual = target_char.current_hp - old_hp

        self.state.add_event(
            EventType.HEAL,
            f"{char.name} heals {target_name} for {actual}",
            actor=char.name,
            target=target_name,
        )
        result = (
            f"{char.name} heals {target_name} for {actual} HP "
            f"({target_char.current_hp}/{target_char.max_hp})"
        )
        self.transcript.add_system_event(result)
        self._event(f"{char.name} heals {target_name} +{actual}HP")
        if self.ui:
            self.ui.game_event(result)
        return result

    # --- Unchanged handlers ---

    def _handle_get_party_status(self, args: dict) -> str:
        lines = []
        for char in self.party:
            slots = ""
            if char.spell_slots:
                slots = f", spell slots: {dict(char.spell_slots)}"
            status = "DOWN" if char.current_hp <= 0 else "OK"
            # Top skills
            top_skills = sorted(
                char.skills.items(), key=lambda x: x[1], reverse=True
            )[:3]
            skill_str = ", ".join(
                f"{s.replace('_', ' ').title()} +{b}" for s, b in top_skills
            )
            lines.append(
                f"- {char.name} ({char.char_class} L{char.level}): "
                f"{char.current_hp}/{char.max_hp} HP, AC {char.ac}, "
                f"attack +{char.attack_bonus}{slots} [{status}]"
            )
            if skill_str:
                lines.append(f"  Skills: {skill_str}")
        return "Party Status:\n" + "\n".join(lines)

    def _handle_search_module(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return "No search query provided."
        if not self.pages:
            return "Module pages not loaded."

        self.module_references.append({
            "tool": "search_module",
            "query": query,
            "turn": self.total_turns,
        })
        self._tick("DM", f"search:{query[:20]}")
        self.transcript.add_system_event(f"DM searches module: \"{query}\"")

        matches = []
        query_lower = query.lower()
        for i, page_text in enumerate(self.pages):
            page_lower = page_text.lower()
            pos = page_lower.find(query_lower)
            if pos != -1:
                # Extract ~150 chars around the match
                start = max(0, pos - 75)
                end = min(len(page_text), pos + len(query) + 75)
                snippet = page_text[start:end].replace("\n", " ").strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(page_text):
                    snippet = snippet + "..."
                matches.append(f"Page {i + 1}: {snippet}")
                if len(matches) >= 5:
                    break

        if not matches:
            return f"No matches found for \"{query}\"."
        return "Search results:\n" + "\n\n".join(matches)

    def _handle_read_page(self, args: dict) -> str:
        page_number = args.get("page_number", 0)
        if not self.pages:
            return "Module pages not loaded."
        if page_number < 1 or page_number > len(self.pages):
            return (
                f"Invalid page number: {page_number}. "
                f"Module has {len(self.pages)} pages (1-{len(self.pages)})."
            )

        self._last_read_page = page_number
        self.module_references.append({
            "tool": "read_page",
            "page": page_number,
            "turn": self.total_turns,
        })
        self._tick("DM", f"p{page_number}")
        self.transcript.add_system_event(f"DM reads page {page_number}")
        return f"--- Page {page_number} ---\n{self.pages[page_number - 1]}"

    def _handle_next_page(self, args: dict) -> str:
        if not self.pages:
            return "Module pages not loaded."
        if self._last_read_page is None:
            target = 1
        else:
            target = self._last_read_page + 1

        if target > len(self.pages):
            return f"Already at the last page ({len(self.pages)})."
        return self._handle_read_page({"page_number": target})

    def _handle_previous_page(self, args: dict) -> str:
        if not self.pages:
            return "Module pages not loaded."
        if self._last_read_page is None:
            return "No page read yet. Use read_page or next_page first."
        target = self._last_read_page - 1
        if target < 1:
            return "Already at the first page."
        return self._handle_read_page({"page_number": target})

    def _handle_end_session(self, args: dict) -> str:
        reason = args.get("reason", "Adventure complete")
        self._terminated = True
        self._flush_ui_wait()  # flush any pending speech
        self.transcript.add_system_event(f"Session ended: {reason}")
        self._event(f"Session ended: {reason}")
        if self.ui:
            self.ui.session_end(reason)
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
            module_references=self.module_references,
        )



@dataclass
class SessionResult:
    """Result of a completed session."""

    transcript: SessionTranscript
    state: GameState
    dm: DMAgent
    players: list[PlayerAgent]
    token_usage: dict
    module_references: list[dict] = field(default_factory=list)

    def get_transcript_text(self) -> str:
        return self.transcript.to_text()
