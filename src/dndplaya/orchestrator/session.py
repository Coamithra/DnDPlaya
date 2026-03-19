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
from ..agents.provider import ProviderGuardrails
from ..agents.context import compact_history
from ..prompts import load_prompt
from .transcript import SessionTranscript

DEFAULT_MAX_TURNS = 100

DIFFICULTY_DC = {
    "very_easy": 5,
    "easy": 10,
    "medium": 15,
    "hard": 20,
    "very_hard": 25,
    "nearly_impossible": 30,
}


def _char_key(name: str) -> str:
    """Normalise a character name to a UI element key."""
    return name.lower().replace(" ", "_")


@dataclass
class ValidationResult:
    """Outcome of validating a player response."""

    status: str  # "ok", "fixed", "needs_retry"
    response: AgentResponse | None = None  # set when status == "fixed"
    hint: str = ""  # correction hint appended to prompt on retry


def _has_excessive_non_ascii(text: str, threshold: float = 0.30) -> bool:
    """Return True if more than *threshold* fraction of characters are non-ASCII.

    Used to detect when a local model (e.g. qwen2.5) switches to Chinese,
    Russian, Thai, etc. mid-response.  Simple heuristic — no external
    language-detection library needed.
    """
    if not text:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / len(text) > threshold


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
        room_map: str = "",
    ):
        self.settings = settings
        self.dice = DiceRoller(seed=seed or settings.seed)

        # Module reference state
        self.pages = pages
        self._room_map = room_map
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
                room_map=room_map,
            )
        else:
            self.dm = DMAgent(
                summary=module_markdown,
                settings=settings,
                map_images=map_images,
                music_tracks=music_tracks,
                enable_reviews=enable_reviews,
                room_map=room_map,
            )

        # Provider guardrails — the provider declares what constraints it needs
        self._guardrails: ProviderGuardrails = self.dm.provider.guardrails

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

        # Staleness detection — consecutive DM turns without module reference
        self._turns_without_module_ref = 0
        self._STALE_THRESHOLD = 3  # nudge after this many turns without a ref

        # All-pass tracking — consecutive group inputs where no player contributed
        self._consecutive_all_pass = 0
        self._ALL_PASS_THRESHOLD = 2  # force story advance after this many

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

    def _log_side_call(
        self, name: str, system: str, prompt: str, result: str,
    ) -> None:
        """Log a side-call (summarizer, synthesizer) to agent_logs/side_calls.txt."""
        if not self._log_dir:
            return
        logs_dir = self._log_dir / "agent_logs"
        logs_dir.mkdir(exist_ok=True)
        with open(logs_dir / "side_calls.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{name} turn={self.total_turns}]\n")
            f.write(f"{'='*60}\n")
            f.write(f"--- SYSTEM ---\n{system}\n\n")
            f.write(f"--- PROMPT ---\n{prompt}\n\n")
            f.write(f"--- RESPONSE ---\n{result}\n\n")

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

    # Bootstrap queries — separate focused questions that each search a
    # small slice of the module.  Chaining across the whole module doesn't
    # work with small models (recency bias overwrites earlier content).
    _BOOTSTRAP_QUERIES = [
        ("introduction overview background",
         "What is this module about? Setting, adventure overview, and tone."),
        ("villain boss leader named",
         "Who are the main NPCs or villains? Names, roles, motivations. No stat blocks."),
        ("hook adventure background quest reward",
         "What are the adventure hooks — reasons for adventurers to visit?"),
        ("entrance door gate tunnel start",
         "Where is the dungeon entrance and what does the party encounter first?"),
    ]

    def _bootstrap_module_knowledge(self) -> str:
        """Run focused search+summarize queries to build a prep sheet.

        Each query targets a different aspect of the module with narrow
        search terms, avoiding the recency-bias problem of chaining
        through 20+ pages in one query.
        """
        if not self.pages:
            return ""

        print("\n  Bootstrapping module knowledge...", end="", flush=True)
        self.transcript.add_system_event("Bootstrapping module knowledge...")

        sections: list[str] = []
        for search_terms, question in self._BOOTSTRAP_QUERIES:
            result = self._handle_search_module({
                "search_terms": search_terms,
                "question": question,
            })
            if result and "No matches found" not in result:
                sections.append(result)

        if not sections:
            return ""

        prep_sheet = "## Module Prep Sheet\n\n" + "\n\n".join(sections)
        print(f" done ({len(sections)} sections)", flush=True)
        self.transcript.add_system_event(f"MODULE PREP SHEET:\n\n{prep_sheet}")
        return prep_sheet

    def run(self) -> SessionResult:
        """Run the complete session via DM conversation loop."""
        print(f"Session start | Party: {', '.join(c.name for c in self.party)}", flush=True)

        try:
            # Bootstrap module knowledge via search queries (replaces upfront summary)
            prep_sheet = self._bootstrap_module_knowledge()

            # Build opening prompt
            party_desc = "\n".join(
                f"- {c.name} ({c.char_class} level {c.level}, {c.pronouns}): "
                f"{c.max_hp} HP, AC {c.ac}"
                for c in self.party
            )
            opening = load_prompt("session_start", party_description=party_desc)
            if prep_sheet:
                opening = prep_sheet + "\n\n" + opening

            # First DM turn
            if self.ui:
                self.ui.thinking_start("dm")
            compact_history(self.dm, self._guardrails.compaction_threshold)
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

                # Staleness detection: track consecutive DM turns without
                # a module reference tool (search_module, read_page, etc.)
                # The counter is reset to 0 inside the ref tool handlers.
                self._turns_without_module_ref += 1

                # Early-out checks
                abort_reason = self._check_early_outs()
                if abort_reason:
                    self._event(f"ABORT: {abort_reason}")
                    self.transcript.add_system_event(
                        f"Session aborted: {abort_reason}"
                    )
                    self._terminated = True
                    break

                # Build a staleness nudge if the DM hasn't used module
                # reference tools recently.
                _stale_nudge = ""
                if (
                    self._turns_without_module_ref >= self._STALE_THRESHOLD
                    and self.pages
                ):
                    _stale_nudge = (
                        "\n[SYSTEM: You haven't consulted the module in "
                        f"{self._turns_without_module_ref} turns. Use "
                        "search_module or read_page to find what happens "
                        "next in the dungeon. Do not improvise — check the "
                        "module.]"
                    )
                    self.transcript.add_system_event(
                        f"Staleness nudge injected (no module ref for "
                        f"{self._turns_without_module_ref} turns)"
                    )

                if response.tool_calls:
                    self._consecutive_no_tool_turns = 0
                    # Process tool calls and submit results back to DM
                    tool_results = self._process_tool_calls(response)

                    if self._terminated or self._is_party_dead():
                        break

                    # Append staleness nudge to the last tool result if needed
                    if _stale_nudge and tool_results:
                        last_id, last_text = tool_results[-1]
                        tool_results[-1] = (last_id, last_text + _stale_nudge)

                    if self.ui:
                        self.ui.thinking_start("dm")
                    compact_history(self.dm, self._guardrails.compaction_threshold)
                    response = self.dm.submit_tool_results(tool_results)
                    if self.ui:
                        self.ui.thinking_stop("dm")
                    self.total_turns += 1
                    self._tick("DM")
                    self._dump_agent_logs()
                else:
                    self._consecutive_no_tool_turns += 1

                    # Local models often write narration as plain text instead
                    # of using tools.  Synthesize tool calls, dispatch them
                    # directly, and feed results back as a regular user message
                    # (not submit_tool_results, which would create invalid history).
                    if response.text and self.settings.provider == "ollama":
                        synthesized = self._synthesize_dm_tools(response.text)
                        if synthesized:
                            self._consecutive_no_tool_turns = 0
                            synth_response = AgentResponse(
                                text=response.text,
                                tool_calls=synthesized,
                                raw_content=response.raw_content,
                                stop_reason="tool_use",
                            )
                            tool_results = self._process_tool_calls(synth_response)
                            if self._terminated or self._is_party_dead():
                                break
                            # Feed results back as a regular message, not submit_tool_results
                            summary = "\n".join(f"[{r}]" for _, r in tool_results)
                            if _stale_nudge:
                                summary += _stale_nudge
                            if self.ui:
                                self.ui.thinking_start("dm")
                            compact_history(self.dm, self._guardrails.compaction_threshold)
                            response = self.dm.send_with_tools(summary)
                            if self.ui:
                                self.ui.thinking_stop("dm")
                            self.total_turns += 1
                            self._tick("DM")
                            self._dump_agent_logs()
                            continue

                    # DM produced text without tools — internal thinking, nudge to use tools
                    if response.text:
                        # Scrub Qwen template tokens before logging
                        import re as _re
                        clean_text = _re.sub(r'<\|im_start\|>.*?(?:<\|im_end\|>)?', '', response.text, flags=_re.DOTALL).strip()
                        if clean_text:
                            self.transcript.add_system_event(f"DM internal: {clean_text}")

                    nudge_msg = load_prompt("session_continue")
                    if _stale_nudge:
                        nudge_msg += _stale_nudge
                    if self.ui:
                        self.ui.thinking_start("dm")
                    compact_history(self.dm, self._guardrails.compaction_threshold)
                    response = self.dm.send_with_tools(nudge_msg)
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
        """Estimate current session cost from all agents. Returns 0 for local providers."""
        if self.settings.provider == "ollama":
            return 0.0

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

    # --- Local model helpers ---

    @staticmethod
    def _clean_local_model_text(text: str) -> str:
        """Strip XML/JSON tool-call artifacts from local model text output."""
        import re
        # Remove Qwen chat template delimiters
        text = re.sub(r'<\|im_start\|>.*?(?:<\|im_end\|>)?', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|im_end\|>', '', text)
        # Remove XML-like tool call tags and their JSON content
        text = re.sub(r'</?tool_call>', '', text)
        # Remove JSON tool call objects {"name": "...", "arguments": {...}}
        text = re.sub(r'\{"name":\s*"\w+",\s*"arguments":\s*\{[^}]*\}\}', '', text)
        # Remove garbled prefixes Qwen generates before tool calls
        text = re.sub(r'\b\w*(StateChange|Button|Asstist)\w*', '', text, flags=re.IGNORECASE)
        # Remove [URGENCY: N] tags (already parsed elsewhere)
        text = re.sub(r'\s*\[URGENCY:\s*\d+\]', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    def _synthesize_dm_tools(self, text: str) -> list:
        """Synthesize tool calls from DM plain-text output (local models).

        Local models often write narration as prose instead of using narrate().
        This extracts the narration and optionally adds request_group_input()
        if the text looks like the DM is asking for player actions.
        """
        import re
        from ..agents.base import ToolCall

        calls: list[ToolCall] = []

        # Strip out any text that looks like tool-call syntax the model wrote
        # (already handled by the provider, but may still appear)
        clean = re.sub(r'\(?request_group_input\s*\(?\)?\)?', '', text).strip()
        clean = re.sub(r'\(?narrate\s*\([^)]*\)\)?', '', clean).strip()
        # Strip Qwen chat template delimiters
        clean = re.sub(r'<\|im_start\|>.*?(?:<\|im_end\|>)?', '', clean, flags=re.DOTALL).strip()

        if clean:
            calls.append(ToolCall(
                id=f"synth_narrate",
                name="narrate",
                arguments={"text": clean},
            ))

        # If the DM asks a question or prompts for action, add group input
        prompt_patterns = [
            r'what do you do',
            r'what does .+ do',
            r'what would you like',
            r'how do you proceed',
            r'what.+action',
            r'discuss amongst yourselves',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in prompt_patterns):
            calls.append(ToolCall(
                id=f"synth_group_input",
                name="request_group_input",
                arguments={},
            ))

        return calls

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
            snapshot = player.snapshot_history()

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

            # Validate + retry loop: send, validate, fix or retry.
            max_retries = 3
            for _attempt in range(max_retries):
                response = player.send_with_tools(prompt)
                result = self._validate_player_response(player, response)

                if result.status == "ok":
                    break
                if result.status == "fixed":
                    response = result.response  # use cleaned-up version
                    break
                # needs_retry — roll back and resend with correction hint
                player.rollback_history(snapshot)
                player.set_cached_context(chat)
                prompt += f"\n\n{result.hint}"

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

        # Local providers (Ollama) can only run one inference at a time,
        # so serial calls avoid thread overhead. Cloud APIs benefit from
        # parallel calls (~4x faster with ThreadPoolExecutor).
        if self.settings.provider == "ollama":
            for p, chat, suffix in tasks:
                results.append(_call(p, chat, suffix))
        else:
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

        # Check if ALL players passed in round 1
        all_passed_r1 = all(
            (text == "pass" or not text.strip() or urgency == 0)
            for _, text, urgency, _ in round1_responses
        )
        if all_passed_r1:
            self._consecutive_all_pass += 1
            self.transcript.add_system_event(
                f"All players passed ({self._consecutive_all_pass} consecutive)."
            )
            # After threshold, tell DM to advance the story instead of
            # requesting more player input (players won't respond on retry).
            if self._consecutive_all_pass >= self._ALL_PASS_THRESHOLD:
                self.transcript.add_system_event(
                    "All-pass threshold reached — injecting story advance nudge."
                )
                return (
                    "Players have not responded for multiple rounds. "
                    "Do NOT call request_group_input again. Instead, advance "
                    "the story: use search_module to find the next encounter, "
                    "roll_initiative to start combat, or use ask_skill_check "
                    "to create a challenge. The party needs action, not more "
                    "questions."
                )
            return "All players passed. No input received."
        else:
            # Reset counter on any meaningful input
            self._consecutive_all_pass = 0

        winner = round1_responses[0]
        thread: list[tuple[str, str, list[str]]] = [
            (winner[0].character.name, winner[1], winner[3])
        ]
        self.transcript.add_player_action(winner[0].character.name, winner[1])

        # Append winner to the chat — subsequent calls see it in the cache.
        # Skip "pass" entries to avoid encouraging other players to pass.
        if winner[1] and winner[1] != "pass":
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

            # Append winner to the chat (skip "pass" to avoid encouraging others)
            if fw[1] and fw[1] != "pass":
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

    def _validate_player_response(
        self, player: PlayerAgent, response: AgentResponse,
    ) -> ValidationResult:
        """Validate a player response. Returns ok / fixed / needs_retry."""
        g = self._guardrails
        name = player.character.name

        # --- Collect all text from the response (free text + say args) ---
        all_text = []
        if response.text:
            all_text.append(response.text)
        for tc in response.tool_calls:
            if tc.name == "say":
                all_text.append(str(tc.arguments.get("text", "")))
        combined = " ".join(all_text)

        # --- Non-ASCII: entire response is unsalvageable → needs_retry ---
        if g.detect_non_ascii and combined.strip():
            if _has_excessive_non_ascii(combined, g.non_ascii_threshold):
                self._event(f"WARN: {name} non-English — needs retry")
                return ValidationResult(
                    status="needs_retry",
                    hint="IMPORTANT: Respond in English only.",
                )

        # --- Role confusion: strip "DM:" content from say() args → fixed ---
        fixed_any = False
        if g.detect_role_confusion and response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "say":
                    text = str(tc.arguments.get("text", ""))
                    dm_match = re.search(r'\bDM:', text)
                    if dm_match:
                        tc.arguments["text"] = text[:dm_match.start()].strip()
                        fixed_any = True
            if fixed_any:
                self._event(f"WARN: {name} role-confused — fixed")
                self.transcript.add_system_event(
                    f"Role confusion detected: {name} "
                    "hallucinated DM response — stripped."
                )

        # --- Empty say: convert say(text="") to pass_turn → fixed ---
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "say":
                    raw = tc.arguments.get("text", "")
                    if not isinstance(raw, str):
                        raw = str(raw)
                    cleaned = re.sub(r'\s*\[URGENCY:\s*\d+\]', '', raw).strip()
                    if not cleaned:
                        tc.name = "pass_turn"
                        tc.arguments = {}
                        fixed_any = True

        if fixed_any:
            return ValidationResult(status="fixed", response=response)

        return ValidationResult(status="ok")

    def _resolve_player_tools(
        self, player: PlayerAgent, response: AgentResponse
    ) -> tuple[str, int, list[str]]:
        """Process player tool calls. Returns (text, urgency, mechanical_results)."""
        mechanical_results: list[str] = []
        say_text = ""
        urgency = 3  # default
        has_pass = False

        if not response.tool_calls:
            # Fallback: player responded with plain text instead of tools.
            # Clean up common local-model garbage (XML tags, JSON fragments).
            text = response.text
            if self.settings.provider == "ollama" and text:
                text = self._clean_local_model_text(text)
            return text, urgency, mechanical_results

        # Process tool calls — validation already happened upstream
        g = self._guardrails
        tool_call_count = 0

        tool_results_for_api: list[tuple[str, str]] = []
        for tc in response.tool_calls:
            if tc.name == "say":
                raw_text = tc.arguments.get("text", "")
                if not isinstance(raw_text, str):
                    raw_text = str(raw_text)
                say_text = re.sub(r'\s*\[URGENCY:\s*\d+\]', '', raw_text).strip()

                tool_call_count += 1
                raw_urg = tc.arguments.get("urgency", 3)
                urgency = max(1, min(5, int(raw_urg) if isinstance(raw_urg, (int, float)) else 3))
                tool_results_for_api.append((tc.id, "Said."))
            elif tc.name == "pass_turn":
                tool_call_count += 1
                has_pass = True
                tool_results_for_api.append((tc.id, "You passed your turn."))
            elif tc.name == "review_note":
                tool_call_count += 1
                note = tc.arguments.get("text", "")
                if note:
                    player.add_engagement_note(note)
                    self.transcript.add_system_event(
                        f"{player.character.name} review note: {note}"
                    )
                tool_results_for_api.append((tc.id, "Noted."))
            elif tc.name == "attack":
                tool_call_count += 1
                result = self._resolve_player_attack(player, tc.arguments)
                mechanical_results.append(result)
                tool_results_for_api.append((tc.id, result))
            elif tc.name == "heal":
                tool_call_count += 1
                result = self._resolve_player_heal(player, tc.arguments)
                mechanical_results.append(result)
                tool_results_for_api.append((tc.id, result))
            else:
                tool_call_count += 1
                tool_results_for_api.append((tc.id, f"Unknown tool: {tc.name}"))

        # Submit results back to player — drain any follow-up tool calls
        # This loop ensures ALL tool_use blocks get matching tool_result blocks,
        # preventing "tool_use ids without tool_result" API errors on future calls.
        # Fix 3: hard cap on total tool calls per player per group input round.
        max_drain = 5  # safety valve
        while tool_results_for_api and max_drain > 0:
            # Provider guardrail: stop draining if we've hit the per-player cap
            if g.drain_loop_cap and tool_call_count >= g.drain_loop_cap:
                self._event(f"WARN: {player.character.name} hit {g.drain_loop_cap}-tool cap")
                self.transcript.add_system_event(
                    f"{player.character.name} hit per-player tool call cap "
                    f"({g.drain_loop_cap}). Stopping drain loop."
                )
                # Still need to submit final results to close the API loop
                player.submit_tool_results(tool_results_for_api)
                break

            max_drain -= 1
            followup = player.submit_tool_results(tool_results_for_api)
            if not followup.tool_calls:
                break
            # Model responded with more tool calls — process them
            tool_results_for_api = []
            for tc in followup.tool_calls:
                if tc.name == "say":
                    if not say_text:
                        raw_text = tc.arguments.get("text", "")
                        if not isinstance(raw_text, str):
                            raw_text = str(raw_text)
                        candidate = re.sub(r'\s*\[URGENCY:\s*\d+\]', '', raw_text).strip()

                        # Empty say in drain loop (checked BEFORE incrementing tool count)
                        if not candidate:
                            has_pass = True
                            tool_results_for_api.append((tc.id, "Empty say — treated as pass."))
                            continue

                        say_text = candidate
                        raw_urg = tc.arguments.get("urgency", 3)
                        urgency = max(1, min(5, int(raw_urg) if isinstance(raw_urg, (int, float)) else 3))
                    tool_call_count += 1
                    tool_results_for_api.append((tc.id, "Said."))
                elif tc.name == "pass_turn":
                    tool_call_count += 1
                    has_pass = True
                    tool_results_for_api.append((tc.id, "You passed your turn."))
                elif tc.name == "review_note":
                    tool_call_count += 1
                    note = tc.arguments.get("text", "")
                    if note:
                        player.add_engagement_note(note)
                    tool_results_for_api.append((tc.id, "Noted."))
                elif tc.name == "attack":
                    tool_call_count += 1
                    result = self._resolve_player_attack(player, tc.arguments)
                    mechanical_results.append(result)
                    tool_results_for_api.append((tc.id, result))
                elif tc.name == "heal":
                    tool_call_count += 1
                    result = self._resolve_player_heal(player, tc.arguments)
                    mechanical_results.append(result)
                    tool_results_for_api.append((tc.id, result))
                else:
                    tool_call_count += 1
                    tool_results_for_api.append((tc.id, f"Unknown tool: {tc.name}"))

        # If we got a real say_text, return it even if empty says also set has_pass
        if say_text:
            return say_text, urgency, mechanical_results

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

        # Fix 5: Guard against healing at full HP — don't waste spell slot
        if target_char.current_hp >= target_char.max_hp:
            result = (
                f"{target_name} is already at full HP "
                f"({target_char.current_hp}/{target_char.max_hp}). "
                f"Heal not needed — spell slot preserved."
            )
            self.transcript.add_system_event(result)
            return result

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

    def _get_page_window(self, page_number: int) -> str:
        """Return the text of a page plus its ±1 neighbors.

        Content that spans page boundaries (room descriptions, stat blocks)
        is captured by including adjacent pages.  The caller is responsible
        for deduplication when multiple hits are close together.

        TODO: Consider chunking by headings instead of PDF pages for
        better logical boundaries.  Also consider a sliding-window
        heuristic that only grabs neighbors when the match is near
        a page edge.
        """
        if not self.pages:
            return ""
        parts: list[str] = []
        for p in (page_number - 1, page_number, page_number + 1):
            if 1 <= p <= len(self.pages):
                parts.append(f"--- Page {p} ---\n{self.pages[p - 1]}")
        return "\n\n".join(parts)

    def _summarize_page_window(
        self, page_window: str, question: str, prior_summary: str = "",
    ) -> str:
        """Side-call: ask the LLM to extract relevant info from page(s).

        Creates a temporary BaseAgent with a focused system prompt, sends
        the page text + question, and returns a short summary (~200 words).

        *page_window* may contain 1-3 pages (the hit page ± neighbors).
        *prior_summary* — if provided, the LLM is asked to add to this
        rather than starting from scratch.  This chains summaries so each
        builds on the last, avoiding redundancy and eliminating the need
        for a separate synthesis step.
        """
        from ..agents.base import BaseAgent

        system = (
            "You are reading pages from a D&D module. Be concise "
            "(200 words max). Include specific numbers (HP, AC, CR, "
            "DC, damage, quantities). Only include information from "
            "the provided pages."
        )
        if self._room_map:
            system += (
                "\n\nFor spatial context, here is the dungeon room map:\n"
                f"{self._room_map}"
            )

        agent = BaseAgent(
            name="PageSummarizer",
            system_prompt=system,
            settings=self.settings,
        )
        if prior_summary:
            prompt = (
                f"Here is a page from a D&D module. The question is: {question}\n\n"
                f"Current summary:\n{prior_summary}\n\n"
                f"New page:\n{page_window}\n\n"
                f"Add any relevant info from this page to the summary and "
                f"return the updated summary:"
            )
        else:
            prompt = (
                f"Here is a page from a D&D module. The question is: {question}\n\n"
                f"{page_window}\n\n"
                f"Extract the relevant information:"
            )
        try:
            result = agent.send(prompt)
            self._log_side_call("PageSummarizer", system, prompt, result)
            return result
        except Exception as e:
            return f"(summarization error: {e})"

    def _synthesize_fragments(
        self, fragments: list[tuple[int, str]], question: str,
    ) -> str:
        """Combine per-page summaries into one coherent answer.

        Only called when multiple pages matched a search.
        """
        from ..agents.base import BaseAgent

        agent = BaseAgent(
            name="Synthesizer",
            system_prompt=(
                "You are a module reference assistant. Given summaries from "
                "different pages of a D&D module, combine them into ONE "
                "concise, comprehensive answer. Keep specific numbers. "
                "Do not add information that isn't in the fragments."
            ),
            settings=self.settings,
        )
        body = "\n\n".join(
            f"Page {page}: {summary}" for page, summary in fragments
        )
        prompt = (
            f"Question: {question}\n\n"
            f"--- Page summaries ---\n{body}\n--- End ---\n\n"
            f"Provide a combined answer:"
        )
        try:
            return agent.send(prompt)
        except Exception as e:
            # Fall back to raw fragments
            return "\n".join(f"Page {p}: {s}" for p, s in fragments)

    def _handle_search_module(self, args: dict) -> str:
        # Accept both "search_terms" (new) and "query" (backward compat)
        search_terms = args.get("search_terms", args.get("query", ""))
        question = args.get("question", "")

        if not search_terms:
            return "No search query provided."
        if not self.pages:
            return "Module pages not loaded."

        self.module_references.append({
            "tool": "search_module",
            "query": search_terms,
            "turn": self.total_turns,
        })
        self._turns_without_module_ref = 0
        self._tick("DM", f"search:{search_terms[:20]}")

        # Transcript: metadata header (single-line → backticks)
        header = f'DM is researching the module... Looking for: "{search_terms}"'
        if question:
            header += f' | Question: "{question}"'
        self.transcript.add_system_event(header)

        # 1. Find matching pages — split on spaces/commas/pipes, match ANY term
        #    Score each page by total term occurrences, take top 5 by score.
        terms = [t.strip().lower() for t in re.split(r'[,|\s]+', search_terms) if len(t.strip()) >= 2]
        if not terms:
            terms = [search_terms.lower()]
        scored: list[tuple[int, int, str]] = []  # (score, page_num, page_text)
        for i, page_text in enumerate(self.pages):
            page_lower = page_text.lower()
            score = sum(page_lower.count(term) for term in terms)
            if score > 0:
                scored.append((score, i + 1, page_text))
        scored.sort(reverse=True)  # highest score first for selection
        top = scored[:5]
        top.reverse()  # chain lowest-score first so best page is summarized last
        matching_pages = [(page_num, text) for _, page_num, text in top]
        page_scores = {page_num: score for score, page_num, _ in top}

        if not matching_pages:
            self.transcript.add_system_event("No matches found.")
            return f'No matches found for "{search_terms}".'

        page_list = ", ".join(
            f"p{p} ({page_scores[p]} hits)" for p, _ in matching_pages
        )
        self.transcript.add_system_event(f"Found matches: {page_list}")

        # 2. If no question, return snippets (backward compatible)
        if not question:
            snippets = []
            for page_num, page_text in matching_pages:
                page_lower = page_text.lower()
                pos = -1
                for term in terms:
                    pos = page_lower.find(term)
                    if pos != -1:
                        break
                if pos == -1:
                    pos = 0
                start = max(0, pos - 75)
                end = min(len(page_text), pos + len(search_terms) + 75)
                snippet = page_text[start:end].replace("\n", " ").strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(page_text):
                    snippet = snippet + "..."
                snippets.append(f"Page {page_num}: {snippet}")
            return "Search results:\n" + "\n\n".join(snippets)

        # 3. Build page windows (±1 neighbors) and chain summaries.
        #    Skip a hit only if its window adds zero new pages.
        #    E.g. hits on pages 2,3,13: page 2 reads {1,2,3}, page 3's
        #    window {2,3,4} adds page 4 so it's NOT skipped.
        num_pages = len(self.pages)
        running_summary = ""
        pages_read: set[int] = set()
        summarized_pages: list[int] = []
        for page_num, _page_text in matching_pages:
            win_lo = max(1, page_num - 1)
            win_hi = min(num_pages, page_num + 1)
            window_pages = set(range(win_lo, win_hi + 1))
            if window_pages.issubset(pages_read):
                continue  # this window adds nothing new
            pages_read.update(window_pages)
            self._tick("DM", f"sum:p{page_num}")
            window = self._get_page_window(page_num)
            running_summary = self._summarize_page_window(
                window, question, prior_summary=running_summary,
            )
            summarized_pages.append(page_num)
            # Log each intermediate summary for debugging
            self.transcript.add_system_event(f"After page {page_num} ({win_lo}-{win_hi}):")
            self.transcript.add_system_event(running_summary)

        # Final label
        if len(summarized_pages) > 1:
            pages_label = ", ".join(str(p) for p in summarized_pages)
            self.transcript.add_system_event(f"Final summary (pages {pages_label}):")
            self.transcript.add_system_event(running_summary)
        return running_summary

    def _handle_read_page(self, args: dict) -> str:
        page_number = args.get("page_number", 0)
        question = args.get("question", "")

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
        self._turns_without_module_ref = 0
        self._tick("DM", f"p{page_number}")

        page_text = self.pages[page_number - 1]

        # If question provided, summarize with ±1 page window
        if question:
            num_pages = len(self.pages)
            win_lo = max(1, page_number - 1)
            win_hi = min(num_pages, page_number + 1)
            self._tick("DM", f"sum:p{page_number}")
            window = self._get_page_window(page_number)
            summary = self._summarize_page_window(window, question)
            self.transcript.add_system_event(
                f'DM reads page {page_number} ({win_lo}-{win_hi}) | Question: "{question}"'
            )
            self.transcript.add_system_event(f"Page {page_number} ({win_lo}-{win_hi}):")
            self.transcript.add_system_event(summary)
            return f"--- Page {page_number} (summary) ---\n{summary}"

        self.transcript.add_system_event(f"DM reads page {page_number}")
        return f"--- Page {page_number} ---\n{page_text}"

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
