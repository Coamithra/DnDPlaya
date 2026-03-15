"""Round resolution with pressure signals for combat simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from dndplaya.mechanics.characters import Character, _compute_spell_slots
from dndplaya.mechanics.dice import DiceRoller
from dndplaya.mechanics.monsters import Monster


class PressureSignal(Enum):
    """Signals emitted during combat to indicate dramatic tension points."""

    CHARACTER_LOW_HP = "character_low_hp"
    ENEMY_BLOODIED = "enemy_bloodied"
    RESOURCE_DEPLETED = "resource_depleted"
    TPK_RISK = "tpk_risk"


@dataclass
class CombatResult:
    """The outcome of a single combat action."""

    damage_dealt: float
    target_name: str
    attacker_name: str
    is_kill: bool
    narrative_hint: str


@dataclass
class RoundResult:
    """The outcome of one full round of combat."""

    results: list[CombatResult] = field(default_factory=list)
    pressure_signals: list[PressureSignal] = field(default_factory=list)
    round_number: int = 0


class CombatResolver:
    """Resolves combat actions using a DiceRoller for randomised outcomes."""

    def __init__(self, dice: DiceRoller) -> None:
        self.dice = dice

    def resolve_attack(
        self,
        attacker: Character | Monster,
        target: Character | Monster,
    ) -> CombatResult:
        """Resolve a single attack from attacker against target.

        Uses the attacker's attack_bonus vs the target's AC, then applies
        variance_roll on the attacker's average damage.
        """
        hit, roll_total = self.dice.check(attacker.attack_bonus, target.ac)

        if not hit:
            return CombatResult(
                damage_dealt=0,
                target_name=target.name,
                attacker_name=attacker.name,
                is_kill=False,
                narrative_hint=f"{attacker.name} misses {target.name} (rolled {roll_total} vs AC {target.ac})",
            )

        # Determine average damage source
        if isinstance(attacker, Character):
            avg_dmg = attacker.avg_damage
        else:
            avg_dmg = attacker.damage_per_round

        damage = max(1.0, self.dice.variance_roll(avg_dmg))
        damage = round(damage, 1)

        remaining = target.current_hp - int(damage)
        is_kill = remaining <= 0

        hint = f"{attacker.name} hits {target.name} for {damage} damage"
        if is_kill:
            hint += f" — {target.name} is slain!"

        return CombatResult(
            damage_dealt=damage,
            target_name=target.name,
            attacker_name=attacker.name,
            is_kill=is_kill,
            narrative_hint=hint,
        )

    def resolve_heal(
        self,
        healer: Character,
        target: Character,
        heal_amount: float,
    ) -> CombatResult:
        """Resolve a healing action."""
        actual_heal = self.dice.variance_roll(heal_amount)
        actual_heal = round(actual_heal, 1)
        actual_heal = min(actual_heal, target.max_hp - target.current_hp)
        actual_heal = max(0, actual_heal)

        return CombatResult(
            damage_dealt=-actual_heal,
            target_name=target.name,
            attacker_name=healer.name,
            is_kill=False,
            narrative_hint=f"{healer.name} heals {target.name} for {actual_heal} HP",
        )

    def resolve_aoe(
        self,
        caster: Character,
        targets: list[Monster],
        damage: float,
    ) -> list[CombatResult]:
        """Resolve an area-of-effect attack against multiple targets."""
        results: list[CombatResult] = []
        for target in targets:
            rolled_damage = max(1.0, self.dice.variance_roll(damage))
            rolled_damage = round(rolled_damage, 1)

            # Saving throw: target rolls vs caster's effective save DC
            # Use a simple DC based on 8 + caster attack_bonus as a proxy
            save_dc = 8 + caster.attack_bonus
            saved, _ = self.dice.check(0, save_dc)
            if saved:
                rolled_damage /= 2  # half damage on save

            remaining = target.current_hp - int(rolled_damage)
            is_kill = remaining <= 0

            hint = f"{caster.name}'s AoE hits {target.name} for {rolled_damage} damage"
            if is_kill:
                hint += f" — {target.name} is destroyed!"

            results.append(
                CombatResult(
                    damage_dealt=rolled_damage,
                    target_name=target.name,
                    attacker_name=caster.name,
                    is_kill=is_kill,
                    narrative_hint=hint,
                )
            )
        return results

    def check_pressure_signals(
        self,
        characters: list[Character],
        monsters: list[Monster],
    ) -> list[PressureSignal]:
        """Evaluate the battlefield and return any active pressure signals."""
        signals: list[PressureSignal] = []

        # CHARACTER_LOW_HP: any character below 25% HP
        for char in characters:
            if char.current_hp > 0 and char.current_hp < char.max_hp * 0.25:
                signals.append(PressureSignal.CHARACTER_LOW_HP)
                break

        # ENEMY_BLOODIED: any monster below 50% HP
        for monster in monsters:
            if monster.current_hp > 0 and monster.current_hp < monster.max_hp * 0.5:
                signals.append(PressureSignal.ENEMY_BLOODIED)
                break

        # RESOURCE_DEPLETED: total party spell slots < 25% of max
        total_current_slots = 0
        total_max_slots = 0
        for char in characters:
            for _level, count in char.spell_slots.items():
                total_current_slots += count
            original_slots = _compute_spell_slots(char.char_class, char.level)
            for _level, count in original_slots.items():
                total_max_slots += count

        if total_max_slots > 0 and total_current_slots < total_max_slots * 0.25:
            signals.append(PressureSignal.RESOURCE_DEPLETED)

        # TPK_RISK: average party HP < 30% of max
        alive_chars = [c for c in characters if c.current_hp > 0]
        if alive_chars:
            avg_hp_pct = sum(c.current_hp / c.max_hp for c in alive_chars) / len(alive_chars)
            if avg_hp_pct < 0.30:
                signals.append(PressureSignal.TPK_RISK)
        elif characters:
            # All dead — definite TPK risk
            signals.append(PressureSignal.TPK_RISK)

        return signals
