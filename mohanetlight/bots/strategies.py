"""Parametrized heuristic bots for league training opponents.

Each bot is a :class:`PlayerSlot` subclass that implements a distinct
Clash Royale Arena-1 strategy.  They are used as training sparring partners
for MohaNetLight via the cr-gym league system.

Available strategies
--------------------
- **PassiveBot** — very easy: high elixir threshold, no defence, slow single-lane pushes.
- **GiantPushBot** — slow push: save elixir, Giant in back + ranged support.
- **BridgeSpamBot** — immediate aggression with cheap/fast troops at the bridge.
- **SpellCycleBot** — defensive troops + spell chip damage on towers.
- **DefensiveCounterBot** — reactive: waits for enemy attack, counters, counter-pushes.
- **BalancedBot** — adapts between aggression and defence based on elixir/HP.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from clash_royale_engine.core.state import State, UnitDetection
from clash_royale_engine.utils.constants import (
    BRIDGE_Y,
    CARD_STATS,
    LANE_DIVIDER_X,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
    RIVER_Y_MAX,
)

from clash_royale_gymnasium.league.player_slot import PlayerSlot


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _infer_player_id(state: State) -> int:
    """Infer whether we are player 0 (bottom) or player 1 (top).

    Uses the ally king-tower position.  P0's king is near y ≈ 0,
    P1's near y ≈ 31.  Falls back to 0 if detection fails.
    """
    mid = N_HEIGHT_TILES // 2  # 16
    for ally in state.allies:
        if ally.unit.category == "building" and "king" in ally.unit.name:
            return 0 if ally.position.tile_y < mid else 1
    for ally in state.allies:
        if ally.unit.category == "building":
            return 0 if ally.position.tile_y < mid else 1
    return 0


def _abs_to_rel_y(tile_y: int, player_id: int) -> int:
    """Convert absolute tile-y to player-relative coordinate.

    Player 0: relative == absolute (own side at bottom).
    Player 1: relative = 31 - absolute  (own side at top, flipped).
    """
    if player_id == 0:
        return tile_y
    return N_HEIGHT_TILES - 1 - tile_y


def _card_name(state: State, idx: int) -> str:
    """Card name at hand index ``idx``."""
    return state.cards[idx].name


def _is_spell(state: State, idx: int) -> bool:
    return state.cards[idx].is_spell


def _card_cost(state: State, idx: int) -> int:
    return state.cards[idx].cost


def _find_card_idx(state: State, names: List[str]) -> Optional[int]:
    """Return the first affordable hand index whose card name is in ``names``."""
    for idx in state.ready:
        if _card_name(state, idx) in names:
            return idx
    return None


def _enemies_on_our_side(state: State, player_id: int = 0) -> List[UnitDetection]:
    """Enemy units that have crossed into our half of the arena.

    Positions in State are *absolute*.  P0's side is y < BRIDGE_Y;
    P1's side is y > RIVER_Y_MAX.
    """
    if player_id == 0:
        return [d for d in state.enemies if d.position.tile_y < BRIDGE_Y]
    return [d for d in state.enemies if d.position.tile_y > RIVER_Y_MAX]


def _weakest_enemy_lane(state: State) -> str:
    """Return ``"left"`` or ``"right"`` based on lowest enemy princess HP."""
    left_hp = state.numbers.left_enemy_princess_hp
    right_hp = state.numbers.right_enemy_princess_hp
    if left_hp == right_hp:
        return random.choice(["left", "right"])
    return "left" if left_hp < right_hp else "right"


def _lane_x(lane: str, rng: random.Random) -> int:
    """Random tile_x in the given lane."""
    if lane == "left":
        return rng.randint(1, LANE_DIVIDER_X - 2)
    return rng.randint(LANE_DIVIDER_X + 1, N_WIDE_TILES - 2)


def _back_y(rng: random.Random) -> int:
    """tile_y in the back of own side (defensive position)."""
    return rng.randint(2, 6)


def _bridge_y(rng: random.Random) -> int:
    """tile_y near the bridge (aggressive position)."""
    return rng.randint(BRIDGE_Y - 3, BRIDGE_Y - 1)


def _spell_target(state: State, rng: random.Random, player_id: int = 0) -> Tuple[int, int]:
    """Target coordinates for a spell — clustered enemies or a tower.

    Returns *player-relative* coords (the engine flips y for P1).
    """
    enemies = state.enemies
    if enemies:
        # Target the cluster centroid (absolute → player-relative)
        xs = [d.position.tile_x for d in enemies]
        ys = [d.position.tile_y for d in enemies]
        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))
        return cx, _abs_to_rel_y(cy, player_id)
    # Target weakest enemy tower
    lane = _weakest_enemy_lane(state)
    tx = 3 if lane == "left" else 14
    ty = N_HEIGHT_TILES - 4  # near enemy tower (player-relative)
    return tx, ty


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Passive Bot  (easiest — curriculum warm-up)
# ═══════════════════════════════════════════════════════════════════════════════


class PassiveBot(PlayerSlot):
    """Very easy opponent for early curriculum phases.

    Controlled by a **passivity** parameter (0–1) that determines how
    often the bot simply does nothing even when it *could* play:

    - ``passivity=0.95`` → 95 % of eligible frames are skipped (very easy).
    - ``passivity=0.80`` → 80 % skipped (slightly harder).

    When it *does* act:

    - **Weak defence** (rare): drops archers or skeletons near threats.
      Never uses knight / mini-pekka / giant to defend.
    - **Slow attack** (very rare): plays knight far back in a single lane.
      Never plays expensive or high-impact troops.
    - **Never plays spells, giant, musketeer, or mini-pekka.**

    The bot is designed to *lose* consistently so the RL agent receives
    positive DamageComponent / TerminalComponent signals from the start.

    Parameters
    ----------
    name : str
        Display name.
    elixir_threshold : float
        Minimum elixir before the bot even *considers* acting.
    passivity : float
        Probability of doing nothing on each eligible frame (0-1).
        Higher = easier opponent.
    defend_chance : float
        Probability of defending (vs ignoring) when threats exist AND the
        passivity roll succeeds.  Default 0.3 — usually ignores threats.
    attack_chance : float
        Probability of attacking when no threats and passivity roll succeeds.
        Default 0.15 — very rarely pushes.
    seed : int
        Random seed.
    """

    DEFEND_TROOPS = ["skeletons", "archers"]
    ATTACK_TROOPS = ["knight"]

    def __init__(
        self,
        name: str = "Passive",
        elixir_threshold: float = 8.5,
        passivity: float = 0.92,
        defend_chance: float = 0.30,
        attack_chance: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__(name)
        self.elixir_threshold = elixir_threshold
        self.passivity = passivity
        self.defend_chance = defend_chance
        self.attack_chance = attack_chance
        self._seed = seed
        self._rng = random.Random(seed)
        self._lane: Optional[str] = None

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        elixir = state.numbers.elixir
        if not state.ready:
            return None

        # Gate 1: need enough elixir to even think about playing
        if elixir < self.elixir_threshold:
            return None

        # Gate 2: passivity roll — most of the time just do nothing
        if self._rng.random() < self.passivity:
            return None

        pid = _infer_player_id(state)
        threats = _enemies_on_our_side(state, pid)

        # Weak defence — only archers / skeletons, and only sometimes
        if threats and self._rng.random() < self.defend_chance:
            defend_idx = _find_card_idx(state, self.DEFEND_TROOPS)
            if defend_idx is not None:
                nearest = min(threats, key=lambda d: d.hp)
                tx = int(nearest.position.tile_x)
                rel_y = _abs_to_rel_y(int(nearest.position.tile_y), pid)
                ty = max(1, rel_y - 1)
                return (tx, ty, defend_idx)

        # Very rare attack — only knight, placed far back
        if self._rng.random() < self.attack_chance:
            if self._lane is None:
                self._lane = self._rng.choice(["left", "right"])
            lx = _lane_x(self._lane, self._rng)
            attack_idx = _find_card_idx(state, self.ATTACK_TROOPS)
            if attack_idx is not None:
                return (lx, _back_y(self._rng), attack_idx)

        # Skip spells / expensive troops entirely
        return None

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._lane = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "passive",
            "threshold": self.elixir_threshold,
            "passivity": self.passivity,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Giant Push Bot
# ═══════════════════════════════════════════════════════════════════════════════


class GiantPushBot(PlayerSlot):
    """Build a slow push: Giant in back, then ranged support behind it.

    Parameters
    ----------
    name : str
        Display name.
    elixir_threshold : float
        Wait until this much elixir before starting a push (default 7).
    seed : int
        Random seed.
    """

    TANKS = ["giant"]
    SUPPORT = ["musketeer", "archers"]
    MELEE = ["knight", "mini_pekka"]

    def __init__(
        self, name: str = "GiantPush", elixir_threshold: float = 7.0, seed: int = 42,
    ) -> None:
        super().__init__(name)
        self.elixir_threshold = elixir_threshold
        self._seed = seed
        self._rng = random.Random(seed)
        self._push_lane: Optional[str] = None

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        elixir = state.numbers.elixir
        if not state.ready:
            return None

        pid = _infer_player_id(state)

        # If enemies on our side, defend first
        threats = _enemies_on_our_side(state, pid)
        if threats and elixir >= 3:
            # Use cheapest troop to defend
            defend_idx = min(
                (i for i in state.ready if not _is_spell(state, i)),
                key=lambda i: _card_cost(state, i),
                default=None,
            )
            if defend_idx is not None:
                tx = int(threats[0].position.tile_x)
                rel_y = _abs_to_rel_y(int(threats[0].position.tile_y), pid)
                ty = max(1, rel_y - 2)
                return (tx, ty, defend_idx)

        # Wait for elixir
        if elixir < self.elixir_threshold:
            return None

        # Pick push lane
        if self._push_lane is None:
            self._push_lane = _weakest_enemy_lane(state)

        lx = _lane_x(self._push_lane, self._rng)

        # Priority: Giant first, then support, then melee
        tank_idx = _find_card_idx(state, self.TANKS)
        if tank_idx is not None:
            return (lx, _back_y(self._rng), tank_idx)

        support_idx = _find_card_idx(state, self.SUPPORT)
        if support_idx is not None:
            return (lx, _back_y(self._rng), support_idx)

        melee_idx = _find_card_idx(state, self.MELEE)
        if melee_idx is not None:
            return (lx, _bridge_y(self._rng), melee_idx)

        # Use spells on clustered enemies
        spell_idx = _find_card_idx(state, ["fireball", "arrows"])
        if spell_idx is not None and state.enemies:
            sx, sy = _spell_target(state, self._rng, pid)
            return (sx, sy, spell_idx)

        # Play cheapest available
        cheapest = min(state.ready, key=lambda i: _card_cost(state, i))
        return (lx, _back_y(self._rng), cheapest)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._push_lane = None

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "type": "giant_push", "threshold": self.elixir_threshold}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Bridge Spam Bot
# ═══════════════════════════════════════════════════════════════════════════════


class BridgeSpamBot(PlayerSlot):
    """Constant pressure — play fast/cheap troops at the bridge immediately.

    Parameters
    ----------
    name : str
        Display name.
    elixir_threshold : float
        Minimum elixir to play (default 4 — aggressive).
    seed : int
        Random seed.
    """

    FAST_TROOPS = ["mini_pekka", "knight", "skeletons", "archers"]

    def __init__(
        self, name: str = "BridgeSpam", elixir_threshold: float = 4.0, seed: int = 42,
    ) -> None:
        super().__init__(name)
        self.elixir_threshold = elixir_threshold
        self._seed = seed
        self._rng = random.Random(seed)

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        elixir = state.numbers.elixir
        if elixir < self.elixir_threshold or not state.ready:
            return None

        pid = _infer_player_id(state)

        lane = self._rng.choice(["left", "right"])
        lx = _lane_x(lane, self._rng)

        # Prefer fast troops at the bridge
        fast_idx = _find_card_idx(state, self.FAST_TROOPS)
        if fast_idx is not None:
            return (lx, _bridge_y(self._rng), fast_idx)

        # Giant as last resort
        tank_idx = _find_card_idx(state, ["giant"])
        if tank_idx is not None:
            return (lx, _bridge_y(self._rng), tank_idx)

        # Spell on enemy tower
        spell_idx = _find_card_idx(state, ["fireball", "arrows"])
        if spell_idx is not None:
            sx, sy = _spell_target(state, self._rng, pid)
            return (sx, sy, spell_idx)

        # Anything
        idx = self._rng.choice(state.ready)
        return (lx, _bridge_y(self._rng), idx)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "type": "bridge_spam", "threshold": self.elixir_threshold}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Spell Cycle Bot
# ═══════════════════════════════════════════════════════════════════════════════


class SpellCycleBot(PlayerSlot):
    """Defend with cheap troops, chip enemy towers with spells.

    Parameters
    ----------
    name : str
        Display name.
    spell_threshold : float
        Elixir to hold in reserve before casting spell (default 5).
    seed : int
        Random seed.
    """

    CHEAP_TROOPS = ["skeletons", "archers", "knight"]
    SPELLS = ["fireball", "arrows"]

    def __init__(
        self, name: str = "SpellCycle", spell_threshold: float = 5.0, seed: int = 42,
    ) -> None:
        super().__init__(name)
        self.spell_threshold = spell_threshold
        self._seed = seed
        self._rng = random.Random(seed)

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        elixir = state.numbers.elixir
        if not state.ready:
            return None

        pid = _infer_player_id(state)

        # Priority 1: defend against threats on our side
        threats = _enemies_on_our_side(state, pid)
        if threats and elixir >= 3:
            cheap = _find_card_idx(state, self.CHEAP_TROOPS)
            if cheap is not None:
                tx = int(threats[0].position.tile_x)
                rel_y = _abs_to_rel_y(int(threats[0].position.tile_y), pid)
                ty = max(1, rel_y - 1)
                return (tx, ty, cheap)

        # Priority 2: cast spell on enemy tower for chip damage
        if elixir >= self.spell_threshold:
            spell_idx = _find_card_idx(state, self.SPELLS)
            if spell_idx is not None:
                # Aim at weakest tower
                lane = _weakest_enemy_lane(state)
                tx = 3 if lane == "left" else 14
                ty = N_HEIGHT_TILES - 4
                return (tx, ty, spell_idx)

        # Priority 3: cycle cheap troops defensively
        if elixir >= 6:
            cheap = _find_card_idx(state, self.CHEAP_TROOPS)
            if cheap is not None:
                lx = _lane_x(
                    _weakest_enemy_lane(state), self._rng,
                )
                return (lx, _back_y(self._rng), cheap)

        return None

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "type": "spell_cycle", "threshold": self.spell_threshold}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Defensive Counter Bot
# ═══════════════════════════════════════════════════════════════════════════════


class DefensiveCounterBot(PlayerSlot):
    """Wait for enemy attacks, counter efficiently, then counter-push.

    Parameters
    ----------
    name : str
        Display name.
    counter_elixir : float
        Minimum elixir to start counter-pushing (default 6).
    seed : int
        Random seed.
    """

    COUNTER_TROOPS = ["mini_pekka", "musketeer", "archers", "knight"]
    CHEAP_CYCLE = ["skeletons"]

    def __init__(
        self, name: str = "DefCounter", counter_elixir: float = 6.0, seed: int = 42,
    ) -> None:
        super().__init__(name)
        self.counter_elixir = counter_elixir
        self._seed = seed
        self._rng = random.Random(seed)

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        elixir = state.numbers.elixir
        if not state.ready:
            return None

        pid = _infer_player_id(state)
        threats = _enemies_on_our_side(state, pid)

        # Active defence: place counter troops on top of threats
        if threats:
            # Find strongest threat (highest HP)
            biggest = max(threats, key=lambda d: d.hp)
            bx = int(biggest.position.tile_x)
            rel_y = _abs_to_rel_y(int(biggest.position.tile_y), pid)
            by = max(1, rel_y - 2)

            # Use high-DPS cards against tanks
            stats = CARD_STATS.get(biggest.unit.name, {})
            is_tank = stats.get("hp", 0) > 500

            if is_tank:
                killer = _find_card_idx(state, ["mini_pekka", "musketeer"])
            else:
                killer = _find_card_idx(state, ["archers", "skeletons", "knight"])

            if killer is not None and elixir >= _card_cost(state, killer):
                return (bx, by, killer)

            # Spell on cluster
            if len(threats) >= 2:
                spell = _find_card_idx(state, ["fireball", "arrows"])
                if spell is not None:
                    sx, sy = _spell_target(state, self._rng, pid)
                    return (int(sx), int(sy), spell)

        # No threats — counter-push if we have elixir advantage
        if elixir >= self.counter_elixir and not threats:
            lane = _weakest_enemy_lane(state)
            lx = _lane_x(lane, self._rng)

            # Play a strong troop at bridge
            push = _find_card_idx(state, self.COUNTER_TROOPS)
            if push is not None:
                return (lx, _bridge_y(self._rng), push)

        # High elixir: cycle cheap cards to avoid waste
        if elixir >= 9.5:
            cheap = _find_card_idx(state, self.CHEAP_CYCLE)
            if cheap is not None:
                lx = _lane_x(
                    _weakest_enemy_lane(state), self._rng,
                )
                return (lx, _back_y(self._rng), cheap)

        return None

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "type": "defensive_counter", "elixir": self.counter_elixir}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Balanced Bot
# ═══════════════════════════════════════════════════════════════════════════════


class BalancedBot(PlayerSlot):
    """Adapts between aggression and defence based on game state.

    - Aggressive when elixir-advantaged or enemy tower is low.
    - Defensive when behind or facing a push.

    Parameters
    ----------
    name : str
        Display name.
    base_threshold : float
        Base elixir threshold (adjusted dynamically).
    seed : int
        Random seed.
    """

    def __init__(
        self, name: str = "Balanced", base_threshold: float = 5.0, seed: int = 42,
    ) -> None:
        super().__init__(name)
        self.base_threshold = base_threshold
        self._seed = seed
        self._rng = random.Random(seed)

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        elixir = state.numbers.elixir
        if not state.ready:
            return None

        pid = _infer_player_id(state)
        threats = _enemies_on_our_side(state, pid)
        n = state.numbers

        # Dynamic threshold: lower when we have HP advantage, higher when losing
        own_hp = n.left_princess_hp + n.right_princess_hp + n.king_hp
        enemy_hp = n.left_enemy_princess_hp + n.right_enemy_princess_hp + n.enemy_king_hp
        hp_ratio = own_hp / max(enemy_hp, 1.0)

        threshold = self.base_threshold
        if hp_ratio > 1.3:
            threshold -= 1.5  # we're ahead, be aggressive
        elif hp_ratio < 0.7:
            threshold += 1.5  # we're behind, be careful

        # Always defend first
        if threats:
            cheapest_troop = min(
                (i for i in state.ready if not _is_spell(state, i)),
                key=lambda i: _card_cost(state, i),
                default=None,
            )
            if cheapest_troop is not None:
                biggest = max(threats, key=lambda d: d.hp)
                bx = int(biggest.position.tile_x)
                rel_y = _abs_to_rel_y(int(biggest.position.tile_y), pid)
                by = max(1, rel_y - 2)
                return (bx, by, cheapest_troop)

            # Spell if multiple enemies
            if len(threats) >= 2:
                spell = _find_card_idx(state, ["fireball", "arrows"])
                if spell is not None:
                    sx, sy = _spell_target(state, self._rng, pid)
                    return (int(sx), int(sy), spell)

        if elixir < threshold:
            return None

        # Attack mode
        lane = _weakest_enemy_lane(state)
        lx = _lane_x(lane, self._rng)

        # Mix push & bridge depending on situation
        if hp_ratio > 1.2:
            # Ahead: bridge spam for quick finish
            fast = _find_card_idx(state, ["mini_pekka", "knight", "archers"])
            if fast is not None:
                return (lx, _bridge_y(self._rng), fast)
        else:
            # Even/behind: build a push from the back
            tank = _find_card_idx(state, ["giant"])
            if tank is not None:
                return (lx, _back_y(self._rng), tank)
            support = _find_card_idx(state, ["musketeer", "archers"])
            if support is not None:
                return (lx, _back_y(self._rng), support)

        # Fallback: cheapest available card
        cheapest = min(state.ready, key=lambda i: _card_cost(state, i))
        return (lx, self._rng.randint(4, BRIDGE_Y - 2), cheapest)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "type": "balanced", "threshold": self.base_threshold}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Optimal Bot — the hardest heuristic opponent
# ═══════════════════════════════════════════════════════════════════════════════


class OptimalBot(PlayerSlot):
    """Expert-level heuristic that plays near-optimal Arena 1 strategy.

    **Defence (elixir-efficient):**
    - Counter knight/mini_pekka cheaply (skeletons on mini_pekka to surround,
      archers on knight from range).
    - *Ignore* lone archers/skeletons — let the tower handle them.
    - Against a giant push:
        1. Arrows on small troops behind the giant.
        2. Place knight/archers to kill ranged support.
        3. Mini-pekka on the giant itself (high DPS tank killer).

    **Offence (giant push + dual-lane pressure):**
    - Build a giant push from the back: Giant → musketeer/archers placed
      far behind so they walk behind it.
    - If elixir ≥ ``dual_lane_threshold``: dual-lane attack —
      giant push in one lane, mini_pekka bridge rush in the other.
    - Use fireball/arrows to clear distractions blocking the push.

    Parameters
    ----------
    name : str
        Display name.
    push_threshold : float
        Elixir to start building a giant push (default 7).
    dual_lane_threshold : float
        Elixir to trigger a dual-lane split push (default 9).
    aggression : float
        0–1 multiplier controlling how eagerly it attacks.
        1.0 = full aggression (default), 0.5 = more conservative.
    seed : int
        Random seed.
    """

    # Troops that are worth countering (expensive / dangerous)
    WORTH_COUNTERING = {"knight", "mini_pekka", "giant", "musketeer"}
    # Cheap trash — let the tower tank them
    IGNORABLE = {"skeletons", "archers"}

    def __init__(
        self,
        name: str = "Optimal",
        push_threshold: float = 7.0,
        dual_lane_threshold: float = 9.0,
        aggression: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(name)
        self.push_threshold = push_threshold
        self.dual_lane_threshold = dual_lane_threshold
        self.aggression = aggression
        self._seed = seed
        self._rng = random.Random(seed)
        self._push_lane: Optional[str] = None
        self._giant_placed = False

    # ── Helpers ───────────────────────────────────────────────────────────

    def _classify_threats(
        self, threats: List[UnitDetection],
    ) -> Tuple[List[UnitDetection], List[UnitDetection], Optional[UnitDetection]]:
        """Split threats into (dangerous, ignorable, giant_if_any)."""
        dangerous: List[UnitDetection] = []
        ignorable: List[UnitDetection] = []
        giant: Optional[UnitDetection] = None
        for t in threats:
            name = t.unit.name
            if name == "giant":
                giant = t
                dangerous.append(t)
            elif name in self.WORTH_COUNTERING:
                dangerous.append(t)
            else:
                ignorable.append(t)
        return dangerous, ignorable, giant

    def _has_support_behind_giant(
        self, giant: UnitDetection, threats: List[UnitDetection], pid: int,
    ) -> List[UnitDetection]:
        """Find enemy support troops walking behind a giant."""
        supports: List[UnitDetection] = []
        gx, gy = giant.position.tile_x, giant.position.tile_y
        for t in threats:
            if t is giant:
                continue
            tx, ty = t.position.tile_x, t.position.tile_y
            # "Behind" the giant means further from our side
            if pid == 0 and ty > gy:  # P0: behind = higher y
                if abs(tx - gx) <= 4:
                    supports.append(t)
            elif pid == 1 and ty < gy:  # P1: behind = lower y
                if abs(tx - gx) <= 4:
                    supports.append(t)
        return supports

    def _other_lane(self, lane: str) -> str:
        return "right" if lane == "left" else "left"

    # ── Main decision ─────────────────────────────────────────────────────

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        elixir = state.numbers.elixir
        if not state.ready:
            return None

        pid = _infer_player_id(state)
        threats = _enemies_on_our_side(state, pid)

        # ── DEFENCE ──────────────────────────────────────────────────────
        if threats:
            dangerous, ignorable, giant = self._classify_threats(threats)

            # 1) Enemy giant push — optimal counter sequence
            if giant is not None:
                return self._counter_giant_push(state, giant, threats, pid, elixir)

            # 2) Counter dangerous troops efficiently
            if dangerous:
                return self._counter_dangerous(state, dangerous, pid, elixir)

            # 3) Ignorable troops only → do nothing, tower handles them
            # (fall through to offence if we have elixir)

        # ── OFFENCE ──────────────────────────────────────────────────────
        return self._build_offence(state, pid, elixir)

    # ── Defence sub-routines ──────────────────────────────────────────────

    def _counter_giant_push(
        self,
        state: State,
        giant: UnitDetection,
        all_threats: List[UnitDetection],
        pid: int,
        elixir: float,
    ) -> Optional[Tuple[int, int, int]]:
        """Counter an enemy giant push optimally.

        Priority:
        1. Arrows on small troops behind the giant (if any).
        2. Knight/archers to kill ranged support behind giant.
        3. Mini-pekka on the giant (tank killer).
        """
        gx = int(giant.position.tile_x)
        g_rel_y = _abs_to_rel_y(int(giant.position.tile_y), pid)

        supports = self._has_support_behind_giant(giant, all_threats, pid)

        # 1) Arrows on small troops / cluster behind the giant
        small_supports = [s for s in supports if s.unit.name in ("archers", "skeletons")]
        if small_supports and elixir >= 3:
            arrows_idx = _find_card_idx(state, ["arrows"])
            if arrows_idx is not None:
                # Target centroid of small troops
                sx = int(sum(s.position.tile_x for s in small_supports) / len(small_supports))
                sy = int(sum(s.position.tile_y for s in small_supports) / len(small_supports))
                return (sx, _abs_to_rel_y(sy, pid), arrows_idx)

        # 2) Place a troop to deal with ranged support (musketeer, archers)
        ranged_supports = [s for s in supports if s.unit.name in ("musketeer", "archers")]
        if ranged_supports and elixir >= 3:
            counter_idx = _find_card_idx(state, ["knight", "archers", "skeletons"])
            if counter_idx is not None:
                target = ranged_supports[0]
                tx = int(target.position.tile_x)
                ty = _abs_to_rel_y(int(target.position.tile_y), pid)
                return (tx, max(1, ty - 1), counter_idx)

        # 3) Mini-pekka on the giant itself (high DPS)
        if elixir >= 4:
            mp_idx = _find_card_idx(state, ["mini_pekka"])
            if mp_idx is not None:
                # Place slightly in front of giant so it walks into mini_pekka
                return (gx, max(1, g_rel_y - 1), mp_idx)

        # 4) Fireball the whole push as fallback
        if elixir >= 4 and len(supports) >= 1:
            fb_idx = _find_card_idx(state, ["fireball"])
            if fb_idx is not None:
                sx, sy = _spell_target(state, self._rng, pid)
                return (int(sx), int(sy), fb_idx)

        # 5) Any troop we can afford near the giant
        any_troop = min(
            (i for i in state.ready if not _is_spell(state, i)),
            key=lambda i: _card_cost(state, i),
            default=None,
        )
        if any_troop is not None and elixir >= _card_cost(state, any_troop):
            return (gx, max(1, g_rel_y - 1), any_troop)

        return None

    def _counter_dangerous(
        self,
        state: State,
        dangerous: List[UnitDetection],
        pid: int,
        elixir: float,
    ) -> Optional[Tuple[int, int, int]]:
        """Counter individual dangerous troops efficiently."""
        target = max(dangerous, key=lambda d: d.hp)
        tx = int(target.position.tile_x)
        t_rel_y = _abs_to_rel_y(int(target.position.tile_y), pid)

        name = target.unit.name

        # Mini-pekka → surround with skeletons (1 elixir vs 4)
        if name == "mini_pekka" and elixir >= 1:
            counter = _find_card_idx(state, ["skeletons"])
            if counter is not None:
                return (tx, max(1, t_rel_y), counter)

        # Knight → archers from range (3 vs 3, but archers survive)
        if name == "knight" and elixir >= 3:
            counter = _find_card_idx(state, ["archers", "musketeer"])
            if counter is not None:
                return (tx, max(1, t_rel_y - 2), counter)

        # Musketeer → knight to tank + close gap (3 vs 4)
        if name == "musketeer" and elixir >= 3:
            counter = _find_card_idx(state, ["knight", "mini_pekka"])
            if counter is not None:
                return (tx, max(1, t_rel_y - 1), counter)

        # Generic: cheapest troop
        cheapest = min(
            (i for i in state.ready if not _is_spell(state, i)),
            key=lambda i: _card_cost(state, i),
            default=None,
        )
        if cheapest is not None and elixir >= _card_cost(state, cheapest):
            return (tx, max(1, t_rel_y - 1), cheapest)

        return None

    # ── Offence sub-routines ──────────────────────────────────────────────

    def _build_offence(
        self, state: State, pid: int, elixir: float,
    ) -> Optional[Tuple[int, int, int]]:
        """Build a giant push or dual-lane split attack."""
        if self._push_lane is None:
            self._push_lane = _weakest_enemy_lane(state)

        lx = _lane_x(self._push_lane, self._rng)

        # ── Dual-lane split push ─────────────────────────────────────────
        if elixir >= self.dual_lane_threshold * self.aggression:
            # If giant already placed in push lane, send mini_pekka other lane
            if self._giant_placed:
                other_lx = _lane_x(self._other_lane(self._push_lane), self._rng)
                mp_idx = _find_card_idx(state, ["mini_pekka", "knight"])
                if mp_idx is not None:
                    return (other_lx, _bridge_y(self._rng), mp_idx)

            # If allies on field (giant walking), place support behind
            ally_giant = any(
                a.unit.name == "giant" and a.unit.category != "building"
                for a in state.allies
            )
            if ally_giant:
                self._giant_placed = True
                # Add support behind the giant
                support_idx = _find_card_idx(state, ["musketeer", "archers"])
                if support_idx is not None:
                    return (lx, _back_y(self._rng), support_idx)
                # Or clear distractions with spell
                if state.enemies:
                    spell_idx = _find_card_idx(state, ["fireball", "arrows"])
                    if spell_idx is not None:
                        sx, sy = _spell_target(state, self._rng, pid)
                        return (int(sx), int(sy), spell_idx)

        # ── Standard giant push from back ────────────────────────────────
        if elixir >= self.push_threshold * self.aggression:
            # Step 1: Giant in back
            giant_idx = _find_card_idx(state, ["giant"])
            if giant_idx is not None:
                self._giant_placed = False  # will be set when we see it on field
                return (lx, _back_y(self._rng), giant_idx)

            # Step 2: Support behind existing push
            support_idx = _find_card_idx(state, ["musketeer", "archers"])
            if support_idx is not None:
                return (lx, _back_y(self._rng), support_idx)

            # Step 3: Melee at bridge
            melee_idx = _find_card_idx(state, ["mini_pekka", "knight"])
            if melee_idx is not None:
                return (lx, _bridge_y(self._rng), melee_idx)

        # ── High elixir dump — don't leak ────────────────────────────────
        if elixir >= 9.5:
            cheapest = min(state.ready, key=lambda i: _card_cost(state, i))
            return (lx, _back_y(self._rng), cheapest)

        return None

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._push_lane = None
        self._giant_placed = False

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "optimal",
            "push_threshold": self.push_threshold,
            "dual_lane_threshold": self.dual_lane_threshold,
            "aggression": self.aggression,
        }


def default_bot_roster(base_seed: int = 42) -> List[PlayerSlot]:
    """Create a default roster of diverse heuristic bots.

    Returns a list of 10 bots with different strategies and parameter
    variations, suitable for a league training curriculum.
    """
    return [
        GiantPushBot("GiantPush-7", elixir_threshold=7.0, seed=base_seed),
        GiantPushBot("GiantPush-5", elixir_threshold=5.0, seed=base_seed + 1),
        BridgeSpamBot("BridgeSpam-4", elixir_threshold=4.0, seed=base_seed + 2),
        BridgeSpamBot("BridgeSpam-3", elixir_threshold=3.0, seed=base_seed + 3),
        SpellCycleBot("SpellCycle-5", spell_threshold=5.0, seed=base_seed + 4),
        SpellCycleBot("SpellCycle-7", spell_threshold=7.0, seed=base_seed + 5),
        DefensiveCounterBot("DefCounter-6", counter_elixir=6.0, seed=base_seed + 6),
        DefensiveCounterBot("DefCounter-8", counter_elixir=8.0, seed=base_seed + 7),
        BalancedBot("Balanced-5", base_threshold=5.0, seed=base_seed + 8),
        BalancedBot("Balanced-3", base_threshold=3.0, seed=base_seed + 9),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Tiered bot rosters for progressive curriculum
# ═══════════════════════════════════════════════════════════════════════════════


def easy_bots(base_seed: int = 42) -> List[PlayerSlot]:
    """Tier 1 — Balanced bots + cautious GiantPush (already beatable)."""
    return [
        BalancedBot("Balanced-7", base_threshold=7.0, seed=base_seed),
        BalancedBot("Balanced-5", base_threshold=5.0, seed=base_seed + 1),
        GiantPushBot("GiantPush-7", elixir_threshold=7.0, seed=base_seed + 2),
    ]


def medium_bots(base_seed: int = 42) -> List[PlayerSlot]:
    """Tier 2 — Learn to counter pushes & face aggression."""
    return [
        GiantPushBot("GiantPush-5", elixir_threshold=5.0, seed=base_seed + 10),
        BalancedBot("Balanced-3", base_threshold=3.0, seed=base_seed + 11),
        BridgeSpamBot("BridgeSpam-4", elixir_threshold=4.0, seed=base_seed + 12),
    ]


def hard_bots(base_seed: int = 42) -> List[PlayerSlot]:
    """Tier 3 — Diverse aggressive strategies that punish mistakes."""
    return [
        BridgeSpamBot("BridgeSpam-3", elixir_threshold=3.0, seed=base_seed + 20),
        GiantPushBot("GiantPush-4", elixir_threshold=4.0, seed=base_seed + 21),
        DefensiveCounterBot("DefCounter-5", counter_elixir=5.0, seed=base_seed + 22),
        SpellCycleBot("SpellCycle-5", spell_threshold=5.0, seed=base_seed + 23),
    ]


def expert_bots(base_seed: int = 42) -> List[PlayerSlot]:
    """Tier 4 — Hardest heuristic bots + conservative OptimalBot."""
    return [
        DefensiveCounterBot("DefCounter-4", counter_elixir=4.0, seed=base_seed + 30),
        BalancedBot("Balanced-2", base_threshold=2.0, seed=base_seed + 31),
        BridgeSpamBot("BridgeSpam-2", elixir_threshold=2.0, seed=base_seed + 32),
        OptimalBot("Optimal-7", push_threshold=7.0, aggression=0.7, seed=base_seed + 33),
    ]


def optimal_bots(base_seed: int = 42) -> List[PlayerSlot]:
    """Tier 5 — Full-strength OptimalBot variants — the ultimate challenge."""
    return [
        OptimalBot("Optimal-7", push_threshold=7.0, dual_lane_threshold=9.0,
                    aggression=1.0, seed=base_seed + 40),
        OptimalBot("Optimal-5", push_threshold=5.0, dual_lane_threshold=8.0,
                    aggression=1.0, seed=base_seed + 41),
    ]
