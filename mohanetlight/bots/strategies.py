"""Parametrized heuristic bots for league training opponents.

Each bot is a :class:`PlayerSlot` subclass that implements a distinct
Clash Royale Arena-1 strategy.  They are used as training sparring partners
for MohaNetLight via the cr-gym league system.

Available strategies
--------------------
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
# Convenience: create the full bot roster
# ═══════════════════════════════════════════════════════════════════════════════


def default_bot_roster(base_seed: int = 42) -> List[PlayerSlot]:
    """Create a default roster of diverse heuristic bots.

    Returns a list of 5 bots with different strategies and parameter
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
