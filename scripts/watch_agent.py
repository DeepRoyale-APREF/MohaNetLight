#!/usr/bin/env python
"""
Visual debugger — watch MohaNetLight play against a bot with real-time
reward and action overlays.

Shows the full Clash Royale GUI with a debug panel on the right displaying:
  • Per-step reward and cumulative episode reward
  • Per-component reward breakdown (colour-coded bars)
  • Agent's chosen action (card, tile position)
  • Critic value estimate V(s)
  • Tower HP deltas and elixir state

Usage
-----
    # Watch a random (untrained) agent
    python scripts/watch_agent.py

    # Watch a trained agent from checkpoint
    python scripts/watch_agent.py --checkpoint logs/mohanet/model_final.pt

    # Choose opponent
    python scripts/watch_agent.py --opponent GiantPush

    # Deterministic mode (argmax actions, no sampling)
    python scripts/watch_agent.py --checkpoint model.pt --deterministic

Press ESC or close the window to quit.

Why frame_skip instead of reducing fps?
---------------------------------------
The physics engine is calibrated for 30 fps — troop speeds, damage ticks,
projectile travel, and collision detection all assume ~33 ms per frame.
Reducing fps to 10 would make combat coarser (troops teleport larger
distances, damage is applied in bigger bursts).  ``frame_skip=3`` keeps
physics at 30 fps for accurate simulation while only asking the RL agent
for a decision every 3 frames (= 10 decisions/second).  The visual debug
script uses frame_skip=1 so you see EVERY physics frame at real speed.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure all three packages are importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# ── cr-engine ─────────────────────────────────────────────────────────────
from clash_royale_engine.visualization.renderer import (
    COL_BG,
    COL_ELIXIR_FILL,
    COL_HP_GREEN,
    COL_HP_RED,
    COL_HP_YELLOW,
    COL_TEXT,
    MARGIN_BOTTOM,
    MARGIN_SIDE,
    MARGIN_TOP,
    ARENA_VISUAL_H,
    WIN_H,
    WIN_W,
    Renderer,
    _ensure_pygame,
    _draw_text,
)

# ── cr-gym ────────────────────────────────────────────────────────────────
from clash_royale_gymnasium.env.clash_env import ClashRoyaleGymEnv

# ── MohaNetLight ──────────────────────────────────────────────────────────
from mohanetlight.config import ModelConfig
from mohanetlight.network.mohanet import MohaNetLight
from mohanetlight.utils.tensor_utils import obs_to_tensors
from mohanetlight.bots.strategies import (
    BalancedBot,
    BridgeSpamBot,
    DefensiveCounterBot,
    GiantPushBot,
    SpellCycleBot,
)

# ══════════════════════════════════════════════════════════════════════════
# Constants for the debug panel
# ══════════════════════════════════════════════════════════════════════════

DEBUG_PANEL_W = 310
EXTENDED_WIN_W = WIN_W + DEBUG_PANEL_W

# Reward component colours (match bar fills)
_COMP_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "DamageComponent": (255, 120, 40),      # orange
    "ElixirComponent": (180, 80, 220),       # purple
    "TerminalComponent": (60, 200, 60),      # green
}


# ══════════════════════════════════════════════════════════════════════════
# DebugRenderer — extends Renderer with a right-side info panel
# ══════════════════════════════════════════════════════════════════════════


class DebugRenderer(Renderer):
    """Extended renderer that shows a debug panel alongside the arena.

    The panel displays reward signals, actions, and training-relevant
    statistics in real time.  The game arena is drawn identically to the
    base :class:`Renderer`.
    """

    def __init__(
        self,
        fps: int = 30,
        title: str = "MohaNet — Visual Debugger",
        speed_multiplier: float = 1.0,
        game_duration: float = 180.0,
    ) -> None:
        super().__init__(
            fps=fps,
            title=title,
            speed_multiplier=speed_multiplier,
            game_duration=game_duration,
        )
        # Debug state (set externally before each render)
        self._debug: Dict[str, Any] = {}
        # Rolling reward history for sparkline
        self._reward_history: List[float] = []
        self._max_history = 200

    # ── Override init to widen the window ──────────────────────────────

    def _init_pygame(self) -> None:
        if self._initialised:
            return
        pg = _ensure_pygame()
        pg.init()
        pg.display.set_caption(self.title)
        # Wider window to fit the debug panel
        self._screen = pg.display.set_mode((EXTENDED_WIN_W, WIN_H))
        self._clock = pg.time.Clock()
        self._font_sm = pg.font.SysFont("consolas", 11)
        self._font_md = pg.font.SysFont("consolas", 14, bold=True)
        self._font_lg = pg.font.SysFont("consolas", 19, bold=True)
        self._font_xl = pg.font.SysFont("consolas", 27, bold=True)
        # Panel-specific fonts
        self._font_panel = pg.font.SysFont("consolas", 12)
        self._font_panel_bold = pg.font.SysFont("consolas", 12, bold=True)
        self._font_section = pg.font.SysFont("consolas", 14, bold=True)

        # Pre-load tower sprites (same as parent)
        from clash_royale_engine.visualization.renderer import (
            _TOWER_IMAGE_FILES,
            TOWER_KING_W,
            TOWER_KING_H,
            TOWER_PRINCESS_W,
            TOWER_PRINCESS_H,
            _load_image,
        )
        for key, path in _TOWER_IMAGE_FILES.items():
            size = (
                (TOWER_KING_W, TOWER_KING_H)
                if "king" in key
                else (TOWER_PRINCESS_W, TOWER_PRINCESS_H)
            )
            self._tower_sprites[key] = _load_image(path, size)

        self._init_music(pg)
        self._initialised = True

    # ── Public API ────────────────────────────────────────────────────

    def set_debug(self, **kwargs: Any) -> None:
        """Update debug data for next render frame."""
        self._debug.update(kwargs)

    def render(self, state: "State") -> bool:  # type: ignore[override]
        """Draw arena + debug panel. Returns False if window was closed."""
        self._init_pygame()
        pg = _ensure_pygame()

        if not self.poll_events():
            return False

        self._update_music(state)
        self._screen.fill(COL_BG)

        # ── Arena (same as parent) ────────────────────────────────────
        self._draw_top_hud(state)
        self._draw_bottom_hud(state)
        self._draw_arena()
        self._draw_tower_sprites(state)
        self._draw_units(state.allies, is_ally=True)
        self._draw_units(state.enemies, is_ally=False)
        self._draw_spells(state)
        self._draw_tower_hp(state)

        # ── Debug panel (right side) ──────────────────────────────────
        self._draw_debug_panel(state)

        pg.display.flip()
        self._clock.tick(self.fps)
        return True

    # ── Debug panel drawing ───────────────────────────────────────────

    def _draw_debug_panel(self, state: "State") -> None:
        pg = _ensure_pygame()
        d = self._debug
        px = WIN_W + 8  # panel x start
        py = 10         # panel y cursor

        # Panel background
        pg.draw.rect(
            self._screen, (28, 24, 20),
            pg.Rect(WIN_W, 0, DEBUG_PANEL_W, WIN_H),
        )
        # Separator line
        pg.draw.line(
            self._screen, (80, 65, 45),
            (WIN_W, 0), (WIN_W, WIN_H), 2,
        )

        # ── Section: MATCH ────────────────────────────────────────────
        py = self._section_header(px, py, "MATCH INFO")
        step = d.get("step", 0)
        time_rem = state.numbers.time_remaining
        is_ot = state.numbers.is_overtime
        ot_rem = state.numbers.overtime_remaining if is_ot else 0.0
        elapsed = 180.0 - time_rem if not is_ot else 180.0 + (60.0 - ot_rem)

        py = self._kv(px, py, "Step", f"{step}")
        py = self._kv(px, py, "Elapsed", f"{elapsed:.1f}s")
        phase = "OVERTIME" if is_ot else ("x2 ELIXIR" if state.numbers.is_double_elixir else "Normal")
        py = self._kv(px, py, "Phase", phase)
        py = self._kv(px, py, "Elixir", f"{state.numbers.elixir:.1f} / 10")

        py += 6

        # ── Section: ACTION ───────────────────────────────────────────
        py = self._section_header(px, py, "AGENT ACTION")
        card_idx = d.get("card_idx", 4)
        tile_x = d.get("tile_x", 0)
        tile_y = d.get("tile_y", 0)
        valid = d.get("action_valid", True)
        card_name = d.get("card_name", "—")

        if card_idx < 4:
            py = self._kv(px, py, "Card", f"[{card_idx}] {card_name}")
            py = self._kv(px, py, "Tile", f"({tile_x}, {tile_y})")
        else:
            py = self._kv(px, py, "Card", "NOOP", col=(130, 130, 130))
        py = self._kv(px, py, "Valid", "YES" if valid else "NO",
                      col=COL_HP_GREEN if valid else COL_HP_RED)

        value = d.get("value", 0.0)
        py = self._kv(px, py, "V(s)", f"{value:+.4f}")

        py += 6

        # ── Section: REWARD ───────────────────────────────────────────
        py = self._section_header(px, py, "REWARD")
        reward = d.get("reward", 0.0)
        cum_reward = d.get("cumulative_reward", 0.0)
        n_episodes = d.get("n_episodes", 0)

        # Colour reward by sign
        r_col = COL_HP_GREEN if reward > 0.001 else (COL_HP_RED if reward < -0.001 else (170, 170, 170))
        py = self._kv(px, py, "Step R", f"{reward:+.6f}", col=r_col)
        py = self._kv(px, py, "Cum R", f"{cum_reward:+.4f}")
        py = self._kv(px, py, "Episodes", f"{n_episodes}")

        py += 4

        # Component breakdown bars
        breakdown = d.get("reward_breakdown", {})
        if breakdown:
            py = self._label(px, py, "Components:", (170, 170, 170))
            bar_max_w = DEBUG_PANEL_W - 30
            # Find max absolute value for scaling
            max_abs = max(abs(v) for v in breakdown.values()) if breakdown else 1.0
            max_abs = max(max_abs, 0.001)

            for comp_name, comp_val in breakdown.items():
                short = comp_name.replace("Component", "")
                col = _COMP_COLOURS.get(comp_name, (180, 180, 180))
                # Bar
                bar_w = int(abs(comp_val) / max_abs * (bar_max_w * 0.5))
                bar_x = px + 80
                bar_y = py + 2
                bar_h = 10

                # Label
                _draw_text(self._screen, f"{short:>10s}", (px, py),
                           self._font_panel, col, shadow=False)

                if comp_val >= 0:
                    if bar_w > 0:
                        pg.draw.rect(self._screen, col,
                                     (bar_x, bar_y, bar_w, bar_h), border_radius=2)
                else:
                    if bar_w > 0:
                        pg.draw.rect(self._screen, col,
                                     (bar_x - bar_w, bar_y, bar_w, bar_h), border_radius=2)

                # Value text
                _draw_text(self._screen, f"{comp_val:+.4f}",
                           (bar_x + bar_max_w * 0.5 + 4, py),
                           self._font_panel, col, shadow=False)
                py += 15

        py += 6

        # ── Section: REWARD SPARKLINE ─────────────────────────────────
        py = self._section_header(px, py, "REWARD HISTORY")
        self._reward_history.append(d.get("reward", 0.0))
        if len(self._reward_history) > self._max_history:
            self._reward_history = self._reward_history[-self._max_history:]

        if len(self._reward_history) > 2:
            spark_x = px
            spark_w = DEBUG_PANEL_W - 20
            spark_h = 60
            # Background
            pg.draw.rect(self._screen, (15, 15, 15),
                         (spark_x, py, spark_w, spark_h))

            vals = self._reward_history
            max_v = max(max(abs(v) for v in vals), 0.001)
            mid_y = py + spark_h // 2

            # Zero line
            pg.draw.line(self._screen, (60, 60, 60),
                         (spark_x, mid_y), (spark_x + spark_w, mid_y), 1)

            # Plot points
            n = len(vals)
            step_w = spark_w / max(n - 1, 1)
            prev_sx, prev_sy = None, None
            for i, v in enumerate(vals):
                sx = int(spark_x + i * step_w)
                sy = int(mid_y - (v / max_v) * (spark_h // 2 - 4))
                sy = max(py + 2, min(py + spark_h - 2, sy))
                col = COL_HP_GREEN if v > 0 else COL_HP_RED if v < 0 else (100, 100, 100)
                if prev_sx is not None:
                    pg.draw.line(self._screen, col, (prev_sx, prev_sy), (sx, sy), 1)
                prev_sx, prev_sy = sx, sy

            pg.draw.rect(self._screen, (60, 60, 60),
                         (spark_x, py, spark_w, spark_h), 1)

            py += spark_h + 2
            # Scale labels
            _draw_text(self._screen, f"+{max_v:.3f}", (spark_x, py - spark_h - 2),
                       self._font_sm, (100, 100, 100), shadow=False)
            _draw_text(self._screen, f"-{max_v:.3f}", (spark_x, py - 12),
                       self._font_sm, (100, 100, 100), shadow=False)

        py += 12

        # ── Section: TOWER HP ─────────────────────────────────────────
        py = self._section_header(px, py, "TOWER HP")
        n = state.numbers
        py = self._kv(px, py, "Own L", f"{n.left_princess_hp:.0f}/1400",
                      col=self._hp_col(n.left_princess_hp, 1400))
        py = self._kv(px, py, "Own R", f"{n.right_princess_hp:.0f}/1400",
                      col=self._hp_col(n.right_princess_hp, 1400))
        py = self._kv(px, py, "Own K", f"{n.king_hp:.0f}/2400",
                      col=self._hp_col(n.king_hp, 2400))
        py = self._kv(px, py, "Ene L", f"{n.left_enemy_princess_hp:.0f}/1400",
                      col=self._hp_col(n.left_enemy_princess_hp, 1400))
        py = self._kv(px, py, "Ene R", f"{n.right_enemy_princess_hp:.0f}/1400",
                      col=self._hp_col(n.right_enemy_princess_hp, 1400))
        py = self._kv(px, py, "Ene K", f"{n.enemy_king_hp:.0f}/2400",
                      col=self._hp_col(n.enemy_king_hp, 2400))

        py += 6

        # ── Section: ENGINE DEBUG ─────────────────────────────────────
        engine_dbg = d.get("engine_debug", {})
        if engine_dbg:
            py = self._section_header(px, py, "ENGINE SIGNALS")
            dmg_dealt = engine_dbg.get("damage_dealt_total", 0.0)
            dmg_recv = engine_dbg.get("damage_received_total", 0.0)
            own_troops = int(engine_dbg.get("own_troops", 0))
            enemy_troops = int(engine_dbg.get("enemy_troops", 0))
            py = self._kv(px, py, "Dmg dealt", f"{dmg_dealt:.1f}",
                          col=COL_HP_GREEN if dmg_dealt > 0 else (130, 130, 130))
            py = self._kv(px, py, "Dmg recv", f"{dmg_recv:.1f}",
                          col=COL_HP_RED if dmg_recv > 0 else (130, 130, 130))
            py = self._kv(px, py, "Own troop", f"{own_troops}")
            py = self._kv(px, py, "Ene troop", f"{enemy_troops}")

        # ── Section: HAND ─────────────────────────────────────────────
        py += 6
        py = self._section_header(px, py, "HAND")
        hand = d.get("hand", [])
        ready = d.get("ready", [])
        for i, card_info in enumerate(hand):
            is_ready = i in ready
            label = f"[{i}] {card_info}"
            col = COL_HP_GREEN if is_ready else (100, 100, 100)
            py = self._label(px, py, label, col)

    # ── Drawing helpers ───────────────────────────────────────────────

    def _section_header(self, x: int, y: int, text: str) -> int:
        """Draw a section header and return updated y."""
        pg = _ensure_pygame()
        _draw_text(self._screen, text, (x, y),
                   self._font_section, (220, 200, 120), shadow=False)
        y += 18
        pg.draw.line(
            self._screen, (80, 70, 50),
            (x, y - 2), (x + DEBUG_PANEL_W - 20, y - 2), 1,
        )
        return y

    def _kv(self, x: int, y: int, key: str, value: str,
            col: Tuple[int, int, int] = COL_TEXT) -> int:
        """Draw a key:value line and return updated y."""
        _draw_text(self._screen, f"{key:>10s}:", (x, y),
                   self._font_panel, (160, 160, 160), shadow=False)
        _draw_text(self._screen, f" {value}", (x + 90, y),
                   self._font_panel, col, shadow=False)
        return y + 14

    def _label(self, x: int, y: int, text: str,
               col: Tuple[int, int, int] = COL_TEXT) -> int:
        """Draw a simple label and return updated y."""
        _draw_text(self._screen, text, (x + 4, y),
                   self._font_panel, col, shadow=False)
        return y + 14

    @staticmethod
    def _hp_col(hp: float, max_hp: float) -> Tuple[int, int, int]:
        """Return colour based on HP ratio."""
        if hp <= 0:
            return (100, 30, 30)
        ratio = hp / max_hp
        if ratio > 0.5:
            return COL_HP_GREEN
        if ratio > 0.25:
            return COL_HP_YELLOW
        return COL_HP_RED


# ══════════════════════════════════════════════════════════════════════════
# Opponent factory
# ══════════════════════════════════════════════════════════════════════════

_BOT_MAP = {
    "GiantPush": lambda s: GiantPushBot("GiantPush", seed=s),
    "BridgeSpam": lambda s: BridgeSpamBot("BridgeSpam", seed=s),
    "SpellCycle": lambda s: SpellCycleBot("SpellCycle", seed=s),
    "DefCounter": lambda s: DefensiveCounterBot("DefCounter", seed=s),
    "Balanced": lambda s: BalancedBot("Balanced", seed=s),
}


def _make_opponent(name: str, seed: int = 42) -> Any:
    """Create a bot by name."""
    factory = _BOT_MAP.get(name)
    if factory is None:
        avail = ", ".join(_BOT_MAP.keys())
        raise ValueError(f"Unknown opponent '{name}'.  Available: {avail}")
    return factory(seed)


# ══════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual debugger — watch MohaNetLight play with reward overlay.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt model checkpoint.  If omitted, uses a random model.",
    )
    parser.add_argument(
        "--opponent", type=str, default="GiantPush",
        help=f"Opponent bot.  Choices: {', '.join(_BOT_MAP)}",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use argmax actions instead of sampling.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the engine.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (cpu / cuda / mps).",
    )
    parser.add_argument(
        "--no-music", action="store_true",
        help="Disable background music.",
    )
    args = parser.parse_args()

    # ── Model ─────────────────────────────────────────────────────────
    cfg = ModelConfig()
    model = MohaNetLight(cfg)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"Checkpoint not found: {ckpt}")
            sys.exit(1)
        state_dict = torch.load(str(ckpt), map_location=args.device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("No checkpoint — using random (untrained) model")

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # ── Opponent ──────────────────────────────────────────────────────
    opp = _make_opponent(args.opponent, seed=args.seed + 100)
    print(f"Opponent: {args.opponent}")

    # ── Environment (frame_skip=1 for full visual fidelity) ───────────
    env = ClashRoyaleGymEnv(
        opponent=opp.to_player_interface(),
        frame_skip=1,  # see every frame at real speed
        fps=30,
        time_limit=180.0,
        speed_multiplier=1.0,
        seed=args.seed,
    )

    # ── Renderer ──────────────────────────────────────────────────────
    renderer = DebugRenderer(
        fps=30,
        title=f"MohaNet vs {args.opponent} — Visual Debugger",
        speed_multiplier=1.0,
        game_duration=180.0,
    )

    if args.no_music:
        renderer._music_enabled = False

    # ── Run the match ─────────────────────────────────────────────────
    obs, info = env.reset()
    hidden = model.init_hidden(batch_size=1)
    hidden = (hidden[0].to(device), hidden[1].to(device))

    running = True
    step = 0
    cumulative_reward = 0.0
    n_episodes = 0
    episode_reward = 0.0

    print("\n" + "=" * 60)
    print("  VISUAL DEBUGGER — press ESC to quit")
    print("=" * 60 + "\n")

    try:
        while running:
            # ── Agent inference ─────────────────────────────────────
            scalars, troops, troop_mask, cards, action_masks = obs_to_tensors(
                obs, device=device,
            )

            with torch.no_grad():
                output = model.act(
                    scalars, troops, troop_mask, cards, action_masks, hidden,
                )

            hidden = output.hidden
            action_dict = {k: int(v.item()) for k, v in output.actions.items()}
            value = output.value.item()

            # ── Step environment ────────────────────────────────────
            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated
            step += 1
            episode_reward += reward
            cumulative_reward += reward

            # ── Gather debug info ───────────────────────────────────
            card_idx = action_dict["card"]
            state = env.engine.get_state(0)

            # Card name — card_idx is a DECK index (0-7), not a hand slot
            card_name = "NOOP"
            deck = state.deck if state.deck else [c.name for c in state.cards]
            if card_idx < len(deck):
                card_name = deck[card_idx].replace("_", " ").title()

            # Hand info
            hand = []
            for c in state.cards:
                hand.append(f"{c.name.replace('_', ' ').title()} ({c.cost})")

            renderer.set_debug(
                step=step,
                card_idx=card_idx,
                card_name=card_name,
                tile_x=action_dict["tile_x"],
                tile_y=action_dict["tile_y"],
                action_valid=info.get("action_valid", True),
                value=value,
                reward=reward,
                cumulative_reward=cumulative_reward,
                n_episodes=n_episodes,
                reward_breakdown=info.get("reward_breakdown", {}),
                engine_debug=info.get("engine_debug", {}),
                hand=hand,
                ready=list(state.ready),
            )

            # ── Render ──────────────────────────────────────────────
            running = renderer.render(state)

            # ── Episode end ─────────────────────────────────────────
            if done:
                winner = env.engine.get_winner()
                n_episodes += 1
                agent_won = winner == 0

                result = "AGENT WINS!" if agent_won else ("OPPONENT WINS!" if winner == 1 else "DRAW!")
                print(f"\n{'=' * 40}")
                print(f"  Episode {n_episodes}: {result}")
                print(f"  Steps: {step}  |  Episode R: {episode_reward:+.4f}")
                print(f"  Total cumulative R: {cumulative_reward:+.4f}")
                print(f"{'=' * 40}\n")

                # Pause 3 seconds so user can see final state
                end_time = time.time() + 3.0
                while time.time() < end_time and running:
                    running = renderer.render(state)

                # Reset for next episode
                episode_reward = 0.0
                step = 0
                obs, info = env.reset()
                hidden = model.init_hidden(batch_size=1)
                hidden = (hidden[0].to(device), hidden[1].to(device))
                renderer._reward_history.clear()
                continue

            obs = next_obs

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        renderer.close()
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()
