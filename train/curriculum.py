"""Curriculum trainer — progressive multi-phase PPO training.

Implements an 8-phase micro-curriculum (~50 updates each):
1. **warmup**: Passive + weak Balanced — free wins to bootstrap policy.
2. **basics**: Balanced(5) + mild Passive — learn elixir management.
3. **push_intro**: Medium bots + GiantPush — learn to counter pushes.
4. **aggression**: Balanced(3) + BridgeSpam — face constant pressure.
5. **diversity**: Mixed strategies (Spam, Spell, DefCounter) — generalise.
6. **hard**: Expert-parametrized bots (low thresholds) + previous bots.
7. **full_league**: All 10 default bots + expert bots — final generalisation.
8. **self_play**: Self-play + league bots — peak competitive play.

Previous-phase opponents are mixed into later phases to prevent
catastrophic forgetting.  Reports are generated at each phase boundary.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from mohanetlight.bots.strategies import (
    BalancedBot,
    BridgeSpamBot,
    GiantPushBot,
    OptimalBot,
    PassiveBot,
    default_bot_roster,
    easy_bots,
    expert_bots,
    hard_bots,
    medium_bots,
    optimal_bots,
)
from mohanetlight.config import ModelConfig, TrainingConfig
from mohanetlight.inference.agent import MohaNetAgent
from mohanetlight.network.mohanet import MohaNetLight
from mohanetlight.training.ppo import PPOTrainer
from mohanetlight.training.rollout import RolloutBuffer
from mohanetlight.training.trainer import LeagueTrainer

from clash_royale_gymnasium.league.player_slot import PlayerSlot


# ═══════════════════════════════════════════════════════════════════════════════
# Phase configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase.

    Parameters
    ----------
    name : str
        Human-readable phase name.
    total_timesteps : int
        Total env steps for this phase.
    training_opponents : list[PlayerSlot]
        Bots to train against (randomly sampled each rollout).
    eval_opponents : list[PlayerSlot] | None
        Bots for periodic evaluation.  If None, uses training_opponents.
    n_steps : int
        Steps per rollout.
    lr : float
        Learning rate for this phase.
    ent_coef : float
        Entropy coefficient.
    eval_interval : int
        Evaluate every N updates.
    checkpoint_interval : int
        Save checkpoint every N updates.
    resume_from : str | None
        Path to a checkpoint to resume from (weights only).
    """

    name: str
    total_timesteps: int
    training_opponents: List[PlayerSlot] = field(default_factory=list)
    eval_opponents: Optional[List[PlayerSlot]] = None
    n_steps: int = 2048
    lr: float = 3e-4
    ent_coef: float = 0.01
    eval_interval: int = 25
    checkpoint_interval: int = 25
    resume_from: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum Trainer
# ═══════════════════════════════════════════════════════════════════════════════


class CurriculumTrainer:
    """Multi-phase curriculum training with metric tracking.

    Each phase creates a :class:`LeagueTrainer` internally, runs it,
    and stores per-update metrics (episode return, critic loss, etc.)
    for later reporting.

    Parameters
    ----------
    phases : list[PhaseConfig]
        Ordered list of training phases.
    model_cfg : ModelConfig
        Network architecture (shared across all phases).
    base_log_dir : str | Path
        Root directory under which ``phase_N_<name>/`` folders are created.
    device : str | None
        Torch device override.  ``None`` = auto-detect.
    frame_skip : int
        Engine frame skip for environments.
    """

    def __init__(
        self,
        phases: List[PhaseConfig],
        model_cfg: ModelConfig | None = None,
        base_log_dir: str | Path = "./logs/curriculum",
        device: Optional[str] = None,
        frame_skip: int = 10,
    ) -> None:
        self.phases = phases
        self.model_cfg = model_cfg or ModelConfig()
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.frame_skip = frame_skip

        # Collected across all phases
        self.all_metrics: Dict[str, List[Dict[str, Any]]] = {}

    def run(self) -> Dict[str, List[Dict[str, Any]]]:
        """Execute all phases sequentially, carrying weights forward.

        Returns
        -------
        dict[str, list[dict]]
            Phase name → list of per-update metric dicts.
        """
        last_checkpoint: Optional[str] = None

        for i, phase in enumerate(self.phases):
            phase_dir = self.base_log_dir / f"phase_{i + 1}_{phase.name}"
            phase_dir.mkdir(parents=True, exist_ok=True)

            # Determine starting weights
            resume_path = phase.resume_from or last_checkpoint

            print("\n" + "=" * 70)
            print(f"  PHASE {i + 1}/{len(self.phases)}: {phase.name}")
            print(f"  Steps: {phase.total_timesteps:,}  |  "
                  f"LR: {phase.lr}  |  Opponents: "
                  f"{[o.name for o in phase.training_opponents]}")
            if resume_path:
                print(f"  Resuming from: {resume_path}")
            print("=" * 70 + "\n")

            # Build TrainingConfig for this phase
            train_kwargs: Dict[str, Any] = dict(
                total_timesteps=phase.total_timesteps,
                n_steps=phase.n_steps,
                lr=phase.lr,
                ent_coef=phase.ent_coef,
                eval_interval=phase.eval_interval,
                checkpoint_interval=phase.checkpoint_interval,
                log_dir=str(phase_dir),
                frame_skip=self.frame_skip,
            )
            if self.device is not None:
                train_kwargs["device"] = self.device

            train_cfg = TrainingConfig(**train_kwargs)

            # Build trainer
            eval_opps = phase.eval_opponents or phase.training_opponents
            trainer = LeagueTrainer(
                model_cfg=self.model_cfg,
                train_cfg=train_cfg,
                training_opponents=phase.training_opponents,
                eval_opponents=eval_opps,
            )

            # Load weights from previous phase
            if resume_path and Path(resume_path).exists():
                state_dict = torch.load(
                    resume_path,
                    map_location=train_cfg.device,
                    weights_only=True,
                )
                trainer.model.load_state_dict(state_dict)
                print(f"  Loaded weights from {resume_path}")

            # Collect metrics via callback
            phase_metrics: List[Dict[str, Any]] = []

            def _on_update(update_idx: int, metrics: Dict[str, Any]) -> None:
                phase_metrics.append(metrics)

            trainer.callback = _on_update

            # Train
            t0 = time.time()
            trainer.train()
            elapsed = time.time() - t0

            print(f"\n  Phase '{phase.name}' complete in {elapsed:.0f}s "
                  f"({len(phase_metrics)} updates)")

            # Save phase metrics
            self.all_metrics[phase.name] = phase_metrics
            metrics_path = phase_dir / "phase_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(phase_metrics, f, indent=2, default=str)

            # Find the best/last checkpoint for next phase
            last_checkpoint = self._find_last_checkpoint(phase_dir)

        # Save combined metrics
        combined_path = self.base_log_dir / "all_phases_metrics.json"
        with open(combined_path, "w") as f:
            json.dump(
                {k: v for k, v in self.all_metrics.items()},
                f, indent=2, default=str,
            )

        print(f"\nAll {len(self.phases)} phases complete.")
        print(f"Combined metrics saved to {combined_path}")

        return self.all_metrics

    @staticmethod
    def _find_last_checkpoint(phase_dir: Path) -> Optional[str]:
        """Find the highest-numbered checkpoint in a phase directory."""
        pts = sorted(phase_dir.glob("mohanet_u*.pt"))
        if not pts:
            return None
        return str(pts[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# Default 8-phase curriculum — converging toward OptimalBot
# ═══════════════════════════════════════════════════════════════════════════════


def default_curriculum(
    steps_per_phase: int = 200_000,
    self_play_steps: int = 200_000,
    n_steps: int = 512,
    lr: float = 3e-4,
    base_seed: int = 42,
    updates_per_phase: int = 50,
) -> List[PhaseConfig]:
    """Build the default 8-phase micro-curriculum.

    Progressive difficulty from Balanced bots → GiantPush → aggressive
    strategies → expert bots → OptimalBot → self-play.  Previous
    opponents are mixed into later phases to prevent forgetting.

    Parameters
    ----------
    steps_per_phase : int
        Ignored when ``updates_per_phase`` is set (kept for CLI compat).
    self_play_steps : int
        Ignored when ``updates_per_phase`` is set (kept for CLI compat).
    n_steps : int
        Rollout length (steps per PPO update).
    lr : float
        Base learning rate.
    base_seed : int
        Seed for bot RNGs.
    updates_per_phase : int
        Number of PPO updates per phase (default 50).

    Returns
    -------
    list[PhaseConfig]
        Eight phases converging toward beating the OptimalBot.
    """
    phase_steps = updates_per_phase * n_steps

    # ── Tiered bot pools ──────────────────────────────────────────────────
    t1 = easy_bots(base_seed)       # Balanced-7, Balanced-5, GiantPush-7
    t2 = medium_bots(base_seed)     # GiantPush-5, Balanced-3, BridgeSpam-4
    t3 = hard_bots(base_seed)       # BridgeSpam-3, GiantPush-4, DefCounter-5, SpellCycle-5
    t4 = expert_bots(base_seed)     # DefCounter-4, Balanced-2, BridgeSpam-2, Optimal-7(0.7)
    t5 = optimal_bots(base_seed)    # Optimal-7(1.0), Optimal-5(1.0)
    all_bots = default_bot_roster(base_seed)

    # Eval set: consistent opponents for tracking progress across phases
    eval_set: List[PlayerSlot] = [
        BalancedBot("eval-Balanced-5", base_threshold=5.0, seed=base_seed + 90),
        GiantPushBot("eval-GiantPush-5", elixir_threshold=5.0, seed=base_seed + 91),
        OptimalBot("eval-Optimal-7", push_threshold=7.0, aggression=1.0, seed=base_seed + 92),
    ]

    return [
        # ── Phase 1: warmup ──────────────────────────────────────────────
        # Balanced + cautious GiantPush — easy wins, the agent already
        # beats these.  Bootstraps the policy.
        PhaseConfig(
            name="warmup",
            total_timesteps=phase_steps,
            training_opponents=t1,
            eval_opponents=eval_set,
            n_steps=n_steps,
            lr=lr,
            ent_coef=0.02,
            eval_interval=25,
            checkpoint_interval=25,
        ),
        # ── Phase 2: push_defense ────────────────────────────────────────
        # Agent learns to counter faster giant pushes + constant pressure.
        # Recalls Balanced-7 to not forget easy matchups.
        PhaseConfig(
            name="push_defense",
            total_timesteps=phase_steps,
            training_opponents=t2 + [t1[0]],  # + Balanced-7 recall
            eval_opponents=eval_set,
            n_steps=n_steps,
            lr=lr,
            ent_coef=0.02,
            eval_interval=25,
            checkpoint_interval=25,
        ),
        # ── Phase 3: aggression ──────────────────────────────────────────
        # Aggressive spam + fast pushes.  Agent learns to defend under
        # constant pressure.  Recalls GiantPush-5.
        PhaseConfig(
            name="aggression",
            total_timesteps=phase_steps,
            training_opponents=t3[:3] + [t2[0]],  # BridgeSpam-3, GiantPush-4, DefCounter-5 + GiantPush-5
            eval_opponents=eval_set,
            n_steps=n_steps,
            lr=lr,
            ent_coef=0.015,
            eval_interval=25,
            checkpoint_interval=25,
        ),
        # ── Phase 4: diversity ───────────────────────────────────────────
        # All hard archetypes: spam, push, counter, spell.  Agent
        # generalises defenses.  Recalls easy bots to prevent forgetting.
        PhaseConfig(
            name="diversity",
            total_timesteps=phase_steps,
            training_opponents=t3 + [t1[0]],  # full hard + Balanced-7 recall
            eval_opponents=eval_set,
            n_steps=n_steps,
            lr=lr * 0.8,
            ent_coef=0.015,
            eval_interval=25,
            checkpoint_interval=25,
        ),
        # ── Phase 5: expert ──────────────────────────────────────────────
        # Expert-param bots including a weakened OptimalBot (aggression=0.7).
        # Agent starts seeing optimal play patterns.  Recalls Balanced-3.
        PhaseConfig(
            name="expert",
            total_timesteps=phase_steps,
            training_opponents=t4 + [t2[1]],  # expert + Balanced-3 recall
            eval_opponents=eval_set,
            n_steps=n_steps,
            lr=lr * 0.6,
            ent_coef=0.01,
            eval_interval=25,
            checkpoint_interval=25,
        ),
        # ── Phase 6: optimal_intro ───────────────────────────────────────
        # Full-strength OptimalBot enters, mixed with expert bots for
        # variety.  Agent can explore strategies that counter optimal play.
        PhaseConfig(
            name="optimal_intro",
            total_timesteps=phase_steps,
            training_opponents=t5 + t4[:2],  # Optimal full + DefCounter-4, Balanced-2 recall
            eval_opponents=eval_set,
            n_steps=n_steps,
            lr=lr * 0.5,
            ent_coef=0.008,
            eval_interval=25,
            checkpoint_interval=25,
        ),
        # ── Phase 7: full_league ─────────────────────────────────────────
        # All bots including OptimalBot — ultimate generalisation test.
        PhaseConfig(
            name="full_league",
            total_timesteps=phase_steps,
            training_opponents=all_bots + t5,  # 10 default + 2 Optimal
            eval_opponents=eval_set,
            n_steps=n_steps,
            lr=lr * 0.4,
            ent_coef=0.008,
            eval_interval=25,
            checkpoint_interval=25,
        ),
        # ── Phase 8: self_play ───────────────────────────────────────────
        # Self-play + OptimalBot.  Opponents injected by runner.
        PhaseConfig(
            name="self_play",
            total_timesteps=phase_steps,
            training_opponents=t5,  # Runner appends MohaNetAgent
            eval_opponents=eval_set,
            n_steps=n_steps,
            lr=lr * 0.3,
            ent_coef=0.005,
            eval_interval=25,
            checkpoint_interval=25,
        ),
    ]
