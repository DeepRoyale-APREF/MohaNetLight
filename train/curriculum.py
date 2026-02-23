"""Curriculum trainer — progressive multi-phase PPO training.

Implements a 4-phase curriculum:
1. **Warm-up**: Train exclusively vs BalancedBot (easiest generaliser).
2. **Tank pressure**: Train exclusively vs GiantPushBot (learn countering).
3. **Full league**: Train vs all 10 heuristic bots (generalise).
4. **Self-play**: Train vs best checkpoint of MohaNet itself.

Each phase produces its own checkpoint, metrics JSON, and can be resumed.
The report generator reads these to produce comparison plots.
"""

from __future__ import annotations
from mohanetlight.bots import BridgeSpamBot

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from mohanetlight.bots.strategies import (
    BalancedBot,
    GiantPushBot,
    default_bot_roster,
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
# Default 4-phase curriculum
# ═══════════════════════════════════════════════════════════════════════════════


def default_curriculum(
    steps_per_phase: int = 200_000,
    self_play_steps: int = 200_000,
    n_steps: int = 512,
    lr: float = 3e-4,
    base_seed: int = 42,
) -> List[PhaseConfig]:
    """Build the default 4-phase curriculum.

    Parameters
    ----------
    steps_per_phase : int
        Env steps for phases 1-3.
    self_play_steps : int
        Env steps for phase 4 (self-play).
    n_steps : int
        Rollout length (steps per PPO update).
    lr : float
        Base learning rate.
    base_seed : int
        Seed for bot RNGs.

    Returns
    -------
    list[PhaseConfig]
        Four phases: Balanced → GiantPush → Full League → Self-play.
    """
    all_bots = default_bot_roster(base_seed)

    # Phase 1: Only Balanced bots (easiest, learn basics)
    balanced_bots = [
        BalancedBot("Balanced-5", base_threshold=5.0, seed=base_seed),
        BalancedBot("Balanced-3", base_threshold=3.0, seed=base_seed + 1),
    ]

    # Phase 2: Only Strong bots (learn to counter tanks)
    strong_bots = [
        GiantPushBot("GiantPush-7", elixir_threshold=7.0, seed=base_seed + 2),
        GiantPushBot("GiantPush-5", elixir_threshold=5.0, seed=base_seed + 3),
        BridgeSpamBot("BridgeSpam-5", seed=base_seed + 4),
        BridgeSpamBot("BridgeSpam-3", seed=base_seed + 5),
    ]

    # Phase 3: Full league (all 10 bots)
    # Phase 4: Self-play is configured separately by the runner

    return [
        PhaseConfig(
            name="balanced",
            total_timesteps=steps_per_phase,
            training_opponents=balanced_bots,
            eval_opponents=balanced_bots,
            n_steps=n_steps,
            lr=lr,
            ent_coef=0.05,  # High entropy to survive early value-loss dominance
            eval_interval=25,
            checkpoint_interval=25,
        ),
        PhaseConfig(
            name="strong",
            total_timesteps=steps_per_phase,
            training_opponents=strong_bots,
            eval_opponents=strong_bots,
            n_steps=n_steps,
            lr=lr,
            ent_coef=0.03,
            eval_interval=25,
            checkpoint_interval=25,
        ),
        PhaseConfig(
            name="full_league",
            total_timesteps=steps_per_phase,
            training_opponents=all_bots,
            eval_opponents=[all_bots[0], all_bots[2], all_bots[4],
                            all_bots[6], all_bots[8]],
            n_steps=n_steps,
            lr=lr * 0.5,  # Reduce LR for fine-tuning
            ent_coef=0.01,
            eval_interval=25,
            checkpoint_interval=25,
        ),
        PhaseConfig(
            name="self_play",
            total_timesteps=self_play_steps,
            training_opponents=[],  # Filled by runner with MohaNetAgent
            eval_opponents=[all_bots[0], all_bots[2], all_bots[4],
                            all_bots[6], all_bots[8]],
            n_steps=n_steps,
            lr=lr * 0.3,  # Even lower LR
            ent_coef=0.005,
            eval_interval=25,
            checkpoint_interval=25,
        ),
    ]
