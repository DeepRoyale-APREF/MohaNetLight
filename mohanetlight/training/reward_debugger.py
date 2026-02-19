"""Reward debugging utilities for diagnosing zero/low reward during training.

Tracks per-step reward statistics, per-component breakdowns, and
cross-rollout episode accumulation so that training progress can be
monitored even when no episode completes within a single rollout.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RewardSnapshot:
    """Statistics for rewards collected during one rollout."""

    # Per-step statistics
    step_rewards: np.ndarray = field(default_factory=lambda: np.array([]))
    n_nonzero: int = 0
    n_positive: int = 0
    n_negative: int = 0

    # Per-component cumulative (keyed by component class name)
    component_sums: Dict[str, float] = field(default_factory=dict)
    component_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(np.sum(self.step_rewards)) if len(self.step_rewards) > 0 else 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.step_rewards)) if len(self.step_rewards) > 0 else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.step_rewards)) if len(self.step_rewards) > 0 else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.step_rewards)) if len(self.step_rewards) > 0 else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self.step_rewards)) if len(self.step_rewards) > 0 else 0.0

    @property
    def abs_mean(self) -> float:
        """Mean of absolute reward values — useful to detect tiny but nonzero signals."""
        return float(np.mean(np.abs(self.step_rewards))) if len(self.step_rewards) > 0 else 0.0

    @property
    def nonzero_frac(self) -> float:
        n = len(self.step_rewards)
        return self.n_nonzero / n if n > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSON logging."""
        d: Dict[str, Any] = {
            "total": round(self.total, 6),
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min, 6),
            "max": round(self.max, 6),
            "abs_mean": round(self.abs_mean, 6),
            "n_steps": len(self.step_rewards),
            "n_nonzero": self.n_nonzero,
            "nonzero_frac": round(self.nonzero_frac, 4),
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }
        if self.component_sums:
            d["components"] = {
                k: round(v, 6) for k, v in self.component_sums.items()
            }
        return d


class RewardDebugger:
    """Tracks reward statistics across rollouts and episodes.

    Designed to be embedded in the trainer to provide visibility into
    why rewards may appear zero or tiny in the log line.

    Parameters
    ----------
    log_dir : Path | str | None
        If set, periodically saves detailed debug logs here.
    verbose : bool
        Print detailed reward diagnostics inline.
    component_tracking : bool
        If True, tracks per-component reward breakdown (requires env info
        to contain ``reward_breakdown``).
    """

    def __init__(
        self,
        log_dir: Optional[Path | str] = None,
        verbose: bool = True,
        component_tracking: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir) if log_dir else None
        self._verbose = verbose
        self._component_tracking = component_tracking

        # Cross-rollout episode tracking
        self._current_ep_reward: float = 0.0
        self._current_ep_steps: int = 0
        self._completed_episodes: List[Dict[str, Any]] = []

        # Per-rollout accumulators
        self._step_rewards: List[float] = []
        self._component_accum: Dict[str, float] = defaultdict(float)
        self._component_count: Dict[str, int] = defaultdict(int)

        # Historical snapshots
        self._rollout_history: List[Dict[str, Any]] = []

        # Diagnosis counters
        self._total_steps: int = 0
        self._total_nonzero_steps: int = 0
        self._all_rewards_zero_count: int = 0  # rollouts with ALL zero rewards

    def on_step(
        self,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called after each environment step during rollout collection.

        Parameters
        ----------
        reward : float
            The reward returned by ``env.step()``.
        done : bool
            Whether the episode ended.
        info : dict, optional
            The info dict from ``env.step()`` (may contain ``reward_breakdown``).
        """
        self._step_rewards.append(reward)
        self._current_ep_reward += reward
        self._current_ep_steps += 1
        self._total_steps += 1

        if abs(reward) > 1e-10:
            self._total_nonzero_steps += 1

        # Track per-component breakdown
        if self._component_tracking and info is not None:
            breakdown = info.get("reward_breakdown", {})
            for comp_name, comp_val in breakdown.items():
                self._component_accum[comp_name] += comp_val
                self._component_count[comp_name] += 1

        if done:
            self._completed_episodes.append({
                "total_reward": self._current_ep_reward,
                "steps": self._current_ep_steps,
            })
            self._current_ep_reward = 0.0
            self._current_ep_steps = 0

    def finish_rollout(self) -> RewardSnapshot:
        """Called at end of rollout. Returns snapshot and resets step buffer.

        Returns
        -------
        RewardSnapshot
            Statistics for the just-completed rollout.
        """
        arr = np.array(self._step_rewards, dtype=np.float32)
        snapshot = RewardSnapshot(
            step_rewards=arr,
            n_nonzero=int(np.count_nonzero(arr)),
            n_positive=int(np.sum(arr > 1e-10)),
            n_negative=int(np.sum(arr < -1e-10)),
            component_sums=dict(self._component_accum),
            component_counts=dict(self._component_count),
        )

        if snapshot.n_nonzero == 0:
            self._all_rewards_zero_count += 1

        self._rollout_history.append(snapshot.to_dict())
        self._step_rewards.clear()
        self._component_accum.clear()
        self._component_count.clear()
        return snapshot

    def get_cross_rollout_episodes(self) -> List[Dict[str, Any]]:
        """Return all completed episodes (persisted across rollout boundaries)."""
        return list(self._completed_episodes)

    def get_mean_episode_reward(self) -> float:
        """Mean reward of completed episodes. 0.0 if none completed yet."""
        if not self._completed_episodes:
            return 0.0
        return float(np.mean([e["total_reward"] for e in self._completed_episodes]))

    def diagnose(self, update_idx: int) -> str:
        """Return a multi-line diagnostic string for the current training state.

        Parameters
        ----------
        update_idx : int
            Current update number (for context).

        Returns
        -------
        str
            Human-readable diagnosis.
        """
        lines = [f"\n{'='*70}", f"  REWARD DIAGNOSIS — update {update_idx}", f"{'='*70}"]

        # Overall stats
        total_nonzero_frac = (
            self._total_nonzero_steps / self._total_steps
            if self._total_steps > 0 else 0.0
        )
        lines.append(f"  Total steps seen:          {self._total_steps:,}")
        lines.append(f"  Steps with nonzero reward:  {self._total_nonzero_steps:,} "
                      f"({total_nonzero_frac:.1%})")
        lines.append(f"  Rollouts with ALL zero R:   {self._all_rewards_zero_count}")

        # Episode completion
        n_eps = len(self._completed_episodes)
        lines.append(f"  Completed episodes:         {n_eps}")
        if n_eps > 0:
            ep_rewards = [e["total_reward"] for e in self._completed_episodes]
            ep_steps = [e["steps"] for e in self._completed_episodes]
            lines.append(f"    Mean ep reward: {np.mean(ep_rewards):.4f}")
            lines.append(f"    Mean ep length: {np.mean(ep_steps):.0f} steps")
        else:
            lines.append("    ⚠  No episodes completed yet!")
            lines.append(f"    Current partial episode: {self._current_ep_steps} steps, "
                          f"R={self._current_ep_reward:.4f}")

        # Last rollout detail
        if self._rollout_history:
            last = self._rollout_history[-1]
            lines.append(f"\n  Last rollout:")
            lines.append(f"    Sum R = {last['total']:.6f}  "
                          f"Mean R = {last['mean']:.6f}  "
                          f"|R| = {last['abs_mean']:.6f}")
            lines.append(f"    Min = {last['min']:.6f}  "
                          f"Max = {last['max']:.6f}  "
                          f"Std = {last['std']:.6f}")
            lines.append(f"    Nonzero: {last['n_nonzero']}/{last['n_steps']} "
                          f"({last['nonzero_frac']:.1%})")
            if "components" in last and last["components"]:
                lines.append(f"    Per-component sums:")
                for comp, val in last["components"].items():
                    lines.append(f"      {comp:25s} = {val:+.6f}")

        lines.append(f"{'='*70}\n")
        return "\n".join(lines)

    def save(self, path: Optional[Path] = None) -> None:
        """Save full debug history to JSON."""
        out = path or (self._log_dir / "reward_debug.json" if self._log_dir else None)
        if out is None:
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_steps": self._total_steps,
            "total_nonzero_steps": self._total_nonzero_steps,
            "all_zero_rollouts": self._all_rewards_zero_count,
            "completed_episodes": self._completed_episodes,
            "rollout_history": self._rollout_history,
        }
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)
