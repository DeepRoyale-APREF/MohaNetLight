"""Baseline trainer — reuses PPO infrastructure with baseline models.

Drop-in replacement for :class:`LeagueTrainer` that accepts any model
implementing the standard interface (``act``, ``evaluate_actions``,
``init_hidden``, ``count_parameters``).

This enables fair comparison: **identical PPO hyperparams, opponents,
and evaluation protocol**, only the model architecture differs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from clash_royale_gymnasium.env import ClashRoyaleGymEnv
from clash_royale_gymnasium.league.match import run_match
from clash_royale_gymnasium.league.player_slot import PlayerSlot

from mohanetlight.baseline.config import BaselineConfig
from mohanetlight.baseline.conv_lstm import ConvLSTMNet
from mohanetlight.baseline.flat_mlp import FlatMLPNet
from mohanetlight.bots.strategies import default_bot_roster
from mohanetlight.config import TrainingConfig
from mohanetlight.inference.baseline_agent import BaselineAgent
from mohanetlight.network.core import LSTMState
from mohanetlight.training.ppo import PPOTrainer
from mohanetlight.training.reward_debugger import RewardDebugger
from mohanetlight.training.rollout import RolloutBuffer
from mohanetlight.utils.tensor_utils import obs_to_tensors

# Type alias for supported baseline models
BaselineModel = Union[ConvLSTMNet, FlatMLPNet]


def build_baseline_model(
    model_type: str,
    cfg: BaselineConfig | None = None,
) -> BaselineModel:
    """Factory for baseline models.

    Parameters
    ----------
    model_type : str
        ``"conv_lstm"`` or ``"flat_mlp"``.
    cfg : BaselineConfig | None
        Architecture config.

    Returns
    -------
    ConvLSTMNet | FlatMLPNet
    """
    cfg = cfg or BaselineConfig()
    if model_type == "flat_mlp":
        return FlatMLPNet(cfg)
    elif model_type == "conv_lstm":
        return ConvLSTMNet(cfg)
    else:
        raise ValueError(f"Unknown baseline model type: {model_type!r}")


class BaselineTrainer:
    """PPO trainer for baseline models — mirrors LeagueTrainer's interface.

    Uses the **same PPO hyperparameters, rollout buffer, opponent pool,
    and evaluation protocol** as the main MohaNet trainer, ensuring a
    fair comparison.

    Parameters
    ----------
    model_type : str
        ``"conv_lstm"`` or ``"flat_mlp"``.
    baseline_cfg : BaselineConfig | None
        Baseline architecture config.
    train_cfg : TrainingConfig | None
        PPO and training loop hyperparameters.
    training_opponents : list[PlayerSlot] | None
        Bots to sample from during data collection.
    eval_opponents : list[PlayerSlot] | None
        Bots for periodic league evaluation.
    env_kwargs : dict | None
        Extra kwargs forwarded to ``ClashRoyaleGymEnv``.
    callback : Callable | None
        Called after each update with ``(update_idx, metrics_dict)``.
    """

    def __init__(
        self,
        model_type: str = "conv_lstm",
        baseline_cfg: BaselineConfig | None = None,
        train_cfg: TrainingConfig | None = None,
        training_opponents: Optional[List[PlayerSlot]] = None,
        eval_opponents: Optional[List[PlayerSlot]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> None:
        self.model_type = model_type
        self.baseline_cfg = baseline_cfg or BaselineConfig()
        self.train_cfg = train_cfg or TrainingConfig()
        self.device = torch.device(self.train_cfg.device)

        # Model
        self.model = build_baseline_model(model_type, self.baseline_cfg)
        self.model.to(self.device)
        model_name = type(self.model).__name__
        print(f"{model_name} — {self.model.count_parameters():,} parameters")

        # PPO (model-agnostic — uses evaluate_actions interface)
        self.ppo = PPOTrainer(self.model, self.train_cfg)

        # Rollout buffer (model-agnostic)
        self.buffer = RolloutBuffer(
            n_steps=self.train_cfg.n_steps,
            gamma=self.train_cfg.gamma,
            gae_lambda=self.train_cfg.gae_lambda,
        )

        # Opponents
        self.training_opponents = training_opponents or default_bot_roster()
        self.eval_opponents = eval_opponents or [
            default_bot_roster()[0],   # GiantPush
            default_bot_roster()[2],   # BridgeSpam
            default_bot_roster()[4],   # SpellCycle
            default_bot_roster()[6],   # DefCounter
            default_bot_roster()[8],   # Balanced
        ]

        self.env_kwargs = env_kwargs or {}
        self.callback = callback
        self._rng = np.random.default_rng(42)

        # Logging
        self.log_dir = Path(self.train_cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_history: List[Dict[str, Any]] = []

        # Reward debugger
        self.reward_debugger = RewardDebugger(
            log_dir=self.log_dir,
            verbose=True,
            component_tracking=True,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop for ``total_timesteps``."""
        cfg = self.train_cfg
        total_updates = cfg.total_timesteps // cfg.n_steps
        global_step = 0
        model_name = type(self.model).__name__

        print(
            f"Training {model_name}: {cfg.total_timesteps:,} steps, "
            f"{total_updates} updates × {cfg.n_steps} steps each\n"
            f"  frame_skip={cfg.frame_skip} → ~{30 / cfg.frame_skip:.1f} decisions/s"
        )

        for update_idx in range(1, total_updates + 1):
            t0 = time.time()

            # 1. Collect rollout
            rollout_info = self._collect_rollout()
            global_step += cfg.n_steps

            # 2. PPO update (with LR annealing)
            progress = (update_idx - 1) / max(total_updates - 1, 1)
            ppo_metrics = self.ppo.update(self.buffer, progress=progress)
            self.buffer.reset()

            elapsed = time.time() - t0
            sps = cfg.n_steps / elapsed

            metrics: Dict[str, Any] = {
                "update": update_idx,
                "global_step": global_step,
                "sps": sps,
                **ppo_metrics,
                **rollout_info,
            }

            # 3. Periodic evaluation
            if update_idx % cfg.eval_interval == 0:
                eval_metrics = self._evaluate(update_idx)
                metrics["eval"] = eval_metrics

            # 4. Checkpoint
            if update_idx % cfg.checkpoint_interval == 0:
                self._save_checkpoint(update_idx)
                self.reward_debugger.save()

            # 5. Log
            self._log(metrics)
            self._metrics_history.append(metrics)

            if self.callback is not None:
                self.callback(update_idx, metrics)

        # Final save
        self._save_checkpoint(total_updates)
        self._save_metrics()
        self.reward_debugger.save()
        print(f"Training {model_name} complete.")

    # ──────────────────────────────────────────────────────────────────────
    # Rollout collection
    # ──────────────────────────────────────────────────────────────────────

    def _collect_rollout(self) -> Dict[str, float]:
        """Collect exactly ``n_steps`` transitions (fixed-length rollout)."""
        opp_idx = int(self._rng.integers(len(self.training_opponents)))
        opp = self.training_opponents[opp_idx]
        opp.reset()

        env = ClashRoyaleGymEnv(
            opponent=opp.to_player_interface(),
            frame_skip=self.train_cfg.frame_skip,
            **self.env_kwargs,
        )

        obs, info = env.reset()
        hidden = self.model.init_hidden(batch_size=1)
        hidden = _to_device(hidden, self.device)

        self.model.eval()

        episode_rewards: List[float] = []
        ep_reward = 0.0
        n_episodes = 0
        step = 0

        while step < self.train_cfg.n_steps:
            scalars, troops, troop_mask, cards, arena_map, action_masks = obs_to_tensors(
                obs, device=self.device,
            )

            output = self.model.act(
                scalars, troops, troop_mask, cards, arena_map, action_masks, hidden,
            )

            action_dict = {k: int(v.item()) for k, v in output.actions.items()}
            self.buffer.add(
                obs=obs,
                action=action_dict,
                log_prob=output.log_prob.item(),
                value=output.value.item(),
                reward=0.0,
                done=False,
                hidden=_detach(hidden),
            )

            hidden = output.hidden

            gym_action = {k: int(v.item()) for k, v in output.actions.items()}
            next_obs, reward, terminated, truncated, info = env.step(gym_action)

            done = terminated or truncated
            self.buffer.rewards[-1] = float(reward)
            self.buffer.dones[-1] = done
            ep_reward += float(reward)

            self.reward_debugger.on_step(
                reward=float(reward),
                done=done,
                info=info,
            )

            obs = next_obs
            step += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                n_episodes += 1
                obs, info = env.reset()
                hidden = self.model.init_hidden(batch_size=1)
                hidden = _to_device(hidden, self.device)

        # Bootstrap value
        with torch.no_grad():
            s, tr, m, c, am_map, am = obs_to_tensors(obs, device=self.device)
            last_output = self.model.act(s, tr, m, c, am_map, am, hidden)
            last_value = last_output.value.item()

        self.buffer.finish(last_value)
        reward_snapshot = self.reward_debugger.finish_rollout()
        env.close()

        cross_ep_reward = self.reward_debugger.get_mean_episode_reward()
        mean_ep_reward = (
            float(np.mean(episode_rewards)) if episode_rewards else cross_ep_reward
        )

        return {
            "mean_ep_reward": mean_ep_reward,
            "n_episodes": n_episodes,
            "opponent": opp.name,
            "rollout_reward_sum": reward_snapshot.total,
            "rollout_reward_mean": reward_snapshot.mean,
            "rollout_reward_abs_mean": reward_snapshot.abs_mean,
            "rollout_reward_nonzero_frac": reward_snapshot.nonzero_frac,
            "rollout_reward_min": reward_snapshot.min,
            "rollout_reward_max": reward_snapshot.max,
            "reward_components": reward_snapshot.component_sums,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate(self, update_idx: int) -> Dict[str, Any]:
        """Play agent against each eval opponent — identical protocol."""
        self.model.eval()
        model_name = type(self.model).__name__

        agent = BaselineAgent(
            name=f"{model_name}-u{update_idx}",
            model=self.model,
            device=str(self.device),
        )

        mpp = self.train_cfg.eval_matches_per_pair
        total_wins = 0
        total_towers = 0
        total_matches = 0

        for opp in self.eval_opponents:
            for m in range(mpp):
                if m % 2 == 0:
                    p0, p1 = agent, opp
                    agent_pid = 0
                else:
                    p0, p1 = opp, agent
                    agent_pid = 1

                result = run_match(
                    p0, p1,
                    frame_skip=self.train_cfg.frame_skip,
                    seed=update_idx * 1000 + total_matches,
                )

                if result.winner == agent_pid:
                    total_wins += 1
                if agent_pid == 0:
                    total_towers += result.p0_towers_destroyed
                else:
                    total_towers += result.p1_towers_destroyed
                total_matches += 1

            print(
                f"    eval vs {opp.name}: "
                f"{total_wins}/{total_matches} wins so far",
                flush=True,
            )

        win_rate = total_wins / max(total_matches, 1)
        avg_crowns = total_towers / max(total_matches, 1)

        print(
            f"  [Eval u{update_idx}] {model_name} — Win rate: {win_rate:.1%}, "
            f"Avg crowns: {avg_crowns:.2f}  ({total_matches} matches)",
            flush=True,
        )

        return {
            "win_rate": win_rate,
            "avg_crowns": avg_crowns,
            "matches": total_matches,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing & logging
    # ──────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, update_idx: int) -> None:
        """Save model state dict."""
        prefix = self.model_type
        path = self.log_dir / f"{prefix}_u{update_idx}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    def _save_metrics(self) -> None:
        """Dump training metrics to JSON."""
        path = self.log_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(self._metrics_history, f, indent=2, default=str)

    def _log(self, metrics: Dict[str, Any]) -> None:
        """Print a compact summary line."""
        u = metrics["update"]
        sps = metrics.get("sps", 0)
        pl = metrics.get("policy_loss", 0)
        vl = metrics.get("value_loss", 0)
        ent = metrics.get("entropy", 0)
        opp = metrics.get("opponent", "?")
        r_sum = metrics.get("rollout_reward_sum", 0.0)
        r_mean = metrics.get("rollout_reward_mean", 0.0)
        r_abs = metrics.get("rollout_reward_abs_mean", 0.0)
        r_nz = metrics.get("rollout_reward_nonzero_frac", 0.0)
        n_eps = metrics.get("n_episodes", 0)
        mr = metrics.get("mean_ep_reward", 0.0)

        print(
            f"[u{u:4d}] sps={sps:.0f}  π={pl:.4f}  v={vl:.4f}  "
            f"H={ent:.3f}  SR={r_sum:+.3f}  mR={r_mean:+.5f}  "
            f"|R|={r_abs:.5f}  nz={r_nz:.0%}  ep={n_eps}  "
            f"epR={mr:+.3f}  vs={opp}",
            flush=True,
        )

        components = metrics.get("reward_components", {})
        if components and u % 25 == 0:
            parts = "  ".join(f"{k}={v:+.4f}" for k, v in components.items())
            print(f"        [components] {parts}", flush=True)

        if u % 50 == 0:
            print(self.reward_debugger.diagnose(u), flush=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LSTM hidden state utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _to_device(hidden: LSTMState, device: torch.device) -> LSTMState:
    return (hidden[0].to(device), hidden[1].to(device))


def _detach(hidden: LSTMState) -> LSTMState:
    return (hidden[0].detach().cpu(), hidden[1].detach().cpu())
