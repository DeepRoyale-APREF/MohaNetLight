"""League trainer — orchestrates PPO training with periodic league evaluation.

Dataflow per iteration:
1. Collect ``n_steps`` environment transitions against a random training opponent.
2. Compute GAE advantages.
3. Run ``n_epochs`` of PPO clipped updates (truncated BPTT).
4. Every ``eval_interval`` updates, run a league tournament against heuristic bots.
5. Save checkpoints periodically.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from clash_royale_gymnasium.env import ClashRoyaleGymEnv
from clash_royale_gymnasium.league.player_slot import HeuristicSlot, PlayerSlot
from clash_royale_gymnasium.league.tournament import LeagueTournament

from mohanetlight.bots.strategies import default_bot_roster
from mohanetlight.config import ModelConfig, TrainingConfig
from mohanetlight.inference.agent import MohaNetAgent
from mohanetlight.network.core import LSTMState
from mohanetlight.network.mohanet import MohaNetLight
from mohanetlight.training.ppo import PPOTrainer
from mohanetlight.training.reward_debugger import RewardDebugger, RewardSnapshot
from mohanetlight.training.rollout import RolloutBuffer
from mohanetlight.utils.tensor_utils import obs_to_tensors


class LeagueTrainer:
    """End-to-end PPO + league evaluation training loop.

    Parameters
    ----------
    model_cfg : ModelConfig
        Network architecture config.
    train_cfg : TrainingConfig
        PPO and training loop hyperparameters.
    training_opponents : list[PlayerSlot] | None
        Bots to sample from during data collection (default: heuristic roster).
    eval_opponents : list[PlayerSlot] | None
        Bots for periodic league evaluation (default: heuristic roster).
    env_kwargs : dict | None
        Extra kwargs forwarded to ``ClashRoyaleGymEnv``.
    callback : Callable | None
        Called after each update with ``(update_idx, metrics_dict)``.
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        train_cfg: TrainingConfig | None = None,
        training_opponents: Optional[List[PlayerSlot]] = None,
        eval_opponents: Optional[List[PlayerSlot]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> None:
        self.model_cfg = model_cfg or ModelConfig()
        self.train_cfg = train_cfg or TrainingConfig()
        self.device = torch.device(self.train_cfg.device)

        # Model
        self.model = MohaNetLight(self.model_cfg).to(self.device)
        print(f"MohaNetLight — {self.model.count_parameters():,} parameters")

        # PPO
        self.ppo = PPOTrainer(self.model, self.train_cfg)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=self.train_cfg.n_steps,
            gamma=self.train_cfg.gamma,
            gae_lambda=self.train_cfg.gae_lambda,
        )

        # Opponents
        self.training_opponents = training_opponents or default_bot_roster()
        self.eval_opponents = eval_opponents or [
            HeuristicSlot("Heuristic-0.3", aggression=0.3, seed=100),
            HeuristicSlot("Heuristic-0.5", aggression=0.5, seed=101),
            HeuristicSlot("Heuristic-0.8", aggression=0.8, seed=102),
        ] + default_bot_roster()[:5]

        self.env_kwargs = env_kwargs or {}
        self.callback = callback
        self._rng = np.random.default_rng(42)

        # Logging
        self.log_dir = Path(self.train_cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_history: List[Dict[str, Any]] = []

        # Reward debugger — tracks per-step and cross-rollout reward signals
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

        print(
            f"Training: {cfg.total_timesteps:,} steps, "
            f"{total_updates} updates × {cfg.n_steps} steps each\n"
            f"  frame_skip={cfg.frame_skip} → ~{30 // cfg.frame_skip} decisions/s, "
            f"~{int(240 * 30 / cfg.frame_skip)} steps/match (240s max)"
        )

        for update_idx in range(1, total_updates + 1):
            t0 = time.time()

            # ── 1. Collect rollout ────────────────────────────────────────
            rollout_info = self._collect_rollout()
            global_step += cfg.n_steps

            # ── 2. PPO update ─────────────────────────────────────────────
            ppo_metrics = self.ppo.update(self.buffer)
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

            # ── 3. Periodic evaluation ────────────────────────────────────
            if update_idx % cfg.eval_interval == 0:
                eval_metrics = self._evaluate(update_idx)
                metrics["eval"] = eval_metrics

            # ── 4. Checkpoint ─────────────────────────────────────────────
            if update_idx % cfg.checkpoint_interval == 0:
                self._save_checkpoint(update_idx)
                self.reward_debugger.save()

            # ── 5. Log ────────────────────────────────────────────────────
            self._log(metrics)
            self._metrics_history.append(metrics)

            if self.callback is not None:
                self.callback(update_idx, metrics)

        # Final save
        self._save_checkpoint(total_updates)
        self._save_metrics()
        self.reward_debugger.save()
        print("Training complete.")

    # ──────────────────────────────────────────────────────────────────────
    # Rollout collection
    # ──────────────────────────────────────────────────────────────────────

    def _collect_rollout(self) -> Dict[str, float]:
        """Collect ``n_steps`` transitions into the rollout buffer."""
        # Pick a random training opponent for this rollout
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

        for _step in range(self.train_cfg.n_steps):
            scalars, troops, troop_mask, cards, action_masks = obs_to_tensors(
                obs, device=self.device,
            )

            output = self.model.act(
                scalars, troops, troop_mask, cards, action_masks, hidden,
            )

            # Store transition
            action_dict = {k: int(v.item()) for k, v in output.actions.items()}
            self.buffer.add(
                obs=obs,
                action=action_dict,
                log_prob=output.log_prob.item(),
                value=output.value.item(),
                reward=0.0,  # filled below
                done=False,  # filled below
                hidden=_detach(hidden),
            )

            hidden = output.hidden

            # Step environment
            gym_action = {k: int(v.item()) for k, v in output.actions.items()}
            next_obs, reward, terminated, truncated, info = env.step(gym_action)

            done = terminated or truncated
            self.buffer.rewards[-1] = float(reward)
            self.buffer.dones[-1] = done
            ep_reward += float(reward)

            # Feed reward debugger with per-step data
            self.reward_debugger.on_step(
                reward=float(reward),
                done=done,
                info=info,
            )

            obs = next_obs

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                n_episodes += 1
                obs, info = env.reset()
                hidden = self.model.init_hidden(batch_size=1)
                hidden = _to_device(hidden, self.device)

        # Bootstrap value for last state
        with torch.no_grad():
            s, tr, m, c, am = obs_to_tensors(obs, device=self.device)
            last_output = self.model.act(s, tr, m, c, am, hidden)
            last_value = last_output.value.item()

        self.buffer.finish(last_value)

        # Finalise reward debug snapshot for this rollout
        reward_snapshot = self.reward_debugger.finish_rollout()

        env.close()

        # Use cross-rollout episode reward if available, else report rollout sum
        cross_ep_reward = self.reward_debugger.get_mean_episode_reward()
        mean_ep_reward = float(np.mean(episode_rewards)) if episode_rewards else cross_ep_reward

        return {
            "mean_ep_reward": mean_ep_reward,
            "n_episodes": n_episodes,
            "opponent": opp.name,
            # New: actual per-step reward stats
            "rollout_reward_sum": reward_snapshot.total,
            "rollout_reward_mean": reward_snapshot.mean,
            "rollout_reward_abs_mean": reward_snapshot.abs_mean,
            "rollout_reward_nonzero_frac": reward_snapshot.nonzero_frac,
            "rollout_reward_min": reward_snapshot.min,
            "rollout_reward_max": reward_snapshot.max,
            "reward_components": reward_snapshot.component_sums,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation via league tournament
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate(self, update_idx: int) -> Dict[str, Any]:
        """Run a mini league tournament and return summary stats."""
        self.model.eval()

        agent = MohaNetAgent(
            name=f"MohaNet-u{update_idx}",
            model=self.model,
            device=str(self.device),
        )
        players: List[PlayerSlot] = [agent] + self.eval_opponents

        league = LeagueTournament(
            players=players,
            matches_per_pair=self.train_cfg.eval_matches_per_pair,
        )
        results = league.run()

        # Find our agent's stats
        agent_stats = results.get(agent.name)
        if agent_stats is not None:
            win_rate = agent_stats.wins / max(agent_stats.matches_played, 1)
            avg_crowns = agent_stats.crowns_scored / max(agent_stats.matches_played, 1)
        else:
            win_rate = 0.0
            avg_crowns = 0.0

        print(
            f"  [Eval u{update_idx}] Win rate: {win_rate:.1%}, "
            f"Avg crowns: {avg_crowns:.2f}"
        )

        return {
            "win_rate": win_rate,
            "avg_crowns": avg_crowns,
            "matches": agent_stats.matches_played if agent_stats else 0,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Checkpointing & logging
    # ──────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, update_idx: int) -> None:
        """Save model state dict."""
        path = self.log_dir / f"mohanet_u{update_idx}.pt"
        torch.save(self.model.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    def _save_metrics(self) -> None:
        """Dump training metrics to JSON."""
        path = self.log_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(self._metrics_history, f, indent=2, default=str)

    def _log(self, metrics: Dict[str, Any]) -> None:
        """Print a compact summary line with actual reward statistics."""
        u = metrics["update"]
        sps = metrics.get("sps", 0)
        pl = metrics.get("policy_loss", 0)
        vl = metrics.get("value_loss", 0)
        ent = metrics.get("entropy", 0)
        opp = metrics.get("opponent", "?")

        # Actual per-step reward stats (not episode-gated)
        r_sum = metrics.get("rollout_reward_sum", 0.0)
        r_mean = metrics.get("rollout_reward_mean", 0.0)
        r_abs = metrics.get("rollout_reward_abs_mean", 0.0)
        r_nz = metrics.get("rollout_reward_nonzero_frac", 0.0)
        n_eps = metrics.get("n_episodes", 0)
        mr = metrics.get("mean_ep_reward", 0.0)

        print(
            f"[u{u:4d}] sps={sps:.0f}  π={pl:.4f}  v={vl:.4f}  "
            f"H={ent:.3f}  ΣR={r_sum:+.3f}  μR={r_mean:+.5f}  "
            f"|R|={r_abs:.5f}  nz={r_nz:.0%}  ep={n_eps}  "
            f"epR={mr:+.3f}  vs={opp}"
        )

        # Print per-component breakdown every 5 updates
        components = metrics.get("reward_components", {})
        if components and u % 5 == 0:
            parts = "  ".join(f"{k}={v:+.4f}" for k, v in components.items())
            print(f"        [components] {parts}")

        # Print full diagnosis periodically (every 10 updates or first 3)
        if u <= 3 or u % 10 == 0:
            print(self.reward_debugger.diagnose(u))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LSTM hidden state utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _to_device(hidden: LSTMState, device: torch.device) -> LSTMState:
    return (hidden[0].to(device), hidden[1].to(device))


def _detach(hidden: LSTMState) -> LSTMState:
    return (hidden[0].detach().cpu(), hidden[1].detach().cpu())
