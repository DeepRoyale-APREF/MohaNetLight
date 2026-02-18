"""PPO trainer — clipped surrogate objective with truncated BPTT.

Implements Proximal Policy Optimisation (Schulman et al., 2017) adapted
for a recurrent actor-critic with hierarchical action heads.  The
aggregated log-prob is the sum of per-head log-probs; the clipping
and advantage estimation operate on this joint probability.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from mohanetlight.config import TrainingConfig
from mohanetlight.network.core import LSTMState
from mohanetlight.network.mohanet import MohaNetLight
from mohanetlight.training.rollout import RolloutBuffer


class PPOTrainer:
    """Stateless PPO update logic.

    Parameters
    ----------
    model : MohaNetLight
        The actor-critic network.
    cfg : TrainingConfig
        PPO hyperparameters.
    """

    def __init__(self, model: MohaNetLight, cfg: TrainingConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO update epochs on the collected rollout.

        Parameters
        ----------
        buffer : RolloutBuffer
            Filled buffer with computed advantages (call ``finish()`` first).

        Returns
        -------
        dict[str, float]
            Training metrics: ``policy_loss``, ``value_loss``,
            ``entropy``, ``approx_kl``, ``clip_fraction``.
        """
        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        n_chunks = 0

        for _epoch in range(self.cfg.n_epochs):
            for chunk in buffer.chunks(self.cfg.batch_chunk_len, device=self.device):
                metrics = self._update_chunk(chunk)
                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_entropy += metrics["entropy"]
                total_kl += metrics["approx_kl"]
                total_clip_frac += metrics["clip_fraction"]
                n_chunks += 1

        n_chunks = max(n_chunks, 1)
        return {
            "policy_loss": total_policy_loss / n_chunks,
            "value_loss": total_value_loss / n_chunks,
            "entropy": total_entropy / n_chunks,
            "approx_kl": total_kl / n_chunks,
            "clip_fraction": total_clip_frac / n_chunks,
        }

    def _update_chunk(self, chunk: Dict[str, Tensor | Dict[str, Tensor] | LSTMState]) -> Dict[str, float]:
        """Process one sequential chunk with truncated BPTT."""
        scalars = chunk["scalars"]  # (T, 16)
        troops = chunk["troops"]    # (T, 100, 14)
        troop_mask = chunk["troop_mask"]  # (T, 100)
        cards = chunk["cards"]      # (T, 4, 4)
        action_masks: Dict[str, Tensor] = chunk["action_masks"]  # type: ignore[assignment]
        actions: Dict[str, Tensor] = chunk["actions"]  # type: ignore[assignment]
        old_log_probs = chunk["old_log_probs"]  # (T,)
        advantages = chunk["advantages"]  # (T,)
        returns = chunk["returns"]  # (T,)
        hidden: LSTMState = chunk["hidden"]  # type: ignore[assignment]

        T = scalars.shape[0]

        # Normalise advantages (per chunk)
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # Forward through each timestep sequentially (truncated BPTT)
        log_probs_list = []
        values_list = []
        entropies_list = []

        h = hidden
        for t in range(T):
            # Slice single timestep — add batch dim
            s_t = scalars[t].unsqueeze(0)       # (1, 16)
            tr_t = troops[t].unsqueeze(0)       # (1, 100, 14)
            m_t = troop_mask[t].unsqueeze(0)    # (1, 100)
            c_t = cards[t].unsqueeze(0)         # (1, 4, 4)

            am_t = {k: v[t].unsqueeze(0) for k, v in action_masks.items()}
            a_t = {k: v[t].unsqueeze(0) for k, v in actions.items()}

            lp, val, ent, h = self.model.evaluate_actions(
                s_t, tr_t, m_t, c_t, am_t, a_t, h,
            )
            log_probs_list.append(lp.squeeze(0))
            values_list.append(val.squeeze(0))
            entropies_list.append(ent.squeeze(0))

        # Detach hidden for truncated BPTT (no gradient flows across chunks)
        # (hidden is already detached between chunks in the buffer)

        new_log_probs = torch.stack(log_probs_list)  # (T,)
        new_values = torch.stack(values_list)          # (T,)
        entropy = torch.stack(entropies_list)          # (T,)

        # ── PPO clipped surrogate loss ────────────────────────────────────
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value loss (clipped) ──────────────────────────────────────────
        value_loss = nn.functional.mse_loss(new_values, returns)

        # ── Entropy bonus ─────────────────────────────────────────────────
        entropy_loss = -entropy.mean()

        # ── Total loss ────────────────────────────────────────────────────
        loss = (
            policy_loss
            + self.cfg.vf_coef * value_loss
            + self.cfg.ent_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        # ── Metrics ───────────────────────────────────────────────────────
        with torch.no_grad():
            approx_kl = ((ratio - 1) - ratio.log()).mean().item()
            clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_frac,
        }
