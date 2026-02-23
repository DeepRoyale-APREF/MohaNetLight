"""Rollout buffer — stores trajectories for on-policy PPO updates.

Handles LSTM hidden states and supports chunked iteration for
truncated BPTT during the PPO update phase.
"""

from __future__ import annotations

from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from mohanetlight.network.core import LSTMState


class RolloutBuffer:
    """Fixed-size buffer that stores one rollout of ``n_steps`` transitions.

    After calling :meth:`finish`, call :meth:`chunks` to iterate over
    sequential chunks for truncated BPTT.

    Parameters
    ----------
    n_steps : int
        Number of environment steps per rollout.
    gamma : float
        Discount factor for GAE.
    gae_lambda : float
        GAE lambda.
    """

    def __init__(
        self,
        n_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage lists (filled during rollout)
        self.obs_list: List[Dict[str, np.ndarray]] = []
        self.actions_list: List[Dict[str, int]] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.hidden_states: List[LSTMState] = []  # LSTM state BEFORE each step

        # Computed after rollout
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
        self._pos = 0

    @property
    def full(self) -> bool:
        """True when buffer has ``n_steps`` transitions."""
        return self._pos >= self.n_steps

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: Dict[str, int],
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        hidden: LSTMState,
    ) -> None:
        """Append one transition.

        Parameters
        ----------
        obs : dict
            Gymnasium observation dict (numpy arrays).
        action : dict
            Sampled actions ``{card, tile_x, tile_y}`` as ints.
        log_prob : float
            Sum of per-head log-probabilities.
        value : float
            Critic state-value estimate.
        reward : float
            Environment reward.
        done : bool
            Episode terminated or truncated.
        hidden : LSTMState
            LSTM hidden state *before* this step.
        """
        self.obs_list.append(obs)
        self.actions_list.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.hidden_states.append(hidden)
        self._pos += 1

    def finish(self, last_value: float) -> None:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_value : float
            V(s_{n_steps}) — bootstrap value for the state after the last
            stored transition.
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + values

    def chunks(
        self,
        chunk_len: int,
        device: torch.device | str = "cpu",
    ) -> Generator[Dict[str, Tensor | Dict[str, Tensor] | LSTMState], None, None]:
        """Yield sequential chunks for truncated BPTT.

        Each chunk is a dict with keys:

        - ``scalars``    ``(T, 16)``
        - ``troops``     ``(T, 100, 14)``
        - ``troop_mask`` ``(T, 100)``
        - ``cards``      ``(T, 4, 4)``
        - ``arena_map``  ``(T, 8, 32, 18)``
        - ``action_masks`` dict of ``(T, K)``
        - ``actions``    dict of ``(T,)``
        - ``old_log_probs`` ``(T,)``
        - ``advantages``  ``(T,)``
        - ``returns``     ``(T,)``
        - ``hidden``     initial LSTMState for this chunk

        Parameters
        ----------
        chunk_len : int
            Number of sequential steps per chunk.
        device : str | torch.device
            Target device.
        """
        assert self.advantages is not None, "Call finish() before chunks()"
        n = self._pos

        for start in range(0, n, chunk_len):
            end = min(start + chunk_len, n)
            sl = slice(start, end)
            T = end - start

            # Observations
            scalars = torch.tensor(
                np.stack([o["scalars"] for o in self.obs_list[sl]]),
                dtype=torch.float32, device=device,
            )
            troops = torch.tensor(
                np.stack([o["troops"] for o in self.obs_list[sl]]),
                dtype=torch.float32, device=device,
            )
            troop_mask = torch.tensor(
                np.stack([o["troop_mask"] for o in self.obs_list[sl]]),
                dtype=torch.bool, device=device,
            )
            cards = torch.tensor(
                np.stack([o["cards"] for o in self.obs_list[sl]]),
                dtype=torch.float32, device=device,
            )
            arena_map = torch.tensor(
                np.stack([o["arena_map"] for o in self.obs_list[sl]]),
                dtype=torch.float32, device=device,
            )

            # Action masks
            mask_keys = list(self.obs_list[0]["action_mask"].keys())
            action_masks: Dict[str, Tensor] = {}
            for k in mask_keys:
                action_masks[k] = torch.tensor(
                    np.stack([o["action_mask"][k] for o in self.obs_list[sl]]),
                    dtype=torch.bool, device=device,
                )

            # Actions
            act_keys = list(self.actions_list[0].keys())
            actions: Dict[str, Tensor] = {}
            for k in act_keys:
                actions[k] = torch.tensor(
                    [a[k] for a in self.actions_list[sl]],
                    dtype=torch.long, device=device,
                )

            # Scalars
            old_log_probs = torch.tensor(
                self.log_probs[start:end], dtype=torch.float32, device=device,
            )
            advantages_t = torch.tensor(
                self.advantages[start:end], dtype=torch.float32, device=device,
            )
            returns_t = torch.tensor(
                self.returns[start:end], dtype=torch.float32, device=device,
            )

            # LSTM hidden at chunk start
            hidden = self.hidden_states[start]
            hidden = (hidden[0].to(device), hidden[1].to(device))

            yield {
                "scalars": scalars,
                "troops": troops,
                "troop_mask": troop_mask,
                "cards": cards,
                "arena_map": arena_map,
                "action_masks": action_masks,
                "actions": actions,
                "old_log_probs": old_log_probs,
                "advantages": advantages_t,
                "returns": returns_t,
                "hidden": hidden,
                "dones": torch.tensor(
                    self.dones[start:end], dtype=torch.bool, device=device,
                ),
            }

    def reset(self) -> None:
        """Clear all stored data for the next rollout."""
        self.obs_list.clear()
        self.actions_list.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.hidden_states.clear()
        self.advantages = None
        self.returns = None
        self._pos = 0
