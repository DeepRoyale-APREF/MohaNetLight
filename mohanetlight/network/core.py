"""Deep LSTM core — temporal integration of encoder outputs."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mohanetlight.config import ModelConfig

LSTMState = Tuple[Tensor, Tensor]  # (h_n, c_n)


class LSTMCore(nn.Module):
    """Two-layer LSTM that integrates scalar, entity, and card encodings.

    Carries hidden state across time-steps within an episode, giving the
    agent temporal context (past troop movements, elixir trends, etc.).

    Architecture::

        LSTM(input=384, hidden=256, layers=2, dropout=0.1)

    Parameters
    ----------
    cfg : ModelConfig
        Architecture configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self._hidden_dim = cfg.lstm_hidden_dim
        self._n_layers = cfg.lstm_layers

        self.lstm = nn.LSTM(
            input_size=cfg.concat_encoder_dim,  # 384
            hidden_size=cfg.lstm_hidden_dim,     # 256
            num_layers=cfg.lstm_layers,          # 2
            batch_first=True,
            dropout=0.1 if cfg.lstm_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: Tensor,
        hidden: Optional[LSTMState] = None,
    ) -> Tuple[Tensor, LSTMState]:
        """Forward through LSTM.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, 384)`` — concatenated encoder outputs
            over ``T`` time-steps (``T=1`` during rollout collection).
        hidden : tuple[Tensor, Tensor], optional
            Previous ``(h_n, c_n)`` each of shape
            ``(n_layers, B, hidden_dim)``.  Zeros if ``None``.

        Returns
        -------
        output : Tensor
            Shape ``(B, T, 256)`` — LSTM output at each step.
        hidden : tuple[Tensor, Tensor]
            Updated ``(h_n, c_n)``.
        """
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, batch_size: int = 1, device: torch.device | str = "cpu") -> LSTMState:
        """Return zero-initialised hidden state.

        Parameters
        ----------
        batch_size : int
            Batch dimension.
        device : torch.device or str
            Target device.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(h_0, c_0)`` each of shape ``(n_layers, B, hidden_dim)``.
        """
        zeros = torch.zeros(self._n_layers, batch_size, self._hidden_dim, device=device)
        return (zeros.clone(), zeros.clone())
