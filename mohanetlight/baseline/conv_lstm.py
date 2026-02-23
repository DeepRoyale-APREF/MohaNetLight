"""ConvLSTM baseline — standard deep RL architecture.

Uses a CNN for the arena spatial map and an MLP for entity/scalar/card
features, concatenates them, and feeds through an LSTM for temporal
context.  Action heads are simple MLPs (no transformer entity encoder,
no FiLM-conditioned spatial decoder, no attention).

This represents a "standard" deep RL approach and sits between the
trivial FlatMLP and the full MohaNetLight in terms of architecture
sophistication.

Architecture::

    Arena:   Conv(8→32,3) → ReLU → Conv(32→64,3) → ReLU → GAP → (64)
    Troops:  flatten(100×14) → Linear(→256) → ReLU → (256)
    Scalars: Linear(16→64) → ReLU → (64)
    Cards:   Linear(8×5→64) → ReLU → (64)
    ──────────────────────────────────────────
    concat(64+256+64+64) = 448
    → LSTM(448, 256, layers=1)
    ├→ CardHead:    Linear(256→128) → ReLU → Linear(128→9)
    ├→ SpatialHead: Linear(256+embed→256) → ReLU → Linear(256→576)
    └→ ValueHead:   Linear(256→64) → ReLU → Linear(64→1)

Approximate parameter count: **≈ 0.7 M**.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mohanetlight.baseline.config import BaselineConfig
from mohanetlight.network.core import LSTMState
from mohanetlight.network.heads import HeadOutput, masked_categorical
from mohanetlight.network.mohanet import ModelOutput


class ConvLSTMNet(nn.Module):
    """CNN + LSTM actor-critic — standard deep RL baseline.

    Parameters
    ----------
    cfg : BaselineConfig | None
        Architecture config. Defaults to standard dimensions.
    """

    def __init__(self, cfg: BaselineConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = BaselineConfig()
        self.cfg = cfg

        # ── Arena CNN ────────────────────────────────────────────────────
        ch = cfg.cnn_channels
        self.arena_cnn = nn.Sequential(
            nn.Conv2d(cfg.arena_channels, ch[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        arena_out_dim = ch[-1]  # 64

        # ── Troop MLP ───────────────────────────────────────────────────
        troop_flat_dim = cfg.max_troops * cfg.troop_feature_dim
        self.troop_mlp = nn.Sequential(
            nn.Linear(troop_flat_dim, 256),
            nn.ReLU(inplace=True),
        )
        troop_out_dim = 256

        # ── Scalar MLP ──────────────────────────────────────────────────
        self.scalar_mlp = nn.Sequential(
            nn.Linear(cfg.scalar_dim, 64),
            nn.ReLU(inplace=True),
        )
        scalar_out_dim = 64

        # ── Card MLP ────────────────────────────────────────────────────
        card_flat_dim = cfg.deck_size * cfg.card_feature_dim
        self.card_mlp = nn.Sequential(
            nn.Linear(card_flat_dim, 64),
            nn.ReLU(inplace=True),
        )
        card_out_dim = 64

        # ── LSTM Core ───────────────────────────────────────────────────
        concat_dim = arena_out_dim + troop_out_dim + scalar_out_dim + card_out_dim
        self._concat_dim = concat_dim
        self._lstm_hidden = cfg.lstm_hidden_dim
        self._lstm_layers = cfg.lstm_layers

        self.lstm = nn.LSTM(
            input_size=concat_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        # ── Card head ───────────────────────────────────────────────────
        self.card_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.n_card_options),
        )

        # ── Card action embedding ───────────────────────────────────────
        self.card_embed = nn.Embedding(cfg.n_card_options, cfg.embedding_dim)

        # ── Spatial head ────────────────────────────────────────────────
        self.spatial_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim + cfg.embedding_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.n_position),
        )

        # ── Value head ──────────────────────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int = 1) -> LSTMState:
        """Return zero-initialised LSTM state."""
        h = torch.zeros(self._lstm_layers, batch_size, self._lstm_hidden)
        c = torch.zeros(self._lstm_layers, batch_size, self._lstm_hidden)
        return (h, c)

    def _encode(
        self,
        scalars: Tensor,
        troops: Tensor,
        troop_mask: Tensor,
        cards: Tensor,
        arena_map: Tensor,
    ) -> Tensor:
        """Encode all observation modalities → concat vector."""
        B = scalars.size(0)
        a = self.arena_cnn(arena_map)                           # (B, 64)
        masked_troops = troops * troop_mask.unsqueeze(-1).float()
        t = self.troop_mlp(masked_troops.reshape(B, -1))       # (B, 256)
        s = self.scalar_mlp(scalars)                             # (B, 64)
        c = self.card_mlp(cards.reshape(B, -1))                  # (B, 64)
        return torch.cat([a, t, s, c], dim=-1)                   # (B, concat)

    # ------------------------------------------------------------------
    # Training: evaluate given actions
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        scalars: Tensor,
        troops: Tensor,
        troop_mask: Tensor,
        cards: Tensor,
        arena_map: Tensor,
        action_masks: Dict[str, Tensor],
        actions: Dict[str, Tensor],
        hidden: LSTMState,
    ) -> Tuple[Tensor, Tensor, Tensor, LSTMState]:
        """Re-evaluate stored actions (PPO training path)."""
        enc = self._encode(scalars, troops, troop_mask, cards, arena_map)

        # LSTM — add time dim
        core_out, new_hidden = self.lstm(enc.unsqueeze(1), hidden)
        core_out = core_out.squeeze(1)  # (B, lstm_hidden)

        B = core_out.size(0)

        # Card
        card_logits = self.card_head(core_out)
        card_out = masked_categorical(
            card_logits, action_masks["card"], action=actions["card"],
        )
        card_emb = self.card_embed(actions["card"])

        # Spatial
        card_action = actions["card"]
        spatial_mask = action_masks["spatial_per_card"][
            torch.arange(B, device=card_action.device), card_action
        ]
        spatial_input = torch.cat([core_out, card_emb], dim=-1)
        pos_logits = self.spatial_head(spatial_input)
        position = actions["tile_y"] * self.cfg.n_tile_x + actions["tile_x"]
        pos_out = masked_categorical(pos_logits, spatial_mask, action=position)

        log_prob = card_out.log_prob + pos_out.log_prob
        entropy = card_out.entropy + pos_out.entropy
        value = self.value_head(core_out).squeeze(-1)

        return log_prob, value, entropy, new_hidden

    # ------------------------------------------------------------------
    # Inference: sample new actions
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        scalars: Tensor,
        troops: Tensor,
        troop_mask: Tensor,
        cards: Tensor,
        arena_map: Tensor,
        action_masks: Dict[str, Tensor],
        hidden: LSTMState,
    ) -> ModelOutput:
        """Sample actions autoregressively (no gradient)."""
        enc = self._encode(scalars, troops, troop_mask, cards, arena_map)

        core_out, new_hidden = self.lstm(enc.unsqueeze(1), hidden)
        core_out = core_out.squeeze(1)

        B = core_out.size(0)

        # Card
        card_logits = self.card_head(core_out)
        card_out = masked_categorical(card_logits, action_masks["card"])
        card_emb = self.card_embed(card_out.action)

        # Spatial
        card_action = card_out.action
        spatial_mask = action_masks["spatial_per_card"][
            torch.arange(B, device=card_action.device), card_action
        ]
        spatial_input = torch.cat([core_out, card_emb], dim=-1)
        pos_logits = self.spatial_head(spatial_input)
        pos_out = masked_categorical(pos_logits, spatial_mask)

        tile_y = pos_out.action // self.cfg.n_tile_x
        tile_x = pos_out.action % self.cfg.n_tile_x

        actions = {
            "card": card_out.action,
            "tile_x": tile_x,
            "tile_y": tile_y,
        }
        log_prob = card_out.log_prob + pos_out.log_prob
        entropy = card_out.entropy + pos_out.entropy
        value = self.value_head(core_out).squeeze(-1)

        return ModelOutput(
            actions=actions,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            hidden=new_hidden,
        )

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
