"""FlatMLP baseline — simplest possible RL architecture.

All observation modalities are flattened and concatenated into a single
vector, then processed by a 3-layer MLP.  No recurrence, no CNN, no
transformer.  This is the lowest bar for comparison with MohaNetLight.

Architecture::

    flatten(scalars + troops + cards + arena_map) → (6_812)
    → Linear(→ 512) → ReLU → Linear(→ 256) → ReLU   (shared backbone)
    ├→ CardHead:   Linear(256 → 128) → ReLU → Linear(128 → 9)
    ├→ SpatialHead: Linear(256 + embed → 256) → ReLU → Linear(256 → 576)
    └→ ValueHead:  Linear(256 → 64) → ReLU → Linear(64 → 1)

**No LSTM** — uses a dummy hidden state for interface compatibility.

Approximate parameter count: **≈ 4 M** (most params from the flat input).
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


class FlatMLPNet(nn.Module):
    """Flat MLP actor-critic — no recurrence, no spatial reasoning.

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

        in_dim = cfg.flat_obs_dim

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Card head
        self.card_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.n_card_options),
        )

        # Card action embedding (autoregressive link)
        self.card_embed = nn.Sequential(
            nn.Embedding(cfg.n_card_options, cfg.embedding_dim),
        )

        # Spatial head: picks flat position conditioned on card choice
        self.spatial_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.embedding_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.n_position),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int = 1) -> LSTMState:
        """Return dummy hidden state (no LSTM — kept for interface compat)."""
        dummy = torch.zeros(1, batch_size, 1)
        return (dummy, dummy)

    def _flatten_obs(
        self,
        scalars: Tensor,
        troops: Tensor,
        troop_mask: Tensor,
        cards: Tensor,
        arena_map: Tensor,
    ) -> Tensor:
        """Flatten all observations into a single vector."""
        B = scalars.size(0)
        # Zero out padded troops for cleaner signal
        masked_troops = troops * troop_mask.unsqueeze(-1).float()
        return torch.cat([
            scalars,                            # (B, 16)
            masked_troops.reshape(B, -1),       # (B, 1400)
            cards.reshape(B, -1),               # (B, 40)
            arena_map.reshape(B, -1),           # (B, 4608)
        ], dim=-1)

    def _forward_shared(
        self,
        scalars: Tensor,
        troops: Tensor,
        troop_mask: Tensor,
        cards: Tensor,
        arena_map: Tensor,
    ) -> Tensor:
        flat = self._flatten_obs(scalars, troops, troop_mask, cards, arena_map)
        return self.backbone(flat)  # (B, hidden_dim)

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
        core = self._forward_shared(scalars, troops, troop_mask, cards, arena_map)
        B = core.size(0)

        # Card
        card_logits = self.card_head(core)
        card_out = masked_categorical(
            card_logits, action_masks["card"], action=actions["card"],
        )
        card_emb = self.card_embed[0](actions["card"])  # (B, embed_dim)

        # Spatial
        card_action = actions["card"]
        spatial_mask = action_masks["spatial_per_card"][
            torch.arange(B, device=card_action.device), card_action
        ]
        spatial_input = torch.cat([core, card_emb], dim=-1)
        pos_logits = self.spatial_head(spatial_input)
        position = actions["tile_y"] * self.cfg.n_tile_x + actions["tile_x"]
        pos_out = masked_categorical(pos_logits, spatial_mask, action=position)

        log_prob = card_out.log_prob + pos_out.log_prob
        entropy = card_out.entropy + pos_out.entropy
        value = self.value_head(core).squeeze(-1)

        return log_prob, value, entropy, hidden

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
        """Sample actions (no gradient)."""
        core = self._forward_shared(scalars, troops, troop_mask, cards, arena_map)
        B = core.size(0)

        # Card
        card_logits = self.card_head(core)
        card_out = masked_categorical(card_logits, action_masks["card"])
        card_emb = self.card_embed[0](card_out.action)

        # Spatial
        card_action = card_out.action
        spatial_mask = action_masks["spatial_per_card"][
            torch.arange(B, device=card_action.device), card_action
        ]
        spatial_input = torch.cat([core, card_emb], dim=-1)
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
        value = self.value_head(core).squeeze(-1)

        return ModelOutput(
            actions=actions,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            hidden=hidden,  # pass-through (no LSTM)
        )

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
