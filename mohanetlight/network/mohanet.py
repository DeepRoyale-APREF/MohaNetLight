"""MohaNetLight — assembled actor-critic network with spatial CNN.

Dataflow
--------
1. Encode inputs in parallel:
   ScalarEncoder, EntityEncoder, CardEncoder, **ArenaEncoder** (CNN)
2. Concatenate → 512-dim  →  LSTMCore  →  256-dim core output
3. Hierarchical action heads (autoregressive):
   card  →  **SpatialDecoder** (ResNet) → position (tile_x, tile_y)
4. Value head (critic) reads core output only.

The ArenaEncoder additionally produces full-resolution feature maps
that are passed to the SpatialDecoder for position selection.

The network exposes two main entry points:

* ``evaluate_actions()``  — training: given stored actions, returns log-probs / values / entropy.
* ``act()``               — inference: sample new actions, return dict + log-prob + value.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mohanetlight.config import ModelConfig
from mohanetlight.network.core import LSTMCore, LSTMState
from mohanetlight.network.encoders import (
    ArenaEncoder,
    CardEncoder,
    EntityEncoder,
    ScalarEncoder,
)
from mohanetlight.network.heads import (
    ActionEmbedding,
    CardHead,
    HeadOutput,
    SpatialDecoder,
    ValueHead,
    masked_categorical,
)


class ModelOutput(NamedTuple):
    """Bundled model outputs for training / inference."""

    actions: Dict[str, Tensor]      # {card, tile_x, tile_y} each (B,)
    log_prob: Tensor                # (B,) sum of per-head log-probs
    value: Tensor                   # (B,) state-value estimate
    entropy: Tensor                 # (B,) sum of per-head entropies
    hidden: LSTMState              # new LSTM hidden state


class MohaNetLight(nn.Module):
    """Lightweight AlphaStar-inspired actor-critic for Clash Royale.

    Features a CNN arena encoder and a ResNet spatial decoder for
    position selection, replacing the old TileX / TileY MLP heads.

    Parameters
    ----------
    cfg : ModelConfig
        Frozen dataclass with all architecture dimensions.

    Approximate parameter count: **≈ 1.8 M** — trainable on Colab free GPU.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # ── Encoders ─────────────────────────────────────────────────────
        self.scalar_enc = ScalarEncoder(cfg)
        self.entity_enc = EntityEncoder(cfg)
        self.card_enc = CardEncoder(cfg)
        self.arena_enc = ArenaEncoder(cfg)

        # ── LSTM Core ────────────────────────────────────────────────────
        self.core = LSTMCore(cfg)

        # ── Card Head ────────────────────────────────────────────────────
        self.card_head = CardHead(cfg)

        # ── Card Action Embedding (autoregressive link to spatial decoder)
        self.card_embed = ActionEmbedding(
            cfg.n_card_options, cfg.embedding_dim, cfg.embedding_proj_dim,
        )

        # ── Spatial Decoder (ResNet) — replaces TileX + TileY heads ─────
        self.spatial_decoder = SpatialDecoder(cfg)

        # ── Value Head (critic) ──────────────────────────────────────────
        self.value_head = ValueHead(cfg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int = 1) -> LSTMState:
        """Return zero-initialized LSTM hidden state.

        Parameters
        ----------
        batch_size : int
            Number of parallel environments (usually 1).

        Returns
        -------
        LSTMState
            Tuple ``(h_0, c_0)`` each ``(layers, B, hidden)``.
        """
        return self.core.init_hidden(batch_size)

    def _encode(
        self,
        scalars: Tensor,
        troops: Tensor,
        troop_mask: Tensor,
        cards: Tensor,
        arena_map: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run the four input encoders.

        Returns
        -------
        concat : Tensor
            ``(B, 512)`` concatenated encoder outputs.
        entity_ctx : Tensor
            ``(B, 128)`` entity context for card head skip connection.
        arena_features : Tensor
            ``(B, C_feat, H, W)`` spatial features for the SpatialDecoder.
        """
        s = self.scalar_enc(scalars)                      # (B, 128)
        e = self.entity_enc(troops, troop_mask)            # (B, 128)
        c = self.card_enc(cards)                           # (B, 128)
        a_emb, arena_features = self.arena_enc(arena_map)  # (B, 128), (B, 64, H, W)
        concat = torch.cat([s, e, c, a_emb], dim=-1)      # (B, 512)
        return concat, e, arena_features

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
        """Re-evaluate stored actions to get fresh log-probs, values, entropy.

        This is the path used inside the PPO loss loop.

        Parameters
        ----------
        scalars : Tensor  ``(B, 16)``
        troops : Tensor   ``(B, 100, 14)``
        troop_mask : Tensor ``(B, 100)``
        cards : Tensor     ``(B, 8, 5)``
        arena_map : Tensor ``(B, 8, 32, 18)``
        action_masks : Dict[str, Tensor]
            ``card (B,9)  spatial_per_card (B,9,576)``
        actions : Dict[str, Tensor]
            ``card (B,)  tile_x (B,)  tile_y (B,)``
        hidden : LSTMState
            LSTM hidden state at the start of this step.

        Returns
        -------
        log_prob : Tensor ``(B,)``
        value : Tensor ``(B,)``
        entropy : Tensor ``(B,)``
        new_hidden : LSTMState
        """
        concat, entity_ctx, arena_features = self._encode(
            scalars, troops, troop_mask, cards, arena_map,
        )

        # LSTM forward — add time dimension T=1
        core_out, new_hidden = self.core(concat.unsqueeze(1), hidden)
        core_out = core_out.squeeze(1)  # (B, 256)

        # ── Card ─────────────────────────────────────────────────────────
        card_logits = self.card_head(core_out, entity_ctx)
        card_out = masked_categorical(
            card_logits, action_masks["card"], action=actions["card"],
        )
        card_emb = self.card_embed(actions["card"])

        # ── Spatial position ─────────────────────────────────────────────
        B = core_out.size(0)
        card_action = actions["card"]

        # Per-card spatial mask: (B, 576)
        spatial_mask = action_masks["spatial_per_card"][
            torch.arange(B, device=card_action.device), card_action
        ]

        # Spatial decoder logits
        pos_logits = self.spatial_decoder(arena_features, core_out, card_emb)

        # Convert stored tile_x/tile_y to flat position for evaluation
        position = actions["tile_y"] * self.cfg.n_tile_x + actions["tile_x"]
        pos_out = masked_categorical(pos_logits, spatial_mask, action=position)

        # ── Aggregated log-prob & entropy ────────────────────────────────
        log_prob = card_out.log_prob + pos_out.log_prob
        entropy = card_out.entropy + pos_out.entropy

        value = self.value_head(core_out)

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
        """Sample actions autoregressively (no gradient).

        Parameters
        ----------
        scalars : Tensor  ``(B, 16)``
        troops : Tensor   ``(B, 100, 14)``
        troop_mask : Tensor ``(B, 100)``
        cards : Tensor     ``(B, 8, 5)``
        arena_map : Tensor ``(B, 8, 32, 18)``
        action_masks : Dict[str, Tensor]
        hidden : LSTMState

        Returns
        -------
        ModelOutput
            Named tuple with ``actions, log_prob, value, entropy, hidden``.
            actions contains ``{card, tile_x, tile_y}`` for engine compat.
        """
        concat, entity_ctx, arena_features = self._encode(
            scalars, troops, troop_mask, cards, arena_map,
        )

        core_out, new_hidden = self.core(concat.unsqueeze(1), hidden)
        core_out = core_out.squeeze(1)

        B = core_out.size(0)

        # ── Card ─────────────────────────────────────────────────────────
        card_logits = self.card_head(core_out, entity_ctx)
        card_out = masked_categorical(card_logits, action_masks["card"])
        card_emb = self.card_embed(card_out.action)

        # ── Spatial position ─────────────────────────────────────────────
        card_action = card_out.action
        spatial_mask = action_masks["spatial_per_card"][
            torch.arange(B, device=card_action.device), card_action
        ]
        pos_logits = self.spatial_decoder(arena_features, core_out, card_emb)
        pos_out = masked_categorical(pos_logits, spatial_mask)

        # Convert flat position → tile_x, tile_y
        tile_y = pos_out.action // self.cfg.n_tile_x
        tile_x = pos_out.action % self.cfg.n_tile_x

        actions = {
            "card": card_out.action,
            "tile_x": tile_x,
            "tile_y": tile_y,
        }
        log_prob = card_out.log_prob + pos_out.log_prob
        entropy = card_out.entropy + pos_out.entropy
        value = self.value_head(core_out)

        return ModelOutput(
            actions=actions,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            hidden=new_hidden,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
