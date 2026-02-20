"""MohaNetLight — assembled AlphaStar-inspired actor-critic network.

Dataflow
--------
1. Encode inputs in parallel:  ScalarEncoder, EntityEncoder, CardEncoder
2. Concatenate → 384-dim  →  LSTMCore  →  256-dim core output
3. Hierarchical action heads (autoregressive):
   strategy  →  card  →  tile_x  →  tile_y
4. Value head (critic) reads core output only.

The network exposes two main entry points:

* ``forward()``  — training: given stored actions, returns log-probs / values / entropy.
* ``act()``      — inference: sample new actions, return dict + log-prob + value.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mohanetlight.config import ModelConfig
from mohanetlight.network.core import LSTMCore, LSTMState
from mohanetlight.network.encoders import CardEncoder, EntityEncoder, ScalarEncoder
from mohanetlight.network.heads import (
    ActionEmbedding,
    CardHead,
    HeadOutput,
    StrategyHead,
    TileXHead,
    TileYHead,
    ValueHead,
    masked_categorical,
)


class ModelOutput(NamedTuple):
    """Bundled model outputs for training / inference."""

    actions: Dict[str, Tensor]      # {strategy, card, tile_x, tile_y} each (B,)
    log_prob: Tensor                # (B,) sum of per-head log-probs
    value: Tensor                   # (B,) state-value estimate
    entropy: Tensor                 # (B,) sum of per-head entropies
    hidden: LSTMState              # new LSTM hidden state


class MohaNetLight(nn.Module):
    """Lightweight AlphaStar-inspired actor-critic for Clash Royale.

    Parameters
    ----------
    cfg : ModelConfig
        Frozen dataclass with all architecture dimensions.

    Approximate parameter count: **≈ 1.6 M** — trainable on Colab free GPU.
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

        # ── LSTM Core ────────────────────────────────────────────────────
        self.core = LSTMCore(cfg)

        # ── Hierarchical Action Heads ────────────────────────────────────
        self.strategy_head = StrategyHead(cfg)
        self.card_head = CardHead(cfg)
        self.tile_x_head = TileXHead(cfg)
        self.tile_y_head = TileYHead(cfg)

        # ── Action Embeddings (autoregressive links) ─────────────────────
        self.strategy_embed = ActionEmbedding(
            cfg.n_strategies, cfg.embedding_dim, cfg.embedding_proj_dim,
        )
        self.card_embed = ActionEmbedding(
            cfg.n_card_options, cfg.embedding_dim, cfg.embedding_proj_dim,
        )
        self.tile_x_embed = ActionEmbedding(
            cfg.n_tile_x, cfg.embedding_dim, cfg.embedding_proj_dim,
        )

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
    ) -> Tuple[Tensor, Tensor]:
        """Run the three input encoders.

        Returns
        -------
        concat : Tensor
            ``(B, 384)`` concatenated encoder outputs.
        entity_ctx : Tensor
            ``(B, 128)`` entity context for head skip connections.
        """
        s = self.scalar_enc(scalars)           # (B, 128)
        e = self.entity_enc(troops, troop_mask)  # (B, 128)
        c = self.card_enc(cards)               # (B, 128)
        concat = torch.cat([s, e, c], dim=-1)  # (B, 384)
        return concat, e

    # ------------------------------------------------------------------
    # Training: evaluate given actions
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        scalars: Tensor,
        troops: Tensor,
        troop_mask: Tensor,
        cards: Tensor,
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
        action_masks : Dict[str, Tensor]
            ``strategy (B,3)  card (B,9)  tile_x_per_card (B,9,18)  tile_y_per_card (B,9,32)``
        actions : Dict[str, Tensor]
            ``strategy (B,)  card (B,)  tile_x (B,)  tile_y (B,)``
        hidden : LSTMState
            LSTM hidden state at the start of this step.

        Returns
        -------
        log_prob : Tensor ``(B,)``
        value : Tensor ``(B,)``
        entropy : Tensor ``(B,)``
        new_hidden : LSTMState
        """
        concat, entity_ctx = self._encode(scalars, troops, troop_mask, cards)

        # LSTM forward — add time dimension T=1
        core_out, new_hidden = self.core(concat.unsqueeze(1), hidden)
        core_out = core_out.squeeze(1)  # (B, 256)

        # ── Strategy ─────────────────────────────────────────────────────
        strat_logits = self.strategy_head(core_out)
        strat_out = masked_categorical(
            strat_logits, action_masks["strategy"], action=actions["strategy"],
        )
        strat_emb = self.strategy_embed(actions["strategy"])

        # ── Card ─────────────────────────────────────────────────────────
        card_logits = self.card_head(core_out, strat_emb, entity_ctx)
        card_out = masked_categorical(
            card_logits, action_masks["card"], action=actions["card"],
        )
        card_emb = self.card_embed(actions["card"])

        # ── Tile X (per-card mask) ───────────────────────────────────────
        B = core_out.size(0)
        card_action = actions["card"]
        tx_mask = action_masks["tile_x_per_card"][torch.arange(B, device=card_action.device), card_action]
        tx_logits = self.tile_x_head(core_out, card_emb, entity_ctx)
        tx_out = masked_categorical(
            tx_logits, tx_mask, action=actions["tile_x"],
        )
        tx_emb = self.tile_x_embed(actions["tile_x"])

        # ── Tile Y (per-card mask) ───────────────────────────────────────
        ty_mask = action_masks["tile_y_per_card"][torch.arange(B, device=card_action.device), card_action]
        ty_logits = self.tile_y_head(core_out, tx_emb, entity_ctx)
        ty_out = masked_categorical(
            ty_logits, ty_mask, action=actions["tile_y"],
        )

        # ── Aggregated log-prob & entropy ────────────────────────────────
        log_prob = (
            strat_out.log_prob + card_out.log_prob + tx_out.log_prob + ty_out.log_prob
        )
        entropy = (
            strat_out.entropy + card_out.entropy + tx_out.entropy + ty_out.entropy
        )

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
        action_masks : Dict[str, Tensor]
        hidden : LSTMState

        Returns
        -------
        ModelOutput
            Named tuple with ``actions, log_prob, value, entropy, hidden``.
        """
        concat, entity_ctx = self._encode(scalars, troops, troop_mask, cards)

        core_out, new_hidden = self.core(concat.unsqueeze(1), hidden)
        core_out = core_out.squeeze(1)

        B = core_out.size(0)

        # Autoregressive sampling
        strat_logits = self.strategy_head(core_out)
        strat_out = masked_categorical(strat_logits, action_masks["strategy"])
        strat_emb = self.strategy_embed(strat_out.action)

        card_logits = self.card_head(core_out, strat_emb, entity_ctx)
        card_out = masked_categorical(card_logits, action_masks["card"])
        card_emb = self.card_embed(card_out.action)

        # Per-card tile masks
        card_action = card_out.action
        tx_mask = action_masks["tile_x_per_card"][torch.arange(B, device=card_action.device), card_action]
        tx_logits = self.tile_x_head(core_out, card_emb, entity_ctx)
        tx_out = masked_categorical(tx_logits, tx_mask)
        tx_emb = self.tile_x_embed(tx_out.action)

        ty_mask = action_masks["tile_y_per_card"][torch.arange(B, device=card_action.device), card_action]
        ty_logits = self.tile_y_head(core_out, tx_emb, entity_ctx)
        ty_out = masked_categorical(ty_logits, ty_mask)

        actions = {
            "strategy": strat_out.action,
            "card": card_out.action,
            "tile_x": tx_out.action,
            "tile_y": ty_out.action,
        }
        log_prob = (
            strat_out.log_prob + card_out.log_prob + tx_out.log_prob + ty_out.log_prob
        )
        entropy = (
            strat_out.entropy + card_out.entropy + tx_out.entropy + ty_out.entropy
        )
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
