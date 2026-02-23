"""Hierarchical action heads, spatial decoder, and value head.

Autoregressive heads: the card head's *sampled* output is embedded and
fed as conditioning to the spatial decoder which selects a 2-D position.

Connections
-----------
- **Card**    ← core + entity_context
- **Spatial** ← CNN feature maps + FiLM(core, card_embed)
- **Value**   ← core  (critic, no action conditioning)
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

from mohanetlight.config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Head output container
# ═══════════════════════════════════════════════════════════════════════════════


class HeadOutput(NamedTuple):
    """Sampled action, log-probability, and entropy from one head."""

    action: Tensor      # (B,) int64
    log_prob: Tensor    # (B,)
    entropy: Tensor     # (B,)


# ═══════════════════════════════════════════════════════════════════════════════
# Masked categorical sampling
# ═══════════════════════════════════════════════════════════════════════════════


def masked_categorical(
    logits: Tensor,
    mask: Tensor,
    action: Optional[Tensor] = None,
) -> HeadOutput:
    """Sample from a masked categorical distribution.

    Parameters
    ----------
    logits : Tensor
        Raw logits of shape ``(B, K)``.
    mask : Tensor
        Boolean mask ``(B, K)`` — ``True`` = valid action.
    action : Tensor, optional
        If given, evaluate this action instead of sampling.

    Returns
    -------
    HeadOutput
        Sampled (or given) action, its log-prob, and entropy.
    """
    # Mask invalid actions by setting logits to -inf
    masked_logits = logits.masked_fill(~mask.bool(), float("-inf"))
    dist = Categorical(logits=masked_logits)

    if action is None:
        action = dist.sample()

    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return HeadOutput(action=action, log_prob=log_prob, entropy=entropy)


# ═══════════════════════════════════════════════════════════════════════════════
# Action embedding (used between heads)
# ═══════════════════════════════════════════════════════════════════════════════


class ActionEmbedding(nn.Module):
    """Embed a discrete action index and project for the next head.

    Architecture::

        Embedding(n_actions, embed_dim) → Linear(embed_dim, proj_dim) → ReLU
    """

    def __init__(self, n_actions: int, embed_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_actions, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, action: Tensor) -> Tensor:
        """Embed action index.

        Parameters
        ----------
        action : Tensor
            Shape ``(B,)`` int64 — sampled action index.

        Returns
        -------
        Tensor
            Shape ``(B, proj_dim)``.
        """
        return self.proj(self.embed(action))


# ═══════════════════════════════════════════════════════════════════════════════
# ① Card Head
# ═══════════════════════════════════════════════════════════════════════════════


class CardHead(nn.Module):
    """MLP head for card selection.

    Input: concat(core, entity_context)
    dim  = 256 + 128 = 384

    Output: logits ``(B, 9)`` → deck slot 0-7 or NOOP (8).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_dim = cfg.card_head_input_dim  # 256 + 128
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.head_hidden_dim, cfg.n_card_options),
        )

    def forward(self, core: Tensor, entity_ctx: Tensor) -> Tensor:
        """Compute card logits.

        Parameters
        ----------
        core : Tensor
            ``(B, 256)`` LSTM output.
        entity_ctx : Tensor
            ``(B, 128)`` entity encoder output.

        Returns
        -------
        Tensor
            ``(B, 9)`` raw logits.
        """
        x = torch.cat([core, entity_ctx], dim=-1)
        return self.mlp(x)


# ═══════════════════════════════════════════════════════════════════════════════
# ② Spatial Decoder (ResNet-style — replaces TileX + TileY heads)
# ═══════════════════════════════════════════════════════════════════════════════


class _ResBlock2d(nn.Module):
    """Simple 2-D residual block: Conv→BN→ReLU→Conv→BN + skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.block(x) + x)


class SpatialDecoder(nn.Module):
    """ResNet-based spatial decoder that outputs per-tile logits.

    Receives CNN feature maps from the ArenaEncoder and conditions them
    on the LSTM core output + selected-card embedding via FiLM
    (Feature-wise Linear Modulation).

    Architecture::

        FiLM conditioning: Linear(core + card_emb → 2 × C_feat) → γ, β
        feat_maps = γ * feat_maps + β          (B, 64, 32, 18)
        → ResBlock(64)
        → Conv(64 → 32, 3, pad=1) + BN + ReLU
        → ResBlock(32)
        → Conv(32 → 1, 1)                     (B, 1, 32, 18)
        → flatten → (B, 576)

    Parameters
    ----------
    cfg : ModelConfig
        Architecture configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        C = cfg.arena_cnn_dim  # 64

        # FiLM conditioning: project (core + card_emb) → γ, β
        self.film = nn.Linear(cfg.spatial_condition_dim, C * 2)

        # Decoder body
        self.res1 = _ResBlock2d(C)
        ch2 = cfg.spatial_decoder_channels[-1]  # 32
        self.down = nn.Sequential(
            nn.Conv2d(C, ch2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
        )
        self.res2 = _ResBlock2d(ch2)
        self.out_conv = nn.Conv2d(ch2, 1, kernel_size=1)

    def forward(
        self,
        feat_maps: Tensor,
        core: Tensor,
        card_emb: Tensor,
    ) -> Tensor:
        """Produce per-tile logits for position selection.

        Parameters
        ----------
        feat_maps : Tensor
            ``(B, C_feat, H, W)`` from ArenaEncoder.
        core : Tensor
            ``(B, 256)`` LSTM core output.
        card_emb : Tensor
            ``(B, 64)`` embedded card action.

        Returns
        -------
        Tensor
            ``(B, H*W)`` flat spatial logits (576 = 32 × 18).
        """
        B, C, H, W = feat_maps.shape

        # FiLM conditioning
        cond = torch.cat([core, card_emb], dim=-1)  # (B, 320)
        film_params = self.film(cond)                 # (B, 2*C)
        gamma = film_params[:, :C].view(B, C, 1, 1)
        beta = film_params[:, C:].view(B, C, 1, 1)
        x = gamma * feat_maps + beta  # (B, C, H, W)

        # Decoder
        x = self.res1(x)       # (B, 64, H, W)
        x = self.down(x)       # (B, 32, H, W)
        x = self.res2(x)       # (B, 32, H, W)
        x = self.out_conv(x)   # (B, 1, H, W)

        return x.view(B, -1)   # (B, H*W)


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy Tile Heads (kept for backward compatibility / reference)
# ═══════════════════════════════════════════════════════════════════════════════


class TileXHead(nn.Module):
    """MLP head for tile column selection.

    Input: concat(core, card_embed, entity_context) = 448
    Output: logits ``(B, 18)``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_dim = cfg.head_input_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.head_hidden_dim, cfg.n_tile_x),
        )

    def forward(self, core: Tensor, prev_embed: Tensor, entity_ctx: Tensor) -> Tensor:
        """Compute tile_x logits ``(B, 18)``."""
        x = torch.cat([core, prev_embed, entity_ctx], dim=-1)
        return self.mlp(x)


# ═══════════════════════════════════════════════════════════════════════════════
# ③ Tile Y Head
# ═══════════════════════════════════════════════════════════════════════════════


class TileYHead(nn.Module):
    """MLP head for tile row selection.

    Input: concat(core, tile_x_embed, entity_context) = 448
    Output: logits ``(B, 32)``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_dim = cfg.head_input_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.head_hidden_dim, cfg.n_tile_y),
        )

    def forward(self, core: Tensor, prev_embed: Tensor, entity_ctx: Tensor) -> Tensor:
        """Compute tile_y logits ``(B, 32)``."""
        x = torch.cat([core, prev_embed, entity_ctx], dim=-1)
        return self.mlp(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Value Head (Critic)
# ═══════════════════════════════════════════════════════════════════════════════


class ValueHead(nn.Module):
    """Critic value head — estimates V(s) from LSTM core output.

    Architecture::

        Linear(256 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 1)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.lstm_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(h, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, core: Tensor) -> Tensor:
        """Compute state value.

        Parameters
        ----------
        core : Tensor
            ``(B, 256)`` LSTM output.

        Returns
        -------
        Tensor
            ``(B,)`` scalar value estimate.
        """
        return self.mlp(core).squeeze(-1)
