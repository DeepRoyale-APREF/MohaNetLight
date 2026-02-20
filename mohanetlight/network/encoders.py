"""Input encoders — scalar MLP, transformer entity encoder, card encoder.

Each encoder maps its raw input to a fixed-size vector of ``encoder_dim``
(default 128) so they can be concatenated before entering the LSTM core.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from mohanetlight.config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Scalar Encoder
# ═══════════════════════════════════════════════════════════════════════════════


class ScalarEncoder(nn.Module):
    """MLP encoder for the 16-dim scalar feature vector.

    Architecture::

        Linear(16 → 64) → ReLU → Linear(64 → 128) → ReLU

    Parameters
    ----------
    cfg : ModelConfig
        Architecture configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.scalar_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, cfg.encoder_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, scalars: Tensor) -> Tensor:
        """Encode scalar features.

        Parameters
        ----------
        scalars : Tensor
            Shape ``(B, 16)`` — normalised scalar features.

        Returns
        -------
        Tensor
            Shape ``(B, encoder_dim)``.
        """
        return self.mlp(scalars)


# ═══════════════════════════════════════════════════════════════════════════════
# Spatial Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════════


class SpatialPositionalEncoding(nn.Module):
    """Learned positional encoding from (tile_x, tile_y) coordinates.

    Projects the 2-D position into the entity model dimension and **adds**
    it to the entity feature embedding, giving the transformer spatial
    awareness of unit placement on the arena.

    Architecture::

        Linear(2 → 32) → ReLU → Linear(32 → entity_model_dim)
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, model_dim),
        )

    def forward(self, positions: Tensor) -> Tensor:
        """Compute positional encoding.

        Parameters
        ----------
        positions : Tensor
            Shape ``(B, N, 2)`` — normalised (tile_x / 17, tile_y / 31).

        Returns
        -------
        Tensor
            Shape ``(B, N, model_dim)`` — additive positional encoding.
        """
        return self.proj(positions)


# ═══════════════════════════════════════════════════════════════════════════════
# Entity (Troop) Encoder
# ═══════════════════════════════════════════════════════════════════════════════


class EntityEncoder(nn.Module):
    """Transformer-based encoder for the variable-length entity list.

    Architecture::

        Linear(14 → 64)          project raw features
        + SpatialPosEnc(x, y)    add spatial position embedding
        → TransformerEncoder(L=2, H=4, d=64, ff=256)
        → masked mean-pool       aggregate over real entities
        → Linear(64 → 128)       project to encoder_dim

    The transformer attends only to real entities (``troop_mask``);
    padding tokens are ignored via ``src_key_padding_mask``.

    Parameters
    ----------
    cfg : ModelConfig
        Architecture configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.entity_model_dim

        # Feature projection
        self.feature_proj = nn.Linear(cfg.troop_feature_dim, d)

        # Spatial positional encoding from (tile_x, tile_y)
        self.pos_enc = SpatialPositionalEncoding(d)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.entity_n_heads,
            dim_feedforward=cfg.entity_ff_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.entity_n_layers,
        )

        # Output projection
        self.out_proj = nn.Linear(d, cfg.encoder_dim)

    def forward(self, troops: Tensor, troop_mask: Tensor) -> Tensor:
        """Encode entity list with spatial-aware transformer.

        Parameters
        ----------
        troops : Tensor
            Shape ``(B, 100, 14)`` — padded entity features.
        troop_mask : Tensor
            Shape ``(B, 100)`` — ``True`` where a real entity exists.

        Returns
        -------
        Tensor
            Shape ``(B, encoder_dim)`` — aggregated entity representation.
        """
        B, N, _ = troops.shape

        # Project features
        x = self.feature_proj(troops)  # (B, N, d)

        # Add spatial positional encoding from tile_x (idx 4) and tile_y (idx 5)
        positions = troops[:, :, 4:6]  # already normalised in [0, 1]
        x = x + self.pos_enc(positions)

        # Transformer with padding mask (True = ignore in PyTorch convention)
        padding_mask = ~troop_mask.bool()  # True for pad tokens
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, N, d)

        # Masked mean-pool: average only over real entities
        mask_f = troop_mask.float().unsqueeze(-1)  # (B, N, 1)
        count = mask_f.sum(dim=1).clamp(min=1)     # (B, 1)
        pooled = (x * mask_f).sum(dim=1) / count    # (B, d)

        return self.out_proj(pooled)  # (B, encoder_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Card Encoder
# ═══════════════════════════════════════════════════════════════════════════════


class CardEncoder(nn.Module):
    """Encoder for the 8 deck cards.

    Architecture::

        Shared Linear(5 → 32) per card → concat 8 → (256) → Linear → ReLU → (128)

    Parameters
    ----------
    cfg : ModelConfig
        Architecture configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.card_hidden_dim  # 32

        # Shared projection per card
        self.card_proj = nn.Sequential(
            nn.Linear(cfg.card_feature_dim, h),
            nn.ReLU(inplace=True),
        )

        # Aggregate 8 cards → encoder_dim
        self.aggregate = nn.Sequential(
            nn.Linear(h * cfg.deck_size, cfg.encoder_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, cards: Tensor) -> Tensor:
        """Encode deck cards.

        Parameters
        ----------
        cards : Tensor
            Shape ``(B, 8, 5)`` — 8 deck cards × 5 features each.

        Returns
        -------
        Tensor
            Shape ``(B, encoder_dim)``.
        """
        B, H, _ = cards.shape
        per_card = self.card_proj(cards)  # (B, 8, 32)
        flat = per_card.reshape(B, -1)    # (B, 256)
        return self.aggregate(flat)        # (B, encoder_dim)
