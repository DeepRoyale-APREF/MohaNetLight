"""Input encoders — scalar MLP, transformer entity encoder, card encoder, arena CNN.

Each encoder maps its raw input to a fixed-size vector of ``encoder_dim``
(default 128) so they can be concatenated before entering the LSTM core.
The arena CNN additionally produces spatial feature maps used by the
spatial decoder head.
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


# ═══════════════════════════════════════════════════════════════════════════════
# Arena Spatial Encoder (CNN)
# ═══════════════════════════════════════════════════════════════════════════════


class ArenaEncoder(nn.Module):
    """CNN encoder for the 2-D arena spatial map.

    Produces two outputs:

    1. A global embedding ``(B, encoder_dim)`` for the LSTM core
       (via global average pooling).
    2. Spatial feature maps ``(B, C_feat, H, W)`` preserved at input
       resolution for use by the SpatialDecoder.

    Architecture::

        Conv(8→32, 3, pad=1) + BN + ReLU
        Conv(32→64, 3, pad=1) + BN + ReLU
        Conv(64→64, 3, pad=1) + BN + ReLU  (residual)
        ── global avg pool → Linear(64→128) → encoder embedding
        ── feature maps (B, 64, 32, 18) → spatial decoder

    Parameters
    ----------
    cfg : ModelConfig
        Architecture configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        ch = cfg.arena_cnn_channels  # (32, 64, 64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(cfg.arena_channels, ch[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
        )
        # Residual block at ch[2] (same as ch[1] for skip connection)
        self.conv3a = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
        )
        self.conv3b = nn.Sequential(
            nn.Conv2d(ch[2], ch[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[2]),
        )
        self.relu = nn.ReLU(inplace=True)

        # Global embedding projection
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch[2], cfg.encoder_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, arena_map: Tensor) -> tuple[Tensor, Tensor]:
        """Encode arena spatial map.

        Parameters
        ----------
        arena_map : Tensor
            Shape ``(B, C, H, W)`` — spatial arena channels.

        Returns
        -------
        embedding : Tensor
            Shape ``(B, encoder_dim)`` — global arena summary for LSTM.
        feature_maps : Tensor
            Shape ``(B, C_feat, H, W)`` — full-resolution features
            for the SpatialDecoder.
        """
        x = self.conv1(arena_map)    # (B, 32, H, W)
        x = self.conv2(x)           # (B, 64, H, W)

        # Residual block
        identity = x
        x = self.conv3a(x)          # (B, 64, H, W)
        x = self.conv3b(x)          # (B, 64, H, W)
        x = self.relu(x + identity) # (B, 64, H, W)

        embedding = self.global_proj(x)  # (B, encoder_dim)
        return embedding, x
