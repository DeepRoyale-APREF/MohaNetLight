"""Hyperparameter configuration for baseline models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    """Architecture hyperparameters shared by baseline models.

    These mirror the observation / action dimensions from
    :class:`mohanetlight.config.ModelConfig` but use simpler internals.

    Parameters
    ----------
    scalar_dim : int
        Scalar feature dimension.
    troop_feature_dim : int
        Per-entity feature dimension.
    card_feature_dim : int
        Per-card feature dimension.
    max_troops : int
        Maximum entities in the observation.
    deck_size : int
        Cards in the deck.
    arena_channels : int
        Arena spatial map input channels.
    arena_h, arena_w : int
        Arena tile dimensions.
    n_card_options : int
        Card head output size (8 deck + 1 noop).
    n_tile_x, n_tile_y : int
        Tile grid dimensions.
    hidden_dim : int
        MLP hidden layer width.
    lstm_hidden_dim : int
        LSTM hidden size (ConvLSTM only).
    lstm_layers : int
        Number of LSTM layers (ConvLSTM only).
    cnn_channels : tuple[int, ...]
        CNN channel progression (ConvLSTM only).
    embedding_dim : int
        Card action embedding dimension.
    """

    # Observation dims (must match cr-gym)
    scalar_dim: int = 16
    troop_feature_dim: int = 14
    card_feature_dim: int = 5
    max_troops: int = 100
    deck_size: int = 8
    arena_channels: int = 8
    arena_h: int = 32
    arena_w: int = 18

    # Action space
    n_card_options: int = 9
    n_tile_x: int = 18
    n_tile_y: int = 32

    # Architecture
    hidden_dim: int = 256
    lstm_hidden_dim: int = 256
    lstm_layers: int = 1
    cnn_channels: tuple[int, ...] = (32, 64)
    embedding_dim: int = 32

    @property
    def n_position(self) -> int:
        return self.n_tile_x * self.n_tile_y

    @property
    def flat_obs_dim(self) -> int:
        """Total flattened observation size (for FlatMLP)."""
        return (
            self.scalar_dim
            + self.max_troops * self.troop_feature_dim
            + self.deck_size * self.card_feature_dim
            + self.arena_channels * self.arena_h * self.arena_w
        )
