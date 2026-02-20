"""Hyperparameter configuration for model architecture and PPO training."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters for MohaNetLight.

    Total ≈ 1.6M parameters — fits comfortably in Google Colab free tier.

    Dimensions
    ----------
    scalar_dim : int
        Input dimension for scalar features (elixir, tower HPs, time, flags).
    troop_feature_dim : int
        Per-entity feature dimension from cr-gym observation.
    card_feature_dim : int
        Per-deck-card feature dimension.
    max_troops : int
        Max padded entities in the observation array.
    deck_size : int
        Number of cards in the deck (8 in Clash Royale).

    Encoder
    -------
    encoder_dim : int
        Output dimension of each encoder (scalar, entity, card).
    entity_model_dim : int
        Internal dimension of the entity transformer.
    entity_n_heads : int
        Number of attention heads in entity transformer.
    entity_ff_dim : int
        Feed-forward dimension in entity transformer layers.
    entity_n_layers : int
        Number of transformer encoder layers.
    card_hidden_dim : int
        Hidden dimension per card before concatenation.

    Core
    ----
    lstm_hidden_dim : int
        LSTM hidden size.
    lstm_layers : int
        Number of stacked LSTM layers.

    Heads
    -----
    head_hidden_dim : int
        Hidden dimension inside action heads.
    embedding_dim : int
        Dimension of sampled-action embeddings (strategy, card, tile_x).
    embedding_proj_dim : int
        Projected embedding dimension fed to next head.

    Action Space
    ------------
    n_strategies : int
        Number of strategy options (AGGRESSIVE, DEFENSIVE, FARMING).
    n_card_options : int
        Deck cards (8) + noop (1) = 9.
    n_tile_x : int
        Number of tile columns.
    n_tile_y : int
        Number of tile rows.
    """

    # Input dims (from cr-gym)
    scalar_dim: int = 16
    troop_feature_dim: int = 14
    card_feature_dim: int = 5
    max_troops: int = 100
    deck_size: int = 8

    # Encoder dims
    encoder_dim: int = 128
    entity_model_dim: int = 64
    entity_n_heads: int = 4
    entity_ff_dim: int = 256
    entity_n_layers: int = 2
    card_hidden_dim: int = 32

    # LSTM core
    lstm_hidden_dim: int = 256
    lstm_layers: int = 2

    # Head dims
    head_hidden_dim: int = 128
    embedding_dim: int = 32
    embedding_proj_dim: int = 64

    # Action space sizes
    n_strategies: int = 3
    n_card_options: int = 9   # 8 deck + noop
    n_tile_x: int = 18
    n_tile_y: int = 32

    @property
    def concat_encoder_dim(self) -> int:
        """Concatenated encoder output: scalar + entity + card."""
        return self.encoder_dim * 3

    @property
    def head_input_dim(self) -> int:
        """Input to card/tile heads: core + prev_embedding + entity_context."""
        return self.lstm_hidden_dim + self.embedding_proj_dim + self.encoder_dim


@dataclass
class TrainingConfig:
    """PPO training hyperparameters.

    Parameters
    ----------
    total_timesteps : int
        Total environment steps to train.
    n_steps : int
        Steps per rollout before each PPO update.  Must be large enough
        for at least one full episode to complete.  At ``frame_skip=3``
        and 30 fps a 240 s match ≈ 2 400 RL steps.  Default 2 560
        provides headroom.
    n_epochs : int
        PPO optimisation epochs per update.
    batch_chunk_len : int
        Sequence chunk length for truncated BPTT during PPO update.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    clip_eps : float
        PPO clipping epsilon.
    vf_coef : float
        Value function loss coefficient.
    ent_coef : float
        Entropy bonus coefficient.
    max_grad_norm : float
        Gradient clipping norm.
    lr : float
        Learning rate.
    frame_skip : int
        Engine frames per gym ``step()``.  Higher = faster but coarser
        RL decisions.  At fps=30, ``frame_skip=3`` → 10 decisions/s.
    eval_interval : int
        Evaluate in league every N updates.
    eval_matches_per_pair : int
        Matches per opponent pair during evaluation.
    checkpoint_interval : int
        Save model every N updates.
    log_dir : str
        Directory for logs and checkpoints.
    device : str
        Torch device (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    total_timesteps: int = 1_000_000
    n_steps: int = 2560
    n_epochs: int = 4
    batch_chunk_len: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    frame_skip: int = 3
    eval_interval: int = 20
    eval_matches_per_pair: int = 6
    checkpoint_interval: int = 50
    log_dir: str = "./logs/mohanet"
    device: str = field(default_factory=lambda: _detect_device())


def _detect_device() -> str:
    """Auto-detect best available torch device.

    Priority: ``cuda`` > ``mps`` > ``cpu``.
    Prints an informational message about which backend is selected.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[MohaNetLight] Using CUDA device: {name}", file=sys.stderr)
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[MohaNetLight] Using Apple MPS device", file=sys.stderr)
        return "mps"

    print("[MohaNetLight] No GPU detected — using CPU", file=sys.stderr)
    return "cpu"
