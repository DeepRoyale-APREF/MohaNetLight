"""MohaNetLight — AlphaStar-inspired lightweight RL network for Clash Royale.

~1.6M parameters, trainable on a single consumer GPU or Google Colab.
"""

from mohanetlight.config import ModelConfig, TrainingConfig
from mohanetlight.network.mohanet import MohaNetLight

__version__ = "0.1.0"

__all__ = [
    "MohaNetLight",
    "ModelConfig",
    "TrainingConfig",
]
