"""Inference agent — wraps baseline models (ConvLSTMNet / FlatMLPNet) for league play."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from clash_royale_engine.core.state import State

from clash_royale_gymnasium.league.player_slot import PlayerSlot

from mohanetlight.baseline.config import BaselineConfig
from mohanetlight.baseline.conv_lstm import ConvLSTMNet
from mohanetlight.baseline.flat_mlp import FlatMLPNet
from mohanetlight.network.core import LSTMState
from mohanetlight.utils.tensor_utils import state_to_obs_tensors

# Type alias for supported baseline models
BaselineModel = Union[ConvLSTMNet, FlatMLPNet]


class BaselineAgent(PlayerSlot):
    """League-compatible PlayerSlot that runs baseline model inference.

    Works with any model implementing the same interface as MohaNetLight:
    ``act()``, ``init_hidden()``, ``count_parameters()``.

    Parameters
    ----------
    name : str
        Agent identifier displayed in league reports.
    model : ConvLSTMNet | FlatMLPNet
        The baseline network.
    device : str | torch.device
        Where to run inference.
    deterministic : bool
        If True, take the argmax action instead of sampling.
    """

    def __init__(
        self,
        name: str,
        model: BaselineModel,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
    ) -> None:
        super().__init__(name)
        self.model = model
        self.device = torch.device(device)
        self.deterministic = deterministic
        self._hidden: Optional[LSTMState] = None
        self.model.to(self.device)
        self.model.eval()

    # ── PlayerSlot interface ──────────────────────────────────────────────

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        """Convert engine state to model action.

        Returns
        -------
        tuple[int, int, int] or None
            ``(tile_x, tile_y, hand_slot)`` for a placement, or
            ``None`` for a noop (no card played this frame).
        """
        if self._hidden is None:
            self._hidden = self.model.init_hidden(batch_size=1)
            self._hidden = _to_device(self._hidden, self.device)

        scalars, troops, troop_mask, cards, arena_map, action_masks = state_to_obs_tensors(
            state, device=self.device,
        )

        output = self.model.act(
            scalars, troops, troop_mask, cards, arena_map, action_masks, self._hidden,
        )
        self._hidden = output.hidden

        actions = output.actions
        card_idx = int(actions["card"].item())

        # NOOP_IDX = 8 means "do nothing"
        if card_idx == 8:
            return None

        # Convert deck_idx → hand_slot
        deck = state.deck if state.deck else [c.name for c in state.cards]
        hand_names = [c.name for c in state.cards]

        if card_idx < len(deck):
            card_name = deck[card_idx]
            if card_name in hand_names:
                hand_slot = hand_names.index(card_name)
            else:
                return None  # card not in hand, treat as noop
        else:
            return None

        tile_x = int(actions["tile_x"].item())
        tile_y = int(actions["tile_y"].item())
        return (tile_x, tile_y, hand_slot)

    def reset(self) -> None:
        """Called before each match — clear LSTM hidden state."""
        self._hidden = None

    def metadata(self) -> Dict[str, Any]:
        """Return agent metadata for reporting."""
        model_type = type(self.model).__name__
        return {
            "name": self.name,
            "type": model_type,
            "params": self.model.count_parameters(),
            "deterministic": self.deterministic,
        }

    # ── Checkpoint helpers ────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        name: str = "Baseline",
        device: str = "cpu",
        model_type: str = "conv_lstm",
        cfg: Optional[BaselineConfig] = None,
        deterministic: bool = False,
    ) -> "BaselineAgent":
        """Load a trained baseline model from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to a ``.pt`` state-dict file.
        name : str
            Agent identifier.
        device : str
            Target device.
        model_type : str
            ``"conv_lstm"`` or ``"flat_mlp"``.
        cfg : BaselineConfig | None
            Architecture config. If ``None``, uses defaults.
        deterministic : bool
            Whether to take argmax actions.
        """
        cfg = cfg or BaselineConfig()
        if model_type == "flat_mlp":
            model: BaselineModel = FlatMLPNet(cfg)
        else:
            model = ConvLSTMNet(cfg)

        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return cls(name=name, model=model, device=device, deterministic=deterministic)


def _to_device(hidden: LSTMState, device: torch.device) -> LSTMState:
    """Move LSTM hidden state tuple to the target device."""
    return (hidden[0].to(device), hidden[1].to(device))
