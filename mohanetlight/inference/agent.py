"""Inference agent — wraps MohaNetLight for league play."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from clash_royale_engine.core.state import State

from clash_royale_gymnasium.league.player_slot import PlayerSlot

from mohanetlight.config import ModelConfig
from mohanetlight.network.core import LSTMState
from mohanetlight.network.mohanet import MohaNetLight
from mohanetlight.utils.tensor_utils import state_to_obs_tensors


class MohaNetAgent(PlayerSlot):
    """League-compatible PlayerSlot that runs MohaNetLight inference.

    Receives a raw engine :class:`State`, converts it to tensors,
    runs a forward pass through the model, and returns an engine action.

    Parameters
    ----------
    name : str
        Agent identifier displayed in league reports.
    model : MohaNetLight
        The trained (or untrained) network.
    device : str | torch.device
        Where to run inference (``"cpu"`` or ``"cuda"``).
    deterministic : bool
        If True, take the argmax action instead of sampling.
    """

    def __init__(
        self,
        name: str,
        model: MohaNetLight,
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
            ``(tile_x, tile_y, card_idx)`` for a placement, or
            ``None`` for a noop (no card played this frame).
        """
        if self._hidden is None:
            self._hidden = self.model.init_hidden(batch_size=1)
            self._hidden = _to_device(self._hidden, self.device)

        scalars, troops, troop_mask, cards, action_masks = state_to_obs_tensors(
            state, device=self.device,
        )

        output = self.model.act(scalars, troops, troop_mask, cards, action_masks, self._hidden)
        self._hidden = output.hidden

        actions = output.actions
        card_idx = int(actions["card"].item())

        # NOOP_IDX = 4 means "do nothing"
        if card_idx == 4:
            return None

        tile_x = int(actions["tile_x"].item())
        tile_y = int(actions["tile_y"].item())
        return (tile_x, tile_y, card_idx)

    def reset(self) -> None:
        """Called before each match — clear LSTM hidden state."""
        self._hidden = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "mohanet",
            "params": self.model.count_parameters(),
            "deterministic": self.deterministic,
        }

    # ── Checkpoint helpers ────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        name: str = "MohaNet",
        device: str = "cpu",
        cfg: Optional[ModelConfig] = None,
        deterministic: bool = False,
    ) -> "MohaNetAgent":
        """Load a trained model from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to a ``.pt`` state-dict file.
        name : str
            Agent identifier.
        device : str
            Target device.
        cfg : ModelConfig | None
            Architecture config. If ``None``, uses defaults.
        deterministic : bool
            Whether to take argmax actions.
        """
        model = MohaNetLight(cfg)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return cls(name=name, model=model, device=device, deterministic=deterministic)


def _to_device(hidden: LSTMState, device: torch.device) -> LSTMState:
    """Move LSTM hidden state tuple to the target device."""
    return (hidden[0].to(device), hidden[1].to(device))
