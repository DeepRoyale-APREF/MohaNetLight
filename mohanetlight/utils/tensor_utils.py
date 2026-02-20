"""Utilities for converting gymnasium observations to batched tensors."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor


def obs_to_tensors(
    obs: Dict[str, np.ndarray | Dict[str, np.ndarray]],
    device: torch.device | str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """Convert a gymnasium observation dict to batched PyTorch tensors.

    Parameters
    ----------
    obs : dict
        Gymnasium Dict observation with keys:
        ``troops (100,14)``, ``troop_mask (100,)``, ``scalars (16,)``,
        ``cards (8,5)``, ``action_mask {strategy, card, tile_x_per_card, tile_y_per_card}``.
    device : str | torch.device
        Target device.

    Returns
    -------
    scalars : Tensor ``(1, 16)``
    troops : Tensor ``(1, 100, 14)``
    troop_mask : Tensor ``(1, 100)`` bool
    cards : Tensor ``(1, 8, 5)``
    action_masks : dict[str, Tensor]
        Each mask is unsqueezed to add a batch dim.
    """
    scalars = torch.as_tensor(obs["scalars"], dtype=torch.float32, device=device).unsqueeze(0)
    troops = torch.as_tensor(obs["troops"], dtype=torch.float32, device=device).unsqueeze(0)
    troop_mask = torch.as_tensor(obs["troop_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    cards = torch.as_tensor(obs["cards"], dtype=torch.float32, device=device).unsqueeze(0)

    mask_dict = obs["action_mask"]
    action_masks = {
        k: torch.as_tensor(v, dtype=torch.bool, device=device).unsqueeze(0)
        for k, v in mask_dict.items()
    }

    return scalars, troops, troop_mask, cards, action_masks


def batch_obs(
    obs_list: list[Dict[str, np.ndarray | Dict[str, np.ndarray]]],
    device: torch.device | str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """Stack a list of observations into batched tensors.

    Parameters
    ----------
    obs_list : list[dict]
        List of gymnasium Dict observations.
    device : str | torch.device
        Target device.

    Returns
    -------
    Same as :func:`obs_to_tensors` but with batch dim = len(obs_list).
    """
    n = len(obs_list)
    scalars = torch.zeros(n, obs_list[0]["scalars"].shape[0], device=device)
    troops = torch.zeros(n, *obs_list[0]["troops"].shape, device=device)
    troop_mask = torch.zeros(n, obs_list[0]["troop_mask"].shape[0], dtype=torch.bool, device=device)
    cards = torch.zeros(n, *obs_list[0]["cards"].shape, device=device)

    mask_keys = list(obs_list[0]["action_mask"].keys())
    action_masks: Dict[str, Tensor] = {
        k: torch.zeros(n, *np.array(obs_list[0]["action_mask"][k]).shape, dtype=torch.bool, device=device)
        for k in mask_keys
    }

    for i, obs in enumerate(obs_list):
        scalars[i] = torch.as_tensor(obs["scalars"], dtype=torch.float32)
        troops[i] = torch.as_tensor(obs["troops"], dtype=torch.float32)
        troop_mask[i] = torch.as_tensor(obs["troop_mask"], dtype=torch.bool)
        cards[i] = torch.as_tensor(obs["cards"], dtype=torch.float32)
        for k in mask_keys:
            action_masks[k][i] = torch.as_tensor(obs["action_mask"][k], dtype=torch.bool)

    return (
        scalars.to(device),
        troops.to(device),
        troop_mask.to(device),
        cards.to(device),
        {k: v.to(device) for k, v in action_masks.items()},
    )


def state_to_obs_tensors(
    state: "State",  # noqa: F821  — avoid circular import
    device: torch.device | str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """Convert a raw engine :class:`State` directly to tensors.

    This is used by the :class:`MohaNetAgent` during league play where the
    full gymnasium env is not available — the agent receives only a ``State``.

    Parameters
    ----------
    state : State
        Raw engine state for one player.
    device : str | torch.device
        Target device.

    Returns
    -------
    Same signature as :func:`obs_to_tensors`.

    Notes
    -----
    Leaked-elixir and frame-ratio are approximated since
    the agent has no access to the engine.
    """
    from clash_royale_engine.core.state import UnitDetection
    from clash_royale_engine.utils.constants import CARD_STATS, CARD_VOCAB

    _CARD_IDX = {name: i for i, name in enumerate(CARD_VOCAB)}
    _CAT = {"troop": 0, "building": 1, "spell": 2}
    _TGT = {"all": 0, "ground": 1, "buildings": 2}
    _TRS = {"ground": 0, "air": 1}

    # ── Troops ────────────────────────────────────────────────────────────
    troop_arr = np.zeros((100, 14), dtype=np.float32)
    mask_arr = np.zeros(100, dtype=bool)

    units: list[Tuple[UnitDetection, bool]] = [
        (d, True) for d in state.allies
    ] + [(d, False) for d in state.enemies]

    for i, (det, is_ally) in enumerate(units[:100]):
        s = CARD_STATS.get(det.unit.name, {})
        troop_arr[i] = [
            _CARD_IDX.get(det.unit.name, 0),
            _CAT.get(det.unit.category, 0),
            _TGT.get(det.unit.target, 0),
            _TRS.get(det.unit.transport, 0),
            float(det.position.tile_x),
            float(det.position.tile_y),
            det.hp / max(det.max_hp, 1),
            float(is_ally),
            s.get("elixir", 0) / 10.0,
            s.get("damage", 0) / 400.0,
            s.get("hit_speed", 1.0) / 2.0,
            s.get("range", 1.0) / 7.0,
            s.get("speed", 60.0) / 100.0,
            s.get("hitbox_radius", 0.5) / 2.0,
        ]
        mask_arr[i] = True

    # ── Scalars ───────────────────────────────────────────────────────────
    n = state.numbers
    troop_elixir = sum(
        CARD_STATS.get(d.unit.name, {}).get("elixir", 0) for d in state.allies
    )
    scalar_arr = np.array([
        n.elixir / 10.0,
        n.left_princess_hp / 1400.0,
        n.right_princess_hp / 1400.0,
        n.king_hp / 2400.0,
        n.left_enemy_princess_hp / 1400.0,
        n.right_enemy_princess_hp / 1400.0,
        n.enemy_king_hp / 2400.0,
        n.time_remaining / 180.0,
        float(n.king_active),
        float(n.enemy_king_active),
        float(n.is_double_elixir),
        float(n.is_overtime),
        n.overtime_remaining / 60.0,
        troop_elixir / 30.0,
        0.0,  # leaked_elixir — unknown without engine
        max(0.0, 1.0 - n.time_remaining / 180.0),  # approx frame_ratio
    ], dtype=np.float32)

    # ── Cards (all 8 deck cards with hand/affordability) ────────────────
    deck = state.deck if state.deck else [c.name for c in state.cards]
    hand_names = [c.name for c in state.cards]
    ready_set = set(state.ready)

    cards_arr = np.zeros((8, 5), dtype=np.float32)
    for i, deck_card_name in enumerate(deck[:8]):
        s = CARD_STATS.get(deck_card_name, {})
        is_in_hand = deck_card_name in hand_names
        is_affordable = False
        if is_in_hand:
            hand_slot = hand_names.index(deck_card_name)
            is_affordable = hand_slot in ready_set
        cards_arr[i] = [
            _CARD_IDX.get(deck_card_name, 0) / 8.0,
            s.get("elixir", 0) / 10.0,
            float(s.get("is_spell", False)),
            float(is_in_hand),
            float(is_affordable),
        ]

    # ── Action masks ──────────────────────────────────────────────────────
    from clash_royale_gymnasium.actions.masking import compute_action_mask

    left_dead = n.left_enemy_princess_hp <= 0
    right_dead = n.right_enemy_princess_hp <= 0
    masks = compute_action_mask(
        state,
        enemy_left_princess_dead=left_dead,
        enemy_right_princess_dead=right_dead,
    )

    obs: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {
        "scalars": scalar_arr,
        "troops": troop_arr,
        "troop_mask": mask_arr,
        "cards": cards_arr,
        "action_mask": {
            "strategy": masks.strategy,
            "card": masks.card,
            "tile_x_per_card": masks.tile_x_per_card,
            "tile_y_per_card": masks.tile_y_per_card,
        },
    }
    return obs_to_tensors(obs, device)
