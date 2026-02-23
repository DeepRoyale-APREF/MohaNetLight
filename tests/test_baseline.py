"""Tests for baseline models and training infrastructure."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mohanetlight.baseline.config import BaselineConfig
from mohanetlight.baseline.conv_lstm import ConvLSTMNet
from mohanetlight.baseline.flat_mlp import FlatMLPNet
from mohanetlight.network.mohanet import ModelOutput


@pytest.fixture
def baseline_cfg() -> BaselineConfig:
    return BaselineConfig()


@pytest.fixture
def baseline_tensors(baseline_cfg: BaselineConfig):
    """Create a batch of random input tensors matching baseline dimensions."""
    B = 4
    cfg = baseline_cfg
    scalars = torch.randn(B, cfg.scalar_dim)
    troops = torch.randn(B, cfg.max_troops, cfg.troop_feature_dim).abs()
    troop_mask = torch.zeros(B, cfg.max_troops, dtype=torch.bool)
    troop_mask[:, :10] = True
    cards = torch.randn(B, cfg.deck_size, cfg.card_feature_dim).abs()
    arena_map = torch.randn(B, cfg.arena_channels, cfg.arena_h, cfg.arena_w).abs()
    action_masks = {
        "card": torch.ones(B, cfg.n_card_options, dtype=torch.bool),
        "tile_x_per_card": torch.ones(B, cfg.n_card_options, cfg.n_tile_x, dtype=torch.bool),
        "tile_y_per_card": torch.ones(B, cfg.n_card_options, cfg.n_tile_y, dtype=torch.bool),
        "spatial_per_card": torch.ones(B, cfg.n_card_options, cfg.n_position, dtype=torch.bool),
    }
    return scalars, troops, troop_mask, cards, arena_map, action_masks


# ═══════════════════════════════════════════════════════════════════════════════
# BaselineConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaselineConfig:
    def test_frozen(self, baseline_cfg: BaselineConfig) -> None:
        with pytest.raises(AttributeError):
            baseline_cfg.scalar_dim = 99  # type: ignore[misc]

    def test_n_position(self, baseline_cfg: BaselineConfig) -> None:
        assert baseline_cfg.n_position == 18 * 32

    def test_flat_obs_dim(self, baseline_cfg: BaselineConfig) -> None:
        expected = (
            16       # scalars
            + 100 * 14  # troops
            + 8 * 5     # cards
            + 8 * 32 * 18  # arena
        )
        assert baseline_cfg.flat_obs_dim == expected

    def test_dims_match_model_config(self) -> None:
        """Baseline obs dims must match MohaNet's obs dims."""
        from mohanetlight.config import ModelConfig
        mc = ModelConfig()
        bc = BaselineConfig()
        assert mc.scalar_dim == bc.scalar_dim
        assert mc.troop_feature_dim == bc.troop_feature_dim
        assert mc.card_feature_dim == bc.card_feature_dim
        assert mc.max_troops == bc.max_troops
        assert mc.deck_size == bc.deck_size
        assert mc.arena_channels == bc.arena_channels
        assert mc.arena_h == bc.arena_h
        assert mc.arena_w == bc.arena_w
        assert mc.n_card_options == bc.n_card_options
        assert mc.n_tile_x == bc.n_tile_x
        assert mc.n_tile_y == bc.n_tile_y


# ═══════════════════════════════════════════════════════════════════════════════
# ConvLSTMNet
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvLSTMNet:
    def test_instantiation(self) -> None:
        model = ConvLSTMNet()
        assert model.count_parameters() > 0

    def test_param_count_reasonable(self) -> None:
        model = ConvLSTMNet()
        params = model.count_parameters()
        # Expected ~0.7M, should be between 0.3M and 2M
        assert 300_000 < params < 2_000_000, f"Unexpected param count: {params:,}"

    def test_init_hidden(self) -> None:
        model = ConvLSTMNet()
        h, c = model.init_hidden(batch_size=2)
        assert h.shape[1] == 2  # batch dim
        assert c.shape[1] == 2

    def test_act_returns_model_output(self, baseline_tensors) -> None:
        model = ConvLSTMNet()
        scalars, troops, troop_mask, cards, arena_map, action_masks = baseline_tensors
        hidden = model.init_hidden(batch_size=4)

        output = model.act(
            scalars, troops, troop_mask, cards, arena_map, action_masks, hidden,
        )

        assert isinstance(output, ModelOutput)
        assert output.actions["card"].shape == (4,)
        assert output.actions["tile_x"].shape == (4,)
        assert output.actions["tile_y"].shape == (4,)
        assert output.log_prob.shape == (4,)
        assert output.value.shape == (4,)
        assert output.entropy.shape == (4,)

    def test_act_actions_in_range(self, baseline_tensors, baseline_cfg) -> None:
        model = ConvLSTMNet()
        scalars, troops, troop_mask, cards, arena_map, action_masks = baseline_tensors
        hidden = model.init_hidden(batch_size=4)

        output = model.act(
            scalars, troops, troop_mask, cards, arena_map, action_masks, hidden,
        )

        assert (output.actions["card"] >= 0).all()
        assert (output.actions["card"] < baseline_cfg.n_card_options).all()
        assert (output.actions["tile_x"] >= 0).all()
        assert (output.actions["tile_x"] < baseline_cfg.n_tile_x).all()
        assert (output.actions["tile_y"] >= 0).all()
        assert (output.actions["tile_y"] < baseline_cfg.n_tile_y).all()

    def test_evaluate_actions(self, baseline_tensors) -> None:
        model = ConvLSTMNet()
        scalars, troops, troop_mask, cards, arena_map, action_masks = baseline_tensors
        hidden = model.init_hidden(batch_size=4)

        # First get actions via act()
        output = model.act(
            scalars, troops, troop_mask, cards, arena_map, action_masks, hidden,
        )

        # Then evaluate those same actions
        log_prob, value, entropy, new_hidden = model.evaluate_actions(
            scalars, troops, troop_mask, cards, arena_map,
            action_masks, output.actions, hidden,
        )

        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert entropy.shape == (4,)

    def test_hidden_state_updates(self, baseline_tensors) -> None:
        model = ConvLSTMNet()
        scalars, troops, troop_mask, cards, arena_map, action_masks = baseline_tensors
        hidden = model.init_hidden(batch_size=4)

        output = model.act(
            scalars, troops, troop_mask, cards, arena_map, action_masks, hidden,
        )

        # Hidden state should have changed
        h_old, c_old = hidden
        h_new, c_new = output.hidden
        assert not torch.allclose(h_old, h_new) or not torch.allclose(c_old, c_new)


# ═══════════════════════════════════════════════════════════════════════════════
# FlatMLPNet
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlatMLPNet:
    def test_instantiation(self) -> None:
        model = FlatMLPNet()
        assert model.count_parameters() > 0

    def test_param_count_reasonable(self) -> None:
        model = FlatMLPNet()
        params = model.count_parameters()
        # Expected ~4M, should be between 2M and 8M
        assert 2_000_000 < params < 8_000_000, f"Unexpected param count: {params:,}"

    def test_init_hidden_dummy(self) -> None:
        model = FlatMLPNet()
        h, c = model.init_hidden(batch_size=3)
        # FlatMLP uses dummy hidden (size 1)
        assert h.shape == (1, 3, 1)

    def test_act_returns_model_output(self, baseline_tensors) -> None:
        model = FlatMLPNet()
        scalars, troops, troop_mask, cards, arena_map, action_masks = baseline_tensors
        hidden = model.init_hidden(batch_size=4)

        output = model.act(
            scalars, troops, troop_mask, cards, arena_map, action_masks, hidden,
        )

        assert isinstance(output, ModelOutput)
        assert output.actions["card"].shape == (4,)
        assert output.actions["tile_x"].shape == (4,)
        assert output.actions["tile_y"].shape == (4,)
        assert output.log_prob.shape == (4,)
        assert output.value.shape == (4,)

    def test_evaluate_actions(self, baseline_tensors) -> None:
        model = FlatMLPNet()
        scalars, troops, troop_mask, cards, arena_map, action_masks = baseline_tensors
        hidden = model.init_hidden(batch_size=4)

        output = model.act(
            scalars, troops, troop_mask, cards, arena_map, action_masks, hidden,
        )

        log_prob, value, entropy, new_hidden = model.evaluate_actions(
            scalars, troops, troop_mask, cards, arena_map,
            action_masks, output.actions, hidden,
        )

        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert entropy.shape == (4,)


# ═══════════════════════════════════════════════════════════════════════════════
# Interface compatibility (both baselines match MohaNetLight interface)
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterfaceCompatibility:
    """Verify baseline models expose the same interface as MohaNetLight."""

    @pytest.mark.parametrize("model_cls", [ConvLSTMNet, FlatMLPNet])
    def test_has_required_methods(self, model_cls) -> None:
        model = model_cls()
        assert hasattr(model, "act")
        assert hasattr(model, "evaluate_actions")
        assert hasattr(model, "init_hidden")
        assert hasattr(model, "count_parameters")
        assert callable(model.act)
        assert callable(model.evaluate_actions)
        assert callable(model.init_hidden)
        assert callable(model.count_parameters)

    @pytest.mark.parametrize("model_cls", [ConvLSTMNet, FlatMLPNet])
    def test_act_output_type(self, model_cls, baseline_tensors) -> None:
        model = model_cls()
        scalars, troops, troop_mask, cards, arena_map, action_masks = baseline_tensors
        hidden = model.init_hidden(batch_size=4)

        output = model.act(
            scalars, troops, troop_mask, cards, arena_map, action_masks, hidden,
        )

        assert isinstance(output, ModelOutput)
        assert "card" in output.actions
        assert "tile_x" in output.actions
        assert "tile_y" in output.actions

    @pytest.mark.parametrize("model_cls", [ConvLSTMNet, FlatMLPNet])
    def test_evaluate_output_shapes(self, model_cls, baseline_tensors) -> None:
        model = model_cls()
        scalars, troops, troop_mask, cards, arena_map, action_masks = baseline_tensors
        hidden = model.init_hidden(batch_size=4)

        output = model.act(
            scalars, troops, troop_mask, cards, arena_map, action_masks, hidden,
        )

        log_prob, value, entropy, new_hidden = model.evaluate_actions(
            scalars, troops, troop_mask, cards, arena_map,
            action_masks, output.actions, hidden,
        )

        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert entropy.shape == (4,)
        assert isinstance(new_hidden, tuple)
        assert len(new_hidden) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# BaselineAgent
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaselineAgent:
    def test_agent_creation(self) -> None:
        from mohanetlight.inference.baseline_agent import BaselineAgent
        model = ConvLSTMNet()
        agent = BaselineAgent(name="test", model=model)
        assert agent.name == "test"

    def test_agent_metadata(self) -> None:
        from mohanetlight.inference.baseline_agent import BaselineAgent
        model = ConvLSTMNet()
        agent = BaselineAgent(name="test-conv", model=model)
        meta = agent.metadata()
        assert meta["name"] == "test-conv"
        assert meta["type"] == "ConvLSTMNet"
        assert meta["params"] > 0

    def test_agent_reset(self) -> None:
        from mohanetlight.inference.baseline_agent import BaselineAgent
        model = ConvLSTMNet()
        agent = BaselineAgent(name="test", model=model)
        agent.reset()
        assert agent._hidden is None


# ═══════════════════════════════════════════════════════════════════════════════
# BaselineTrainer (unit-level — no env dependency)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaselineTrainerFactory:
    def test_build_conv_lstm(self) -> None:
        from mohanetlight.training.baseline_trainer import build_baseline_model
        model = build_baseline_model("conv_lstm")
        assert isinstance(model, ConvLSTMNet)

    def test_build_flat_mlp(self) -> None:
        from mohanetlight.training.baseline_trainer import build_baseline_model
        model = build_baseline_model("flat_mlp")
        assert isinstance(model, FlatMLPNet)

    def test_build_unknown_raises(self) -> None:
        from mohanetlight.training.baseline_trainer import build_baseline_model
        with pytest.raises(ValueError, match="Unknown baseline"):
            build_baseline_model("unknown_model")
