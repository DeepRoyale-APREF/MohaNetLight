"""Tests for MohaNetLight — shape checks, forward passes, and component tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mohanetlight.config import ModelConfig, TrainingConfig
from mohanetlight.network.encoders import CardEncoder, EntityEncoder, ScalarEncoder
from mohanetlight.network.core import LSTMCore
from mohanetlight.network.heads import (
    ActionEmbedding,
    CardHead,
    HeadOutput,
    StrategyHead,
    TileXHead,
    TileYHead,
    ValueHead,
    masked_categorical,
)
from mohanetlight.network.mohanet import MohaNetLight, ModelOutput


@pytest.fixture
def cfg() -> ModelConfig:
    return ModelConfig()


@pytest.fixture
def model(cfg: ModelConfig) -> MohaNetLight:
    return MohaNetLight(cfg)


@pytest.fixture
def batch_tensors(cfg: ModelConfig):
    """Create a batch of random input tensors."""
    B = 4
    scalars = torch.randn(B, cfg.scalar_dim)
    troops = torch.randn(B, cfg.max_troops, cfg.troop_feature_dim).abs()
    troop_mask = torch.zeros(B, cfg.max_troops, dtype=torch.bool)
    troop_mask[:, :10] = True  # 10 active troops
    cards = torch.randn(B, cfg.hand_size, cfg.card_feature_dim).abs()
    action_masks = {
        "strategy": torch.ones(B, cfg.n_strategies, dtype=torch.bool),
        "card": torch.ones(B, cfg.n_card_options, dtype=torch.bool),
        "tile_x": torch.ones(B, cfg.n_tile_x, dtype=torch.bool),
        "tile_y": torch.ones(B, cfg.n_tile_y, dtype=torch.bool),
    }
    return scalars, troops, troop_mask, cards, action_masks


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfig:
    def test_model_config_frozen(self):
        cfg = ModelConfig()
        with pytest.raises(AttributeError):
            cfg.scalar_dim = 99  # type: ignore[misc]

    def test_concat_encoder_dim(self, cfg: ModelConfig):
        assert cfg.concat_encoder_dim == 384

    def test_head_input_dim(self, cfg: ModelConfig):
        # lstm_hidden(256) + embedding_proj(64) + encoder(128) = 448
        assert cfg.head_input_dim == 448

    def test_training_config_mutable(self):
        tc = TrainingConfig()
        tc.lr = 1e-5
        assert tc.lr == 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class TestScalarEncoder:
    def test_output_shape(self, cfg: ModelConfig):
        enc = ScalarEncoder(cfg)
        x = torch.randn(2, cfg.scalar_dim)
        out = enc(x)
        assert out.shape == (2, cfg.encoder_dim)

    def test_single_sample(self, cfg: ModelConfig):
        enc = ScalarEncoder(cfg)
        x = torch.randn(1, cfg.scalar_dim)
        out = enc(x)
        assert out.shape == (1, cfg.encoder_dim)


class TestEntityEncoder:
    def test_output_shape(self, cfg: ModelConfig):
        enc = EntityEncoder(cfg)
        x = torch.randn(2, cfg.max_troops, cfg.troop_feature_dim)
        mask = torch.zeros(2, cfg.max_troops, dtype=torch.bool)
        mask[:, :5] = True
        out = enc(x, mask)
        assert out.shape == (2, cfg.encoder_dim)

    def test_all_masked_no_nan(self, cfg: ModelConfig):
        enc = EntityEncoder(cfg)
        x = torch.randn(2, cfg.max_troops, cfg.troop_feature_dim)
        mask = torch.zeros(2, cfg.max_troops, dtype=torch.bool)
        out = enc(x, mask)
        assert not torch.isnan(out).any()

    def test_single_entity(self, cfg: ModelConfig):
        enc = EntityEncoder(cfg)
        x = torch.randn(1, cfg.max_troops, cfg.troop_feature_dim)
        mask = torch.zeros(1, cfg.max_troops, dtype=torch.bool)
        mask[0, 0] = True
        out = enc(x, mask)
        assert out.shape == (1, cfg.encoder_dim)


class TestCardEncoder:
    def test_output_shape(self, cfg: ModelConfig):
        enc = CardEncoder(cfg)
        x = torch.randn(3, cfg.hand_size, cfg.card_feature_dim)
        out = enc(x)
        assert out.shape == (3, cfg.encoder_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM Core
# ═══════════════════════════════════════════════════════════════════════════════


class TestLSTMCore:
    def test_output_shape(self, cfg: ModelConfig):
        core = LSTMCore(cfg)
        x = torch.randn(2, 5, cfg.concat_encoder_dim)  # (B, T, 384)
        h = core.init_hidden(2)
        out, new_h = core(x, h)
        assert out.shape == (2, 5, cfg.lstm_hidden_dim)
        assert new_h[0].shape == (cfg.lstm_layers, 2, cfg.lstm_hidden_dim)

    def test_single_step(self, cfg: ModelConfig):
        core = LSTMCore(cfg)
        x = torch.randn(1, 1, cfg.concat_encoder_dim)
        h = core.init_hidden(1)
        out, new_h = core(x, h)
        assert out.shape == (1, 1, cfg.lstm_hidden_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Heads
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaskedCategorical:
    def test_basic_sampling(self):
        logits = torch.randn(4, 5)
        mask = torch.ones(4, 5, dtype=torch.bool)
        out = masked_categorical(logits, mask)
        assert out.action.shape == (4,)
        assert out.log_prob.shape == (4,)
        assert out.entropy.shape == (4,)
        assert (out.action >= 0).all() and (out.action < 5).all()

    def test_mask_blocks_action(self):
        logits = torch.randn(100, 3)
        mask = torch.zeros(100, 3, dtype=torch.bool)
        mask[:, 1] = True  # only action 1 is valid
        out = masked_categorical(logits, mask)
        assert (out.action == 1).all()

    def test_evaluate_action(self):
        logits = torch.randn(4, 5)
        mask = torch.ones(4, 5, dtype=torch.bool)
        action = torch.tensor([0, 1, 2, 3])
        out = masked_categorical(logits, mask, action=action)
        assert (out.action == action).all()


class TestStrategyHead:
    def test_output_shape(self, cfg: ModelConfig):
        head = StrategyHead(cfg)
        core = torch.randn(2, cfg.lstm_hidden_dim)
        logits = head(core)
        assert logits.shape == (2, cfg.n_strategies)


class TestCardHead:
    def test_output_shape(self, cfg: ModelConfig):
        head = CardHead(cfg)
        core = torch.randn(2, cfg.lstm_hidden_dim)
        emb = torch.randn(2, cfg.embedding_proj_dim)
        ent = torch.randn(2, cfg.encoder_dim)
        logits = head(core, emb, ent)
        assert logits.shape == (2, cfg.n_card_options)


class TestTileHeads:
    def test_tile_x_shape(self, cfg: ModelConfig):
        head = TileXHead(cfg)
        core = torch.randn(2, cfg.lstm_hidden_dim)
        emb = torch.randn(2, cfg.embedding_proj_dim)
        ent = torch.randn(2, cfg.encoder_dim)
        assert head(core, emb, ent).shape == (2, cfg.n_tile_x)

    def test_tile_y_shape(self, cfg: ModelConfig):
        head = TileYHead(cfg)
        core = torch.randn(2, cfg.lstm_hidden_dim)
        emb = torch.randn(2, cfg.embedding_proj_dim)
        ent = torch.randn(2, cfg.encoder_dim)
        assert head(core, emb, ent).shape == (2, cfg.n_tile_y)


class TestValueHead:
    def test_output_shape(self, cfg: ModelConfig):
        head = ValueHead(cfg)
        core = torch.randn(2, cfg.lstm_hidden_dim)
        val = head(core)
        assert val.shape == (2,)


class TestActionEmbedding:
    def test_output_shape(self, cfg: ModelConfig):
        emb = ActionEmbedding(cfg.n_strategies, cfg.embedding_dim, cfg.embedding_proj_dim)
        action = torch.tensor([0, 1, 2])
        out = emb(action)
        assert out.shape == (3, cfg.embedding_proj_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════════════════


class TestMohaNetLight:
    def test_parameter_count(self, model: MohaNetLight):
        params = model.count_parameters()
        assert 1_000_000 < params < 3_000_000, f"Expected ~1.6M params, got {params:,}"

    def test_act_shapes(self, model: MohaNetLight, batch_tensors):
        scalars, troops, troop_mask, cards, action_masks = batch_tensors
        B = scalars.shape[0]
        hidden = model.init_hidden(B)

        output = model.act(scalars, troops, troop_mask, cards, action_masks, hidden)

        assert isinstance(output, ModelOutput)
        assert output.actions["strategy"].shape == (B,)
        assert output.actions["card"].shape == (B,)
        assert output.actions["tile_x"].shape == (B,)
        assert output.actions["tile_y"].shape == (B,)
        assert output.log_prob.shape == (B,)
        assert output.value.shape == (B,)
        assert output.entropy.shape == (B,)

    def test_act_single_sample(self, model: MohaNetLight, cfg: ModelConfig):
        scalars = torch.randn(1, cfg.scalar_dim)
        troops = torch.randn(1, cfg.max_troops, cfg.troop_feature_dim).abs()
        troop_mask = torch.zeros(1, cfg.max_troops, dtype=torch.bool)
        troop_mask[0, :3] = True
        cards = torch.randn(1, cfg.hand_size, cfg.card_feature_dim).abs()
        action_masks = {
            "strategy": torch.ones(1, cfg.n_strategies, dtype=torch.bool),
            "card": torch.ones(1, cfg.n_card_options, dtype=torch.bool),
            "tile_x": torch.ones(1, cfg.n_tile_x, dtype=torch.bool),
            "tile_y": torch.ones(1, cfg.n_tile_y, dtype=torch.bool),
        }
        hidden = model.init_hidden(1)
        output = model.act(scalars, troops, troop_mask, cards, action_masks, hidden)
        assert output.actions["strategy"].shape == (1,)

    def test_evaluate_actions(self, model: MohaNetLight, batch_tensors, cfg: ModelConfig):
        scalars, troops, troop_mask, cards, action_masks = batch_tensors
        B = scalars.shape[0]
        hidden = model.init_hidden(B)

        # First sample actions
        with torch.no_grad():
            output = model.act(scalars, troops, troop_mask, cards, action_masks, hidden)

        # Then evaluate them
        log_prob, value, entropy, new_hidden = model.evaluate_actions(
            scalars, troops, troop_mask, cards, action_masks, output.actions, hidden,
        )
        assert log_prob.shape == (B,)
        assert value.shape == (B,)
        assert entropy.shape == (B,)

    def test_hidden_state_propagation(self, model: MohaNetLight, cfg: ModelConfig):
        """Hidden state should change after a forward pass."""
        scalars = torch.randn(1, cfg.scalar_dim)
        troops = torch.randn(1, cfg.max_troops, cfg.troop_feature_dim).abs()
        troop_mask = torch.zeros(1, cfg.max_troops, dtype=torch.bool)
        troop_mask[0, :2] = True
        cards = torch.randn(1, cfg.hand_size, cfg.card_feature_dim).abs()
        action_masks = {
            "strategy": torch.ones(1, cfg.n_strategies, dtype=torch.bool),
            "card": torch.ones(1, cfg.n_card_options, dtype=torch.bool),
            "tile_x": torch.ones(1, cfg.n_tile_x, dtype=torch.bool),
            "tile_y": torch.ones(1, cfg.n_tile_y, dtype=torch.bool),
        }

        h0 = model.init_hidden(1)
        out1 = model.act(scalars, troops, troop_mask, cards, action_masks, h0)
        h1 = out1.hidden

        # Hidden state should have changed
        assert not torch.allclose(h0[0], h1[0])

    def test_masked_actions_are_valid(self, model: MohaNetLight, cfg: ModelConfig):
        """Actions should respect masks."""
        B = 50
        scalars = torch.randn(B, cfg.scalar_dim)
        troops = torch.randn(B, cfg.max_troops, cfg.troop_feature_dim).abs()
        troop_mask = torch.zeros(B, cfg.max_troops, dtype=torch.bool)
        troop_mask[:, :5] = True
        cards = torch.randn(B, cfg.hand_size, cfg.card_feature_dim).abs()

        # Only strategy 0 and card 4 (noop) are valid
        action_masks = {
            "strategy": torch.zeros(B, cfg.n_strategies, dtype=torch.bool),
            "card": torch.zeros(B, cfg.n_card_options, dtype=torch.bool),
            "tile_x": torch.ones(B, cfg.n_tile_x, dtype=torch.bool),
            "tile_y": torch.ones(B, cfg.n_tile_y, dtype=torch.bool),
        }
        action_masks["strategy"][:, 0] = True
        action_masks["card"][:, 4] = True  # noop only

        hidden = model.init_hidden(B)
        output = model.act(scalars, troops, troop_mask, cards, action_masks, hidden)

        assert (output.actions["strategy"] == 0).all()
        assert (output.actions["card"] == 4).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout Buffer
# ═══════════════════════════════════════════════════════════════════════════════


class TestRolloutBuffer:
    def test_add_and_finish(self):
        from mohanetlight.training.rollout import RolloutBuffer

        buf = RolloutBuffer(n_steps=8, gamma=0.99, gae_lambda=0.95)
        hidden = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256))

        for i in range(8):
            obs = {
                "scalars": np.zeros(16, dtype=np.float32),
                "troops": np.zeros((100, 14), dtype=np.float32),
                "troop_mask": np.zeros(100, dtype=bool),
                "cards": np.zeros((4, 4), dtype=np.float32),
                "action_mask": {
                    "strategy": np.ones(3, dtype=bool),
                    "card": np.ones(5, dtype=bool),
                    "tile_x": np.ones(18, dtype=bool),
                    "tile_y": np.ones(32, dtype=bool),
                },
            }
            action = {"strategy": 0, "card": 4, "tile_x": 9, "tile_y": 15}
            buf.add(obs, action, log_prob=-1.0, value=0.5, reward=1.0,
                    done=False, hidden=hidden)

        assert buf.full
        buf.finish(last_value=0.0)
        assert buf.advantages is not None
        assert buf.returns is not None
        assert len(buf.advantages) == 8

    def test_chunks(self):
        from mohanetlight.training.rollout import RolloutBuffer

        buf = RolloutBuffer(n_steps=8)
        hidden = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256))

        for i in range(8):
            obs = {
                "scalars": np.random.randn(16).astype(np.float32),
                "troops": np.random.randn(100, 14).astype(np.float32),
                "troop_mask": np.zeros(100, dtype=bool),
                "cards": np.random.randn(4, 4).astype(np.float32),
                "action_mask": {
                    "strategy": np.ones(3, dtype=bool),
                    "card": np.ones(5, dtype=bool),
                    "tile_x": np.ones(18, dtype=bool),
                    "tile_y": np.ones(32, dtype=bool),
                },
            }
            action = {"strategy": 1, "card": 2, "tile_x": 5, "tile_y": 10}
            buf.add(obs, action, log_prob=-0.5, value=1.0, reward=0.5,
                    done=(i == 7), hidden=hidden)

        buf.finish(last_value=0.0)

        chunks = list(buf.chunks(chunk_len=4))
        assert len(chunks) == 2
        assert chunks[0]["scalars"].shape == (4, 16)
        assert chunks[0]["actions"]["strategy"].shape == (4,)

    def test_reset(self):
        from mohanetlight.training.rollout import RolloutBuffer

        buf = RolloutBuffer(n_steps=4)
        hidden = (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256))

        for _ in range(4):
            obs = {
                "scalars": np.zeros(16, dtype=np.float32),
                "troops": np.zeros((100, 14), dtype=np.float32),
                "troop_mask": np.zeros(100, dtype=bool),
                "cards": np.zeros((4, 4), dtype=np.float32),
                "action_mask": {k: np.ones(s, dtype=bool)
                    for k, s in [("strategy", 3), ("card", 5),
                                  ("tile_x", 18), ("tile_y", 32)]},
            }
            buf.add(obs, {"strategy": 0, "card": 4, "tile_x": 0, "tile_y": 0},
                    -1.0, 0.0, 0.0, False, hidden)

        buf.finish(0.0)
        buf.reset()
        assert not buf.full
        assert len(buf.obs_list) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor Utilities
# ═══════════════════════════════════════════════════════════════════════════════


class TestTensorUtils:
    def test_obs_to_tensors(self):
        from mohanetlight.utils.tensor_utils import obs_to_tensors

        obs = {
            "scalars": np.random.randn(16).astype(np.float32),
            "troops": np.random.randn(100, 14).astype(np.float32),
            "troop_mask": np.zeros(100, dtype=bool),
            "cards": np.random.randn(4, 4).astype(np.float32),
            "action_mask": {
                "strategy": np.ones(3, dtype=bool),
                "card": np.ones(5, dtype=bool),
                "tile_x": np.ones(18, dtype=bool),
                "tile_y": np.ones(32, dtype=bool),
            },
        }
        scalars, troops, troop_mask, cards, masks = obs_to_tensors(obs)
        assert scalars.shape == (1, 16)
        assert troops.shape == (1, 100, 14)
        assert troop_mask.shape == (1, 100)
        assert cards.shape == (1, 4, 4)
        assert masks["strategy"].shape == (1, 3)

    def test_batch_obs(self):
        from mohanetlight.utils.tensor_utils import batch_obs

        obs_list = []
        for _ in range(3):
            obs_list.append({
                "scalars": np.random.randn(16).astype(np.float32),
                "troops": np.random.randn(100, 14).astype(np.float32),
                "troop_mask": np.zeros(100, dtype=bool),
                "cards": np.random.randn(4, 4).astype(np.float32),
                "action_mask": {
                    "strategy": np.ones(3, dtype=bool),
                    "card": np.ones(5, dtype=bool),
                    "tile_x": np.ones(18, dtype=bool),
                    "tile_y": np.ones(32, dtype=bool),
                },
            })
        scalars, troops, troop_mask, cards, masks = batch_obs(obs_list)
        assert scalars.shape == (3, 16)
        assert masks["card"].shape == (3, 5)


# ═══════════════════════════════════════════════════════════════════════════════
# Bots (smoke tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBots:
    def test_default_roster_length(self):
        from mohanetlight.bots.strategies import default_bot_roster
        roster = default_bot_roster()
        assert len(roster) == 10

    def test_bot_names_unique(self):
        from mohanetlight.bots.strategies import default_bot_roster
        roster = default_bot_roster()
        names = [b.name for b in roster]
        assert len(names) == len(set(names))

    def test_all_bots_have_metadata(self):
        from mohanetlight.bots.strategies import default_bot_roster
        for bot in default_bot_roster():
            meta = bot.metadata()
            assert "name" in meta
            assert "type" in meta
