#!/usr/bin/env python3
"""Train MohaNetLight with PPO via league play against heuristic bots.

Usage::

    python scripts/train_league.py
    python scripts/train_league.py --device cuda --total-steps 2000000
    python scripts/train_league.py --device mps --n-steps 256 --lr 1e-4

Configuration can be overridden via CLI args or by editing defaults in
``mohanetlight.config``.
"""

from __future__ import annotations

import argparse
import sys

from mohanetlight.config import ModelConfig, TrainingConfig
from mohanetlight.training.trainer import LeagueTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MohaNetLight with PPO + league evaluation",
    )
    # Training
    p.add_argument("--total-steps", type=int, default=1_000_000,
                    help="Total environment steps (default: 1M)")
    p.add_argument("--n-steps", type=int, default=512,
                    help="Steps per rollout (default: 512, ~2-3 matches at frame_skip=30)")
    p.add_argument("--n-epochs", type=int, default=4,
                    help="PPO epochs per update (default: 4)")
    p.add_argument("--chunk-len", type=int, default=32,
                    help="Truncated BPTT chunk length (default: 32)")
    p.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (default: 3e-4)")
    p.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor (default: 0.99)")
    p.add_argument("--gae-lambda", type=float, default=0.95,
                    help="GAE lambda (default: 0.95)")
    p.add_argument("--clip-eps", type=float, default=0.2,
                    help="PPO clip epsilon (default: 0.2)")
    p.add_argument("--ent-coef", type=float, default=0.01,
                    help="Entropy coefficient (default: 0.01)")
    p.add_argument("--vf-coef", type=float, default=0.5,
                    help="Value function coefficient (default: 0.5)")

    # Eval
    p.add_argument("--eval-interval", type=int, default=20,
                    help="League eval every N updates (default: 20)")
    p.add_argument("--eval-matches", type=int, default=6,
                    help="Matches per pair in eval (default: 6)")
    p.add_argument("--checkpoint-interval", type=int, default=50,
                    help="Save model every N updates (default: 50)")

    # System
    p.add_argument("--frame-skip", type=int, default=10,
                    help="Physics frames per RL step (default: 10, ~3 decisions/s)")
    p.add_argument("--device", type=str, default=None,
                    choices=["cpu", "cuda", "mps"],
                    help="Torch device (default: auto-detect)")
    p.add_argument("--log-dir", type=str, default="./logs/mohanet",
                    help="Log directory (default: ./logs/mohanet)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg = ModelConfig()

    # Build TrainingConfig — if --device not given, let auto-detect pick
    train_kwargs: dict = dict(
        total_timesteps=args.total_steps,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_chunk_len=args.chunk_len,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        eval_interval=args.eval_interval,
        eval_matches_per_pair=args.eval_matches,
        checkpoint_interval=args.checkpoint_interval,
        log_dir=args.log_dir,
        frame_skip=args.frame_skip,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    train_cfg = TrainingConfig(**train_kwargs)

    trainer = LeagueTrainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
