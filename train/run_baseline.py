#!/usr/bin/env python3
"""Train a baseline model (ConvLSTM / FlatMLP) for comparison with MohaNetLight.

Uses the **same PPO hyperparameters, curriculum, opponents, and evaluation
protocol** as MohaNet, ensuring a fair comparison.  Supports reward-shaping
variation via ``--reward-preset`` for Research Question 2.

Examples
--------
Train ConvLSTM with default reward shaping (same as MohaNet):

    python train/run_baseline.py --model conv_lstm --output-dir logs/conv_lstm

Train with sparse reward (terminal-only) for reward-shaping ablation:

    python train/run_baseline.py --model conv_lstm --reward-preset sparse \
        --output-dir logs/conv_lstm_sparse

Quick test run (fewer steps):

    python train/run_baseline.py --model conv_lstm --steps 5000 \
        --output-dir logs/conv_lstm_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from mohanetlight.baseline.config import BaselineConfig
from mohanetlight.bots.strategies import (
    BalancedBot,
    BridgeSpamBot,
    GiantPushBot,
    default_bot_roster,
)
from mohanetlight.config import TrainingConfig
from mohanetlight.training.baseline_trainer import BaselineTrainer

from train.report import generate_full_report

# ═══════════════════════════════════════════════════════════════════════════════
# Reward presets — same structure as default_reward_function but configurable
# ═══════════════════════════════════════════════════════════════════════════════

REWARD_PRESETS = {
    "default": {
        "damage_weight": 5.0,
        "defensive_weight": 2.0,
        "elixir_weight": 0.2,
        "terminal_weight": 0.5,
        "princess_reward": 5.0,
        "win_reward": 10.0,
    },
    "sparse": {
        "damage_weight": 0.0,
        "defensive_weight": 0.0,
        "elixir_weight": 0.0,
        "terminal_weight": 1.0,
        "princess_reward": 10.0,
        "win_reward": 20.0,
    },
    "dense": {
        "damage_weight": 8.0,
        "defensive_weight": 4.0,
        "elixir_weight": 0.5,
        "terminal_weight": 0.3,
        "princess_reward": 3.0,
        "win_reward": 5.0,
    },
    "damage_only": {
        "damage_weight": 10.0,
        "defensive_weight": 0.0,
        "elixir_weight": 0.0,
        "terminal_weight": 0.5,
        "princess_reward": 5.0,
        "win_reward": 10.0,
    },
}


def _build_env_kwargs(preset_name: str) -> dict:
    """Build reward_fn kwargs to pass to the environment."""
    from clash_royale_gymnasium.rewards.default import default_reward_function

    preset = REWARD_PRESETS[preset_name]
    reward_fn = default_reward_function(**preset)
    return {"reward_fn": reward_fn}


def main() -> None:
    """Entry point for baseline training."""
    parser = argparse.ArgumentParser(
        description="Train a baseline model for comparison with MohaNetLight.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="conv_lstm",
        choices=["conv_lstm", "flat_mlp"],
        help="Baseline model architecture.",
    )

    # Training
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
        help="Total environment timesteps.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2560,
        help="Steps per rollout before PPO update.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Engine frames per gym step.",
    )

    # Reward shaping
    parser.add_argument(
        "--reward-preset",
        type=str,
        default="default",
        choices=list(REWARD_PRESETS.keys()),
        help="Reward shaping preset for ablation studies.",
    )

    # Evaluation
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=200,
        help="Evaluate every N updates.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N updates.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./logs/baseline",
        help="Directory for logs, checkpoints, and reports.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (auto-detect if omitted).",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=50,
        help="Moving average window for report plots.",
    )

    args = parser.parse_args()

    # ── Build training config ─────────────────────────────────────────────
    train_kwargs = dict(
        total_timesteps=args.steps,
        n_steps=args.n_steps,
        lr=args.lr,
        ent_coef=args.ent_coef,
        frame_skip=args.frame_skip,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        log_dir=args.output_dir,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    train_cfg = TrainingConfig(**train_kwargs)
    baseline_cfg = BaselineConfig()

    # ── Build environment kwargs (reward shaping) ─────────────────────────
    env_kwargs = _build_env_kwargs(args.reward_preset)

    # ── Create trainer ────────────────────────────────────────────────────
    trainer = BaselineTrainer(
        model_type=args.model,
        baseline_cfg=baseline_cfg,
        train_cfg=train_cfg,
        env_kwargs=env_kwargs,
    )

    # ── Print banner ──────────────────────────────────────────────────────
    total_updates = args.steps // args.n_steps
    print("=" * 70)
    print(f"  Baseline Training — {args.model}")
    print(f"  Parameters: {trainer.model.count_parameters():,}")
    print(f"  Total steps: {args.steps:,} ({total_updates} updates)")
    print(f"  Reward preset: {args.reward_preset}")
    print(f"  Device: {train_cfg.device}")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    # ── Collect metrics via callback ──────────────────────────────────────
    all_metrics: list[dict] = []

    def _on_update(update_idx: int, metrics: dict) -> None:
        all_metrics.append(metrics)

    trainer.callback = _on_update

    # ── Train ─────────────────────────────────────────────────────────────
    t_start = time.time()
    trainer.train()
    total_time = time.time() - t_start

    # ── Save combined metrics ─────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    combined_metrics = {f"{args.model}_{args.reward_preset}": all_metrics}
    metrics_path = output_dir / "all_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(combined_metrics, f, indent=2, default=str)

    # ── Generate reports ──────────────────────────────────────────────────
    report_dir = output_dir / "reports"
    print(f"\nGenerating reports in {report_dir}/ ...")
    try:
        generated = generate_full_report(combined_metrics, report_dir, args.ma_window)
        print(f"Generated {len(generated)} report plots.")
    except ImportError:
        print("matplotlib not installed — skipping report generation.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BASELINE TRAINING COMPLETE")
    print(f"  Model: {args.model} ({trainer.model.count_parameters():,} params)")
    print(f"  Reward preset: {args.reward_preset}")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    if all_metrics:
        last = all_metrics[-1]
        print(f"  Final return: {last.get('mean_ep_reward', 'N/A'):.3f}")
        if "eval" in last:
            print(f"  Final eval: win_rate={last['eval']['win_rate']:.1%}, "
                  f"avg_crowns={last['eval']['avg_crowns']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
