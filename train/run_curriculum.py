#!/usr/bin/env python3
"""Run the full curriculum training pipeline.

Usage::

    # Default curriculum (8 micro-phases, ~50 updates each)
    python train/run_curriculum.py

    # Custom updates per phase
    python train/run_curriculum.py --updates-per-phase 100

    # Quick smoke test
    python train/run_curriculum.py --updates-per-phase 5 --n-steps 512

    # Force device
    python train/run_curriculum.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Ensure project root is on sys.path so both 'train' and 'mohanetlight' resolve
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mohanetlight.config import ModelConfig
from mohanetlight.inference.agent import MohaNetAgent

from train.curriculum import CurriculumTrainer, default_curriculum
from train.report import generate_full_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run multi-phase curriculum training for MohaNetLight",
    )
    p.add_argument(
        "--updates-per-phase", type=int, default=50,
        help="PPO updates per curriculum phase (default: 50)",
    )
    p.add_argument(
        "--steps-per-phase", type=int, default=200_000,
        help="(Legacy) Env steps for phases — ignored when --updates-per-phase is set.",
    )
    p.add_argument(
        "--self-play-steps", type=int, default=200_000,
        help="(Legacy) Env steps for self-play phase — ignored when --updates-per-phase is set.",
    )
    p.add_argument(
        "--n-steps", type=int, default=512,
        help="Rollout length — steps per PPO update (default: 512)",
    )
    p.add_argument(
        "--lr", type=float, default=3e-4,
        help="Base learning rate (default: 3e-4)",
    )
    p.add_argument(
        "--frame-skip", type=int, default=10,
        help="Engine frame skip (default: 10)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Torch device (default: auto-detect)",
    )
    p.add_argument(
        "--output-dir", type=str, default="./logs/curriculum",
        help="Root output directory (default: ./logs/curriculum)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    p.add_argument(
        "--skip-self-play", action="store_true",
        help="Skip the self-play phase (useful for quick runs)",
    )
    p.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to a .pt checkpoint to resume from (loads weights before "
             "the starting phase). Example: logs/curriculum/phase_1_balanced/mohanet_u250.pt",
    )
    p.add_argument(
        "--start-phase", type=int, default=1,
        help="1-indexed phase to start from (default: 1). "
             "Phases before this are skipped. Use with --resume-from.",
    )
    p.add_argument(
        "--ma-window", type=int, default=50,
        help="Moving average window for report plots (default: 50)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build curriculum ──────────────────────────────────────────────────
    phases = default_curriculum(
        steps_per_phase=args.steps_per_phase,
        self_play_steps=args.self_play_steps,
        n_steps=args.n_steps,
        lr=args.lr,
        base_seed=args.seed,
        updates_per_phase=args.updates_per_phase,
    )

    if args.skip_self_play:
        phases = [p for p in phases if p.name != "self_play"]
        print("Self-play phase skipped.")

    # ── Configure self-play opponent ──────────────────────────────────────
    # Self-play uses a frozen copy of the best model from phase 3.
    # We set a placeholder here — CurriculumTrainer will load the weights
    # from the previous phase checkpoint, and we need to create the
    # self-play opponent *after* phase 3 completes.
    # Strategy: subclass or patch run() to inject self-play opponent.

    model_cfg = ModelConfig()
    trainer = CurriculumTrainer(
        phases=phases,
        model_cfg=model_cfg,
        base_log_dir=args.output_dir,
        device=args.device,
        frame_skip=args.frame_skip,
    )

    # Patch self-play phase: inject a frozen MohaNetAgent as opponent
    # after phase 3 completes. We override the run loop to do this.
    original_run = trainer.run

    def run_with_self_play() -> dict:
        """Run curriculum, injecting self-play opponent between phases."""
        last_checkpoint = args.resume_from  # initialise from CLI flag
        start_idx = max(0, args.start_phase - 1)  # 1-indexed → 0-indexed

        if start_idx > 0:
            print(f"  Skipping phases 1–{start_idx} (starting at phase {start_idx + 1})")
            if last_checkpoint:
                print(f"  Resuming weights from: {last_checkpoint}")

        for i, phase in enumerate(trainer.phases):
            if i < start_idx:
                continue  # skip completed phases

            phase_dir = trainer.base_log_dir / f"phase_{i + 1}_{phase.name}"
            phase_dir.mkdir(parents=True, exist_ok=True)

            # If this is self-play phase, create opponent from last checkpoint
            if phase.name == "self_play" and last_checkpoint:
                print(f"\n  Creating self-play opponent from: {last_checkpoint}")
                sp_agent = MohaNetAgent.from_checkpoint(
                    path=last_checkpoint,
                    name="MohaNet-SelfPlay",
                    device=args.device or "cpu",
                    cfg=model_cfg,
                    deterministic=True,
                )
                phase.training_opponents = [sp_agent]
                print(f"  Self-play opponent ready: {sp_agent.name}")

            # Determine starting weights
            resume_path = phase.resume_from or last_checkpoint

            print("\n" + "=" * 70)
            print(f"  PHASE {i + 1}/{len(trainer.phases)}: {phase.name}")
            opp_names = [o.name for o in phase.training_opponents]
            print(f"  Steps: {phase.total_timesteps:,}  |  "
                  f"LR: {phase.lr}  |  Opponents: {opp_names}")
            if resume_path:
                print(f"  Resuming from: {resume_path}")
            print("=" * 70 + "\n")

            # Build TrainingConfig
            from mohanetlight.config import TrainingConfig
            train_kwargs = dict(
                total_timesteps=phase.total_timesteps,
                n_steps=phase.n_steps,
                lr=phase.lr,
                ent_coef=phase.ent_coef,
                eval_interval=phase.eval_interval,
                checkpoint_interval=phase.checkpoint_interval,
                log_dir=str(phase_dir),
                frame_skip=trainer.frame_skip,
            )
            if trainer.device is not None:
                train_kwargs["device"] = trainer.device

            train_cfg = TrainingConfig(**train_kwargs)

            from mohanetlight.training.trainer import LeagueTrainer
            eval_opps = phase.eval_opponents or phase.training_opponents
            league_trainer = LeagueTrainer(
                model_cfg=trainer.model_cfg,
                train_cfg=train_cfg,
                training_opponents=phase.training_opponents,
                eval_opponents=eval_opps,
            )

            # Load weights from previous phase
            if resume_path and Path(resume_path).exists():
                state_dict = torch.load(
                    resume_path,
                    map_location=train_cfg.device,
                    weights_only=True,
                )
                league_trainer.model.load_state_dict(state_dict)
                print(f"  Loaded weights from {resume_path}")

            # Collect metrics via callback
            phase_metrics = []

            def _on_update(update_idx: int, metrics: dict) -> None:
                phase_metrics.append(metrics)

            league_trainer.callback = _on_update

            # Train
            t0 = time.time()
            league_trainer.train()
            elapsed = time.time() - t0

            print(f"\n  Phase '{phase.name}' complete in {elapsed:.0f}s "
                  f"({len(phase_metrics)} updates)")

            # Save phase metrics
            trainer.all_metrics[phase.name] = phase_metrics
            metrics_path = phase_dir / "phase_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(phase_metrics, f, indent=2, default=str)

            last_checkpoint = trainer._find_last_checkpoint(phase_dir)

            # ── Generate intermediate report after each phase ─────────────
            report_dir = Path(args.output_dir) / "reports"
            try:
                generated = generate_full_report(
                    trainer.all_metrics, report_dir, ma_window=args.ma_window,
                )
                print(f"  Phase report: {len(generated)} plots in {report_dir}/")
            except ImportError:
                pass  # matplotlib not available

        # Save combined metrics
        combined_path = trainer.base_log_dir / "all_phases_metrics.json"
        with open(combined_path, "w") as f:
            json.dump(trainer.all_metrics, f, indent=2, default=str)

        print(f"\nAll {len(trainer.phases)} phases complete.")
        print(f"Combined metrics saved to {combined_path}")

        return trainer.all_metrics

    # ── Execute ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("  MohaNetLight — Curriculum Training Pipeline")
    print(f"  Phases: {len(phases)}")
    for i, p in enumerate(phases):
        print(f"    {i + 1}. {p.name} — {p.total_timesteps:,} steps")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    t_start = time.time()
    all_metrics = run_with_self_play()
    total_time = time.time() - t_start

    # ── Generate reports ──────────────────────────────────────────────────
    report_dir = Path(args.output_dir) / "reports"
    print(f"\nGenerating reports in {report_dir}/ ...")
    try:
        generated = generate_full_report(
            all_metrics, report_dir, ma_window=args.ma_window,
        )
        print(f"Generated {len(generated)} report plots.")
    except ImportError:
        print("matplotlib not installed — skipping report generation.")
        print("Install with: pip install matplotlib")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    for phase_name, metrics in all_metrics.items():
        if metrics:
            last = metrics[-1]
            print(f"  {phase_name}: {len(metrics)} updates, "
                  f"final return={last.get('mean_ep_reward', 'N/A'):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
