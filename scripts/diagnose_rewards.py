#!/usr/bin/env python
"""Quick diagnostic: run one rollout and inspect per-step reward signals.

Usage:
    python scripts/diagnose_rewards.py
"""
from __future__ import annotations

import numpy as np

from clash_royale_gymnasium.env import ClashRoyaleGymEnv
from clash_royale_engine.players.player_interface import HeuristicBot


def main() -> None:
    env = ClashRoyaleGymEnv(
        opponent=HeuristicBot(aggression=0.5, seed=42),
        frame_skip=3,
        seed=0,
    )

    obs, info = env.reset()
    n_steps = 2560  # enough for a full match at frame_skip=3

    rewards = []
    breakdowns = []

    for step in range(n_steps):
        # Random masked action
        mask = obs["action_mask"]
        action = {
            "card": int(np.random.choice(np.where(mask["card"])[0])),
            "tile_x": int(np.random.choice(np.where(mask["tile_x"])[0])),
            "tile_y": int(np.random.choice(np.where(mask["tile_y"])[0])),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        breakdowns.append(info.get("reward_breakdown", {}))

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            obs, info = env.reset()

    env.close()

    rewards = np.array(rewards)

    print(f"\n{'='*60}")
    print(f"  REWARD DIAGNOSTIC — {n_steps} steps")
    print(f"{'='*60}")
    print(f"  Sum:      {rewards.sum():.6f}")
    print(f"  Mean:     {rewards.mean():.6f}")
    print(f"  Std:      {rewards.std():.6f}")
    print(f"  Min:      {rewards.min():.6f}")
    print(f"  Max:      {rewards.max():.6f}")
    print(f"  |R| mean: {np.abs(rewards).mean():.6f}")
    print(f"  Nonzero:  {np.count_nonzero(rewards)}/{n_steps} "
          f"({np.count_nonzero(rewards)/n_steps:.1%})")
    print(f"  Positive: {(rewards > 1e-10).sum()}")
    print(f"  Negative: {(rewards < -1e-10).sum()}")

    # Per-component breakdown
    if breakdowns and breakdowns[0]:
        comp_names = list(breakdowns[0].keys())
        print(f"\n  Per-component sums over {n_steps} steps:")
        for name in comp_names:
            vals = [bd.get(name, 0.0) for bd in breakdowns]
            arr = np.array(vals)
            print(f"    {name:25s}  sum={arr.sum():+.6f}  "
                  f"mean={arr.mean():+.8f}  nonzero={np.count_nonzero(arr)}")

    # Engine debug signals (sample from last info)
    engine_dbg = info.get("engine_debug", {})
    if engine_dbg:
        print(f"\n  Engine debug signals (last step):")
        for k, v in sorted(engine_dbg.items()):
            print(f"    {k:30s} = {v:.4f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
