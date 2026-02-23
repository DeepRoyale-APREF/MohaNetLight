#!/usr/bin/env python3
"""Compare trained MohaNet vs Baseline model — fair head-to-head evaluation.

Loads checkpoints from both models, runs a tournament against each other
AND against the heuristic bot roster, then produces a structured comparison
with ≥ 2 metrics (win rate, avg crowns, plus optional training-curve stats).

Designed for academic investigation:
- RQ1: Does HRL (MohaNet) outperform flat architecture (ConvLSTM)?
- RQ2: How does reward shaping affect each architecture?  (compare logs)
- RQ3: How robust is each model against diverse opponents?

Examples
--------
Compare two checkpoints:

    python scripts/compare_models.py \\
        --mohanet-ckpt logs/mohanet/mohanet_u200.pt \\
        --baseline-ckpt logs/conv_lstm/conv_lstm_u200.pt \\
        --baseline-type conv_lstm \\
        --matches-per-pair 10

Add training metrics from two runs for learning-curve comparison:

    python scripts/compare_models.py \\
        --mohanet-ckpt logs/mohanet/mohanet_u200.pt \\
        --baseline-ckpt logs/conv_lstm/conv_lstm_u200.pt \\
        --mohanet-metrics logs/mohanet/metrics.json \\
        --baseline-metrics logs/conv_lstm/metrics.json \\
        --output-dir logs/comparison
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clash_royale_gymnasium.league.match import MatchResult, run_match
from clash_royale_gymnasium.league.player_slot import PlayerSlot

from mohanetlight.baseline.config import BaselineConfig
from mohanetlight.bots.strategies import default_bot_roster
from mohanetlight.config import ModelConfig
from mohanetlight.inference.agent import MohaNetAgent
from mohanetlight.inference.baseline_agent import BaselineAgent


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation engine
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_agent_vs_bots(
    agent: PlayerSlot,
    opponents: List[PlayerSlot],
    matches_per_pair: int = 10,
    frame_skip: int = 10,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """Run agent against each opponent and aggregate metrics.

    Returns
    -------
    dict with keys:
        overall_win_rate, overall_avg_crowns, per_opponent (list of dicts),
        total_wins, total_matches, results (list of MatchResult-like dicts)
    """
    per_opp: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    total_wins = 0
    total_crowns = 0
    total_matches = 0

    for opp in opponents:
        opp_wins = 0
        opp_crowns = 0

        for m in range(matches_per_pair):
            # Alternate sides for fairness
            if m % 2 == 0:
                p0, p1 = agent, opp
                agent_pid = 0
            else:
                p0, p1 = opp, agent
                agent_pid = 1

            result = run_match(
                p0, p1,
                frame_skip=frame_skip,
                seed=seed_offset + total_matches,
            )

            won = result.winner == agent_pid
            crowns = (
                result.p0_towers_destroyed if agent_pid == 0
                else result.p1_towers_destroyed
            )

            if won:
                opp_wins += 1
                total_wins += 1
            opp_crowns += crowns
            total_crowns += crowns
            total_matches += 1

            all_results.append({
                "opponent": opp.name,
                "agent_side": agent_pid,
                "winner": result.winner,
                "agent_won": won,
                "agent_crowns": crowns,
                "game_duration": result.game_duration,
            })

        per_opp.append({
            "opponent": opp.name,
            "wins": opp_wins,
            "matches": matches_per_pair,
            "win_rate": opp_wins / matches_per_pair,
            "avg_crowns": opp_crowns / matches_per_pair,
        })

    return {
        "overall_win_rate": total_wins / max(total_matches, 1),
        "overall_avg_crowns": total_crowns / max(total_matches, 1),
        "total_wins": total_wins,
        "total_matches": total_matches,
        "per_opponent": per_opp,
        "results": all_results,
    }


def head_to_head(
    agent_a: PlayerSlot,
    agent_b: PlayerSlot,
    n_matches: int = 20,
    frame_skip: int = 10,
    seed_offset: int = 10000,
) -> Dict[str, Any]:
    """Play two agents directly against each other.

    Returns
    -------
    dict with a_wins, b_wins, draws, a_win_rate, a_avg_crowns, b_avg_crowns
    """
    a_wins = 0
    b_wins = 0
    draws = 0
    a_crowns = 0
    b_crowns = 0

    for m in range(n_matches):
        if m % 2 == 0:
            p0, p1 = agent_a, agent_b
            a_pid = 0
        else:
            p0, p1 = agent_b, agent_a
            a_pid = 1

        result = run_match(p0, p1, frame_skip=frame_skip, seed=seed_offset + m)

        a_cr = result.p0_towers_destroyed if a_pid == 0 else result.p1_towers_destroyed
        b_cr = result.p1_towers_destroyed if a_pid == 0 else result.p0_towers_destroyed
        a_crowns += a_cr
        b_crowns += b_cr

        if result.winner == a_pid:
            a_wins += 1
        elif result.winner is not None:
            b_wins += 1
        else:
            draws += 1

    return {
        "a_name": agent_a.name,
        "b_name": agent_b.name,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "total": n_matches,
        "a_win_rate": a_wins / max(n_matches, 1),
        "b_win_rate": b_wins / max(n_matches, 1),
        "a_avg_crowns": a_crowns / max(n_matches, 1),
        "b_avg_crowns": b_crowns / max(n_matches, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training curve comparison
# ═══════════════════════════════════════════════════════════════════════════════


def compare_training_curves(
    mohanet_metrics_path: Optional[str],
    baseline_metrics_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Extract key training-curve statistics for comparison.

    Metrics analysed:
    - Sample efficiency: steps to reach specific win-rate thresholds
    - Stability: variance of episode returns over last 25% of training
    - Final performance: mean return over last 10% of updates
    """
    if not mohanet_metrics_path or not baseline_metrics_path:
        return None

    def _load_metrics(path: str) -> List[Dict[str, Any]]:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            # all_phases_metrics.json format: {phase_name: [metrics]}
            combined = []
            for metrics in data.values():
                combined.extend(metrics)
            return combined
        return data

    def _analyse(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        returns = [m.get("mean_ep_reward", 0.0) for m in metrics]
        n = len(returns)
        if n == 0:
            return {}

        # Final performance — mean over last 10%
        tail = max(1, n // 10)
        final_mean = float(np.mean(returns[-tail:]))
        final_std = float(np.std(returns[-tail:]))

        # Stability — variance over last 25%
        quarter = max(1, n // 4)
        stability_std = float(np.std(returns[-quarter:]))

        # Sample efficiency — steps to first eval with win_rate > thresholds
        thresholds = {0.3: None, 0.5: None, 0.7: None}
        for m in metrics:
            ev = m.get("eval", {})
            wr = ev.get("win_rate")
            step = m.get("global_step", 0)
            if wr is not None:
                for thr in thresholds:
                    if thresholds[thr] is None and wr >= thr:
                        thresholds[thr] = step

        return {
            "total_updates": n,
            "final_mean_return": round(final_mean, 4),
            "final_std_return": round(final_std, 4),
            "stability_std_last_25pct": round(stability_std, 4),
            "steps_to_30pct_wr": thresholds[0.3],
            "steps_to_50pct_wr": thresholds[0.5],
            "steps_to_70pct_wr": thresholds[0.7],
        }

    try:
        m_metrics = _load_metrics(mohanet_metrics_path)
        b_metrics = _load_metrics(baseline_metrics_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not load training metrics: {e}")
        return None

    return {
        "mohanet": _analyse(m_metrics),
        "baseline": _analyse(b_metrics),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_comparison_report(
    comparison: Dict[str, Any],
    output_dir: Path,
) -> List[Path]:
    """Generate comparison plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping comparison plots.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    # ── Bar chart: Win rate vs each opponent ──────────────────────────────
    mohanet_eval = comparison.get("mohanet_vs_bots", {})
    baseline_eval = comparison.get("baseline_vs_bots", {})

    if mohanet_eval and baseline_eval:
        m_per_opp = mohanet_eval.get("per_opponent", [])
        b_per_opp = baseline_eval.get("per_opponent", [])

        if m_per_opp and b_per_opp:
            opp_names = [o["opponent"] for o in m_per_opp]
            m_wr = [o["win_rate"] for o in m_per_opp]
            b_wr = [o["win_rate"] for o in b_per_opp]

            x = np.arange(len(opp_names))
            width = 0.35

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            # Win rate
            axes[0].bar(x - width / 2, m_wr, width, label="MohaNet", color="#2196F3")
            axes[0].bar(x + width / 2, b_wr, width, label="Baseline", color="#FF9800")
            axes[0].set_ylabel("Win Rate")
            axes[0].set_title("Win Rate vs Heuristic Bots")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(opp_names, rotation=30, ha="right", fontsize=8)
            axes[0].legend()
            axes[0].set_ylim(0, 1.05)

            # Avg crowns
            m_cr = [o["avg_crowns"] for o in m_per_opp]
            b_cr = [o["avg_crowns"] for o in b_per_opp]
            axes[1].bar(x - width / 2, m_cr, width, label="MohaNet", color="#2196F3")
            axes[1].bar(x + width / 2, b_cr, width, label="Baseline", color="#FF9800")
            axes[1].set_ylabel("Avg Crowns")
            axes[1].set_title("Avg Crowns Destroyed vs Heuristic Bots")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(opp_names, rotation=30, ha="right", fontsize=8)
            axes[1].legend()

            plt.tight_layout()
            path = output_dir / "winrate_crowns_comparison.png"
            fig.savefig(str(path), dpi=150)
            plt.close(fig)
            generated.append(path)

    # ── Head-to-head summary ─────────────────────────────────────────────
    h2h = comparison.get("head_to_head")
    if h2h:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [h2h["a_name"], h2h["b_name"], "Draws"]
        values = [h2h["a_wins"], h2h["b_wins"], h2h["draws"]]
        colors = ["#2196F3", "#FF9800", "#9E9E9E"]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Matches")
        ax.set_title(
            f"Head-to-Head ({h2h['total']} matches)\n"
            f"{h2h['a_name']}: {h2h['a_win_rate']:.0%} WR  |  "
            f"{h2h['b_name']}: {h2h['b_win_rate']:.0%} WR"
        )
        plt.tight_layout()
        path = output_dir / "head_to_head.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        generated.append(path)

    return generated


def print_comparison_table(comparison: Dict[str, Any]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 74)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 74)

    # ── Summary metrics ──────────────────────────────────────────────────
    m_eval = comparison.get("mohanet_vs_bots", {})
    b_eval = comparison.get("baseline_vs_bots", {})
    b_type = comparison.get("baseline_type", "baseline")

    if m_eval and b_eval:
        print(f"\n{'Metric':<30} {'MohaNet':>14} {b_type:>14}")
        print("-" * 60)
        print(
            f"{'Win Rate (vs bots)' :<30} "
            f"{m_eval['overall_win_rate']:>13.1%} "
            f"{b_eval['overall_win_rate']:>13.1%}"
        )
        print(
            f"{'Avg Crowns (vs bots)' :<30} "
            f"{m_eval['overall_avg_crowns']:>14.2f} "
            f"{b_eval['overall_avg_crowns']:>14.2f}"
        )

        # Per-opponent breakdown
        m_per = {o["opponent"]: o for o in m_eval.get("per_opponent", [])}
        b_per = {o["opponent"]: o for o in b_eval.get("per_opponent", [])}
        if m_per:
            print(f"\n{'  Per-Opponent WR':<30} {'MohaNet':>14} {b_type:>14}")
            print("  " + "-" * 56)
            for opp_name in m_per:
                m_wr = m_per[opp_name]["win_rate"]
                b_wr = b_per.get(opp_name, {}).get("win_rate", 0)
                marker = " *" if abs(m_wr - b_wr) > 0.15 else ""
                print(
                    f"  {opp_name:<28} {m_wr:>13.1%} {b_wr:>13.1%}{marker}"
                )

    # ── Head-to-head ─────────────────────────────────────────────────────
    h2h = comparison.get("head_to_head")
    if h2h:
        print(f"\n  Head-to-Head ({h2h['total']} matches):")
        print(f"    {h2h['a_name']}: {h2h['a_wins']} wins ({h2h['a_win_rate']:.0%}), "
              f"avg crowns={h2h['a_avg_crowns']:.2f}")
        print(f"    {h2h['b_name']}: {h2h['b_wins']} wins ({h2h['b_win_rate']:.0%}), "
              f"avg crowns={h2h['b_avg_crowns']:.2f}")
        print(f"    Draws: {h2h['draws']}")

    # ── Training curve stats ─────────────────────────────────────────────
    curve = comparison.get("training_curves")
    if curve:
        print(f"\n{'Training Curve':<30} {'MohaNet':>14} {b_type:>14}")
        print("-" * 60)
        mc = curve.get("mohanet", {})
        bc = curve.get("baseline", {})
        for key, label in [
            ("final_mean_return", "Final Mean Return"),
            ("final_std_return", "Final Std Return"),
            ("stability_std_last_25pct", "Stability (σ last 25%)"),
            ("steps_to_30pct_wr", "Steps → 30% WR"),
            ("steps_to_50pct_wr", "Steps → 50% WR"),
            ("steps_to_70pct_wr", "Steps → 70% WR"),
        ]:
            mv = mc.get(key, "N/A")
            bv = bc.get(key, "N/A")
            if isinstance(mv, float):
                mv = f"{mv:.4f}"
            if isinstance(bv, float):
                bv = f"{bv:.4f}"
            mv = str(mv) if mv is not None else "—"
            bv = str(bv) if bv is not None else "—"
            print(f"{'  ' + label:<30} {mv:>14} {bv:>14}")

    # ── Model info ───────────────────────────────────────────────────────
    info = comparison.get("model_info")
    if info:
        print(f"\n  Model Info:")
        for name, mi in info.items():
            print(f"    {name}: {mi.get('params', '?'):,} params, "
                  f"type={mi.get('type', '?')}")

    print("\n" + "=" * 74)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Compare MohaNet vs Baseline model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mohanet-ckpt",
        type=str,
        required=True,
        help="Path to MohaNet checkpoint (.pt).",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        required=True,
        help="Path to baseline checkpoint (.pt).",
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        default="conv_lstm",
        choices=["conv_lstm", "flat_mlp"],
        help="Baseline model architecture.",
    )
    parser.add_argument(
        "--mohanet-metrics",
        type=str,
        default=None,
        help="Path to MohaNet training metrics JSON (for curve comparison).",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default=None,
        help="Path to baseline training metrics JSON (for curve comparison).",
    )
    parser.add_argument(
        "--matches-per-pair",
        type=int,
        default=10,
        help="Matches per opponent pair in bot evaluation.",
    )
    parser.add_argument(
        "--head-to-head-matches",
        type=int,
        default=20,
        help="Direct matches between MohaNet and Baseline.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Frame skip for evaluation matches.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./logs/comparison",
        help="Output directory for results and plots.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────
    print("Loading MohaNet checkpoint...")
    mohanet_agent = MohaNetAgent.from_checkpoint(
        path=args.mohanet_ckpt,
        name="MohaNet",
        device=args.device,
        deterministic=True,
    )

    print(f"Loading Baseline ({args.baseline_type}) checkpoint...")
    baseline_agent = BaselineAgent.from_checkpoint(
        path=args.baseline_ckpt,
        name=f"Baseline-{args.baseline_type}",
        device=args.device,
        model_type=args.baseline_type,
        deterministic=True,
    )

    # ── Evaluation bots ───────────────────────────────────────────────────
    eval_bots = [
        default_bot_roster()[0],   # GiantPush
        default_bot_roster()[2],   # BridgeSpam
        default_bot_roster()[4],   # SpellCycle
        default_bot_roster()[6],   # DefCounter
        default_bot_roster()[8],   # Balanced
    ]

    comparison: Dict[str, Any] = {
        "baseline_type": args.baseline_type,
        "model_info": {
            "MohaNet": mohanet_agent.metadata(),
            f"Baseline-{args.baseline_type}": baseline_agent.metadata(),
        },
    }

    # ── Metric 1 & 2: Win rate + Avg crowns vs bots ──────────────────────
    print(f"\nEvaluating MohaNet vs {len(eval_bots)} bots "
          f"({args.matches_per_pair} matches each)...")
    t0 = time.time()
    comparison["mohanet_vs_bots"] = evaluate_agent_vs_bots(
        mohanet_agent, eval_bots,
        matches_per_pair=args.matches_per_pair,
        frame_skip=args.frame_skip,
        seed_offset=0,
    )
    print(f"  MohaNet done in {time.time() - t0:.1f}s — "
          f"WR={comparison['mohanet_vs_bots']['overall_win_rate']:.1%}")

    print(f"\nEvaluating Baseline vs {len(eval_bots)} bots "
          f"({args.matches_per_pair} matches each)...")
    t0 = time.time()
    comparison["baseline_vs_bots"] = evaluate_agent_vs_bots(
        baseline_agent, eval_bots,
        matches_per_pair=args.matches_per_pair,
        frame_skip=args.frame_skip,
        seed_offset=5000,
    )
    print(f"  Baseline done in {time.time() - t0:.1f}s — "
          f"WR={comparison['baseline_vs_bots']['overall_win_rate']:.1%}")

    # ── Head-to-head ─────────────────────────────────────────────────────
    print(f"\nHead-to-head: MohaNet vs Baseline ({args.head_to_head_matches} matches)...")
    t0 = time.time()
    comparison["head_to_head"] = head_to_head(
        mohanet_agent, baseline_agent,
        n_matches=args.head_to_head_matches,
        frame_skip=args.frame_skip,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Training curve comparison ─────────────────────────────────────────
    curve_data = compare_training_curves(
        args.mohanet_metrics, args.baseline_metrics,
    )
    if curve_data:
        comparison["training_curves"] = curve_data

    # ── Print results ─────────────────────────────────────────────────────
    print_comparison_table(comparison)

    # ── Save JSON ─────────────────────────────────────────────────────────
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # ── Generate plots ────────────────────────────────────────────────────
    report_dir = output_dir / "plots"
    plots = generate_comparison_report(comparison, report_dir)
    if plots:
        print(f"Generated {len(plots)} comparison plots in {report_dir}/")

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
