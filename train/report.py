"""Training report — generates per-phase and cross-phase metric plots.

Per-phase report (3×2 grid):
- Episode Return + Win Rate
- Critic Loss + Policy Loss
- Entropy (H) + Entropy Choice (Hc)

Combined cross-phase report overlays all phases with phase separators.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _moving_average(values: List[float], window: int) -> np.ndarray:
    """Compute centred moving average with adaptive window.

    When the data has fewer points than *window*, the window is shrunk
    automatically so that smoothing is always applied (minimum window = 3).
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) <= 2:
        return arr
    # Adapt window to data length (at least 3)
    w = min(window, max(3, len(arr) // 2 | 1))  # enforce odd for centering
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    ma = np.convolve(arr, kernel, mode="same")
    # Fix edge bias from partial overlap
    half = w // 2
    for i in range(half):
        ma[i] = np.mean(arr[: i + half + 1])
        ma[-(i + 1)] = np.mean(arr[-(i + half + 1) :])
    return ma


# ═══════════════════════════════════════════════════════════════════════════════
# Colour palette — supports up to 10 phases
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_COLORS = [
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#F44336",  # red
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
    "#E91E63",  # pink
    "#CDDC39",  # lime
]
PHASE_COLORS_MA = [
    "#0D47A1",
    "#1B5E20",
    "#E65100",
    "#4A148C",
    "#B71C1C",
    "#006064",
    "#3E2723",
    "#263238",
    "#880E4F",
    "#827717",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers to extract eval metrics (win_rate, avg_crowns)
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_eval_series(
    metrics: List[Dict[str, Any]],
) -> tuple[list[int], list[float], list[float]]:
    """Extract eval updates, win_rate, and avg_crowns from metrics.

    Returns (update_indices, win_rates, avg_crowns) — only for updates
    that have an ``eval`` sub-dict.
    """
    updates: list[int] = []
    win_rates: list[float] = []
    crowns: list[float] = []
    for m in metrics:
        ev = m.get("eval")
        if ev is not None:
            updates.append(m.get("update", 0))
            win_rates.append(ev.get("win_rate", 0.0))
            crowns.append(ev.get("avg_crowns", 0.0))
    return updates, win_rates, crowns


# ═══════════════════════════════════════════════════════════════════════════════
# Per-phase report (3×2 grid)
# ═══════════════════════════════════════════════════════════════════════════════


def generate_phase_report(
    phase_name: str,
    metrics: List[Dict[str, Any]],
    output_dir: str | Path,
    ma_window: int = 50,
) -> List[Path]:
    """Generate a 3×2 report for a single phase.

    Panels: Return, Win Rate, Critic Loss, Policy Loss, Entropy (H),
    Entropy Choice (Hc).

    Parameters
    ----------
    phase_name : str
        Phase name (used in titles and filenames).
    metrics : list[dict]
        Per-update metric dicts from the trainer callback.
    output_dir : str | Path
        Directory to save the PNG files.
    ma_window : int
        Moving average window size (default 50).

    Returns
    -------
    list[Path]
        Paths to the generated PNG files.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Report generation requires matplotlib. Install with: pip install matplotlib"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    if not metrics:
        return generated

    updates = list(range(1, len(metrics) + 1))

    # ── Extract per-update series ─────────────────────────────────────────
    ep_returns = [m.get("mean_ep_reward", 0.0) for m in metrics]
    critic_losses = [m.get("value_loss", 0.0) for m in metrics]
    policy_losses = [m.get("policy_loss", 0.0) for m in metrics]
    entropies = [m.get("entropy", 0.0) for m in metrics]
    entropies_c = [m.get("entropy_choice", 0.0) for m in metrics]

    ep_ma = _moving_average(ep_returns, ma_window)
    critic_ma = _moving_average(critic_losses, ma_window)
    policy_ma = _moving_average(policy_losses, ma_window)
    ent_ma = _moving_average(entropies, ma_window)
    ent_c_ma = _moving_average(entropies_c, ma_window)

    eval_updates, win_rates, avg_crowns = _extract_eval_series(metrics)

    # ── 3×2 figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Phase: {phase_name}", fontsize=14, fontweight="bold")

    # (0,0) — Episode Return
    ax = axes[0, 0]
    ax.set_title("Retorno por episodio")
    ax.set_xlabel("Update")
    ax.set_ylabel("Retorno")
    ax.plot(updates, ep_returns, alpha=0.35, color="#2196F3", linewidth=0.8)
    ax.plot(updates, ep_ma, color="#0D47A1", linewidth=2, label=f"MA({ma_window})")
    ax.legend(fontsize=8)

    # (0,1) — Win Rate + Avg Crowns
    ax = axes[0, 1]
    ax.set_title("Win Rate & Coronas")
    ax.set_xlabel("Update")
    ax.set_ylabel("Win Rate", color="#4CAF50")
    if eval_updates:
        ax.plot(eval_updates, win_rates, "o-", color="#4CAF50", markersize=4, label="Win Rate")
        ax.set_ylim(-0.05, 1.05)
        ax2 = ax.twinx()
        ax2.plot(
            eval_updates, avg_crowns, "s--", color="#FF9800", markersize=4, label="Avg Crowns"
        )
        ax2.set_ylabel("Avg Crowns", color="#FF9800")
        ax2.tick_params(axis="y", labelcolor="#FF9800")
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No eval data", transform=ax.transAxes, ha="center", va="center")

    # (1,0) — Critic Loss
    ax = axes[1, 0]
    ax.set_title("Pérdida del crítico (TD error²)")
    ax.set_xlabel("Update")
    ax.set_ylabel("Critic Loss")
    ax.plot(updates, critic_losses, alpha=0.35, color="#FF9800", linewidth=0.8)
    ax.plot(updates, critic_ma, color="#E65100", linewidth=2, label=f"MA({ma_window})")
    ax.legend(fontsize=8)

    # (1,1) — Policy Loss
    ax = axes[1, 1]
    ax.set_title("Policy Loss")
    ax.set_xlabel("Update")
    ax.set_ylabel("Policy Loss")
    ax.plot(updates, policy_losses, alpha=0.35, color="#9C27B0", linewidth=0.8)
    ax.plot(updates, policy_ma, color="#4A148C", linewidth=2, label=f"MA({ma_window})")
    ax.legend(fontsize=8)

    # (2,0) — Entropy
    ax = axes[2, 0]
    ax.set_title("Entropía (H)")
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy")
    ax.plot(updates, entropies, alpha=0.35, color="#00BCD4", linewidth=0.8)
    ax.plot(updates, ent_ma, color="#006064", linewidth=2, label=f"MA({ma_window})")
    ax.legend(fontsize=8)

    # (2,1) — Entropy Choice (Hc)
    ax = axes[2, 1]
    ax.set_title("Entropía Choice (Hc)")
    ax.set_xlabel("Update")
    ax.set_ylabel("Hc")
    ax.plot(updates, entropies_c, alpha=0.35, color="#E91E63", linewidth=0.8)
    ax.plot(updates, ent_c_ma, color="#880E4F", linewidth=2, label=f"MA({ma_window})")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / f"{phase_name}_report.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    generated.append(path)

    return generated


# ═══════════════════════════════════════════════════════════════════════════════
# Combined cross-phase report (3×2 grid, all phases overlaid)
# ═══════════════════════════════════════════════════════════════════════════════


def generate_combined_report(
    all_metrics: Dict[str, List[Dict[str, Any]]],
    output_dir: str | Path,
    ma_window: int = 50,
) -> List[Path]:
    """Generate combined cross-phase comparison plots.

    Overlays all phases in a 3×2 grid: Return, Win Rate, Critic Loss,
    Policy Loss, Entropy (H), Entropy Choice (Hc).

    Parameters
    ----------
    all_metrics : dict[str, list[dict]]
        Phase name -> per-update metrics.
    output_dir : str | Path
        Directory to save PNGs.
    ma_window : int
        Moving average window.

    Returns
    -------
    list[Path]
        Paths to generated PNGs.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    if not all_metrics:
        return generated

    # ── 3×2 combined figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Curriculum Training — All Phases", fontsize=14, fontweight="bold")

    global_update = 0
    # Collect eval data with global offsets
    all_eval_updates: list[list[int]] = []
    all_win_rates: list[list[float]] = []
    all_crowns: list[list[float]] = []

    for i, (phase_name, metrics) in enumerate(all_metrics.items()):
        if not metrics:
            continue
        n = len(metrics)
        updates = list(range(global_update + 1, global_update + n + 1))
        color = PHASE_COLORS[i % len(PHASE_COLORS)]
        color_ma = PHASE_COLORS_MA[i % len(PHASE_COLORS_MA)]

        # Phase separator
        if global_update > 0:
            for row in range(3):
                for col in range(2):
                    axes[row, col].axvline(
                        x=global_update + 0.5, color="gray", linestyle="--", alpha=0.4
                    )

        # ── (0,0) Episode Return ─────────────────────────────────────────
        ep_returns = [m.get("mean_ep_reward", 0.0) for m in metrics]
        ep_ma = _moving_average(ep_returns, ma_window)
        axes[0, 0].plot(updates, ep_returns, alpha=0.3, color=color, linewidth=0.6)
        axes[0, 0].plot(updates, ep_ma, color=color_ma, linewidth=1.8, label=phase_name)

        # ── (0,1) Win Rate (scatter) ─────────────────────────────────────
        eval_up, wrs, crs = _extract_eval_series(metrics)
        # Offset eval updates to global x-axis
        eval_up_global = [u + global_update for u in eval_up]
        all_eval_updates.append(eval_up_global)
        all_win_rates.append(wrs)
        all_crowns.append(crs)
        if eval_up_global:
            axes[0, 1].plot(
                eval_up_global, wrs, "o-", color=color_ma, markersize=3,
                linewidth=1.5, label=phase_name,
            )

        # ── (1,0) Critic Loss ────────────────────────────────────────────
        vl = [m.get("value_loss", 0.0) for m in metrics]
        vl_ma = _moving_average(vl, ma_window)
        axes[1, 0].plot(updates, vl, alpha=0.3, color=color, linewidth=0.6)
        axes[1, 0].plot(updates, vl_ma, color=color_ma, linewidth=1.8, label=phase_name)

        # ── (1,1) Policy Loss ────────────────────────────────────────────
        pl = [m.get("policy_loss", 0.0) for m in metrics]
        pl_ma = _moving_average(pl, ma_window)
        axes[1, 1].plot(updates, pl, alpha=0.3, color=color, linewidth=0.6)
        axes[1, 1].plot(updates, pl_ma, color=color_ma, linewidth=1.8, label=phase_name)

        # ── (2,0) Entropy ────────────────────────────────────────────────
        ent = [m.get("entropy", 0.0) for m in metrics]
        ent_ma = _moving_average(ent, ma_window)
        axes[2, 0].plot(updates, ent, alpha=0.3, color=color, linewidth=0.6)
        axes[2, 0].plot(updates, ent_ma, color=color_ma, linewidth=1.8, label=phase_name)

        # ── (2,1) Entropy Choice (Hc) ────────────────────────────────────
        hc = [m.get("entropy_choice", 0.0) for m in metrics]
        hc_ma = _moving_average(hc, ma_window)
        axes[2, 1].plot(updates, hc, alpha=0.3, color=color, linewidth=0.6)
        axes[2, 1].plot(updates, hc_ma, color=color_ma, linewidth=1.8, label=phase_name)

        global_update += n

    # ── Titles / labels ───────────────────────────────────────────────────
    panel_cfg = [
        (0, 0, "Retorno por update", "Update (global)", "Retorno"),
        (0, 1, "Win Rate", "Update (global)", "Win Rate"),
        (1, 0, "Pérdida del crítico", "Update (global)", "Critic Loss"),
        (1, 1, "Policy Loss", "Update (global)", "Policy Loss"),
        (2, 0, "Entropía (H)", "Update (global)", "Entropy"),
        (2, 1, "Entropía Choice (Hc)", "Update (global)", "Hc"),
    ]
    for r, c, title, xlabel, ylabel in panel_cfg:
        axes[r, c].set_title(title)
        axes[r, c].set_xlabel(xlabel)
        axes[r, c].set_ylabel(ylabel)
        axes[r, c].legend(fontsize=7, loc="best")

    axes[0, 1].set_ylim(-0.05, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "combined_report.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    generated.append(path)

    # ── Separate Avg Crowns plot ──────────────────────────────────────────
    fig2, ax_cr = plt.subplots(figsize=(10, 4))
    ax_cr.set_title("Avg Crowns por fase")
    ax_cr.set_xlabel("Update (global)")
    ax_cr.set_ylabel("Avg Crowns")
    for i, (phase_name, _) in enumerate(all_metrics.items()):
        if i < len(all_eval_updates) and all_eval_updates[i]:
            color_ma = PHASE_COLORS_MA[i % len(PHASE_COLORS_MA)]
            ax_cr.plot(
                all_eval_updates[i], all_crowns[i], "s-", color=color_ma,
                markersize=3, linewidth=1.5, label=phase_name,
            )
    ax_cr.legend(fontsize=8)
    plt.tight_layout()
    path2 = output_dir / "combined_crowns.png"
    fig2.savefig(str(path2), dpi=150)
    plt.close(fig2)
    generated.append(path2)

    return generated


# ═══════════════════════════════════════════════════════════════════════════════
# Full report entry-point
# ═══════════════════════════════════════════════════════════════════════════════


def generate_full_report(
    all_metrics: Dict[str, List[Dict[str, Any]]],
    output_dir: str | Path,
    ma_window: int = 50,
) -> List[Path]:
    """Generate all reports: per-phase + combined.

    Parameters
    ----------
    all_metrics : dict[str, list[dict]]
        Phase name -> per-update metrics.
    output_dir : str | Path
        Root directory for reports.
    ma_window : int
        Moving average window.

    Returns
    -------
    list[Path]
        All generated PNG paths.
    """
    output_dir = Path(output_dir)
    all_generated: List[Path] = []

    # Per-phase reports
    for phase_name, metrics in all_metrics.items():
        phase_dir = output_dir / "phases"
        paths = generate_phase_report(phase_name, metrics, phase_dir, ma_window)
        all_generated.extend(paths)

    # Combined reports
    combined_paths = generate_combined_report(all_metrics, output_dir, ma_window)
    all_generated.extend(combined_paths)

    print(f"\nGenerated {len(all_generated)} report plots in {output_dir}/")
    for p in all_generated:
        print(f"  {p.relative_to(output_dir)}")

    return all_generated
