"""Training report — generates per-phase and cross-phase metric plots.

Produces two key charts per phase:
- **Episode Return** with MA(50) smoothing.
- **Critic (Value) Loss** with MA(50) smoothing.

Also generates a combined cross-phase summary plot.
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
        ma[-(i + 1)] = np.mean(arr[-(i + half + 1):])
    return ma


def generate_phase_report(
    phase_name: str,
    metrics: List[Dict[str, Any]],
    output_dir: str | Path,
    ma_window: int = 50,
) -> List[Path]:
    """Generate Episode Return and Critic Loss plots for a single phase.

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
            "Report generation requires matplotlib. "
            "Install with: pip install matplotlib"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    if not metrics:
        return generated

    updates = list(range(1, len(metrics) + 1))

    ep_returns = [m.get("mean_ep_reward", 0.0) for m in metrics]
    critic_losses = [m.get("value_loss", 0.0) for m in metrics]
    ep_ma = _moving_average(ep_returns, ma_window)
    critic_ma = _moving_average(critic_losses, ma_window)

    # ── Side-by-side: Return + Critic Loss ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Left — Episode Return (default blue raw, orange MA)
    axes[0].set_title(f"{phase_name}: Retorno por update")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Retorno")
    axes[0].plot(updates, ep_returns, alpha=0.4, label="Retorno")
    axes[0].plot(updates, ep_ma, label=f"MA({ma_window})")
    axes[0].legend()

    # Right — Critic Loss (orange raw, red MA)
    axes[1].set_title("Pérdida del crítico (TD error²)")
    axes[1].set_xlabel("Update")
    axes[1].set_ylabel("Critic Loss (promedio por update)")
    axes[1].plot(updates, critic_losses, alpha=0.4, color="orange",
                 label="Critic Loss")
    axes[1].plot(updates, critic_ma, color="red", label=f"MA({ma_window})")
    axes[1].legend()

    plt.tight_layout()
    path = output_dir / f"{phase_name}_report.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    generated.append(path)

    return generated


def generate_combined_report(
    all_metrics: Dict[str, List[Dict[str, Any]]],
    output_dir: str | Path,
    ma_window: int = 50,
) -> List[Path]:
    """Generate combined cross-phase comparison plots.

    Overlays all phases in a single figure for Episode Return and Critic Loss.

    Parameters
    ----------
    all_metrics : dict[str, list[dict]]
        Phase name → per-update metrics.
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

    # Colour palette for phases
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    # Matching darker shades for MA lines
    colors_ma = ["#0D47A1", "#1B5E20", "#E65100", "#4A148C", "#B71C1C"]

    # ── Combined side-by-side: Return + Critic Loss ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    global_update = 0

    for i, (phase_name, metrics) in enumerate(all_metrics.items()):
        if not metrics:
            continue
        n = len(metrics)
        updates = list(range(global_update + 1, global_update + n + 1))
        color = colors[i % len(colors)]
        color_ma = colors_ma[i % len(colors_ma)]

        # Left — Episode Return
        ep_returns = [m.get("mean_ep_reward", 0.0) for m in metrics]
        ep_ma = _moving_average(ep_returns, ma_window)
        axes[0].plot(updates, ep_returns, alpha=0.4, color=color, linewidth=0.7)
        axes[0].plot(updates, ep_ma, color=color_ma, linewidth=2.0,
                     label=f"{phase_name}")

        # Right — Critic Loss
        critic_losses = [m.get("value_loss", 0.0) for m in metrics]
        critic_ma = _moving_average(critic_losses, ma_window)
        axes[1].plot(updates, critic_losses, alpha=0.4, color=color,
                     linewidth=0.7)
        axes[1].plot(updates, critic_ma, color=color_ma, linewidth=2.0,
                     label=f"{phase_name}")

        # Phase separator
        if global_update > 0:
            axes[0].axvline(x=global_update + 0.5, color="gray",
                            linestyle="--", alpha=0.5)
            axes[1].axvline(x=global_update + 0.5, color="gray",
                            linestyle="--", alpha=0.5)

        global_update += n

    axes[0].set_title("Retorno por update — todas las fases")
    axes[0].set_xlabel("Update (global)")
    axes[0].set_ylabel("Retorno")
    axes[0].legend()

    axes[1].set_title("Pérdida del crítico (TD error²)")
    axes[1].set_xlabel("Update (global)")
    axes[1].set_ylabel("Critic Loss (promedio por update)")
    axes[1].legend()

    plt.tight_layout()
    path = output_dir / "combined_return_critic.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    generated.append(path)

    # ── Additional side-by-side: Policy Loss + Entropy ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    global_update = 0

    for i, (phase_name, metrics) in enumerate(all_metrics.items()):
        if not metrics:
            continue
        n = len(metrics)
        updates = list(range(global_update + 1, global_update + n + 1))
        color = colors[i % len(colors)]
        color_ma = colors_ma[i % len(colors_ma)]

        # Policy loss
        pl = [m.get("policy_loss", 0.0) for m in metrics]
        pl_ma = _moving_average(pl, ma_window)
        axes[0].plot(updates, pl, alpha=0.4, color=color, linewidth=0.7)
        axes[0].plot(updates, pl_ma, color=color_ma, linewidth=2.0,
                     label=f"{phase_name}")

        # Entropy
        ent = [m.get("entropy", 0.0) for m in metrics]
        ent_ma = _moving_average(ent, ma_window)
        axes[1].plot(updates, ent, alpha=0.4, color=color, linewidth=0.7)
        axes[1].plot(updates, ent_ma, color=color_ma, linewidth=2.0,
                     label=f"{phase_name}")

        if global_update > 0:
            axes[0].axvline(x=global_update + 0.5, color="gray",
                            linestyle="--", alpha=0.5)
            axes[1].axvline(x=global_update + 0.5, color="gray",
                            linestyle="--", alpha=0.5)

        global_update += n

    axes[0].set_title("Policy Loss por update")
    axes[0].set_xlabel("Update (global)")
    axes[0].set_ylabel("Policy Loss")
    axes[0].legend()

    axes[1].set_title("Entropía por update")
    axes[1].set_xlabel("Update (global)")
    axes[1].set_ylabel("Entropy")
    axes[1].legend()

    plt.tight_layout()
    path = output_dir / "combined_policy_entropy.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    generated.append(path)

    return generated


def generate_full_report(
    all_metrics: Dict[str, List[Dict[str, Any]]],
    output_dir: str | Path,
    ma_window: int = 50,
) -> List[Path]:
    """Generate all reports: per-phase + combined.

    Parameters
    ----------
    all_metrics : dict[str, list[dict]]
        Phase name → per-update metrics.
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
