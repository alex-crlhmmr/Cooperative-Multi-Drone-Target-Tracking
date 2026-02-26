"""Diagnostic visualization for filter comparison.

All plots accept multiple filters for side-by-side comparison.
Designed for paper-ready output with publication-quality formatting.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


FILTER_COLORS = {
    "EKF": "#2196F3",
    "UKF": "#FF9800",
    "PF": "#4CAF50",
    "IMM": "#E91E63",
    "ConsensusEKF": "#7B1FA2",
    "ConsensusIMM": "#D32F2F",
}
FILTER_STYLES = {
    "EKF": "-",
    "UKF": "--",
    "PF": "-.",
    "IMM": "-",
    "ConsensusEKF": "--",
    "ConsensusIMM": "-.",
}

# Ordered patterns for substring matching (more specific first)
_COLOR_PATTERNS = [
    ("Consensus IMM", "#D32F2F"),   # deep red
    ("Consensus EKF", "#7B1FA2"),   # purple
    ("IMM", "#E91E63"),             # pink
    ("PF", "#4CAF50"),              # green
    ("UKF", "#FF9800"),             # orange
    ("EKF", "#2196F3"),             # blue
]
_STYLE_PATTERNS = [
    ("Consensus IMM", "-."),
    ("Consensus EKF", "--"),
    ("IMM", (0, (5, 1))),           # densely dashed
    ("PF", "-."),
    ("UKF", "--"),
    ("EKF", "-"),
]


def _get_color(name):
    # Exact match first
    if name in FILTER_COLORS:
        return FILTER_COLORS[name]
    # Substring match
    for pattern, color in _COLOR_PATTERNS:
        if pattern in name:
            return color
    return "#999999"


def _get_style(name):
    if name in FILTER_STYLES:
        return FILTER_STYLES[name]
    for pattern, style in _STYLE_PATTERNS:
        if pattern in name:
            return style
    return "-"


def _save_or_show(fig, save_path):
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_state_estimates(
    time: np.ndarray,
    true_states: np.ndarray,
    estimates: np.ndarray,
    covariances: np.ndarray,
    filter_names: list[str],
    save_path: str | None = None,
):
    """6-panel plot: px,py,pz,vx,vy,vz with truth, estimates, and 3-sigma bounds.

    Args:
        time: (T,) time axis
        true_states: (T, 6) ground truth
        estimates: (n_filters, T, 6)
        covariances: (n_filters, T, 6, 6)
        filter_names: list of filter names
    """
    labels = ["px (m)", "py (m)", "pz (m)", "vx (m/s)", "vy (m/s)", "vz (m/s)"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for dim in range(6):
        ax = axes[dim]
        ax.plot(time, true_states[:, dim], "k-", linewidth=1.5, label="Truth", alpha=0.8)

        for fi, name in enumerate(filter_names):
            color = _get_color(name)
            style = _get_style(name)
            ax.plot(time, estimates[fi, :, dim], color=color, linestyle=style,
                    linewidth=1.0, label=name, alpha=0.85)

            # 3-sigma bounds
            sigma = np.sqrt(covariances[fi, :, dim, dim])
            ax.fill_between(
                time,
                estimates[fi, :, dim] - 3 * sigma,
                estimates[fi, :, dim] + 3 * sigma,
                color=color, alpha=0.08,
            )

        ax.set_ylabel(labels[dim])
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("State Estimates vs Ground Truth", fontsize=14)
    _save_or_show(fig, save_path)


def plot_3d_estimate_trajectory(
    true_states: np.ndarray,
    estimates: np.ndarray,
    filter_names: list[str],
    save_path: str | None = None,
):
    """3D trajectory plot with true and estimated paths."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        true_states[:, 0], true_states[:, 1], true_states[:, 2],
        "k-", linewidth=2, label="Truth", alpha=0.8,
    )
    ax.scatter(*true_states[0, :3], color="k", marker="*", s=150, zorder=5)

    for fi, name in enumerate(filter_names):
        color = _get_color(name)
        style = _get_style(name)
        ax.plot(
            estimates[fi, :, 0], estimates[fi, :, 1], estimates[fi, :, 2],
            color=color, linestyle=style, linewidth=1.2, label=name, alpha=0.8,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Estimated Trajectories")
    ax.legend()
    _save_or_show(fig, save_path)


def plot_rmse(
    time: np.ndarray,
    true_states: np.ndarray,
    estimates: np.ndarray,
    filter_names: list[str],
    save_path: str | None = None,
):
    """Position and velocity RMSE over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for fi, name in enumerate(filter_names):
        color = _get_color(name)
        style = _get_style(name)
        pos_err = np.linalg.norm(true_states[:, :3] - estimates[fi, :, :3], axis=1)
        vel_err = np.linalg.norm(true_states[:, 3:6] - estimates[fi, :, 3:6], axis=1)

        ax1.plot(time, pos_err, color=color, linestyle=style, linewidth=1.2, label=name)
        ax2.plot(time, vel_err, color=color, linestyle=style, linewidth=1.2, label=name)

    ax1.set_ylabel("Position Error (m)")
    ax1.set_title("Estimation Error Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2.set_ylabel("Velocity Error (m/s)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    _save_or_show(fig, save_path)


def plot_nees(
    time: np.ndarray,
    nees: np.ndarray,
    filter_names: list[str],
    state_dim: int = 6,
    save_path: str | None = None,
):
    """NEES over time with chi-squared 95% confidence bounds.

    Args:
        nees: (n_filters, T) NEES values
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Chi-squared 95% bounds for NEES
    T = nees.shape[1]
    lower = stats.chi2.ppf(0.025, state_dim)
    upper = stats.chi2.ppf(0.975, state_dim)

    ax.axhline(state_dim, color="gray", linestyle=":", linewidth=1, label=f"Expected ({state_dim})")
    ax.axhline(lower, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(upper, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"95% bounds [{lower:.1f}, {upper:.1f}]")

    for fi, name in enumerate(filter_names):
        color = _get_color(name)
        style = _get_style(name)
        # Clip for display (NEES can be huge before convergence)
        nees_clipped = np.clip(nees[fi], 0, state_dim * 20)
        ax.plot(time, nees_clipped, color=color, linestyle=style,
                linewidth=1.0, label=name, alpha=0.7)

    ax.set_ylabel("NEES")
    ax.set_xlabel("Time (s)")
    ax.set_title("Normalized Estimation Error Squared (NEES)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, state_dim * 10)
    _save_or_show(fig, save_path)


def plot_covariance_trace(
    time: np.ndarray,
    covariances: np.ndarray,
    filter_names: list[str],
    save_path: str | None = None,
):
    """Trace of covariance matrix over time (convergence rate)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for fi, name in enumerate(filter_names):
        color = _get_color(name)
        style = _get_style(name)
        # Position trace (first 3 diagonal elements)
        pos_trace = np.array([np.trace(covariances[fi, t, :3, :3])
                              for t in range(covariances.shape[1])])
        # Velocity trace (last 3 diagonal elements)
        vel_trace = np.array([np.trace(covariances[fi, t, 3:, 3:])
                              for t in range(covariances.shape[1])])

        ax1.plot(time, pos_trace, color=color, linestyle=style, linewidth=1.2, label=name)
        ax2.plot(time, vel_trace, color=color, linestyle=style, linewidth=1.2, label=name)

    ax1.set_ylabel("tr(P_pos) (m²)")
    ax1.set_title("Covariance Trace — Convergence Rate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2.set_ylabel("tr(P_vel) (m²/s²)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    _save_or_show(fig, save_path)


def plot_monte_carlo_comparison(
    results: dict,
    filter_names: list[str],
    traj_types: list[str],
    save_path: str | None = None,
):
    """Box plots comparing filters across MC runs.

    Args:
        results: dict with keys like (filter_name, traj_type) -> dict of metric arrays
            Each inner dict has: pos_rmse, vel_rmse, anees, convergence_time, track_loss
        filter_names: list of filter names
        traj_types: list of trajectory types
    """
    n_traj = len(traj_types)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("pos_rmse", "Position RMSE (m)", axes[0, 0]),
        ("vel_rmse", "Velocity RMSE (m/s)", axes[0, 1]),
        ("anees", "ANEES", axes[1, 0]),
        ("convergence_time", "Convergence Time (steps)", axes[1, 1]),
    ]

    for metric_key, ylabel, ax in metrics:
        positions = []
        data_all = []
        labels = []
        colors_all = []

        for ti, traj in enumerate(traj_types):
            for fi, fname in enumerate(filter_names):
                key = (fname, traj)
                if key not in results:
                    continue
                vals = results[key].get(metric_key, [])
                if len(vals) == 0:
                    continue
                pos = ti * (len(filter_names) + 1) + fi
                positions.append(pos)
                data_all.append(vals)
                labels.append(f"{fname}\n{traj}")
                colors_all.append(_get_color(fname))

        if not data_all:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        bp = ax.boxplot(data_all, positions=positions, widths=0.7, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7, rotation=30)
        ax.grid(True, alpha=0.3, axis="y")

        # ANEES reference line
        if metric_key == "anees":
            ax.axhline(6, color="gray", linestyle=":", linewidth=1, label="Expected (6)")
            ax.legend(fontsize=8)

    fig.suptitle("Monte Carlo Filter Comparison", fontsize=14)
    _save_or_show(fig, save_path)


def plot_track_loss_bar(
    results: dict,
    filter_names: list[str],
    traj_types: list[str],
    threshold: float = 100.0,
    save_path: str | None = None,
):
    """Stacked bar chart of track loss rates: error-based vs sensor blackout."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(traj_types))
    width = 0.8 / max(len(filter_names), 1)

    has_breakdown = any(
        "error_loss" in results.get((fn, tj), {})
        for fn in filter_names for tj in traj_types
    )

    for fi, fname in enumerate(filter_names):
        err_rates = []
        blind_rates = []
        total_rates = []
        for traj in traj_types:
            key = (fname, traj)
            r = results.get(key, {})
            total_rates.append(np.mean(r.get("track_loss", [0])) * 100)
            err_rates.append(np.mean(r.get("error_loss", [0])) * 100)
            blind_rates.append(np.mean(r.get("blackout_loss", [0])) * 100)

        pos = x + fi * width
        if has_breakdown:
            ax.bar(pos, err_rates, width, label=f"{fname} (error >100m)",
                   color=_get_color(fname), alpha=0.7)
            ax.bar(pos, blind_rates, width, bottom=err_rates,
                   label=f"{fname} (blackout)",
                   color=_get_color(fname), alpha=0.35, hatch="//")
        else:
            ax.bar(pos, total_rates, width, label=fname,
                   color=_get_color(fname), alpha=0.7)

    # Annotate if everything is zero
    all_zero = all(
        np.mean(results.get((fn, tj), {}).get("track_loss", [0])) == 0
        for fn in filter_names for tj in traj_types
    )
    if all_zero:
        ax.text(0.5, 0.5, "No track loss observed", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#888888")

    ax.set_ylabel("Track Loss Rate (%)")
    ax.set_title(f"Track Loss Rate (error >{threshold:.0f}m  or  sensor blackout)")
    ax.set_xticks(x + width * (len(filter_names) - 1) / 2)
    ax.set_xticklabels(traj_types)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    _save_or_show(fig, save_path)
