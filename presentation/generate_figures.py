"""Generate presentation-quality figures for AA273 slides.

Usage:
    python presentation/generate_figures.py              # all figures
    python presentation/generate_figures.py --only layer1 # just layer 1
    python presentation/generate_figures.py --only layer2 # just layer 2
    python presentation/generate_figures.py --run-mc      # run fresh MC (slow)
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Presentation color scheme
COLORS = {
    "EKF": "#2563eb",
    "UKF": "#d97706",
    "PF": "#059669",
    "IMM": "#dc2626",
    "Centralized EKF": "#2563eb",
    "Centralized IMM": "#dc2626",
    "Consensus EKF": "#7c3aed",
    "Consensus IMM": "#0891b2",
}

STYLES = {
    "EKF": "-",
    "UKF": "--",
    "PF": "-.",
    "IMM": ":",
    "Centralized EKF": "-",
    "Centralized IMM": "-",
    "Consensus EKF": "--",
    "Consensus IMM": "--",
}


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    """Apply clean presentation styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)
    if title:
        ax.set_title(title, fontsize=14, fontweight="600", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, linewidth=0.5)


# ── Layer 1: Single-run filter comparison from saved data ──────────────

def fig_layer1_rmse():
    """RMSE convergence curves for EKF vs IMM (+ consensus variants)."""
    data_path = "results/imm_data_evasive_s42.npz"
    if not os.path.exists(data_path):
        print(f"  SKIP: {data_path} not found — run scripts/run_consensus_imm.py first")
        return

    d = np.load(data_path, allow_pickle=True)
    true_states = d["true_states"]
    estimates = d["estimates"]
    filter_names = d["filter_names"]
    dt = float(d["dt"])
    T = len(true_states)
    time = np.arange(T) * dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Position RMSE
    for i, name in enumerate(filter_names):
        pos_err = np.linalg.norm(estimates[i, :, :3] - true_states[:, :3], axis=1)
        # Smooth with rolling window
        window = 100
        if len(pos_err) > window:
            smoothed = np.convolve(pos_err, np.ones(window)/window, mode="valid")
            t_smooth = time[window-1:]
        else:
            smoothed = pos_err
            t_smooth = time
        color = COLORS.get(name, "#666")
        style = STYLES.get(name, "-")
        label = name.replace("(full)", "").strip()
        ax1.plot(t_smooth, smoothed, color=color, linestyle=style,
                 linewidth=2, label=label, alpha=0.9)

    style_axis(ax1, title="Position Error", xlabel="Time (s)", ylabel="Error (m)")
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.set_ylim(bottom=0)

    # tr(P) convergence
    covariances = d["covariances"]
    for i, name in enumerate(filter_names):
        trP = np.array([np.trace(covariances[i, t, :3, :3]) for t in range(T)])
        window = 100
        if len(trP) > window:
            smoothed = np.convolve(trP, np.ones(window)/window, mode="valid")
            t_smooth = time[window-1:]
        else:
            smoothed = trP
            t_smooth = time
        color = COLORS.get(name, "#666")
        style = STYLES.get(name, "-")
        label = name.replace("(full)", "").strip()
        ax2.plot(t_smooth, smoothed, color=color, linestyle=style,
                 linewidth=2, label=label, alpha=0.9)

    style_axis(ax2, title="Covariance Trace — tr(P_pos)", xlabel="Time (s)", ylabel="tr(P)")
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.set_ylim(bottom=0)
    ax2.set_yscale("log")

    plt.tight_layout()
    path = OUT_DIR / "layer1_rmse_trP.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


def fig_layer1_nees():
    """NEES consistency plot with chi-squared bounds."""
    data_path = "results/imm_data_evasive_s42.npz"
    if not os.path.exists(data_path):
        print(f"  SKIP: {data_path} not found")
        return

    d = np.load(data_path, allow_pickle=True)
    nees = d["nees"]
    filter_names = d["filter_names"]
    dt = float(d["dt"])
    T = nees.shape[1]
    time = np.arange(T) * dt

    fig, ax = plt.subplots(figsize=(12, 5))

    # Chi-squared bounds (6 DOF, 95% interval)
    from scipy import stats
    n_dof = 6
    alpha = 0.05
    chi2_lower = stats.chi2.ppf(alpha / 2, n_dof)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, n_dof)
    ax.axhline(chi2_lower, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(chi2_upper, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(n_dof, color="#94a3b8", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.fill_between(time, chi2_lower, chi2_upper, alpha=0.08, color="#94a3b8")

    for i, name in enumerate(filter_names):
        window = 200
        nees_i = nees[i]
        if len(nees_i) > window:
            smoothed = np.convolve(nees_i, np.ones(window)/window, mode="valid")
            t_smooth = time[window-1:]
        else:
            smoothed = nees_i
            t_smooth = time
        color = COLORS.get(name, "#666")
        style = STYLES.get(name, "-")
        label = name.replace("(full)", "").strip()
        ax.plot(t_smooth, smoothed, color=color, linestyle=style,
                linewidth=2, label=label, alpha=0.9)

    style_axis(ax, title="NEES Consistency (6-DOF, 95% bounds)",
               xlabel="Time (s)", ylabel="NEES")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0, min(50, np.percentile(nees, 99)))

    plt.tight_layout()
    path = OUT_DIR / "layer1_nees.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


def fig_imm_mode_probs():
    """IMM mode probability evolution."""
    data_path = "results/imm_data_evasive_s42.npz"
    if not os.path.exists(data_path):
        print(f"  SKIP: {data_path} not found")
        return

    d = np.load(data_path, allow_pickle=True)
    dt = float(d["dt"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax_idx, key in enumerate(["mode_probs_Centralized IMM", "mode_probs_Consensus IMM (full)"]):
        if key not in d:
            continue
        probs = d[key]  # (T, 2)
        T = len(probs)
        time = np.arange(T) * dt
        ax = axes[ax_idx]

        ax.fill_between(time, 0, probs[:, 0], alpha=0.4, color="#2563eb", label="CV (gentle)")
        ax.fill_between(time, probs[:, 0], 1, alpha=0.4, color="#dc2626", label="CA (aggressive)")
        ax.plot(time, probs[:, 0], color="#2563eb", linewidth=1.5)

        title = "Centralized IMM" if "Centralized" in key else "Consensus IMM"
        style_axis(ax, title=f"{title} — Mode Probabilities",
                   xlabel="Time (s)", ylabel="P(mode)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10, loc="center right")

    plt.tight_layout()
    path = OUT_DIR / "imm_mode_probs.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Layer 2: Consensus-specific plots ──────────────────────────────────

def fig_layer2_per_drone():
    """Per-drone local estimates vs consensus average."""
    data_path = "results/imm_data_evasive_s42.npz"
    if not os.path.exists(data_path):
        print(f"  SKIP: {data_path} not found")
        return

    d = np.load(data_path, allow_pickle=True)
    true_states = d["true_states"]
    local_estimates = d["local_estimates"]  # (5, T, 6)
    dt = float(d["dt"])
    N, T = local_estimates.shape[:2]
    T = min(T, len(true_states))
    time = np.arange(T) * dt

    drone_colors = ["#dc2626", "#7c3aed", "#2563eb", "#059669", "#d97706"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["$p_x$ (m)", "$p_y$ (m)", "$p_z$ (m)"]

    for dim in range(3):
        ax = axes[dim]
        ax.plot(time[:T], true_states[:T, dim], "k-", linewidth=2,
                label="Truth", alpha=0.8)

        for i in range(N):
            ax.plot(time[:T], local_estimates[i, :T, dim],
                    color=drone_colors[i], linewidth=0.8, alpha=0.5,
                    label=f"Drone {i+1}" if dim == 0 else None)

        # Consensus average
        avg = np.mean(local_estimates[:, :T, dim], axis=0)
        ax.plot(time[:T], avg, color="#0891b2", linewidth=2,
                linestyle="--", label="Consensus avg" if dim == 0 else None)

        style_axis(ax, ylabel=labels[dim])
        if dim == 0:
            ax.legend(fontsize=9, ncol=4, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_title("Per-Drone Estimates vs Truth (Consensus IMM)", fontsize=14, fontweight="600")

    plt.tight_layout()
    path = OUT_DIR / "layer2_per_drone.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


def fig_layer2_disagreement():
    """Consensus disagreement (RMS spread) over time."""
    data_path = "results/imm_data_evasive_s42.npz"
    if not os.path.exists(data_path):
        print(f"  SKIP: {data_path} not found")
        return

    d = np.load(data_path, allow_pickle=True)
    disagreement = d["disagreements"]
    dt = float(d["dt"])
    T = len(disagreement)
    time = np.arange(T) * dt

    fig, ax = plt.subplots(figsize=(10, 4))

    window = 100
    if T > window:
        smoothed = np.convolve(disagreement, np.ones(window)/window, mode="valid")
        t_smooth = time[window-1:]
    else:
        smoothed = disagreement
        t_smooth = time

    ax.fill_between(t_smooth, 0, smoothed, alpha=0.3, color="#7c3aed")
    ax.plot(t_smooth, smoothed, color="#7c3aed", linewidth=2)

    style_axis(ax, title="Inter-Drone Disagreement (RMS Spread of Local Estimates)",
               xlabel="Time (s)", ylabel="RMS disagreement (m)")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = OUT_DIR / "layer2_disagreement.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


def fig_topology_diagram():
    """Communication topology diagrams (ring, star, full)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    topologies = {
        "Ring": [(0,1),(1,2),(2,3),(3,4),(4,0)],
        "Star": [(0,1),(0,2),(0,3),(0,4)],
        "Full": [(i,j) for i in range(5) for j in range(i+1, 5)],
    }
    topo_colors = {"Ring": "#d97706", "Star": "#059669", "Full": "#2563eb"}

    # Place 5 nodes in a pentagon
    angles = np.linspace(0, 2*np.pi, 6)[:-1] - np.pi/2
    positions = np.column_stack([np.cos(angles), np.sin(angles)])

    for ax_idx, (name, edges) in enumerate(topologies.items()):
        ax = axes[ax_idx]
        color = topo_colors[name]

        # Draw edges
        for i, j in edges:
            ax.plot([positions[i,0], positions[j,0]],
                    [positions[i,1], positions[j,1]],
                    color=color, linewidth=2, alpha=0.6)

        # Draw nodes
        for i in range(5):
            circle = plt.Circle(positions[i], 0.12, color=color,
                              ec="white", linewidth=2, zorder=5)
            ax.add_patch(circle)
            ax.text(positions[i,0], positions[i,1], str(i+1),
                   ha="center", va="center", fontsize=10,
                   fontweight="bold", color="white", zorder=6)

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=14, fontweight="600", color=color)
        ax.axis("off")

    plt.tight_layout()
    path = OUT_DIR / "topology_diagrams.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Summary / overview figures ─────────────────────────────────────────

def fig_geometry_matters():
    """Visual showing why angular geometry matters for bearing-only tracking."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Good geometry: diverse angles
    ax = axes[0]
    target = np.array([0, 0])
    good_angles = np.array([0, 72, 144, 216, 288]) * np.pi / 180
    drone_r = 60
    for i, angle in enumerate(good_angles):
        pos = target + drone_r * np.array([np.cos(angle), np.sin(angle)])
        ax.plot(*pos, "o", color="#2563eb", markersize=12, zorder=5)
        # Bearing line
        ax.plot([pos[0], target[0]], [pos[1], target[1]],
                color="#2563eb", linewidth=1.5, alpha=0.4)
        ax.annotate("", xy=target, xytext=pos,
                    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.5, alpha=0.6))

    ax.plot(*target, "r*", markersize=18, zorder=6, label="Target")
    # Uncertainty ellipse (small)
    ellipse = mpatches.Ellipse(target, 8, 8, angle=0, fill=False,
                                edgecolor="#dc2626", linewidth=2, linestyle="--")
    ax.add_patch(ellipse)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    style_axis(ax, title="Good Geometry → Low Uncertainty")
    ax.text(0, -80, "tr(P) ≈ 9.6", ha="center", fontsize=13,
            fontweight="bold", color="#059669")

    # Bad geometry: clustered
    ax = axes[1]
    cluster_angles = np.array([30, 35, 40, 45, 50]) * np.pi / 180
    for i, angle in enumerate(cluster_angles):
        pos = target + drone_r * np.array([np.cos(angle), np.sin(angle)])
        ax.plot(*pos, "o", color="#2563eb", markersize=12, zorder=5)
        ax.plot([pos[0], target[0]], [pos[1], target[1]],
                color="#2563eb", linewidth=1.5, alpha=0.4)
        ax.annotate("", xy=target, xytext=pos,
                    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.5, alpha=0.6))

    ax.plot(*target, "r*", markersize=18, zorder=6)
    # Uncertainty ellipse (large, elongated)
    ellipse = mpatches.Ellipse(target, 50, 12, angle=40, fill=False,
                                edgecolor="#dc2626", linewidth=2, linestyle="--")
    ax.add_patch(ellipse)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    style_axis(ax, title="Bad Geometry → High Uncertainty")
    ax.text(0, -80, "tr(P) ≈ 6,382", ha="center", fontsize=13,
            fontweight="bold", color="#dc2626")

    plt.tight_layout()
    path = OUT_DIR / "geometry_matters.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


def fig_sensor_model():
    """Sensor noise and detection probability vs range."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Noise vs range
    r = np.linspace(10, 400, 200)
    sigma_base = np.deg2rad(2.0)
    r_ref = 100.0
    sigma_r = sigma_base * (r / r_ref)

    ax1.plot(r, np.rad2deg(sigma_r), color="#2563eb", linewidth=2.5)
    ax1.axvline(r_ref, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)
    ax1.text(r_ref + 5, np.rad2deg(sigma_base) * 3.5, f"$r_{{ref}}$={r_ref}m",
             fontsize=11, color="#64748b")
    style_axis(ax1, title="Bearing Noise vs Range",
               xlabel="Range (m)", ylabel="σ (degrees)")

    # Detection probability vs range
    p_max = 0.99
    r_half = 250.0
    k = np.log(p_max / 0.5) / r_half
    p_detect = p_max * np.exp(-k * r)

    ax2.plot(r, p_detect, color="#059669", linewidth=2.5)
    ax2.axhline(0.5, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axvline(r_half, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.5)
    ax2.text(r_half + 5, 0.55, f"$r_{{1/2}}$={r_half}m", fontsize=11, color="#64748b")
    style_axis(ax2, title="Detection Probability vs Range",
               xlabel="Range (m)", ylabel="P(detect)")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    path = OUT_DIR / "sensor_model.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


def fig_3d_trajectory():
    """3D trajectory plot from saved data."""
    data_path = "results/imm_data_evasive_s42.npz"
    if not os.path.exists(data_path):
        print(f"  SKIP: {data_path} not found")
        return

    d = np.load(data_path, allow_pickle=True)
    true_states = d["true_states"]  # (T, 6)
    drone_positions = d["drone_positions"]  # (T, 5, 3)
    T = len(true_states)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Target trajectory
    ax.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2],
            "r-", linewidth=2, label="Target", alpha=0.8)
    ax.scatter(*true_states[0, :3], c="red", s=80, marker="*", zorder=5)

    # Drone trajectories
    drone_colors = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626"]
    for i in range(5):
        ax.plot(drone_positions[:, i, 0], drone_positions[:, i, 1],
                drone_positions[:, i, 2], color=drone_colors[i],
                linewidth=1, alpha=0.5, label=f"Drone {i+1}")

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_zlabel("Z (m)", fontsize=11)
    ax.set_title("3D Tracking Scenario — Evasive Target", fontsize=14, fontweight="600")
    ax.legend(fontsize=9, loc="upper left")

    path = OUT_DIR / "3d_trajectory.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Monte Carlo figures (require running MC) ───────────────────────────

def fig_mc_layer1(run_mc=False):
    """MC comparison boxplots for Layer 1 filters."""
    if run_mc:
        print("  Running Layer 1 Monte Carlo (this takes a few minutes)...")
        os.system("cd .. && python scripts/monte_carlo.py --runs 50 --traj evasive --save")
    else:
        print("  SKIP MC Layer 1 (use --run-mc to generate). Using placeholder.")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        choices=["layer1", "layer2", "overview"])
    parser.add_argument("--run-mc", action="store_true")
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    print("Generating presentation figures...")
    print(f"Output: {OUT_DIR.resolve()}\n")

    if args.only is None or args.only == "overview":
        print("[Overview]")
        fig_geometry_matters()
        fig_sensor_model()
        fig_3d_trajectory()
        fig_topology_diagram()
        print()

    if args.only is None or args.only == "layer1":
        print("[Layer 1]")
        fig_layer1_rmse()
        fig_layer1_nees()
        fig_imm_mode_probs()
        if args.run_mc:
            fig_mc_layer1(run_mc=True)
        print()

    if args.only is None or args.only == "layer2":
        print("[Layer 2]")
        fig_layer2_per_drone()
        fig_layer2_disagreement()
        print()

    print("Done! Figures saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
