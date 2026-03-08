"""Generate Layer 1 presentation figures: EKF/UKF/PF slide + IMM slide."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

OUT = Path("presentation/figures")
OUT.mkdir(exist_ok=True)

COLORS = {
    "EKF": "#2563eb",
    "UKF": "#d97706",
    "PF": "#059669",
    "IMM": "#dc2626",
}


def style(ax, title=None, xlabel=None, ylabel=None):
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


def smooth(arr, w=50):
    if len(arr) < w:
        return arr, np.arange(len(arr))
    s = np.convolve(arr, np.ones(w) / w, mode="valid")
    return s, np.arange(w - 1, len(arr))


# ── Slide 1: EKF vs UKF vs PF (3 panels, NO IMM) ────────────────────

def fig_ekf_ukf_pf():
    path = "results/filter_data_evasive_s42.npz"
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    d = np.load(path, allow_pickle=True)
    true_states = d["true_states"]
    estimates = d["estimates"]
    covariances = d["covariances"]
    filter_names = list(d["filter_names"])
    dt = float(d["dt"])
    T = estimates.shape[1]
    time = np.arange(T) * dt

    # Only plot EKF, UKF, PF (not IMM)
    plot_filters = [(i, n) for i, n in enumerate(filter_names) if n != "IMM"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Position RMSE
    ax = axes[0]
    for i, name in plot_filters:
        pos_err = np.linalg.norm(estimates[i, :, :3] - true_states[:T, :3], axis=1)
        s, idx = smooth(pos_err, 30)
        ax.plot(time[idx], s, color=COLORS.get(name, "#666"), linewidth=2.5,
                label=name, alpha=0.9)
    style(ax, title="Position Error", xlabel="Time (s)", ylabel="Error (m)")
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    # Panel 2: tr(P) convergence
    ax = axes[1]
    for i, name in plot_filters:
        trP = np.array([np.trace(covariances[i, t, :3, :3]) for t in range(T)])
        s, idx = smooth(trP, 30)
        ax.plot(time[idx], s, color=COLORS.get(name, "#666"), linewidth=2.5,
                label=name, alpha=0.9)
    style(ax, title="Covariance Trace — tr(P)", xlabel="Time (s)", ylabel="tr(P)")
    ax.set_yscale("log")
    ax.legend(fontsize=11)

    # Panel 3: NEES
    ax = axes[2]
    from scipy import stats
    chi2_lo = stats.chi2.ppf(0.025, 6)
    chi2_hi = stats.chi2.ppf(0.975, 6)
    ax.axhspan(chi2_lo, chi2_hi, alpha=0.15, color="#94a3b8", label="95% bounds")
    ax.axhline(6, color="#94a3b8", linewidth=1, alpha=0.5, linestyle="--")

    nees = d["nees"]
    for i, name in plot_filters:
        s, idx = smooth(nees[i], 30)
        ax.plot(time[idx], s, color=COLORS.get(name, "#666"), linewidth=2.5,
                label=name, alpha=0.9)
    style(ax, title="NEES (6-DOF, 95% bounds)", xlabel="Time (s)", ylabel="NEES")
    ax.legend(fontsize=10)
    ax.set_ylim(0, min(40, np.nanpercentile(nees[0], 98)))

    plt.tight_layout()
    p = OUT / "layer1_ekf_ukf_pf.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {p}")
    plt.close(fig)


# ── Slide 2: IMM — mode probs + all 4 filters comparison ────────────

def fig_imm():
    path = "results/filter_data_evasive_s42.npz"
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return

    d = np.load(path, allow_pickle=True)
    true_states = d["true_states"]
    estimates = d["estimates"]
    filter_names = list(d["filter_names"])
    dt = float(d["dt"])
    T = estimates.shape[1]
    time = np.arange(T) * dt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Panel 1: 3-mode probabilities
    ax = axes[0]
    if "imm_mode_probs" in d:
        probs = d["imm_mode_probs"]
        tp = np.arange(len(probs)) * dt
        # Stack fill: gentle, moderate, aggressive
        ax.fill_between(tp, 0, probs[:, 0], alpha=0.4, color="#2563eb",
                         label="Gentle (σ_a=0.3)")
        ax.fill_between(tp, probs[:, 0], probs[:, 0] + probs[:, 1],
                         alpha=0.4, color="#d97706", label="Moderate (σ_a=5)")
        ax.fill_between(tp, probs[:, 0] + probs[:, 1], 1,
                         alpha=0.4, color="#dc2626", label="Aggressive (σ_a=25)")
    style(ax, title="IMM Mode Probabilities (3-mode)", xlabel="Time (s)",
          ylabel="P(mode)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="center right")

    # Panel 2: All 4 filters position error
    ax = axes[1]
    for i, name in enumerate(filter_names):
        pos_err = np.linalg.norm(estimates[i, :, :3] - true_states[:T, :3], axis=1)
        s, idx = smooth(pos_err, 30)
        color = COLORS.get(name, "#666")
        lw = 3.0 if name in ("PF", "IMM") else 2.0
        ax.plot(time[idx], s, color=color, linewidth=lw, label=name, alpha=0.9)
    style(ax, title="Position Error — All Filters", xlabel="Time (s)",
          ylabel="Error (m)")
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    p = OUT / "layer1_imm.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating Layer 1 presentation figures...\n")
    print("[EKF / UKF / PF]")
    fig_ekf_ukf_pf()
    print("\n[IMM]")
    fig_imm()
    print("\nDone!")
