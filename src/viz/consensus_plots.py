"""Visualization for distributed consensus EKF experiments.

Includes per-drone estimate plots, topology comparison, dropout degradation,
consensus iteration sweeps, and topology diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt

# Drone colors for per-drone plots
DRONE_COLORS = ["#E91E63", "#9C27B0", "#3F51B5", "#009688", "#FF5722"]
TOPO_COLORS = {"full": "#2196F3", "ring": "#FF9800", "star": "#4CAF50"}
TOPO_MARKERS = {"full": "o", "ring": "s", "star": "^"}


def _save_or_show(fig, save_path):
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_per_drone_estimates(
    time: np.ndarray,
    true_states: np.ndarray,
    local_estimates: np.ndarray,
    consensus_avg: np.ndarray,
    centralized_est: np.ndarray,
    consensus_label: str = "Consensus",
    save_path: str | None = None,
):
    """Plot per-drone local estimates vs consensus average vs centralized.

    Args:
        time: (T,) time axis
        true_states: (T, 6) ground truth
        local_estimates: (N, T, 6) per-drone estimates
        consensus_avg: (T, 6) consensus average estimate
        centralized_est: (T, 6) centralized EKF estimate
        consensus_label: label for consensus filter
        save_path: if provided, save figure
    """
    N, T_local = local_estimates.shape[:2]
    T = min(len(time), T_local, len(true_states))

    labels = ["px (m)", "py (m)", "pz (m)"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax_i, (ax, label) in enumerate(zip(axes, labels)):
        # Truth
        ax.plot(time[:T], true_states[:T, ax_i], "k-", lw=2, label="Truth", zorder=5)

        # Per-drone locals (thin)
        for di in range(N):
            ax.plot(time[:T], local_estimates[di, :T, ax_i],
                    color=DRONE_COLORS[di % len(DRONE_COLORS)],
                    alpha=0.4, lw=0.8,
                    label=f"Drone {di}" if ax_i == 0 else None)

        # Consensus average
        ax.plot(time[:T], consensus_avg[:T, ax_i],
                "b--", lw=1.5, label=consensus_label if ax_i == 0 else None, zorder=4)

        # Centralized
        ax.plot(time[:T], centralized_est[:T, ax_i],
                "r-", lw=1.5, alpha=0.7, label="Centralized" if ax_i == 0 else None, zorder=3)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Per-Drone Local Estimates vs Consensus vs Centralized", fontsize=13)
    _save_or_show(fig, save_path)


def plot_topology_comparison(
    time: np.ndarray,
    true_states: np.ndarray,
    estimates: np.ndarray,
    filter_names: list[str],
    save_path: str | None = None,
):
    """Position RMSE over time for each filter/topology.

    Args:
        time: (T,) time axis
        true_states: (T, 6) ground truth
        estimates: (n_filters, T, 6) filter estimates
        filter_names: list of filter labels
        save_path: if provided, save figure
    """
    T = min(len(time), estimates.shape[1], len(true_states))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    colors = ["#F44336"]  # centralized = red
    for name in filter_names[1:]:
        name_lower = name.lower()
        if "full" in name_lower:
            colors.append(TOPO_COLORS["full"])
        elif "ring" in name_lower:
            colors.append(TOPO_COLORS["ring"])
        elif "star" in name_lower:
            colors.append(TOPO_COLORS["star"])
        else:
            colors.append("#999999")

    # Position RMSE
    for fi, name in enumerate(filter_names):
        pos_err = np.linalg.norm(true_states[:T, :3] - estimates[fi, :T, :3], axis=1)
        axes[0].plot(time[:T], pos_err, color=colors[fi], lw=1.5, label=name)
    axes[0].set_ylabel("Position Error (m)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Velocity RMSE
    for fi, name in enumerate(filter_names):
        vel_err = np.linalg.norm(true_states[:T, 3:6] - estimates[fi, :T, 3:6], axis=1)
        axes[1].plot(time[:T], vel_err, color=colors[fi], lw=1.5, label=name)
    axes[1].set_ylabel("Velocity Error (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Topology Comparison: Error Over Time", fontsize=13)
    _save_or_show(fig, save_path)


def plot_consensus_convergence(
    time: np.ndarray,
    disagreements: np.ndarray,
    save_path: str | None = None,
):
    """Consensus disagreement (RMS spread of local estimates) over time.

    Args:
        time: (T,) time axis
        disagreements: (T,) disagreement values
        save_path: if provided, save figure
    """
    T = min(len(time), len(disagreements))
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(time[:T], disagreements[:T], color="#2196F3", lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Disagreement (m)")
    ax.set_title("Consensus Convergence: Local Estimate Disagreement")
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_dropout_degradation(
    dropout_probs: list[float],
    metrics: dict[str, dict],
    metric_key: str = "pos_rmse",
    ylabel: str = "Position RMSE (m)",
    save_path: str | None = None,
):
    """Dropout probability vs metric, one curve per topology.

    Args:
        dropout_probs: list of dropout probabilities
        metrics: {topology: {dropout: {metric_key: value, ...}, ...}, ...}
        metric_key: which metric to plot
        ylabel: y-axis label
        save_path: if provided, save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for topo in metrics:
        color = TOPO_COLORS.get(topo, "#999999")
        marker = TOPO_MARKERS.get(topo, "o")
        vals = []
        probs = []
        for dp in dropout_probs:
            if dp in metrics[topo]:
                vals.append(metrics[topo][dp].get(metric_key, np.nan))
                probs.append(dp)
        ax.plot(probs, vals, color=color, marker=marker, lw=2, markersize=8,
                label=topo.capitalize())

    ax.set_xlabel("Dropout Probability")
    ax.set_ylabel(ylabel)
    ax.set_title("Communication Dropout Degradation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_iteration_sweep(
    iterations: list[int],
    metrics: dict[str, list[float]],
    ylabel: str = "Position RMSE (m)",
    save_path: str | None = None,
):
    """Consensus iterations L vs metric, one curve per topology.

    Args:
        iterations: list of L values
        metrics: {topology: [metric_at_L1, metric_at_L2, ...]}
        ylabel: y-axis label
        save_path: if provided, save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for topo, vals in metrics.items():
        color = TOPO_COLORS.get(topo, "#999999")
        marker = TOPO_MARKERS.get(topo, "o")
        ax.plot(iterations[:len(vals)], vals, color=color, marker=marker,
                lw=2, markersize=8, label=topo.capitalize())

    ax.set_xlabel("Consensus Iterations (L)")
    ax.set_ylabel(ylabel)
    ax.set_title("Effect of Consensus Iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def draw_topology_diagrams(n: int = 5, save_path: str | None = None):
    """Draw 3-panel figure showing full, ring, and star topologies.

    Args:
        n: number of drones
        save_path: if provided, save figure
    """
    from src.filters.topology import generate_adjacency

    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed — skipping topology diagrams")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    topos = ["full", "ring", "star"]
    titles = ["Full (Complete)", "Ring", "Star (Hub-Spoke)"]

    for ax, topo, title in zip(axes, topos, titles):
        adj = generate_adjacency(n, topo)
        G = nx.from_numpy_array(adj)
        pos = nx.circular_layout(G)

        node_colors = ["#F44336" if (topo == "star" and node == 0) else "#2196F3"
                       for node in G.nodes()]
        nx.draw(
            G, pos, ax=ax,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_size=12,
            font_color="white",
            font_weight="bold",
            edge_color="#666666",
            width=2.0,
        )
        ax.set_title(title, fontsize=12)

    fig.suptitle("Communication Topologies (5 Drones)", fontsize=14, y=1.02)
    _save_or_show(fig, save_path)


def plot_mc_dropout_comparison(
    results: dict,
    topologies: list[str],
    dropout_probs: list[float],
    metric_key: str = "pos_rmse",
    centralized_baseline: float | None = None,
    ylabel: str = "Position RMSE (m)",
    save_path: str | None = None,
):
    """MC results: box plots or curves of metric vs dropout per topology.

    Args:
        results: {(topology, dropout): {metric_key: array, ...}, ...}
        topologies: list of topologies
        dropout_probs: list of dropout probs
        metric_key: which metric to plot
        centralized_baseline: if provided, draw horizontal line
        ylabel: y-axis label
        save_path: if provided, save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for topo in topologies:
        color = TOPO_COLORS.get(topo, "#999999")
        marker = TOPO_MARKERS.get(topo, "o")
        medians = []
        q25 = []
        q75 = []
        valid_probs = []

        for dp in dropout_probs:
            key = (topo, dp)
            if key in results and metric_key in results[key]:
                vals = results[key][metric_key]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    medians.append(np.median(vals))
                    q25.append(np.percentile(vals, 25))
                    q75.append(np.percentile(vals, 75))
                    valid_probs.append(dp)

        if valid_probs:
            medians = np.array(medians)
            q25 = np.array(q25)
            q75 = np.array(q75)
            ax.plot(valid_probs, medians, color=color, marker=marker,
                    lw=2, markersize=8, label=topo.capitalize())
            ax.fill_between(valid_probs, q25, q75, color=color, alpha=0.15)

    if centralized_baseline is not None:
        ax.axhline(centralized_baseline, color="#F44336", ls="--", lw=1.5,
                   label="Centralized EKF")

    ax.set_xlabel("Dropout Probability")
    ax.set_ylabel(ylabel)
    ax.set_title("Dropout Degradation (Monte Carlo)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_mc_topology_boxes(
    results: dict,
    topologies: list[str],
    traj_types: list[str],
    metric_key: str = "pos_rmse",
    centralized_results: dict | None = None,
    ylabel: str = "Position RMSE (m)",
    save_path: str | None = None,
):
    """Box plots comparing topologies across trajectory types.

    Args:
        results: {(topology, traj): {metric_key: array, ...}, ...}
        topologies: list of topology names
        traj_types: list of trajectory types
        metric_key: metric to plot
        centralized_results: {traj: {metric_key: array}} for baseline
        ylabel: y-axis label
        save_path: if provided, save figure
    """
    n_traj = len(traj_types)
    n_topo = len(topologies)
    fig, axes = plt.subplots(1, n_traj, figsize=(4 * n_traj, 5), sharey=True)
    if n_traj == 1:
        axes = [axes]

    width = 0.6 / (n_topo + 1)
    for ti, traj in enumerate(traj_types):
        ax = axes[ti]
        positions = []
        data = []
        colors = []
        labels = []

        # Centralized baseline
        if centralized_results and traj in centralized_results:
            vals = centralized_results[traj].get(metric_key, np.array([]))
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                positions.append(0)
                data.append(vals)
                colors.append("#F44336")
                labels.append("Central")

        for topi, topo in enumerate(topologies):
            key = (topo, traj)
            if key in results and metric_key in results[key]:
                vals = results[key][metric_key]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    positions.append(topi + 1)
                    data.append(vals)
                    colors.append(TOPO_COLORS.get(topo, "#999999"))
                    labels.append(topo.capitalize())

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.5,
                           patch_artist=True, showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_title(traj, fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel(ylabel)
    fig.suptitle("Topology Comparison (Monte Carlo)", fontsize=13)
    _save_or_show(fig, save_path)
