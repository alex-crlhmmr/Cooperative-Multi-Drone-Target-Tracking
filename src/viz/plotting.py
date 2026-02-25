"""Matplotlib plotting utilities for tracking results."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_trajectories(
    drone_trajectories: list[np.ndarray],
    target_trajectory: np.ndarray,
    measurements: list[list] | None = None,
    title: str = "Multi-Drone Target Tracking",
    save_path: str | None = None,
    plot_box: float | None = None,
):
    """Plot 3D trajectories of drones and target.

    Args:
        drone_trajectories: list of (T, 3) arrays, one per tracker drone
        target_trajectory: (T, 3) array of target positions
        measurements: optional list of measurement lists per timestep
        title: plot title
        save_path: if provided, save figure to this path
        plot_box: half-size of bounding box in meters. If None, auto-fit to data.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#00B4D8', '#FF6B35', '#2EC4B6', '#9B5DE5', '#F4A261', '#06D6A0', '#EF476F', '#118AB2']
    drone_labels = [f'Tracker {i}' for i in range(len(drone_trajectories))]

    # Plot drone trajectories
    for i, traj in enumerate(drone_trajectories):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=colors[i % len(colors)], linewidth=1.5,
                label=drone_labels[i], alpha=0.8)
        # Start/end markers
        ax.scatter(*traj[0], color=colors[i % len(colors)], marker='o', s=60)
        ax.scatter(*traj[-1], color=colors[i % len(colors)], marker='s', s=60)

    # Plot target trajectory
    ax.plot(target_trajectory[:, 0], target_trajectory[:, 1], target_trajectory[:, 2],
            color='red', linewidth=2.5, label='Target', linestyle='--')
    ax.scatter(*target_trajectory[0], color='red', marker='*', s=150, zorder=5)
    ax.scatter(*target_trajectory[-1], color='darkred', marker='X', s=100, zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend(loc='upper left')

    # Bounding box: use plot_box if given, otherwise auto-fit
    all_pts = np.vstack([target_trajectory] + drone_trajectories)
    mid = all_pts.mean(axis=0)
    if plot_box is not None:
        half = plot_box
    else:
        half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(max(0, mid[2] - half), mid[2] + half)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_2d_topdown(
    drone_trajectories: list[np.ndarray],
    target_trajectory: np.ndarray,
    title: str = "Top-Down View",
    save_path: str | None = None,
    plot_box: float | None = None,
):
    """Plot 2D top-down (XY) view of trajectories."""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['#00B4D8', '#FF6B35', '#2EC4B6', '#9B5DE5', '#F4A261', '#06D6A0', '#EF476F', '#118AB2']

    for i, traj in enumerate(drone_trajectories):
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)],
                linewidth=1.5, label=f'Tracker {i}', alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i % len(colors)],
                   marker='o', s=80, zorder=5)

    ax.plot(target_trajectory[:, 0], target_trajectory[:, 1],
            color='red', linewidth=2.5, label='Target', linestyle='--')
    ax.scatter(target_trajectory[0, 0], target_trajectory[0, 1],
               color='red', marker='*', s=200, zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if plot_box is not None:
        all_pts = np.vstack([target_trajectory[:, :2]] + [t[:, :2] for t in drone_trajectories])
        mid = all_pts.mean(axis=0)
        ax.set_xlim(mid[0] - plot_box, mid[0] + plot_box)
        ax.set_ylim(mid[1] - plot_box, mid[1] + plot_box)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_distances(
    drone_trajectories: list[np.ndarray],
    target_trajectory: np.ndarray,
    dt: float = 1.0,
    save_path: str | None = None,
):
    """Plot drone-to-target distances over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#00B4D8', '#FF6B35', '#2EC4B6', '#9B5DE5', '#F4A261', '#06D6A0', '#EF476F', '#118AB2']
    T = min(len(target_trajectory), min(len(t) for t in drone_trajectories))
    time = np.arange(T) * dt

    for i, traj in enumerate(drone_trajectories):
        dist = np.linalg.norm(traj[:T] - target_trajectory[:T], axis=1)
        ax.plot(time, dist, color=colors[i % len(colors)],
                linewidth=1.5, label=f'Tracker {i}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to Target (m)')
    ax.set_title('Drone-Target Distances')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
