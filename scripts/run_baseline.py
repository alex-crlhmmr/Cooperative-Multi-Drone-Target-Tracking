#!/usr/bin/env python3
"""Baseline heuristic for multi-drone tracking: chase-with-offset.

Each drone chases its local ConsensusIMM estimate with a fixed radial offset,
using PD control + feedforward velocity from the filter.

Usage:
    python -m scripts.run_baseline
    python -m scripts.run_baseline --episodes 20 --traj evasive --save results/baseline
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from src.env.tracking_aviary import TrackingAviary
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from src.rl.ppo import TrackingConfig


def run_baseline_episode(cfg: TrackingConfig, rng: np.random.Generator,
                         R_desired: float = 60.0, Kp: float = 2.0, Kd: float = 0.5):
    """Run one episode with chase-with-offset heuristic.

    Each drone maintains a radial offset R_desired from the local estimate,
    using PD control toward the desired position + feedforward velocity.

    Returns:
        dict with timeseries of metrics
    """
    N = cfg.num_drones

    aviary = TrackingAviary(
        num_trackers=N,
        target_speed=cfg.target_speed,
        target_trajectory=cfg.target_trajectory,
        target_sigma_a=cfg.target_sigma_a,
        episode_length=cfg.episode_length,
        sensor_config=cfg.sensor_config,
        pyb_freq=cfg.pyb_freq,
        ctrl_freq=cfg.ctrl_freq,
        gui=False,
        rng=rng,
    )
    aviary.reset()

    adj = generate_adjacency(N, cfg.topology)
    filt = ConsensusIMM(
        dt=cfg.dt,
        sigma_a_modes=list(cfg.imm_sigma_a_modes),
        sigma_bearing=cfg.sigma_bearing_rad,
        range_ref=cfg.range_ref,
        transition_matrix=cfg.transition_matrix,
        num_drones=N,
        adjacency=adj,
        num_consensus_iters=cfg.consensus_iters,
        consensus_step_size=cfg.consensus_step_size,
        dropout_prob=cfg.dropout_prob,
        P0_pos=cfg.P0_pos,
        P0_vel=cfg.P0_vel,
        rng=rng,
    )

    tr_P_history = []
    rmse_history = []
    detection_history = []
    reward_history = []

    prev_drone_positions = None

    for step in range(cfg.episode_length):
        drone_positions = np.array([
            aviary._getDroneStateVector(i)[:3] for i in range(N)
        ])

        # Compute heuristic velocity commands
        if filt.initialized:
            velocities = np.zeros((N, 3))
            per_drone_estimates = np.zeros((N, 3))

            for i in range(N):
                local_est = filt.get_local_estimate(i)
                est_pos = local_est[:3]
                est_vel = local_est[3:6]
                per_drone_estimates[i] = est_pos

                # Radial direction from estimate to drone
                radial = drone_positions[i] - est_pos
                radial_dist = np.linalg.norm(radial)

                if radial_dist > 1e-3:
                    radial_dir = radial / radial_dist
                else:
                    # Random direction if too close
                    radial_dir = rng.standard_normal(3)
                    radial_dir /= np.linalg.norm(radial_dir)

                # Desired position: estimate + radial offset
                desired_pos = est_pos + radial_dir * R_desired

                # PD control toward desired position
                pos_error = desired_pos - drone_positions[i]
                vel_error = np.zeros(3)
                if prev_drone_positions is not None:
                    vel_error = -(drone_positions[i] - prev_drone_positions[i]) / cfg.dt

                velocity = Kp * pos_error + Kd * vel_error + est_vel

                # Clamp to v_max
                speed = np.linalg.norm(velocity)
                if speed > cfg.v_max:
                    velocity = velocity * cfg.v_max / speed

                velocities[i] = velocity

            waypoints = drone_positions + velocities * cfg.dt
            waypoints[:, 2] = np.maximum(waypoints[:, 2], cfg.min_altitude)

            result = aviary.step_tracking(
                tracker_targets=waypoints,
                tracker_target_vels=velocities,
                per_drone_estimates=per_drone_estimates,
            )
        else:
            # Hold position until filter initializes
            result = aviary.step_tracking(
                tracker_targets=drone_positions,
                tracker_target_vels=np.zeros((N, 3)),
            )

        prev_drone_positions = drone_positions.copy()

        if result.get("done", False) and "drone_positions" not in result:
            break

        # Filter update
        if filt.initialized:
            filt.predict()
        filt.update(result["measurements"], result["drone_positions"])

        # Metrics
        if filt.initialized:
            P = filt.get_covariance()
            tr_P = np.trace(P[:3, :3])
            est = filt.get_estimate()
            rmse = np.linalg.norm(est[:3] - result["target_true_pos"])
        else:
            tr_P = cfg.P0_pos * 3
            rmse = np.nan

        n_detections = sum(1 for m in result["measurements"] if m is not None)

        tr_P_history.append(tr_P)
        rmse_history.append(rmse)
        detection_history.append(n_detections)

        # Compute reward (same as RL env)
        cov_reward = -cfg.w_cov * tr_P / cfg.cov_scale
        reward_history.append(cov_reward)

    aviary.close()

    return {
        "tr_P": np.array(tr_P_history),
        "rmse": np.array(rmse_history),
        "detections": np.array(detection_history),
        "reward": np.array(reward_history),
    }


def plot_baseline(all_results: list, save_dir: str, cfg):
    """Generate baseline evaluation plots."""
    os.makedirs(save_dir, exist_ok=True)
    n_eps = len(all_results)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 1: Per-step timeseries (averaged over episodes) ---
    def _mean_timeseries(key):
        max_len = max(len(r[key]) for r in all_results)
        padded = np.full((n_eps, max_len), np.nan)
        for i, r in enumerate(all_results):
            padded[i, :len(r[key])] = r[key]
        return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)

    # tr(P) over time
    ax = axes[0, 0]
    mean_trP, std_trP = _mean_timeseries("tr_P")
    steps = np.arange(len(mean_trP))
    ax.plot(steps, mean_trP, color="C1")
    ax.fill_between(steps, mean_trP - std_trP, mean_trP + std_trP, alpha=0.2, color="C1")
    ax.set_xlabel("Step")
    ax.set_ylabel("tr(P_pos)")
    ax.set_title("Filter Uncertainty Over Time")
    ax.grid(True, alpha=0.3)

    # RMSE over time
    ax = axes[0, 1]
    mean_rmse, std_rmse = _mean_timeseries("rmse")
    steps = np.arange(len(mean_rmse))
    ax.plot(steps, mean_rmse, color="C0")
    ax.fill_between(steps, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.2, color="C0")
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Position RMSE Over Time")
    ax.grid(True, alpha=0.3)

    # Cumulative reward over time
    ax = axes[0, 2]
    mean_rew, _ = _mean_timeseries("reward")
    cum_rew = np.nancumsum(mean_rew)
    ax.plot(np.arange(len(cum_rew)), cum_rew, color="C2")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward")
    ax.grid(True, alpha=0.3)

    # --- Row 2: Per-episode distributions ---
    ep_trPs = [np.nanmean(r["tr_P"]) for r in all_results]
    ep_rmses = [np.nanmean(r["rmse"]) for r in all_results]
    ep_rewards = [np.sum(r["reward"]) for r in all_results]

    # tr(P) distribution
    ax = axes[1, 0]
    ax.hist(ep_trPs, bins=min(20, n_eps), color="C1", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(ep_trPs), color="red", linestyle="--", label=f"Mean={np.mean(ep_trPs):.1f}")
    ax.set_xlabel("Mean tr(P)")
    ax.set_ylabel("Count")
    ax.set_title("Episode tr(P) Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE distribution
    ax = axes[1, 1]
    ax.hist(ep_rmses, bins=min(20, n_eps), color="C0", alpha=0.7, edgecolor="black")
    ax.axvline(np.nanmean(ep_rmses), color="red", linestyle="--", label=f"Mean={np.nanmean(ep_rmses):.2f}")
    ax.set_xlabel("Mean RMSE (m)")
    ax.set_ylabel("Count")
    ax.set_title("Episode RMSE Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Detections over time
    ax = axes[1, 2]
    mean_det, std_det = _mean_timeseries("detections")
    steps = np.arange(len(mean_det))
    ax.plot(steps, mean_det, color="C4")
    ax.fill_between(steps, mean_det - std_det, mean_det + std_det, alpha=0.2, color="C4")
    ax.set_xlabel("Step")
    ax.set_ylabel("Detections")
    ax.set_title(f"Active Detections (of {cfg.num_drones})")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Baseline (Chase-with-Offset) — {cfg.num_drones} Drones, "
                 f"{n_eps} Episodes, {cfg.target_trajectory}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "baseline_plots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved plots: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run baseline chase-with-offset heuristic")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--save", type=str, default=None, help="Save results to directory")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-drones", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-from", type=str, default=None,
                        help="Load config from a training checkpoint .pt (ensures identical setup)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    # Build config — from checkpoint or defaults, then apply CLI overrides
    if args.config_from:
        import torch
        ckpt = torch.load(args.config_from, map_location="cpu", weights_only=False)
        if "full_config" in ckpt:
            cfg = TrackingConfig(**ckpt["full_config"])
            print(f"Loaded config from {args.config_from}")
        else:
            cfg = TrackingConfig()
            print(f"Warning: checkpoint has no full_config, using defaults")
    else:
        cfg = TrackingConfig()

    # CLI overrides (take precedence over checkpoint config)
    if args.traj is not None:
        cfg.target_trajectory = args.traj
    if args.steps is not None:
        cfg.episode_length = args.steps
    if args.num_drones is not None:
        cfg.num_drones = args.num_drones

    rng = np.random.default_rng(args.seed)

    all_results = []
    all_tr_P = []
    all_rmse = []
    all_reward = []

    print(f"Running baseline: {args.episodes} episodes, {cfg.num_drones} drones, "
          f"traj={cfg.target_trajectory}, steps={cfg.episode_length}")

    for ep in range(args.episodes):
        result = run_baseline_episode(cfg, rng)
        all_results.append(result)
        mean_tr_P = np.nanmean(result["tr_P"])
        mean_rmse = np.nanmean(result["rmse"])
        mean_reward = np.mean(result["reward"])
        all_tr_P.append(mean_tr_P)
        all_rmse.append(mean_rmse)
        all_reward.append(mean_reward)
        print(f"  Episode {ep+1}/{args.episodes}: "
              f"tr(P)={mean_tr_P:.1f}  RMSE={mean_rmse:.2f}  reward={mean_reward:.3f}")

    print(f"\n{'='*50}")
    print(f"Baseline Summary ({args.episodes} episodes)")
    print(f"{'='*50}")
    print(f"  Mean tr(P):  {np.mean(all_tr_P):.1f} +/- {np.std(all_tr_P):.1f}")
    print(f"  Mean RMSE:   {np.nanmean(all_rmse):.2f} +/- {np.nanstd(all_rmse):.2f}")
    print(f"  Mean Reward: {np.mean(all_reward):.3f} +/- {np.std(all_reward):.3f}")

    save_dir = args.save or "results/baseline"

    if args.save:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, "baseline_results.npz"),
            tr_P=np.array(all_tr_P),
            rmse=np.array(all_rmse),
            reward=np.array(all_reward),
        )
        print(f"  Saved to {save_dir}/baseline_results.npz")

    if not args.no_plot:
        plot_baseline(all_results, save_dir, cfg)


if __name__ == "__main__":
    main()
