#!/usr/bin/env python3
"""Run two baselines for comparison with RL v4.

Baseline 1: "Cheating" formation — follow TRUE target trajectory with fixed
    offsets from initial hollow-sphere spawn. Normal spawn. Local gimbal.
Baseline 2: Chase+offset from filter estimate. Cluster spawn. Local gimbal.

Both use ConsensusIMM, 5 drones, evasive trajectory, 5K steps.
"""
import argparse
import numpy as np

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from src.rl.ppo.tracking_config import TrackingConfig


def run_formation_oracle(cfg, seed):
    """Baseline 1: formation following TRUE target, normal spawn."""
    N = cfg.num_drones
    rng = np.random.default_rng(seed)

    target_pos = np.array([0.0, 0.0, 50.0])
    tracker_positions = spawn_in_hollow_sphere(
        center=target_pos, r_min=100.0, r_max=300.0, n=N, rng=rng,
        min_altitude=cfg.min_altitude,
    )

    aviary = TrackingAviary(
        num_trackers=N, tracker_positions=tracker_positions,
        target_initial_pos=target_pos, target_speed=cfg.target_speed,
        target_trajectory=cfg.target_trajectory, target_sigma_a=cfg.target_sigma_a,
        episode_length=cfg.episode_length, sensor_config=cfg.sensor_config,
        pyb_freq=cfg.pyb_freq, ctrl_freq=cfg.ctrl_freq, gui=False, rng=rng,
    )
    aviary.reset()

    formation_offset = tracker_positions - target_pos

    adj = generate_adjacency(N, cfg.topology)
    filt = ConsensusIMM(
        dt=cfg.dt, sigma_a_modes=list(cfg.imm_sigma_a_modes),
        sigma_bearing=cfg.sigma_bearing_rad, range_ref=cfg.range_ref,
        transition_matrix=cfg.transition_matrix, num_drones=N, adjacency=adj,
        num_consensus_iters=cfg.consensus_iters,
        consensus_step_size=cfg.consensus_step_size,
        dropout_prob=cfg.dropout_prob, P0_pos=cfg.P0_pos, P0_vel=cfg.P0_vel, rng=rng,
    )

    tr_Ps, rmses = [], []

    for step in range(cfg.episode_length):
        drone_pos = np.array([aviary._getDroneStateVector(i)[:3] for i in range(N)])

        target_full = aviary._getDroneStateVector(N)
        target_pos_true = target_full[:3]
        target_vel = target_full[10:13]

        waypoints = np.zeros((N, 3))
        vels = np.tile(target_vel, (N, 1))
        for i in range(N):
            waypoints[i] = target_pos_true + formation_offset[i]
            waypoints[i, 2] = max(cfg.min_altitude, waypoints[i, 2])

        per_drone_est = None
        if filt.initialized:
            per_drone_est = np.array([
                filt.get_local_estimate(i)[:3] for i in range(N)
            ])

        result = aviary.step_tracking(waypoints, vels, per_drone_estimates=per_drone_est)

        drone_pos = np.array([aviary._getDroneStateVector(i)[:3] for i in range(N)])
        filt.predict()
        filt.update(result["measurements"], drone_pos)

        if filt.initialized:
            P = filt.get_covariance()
            tr_Ps.append(np.trace(P[:3, :3]))
            est = filt.get_estimate()
            rmses.append(np.linalg.norm(est[:3] - result["target_true_pos"]))

    aviary.close()
    return np.nanmean(tr_Ps), np.nanmean(rmses)


def run_chase_cluster(cfg, seed):
    """Baseline 2: chase+offset from estimate, cluster spawn."""
    import pybullet as p
    N = cfg.num_drones
    rng = np.random.default_rng(seed)

    aviary = TrackingAviary(
        num_trackers=N, target_speed=cfg.target_speed,
        target_trajectory=cfg.target_trajectory, target_sigma_a=cfg.target_sigma_a,
        episode_length=cfg.episode_length, sensor_config=cfg.sensor_config,
        pyb_freq=cfg.pyb_freq, ctrl_freq=cfg.ctrl_freq, gui=False, rng=rng,
    )
    aviary.reset()

    # Cluster spawn: all drones near one point, ±5m spread
    target_pos = np.array([0.0, 0.0, 50.0])
    direction = rng.standard_normal(3)
    direction[2] = abs(direction[2])
    direction /= np.linalg.norm(direction) + 1e-8
    cluster_center = target_pos + direction * 100.0
    cluster_center[2] = max(cluster_center[2], cfg.min_altitude + 10.0)
    for i in range(N):
        jitter = rng.uniform(-5.0, 5.0, size=3)
        new_pos = cluster_center + jitter
        new_pos[2] = max(new_pos[2], cfg.min_altitude)
        p.resetBasePositionAndOrientation(
            aviary.DRONE_IDS[i], new_pos, [0, 0, 0, 1],
            physicsClientId=aviary.CLIENT,
        )

    adj = generate_adjacency(N, cfg.topology)
    filt = ConsensusIMM(
        dt=cfg.dt, sigma_a_modes=list(cfg.imm_sigma_a_modes),
        sigma_bearing=cfg.sigma_bearing_rad, range_ref=cfg.range_ref,
        transition_matrix=cfg.transition_matrix, num_drones=N, adjacency=adj,
        num_consensus_iters=cfg.consensus_iters,
        consensus_step_size=cfg.consensus_step_size,
        dropout_prob=cfg.dropout_prob, P0_pos=cfg.P0_pos, P0_vel=cfg.P0_vel, rng=rng,
    )

    R_desired, Kp = 60.0, 2.0
    tr_Ps, rmses = [], []

    for step in range(cfg.episode_length):
        drone_pos = np.array([aviary._getDroneStateVector(i)[:3] for i in range(N)])

        if filt.initialized:
            velocities = np.zeros((N, 3))
            per_drone_est = np.zeros((N, 3))
            for i in range(N):
                local_est = filt.get_local_estimate(i)
                est_pos, est_vel = local_est[:3], local_est[3:6]
                per_drone_est[i] = est_pos
                radial = drone_pos[i] - est_pos
                rdist = np.linalg.norm(radial)
                if rdist > 1e-3:
                    rdir = radial / rdist
                else:
                    rdir = rng.standard_normal(3)
                    rdir /= np.linalg.norm(rdir)
                desired = est_pos + rdir * R_desired
                vel = Kp * (desired - drone_pos[i]) + est_vel
                speed = np.linalg.norm(vel)
                if speed > cfg.v_max:
                    vel = vel / speed * cfg.v_max
                velocities[i] = vel
            waypoints = drone_pos + velocities * cfg.dt
            waypoints[:, 2] = np.maximum(waypoints[:, 2], cfg.min_altitude)
            result = aviary.step_tracking(waypoints, velocities,
                                          per_drone_estimates=per_drone_est)
        else:
            result = aviary.step_tracking(drone_pos, np.zeros((N, 3)))

        drone_pos = np.array([aviary._getDroneStateVector(i)[:3] for i in range(N)])
        filt.predict()
        filt.update(result["measurements"], drone_pos)

        if filt.initialized:
            P = filt.get_covariance()
            tr_Ps.append(np.trace(P[:3, :3]))
            est = filt.get_estimate()
            rmses.append(np.linalg.norm(est[:3] - result["target_true_pos"]))

    aviary.close()
    return np.nanmean(tr_Ps), np.nanmean(rmses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--traj", type=str, default="evasive")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrackingConfig()
    cfg.episode_length = args.steps
    cfg.target_trajectory = args.traj

    seeds = [args.seed + i for i in range(args.episodes)]

    print(f"{'='*65}")
    print(f"  V4 Baselines: {args.episodes} episodes x {args.steps} steps, {args.traj}")
    print(f"{'='*65}\n")

    # Baseline 1: Formation oracle, normal spawn
    print("Baseline 1: Formation oracle (TRUE target), normal spawn, local gimbal")
    b1_trP, b1_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_formation_oracle(cfg, seed)
        b1_trP.append(trP)
        b1_rmse.append(rmse)
        print(f"  Ep {i+1}/{args.episodes}: tr(P)={trP:.1f}  RMSE={rmse:.2f}")

    print(f"  >>> MEAN: tr(P)={np.mean(b1_trP):.1f} ± {np.std(b1_trP):.1f}  "
          f"RMSE={np.mean(b1_rmse):.2f} ± {np.std(b1_rmse):.2f}\n")

    # Baseline 2: Chase+offset, cluster spawn
    print("Baseline 2: Chase+offset, cluster spawn, local gimbal")
    b2_trP, b2_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_chase_cluster(cfg, seed)
        b2_trP.append(trP)
        b2_rmse.append(rmse)
        print(f"  Ep {i+1}/{args.episodes}: tr(P)={trP:.1f}  RMSE={rmse:.2f}")

    print(f"  >>> MEAN: tr(P)={np.mean(b2_trP):.1f} ± {np.std(b2_trP):.1f}  "
          f"RMSE={np.mean(b2_rmse):.2f} ± {np.std(b2_rmse):.2f}\n")

    # Summary
    print(f"{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Scenario':<45} {'tr(P)':>12} {'RMSE':>12}")
    print(f"  {'-'*65}")
    print(f"  {'1. Formation oracle, normal spawn':<45} "
          f"{np.mean(b1_trP):>7.1f}±{np.std(b1_trP):<4.1f} "
          f"{np.mean(b1_rmse):>7.2f}±{np.std(b1_rmse):.2f}")
    print(f"  {'2. Chase+offset, cluster spawn':<45} "
          f"{np.mean(b2_trP):>7.1f}±{np.std(b2_trP):<4.1f} "
          f"{np.mean(b2_rmse):>7.2f}±{np.std(b2_rmse):.2f}")
    print(f"  {'3. RL v4, cluster spawn':<45} {'(after training)':>12}")
    print()


if __name__ == "__main__":
    main()
