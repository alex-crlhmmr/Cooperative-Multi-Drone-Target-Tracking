#!/usr/bin/env python3
"""Evaluate spread heuristic + chase baseline (NO RL) on cluster spawn.

Tests whether the ~50% cluster failure rate is fundamental to bearing geometry
or specific to the RL policy. Uses same env setup as GatedTrackingEnv but
with rl_weight=0 throughout (pure baseline control).

Spread phase: fly away from centroid for N steps at max speed.
Chase phase: chase+offset from filter estimate (same as GatedTrackingEnv baseline).
"""
import argparse
import sys
import numpy as np
import pybullet as p

from src.env.tracking_aviary import TrackingAviary
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from src.rl.ppo.tracking_config import TrackingConfig


def run_spread_chase(cfg, seed, spread_steps=200, R_desired=60.0, Kp=2.0):
    """Spread heuristic → chase+offset baseline, cluster spawn, no RL."""
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

    tr_Ps, rmses = [], []

    for step in range(cfg.episode_length):
        drone_pos = np.array([aviary._getDroneStateVector(i)[:3] for i in range(N)])

        # === PHASE 1: Spread from centroid ===
        if step < spread_steps:
            centroid = drone_pos.mean(axis=0)
            velocities = np.zeros((N, 3))
            for i in range(N):
                away = drone_pos[i] - centroid
                dist = np.linalg.norm(away)
                if dist > 1e-3:
                    direction = away / dist
                else:
                    angle = 2 * np.pi * i / N
                    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                velocities[i] = direction * cfg.v_max

            waypoints = drone_pos + velocities * cfg.dt
            waypoints[:, 2] = np.maximum(waypoints[:, 2], cfg.min_altitude)

            per_drone_est = None
            if filt.initialized:
                per_drone_est = np.array([
                    filt.get_local_estimate(i)[:3] for i in range(N)
                ])

            result = aviary.step_tracking(waypoints, velocities,
                                          per_drone_estimates=per_drone_est)

        # === PHASE 2: Chase+offset from estimate ===
        else:
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
            P_mat = filt.get_covariance()
            tr_Ps.append(np.trace(P_mat[:3, :3]))
            est = filt.get_estimate()
            rmses.append(np.linalg.norm(est[:3] - result["target_true_pos"]))

    aviary.close()
    return np.nanmean(tr_Ps), np.nanmean(rmses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--spread-steps", type=int, default=200)
    parser.add_argument("--traj", type=str, default="evasive")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrackingConfig()
    cfg.episode_length = args.steps
    cfg.target_trajectory = args.traj

    seeds = [args.seed + i for i in range(args.episodes)]

    print(f"Spread({args.spread_steps}) + Chase baseline, cluster spawn")
    print(f"{args.episodes} episodes x {args.steps} steps, {args.traj}")
    print(f"{'='*60}")
    sys.stdout.flush()

    all_trP, all_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_spread_chase(cfg, seed, spread_steps=args.spread_steps)
        all_trP.append(trP)
        all_rmse.append(rmse)
        print(f"  Ep {i+1:>2}/{args.episodes}: tr(P)={trP:>10.1f}  RMSE={rmse:>8.2f}")
        sys.stdout.flush()

    print(f"\n  MEAN:   tr(P)={np.mean(all_trP):.1f} +/- {np.std(all_trP):.1f}  "
          f"RMSE={np.mean(all_rmse):.2f} +/- {np.std(all_rmse):.2f}")
    print(f"  MEDIAN: tr(P)={np.median(all_trP):.1f}  RMSE={np.median(all_rmse):.2f}")

    good = sum(1 for t in all_trP if t < 100)
    print(f"  Good episodes (tr(P)<100): {good}/{args.episodes}")


if __name__ == "__main__":
    main()
