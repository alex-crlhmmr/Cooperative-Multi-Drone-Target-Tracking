#!/usr/bin/env python3
"""Paper-quality evaluation: 3 scenarios, 20 episodes each, 10K steps.

1. Formation oracle (TRUE target), normal spawn — upper bound
2. Chase+offset baseline, cluster spawn — lower bound
3. RL v4, cluster spawn — our method

All use ConsensusIMM, 5 drones, evasive trajectory, local gimbal.
"""
import argparse
import numpy as np
import pybullet as p
import torch

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from src.rl.ppo.tracking_config import TrackingConfig
from src.rl.ppo.tracking_networks import TrackingActor
from src.rl.ppo.tracking_trainer import RunningMeanStd
from experiments.tracking_env_gated import GatedTrackingEnv


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
    N = cfg.num_drones
    rng = np.random.default_rng(seed)

    aviary = TrackingAviary(
        num_trackers=N, target_speed=cfg.target_speed,
        target_trajectory=cfg.target_trajectory, target_sigma_a=cfg.target_sigma_a,
        episode_length=cfg.episode_length, sensor_config=cfg.sensor_config,
        pyb_freq=cfg.pyb_freq, ctrl_freq=cfg.ctrl_freq, gui=False, rng=rng,
    )
    aviary.reset()

    # Cluster spawn
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


def run_rl_cluster(cfg, actor, obs_norm, device, seed, gate_threshold,
                   max_rl_authority, min_rl_authority, base_R_desired):
    """RL policy, cluster spawn."""
    env = GatedTrackingEnv(
        cfg, seed=seed,
        gate_threshold=gate_threshold,
        max_rl_authority=max_rl_authority,
        min_rl_authority=min_rl_authority,
        base_R_desired=base_R_desired,
    )
    env.spawn_mode = "cluster"
    obs, _ = env.reset()
    N = cfg.num_drones

    tr_Ps, rmses = [], []

    for step in range(cfg.episode_length):
        if obs_norm is not None:
            obs_n = obs_norm.normalize(obs)
        else:
            obs_n = obs

        obs_t = torch.tensor(obs_n.reshape(N, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = actor.forward(obs_t)
            action = torch.clamp(dist.mean, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

        if "tr_P_pos" in info:
            tr_Ps.append(info["tr_P_pos"])
        if info.get("filter_initialized", False) and "result" in info:
            est = env._filter.get_estimate()
            true_pos = info["result"]["target_true_pos"]
            rmses.append(np.linalg.norm(est[:3] - true_pos))

        if terminated or truncated:
            break

    env.close()
    return np.nanmean(tr_Ps), np.nanmean(rmses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--traj", type=str, default="evasive")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load RL checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    rl_cfg = TrackingConfig()
    rl_cfg.episode_length = args.steps
    rl_cfg.target_trajectory = args.traj

    gating = ckpt.get("gating", {})
    gate_threshold = gating.get("gate_threshold", 500.0)
    max_rl_authority = gating.get("max_rl_authority", 1.0)
    min_rl_authority = gating.get("min_rl_authority", 0.3)
    base_R_desired = gating.get("base_R_desired", 60.0)

    actor = TrackingActor(
        device=device, obs_dim=32, action_dim=3,
        ego_hidden=rl_cfg.ego_hidden, neighbor_hidden=rl_cfg.neighbor_hidden,
        neighbor_embed_dim=rl_cfg.neighbor_embed_dim,
        policy_hidden=rl_cfg.policy_hidden,
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_norm = None
    if "obs_normalizer" in ckpt:
        obs_norm = RunningMeanStd(shape=(32,))
        obs_norm.mean = ckpt["obs_normalizer"]["mean"]
        obs_norm.var = ckpt["obs_normalizer"]["var"]
        obs_norm.count = ckpt["obs_normalizer"]["count"]

    # Baseline config
    cfg = TrackingConfig()
    cfg.episode_length = args.steps
    cfg.target_trajectory = args.traj

    seeds = [args.seed + i for i in range(args.episodes)]

    print(f"{'='*70}")
    print(f"  Paper Evaluation: {args.episodes} episodes x {args.steps} steps, {args.traj}")
    print(f"  RL checkpoint: {args.checkpoint}")
    print(f"{'='*70}\n")

    # --- Scenario 1: Formation oracle ---
    print("1. Formation oracle (TRUE target), normal spawn, local gimbal")
    b1_trP, b1_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_formation_oracle(cfg, seed)
        b1_trP.append(trP)
        b1_rmse.append(rmse)
        print(f"   Ep {i+1:>2d}/{args.episodes}: tr(P)={trP:>10.1f}  RMSE={rmse:>8.2f}")
    print(f"   MEAN: tr(P)={np.mean(b1_trP):.1f} +/- {np.std(b1_trP):.1f}  "
          f"RMSE={np.mean(b1_rmse):.2f} +/- {np.std(b1_rmse):.2f}\n")

    # --- Scenario 2: Chase+offset, cluster ---
    print("2. Chase+offset baseline, cluster spawn, local gimbal")
    b2_trP, b2_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_chase_cluster(cfg, seed)
        b2_trP.append(trP)
        b2_rmse.append(rmse)
        print(f"   Ep {i+1:>2d}/{args.episodes}: tr(P)={trP:>10.1f}  RMSE={rmse:>8.2f}")
    print(f"   MEAN: tr(P)={np.mean(b2_trP):.1f} +/- {np.std(b2_trP):.1f}  "
          f"RMSE={np.mean(b2_rmse):.2f} +/- {np.std(b2_rmse):.2f}\n")

    # --- Scenario 3: RL, cluster ---
    print("3. RL v4 (gated), cluster spawn")
    b3_trP, b3_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_rl_cluster(
            rl_cfg, actor, obs_norm, device, seed,
            gate_threshold, max_rl_authority, min_rl_authority, base_R_desired,
        )
        b3_trP.append(trP)
        b3_rmse.append(rmse)
        print(f"   Ep {i+1:>2d}/{args.episodes}: tr(P)={trP:>10.1f}  RMSE={rmse:>8.2f}")
    print(f"   MEAN: tr(P)={np.mean(b3_trP):.1f} +/- {np.std(b3_trP):.1f}  "
          f"RMSE={np.mean(b3_rmse):.2f} +/- {np.std(b3_rmse):.2f}\n")

    # --- Paper table ---
    print(f"{'='*70}")
    print(f"  PAPER TABLE ({args.episodes} episodes, {args.steps} steps, {args.traj})")
    print(f"{'='*70}")
    print(f"  {'Scenario':<40} {'tr(P)':>14} {'RMSE (m)':>14}")
    print(f"  {'-'*68}")

    def fmt(label, trPs, rmses):
        return (f"  {label:<40} "
                f"{np.mean(trPs):>6.1f} +/- {np.std(trPs):<6.1f} "
                f"{np.mean(rmses):>6.2f} +/- {np.std(rmses):.2f}")

    print(fmt("Formation oracle (normal, cheating)", b1_trP, b1_rmse))
    print(fmt("Chase+offset (cluster)", b2_trP, b2_rmse))
    print(fmt("RL v4 gated (cluster)", b3_trP, b3_rmse))

    # Improvement
    if np.mean(b2_trP) > 0:
        improv = np.mean(b2_trP) / max(np.mean(b3_trP), 1e-6)
        print(f"\n  RL vs chase baseline: {improv:.0f}x improvement in tr(P)")
    print()


if __name__ == "__main__":
    main()
