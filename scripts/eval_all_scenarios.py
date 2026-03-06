#!/usr/bin/env python3
"""Clean evaluation of all 5 scenarios with consistent methodology.

1. Baseline (chase+offset), normal spawn, 10K steps
2. Baseline (chase+offset), cluster spawn, 10K steps
3. Baseline (fixed formation from TRUE target), normal spawn, 10K steps
4. Best RL model, normal spawn, 10K steps
5. Best RL model, cluster spawn, 10K steps

All scenarios: 10 episodes, same seed sequence, evasive trajectory.
"""
import argparse
import numpy as np
import pybullet as p
import torch

from src.env.tracking_aviary import TrackingAviary
from src.env.tracking_aviary import spawn_in_hollow_sphere
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from src.rl.ppo import TrackingConfig
from src.rl.ppo.tracking_networks import TrackingActor
from src.rl.ppo.tracking_env import MultiDroneTrackingEnv
from src.rl.ppo.tracking_trainer import RunningMeanStd


def load_rl_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = TrackingConfig(**(ckpt.get("full_config", {})))
    actor = TrackingActor(
        device=device, obs_dim=32, action_dim=3,
        ego_hidden=cfg.ego_hidden, neighbor_hidden=cfg.neighbor_hidden,
        neighbor_embed_dim=cfg.neighbor_embed_dim,
        policy_hidden=cfg.policy_hidden,
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    obs_norm = None
    if "obs_normalizer" in ckpt:
        obs_norm = RunningMeanStd(shape=(32,))
        obs_norm.mean = ckpt["obs_normalizer"]["mean"]
        obs_norm.var = ckpt["obs_normalizer"]["var"]
        obs_norm.count = ckpt["obs_normalizer"]["count"]
    return actor, cfg, obs_norm


def apply_cluster_spawn(aviary, N, rng, min_altitude):
    """Move all drones to within 2m of each other, 100m from target."""
    target_pos = np.array([0.0, 0.0, 50.0])
    cluster_center = target_pos + np.array([100.0, 0.0, 0.0])
    positions = []
    for i in range(N):
        jitter = rng.uniform(-1.0, 1.0, size=3)
        new_pos = cluster_center + jitter
        new_pos[2] = max(new_pos[2], min_altitude)
        p.resetBasePositionAndOrientation(
            aviary.DRONE_IDS[i], new_pos, [0, 0, 0, 1],
            physicsClientId=aviary.CLIENT,
        )
        positions.append(new_pos)
    return np.array(positions)


def run_baseline_chase(cfg, seed, cluster=False):
    """Scenario 1 & 2: chase-with-offset baseline."""
    N = cfg.num_drones
    rng = np.random.default_rng(seed)

    aviary = TrackingAviary(
        num_trackers=N, target_speed=cfg.target_speed,
        target_trajectory=cfg.target_trajectory, target_sigma_a=cfg.target_sigma_a,
        episode_length=cfg.episode_length, sensor_config=cfg.sensor_config,
        pyb_freq=cfg.pyb_freq, ctrl_freq=cfg.ctrl_freq, gui=False, rng=rng,
    )
    aviary.reset()

    if cluster:
        apply_cluster_spawn(aviary, N, rng, cfg.min_altitude)

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


def run_baseline_formation(cfg, seed):
    """Scenario 3: fixed formation offset from TRUE target position (Layer 2 style)."""
    N = cfg.num_drones
    rng = np.random.default_rng(seed)

    target_initial_pos = np.array([0.0, 0.0, 50.0])
    tracker_positions = spawn_in_hollow_sphere(
        center=target_initial_pos, r_min=100.0, r_max=300.0, n=N, rng=rng,
        min_altitude=cfg.min_altitude,
    )

    aviary = TrackingAviary(
        num_trackers=N, tracker_positions=tracker_positions,
        target_initial_pos=target_initial_pos,
        target_speed=cfg.target_speed,
        target_trajectory=cfg.target_trajectory, target_sigma_a=cfg.target_sigma_a,
        episode_length=cfg.episode_length, sensor_config=cfg.sensor_config,
        pyb_freq=cfg.pyb_freq, ctrl_freq=cfg.ctrl_freq, gui=False, rng=rng,
    )
    aviary.reset()

    formation_offset = tracker_positions - target_initial_pos

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

        # Get true target state for formation following
        # Target is the last "drone" in the aviary (index N)
        target_full = aviary._getDroneStateVector(N)
        target_pos_true = target_full[:3]
        target_vel = target_full[10:13]  # velocity is at indices 10:13

        # Waypoints: true target + fixed offset
        waypoints = np.zeros((N, 3))
        vels = np.tile(target_vel, (N, 1))
        for i in range(N):
            waypoints[i] = target_pos_true + formation_offset[i]
            waypoints[i, 2] = max(cfg.min_altitude, waypoints[i, 2])

        # Use per-drone estimates for gimbal if filter initialized
        per_drone_est = None
        if filt.initialized:
            per_drone_est = np.array([
                filt.get_local_estimate(i)[:3] for i in range(N)
            ])

        result = aviary.step_tracking(waypoints, vels,
                                      per_drone_estimates=per_drone_est)

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


def run_rl_eval(cfg, actor, obs_norm, device, seed, cluster=False):
    """Scenario 4 & 5: RL policy evaluation."""
    env = MultiDroneTrackingEnv(cfg, seed=seed)
    obs, _ = env.reset()
    N = cfg.num_drones

    if cluster:
        apply_cluster_spawn(env._aviary, N,
                            np.random.default_rng(seed + 5000),
                            cfg.min_altitude)
        # Take a no-op step to get valid obs after repositioning
        obs, _, _, _, _ = env.step(np.zeros((N, 3)))

    tr_Ps, rmses = [], []

    for step in range(cfg.episode_length):
        if obs_norm is not None:
            obs_n = obs_norm.normalize(obs)
        else:
            obs_n = obs

        obs_t = torch.tensor(obs_n.reshape(N, -1),
                             dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = actor.forward(obs_t)
            action = torch.clamp(dist.mean, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

        if "tr_P_pos" in info:
            tr_Ps.append(info["tr_P_pos"])
        if info.get("filter_initialized", False):
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--traj", type=str, default="evasive")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load RL model
    actor, rl_cfg, obs_norm = load_rl_checkpoint(args.checkpoint, device)

    # Build config for baselines (match RL config)
    cfg = TrackingConfig()
    cfg.episode_length = args.steps
    cfg.target_trajectory = args.traj

    # Also override RL config episode length
    rl_cfg.episode_length = args.steps
    rl_cfg.target_trajectory = args.traj

    seeds = [args.seed + i for i in range(args.episodes)]

    print(f"{'='*65}")
    print(f"  Clean Evaluation: {args.episodes} episodes x {args.steps} steps")
    print(f"  Trajectory: {args.traj}")
    print(f"  RL checkpoint: {args.checkpoint}")
    print(f"{'='*65}\n")

    # ---- Scenario 1: Baseline normal spawn ----
    print("Scenario 1: Baseline (chase+offset), normal spawn...")
    s1_trP, s1_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_baseline_chase(cfg, seed, cluster=False)
        s1_trP.append(trP)
        s1_rmse.append(rmse)
        print(f"  Episode {i+1}/{args.episodes}: tr(P)={trP:.1f}  RMSE={rmse:.2f}")
    print()

    # ---- Scenario 2: Baseline cluster spawn ----
    print("Scenario 2: Baseline (chase+offset), cluster spawn...")
    s2_trP, s2_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_baseline_chase(cfg, seed, cluster=True)
        s2_trP.append(trP)
        s2_rmse.append(rmse)
        print(f"  Episode {i+1}/{args.episodes}: tr(P)={trP:.1f}  RMSE={rmse:.2f}")
    print()

    # ---- Scenario 3: Formation baseline (true target) ----
    print("Scenario 3: Baseline (fixed formation, TRUE target), normal spawn...")
    s3_trP, s3_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_baseline_formation(cfg, seed)
        s3_trP.append(trP)
        s3_rmse.append(rmse)
        print(f"  Episode {i+1}/{args.episodes}: tr(P)={trP:.1f}  RMSE={rmse:.2f}")
    print()

    # ---- Scenario 4: RL normal spawn ----
    print("Scenario 4: RL policy, normal spawn...")
    s4_trP, s4_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_rl_eval(rl_cfg, actor, obs_norm, device, seed, cluster=False)
        s4_trP.append(trP)
        s4_rmse.append(rmse)
        print(f"  Episode {i+1}/{args.episodes}: tr(P)={trP:.1f}  RMSE={rmse:.2f}")
    print()

    # ---- Scenario 5: RL cluster spawn ----
    print("Scenario 5: RL policy, cluster spawn...")
    s5_trP, s5_rmse = [], []
    for i, seed in enumerate(seeds):
        trP, rmse = run_rl_eval(rl_cfg, actor, obs_norm, device, seed, cluster=True)
        s5_trP.append(trP)
        s5_rmse.append(rmse)
        print(f"  Episode {i+1}/{args.episodes}: tr(P)={trP:.1f}  RMSE={rmse:.2f}")
    print()

    # ---- Summary ----
    print(f"\n{'='*65}")
    print(f"  SUMMARY ({args.episodes} episodes, {args.steps} steps, {args.traj})")
    print(f"{'='*65}")
    print(f"  {'Scenario':<45} {'tr(P)':>10} {'RMSE':>10}")
    print(f"  {'-'*65}")

    def fmt(label, trPs, rmses):
        return (f"  {label:<45} "
                f"{np.mean(trPs):>8.1f}+-{np.std(trPs):<5.1f} "
                f"{np.mean(rmses):>6.2f}+-{np.std(rmses):.2f}")

    print(fmt("1. Baseline chase+offset, normal", s1_trP, s1_rmse))
    print(fmt("2. Baseline chase+offset, cluster", s2_trP, s2_rmse))
    print(fmt("3. Baseline formation (TRUE target)", s3_trP, s3_rmse))
    print(fmt("4. RL policy, normal", s4_trP, s4_rmse))
    print(fmt("5. RL policy, cluster", s5_trP, s5_rmse))
    print()


if __name__ == "__main__":
    main()
