#!/usr/bin/env python3
"""Replay a gated policy checkpoint using animate_rl_tracking."""
import argparse
import numpy as np
import torch

from src.rl.ppo.tracking_config import TrackingConfig
from src.rl.ppo.tracking_networks import TrackingActor
from src.rl.ppo.tracking_trainer import RunningMeanStd
from experiments.tracking_env_gated import GatedTrackingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TrackingConfig()
    cfg.episode_length = args.steps
    cfg.target_trajectory = "evasive"

    gating = ckpt.get("gating", {})
    gate_threshold = gating.get("gate_threshold", 500.0)
    max_rl_authority = gating.get("max_rl_authority", 1.0)
    min_rl_authority = gating.get("min_rl_authority", 0.3)
    base_R_desired = gating.get("base_R_desired", 60.0)

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

    N = cfg.num_drones

    # Create gated env (headless — we record then animate)
    env = GatedTrackingEnv(
        cfg, seed=args.seed,
        gate_threshold=gate_threshold,
        max_rl_authority=max_rl_authority,
        min_rl_authority=min_rl_authority,
        base_R_desired=base_R_desired,
    )
    if args.cluster:
        env.spawn_mode = "cluster"

    obs, _ = env.reset()

    # Collect trajectory data
    tr_P_steps = []
    rmse_steps = []
    all_drone_pos = []
    all_target_states = []
    all_consensus_est = []
    all_local_est = []
    all_measurements = []
    all_active_edges = []

    print(f"\nReplaying {args.checkpoint}")
    print(f"  Cluster: {args.cluster}, Steps: {args.steps}")
    print(f"  Gating: threshold={gate_threshold}, max_rl={max_rl_authority}, min_rl={min_rl_authority}")
    print()

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
            tr_P_steps.append(info["tr_P_pos"])
        if info.get("filter_initialized", False) and "result" in info:
            est = env._filter.get_estimate()
            true_pos = info["result"]["target_true_pos"]
            rmse_steps.append(np.linalg.norm(est[:3] - true_pos))

        # Record trajectory
        if "result" in info:
            result = info["result"]
            all_drone_pos.append(result["drone_positions"].copy())
            all_target_states.append(result["target_true_state"].copy())
            all_measurements.append([m for m in result["measurements"]])
            if env._filter.initialized:
                all_consensus_est.append(env._filter.get_estimate().copy())
                local_step = [env._filter.get_local_estimate(i).copy() for i in range(N)]
                all_local_est.append(np.array(local_step))
                all_active_edges.append(env._filter.get_active_edges())
            else:
                all_consensus_est.append(np.zeros(6))
                all_local_est.append(np.zeros((N, 6)))
                all_active_edges.append(np.zeros((N, N)))

        if step % 500 == 0 and tr_P_steps:
            rmse_val = rmse_steps[-1] if rmse_steps else float("nan")
            print(f"  Step {step:>5d}: tr(P)={tr_P_steps[-1]:.1f}  RMSE={rmse_val:.1f}")

        if terminated or truncated:
            break

    env.close()

    mean_tr_P = np.nanmean(tr_P_steps) if tr_P_steps else float("nan")
    mean_rmse = np.nanmean(rmse_steps) if rmse_steps else float("nan")
    print(f"\n  Final: tr(P)={mean_tr_P:.1f}  RMSE={mean_rmse:.1f}")

    # Animate
    from src.viz.animation import animate_rl_tracking
    animate_rl_tracking(
        drone_positions=np.array(all_drone_pos),
        target_true_states=np.array(all_target_states),
        consensus_est=np.array(all_consensus_est),
        local_estimates=np.array(all_local_est).transpose(1, 0, 2),
        adjacency=env._adj.copy(),
        tr_P_history=np.array(tr_P_steps),
        measurements=all_measurements,
        active_edges=all_active_edges,
        dt=cfg.dt,
        title=f"Gated RL Replay ({cfg.episode_length} steps, {cfg.target_trajectory})"
              + (" [cluster spawn]" if args.cluster else ""),
        topology_name=cfg.topology,
        controller_label="Gated RL (inverted)",
    )


if __name__ == "__main__":
    main()
