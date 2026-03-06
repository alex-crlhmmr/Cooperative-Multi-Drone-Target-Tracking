#!/usr/bin/env python3
"""Training script for adaptive-gated residual policy.

Key difference from scripts/train_rl.py:
- Uses GatedTrackingEnv instead of MultiDroneTrackingEnv
- RL authority adapts based on tr(P): high uncertainty → pure baseline, low → RL kicks in
- Configurable gate_threshold and max_rl_authority

Usage:
    python -m experiments.train_gated --steps 5000000 --save-path output/rl_gated_v1 --tag gated_v1
    python -m experiments.train_gated --gate-threshold 200 --max-rl-authority 0.7
    python -m experiments.train_gated --resume output/rl_gated_v1/checkpoint_best.pt
"""
import argparse
import os
import sys
import time
import torch
import numpy as np

from src.rl.ppo.tracking_config import TrackingConfig
from src.rl.ppo.tracking_networks import TrackingActor, TrackingCritic
from src.rl.ppo.ppo_agent import PPOAgent
from src.rl.ppo.tracking_buffer import MultiAgentRolloutBuffer
from src.rl.ppo.tracking_trainer import RunningMeanStd
from src.rl.ppo.vec_env import SubprocVecEnv
from experiments.tracking_env_gated import GatedTrackingEnv


def make_env_fn(cfg, seed, gate_threshold, max_rl_authority, min_rl_authority, base_R_desired, augment_obs=False, spread_steps=200):
    """Factory for GatedTrackingEnv."""
    def _init():
        return GatedTrackingEnv(
            cfg, seed=seed,
            gate_threshold=gate_threshold,
            max_rl_authority=max_rl_authority,
            min_rl_authority=min_rl_authority,
            base_R_desired=base_R_desired,
            augment_obs=augment_obs,
            spread_steps=spread_steps,
        )
    return _init


def run_eval(cfg, actor, obs_normalizer, device, gate_threshold, max_rl_authority,
             min_rl_authority, base_R_desired, spawn_mode="normal", num_episodes=3,
             spread_steps=200):
    """Run deterministic eval episodes."""
    rewards, tr_Ps, rmses = [], [], []

    for ep in range(num_episodes):
        env = GatedTrackingEnv(
            cfg, seed=10000 + ep,
            gate_threshold=gate_threshold,
            max_rl_authority=max_rl_authority,
            min_rl_authority=min_rl_authority,
            base_R_desired=base_R_desired,
            spread_steps=spread_steps,
        )
        env.spawn_mode = spawn_mode
        obs, _ = env.reset()
        N = cfg.num_drones
        ep_reward = 0.0
        ep_tr_P, ep_rmse = [], []

        for step in range(cfg.episode_length):
            if obs_normalizer is not None:
                obs_norm = obs_normalizer.normalize(obs)
            else:
                obs_norm = obs

            obs_t = torch.tensor(
                obs_norm.reshape(N, -1), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                dist = actor.forward(obs_t)
                action = torch.clamp(dist.mean, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            ep_reward += reward.mean()

            if "tr_P_pos" in info:
                ep_tr_P.append(info["tr_P_pos"])
            if info.get("filter_initialized", False) and "result" in info:
                est = env._filter.get_estimate()
                true_pos = info["result"]["target_true_pos"]
                ep_rmse.append(np.linalg.norm(est[:3] - true_pos))

            if terminated or truncated:
                break

        env.close()
        rewards.append(ep_reward)
        tr_Ps.append(np.nanmean(ep_tr_P) if ep_tr_P else np.nan)
        rmses.append(np.nanmean(ep_rmse) if ep_rmse else np.nan)

    return {
        "reward": np.nanmean(rewards),
        "tr_P": np.nanmean(tr_Ps),
        "rmse": np.nanmean(rmses),
    }


def append_eval_csv(save_path, step, result):
    csv_path = os.path.join(save_path, "eval_results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write("step,reward,tr_P,rmse\n")
        f.write(f"{step},{result['reward']:.4f},{result['tr_P']:.4f},{result['rmse']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Train gated residual RL policy")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--save-path", type=str, default="./output/rl_gated_v1")
    parser.add_argument("--tag", type=str, default="gated")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--traj", type=str, default="evasive")
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--spawn-mode", type=str, default="normal",
                        choices=["normal", "mixed", "cluster"])
    # Gating params (INVERTED: high tr(P) → more RL)
    parser.add_argument("--gate-threshold", type=float, default=500.0,
                        help="tr(P) at which RL weight reaches max (saturation point)")
    parser.add_argument("--max-rl-authority", type=float, default=1.0,
                        help="RL weight when tr(P) >= threshold (high uncertainty)")
    parser.add_argument("--min-rl-authority", type=float, default=0.3,
                        help="RL weight when tr(P) ~ 0 (low uncertainty, baseline dominates)")
    parser.add_argument("--base-R-desired", type=float, default=60.0,
                        help="Radial offset for base chase+offset controller")
    # Reward weights
    parser.add_argument("--w-cov", type=float, default=None)
    parser.add_argument("--w-dist", type=float, default=None)
    parser.add_argument("--w-sep", type=float, default=None)
    parser.add_argument("--w-detect", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor (default from config: 0.95)")
    parser.add_argument("--clip-epsilon", type=float, default=None,
                        help="PPO clip epsilon (default 0.2)")
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help="Rollout steps per update (default 1500)")
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="Entropy coefficient (default 0.01)")
    parser.add_argument("--augment-obs", action="store_true",
                        help="Random yaw rotation augmentation for rotation invariance")
    parser.add_argument("--spread-steps", type=int, default=200,
                        help="Heuristic spread phase duration (default 200)")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    cfg = TrackingConfig()
    cfg.target_trajectory = args.traj
    cfg.max_training_steps = args.steps
    cfg.num_envs = args.num_envs
    cfg.save_path = args.save_path
    cfg.lr = args.lr

    if args.episode_length is not None:
        cfg.episode_length = args.episode_length
    if args.w_cov is not None:
        cfg.w_cov = args.w_cov
    if args.w_dist is not None:
        cfg.w_dist = args.w_dist
    if args.w_sep is not None:
        cfg.w_separation = args.w_sep
    if args.w_detect is not None:
        cfg.w_detection = args.w_detect
    if args.gamma is not None:
        cfg.gamma = args.gamma
    if args.clip_epsilon is not None:
        cfg.clip_epsilon = args.clip_epsilon
    if args.rollout_steps is not None:
        cfg.rollout_steps = args.rollout_steps
    if args.entropy_coef is not None:
        cfg.entropy_coef = args.entropy_coef

    N = cfg.num_drones
    E = cfg.num_envs
    OBS_DIM = 32  # Per-drone observation dimension (from MultiDroneTrackingEnv)

    # Resume
    start_step = 0
    obs_normalizer = RunningMeanStd(shape=(OBS_DIM,))

    actor = TrackingActor(
        device=device, obs_dim=OBS_DIM, action_dim=3,
        ego_hidden=cfg.ego_hidden, neighbor_hidden=cfg.neighbor_hidden,
        neighbor_embed_dim=cfg.neighbor_embed_dim, policy_hidden=cfg.policy_hidden,
    ).to(device)

    critic = TrackingCritic(
        device=device, obs_dim=OBS_DIM,
        ego_hidden=cfg.ego_hidden, neighbor_hidden=cfg.neighbor_hidden,
        neighbor_embed_dim=cfg.neighbor_embed_dim,
        value_hidden=cfg.policy_hidden,
    ).to(device)

    agent = PPOAgent(
        actor=actor, critic=critic,
        lr=cfg.lr, clip_epsilon=cfg.clip_epsilon,
        num_epochs=cfg.num_epochs,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        mini_batch_size=cfg.mini_batch_size,
        max_grad_norm=cfg.max_grad_norm,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            agent.optimizer.load_state_dict(ckpt["optimizer"])
        if "obs_normalizer" in ckpt:
            obs_normalizer.mean = ckpt["obs_normalizer"]["mean"]
            obs_normalizer.var = ckpt["obs_normalizer"]["var"]
            obs_normalizer.count = ckpt["obs_normalizer"]["count"]
        start_step = ckpt.get("step", 0)
        # Update LR if specified
        for pg in agent.optimizer.param_groups:
            pg["lr"] = cfg.lr
        print(f"  Resumed at step {start_step}, LR set to {cfg.lr}")

    # Create parallel envs
    env_fns = [
        make_env_fn(cfg, args.seed + i, args.gate_threshold,
                    args.max_rl_authority, args.min_rl_authority, args.base_R_desired,
                    augment_obs=args.augment_obs, spread_steps=args.spread_steps)
        for i in range(E)
    ]
    vec_env = SubprocVecEnv(env_fns)

    # Set spawn mode
    vec_env.set_attr("spawn_mode", args.spawn_mode)

    # Buffer
    buffer = MultiAgentRolloutBuffer(
        rollout_steps=cfg.rollout_steps,
        num_envs=E, num_drones=N,
        obs_dim=OBS_DIM, action_dim=3,
        device=device,
    )

    # Print config
    print(f"\n{'='*60}")
    print(f"  INVERTED Gated Residual RL Training")
    print(f"{'='*60}")
    print(f"  Gate threshold:    {args.gate_threshold}")
    print(f"  Max RL authority:  {args.max_rl_authority} (high uncertainty)")
    print(f"  Min RL authority:  {args.min_rl_authority} (low uncertainty)")
    print(f"  Base R_desired:    {args.base_R_desired}")
    print(f"  Spawn mode:        {args.spawn_mode}")
    print(f"  Episode length:    {cfg.episode_length}")
    print(f"  Num envs:          {E}")
    print(f"  Rollout steps:     {cfg.rollout_steps}")
    print(f"  Max steps:         {cfg.max_training_steps}")
    print(f"  LR:                {cfg.lr}")
    print(f"  Gamma:             {cfg.gamma}")
    print(f"  Reward weights:    cov={cfg.w_cov}, dist={cfg.w_dist}, "
          f"sep={cfg.w_separation}, det={cfg.w_detection}")
    print(f"  Save path:         {args.save_path}")
    if args.resume:
        print(f"  Resumed from:      {args.resume} (step {start_step})")
    print(f"{'='*60}\n")

    # Training loop
    obs = vec_env.reset()  # (E, N, obs_dim)
    total_steps = start_step
    num_rollouts = 0
    t_start = time.time()
    best_tr_P = float("inf")

    while total_steps < cfg.max_training_steps:
        # Collect rollout
        buffer.reset()

        with torch.no_grad():
            for t in range(cfg.rollout_steps):
                obs_flat = obs.reshape(E * N, -1)
                obs_normalizer.update(obs_flat)
                obs_norm = obs_normalizer.normalize(obs_flat)

                obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device)

                # Policy forward pass (act handles tanh squashing + Jacobian)
                actions_flat, log_probs_flat = actor.act(obs_t)
                values_flat = critic(obs_t)

                # Reshape back to (E, N, ...)
                actions_en = actions_flat.reshape(E, N, -1)
                log_probs_en = log_probs_flat.reshape(E, N)
                values_en = values_flat.reshape(E, N)

                # Step all envs
                action_np = actions_en.cpu().numpy()
                next_obs, rewards, dones, infos = vec_env.step(action_np)

                buffer.push(
                    states=torch.tensor(obs_norm.reshape(E, N, -1), dtype=torch.float32, device=device),
                    actions=actions_en,
                    rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
                    dones=torch.tensor(dones, dtype=torch.float32, device=device),
                    log_probs=log_probs_en,
                    values=values_en,
                )

                obs = next_obs
                total_steps += E * N

            # Compute last values for GAE bootstrapping
            obs_flat = obs.reshape(E * N, -1)
            obs_norm = obs_normalizer.normalize(obs_flat)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device)
            last_values = critic(obs_t).reshape(E, N)

        buffer.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)

        # PPO update
        agent.update(buffer)
        num_rollouts += 1

        # Logging
        if num_rollouts % 5 == 0:
            elapsed = time.time() - t_start
            sps = total_steps / elapsed
            print(f"  Step {total_steps:>8d} | SPS: {sps:.0f}")

        # Eval
        if total_steps % cfg.eval_freq < E * N * cfg.rollout_steps:
            eval_result = run_eval(
                cfg, actor, obs_normalizer, device,
                args.gate_threshold, args.max_rl_authority,
                args.min_rl_authority, args.base_R_desired,
                spawn_mode=args.spawn_mode,
                spread_steps=args.spread_steps,
            )
            print(f"  [EVAL @ {total_steps}] "
                  f"reward={eval_result['reward']:.2f}  "
                  f"tr(P)={eval_result['tr_P']:.2f}  "
                  f"RMSE={eval_result['rmse']:.2f}")
            append_eval_csv(args.save_path, total_steps, eval_result)

            # Save best
            if eval_result["tr_P"] < best_tr_P:
                best_tr_P = eval_result["tr_P"]
                save_checkpoint(
                    os.path.join(args.save_path, "checkpoint_best.pt"),
                    actor, critic, agent, obs_normalizer, total_steps, cfg,
                    args,
                )
                print(f"    -> New best tr(P)={best_tr_P:.2f}, saved checkpoint_best.pt")

        # Periodic save
        if total_steps % cfg.save_freq < E * N * cfg.rollout_steps:
            save_checkpoint(
                os.path.join(args.save_path, f"checkpoint_step_{total_steps}.pt"),
                actor, critic, agent, obs_normalizer, total_steps, cfg,
                args,
            )

    # Final save
    save_checkpoint(
        os.path.join(args.save_path, "checkpoint_final.pt"),
        actor, critic, agent, obs_normalizer, total_steps, cfg,
        args,
    )
    vec_env.close()
    print(f"\nTraining complete. {total_steps} steps, best tr(P)={best_tr_P:.2f}")


def save_checkpoint(path, actor, critic, agent, obs_normalizer, step, cfg, args):
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "obs_normalizer": {
            "mean": obs_normalizer.mean,
            "var": obs_normalizer.var,
            "count": obs_normalizer.count,
        },
        "step": step,
        "full_config": {
            k: v for k, v in cfg.__dict__.items()
            if not k.startswith("_") and not callable(v)
        },
        "gating": {
            "gate_threshold": args.gate_threshold,
            "max_rl_authority": args.max_rl_authority,
            "min_rl_authority": args.min_rl_authority,
            "base_R_desired": args.base_R_desired,
        },
    }, path)


if __name__ == "__main__":
    main()
