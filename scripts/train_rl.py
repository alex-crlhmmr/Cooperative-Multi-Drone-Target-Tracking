#!/usr/bin/env python3
"""Train PPO agent for multi-drone active tracking.

Usage:
    python -m scripts.train_rl
    python -m scripts.train_rl --num-envs 2 --steps 5000  # quick test
    python -m scripts.train_rl --num-envs 16 --steps 5000000 --traj multi_segment
"""

import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.rl.ppo import (
    TrackingConfig, MultiDroneTrackingEnv, TrackingActor, TrackingCritic,
    MultiAgentRolloutBuffer, TrackingTrainer, RunningMeanStd, PPOAgent, SubprocVecEnv,
)


def make_env_fn(cfg: TrackingConfig, seed: int):
    """Create a factory function for SubprocVecEnv workers."""
    def _make():
        return MultiDroneTrackingEnv(cfg, seed=seed)
    return _make


def plot_training(history: dict, save_dir: str):
    """Generate end-of-training plots and save to disk."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 1: Episode metrics ---
    # Reward
    ax = axes[0, 0]
    ep_r = history["episode_rewards"]
    if len(ep_r) > 0:
        ax.plot(ep_r, alpha=0.3, color="C0", linewidth=0.5)
        window = min(50, max(1, len(ep_r) // 10))
        if len(ep_r) >= window:
            smoothed = np.convolve(ep_r, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(ep_r)), smoothed, color="C0", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Training Reward")
        ax.grid(True, alpha=0.3)

    # tr(P)
    ax = axes[0, 1]
    ep_trP = history["episode_tr_Ps"]
    if len(ep_trP) > 0:
        ax.plot(ep_trP, alpha=0.3, color="C1", linewidth=0.5)
        window = min(50, max(1, len(ep_trP) // 10))
        if len(ep_trP) >= window:
            smoothed = np.convolve(ep_trP, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(ep_trP)), smoothed, color="C1", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean tr(P_pos)")
        ax.set_title("Filter Uncertainty")
        ax.grid(True, alpha=0.3)

    # Eval curves
    ax = axes[0, 2]
    eval_steps = history["eval_steps"]
    if len(eval_steps) > 0:
        ax.plot(eval_steps, history["eval_rewards"], "o-", color="C2", label="Eval Reward", markersize=3)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Eval Reward")
        ax.set_title("Evaluation (Deterministic)")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(eval_steps, history["eval_tr_Ps"], "s--", color="C3", label="Eval tr(P)", markersize=3)
        ax2.set_ylabel("Eval tr(P)", color="C3")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    # --- Row 2: Loss components ---
    # Policy loss
    ax = axes[1, 0]
    pl = history["policy_losses"]
    if len(pl) > 0:
        ax.plot(pl, color="C4")
        ax.set_xlabel("Rollout")
        ax.set_ylabel("Policy Loss")
        ax.set_title("Policy Loss")
        ax.grid(True, alpha=0.3)

    # Value loss
    ax = axes[1, 1]
    vl = history["value_losses"]
    if len(vl) > 0:
        ax.plot(vl, color="C5")
        ax.set_xlabel("Rollout")
        ax.set_ylabel("Value Loss")
        ax.set_title("Value Loss")
        ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 2]
    ent = history["entropy_losses"]
    if len(ent) > 0:
        ax.plot(ent, color="C6")
        ax.set_xlabel("Rollout")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy")
        ax.grid(True, alpha=0.3)

    fig.suptitle("PPO Active Tracking — Training Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved training plots: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train PPO for multi-drone tracking")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel envs")
    parser.add_argument("--steps", type=int, default=None, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traj", type=str, default=None, help="Target trajectory type")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-norm", action="store_true", help="Disable obs normalization")
    parser.add_argument("--tag", type=str, default="", help="Run tag for TensorBoard")
    parser.add_argument("--num-drones", type=int, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Skip end-of-training plots")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint .pt")
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--spawn-mode", type=str, default=None, choices=["normal", "mixed", "cluster"])
    parser.add_argument("--w-cov", type=float, default=None)
    parser.add_argument("--w-dist", type=float, default=None)
    parser.add_argument("--w-sep", type=float, default=None)
    parser.add_argument("--w-detect", type=float, default=None)
    args = parser.parse_args()

    # Build config — from checkpoint if resuming, else fresh defaults
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "full_config" in ckpt:
            cfg = TrackingConfig(**ckpt["full_config"])
        else:
            cfg = TrackingConfig(seed=args.seed)
        resume_step = ckpt.get("step", 0)
        print(f"Resuming from {args.resume} (step {resume_step})")
    else:
        cfg = TrackingConfig(seed=args.seed)
        ckpt = None
        resume_step = 0
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.steps is not None:
        cfg.max_training_steps = args.steps
    if args.traj is not None:
        cfg.target_trajectory = args.traj
    if args.lr is not None:
        cfg.lr = args.lr
    if args.no_norm:
        cfg.normalize_obs = False
    if args.num_drones is not None:
        cfg.num_drones = args.num_drones
    if args.save_path is not None:
        cfg.save_path = args.save_path
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

    # Seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create vectorized environment
    env_fns = [make_env_fn(cfg, seed=cfg.seed + i) for i in range(cfg.num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    N = cfg.num_drones
    obs_dim = 31  # matches MultiDroneTrackingEnv
    action_dim = 3

    # Create networks
    actor = TrackingActor(
        device=device,
        obs_dim=obs_dim,
        action_dim=action_dim,
        ego_hidden=cfg.ego_hidden,
        neighbor_hidden=cfg.neighbor_hidden,
        neighbor_embed_dim=cfg.neighbor_embed_dim,
        policy_hidden=cfg.policy_hidden,
    ).to(device)

    critic = TrackingCritic(
        device=device,
        obs_dim=obs_dim,
        ego_hidden=cfg.ego_hidden,
        neighbor_hidden=cfg.neighbor_hidden,
        neighbor_embed_dim=cfg.neighbor_embed_dim,
        value_hidden=cfg.policy_hidden,
    ).to(device)

    # PPO agent
    agent = PPOAgent(
        actor=actor,
        critic=critic,
        lr=cfg.lr,
        clip_epsilon=cfg.clip_epsilon,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        num_epochs=cfg.num_epochs,
        mini_batch_size=cfg.mini_batch_size,
        max_grad_norm=cfg.max_grad_norm,
    )

    # Rollout buffer
    buffer = MultiAgentRolloutBuffer(
        rollout_steps=cfg.rollout_steps,
        num_envs=cfg.num_envs,
        num_drones=N,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    # Observation normalizer
    obs_normalizer = RunningMeanStd(shape=(obs_dim,)) if cfg.normalize_obs else None

    # Resume from checkpoint if provided
    if ckpt is not None:
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        if obs_normalizer is not None and "obs_normalizer" in ckpt:
            obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        if "history" in ckpt:
            trainer_history = ckpt["history"]
        else:
            trainer_history = None
        print(f"  Loaded weights, optimizer, normalizer from step {resume_step}")

    # TensorBoard
    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"rl_tracking_{N}d_{cfg.num_envs}e{tag}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Trainer
    spawn_mode = args.spawn_mode or "mixed"
    trainer = TrackingTrainer(
        vec_env=vec_env,
        agent=agent,
        buffer=buffer,
        cfg=cfg,
        device=device,
        writer=writer,
        obs_normalizer=obs_normalizer,
        start_step=resume_step,
        spawn_mode=spawn_mode,
    )

    # Restore training history if resuming
    if ckpt is not None and "history" in ckpt:
        trainer.history = ckpt["history"]

    # Print full config
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    transitions_per_rollout = cfg.rollout_steps * cfg.num_envs * N
    total_rollouts = cfg.max_training_steps // (cfg.rollout_steps * cfg.num_envs * N)

    print(f"\n{'='*60}")
    print(f"  PPO Active Tracking Training")
    print(f"{'='*60}")
    print(f"  Device:         {device}")
    print(f"  TensorBoard:    runs/{run_name}")
    print(f"  Checkpoints:    {cfg.save_path}")
    print()
    print(f"  --- Environment ---")
    print(f"  Drones:         {N}")
    print(f"  Parallel envs:  {cfg.num_envs}")
    print(f"  Episode length: {cfg.episode_length} steps")
    print(f"  Trajectory:     {cfg.target_trajectory}")
    print(f"  Target speed:   {cfg.target_speed} m/s")
    print(f"  v_max:          {cfg.v_max} m/s")
    print(f"  Sensor FOV:     {cfg.fov_half_deg} deg half-angle")
    print(f"  Sensor range:   {cfg.max_range} m")
    print()
    print(f"  --- Filter ---")
    print(f"  IMM modes:      sigma_a = {cfg.imm_sigma_a_modes}")
    print(f"  Topology:       {cfg.topology}")
    print(f"  Consensus iters:{cfg.consensus_iters}")
    print()
    print(f"  --- Reward ---")
    print(f"  w_cov:          {cfg.w_cov}")
    print(f"  w_dist:         {cfg.w_dist}")
    print(f"  dist_scale:     {cfg.dist_scale} m")
    print(f"  w_detection:    {cfg.w_detection}")
    print(f"  w_separation:   {cfg.w_separation}")
    print(f"  cov_scale:      {cfg.cov_scale}")
    print()
    print(f"  --- PPO ---")
    print(f"  lr:             {cfg.lr}")
    print(f"  clip_epsilon:   {cfg.clip_epsilon}")
    print(f"  entropy_coef:   {cfg.entropy_coef}")
    print(f"  value_coef:     {cfg.value_coef}")
    print(f"  num_epochs:     {cfg.num_epochs}")
    print(f"  mini_batch:     {cfg.mini_batch_size}")
    print(f"  gamma:          {cfg.gamma}")
    print(f"  gae_lambda:     {cfg.gae_lambda}")
    print(f"  max_grad_norm:  {cfg.max_grad_norm}")
    print()
    print(f"  --- Rollout ---")
    print(f"  Rollout steps:  {cfg.rollout_steps}")
    print(f"  Transitions:    {transitions_per_rollout:,} per rollout")
    print(f"  Max steps:      {cfg.max_training_steps:,}")
    print(f"  Est. rollouts:  ~{total_rollouts}")
    print(f"  Eval every:     {cfg.eval_freq:,} steps")
    print(f"  Save every:     {cfg.save_freq:,} steps")
    print()
    print(f"  --- Networks ---")
    print(f"  Ego hidden:     {cfg.ego_hidden}")
    print(f"  Neighbor hidden:{cfg.neighbor_hidden}")
    print(f"  Neighbor embed: {cfg.neighbor_embed_dim}")
    print(f"  Policy hidden:  {cfg.policy_hidden}")
    print(f"  Actor params:   {actor_params:,}")
    print(f"  Critic params:  {critic_params:,}")
    print(f"  Obs norm:       {cfg.normalize_obs}")
    print(f"{'='*60}\n")

    try:
        history = trainer.train()
    finally:
        vec_env.close()
        writer.close()

    # End-of-training plots
    if not args.no_plot:
        plot_training(history, cfg.save_path)

    ep_r = history["episode_rewards"]
    if ep_r:
        print(f"\nDone. Final mean reward (last 20 ep): {np.mean(ep_r[-20:]):.3f}")
    else:
        print(f"\nDone. No completed episodes (try longer --steps or shorter episode_length).")


if __name__ == "__main__":
    main()
