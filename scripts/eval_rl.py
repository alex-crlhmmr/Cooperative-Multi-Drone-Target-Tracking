#!/usr/bin/env python3
"""Evaluate trained RL policy and compare to baseline.

Loads checkpoint, runs deterministic eval episodes, optionally runs baseline
with same seeds, and generates comparison plots.

Usage:
    python -m scripts.eval_rl --checkpoint output/rl_tracking/checkpoint_final.pt
    python -m scripts.eval_rl --checkpoint output/rl_tracking/checkpoint_final.pt --compare-baseline
    python -m scripts.eval_rl --checkpoint output/rl_tracking/checkpoint_final.pt --episodes 20 --save results/eval
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.rl.ppo import (
    TrackingConfig, MultiDroneTrackingEnv, TrackingActor, TrackingCritic, RunningMeanStd,
)


def load_checkpoint(path: str, device: torch.device):
    """Load checkpoint and reconstruct actor + config."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["config"]

    # Reconstruct full config if saved, else use defaults
    if "full_config" in ckpt:
        cfg = TrackingConfig(**ckpt["full_config"])
    else:
        cfg = TrackingConfig(num_drones=meta["num_drones"])

    obs_dim = meta["obs_dim"]
    action_dim = meta["action_dim"]

    actor = TrackingActor(
        device=device,
        obs_dim=obs_dim,
        action_dim=action_dim,
        ego_hidden=cfg.ego_hidden,
        neighbor_hidden=cfg.neighbor_hidden,
        neighbor_embed_dim=cfg.neighbor_embed_dim,
        policy_hidden=cfg.policy_hidden,
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_normalizer = None
    if "obs_normalizer" in ckpt:
        obs_normalizer = RunningMeanStd(shape=(obs_dim,))
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

    training_step = ckpt.get("step", "?")
    return actor, cfg, obs_normalizer, training_step


def eval_episode(env: MultiDroneTrackingEnv, actor: TrackingActor,
                 obs_normalizer, device: torch.device, cfg: TrackingConfig):
    """Run one eval episode with deterministic policy. Returns per-step metrics."""
    obs, _ = env.reset()
    N = cfg.num_drones

    tr_P_steps = []
    rmse_steps = []
    reward_steps = []
    detection_steps = []

    for step in range(cfg.episode_length):
        if obs_normalizer is not None:
            obs_norm = obs_normalizer.normalize(obs)
        else:
            obs_norm = obs

        obs_t = torch.tensor(obs_norm.reshape(N, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = actor.forward(obs_t)
            action = torch.tanh(dist.mean)

        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

        reward_steps.append(float(reward.mean()))

        if "tr_P_pos" in info:
            tr_P_steps.append(info["tr_P_pos"])
        if info.get("filter_initialized", False) and "result" in info:
            est = env._filter.get_estimate()
            true_pos = info["result"]["target_true_pos"]
            rmse_steps.append(np.linalg.norm(est[:3] - true_pos))
            n_det = sum(1 for m in info["result"]["measurements"] if m is not None)
            detection_steps.append(n_det)

        if terminated or truncated:
            break

    return {
        "reward": sum(reward_steps),
        "tr_P": np.array(tr_P_steps),
        "rmse": np.array(rmse_steps),
        "detections": np.array(detection_steps),
        "reward_steps": np.array(reward_steps),
    }


def plot_comparison(rl_results: list, baseline_results: list | None, save_dir: str, N: int):
    """Generate comparison plots: per-step timeseries + summary bar charts."""
    os.makedirs(save_dir, exist_ok=True)
    has_baseline = baseline_results is not None and len(baseline_results) > 0

    n_rows = 2 if has_baseline else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # make 2D

    # --- RL timeseries (averaged over episodes) ---
    def _mean_timeseries(results, key):
        """Compute mean timeseries, padding shorter episodes."""
        max_len = max(len(r[key]) for r in results)
        padded = np.full((len(results), max_len), np.nan)
        for i, r in enumerate(results):
            padded[i, :len(r[key])] = r[key]
        return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)

    # tr(P) over time
    ax = axes[0, 0]
    mean_trP, std_trP = _mean_timeseries(rl_results, "tr_P")
    steps = np.arange(len(mean_trP))
    ax.plot(steps, mean_trP, color="C0", label="RL")
    ax.fill_between(steps, mean_trP - std_trP, mean_trP + std_trP, alpha=0.2, color="C0")
    if has_baseline:
        bl_mean, bl_std = _mean_timeseries(baseline_results, "tr_P")
        bl_steps = np.arange(len(bl_mean))
        ax.plot(bl_steps, bl_mean, color="C1", linestyle="--", label="Baseline")
        ax.fill_between(bl_steps, bl_mean - bl_std, bl_mean + bl_std, alpha=0.2, color="C1")
    ax.set_xlabel("Step")
    ax.set_ylabel("tr(P_pos)")
    ax.set_title("Filter Uncertainty Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE over time
    ax = axes[0, 1]
    mean_rmse, std_rmse = _mean_timeseries(rl_results, "rmse")
    steps = np.arange(len(mean_rmse))
    ax.plot(steps, mean_rmse, color="C0", label="RL")
    ax.fill_between(steps, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.2, color="C0")
    if has_baseline:
        bl_mean, bl_std = _mean_timeseries(baseline_results, "rmse")
        bl_steps = np.arange(len(bl_mean))
        ax.plot(bl_steps, bl_mean, color="C1", linestyle="--", label="Baseline")
        ax.fill_between(bl_steps, bl_mean - bl_std, bl_mean + bl_std, alpha=0.2, color="C1")
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Position RMSE Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative reward
    ax = axes[0, 2]
    mean_rew, std_rew = _mean_timeseries(rl_results, "reward_steps")
    cum_rew = np.nancumsum(mean_rew)
    ax.plot(np.arange(len(cum_rew)), cum_rew, color="C0", label="RL")
    if has_baseline:
        bl_mean, _ = _mean_timeseries(baseline_results, "reward_steps")
        bl_cum = np.nancumsum(bl_mean)
        ax.plot(np.arange(len(bl_cum)), bl_cum, color="C1", linestyle="--", label="Baseline")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Row 2: Summary bar charts (if comparing) ---
    if has_baseline:
        rl_rewards = [r["reward"] for r in rl_results]
        bl_rewards = [sum(r["reward_steps"]) for r in baseline_results]
        rl_trPs = [np.nanmean(r["tr_P"]) for r in rl_results]
        bl_trPs = [np.nanmean(r["tr_P"]) for r in baseline_results]
        rl_rmses = [np.nanmean(r["rmse"]) for r in rl_results]
        bl_rmses = [np.nanmean(r["rmse"]) for r in baseline_results]

        for ax_i, (rl_vals, bl_vals, ylabel, title) in enumerate([
            (rl_rewards, bl_rewards, "Total Reward", "Episode Reward"),
            (rl_trPs, bl_trPs, "Mean tr(P)", "Mean Filter Uncertainty"),
            (rl_rmses, bl_rmses, "Mean RMSE (m)", "Mean Position Error"),
        ]):
            ax = axes[1, ax_i]
            x = np.arange(2)
            means = [np.nanmean(rl_vals), np.nanmean(bl_vals)]
            stds = [np.nanstd(rl_vals), np.nanstd(bl_vals)]
            bars = ax.bar(x, means, yerr=stds, capsize=5,
                          color=["C0", "C1"], width=0.5, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(["RL Policy", "Baseline"])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="y")
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"RL vs Baseline — {N} Drones, {len(rl_results)} Episodes",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "eval_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved comparison plot: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL tracking policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save", type=str, default=None, help="Directory to save plots")
    parser.add_argument("--steps", type=int, default=None, help="Override episode length for eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    actor, cfg, obs_normalizer, training_step = load_checkpoint(args.checkpoint, device)
    if args.traj:
        cfg.target_trajectory = args.traj
    if args.steps:
        cfg.episode_length = args.steps

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Training step: {training_step}")
    print(f"  Drones: {cfg.num_drones}, Trajectory: {cfg.target_trajectory}")
    print()

    # Run RL evaluation
    rl_results = []
    for ep in range(args.episodes):
        env = MultiDroneTrackingEnv(cfg, seed=args.seed + ep)
        result = eval_episode(env, actor, obs_normalizer, device, cfg)
        env.close()
        rl_results.append(result)

        mean_tr_P = np.nanmean(result["tr_P"]) if len(result["tr_P"]) > 0 else np.nan
        mean_rmse = np.nanmean(result["rmse"]) if len(result["rmse"]) > 0 else np.nan
        print(f"  RL Episode {ep+1}/{args.episodes}: "
              f"reward={result['reward']:.2f}  tr(P)={mean_tr_P:.1f}  RMSE={mean_rmse:.2f}")

    rl_rewards = [r["reward"] for r in rl_results]
    rl_trPs = [np.nanmean(r["tr_P"]) for r in rl_results]
    rl_rmses = [np.nanmean(r["rmse"]) for r in rl_results]

    print(f"\n{'='*55}")
    print(f"  RL Policy ({args.episodes} episodes)")
    print(f"{'='*55}")
    print(f"  Reward:  {np.mean(rl_rewards):>8.3f} +/- {np.std(rl_rewards):.3f}")
    print(f"  tr(P):   {np.nanmean(rl_trPs):>8.1f} +/- {np.nanstd(rl_trPs):.1f}")
    print(f"  RMSE:    {np.nanmean(rl_rmses):>8.2f} +/- {np.nanstd(rl_rmses):.2f}")

    # Baseline comparison
    baseline_results = None
    if args.compare_baseline:
        from scripts.run_baseline import run_baseline_episode

        print(f"\nRunning baseline ({args.episodes} episodes)...")
        baseline_results = []
        rng = np.random.default_rng(args.seed)

        for ep in range(args.episodes):
            bl = run_baseline_episode(cfg, rng)
            # Adapt baseline result format to match RL format
            bl["reward"] = float(np.sum(bl["reward"]))
            bl["reward_steps"] = bl["reward"] if isinstance(bl["reward"], np.ndarray) else np.array([bl["reward"]])
            baseline_results.append(bl)
            mean_tr_P = np.nanmean(bl["tr_P"])
            mean_rmse = np.nanmean(bl["rmse"])
            print(f"  Baseline Episode {ep+1}/{args.episodes}: "
                  f"tr(P)={mean_tr_P:.1f}  RMSE={mean_rmse:.2f}")

        bl_trPs = [np.nanmean(r["tr_P"]) for r in baseline_results]
        bl_rmses = [np.nanmean(r["rmse"]) for r in baseline_results]
        bl_rewards = [r["reward"] for r in baseline_results]

        print(f"\n{'='*55}")
        print(f"  Comparison: RL vs Baseline")
        print(f"{'='*55}")
        print(f"  {'Metric':<10} {'RL':>12} {'Baseline':>12} {'Change':>12}")
        print(f"  {'-'*46}")

        rl_trP = np.nanmean(rl_trPs)
        bl_trP = np.nanmean(bl_trPs)
        pct = (1 - rl_trP / bl_trP) * 100 if bl_trP > 0 else 0
        print(f"  {'tr(P)':<10} {rl_trP:>12.1f} {bl_trP:>12.1f} {pct:>+11.1f}%")

        rl_rmse = np.nanmean(rl_rmses)
        bl_rmse = np.nanmean(bl_rmses)
        pct = (1 - rl_rmse / bl_rmse) * 100 if bl_rmse > 0 else 0
        print(f"  {'RMSE':<10} {rl_rmse:>12.2f} {bl_rmse:>12.2f} {pct:>+11.1f}%")

        rl_rew = np.mean(rl_rewards)
        bl_rew = np.mean(bl_rewards)
        print(f"  {'Reward':<10} {rl_rew:>12.3f} {bl_rew:>12.3f}")

    # Plots
    save_dir = args.save or os.path.dirname(args.checkpoint) or "."
    plot_comparison(rl_results, baseline_results, save_dir, cfg.num_drones)


if __name__ == "__main__":
    main()
