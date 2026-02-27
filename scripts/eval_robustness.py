#!/usr/bin/env python3
"""Robustness evaluation: test trained RL policy under distribution shifts.

Tests how the policy (trained with full topology, 50% dropout, 5 drones,
evasive trajectory, 1000-step episodes) generalizes to:
  - Different topologies (ring, star, line)
  - Different dropout rates (0%, 25%, 75%)
  - Different drone counts (3, 4, 5, 7)
  - Different trajectories (circle, multi_segment, evasive, random_walk)
  - Longer episodes (2000, 3000, 5000 steps)

Each condition is also run with the baseline heuristic for comparison.

Usage:
    python -m scripts.eval_robustness --checkpoint output/rl_tracking/checkpoint_final.pt
    python -m scripts.eval_robustness --checkpoint output/rl_tracking/checkpoint_final.pt --episodes 10 --save results/robustness
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.rl.ppo import (
    TrackingConfig, MultiDroneTrackingEnv, TrackingActor, RunningMeanStd,
)
from scripts.run_baseline import run_baseline_episode


def load_checkpoint(path: str, device: torch.device):
    """Load checkpoint and reconstruct actor + config."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["config"]

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

    return actor, cfg, obs_normalizer


def eval_rl_episode(env, actor, obs_normalizer, device, episode_length):
    """Run one RL episode, return summary metrics."""
    obs, _ = env.reset()
    N = obs.shape[0]
    tr_Ps, rmses = [], []

    for step in range(episode_length):
        obs_norm = obs_normalizer.normalize(obs) if obs_normalizer else obs
        obs_t = torch.tensor(obs_norm.reshape(N, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = actor.forward(obs_t)
            action = torch.tanh(dist.mean)
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

        if "tr_P_pos" in info:
            tr_Ps.append(info["tr_P_pos"])
        if info.get("filter_initialized", False) and "result" in info:
            est = env._filter.get_estimate()
            true_pos = info["result"]["target_true_pos"]
            rmses.append(np.linalg.norm(est[:3] - true_pos))

        if terminated or truncated:
            break

    return {
        "mean_tr_P": np.nanmean(tr_Ps) if tr_Ps else np.nan,
        "mean_rmse": np.nanmean(rmses) if rmses else np.nan,
    }


def eval_condition(name, cfg_overrides, actor, base_cfg, obs_normalizer,
                   device, num_episodes, seed):
    """Evaluate RL + baseline under one condition. Returns dict of results."""
    cfg = TrackingConfig(**{**base_cfg.__dict__, **cfg_overrides})

    rl_trPs, rl_rmses = [], []
    bl_trPs, bl_rmses = [], []

    for ep in range(num_episodes):
        # RL
        env = MultiDroneTrackingEnv(cfg, seed=seed + ep)
        r = eval_rl_episode(env, actor, obs_normalizer, device, cfg.episode_length)
        env.close()
        rl_trPs.append(r["mean_tr_P"])
        rl_rmses.append(r["mean_rmse"])

        # Baseline
        rng = np.random.default_rng(seed + ep)
        bl = run_baseline_episode(cfg, rng)
        bl_trPs.append(np.nanmean(bl["tr_P"]))
        bl_rmses.append(np.nanmean(bl["rmse"]))

    return {
        "name": name,
        "rl_tr_P": np.nanmean(rl_trPs),
        "rl_tr_P_std": np.nanstd(rl_trPs),
        "rl_rmse": np.nanmean(rl_rmses),
        "rl_rmse_std": np.nanstd(rl_rmses),
        "bl_tr_P": np.nanmean(bl_trPs),
        "bl_tr_P_std": np.nanstd(bl_trPs),
        "bl_rmse": np.nanmean(bl_rmses),
        "bl_rmse_std": np.nanstd(bl_rmses),
    }


def plot_robustness(all_results: dict, save_dir: str):
    """Generate grouped bar charts for each sweep dimension."""
    os.makedirs(save_dir, exist_ok=True)

    dims = list(all_results.keys())
    n_dims = len(dims)

    fig, axes = plt.subplots(2, n_dims, figsize=(6 * n_dims, 10))
    if n_dims == 1:
        axes = axes[:, np.newaxis]

    for col, dim_name in enumerate(dims):
        results = all_results[dim_name]
        labels = [r["name"] for r in results]
        x = np.arange(len(labels))
        width = 0.35

        # tr(P) bar chart
        ax = axes[0, col]
        rl_vals = [r["rl_tr_P"] for r in results]
        rl_errs = [r["rl_tr_P_std"] for r in results]
        bl_vals = [r["bl_tr_P"] for r in results]
        bl_errs = [r["bl_tr_P_std"] for r in results]

        ax.bar(x - width/2, rl_vals, width, yerr=rl_errs, capsize=3,
               label="RL", color="C0", alpha=0.8)
        ax.bar(x + width/2, bl_vals, width, yerr=bl_errs, capsize=3,
               label="Baseline", color="C1", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean tr(P)")
        ax.set_title(f"{dim_name} — Filter Uncertainty")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # RMSE bar chart
        ax = axes[1, col]
        rl_vals = [r["rl_rmse"] for r in results]
        rl_errs = [r["rl_rmse_std"] for r in results]
        bl_vals = [r["bl_rmse"] for r in results]
        bl_errs = [r["bl_rmse_std"] for r in results]

        ax.bar(x - width/2, rl_vals, width, yerr=rl_errs, capsize=3,
               label="RL", color="C0", alpha=0.8)
        ax.bar(x + width/2, bl_vals, width, yerr=bl_errs, capsize=3,
               label="Baseline", color="C1", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean RMSE (m)")
        ax.set_title(f"{dim_name} — Position Error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Robustness Evaluation — RL vs Baseline Under Distribution Shifts",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "robustness_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved robustness plot: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation for RL tracking policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per condition")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--save", type=str, default="results/robustness")
    parser.add_argument("--sweeps", type=str, default="topology,dropout,trajectory,horizon",
                        help="Comma-separated list of sweeps to run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, base_cfg, obs_normalizer = load_checkpoint(args.checkpoint, device)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Training config: {base_cfg.num_drones} drones, topology={base_cfg.topology}, "
          f"dropout={base_cfg.dropout_prob}, traj={base_cfg.target_trajectory}")
    print(f"Episodes per condition: {args.episodes}")
    print()

    all_results = {}
    sweep_list = [s.strip() for s in args.sweeps.split(",")]
    t_start = time.time()

    # --- Topology sweep ---
    if "topology" in sweep_list:
        print("=" * 60)
        print("  SWEEP: Topology (trained on 'full')")
        print("=" * 60)
        results = []
        for topo in ["full", "ring", "star", "line"]:
            print(f"  Testing topology={topo}...", end=" ", flush=True)
            r = eval_condition(
                name=topo,
                cfg_overrides={"topology": topo, "target_trajectory": "evasive"},
                actor=actor, base_cfg=base_cfg, obs_normalizer=obs_normalizer,
                device=device, num_episodes=args.episodes, seed=args.seed,
            )
            results.append(r)
            print(f"RL tr(P)={r['rl_tr_P']:.1f}  BL tr(P)={r['bl_tr_P']:.1f}  "
                  f"RL RMSE={r['rl_rmse']:.2f}  BL RMSE={r['bl_rmse']:.2f}")
        all_results["Topology"] = results

    # --- Dropout sweep ---
    if "dropout" in sweep_list:
        print()
        print("=" * 60)
        print("  SWEEP: Dropout (trained with 0.5)")
        print("=" * 60)
        results = []
        for dp in [0.0, 0.25, 0.5, 0.75]:
            print(f"  Testing dropout={dp}...", end=" ", flush=True)
            r = eval_condition(
                name=f"{int(dp*100)}%",
                cfg_overrides={"dropout_prob": dp, "target_trajectory": "evasive"},
                actor=actor, base_cfg=base_cfg, obs_normalizer=obs_normalizer,
                device=device, num_episodes=args.episodes, seed=args.seed,
            )
            results.append(r)
            print(f"RL tr(P)={r['rl_tr_P']:.1f}  BL tr(P)={r['bl_tr_P']:.1f}  "
                  f"RL RMSE={r['rl_rmse']:.2f}  BL RMSE={r['bl_rmse']:.2f}")
        all_results["Dropout"] = results

    # --- Trajectory sweep ---
    if "trajectory" in sweep_list:
        print()
        print("=" * 60)
        print("  SWEEP: Trajectory (trained on 'evasive')")
        print("=" * 60)
        results = []
        for traj in ["circle", "multi_segment", "evasive", "random_walk"]:
            print(f"  Testing trajectory={traj}...", end=" ", flush=True)
            r = eval_condition(
                name=traj,
                cfg_overrides={"target_trajectory": traj},
                actor=actor, base_cfg=base_cfg, obs_normalizer=obs_normalizer,
                device=device, num_episodes=args.episodes, seed=args.seed,
            )
            results.append(r)
            print(f"RL tr(P)={r['rl_tr_P']:.1f}  BL tr(P)={r['bl_tr_P']:.1f}  "
                  f"RL RMSE={r['rl_rmse']:.2f}  BL RMSE={r['bl_rmse']:.2f}")
        all_results["Trajectory"] = results

    # --- Episode length (horizon) sweep ---
    if "horizon" in sweep_list:
        print()
        print("=" * 60)
        print("  SWEEP: Episode Length (trained on 1000)")
        print("=" * 60)
        results = []
        for length in [500, 1000, 2000, 5000]:
            print(f"  Testing episode_length={length}...", end=" ", flush=True)
            r = eval_condition(
                name=f"{length}",
                cfg_overrides={"episode_length": length, "target_trajectory": "evasive"},
                actor=actor, base_cfg=base_cfg, obs_normalizer=obs_normalizer,
                device=device, num_episodes=args.episodes, seed=args.seed,
            )
            results.append(r)
            print(f"RL tr(P)={r['rl_tr_P']:.1f}  BL tr(P)={r['bl_tr_P']:.1f}  "
                  f"RL RMSE={r['rl_rmse']:.2f}  BL RMSE={r['bl_rmse']:.2f}")
        all_results["Horizon"] = results

    # --- Print summary table ---
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  ROBUSTNESS SUMMARY  ({elapsed:.0f}s total)")
    print(f"{'=' * 70}")
    print(f"  {'Condition':<20} {'RL tr(P)':>10} {'BL tr(P)':>10} {'RL RMSE':>10} {'BL RMSE':>10} {'tr(P) Δ':>10}")
    print(f"  {'-'*70}")

    for dim_name, results in all_results.items():
        print(f"  --- {dim_name} ---")
        for r in results:
            pct = (1 - r["rl_tr_P"] / r["bl_tr_P"]) * 100 if r["bl_tr_P"] > 0 else 0
            print(f"  {r['name']:<20} {r['rl_tr_P']:>10.1f} {r['bl_tr_P']:>10.1f} "
                  f"{r['rl_rmse']:>10.2f} {r['bl_rmse']:>10.2f} {pct:>+9.0f}%")

    # --- Plot ---
    plot_robustness(all_results, args.save)

    # --- Save raw data ---
    os.makedirs(args.save, exist_ok=True)
    import json
    with open(os.path.join(args.save, "robustness_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved raw data: {args.save}/robustness_results.json")


if __name__ == "__main__":
    main()
