"""V6 Replay — run single episode for any scenario and animate."""

import argparse
import numpy as np
import torch

from experiments.v2.networks import BetaActor
from src.rl.ppo.tracking_trainer import RunningMeanStd
from src.viz.animation import animate_tracking

from .config import V6Config
from .spread_env import SpreadEnv
from .track_env import TrackEnv
from .baselines import (
    run_baseline_chase_only,
    run_heuristic_twophase,
    run_rl_twophase,
)


def main():
    parser = argparse.ArgumentParser(description="V6 Replay Animation")
    parser.add_argument("--scenario", type=str, required=True,
                        choices=["baseline", "heuristic", "rl"],
                        help="Which scenario to replay")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Track checkpoint (for rl scenario)")
    parser.add_argument("--steps", type=int, default=5000, help="Total steps")
    parser.add_argument("--spread-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None, help="Save animation to file")
    args = parser.parse_args()

    device = torch.device("cpu")
    cfg = V6Config()
    spread_steps = args.spread_steps
    track_steps = args.steps - spread_steps

    if args.scenario == "baseline":
        print("Running baseline (cluster + chase+offset, no angular repulsion)...")
        track_env = TrackEnv(cfg, seed=args.seed)
        track_env.spawn_mode = "cluster"
        saved = cfg.angular_weight
        cfg.angular_weight = 0.0
        result = run_baseline_chase_only(track_env, cfg, total_steps=args.steps)
        cfg.angular_weight = saved
        track_env.close()
        label = "Baseline (cluster, no spread)"

    elif args.scenario == "heuristic":
        print("Running V6 heuristic (repulsion spread → angular-repulsion tracking)...")
        spread_env = SpreadEnv(cfg, seed=args.seed)
        track_env = TrackEnv(cfg, seed=args.seed)
        spread_env.reset(seed=args.seed)
        result = run_heuristic_twophase(spread_env, track_env, cfg,
                                        spread_steps, track_steps)
        spread_env.close()
        track_env.close()
        label = "V6 Heuristic (spread→angular)"

    elif args.scenario == "rl":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for rl scenario")
            return

        print("Running V6 RL (spread → angular tracking + RL)...")

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        track_actor = BetaActor(device, obs_dim=cfg.track_actor_obs_dim, action_dim=3,
                                hidden_sizes=cfg.actor_hidden)
        track_actor.load_state_dict(ckpt["actor"])
        track_actor.eval()

        track_obs_norm = None
        if "obs_normalizer" in ckpt:
            track_obs_norm = RunningMeanStd(shape=(cfg.track_critic_obs_dim,))
            track_obs_norm.load_state_dict(ckpt["obs_normalizer"])

        spread_env = SpreadEnv(cfg, seed=args.seed)
        track_env = TrackEnv(cfg, seed=args.seed)
        spread_env.reset(seed=args.seed)

        result = run_rl_twophase(
            track_actor, spread_env, track_env, cfg,
            track_obs_normalizer=track_obs_norm,
            device=device,
            spread_steps=spread_steps, track_steps=track_steps,
        )
        spread_env.close()
        track_env.close()
        label = "V6 RL (spread→angular+RL)"

    # Animate
    drone_trajectories = result["drone_trajectories"]
    target_trajectory = result["target_trajectory"]
    measurements = result["measurements"]
    tr_P_history = result.get("tr_P_history", [])

    T = len(target_trajectory)
    title = f"{label} — {T} steps"
    if tr_P_history:
        final_trP = np.mean(tr_P_history[-100:]) if len(tr_P_history) > 100 else np.mean(tr_P_history)
        title += f" — tr(P)={final_trP:.1f}"

    print(f"\nAnimating {title}...")
    print(f"  Drone trajectory lengths: {[len(t) for t in drone_trajectories]}")
    print(f"  Target trajectory length: {T}")
    if tr_P_history:
        print(f"  tr(P) final 100-step mean: {final_trP:.1f}")

    animate_tracking(
        drone_trajectories=drone_trajectories,
        target_trajectory=target_trajectory,
        measurements=measurements,
        dt=cfg.dt,
        title=title,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
