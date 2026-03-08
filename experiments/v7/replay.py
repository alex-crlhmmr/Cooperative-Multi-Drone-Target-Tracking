"""V7 Replay — animate single episode."""

import argparse
import numpy as np
import torch

from experiments.v2.networks import BetaActor
from src.rl.ppo.tracking_trainer import RunningMeanStd
from src.viz.animation import animate_tracking

from .config import V7Config
from .spread_env import SpreadEnv
from .track_env import TrackEnv
from .baselines import (
    run_baseline_no_spread,
    run_heuristic_spread_chase,
    run_rl_spread_chase,
)


def main():
    parser = argparse.ArgumentParser(description="V7 Replay Animation")
    parser.add_argument("--scenario", type=str, required=True,
                        choices=["baseline", "heuristic", "rl"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--spread-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    cfg = V7Config()
    spread_steps = args.spread_steps
    track_steps = args.steps - spread_steps

    if args.scenario == "baseline":
        print("Running baseline (cluster + chase, no spread)...")
        track_env = TrackEnv(cfg, seed=args.seed)
        track_env.spawn_mode = "cluster"
        result = run_baseline_no_spread(track_env, cfg, total_steps=args.steps)
        track_env.close()
        label = "Baseline (cluster, no spread)"

    elif args.scenario == "heuristic":
        print("Running heuristic (repulsion spread → chase)...")
        spread_env = SpreadEnv(cfg, seed=args.seed)
        track_env = TrackEnv(cfg, seed=args.seed)
        spread_env.reset(seed=args.seed)
        result = run_heuristic_spread_chase(spread_env, track_env, cfg,
                                             spread_steps, track_steps)
        spread_env.close()
        track_env.close()
        label = "Heuristic (spread→chase)"

    elif args.scenario == "rl":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for rl scenario")
            return

        print("Running V7 RL spread → chase...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        spread_actor = BetaActor(device, obs_dim=cfg.spread_obs_dim, action_dim=3,
                                 hidden_sizes=cfg.actor_hidden)
        spread_actor.load_state_dict(ckpt["actor"])
        spread_actor.eval()

        spread_obs_norm = None
        if "obs_normalizer" in ckpt:
            spread_obs_norm = RunningMeanStd(shape=(cfg.spread_critic_obs_dim,))
            spread_obs_norm.load_state_dict(ckpt["obs_normalizer"])

        spread_env = SpreadEnv(cfg, seed=args.seed)
        track_env = TrackEnv(cfg, seed=args.seed)
        spread_env.reset(seed=args.seed)

        result = run_rl_spread_chase(
            spread_actor, spread_env, track_env, cfg,
            spread_obs_normalizer=spread_obs_norm,
            device=device,
            spread_steps=spread_steps, track_steps=track_steps,
        )
        spread_env.close()
        track_env.close()
        label = "V7 RL spread→chase"

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
