"""V6 Evaluation — 4 scenarios comparison on cluster spawn.

| # | Spread | Track | Label |
|---|--------|-------|-------|
| 1 | None (cluster) | Plain chase+offset | "Baseline" |
| 2 | Repulsion (400 steps) | Plain chase+offset | "V4 Heuristic" |
| 3 | Repulsion (400 steps) | Angular-repulsion chase (no RL) | "V6 Heuristic" |
| 4 | Repulsion (400 steps) | Angular-repulsion chase + RL | "V6 RL" |
"""

import argparse
import numpy as np
import torch
import time

from experiments.v2.networks import BetaActor
from src.rl.ppo.tracking_trainer import RunningMeanStd

from .config import V6Config
from .spread_env import SpreadEnv
from .track_env import TrackEnv
from .baselines import (
    run_baseline_chase_only,
    run_baseline_no_spread,
    run_heuristic_twophase,
    run_rl_twophase,
)


def run_scenario(name, cfg, num_episodes, spread_steps, track_steps,
                 scenario_fn, **kwargs):
    """Run eval episodes for one scenario, collecting tr(P)."""
    all_tr_P = []
    N = cfg.num_drones

    for ep in range(num_episodes):
        seed = 20000 + ep

        spread_env = SpreadEnv(cfg, seed=seed)
        track_env = TrackEnv(cfg, seed=seed)

        result = scenario_fn(
            spread_env=spread_env, track_env=track_env, cfg=cfg,
            spread_steps=spread_steps, track_steps=track_steps,
            seed=seed, **kwargs,
        )

        tr_P_history = result["tr_P_history"]
        mean_trP = np.mean(tr_P_history) if tr_P_history else np.nan
        all_tr_P.append(mean_trP)

        spread_env.close()
        track_env.close()

        print(f"  {name} ep {ep+1}/{num_episodes}: tr(P)={mean_trP:.1f}")

    return {
        "name": name,
        "tr_P_mean": np.nanmean(all_tr_P),
        "tr_P_std": np.nanstd(all_tr_P),
        "tr_P_all": all_tr_P,
    }


def _run_scenario1(spread_env, track_env, cfg, spread_steps, track_steps, seed, **kwargs):
    """Scenario 1: Baseline — cluster spawn + plain chase+offset."""
    track_env.spawn_mode = "cluster"
    # Temporarily zero angular_weight for plain chase
    saved = cfg.angular_weight
    cfg.angular_weight = 0.0
    result = run_baseline_chase_only(track_env, cfg, total_steps=spread_steps + track_steps)
    cfg.angular_weight = saved
    return result


def _run_scenario2(spread_env, track_env, cfg, spread_steps, track_steps, seed, **kwargs):
    """Scenario 2: V4 Heuristic — repulsion spread + plain chase+offset."""
    spread_env.reset(seed=seed)
    return run_baseline_no_spread(spread_env, track_env, cfg, spread_steps, track_steps)


def _run_scenario3(spread_env, track_env, cfg, spread_steps, track_steps, seed, **kwargs):
    """Scenario 3: V6 Heuristic — repulsion spread + angular-repulsion tracking."""
    spread_env.reset(seed=seed)
    return run_heuristic_twophase(spread_env, track_env, cfg, spread_steps, track_steps)


def _run_scenario4(spread_env, track_env, cfg, spread_steps, track_steps, seed,
                   track_actor=None, track_obs_normalizer=None, device=None, **kwargs):
    """Scenario 4: V6 RL — repulsion spread + angular-repulsion tracking + RL."""
    spread_env.reset(seed=seed)
    return run_rl_twophase(
        track_actor, spread_env, track_env, cfg,
        track_obs_normalizer=track_obs_normalizer,
        device=device, spread_steps=spread_steps, track_steps=track_steps,
    )


def main():
    parser = argparse.ArgumentParser(description="V6 Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Track checkpoint for V6 RL scenario")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000,
                        help="Total steps (spread + track)")
    parser.add_argument("--spread-steps", type=int, default=400)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = V6Config()
    spread_steps = args.spread_steps
    track_steps = args.steps - spread_steps

    scenarios = []

    # ── Scenario 1: Baseline (no spread, no angular repulsion) ──
    print("=" * 60)
    print("Scenario 1: Baseline chase+offset, cluster spawn (no spread)")
    print("=" * 60)
    r = run_scenario("Baseline (no spread)", cfg, args.episodes,
                     spread_steps, track_steps, _run_scenario1)
    scenarios.append(r)

    # ── Scenario 2: V4 Heuristic (spread + plain chase) ──
    print()
    print("=" * 60)
    print("Scenario 2: Repulsion spread → plain chase+offset")
    print("=" * 60)
    r = run_scenario("V4 Heuristic (spread→chase)", cfg, args.episodes,
                     spread_steps, track_steps, _run_scenario2)
    scenarios.append(r)

    # ── Scenario 3: V6 Heuristic (spread + angular tracking) ──
    print()
    print("=" * 60)
    print("Scenario 3: Repulsion spread → angular-repulsion tracking")
    print("=" * 60)
    r = run_scenario("V6 Heuristic (spread→angular)", cfg, args.episodes,
                     spread_steps, track_steps, _run_scenario3)
    scenarios.append(r)

    if not args.baseline_only:
        # ── Scenario 4: V6 RL ──
        if args.checkpoint:
            print()
            print("=" * 60)
            print("Scenario 4: Repulsion spread → angular tracking + RL residual")
            print("=" * 60)

            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            track_actor = BetaActor(device, obs_dim=cfg.track_actor_obs_dim, action_dim=3,
                                    hidden_sizes=cfg.actor_hidden)
            track_actor.load_state_dict(ckpt["actor"])
            track_actor.eval()

            track_obs_normalizer = None
            if "obs_normalizer" in ckpt:
                track_obs_normalizer = RunningMeanStd(shape=(cfg.track_critic_obs_dim,))
                track_obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

            r = run_scenario(
                "V6 RL (spread→angular+RL)", cfg, args.episodes,
                spread_steps, track_steps, _run_scenario4,
                track_actor=track_actor,
                track_obs_normalizer=track_obs_normalizer,
                device=device,
            )
            scenarios.append(r)
        else:
            print("\nSkipping V6 RL scenario (no --checkpoint provided)")

    # ── Summary table ──
    print()
    print("=" * 70)
    print(f"{'Scenario':<40} {'tr(P)':>20}")
    print("-" * 70)
    for s in scenarios:
        trP_str = f"{s['tr_P_mean']:.1f} +/- {s['tr_P_std']:.1f}"
        print(f"{s['name']:<40} {trP_str:>20}")
    print("=" * 70)


if __name__ == "__main__":
    main()
