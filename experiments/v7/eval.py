"""V7 Evaluation — compare heuristic spread vs RL spread, both with chase+offset tracking."""

import argparse
import numpy as np
import torch

from experiments.v2.networks import BetaActor
from src.rl.ppo.tracking_trainer import RunningMeanStd

from .config import V7Config
from .spread_env import SpreadEnv
from .track_env import TrackEnv
from .baselines import (
    run_baseline_no_spread,
    run_heuristic_spread_chase,
    run_rl_spread_chase,
)


def run_scenario(name, cfg, num_episodes, scenario_fn, **kwargs):
    all_tr_P = []

    for ep in range(num_episodes):
        seed = 20000 + ep
        result = scenario_fn(cfg=cfg, seed=seed, **kwargs)

        tr_P_history = result["tr_P_history"]
        mean_trP = np.mean(tr_P_history) if tr_P_history else np.nan
        all_tr_P.append(mean_trP)
        print(f"  {name} ep {ep+1}/{num_episodes}: tr(P)={mean_trP:.1f}")

    return {
        "name": name,
        "tr_P_mean": np.nanmean(all_tr_P),
        "tr_P_std": np.nanstd(all_tr_P),
        "tr_P_all": all_tr_P,
    }


def _run_baseline(cfg, seed, **kwargs):
    track_env = TrackEnv(cfg, seed=seed)
    track_env.spawn_mode = "cluster"
    result = run_baseline_no_spread(track_env, cfg, total_steps=5000)
    track_env.close()
    return result


def _run_heuristic(cfg, seed, **kwargs):
    spread_env = SpreadEnv(cfg, seed=seed)
    track_env = TrackEnv(cfg, seed=seed)
    spread_env.reset(seed=seed)
    result = run_heuristic_spread_chase(spread_env, track_env, cfg,
                                         spread_steps=400, track_steps=4600)
    spread_env.close()
    track_env.close()
    return result


def _run_rl(cfg, seed, spread_actor=None, spread_obs_normalizer=None,
            device=None, **kwargs):
    spread_env = SpreadEnv(cfg, seed=seed)
    track_env = TrackEnv(cfg, seed=seed)
    spread_env.reset(seed=seed)
    result = run_rl_spread_chase(
        spread_actor, spread_env, track_env, cfg,
        spread_obs_normalizer=spread_obs_normalizer,
        device=device, spread_steps=400, track_steps=4600,
    )
    spread_env.close()
    track_env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="V7 Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = V7Config()
    scenarios = []

    # Scenario 1: Baseline (cluster, no spread)
    print("=" * 60)
    print("Scenario 1: Baseline (cluster spawn, chase+offset only)")
    print("=" * 60)
    r = run_scenario("Baseline (no spread)", cfg, args.episodes, _run_baseline)
    scenarios.append(r)

    # Scenario 2: Heuristic spread + chase
    print()
    print("=" * 60)
    print("Scenario 2: Heuristic spread (400 steps) → chase+offset")
    print("=" * 60)
    r = run_scenario("Heuristic spread→chase", cfg, args.episodes, _run_heuristic)
    scenarios.append(r)

    if not args.baseline_only and args.checkpoint:
        # Scenario 3: RL spread + chase
        print()
        print("=" * 60)
        print("Scenario 3: V7 RL spread (400 steps) → chase+offset")
        print("=" * 60)

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        spread_actor = BetaActor(device, obs_dim=cfg.spread_obs_dim, action_dim=3,
                                 hidden_sizes=cfg.actor_hidden)
        spread_actor.load_state_dict(ckpt["actor"])
        spread_actor.eval()

        spread_obs_normalizer = None
        if "obs_normalizer" in ckpt:
            spread_obs_normalizer = RunningMeanStd(shape=(cfg.spread_critic_obs_dim,))
            spread_obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

        r = run_scenario(
            "V7 RL spread→chase", cfg, args.episodes, _run_rl,
            spread_actor=spread_actor,
            spread_obs_normalizer=spread_obs_normalizer,
            device=device,
        )
        scenarios.append(r)
    elif not args.baseline_only:
        print("\nSkipping V7 RL scenario (no --checkpoint provided)")

    # Summary
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
