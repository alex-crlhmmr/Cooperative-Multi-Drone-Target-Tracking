"""V5 Evaluation — 3 scenarios, all with chase+offset tracking."""

import argparse
import numpy as np
import torch

from experiments.v2.networks import BetaActor
from src.rl.ppo.tracking_trainer import RunningMeanStd

from .config import V5Config
from .spread_env import SpreadEnv
from .track_env import TrackEnv
from .baselines import (
    run_baseline_no_spread,
    run_heuristic_twophase,
    run_rl_spread_heuristic_track,
    repulsion_controller,
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


def _run_baseline(spread_env, track_env, cfg, spread_steps, track_steps, seed, **kwargs):
    """Baseline: chase+offset from cluster spawn (no spread)."""
    track_env.spawn_mode = "cluster"
    return run_baseline_no_spread(track_env, cfg, total_steps=spread_steps + track_steps)


def _run_heuristic(spread_env, track_env, cfg, spread_steps, track_steps, seed, **kwargs):
    """Heuristic: repulsion spread -> chase+offset."""
    spread_env.reset(seed=seed)
    return run_heuristic_twophase(spread_env, track_env, cfg, spread_steps, track_steps)


def _run_rl_spread(spread_env, track_env, cfg, spread_steps, track_steps, seed,
                   spread_actor=None, spread_obs_normalizer=None,
                   device=None, **kwargs):
    """V5 RL spread -> heuristic chase+offset."""
    spread_env.reset(seed=seed)
    return run_rl_spread_heuristic_track(
        spread_actor, spread_env, track_env, cfg,
        spread_obs_normalizer=spread_obs_normalizer,
        device=device, spread_steps=spread_steps, track_steps=track_steps,
    )


def main():
    parser = argparse.ArgumentParser(description="V5 Evaluation")
    parser.add_argument("--spread-checkpoint", type=str, default=None,
                        help="V5 spread checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000,
                        help="Total steps (spread + track)")
    parser.add_argument("--spread-steps", type=int, default=400)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = V5Config()
    spread_steps = args.spread_steps
    track_steps = args.steps - spread_steps

    scenarios = []

    # ── Scenario 1: Baseline (no spread, cluster spawn) ──
    print("=" * 60)
    print("Scenario 1: Baseline chase+offset, cluster spawn (no spread)")
    print("=" * 60)
    r = run_scenario("Baseline (no spread)", cfg, args.episodes,
                     spread_steps, track_steps, _run_baseline)
    scenarios.append(r)

    # ── Scenario 2: Repulsion heuristic spread -> chase+offset ──
    print()
    print("=" * 60)
    print(f"Scenario 2: Repulsion heuristic ({spread_steps} steps) -> chase+offset")
    print("=" * 60)
    r = run_scenario("Heuristic (repulsion->chase)", cfg, args.episodes,
                     spread_steps, track_steps, _run_heuristic)
    scenarios.append(r)

    if not args.baseline_only:
        # ── Scenario 3: V5 RL spread -> chase+offset ──
        if args.spread_checkpoint:
            print()
            print("=" * 60)
            print(f"Scenario 3: V5 RL spread ({spread_steps} steps) -> chase+offset")
            print("=" * 60)

            spread_ckpt = torch.load(args.spread_checkpoint, map_location=device,
                                     weights_only=False)
            spread_actor = BetaActor(device, obs_dim=cfg.spread_obs_dim, action_dim=3,
                                     hidden_sizes=cfg.actor_hidden)
            spread_actor.load_state_dict(spread_ckpt["actor"])
            spread_actor.eval()

            spread_obs_normalizer = None
            if "obs_normalizer" in spread_ckpt:
                spread_obs_normalizer = RunningMeanStd(shape=(cfg.spread_critic_obs_dim,))
                spread_obs_normalizer.load_state_dict(spread_ckpt["obs_normalizer"])

            r = run_scenario(
                "V5 RL spread->chase", cfg, args.episodes,
                spread_steps, track_steps, _run_rl_spread,
                spread_actor=spread_actor,
                spread_obs_normalizer=spread_obs_normalizer,
                device=device,
            )
            scenarios.append(r)
        else:
            print("\nSkipping V5 RL scenario (no --spread-checkpoint provided)")

    # ── Summary table ──
    print()
    print("=" * 70)
    print(f"{'Scenario':<35} {'tr(P)':>20}")
    print("-" * 70)
    for s in scenarios:
        trP_str = f"{s['tr_P_mean']:.1f} +/- {s['tr_P_std']:.1f}"
        print(f"{s['name']:<35} {trP_str:>20}")
    print("=" * 70)


if __name__ == "__main__":
    main()
