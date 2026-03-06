"""V2 Evaluation — compare RL policy vs baseline on multiple scenarios."""

import argparse
import numpy as np
import torch
import time

from .config import V2TrackingConfig
from .env import V2TrackingEnv
from .networks import BetaActor, CentralizedCritic
from src.rl.ppo.tracking_trainer import RunningMeanStd


def run_scenario(
    name: str,
    cfg: V2TrackingConfig,
    actor: BetaActor | None,
    device: torch.device,
    obs_normalizer: RunningMeanStd | None,
    spawn_mode: str,
    num_episodes: int = 10,
    max_steps: int = 10000,
) -> dict:
    """Run eval episodes for one scenario.

    If actor is None, runs baseline (zero RL correction).
    """
    N = cfg.num_drones
    all_tr_P, all_rmse = [], []

    for ep in range(num_episodes):
        env = V2TrackingEnv(cfg, seed=20000 + ep)
        env.spawn_mode = spawn_mode
        obs, _ = env.reset()

        ep_tr_P, ep_rmse = [], []

        for step in range(max_steps):
            if actor is not None:
                if obs_normalizer is not None:
                    obs_norm = np.clip(obs_normalizer.normalize(obs), -10.0, 10.0)
                else:
                    obs_norm = obs
                actor_obs = torch.tensor(
                    obs_norm[:, :cfg.actor_obs_dim].reshape(N, -1),
                    dtype=torch.float32, device=device,
                )
                with torch.no_grad():
                    action = actor.deterministic_action(actor_obs).cpu().numpy()
            else:
                # Baseline: zero residual correction
                action = np.zeros((N, 3), dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if "tr_P_pos" in info:
                ep_tr_P.append(info["tr_P_pos"])
            if info.get("filter_initialized", False) and "result" in info:
                est = env._filter.get_estimate()
                true_pos = info["result"]["target_true_pos"]
                ep_rmse.append(np.linalg.norm(est[:3] - true_pos))

            if terminated or truncated:
                break

        env.close()
        mean_trP = np.mean(ep_tr_P) if ep_tr_P else np.nan
        mean_rmse = np.mean(ep_rmse) if ep_rmse else np.nan
        all_tr_P.append(mean_trP)
        all_rmse.append(mean_rmse)
        print(f"  {name} ep {ep+1}/{num_episodes}: "
              f"tr(P)={mean_trP:.1f}  RMSE={mean_rmse:.2f}  steps={len(ep_tr_P)}")

    return {
        "name": name,
        "tr_P_mean": np.nanmean(all_tr_P),
        "tr_P_std": np.nanstd(all_tr_P),
        "rmse_mean": np.nanmean(all_rmse),
        "rmse_std": np.nanstd(all_rmse),
    }


def main():
    parser = argparse.ArgumentParser(description="V2 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--cluster-only", action="store_true",
                        help="Only eval on cluster spawn")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("full_config", {})

    cfg = V2TrackingConfig()
    for k, v in ckpt_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Build networks
    actor = BetaActor(device, obs_dim=cfg.actor_obs_dim, action_dim=3,
                      hidden_sizes=cfg.actor_hidden)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_normalizer = None
    if "obs_normalizer" in ckpt:
        obs_normalizer = RunningMeanStd(shape=(cfg.critic_obs_dim,))
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")
    print(f"Config: spawn={cfg.spawn_mode}, residual={cfg.residual_scale}, "
          f"actor_obs={cfg.actor_obs_dim}, critic_obs={cfg.critic_obs_dim}")
    print()

    # Run scenarios
    scenarios = []

    # 1. Baseline, cluster
    print("=" * 60)
    print("Scenario 1: Baseline chase+offset, cluster spawn")
    print("=" * 60)
    r = run_scenario("Baseline (cluster)", cfg, None, device, None,
                     "cluster", args.episodes, args.steps)
    scenarios.append(r)

    # 2. RL, cluster
    print()
    print("=" * 60)
    print("Scenario 2: V2 RL policy, cluster spawn")
    print("=" * 60)
    r = run_scenario("V2 RL (cluster)", cfg, actor, device, obs_normalizer,
                     "cluster", args.episodes, args.steps)
    scenarios.append(r)

    if not args.cluster_only:
        # 3. Baseline, normal
        print()
        print("=" * 60)
        print("Scenario 3: Baseline chase+offset, normal spawn")
        print("=" * 60)
        r = run_scenario("Baseline (normal)", cfg, None, device, None,
                         "normal", args.episodes, args.steps)
        scenarios.append(r)

        # 4. RL, normal
        print()
        print("=" * 60)
        print("Scenario 4: V2 RL policy, normal spawn")
        print("=" * 60)
        r = run_scenario("V2 RL (normal)", cfg, actor, device, obs_normalizer,
                         "normal", args.episodes, args.steps)
        scenarios.append(r)

    # Summary table
    print()
    print("=" * 70)
    print(f"{'Scenario':<30} {'tr(P)':>15} {'RMSE (m)':>15}")
    print("-" * 70)
    for s in scenarios:
        trP_str = f"{s['tr_P_mean']:.1f} ± {s['tr_P_std']:.1f}"
        rmse_str = f"{s['rmse_mean']:.2f} ± {s['rmse_std']:.2f}"
        print(f"{s['name']:<30} {trP_str:>15} {rmse_str:>15}")
    print("=" * 70)


if __name__ == "__main__":
    main()
