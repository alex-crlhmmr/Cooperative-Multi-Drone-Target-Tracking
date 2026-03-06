"""V2 Replay — run single episode and animate."""

import argparse
import numpy as np
import torch

from .config import V2TrackingConfig
from .env import V2TrackingEnv
from .networks import BetaActor
from src.rl.ppo.tracking_trainer import RunningMeanStd
from src.viz.animation import animate_tracking


def main():
    parser = argparse.ArgumentParser(description="V2 Replay Animation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cluster", action="store_true", help="Use cluster spawn")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no RL)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None, help="Save animation to file")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("full_config", {})
    cfg = V2TrackingConfig()
    for k, v in ckpt_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.episode_length = args.steps

    # Build actor
    actor = None
    obs_normalizer = None
    if not args.baseline:
        actor = BetaActor(device, obs_dim=cfg.actor_obs_dim, action_dim=3,
                          hidden_sizes=cfg.actor_hidden)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        if "obs_normalizer" in ckpt:
            obs_normalizer = RunningMeanStd(shape=(cfg.critic_obs_dim,))
            obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

    # Run episode
    env = V2TrackingEnv(cfg, seed=args.seed)
    env.spawn_mode = "cluster" if args.cluster else "normal"
    obs, _ = env.reset()

    N = cfg.num_drones
    drone_trajectories = [[] for _ in range(N)]
    target_trajectory = []
    measurements_list = []

    for step in range(args.steps):
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
            action = np.zeros((N, 3), dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if "result" in info:
            result = info["result"]
            positions = result["drone_positions"]
            for i in range(N):
                drone_trajectories[i].append(positions[i].copy())
            target_trajectory.append(result["target_true_pos"].copy())
            measurements_list.append(result["measurements"])

        if terminated or truncated:
            break

        if step % 1000 == 0:
            tr_P = info.get("tr_P_pos", np.nan)
            print(f"Step {step}: tr(P)={tr_P:.1f}")

    env.close()

    # Convert to arrays
    drone_trajectories = [np.array(t) for t in drone_trajectories]
    target_trajectory = np.array(target_trajectory)

    mode = "baseline" if args.baseline else "V2 RL"
    spawn = "cluster" if args.cluster else "normal"
    title = f"{mode} ({spawn} spawn) — {len(target_trajectory)} steps"

    print(f"\nAnimating {title}...")
    animate_tracking(
        drone_trajectories=drone_trajectories,
        target_trajectory=target_trajectory,
        measurements=measurements_list,
        dt=cfg.dt,
        title=title,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
