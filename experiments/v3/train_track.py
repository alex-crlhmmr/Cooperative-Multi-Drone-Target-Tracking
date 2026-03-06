"""V3 Phase 2 Training — MAPPO on TrackEnv with Phase 1 warm-start.

Each training episode:
1. Spawn cluster
2. Run Phase 1 policy (or heuristic) for spread_steps
3. Record final drone positions
4. Initialize TrackEnv from those positions
5. Train Phase 2 policy from there
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

from src.rl.ppo.tracking_buffer import MultiAgentRolloutBuffer
from src.rl.ppo.tracking_trainer import RunningMeanStd

from experiments.v2.networks import BetaActor, CentralizedCritic
from .config import V3Config
from .baselines import repulsion_controller


def _safe_normalize(obs, normalizer):
    out = normalizer.normalize(obs)
    return np.clip(out, -10.0, 10.0)


# ── SubprocVecEnv for Phase 2 (with Phase 1 warm-start) ──

class _TrackEnvFactory:
    """Creates TrackEnv that runs Phase 1 on each reset."""
    def __init__(self, cfg, seed, spread_mode, spread_ckpt_path=None):
        self.cfg = cfg
        self.seed = seed
        self.spread_mode = spread_mode  # "rl", "repulsion", or "none"
        self.spread_ckpt_path = spread_ckpt_path

    def __call__(self):
        from experiments.v3.track_env import TrackEnv
        env = _WarmStartTrackEnv(
            self.cfg, seed=self.seed,
            spread_mode=self.spread_mode,
            spread_ckpt_path=self.spread_ckpt_path,
        )
        return env


class _WarmStartTrackEnv:
    """Wrapper that runs Phase 1 (spread) before each Phase 2 reset.

    IMPORTANT: Defers env creation to first reset() to avoid PyBullet
    deadlocks in forked subprocesses.
    """

    def __init__(self, cfg, seed, spread_mode, spread_ckpt_path=None):
        self.cfg = cfg
        self._seed = seed
        self.spread_mode = spread_mode
        self._spread_ckpt_path = spread_ckpt_path

        self._spread_env = None
        self._track_env = None
        self._spread_actor = None
        self._spread_obs_normalizer = None
        self._initialized = False

        # Set gym attributes from config (no env needed)
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(cfg.num_drones, cfg.track_critic_obs_dim), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(cfg.num_drones, 3), dtype=np.float32,
        )
        self.spawn_mode = "cluster"

    def _lazy_init(self):
        """Create envs on first use (inside subprocess)."""
        if self._initialized:
            return
        from experiments.v3.spread_env import SpreadEnv
        from experiments.v3.track_env import TrackEnv

        self._spread_env = SpreadEnv(self.cfg, seed=self._seed)
        self._track_env = TrackEnv(self.cfg, seed=self._seed)

        if self.spread_mode == "rl" and self._spread_ckpt_path:
            device = torch.device("cpu")
            ckpt = torch.load(self._spread_ckpt_path, map_location=device, weights_only=False)
            self._spread_actor = BetaActor(
                device, obs_dim=self.cfg.spread_obs_dim, action_dim=3,
                hidden_sizes=self.cfg.actor_hidden,
            )
            self._spread_actor.load_state_dict(ckpt["actor"])
            self._spread_actor.eval()
            if "obs_normalizer" in ckpt:
                self._spread_obs_normalizer = RunningMeanStd(
                    shape=(self.cfg.spread_critic_obs_dim,)
                )
                self._spread_obs_normalizer.load_state_dict(ckpt["obs_normalizer"])

        self._initialized = True

    def reset(self, seed=None, options=None):
        self._lazy_init()
        # Phase 1: spread
        obs, _ = self._spread_env.reset(seed=seed)
        N = self.cfg.num_drones

        for step in range(self.cfg.spread_steps):
            if self.spread_mode == "rl" and self._spread_actor is not None:
                if self._spread_obs_normalizer is not None:
                    obs_norm = np.clip(
                        self._spread_obs_normalizer.normalize(obs), -10.0, 10.0
                    )
                else:
                    obs_norm = obs
                actor_obs = torch.tensor(
                    obs_norm[:, :self.cfg.spread_obs_dim].reshape(N, -1),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    action = self._spread_actor.deterministic_action(actor_obs).numpy()
            elif self.spread_mode == "repulsion":
                drone_positions = self._spread_env.get_drone_positions()
                target_pos = np.array([0.0, 0.0, 50.0])  # initial target pos
                action = np.zeros((N, 3), dtype=np.float32)
                for i in range(N):
                    vel = repulsion_controller(
                        drone_positions[i], drone_positions, i,
                        v_max=self.cfg.v_max, gain=1.0,
                        target_pos=target_pos, surround_weight=0.3,
                    )
                    action[i] = np.clip(vel / self.cfg.v_max, -1.0, 1.0)
            else:
                action = np.zeros((N, 3), dtype=np.float32)

            obs, _, terminated, truncated, _ = self._spread_env.step(action)
            if terminated or truncated:
                break

        # Get Phase 1 final positions
        final_positions = self._spread_env.get_drone_positions()

        # Phase 2: track from spread positions
        self._track_env.set_initial_positions(final_positions)
        return self._track_env.reset(seed=seed)

    def step(self, action):
        return self._track_env.step(action)

    def close(self):
        if self._spread_env is not None:
            self._spread_env.close()
        if self._track_env is not None:
            self._track_env.close()


def _worker(pipe, env_fn):
    env = env_fn()
    while True:
        cmd, data = pipe.recv()
        if cmd == "reset":
            obs, info = env.reset(seed=data)
            pipe.send((obs, info))
        elif cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                info["terminal_obs"] = obs.copy()
                obs, _ = env.reset()
            pipe.send((obs, reward, done, info))
        elif cmd == "close":
            env.close()
            pipe.close()
            break
        elif cmd == "set_attr":
            attr_name, attr_value = data
            setattr(env, attr_name, attr_value)
            pipe.send(True)
        elif cmd == "get_spaces":
            pipe.send((env.observation_space, env.action_space))
        else:
            raise ValueError(f"Unknown command: {cmd}")


class TrackVecEnv:
    def __init__(self, env_fns):
        import multiprocessing as mp
        self.num_envs = len(env_fns)
        self.closed = False
        ctx = mp.get_context("fork")
        self._parent_pipes = []
        self._processes = []
        for env_fn in env_fns:
            parent_pipe, child_pipe = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child_pipe, env_fn), daemon=True)
            proc.start()
            child_pipe.close()
            self._parent_pipes.append(parent_pipe)
            self._processes.append(proc)
        self._parent_pipes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self._parent_pipes[0].recv()

    def reset(self, seeds=None):
        if seeds is None:
            seeds = [None] * self.num_envs
        for pipe, seed in zip(self._parent_pipes, seeds):
            pipe.send(("reset", seed))
        results = [pipe.recv() for pipe in self._parent_pipes]
        obs = np.stack([r[0] for r in results])
        return obs

    def step(self, actions):
        for pipe, action in zip(self._parent_pipes, actions):
            pipe.send(("step", action))
        results = [pipe.recv() for pipe in self._parent_pipes]
        obs = np.stack([r[0] for r in results])
        rewards = np.stack([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def close(self):
        if self.closed:
            return
        self.closed = True
        for pipe in self._parent_pipes:
            try:
                pipe.send(("close", None))
            except BrokenPipeError:
                pass
        for proc in self._processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


# ── PPO Update ──

def ppo_update(actor, critic, actor_opt, critic_opt, buffer, cfg, value_normalizer=None):
    policy_losses, value_losses, entropies, clip_fracs = [], [], [], []

    if value_normalizer is not None:
        returns_np = buffer.returns.cpu().numpy().flatten()
        value_normalizer.update(returns_np)
        ret_std = np.sqrt(value_normalizer.var) + 1e-8
        buffer.returns = buffer.returns / ret_std

    for _ in range(cfg.ppo_epochs):
        for batch in buffer.get_batches(cfg.mini_batch_size):
            states, actions, old_log_probs, returns, advantages = batch

            actor_obs = states[:, :cfg.track_actor_obs_dim]
            critic_obs = states[:, :cfg.track_critic_obs_dim]

            new_log_probs, entropy = actor.evaluate(actor_obs, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - cfg.clip_epsilon,
                                  1 + cfg.clip_epsilon) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            entropy_loss = -entropy.mean()

            actor_total = policy_loss + cfg.entropy_coef * entropy_loss
            actor_opt.zero_grad()
            actor_total.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm_actor)
            actor_opt.step()

            new_values = critic(critic_obs)
            value_loss = nn.functional.mse_loss(new_values, returns)
            critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm_critic)
            critic_opt.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(-entropy_loss.item())
            clip_fracs.append(((ratio - 1.0).abs() > cfg.clip_epsilon).float().mean().item())

    return {
        "policy_loss": np.mean(policy_losses),
        "value_loss": np.mean(value_losses),
        "entropy": np.mean(entropies),
        "clip_frac": np.mean(clip_fracs),
    }


# ── Eval ──

def run_eval(actor, critic, cfg, device, obs_normalizer=None,
             spread_mode="repulsion", spread_ckpt_path=None,
             num_episodes=5):
    N = cfg.num_drones
    all_tr_P, all_rmse = [], []

    for ep in range(num_episodes):
        env = _WarmStartTrackEnv(
            cfg, seed=10000 + ep,
            spread_mode=spread_mode,
            spread_ckpt_path=spread_ckpt_path,
        )
        obs, _ = env.reset()
        ep_tr_P, ep_rmse = [], []

        for step in range(cfg.track_episode_length):
            if obs_normalizer is not None:
                obs_norm = _safe_normalize(obs, obs_normalizer)
            else:
                obs_norm = obs

            actor_obs = torch.tensor(
                obs_norm[:, :cfg.track_actor_obs_dim].reshape(N, -1),
                dtype=torch.float32, device=device,
            )
            with torch.no_grad():
                action = actor.deterministic_action(actor_obs)

            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

            if "tr_P_pos" in info:
                ep_tr_P.append(info["tr_P_pos"])
            if info.get("filter_initialized", False) and "result" in info:
                est = env._track_env._filter.get_estimate()
                true_pos = info["result"]["target_true_pos"]
                ep_rmse.append(np.linalg.norm(est[:3] - true_pos))

            if terminated or truncated:
                break

        env.close()
        all_tr_P.append(np.mean(ep_tr_P) if ep_tr_P else np.nan)
        all_rmse.append(np.mean(ep_rmse) if ep_rmse else np.nan)

    return {
        "tr_P": np.nanmean(all_tr_P),
        "rmse": np.nanmean(all_rmse),
    }


# ── Training Loop ──

def train(cfg: V3Config, spread_mode: str = "repulsion",
          spread_ckpt_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(cfg.save_path, "track")
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_path, "tb"))

    # Use longer rollouts for tracking
    track_rollout_steps = 1000

    env_fns = [
        _TrackEnvFactory(cfg, seed=cfg.seed + i,
                         spread_mode=spread_mode,
                         spread_ckpt_path=spread_ckpt_path)
        for i in range(cfg.num_envs)
    ]
    vec_env = TrackVecEnv(env_fns)

    E, N = cfg.num_envs, cfg.num_drones

    actor = BetaActor(device, obs_dim=cfg.track_actor_obs_dim, action_dim=3,
                      hidden_sizes=cfg.actor_hidden)
    critic = CentralizedCritic(device, obs_dim=cfg.track_critic_obs_dim,
                               hidden_sizes=cfg.critic_hidden)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr)

    buffer = MultiAgentRolloutBuffer(
        track_rollout_steps, E, N,
        obs_dim=cfg.track_critic_obs_dim,
        action_dim=3, device=device,
    )

    obs_normalizer = RunningMeanStd(shape=(cfg.track_critic_obs_dim,)) if cfg.normalize_obs else None
    value_normalizer = RunningMeanStd(shape=()) if cfg.normalize_value_targets else None

    history = {
        "episode_rewards": [], "episode_tr_Ps": [], "episode_rmses": [],
        "eval_steps": [], "eval_tr_Ps": [], "eval_rmses": [],
        "policy_losses": [], "value_losses": [], "entropy_losses": [],
    }

    total_steps = 0
    best_eval_trP = float("inf")
    obs = vec_env.reset()

    ep_reward_accum = np.zeros(E)
    ep_tr_P_accum = np.zeros(E)
    ep_length = np.zeros(E, dtype=int)

    print(f"[V3 Track] Training: {cfg.track_training_steps} steps, {E} envs, "
          f"spread_mode={spread_mode}")
    t_start = time.time()
    num_rollouts = 0

    while total_steps < cfg.track_training_steps:
        frac = 1.0 - total_steps / cfg.track_training_steps
        for pg in actor_optimizer.param_groups:
            pg["lr"] = cfg.lr * frac
        for pg in critic_optimizer.param_groups:
            pg["lr"] = cfg.lr * frac

        buffer.reset()
        with torch.no_grad():
            for t in range(track_rollout_steps):
                if obs_normalizer is not None:
                    obs_normalizer.update(obs)
                    obs_norm = _safe_normalize(obs, obs_normalizer)
                else:
                    obs_norm = obs

                obs_flat = torch.tensor(
                    obs_norm.reshape(E * N, -1), dtype=torch.float32, device=device
                )
                actor_obs_flat = obs_flat[:, :cfg.track_actor_obs_dim]
                critic_obs_flat = obs_flat[:, :cfg.track_critic_obs_dim]

                actions_flat, log_probs_flat = actor.act(actor_obs_flat)
                values_flat = critic(critic_obs_flat)

                actions_en = actions_flat.reshape(E, N, -1)
                log_probs_en = log_probs_flat.reshape(E, N)
                values_en = values_flat.reshape(E, N)

                actions_np = actions_en.cpu().numpy()
                next_obs, rewards, dones, infos = vec_env.step(actions_np)

                buffer.push(
                    states=torch.tensor(obs_norm, dtype=torch.float32, device=device),
                    actions=actions_en,
                    rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
                    dones=torch.tensor(dones, dtype=torch.float32, device=device),
                    log_probs=log_probs_en,
                    values=values_en,
                )

                ep_reward_accum += rewards.mean(axis=1)
                ep_length += 1
                for e in range(E):
                    if "tr_P_pos" in infos[e]:
                        ep_tr_P_accum[e] += infos[e]["tr_P_pos"]

                for e in range(E):
                    if dones[e]:
                        history["episode_rewards"].append(ep_reward_accum[e])
                        if ep_length[e] > 0:
                            history["episode_tr_Ps"].append(ep_tr_P_accum[e] / ep_length[e])
                        ep_reward_accum[e] = 0.0
                        ep_tr_P_accum[e] = 0.0
                        ep_length[e] = 0

                obs = next_obs
                total_steps += E * N

            if obs_normalizer is not None:
                obs_norm = _safe_normalize(obs, obs_normalizer)
            else:
                obs_norm = obs
            obs_flat = torch.tensor(
                obs_norm.reshape(E * N, -1), dtype=torch.float32, device=device
            )
            last_values = critic(obs_flat[:, :cfg.track_critic_obs_dim]).reshape(E, N)

        buffer.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)
        loss_info = ppo_update(
            actor, critic, actor_optimizer, critic_optimizer,
            buffer, cfg, value_normalizer,
        )
        num_rollouts += 1

        history["policy_losses"].append(loss_info["policy_loss"])
        history["value_losses"].append(loss_info["value_loss"])
        history["entropy_losses"].append(loss_info["entropy"])

        ep_rewards = history["episode_rewards"]
        ep_tr_Ps = history["episode_tr_Ps"]

        if writer and len(ep_rewards) > 0:
            writer.add_scalar("Reward/Episode", np.mean(ep_rewards[-10:]), total_steps)
            if ep_tr_Ps:
                writer.add_scalar("Filter/MeanTrP", np.mean(ep_tr_Ps[-10:]), total_steps)
            writer.add_scalar("Loss/Policy", loss_info["policy_loss"], total_steps)
            writer.add_scalar("Loss/Value", loss_info["value_loss"], total_steps)
            writer.add_scalar("Loss/Entropy", loss_info["entropy"], total_steps)

        if num_rollouts % 5 == 0:
            elapsed = time.time() - t_start
            sps = total_steps / elapsed if elapsed > 0 else 0
            recent_r = np.mean(ep_rewards[-10:]) if ep_rewards else 0
            recent_trP = np.mean(ep_tr_Ps[-10:]) if ep_tr_Ps else 0
            print(f"[{total_steps:>8d} | {elapsed:.0f}s | {sps:.0f} sps] "
                  f"r={recent_r:.2f}  trP={recent_trP:.1f}  "
                  f"ploss={loss_info['policy_loss']:.4f}  "
                  f"ent={loss_info['entropy']:.4f}  "
                  f"eps={len(ep_rewards)}")

        # Eval
        if total_steps % cfg.eval_freq < E * N * track_rollout_steps:
            eval_result = run_eval(
                actor, critic, cfg, device, obs_normalizer,
                spread_mode=spread_mode, spread_ckpt_path=spread_ckpt_path,
            )
            history["eval_steps"].append(total_steps)
            history["eval_tr_Ps"].append(eval_result["tr_P"])
            history["eval_rmses"].append(eval_result["rmse"])

            if writer:
                writer.add_scalar("Eval/TrP", eval_result["tr_P"], total_steps)
                writer.add_scalar("Eval/RMSE", eval_result["rmse"], total_steps)

            print(f"  [EVAL @ {total_steps}] "
                  f"tr(P)={eval_result['tr_P']:.1f}  "
                  f"RMSE={eval_result['rmse']:.2f}")

            # CSV logging
            csv_path = os.path.join(save_path, "eval_results.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                if write_header:
                    f.write("step,tr_P,rmse,reward\n")
                recent_r = np.mean(ep_rewards[-10:]) if ep_rewards else 0
                f.write(f"{total_steps},{eval_result['tr_P']:.4f},"
                        f"{eval_result['rmse']:.4f},{recent_r:.4f}\n")

            if eval_result["tr_P"] < best_eval_trP:
                best_eval_trP = eval_result["tr_P"]
                _save_checkpoint(save_path, actor, critic, actor_optimizer, critic_optimizer,
                                 obs_normalizer, value_normalizer, total_steps, history,
                                 cfg, tag="best")

        # Save periodic
        if total_steps % cfg.save_freq < E * N * track_rollout_steps:
            _save_checkpoint(save_path, actor, critic, actor_optimizer, critic_optimizer,
                             obs_normalizer, value_normalizer, total_steps, history,
                             cfg, tag=f"step_{total_steps}")

    # Final
    _save_checkpoint(save_path, actor, critic, actor_optimizer, critic_optimizer,
                     obs_normalizer, value_normalizer, total_steps, history,
                     cfg, tag="final")

    vec_env.close()
    writer.close()
    elapsed = time.time() - t_start
    print(f"[V3 Track] Training complete: {total_steps} steps, "
          f"{len(history['episode_rewards'])} episodes, {elapsed:.0f}s, "
          f"best eval tr(P)={best_eval_trP:.1f}")
    return history


def _save_checkpoint(save_path, actor, critic, actor_opt, critic_opt,
                     obs_norm, val_norm, step, history, cfg, tag=""):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"checkpoint_{tag}.pt")
    save_dict = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_opt.state_dict(),
        "critic_optimizer": critic_opt.state_dict(),
        "step": step,
        "phase": "track",
        "config": {
            "num_drones": cfg.num_drones,
            "actor_obs_dim": cfg.track_actor_obs_dim,
            "critic_obs_dim": cfg.track_critic_obs_dim,
            "action_dim": 3,
        },
        "full_config": asdict(cfg),
        "history": history,
    }
    if obs_norm is not None:
        save_dict["obs_normalizer"] = obs_norm.state_dict()
    if val_norm is not None:
        save_dict["value_normalizer"] = val_norm.state_dict()
    torch.save(save_dict, path)
    print(f"  Saved: {path}")


# ── CLI ──

def parse_args():
    parser = argparse.ArgumentParser(description="V3 Phase 2: Train Track Policy")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--spread-checkpoint", type=str, default=None,
                        help="Phase 1 checkpoint (uses RL spread)")
    parser.add_argument("--spread-heuristic", type=str, default=None,
                        choices=["repulsion", "none"],
                        help="Use heuristic spread instead of RL")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = V3Config()

    if args.steps is not None:
        cfg.track_training_steps = args.steps
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.tag is not None:
        cfg.save_path = f"./output/v3_{args.tag}"

    # Determine spread mode
    spread_mode = "repulsion"  # default
    spread_ckpt_path = None
    if args.spread_checkpoint:
        spread_mode = "rl"
        spread_ckpt_path = args.spread_checkpoint
    elif args.spread_heuristic:
        spread_mode = args.spread_heuristic

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "full_config" in ckpt:
            for k, v in ckpt["full_config"].items():
                if hasattr(cfg, k) and k not in ("track_training_steps", "save_path",
                                                   "num_envs"):
                    setattr(cfg, k, v)

    train(cfg, spread_mode=spread_mode, spread_ckpt_path=spread_ckpt_path)


if __name__ == "__main__":
    main()
