"""V7 Training — RL residual spread with angular separation reward.

Trains spread policy only. Tracking uses plain chase+offset (no RL).
Eval runs short tracking after spread to measure downstream tr(P).
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
from .config import V7Config


def _safe_normalize(obs, normalizer):
    out = normalizer.normalize(obs)
    return np.clip(out, -10.0, 10.0)


# ── SubprocVecEnv ──

class _SpreadEnvFactory:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = seed
    def __call__(self):
        from experiments.v7.spread_env import SpreadEnv
        return SpreadEnv(self.cfg, seed=self.seed)


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
                if hasattr(env, 'get_drone_positions'):
                    info["final_positions"] = env.get_drone_positions().copy()
                obs, _ = env.reset()
            pipe.send((obs, reward, done, info))
        elif cmd == "close":
            env.close()
            pipe.close()
            break
        elif cmd == "get_spaces":
            pipe.send((env.observation_space, env.action_space))
        else:
            raise ValueError(f"Unknown command: {cmd}")


class SpreadVecEnv:
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

            # Per-minibatch advantage normalization
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            actor_obs = states[:, :cfg.spread_obs_dim]
            critic_obs = states[:, :cfg.spread_critic_obs_dim]

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


# ── Eval (spread + tracking) ──

def run_eval(actor, cfg, device, obs_normalizer=None, num_episodes=5,
             track_steps=2000):
    """Eval: RL spread → chase+offset tracking. Measures downstream tr(P)."""
    from .spread_env import SpreadEnv
    from .track_env import TrackEnv

    N = cfg.num_drones
    all_min_angles, all_mean_ranges, all_tr_P = [], [], []

    for ep in range(num_episodes):
        seed = 10000 + ep
        spread_env = SpreadEnv(cfg, seed=seed)
        obs, _ = spread_env.reset()
        ep_min_angle = 0.0
        ep_mean_range = 0.0

        for step in range(cfg.spread_steps):
            if obs_normalizer is not None:
                obs_norm = _safe_normalize(obs, obs_normalizer)
            else:
                obs_norm = obs

            actor_obs = torch.tensor(
                obs_norm[:, :cfg.spread_obs_dim].reshape(N, -1),
                dtype=torch.float32, device=device,
            )
            with torch.no_grad():
                action = actor.deterministic_action(actor_obs)

            obs, reward, terminated, truncated, info = spread_env.step(action.cpu().numpy())
            ep_min_angle = info.get("min_angular_sep", ep_min_angle)
            ep_mean_range = info.get("mean_range_to_target", ep_mean_range)

            if terminated or truncated:
                break

        final_positions = spread_env.get_drone_positions()
        spread_env.close()

        all_min_angles.append(np.degrees(ep_min_angle))
        all_mean_ranges.append(ep_mean_range)

        # Short tracking run
        track_env = TrackEnv(cfg, seed=seed)
        track_env.set_initial_positions(final_positions)
        track_obs, _ = track_env.reset()
        ep_tr_P = []

        for step in range(track_steps):
            action_zero = np.zeros((N, 3), dtype=np.float32)
            track_obs, _, terminated, truncated, info = track_env.step(action_zero)
            if "tr_P_pos" in info:
                ep_tr_P.append(info["tr_P_pos"])
            if terminated or truncated:
                break

        track_env.close()
        all_tr_P.append(np.mean(ep_tr_P) if ep_tr_P else np.nan)

    return {
        "min_angle_deg": np.mean(all_min_angles),
        "mean_range": np.mean(all_mean_ranges),
        "tr_P": np.nanmean(all_tr_P),
    }


# ── Training Loop ──

def train(cfg: V7Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(cfg.save_path, "spread")
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_path, "tb"))

    env_fns = [_SpreadEnvFactory(cfg, seed=cfg.seed + i) for i in range(cfg.num_envs)]
    vec_env = SpreadVecEnv(env_fns)

    E, N = cfg.num_envs, cfg.num_drones

    actor = BetaActor(device, obs_dim=cfg.spread_obs_dim, action_dim=3,
                      hidden_sizes=cfg.actor_hidden)
    critic = CentralizedCritic(device, obs_dim=cfg.spread_critic_obs_dim,
                               hidden_sizes=cfg.critic_hidden)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr)

    buffer = MultiAgentRolloutBuffer(
        cfg.rollout_steps, E, N,
        obs_dim=cfg.spread_critic_obs_dim,
        action_dim=3, device=device,
    )

    obs_normalizer = RunningMeanStd(shape=(cfg.spread_critic_obs_dim,)) if cfg.normalize_obs else None
    value_normalizer = RunningMeanStd(shape=()) if cfg.normalize_value_targets else None

    history = {
        "episode_rewards": [], "episode_min_angles": [], "episode_mean_ranges": [],
        "eval_steps": [], "eval_min_angles": [], "eval_tr_Ps": [], "eval_ranges": [],
        "policy_losses": [], "value_losses": [], "entropy_losses": [],
    }

    total_steps = 0
    best_eval_trP = float("inf")
    obs = vec_env.reset()

    ep_reward_accum = np.zeros(E)
    ep_length = np.zeros(E, dtype=int)

    print(f"[V7] Training: {cfg.spread_training_steps} steps, {E} envs")
    print(f"[V7] Reward: angular separation (min_angle={cfg.reward_min_angle}, "
          f"mean_angle={cfg.reward_mean_angle}, range={cfg.reward_range})")
    print(f"[V7] Spread: residual_scale={cfg.spread_residual_scale}, "
          f"repulsion_gain={cfg.spread_repulsion_gain}, "
          f"surround_weight={cfg.spread_surround_weight}")
    t_start = time.time()
    num_rollouts = 0

    while total_steps < cfg.spread_training_steps:
        # LR annealing
        frac = 1.0 - total_steps / cfg.spread_training_steps
        for pg in actor_optimizer.param_groups:
            pg["lr"] = cfg.lr * frac
        for pg in critic_optimizer.param_groups:
            pg["lr"] = cfg.lr * frac

        buffer.reset()
        with torch.no_grad():
            for t in range(cfg.rollout_steps):
                if obs_normalizer is not None:
                    obs_normalizer.update(obs)
                    obs_norm = _safe_normalize(obs, obs_normalizer)
                else:
                    obs_norm = obs

                obs_flat = torch.tensor(
                    obs_norm.reshape(E * N, -1), dtype=torch.float32, device=device
                )
                actor_obs_flat = obs_flat[:, :cfg.spread_obs_dim]
                critic_obs_flat = obs_flat[:, :cfg.spread_critic_obs_dim]

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
                    if dones[e]:
                        history["episode_rewards"].append(ep_reward_accum[e])
                        min_angle = infos[e].get("min_angular_sep", 0.0)
                        history["episode_min_angles"].append(np.degrees(min_angle))
                        mean_range = infos[e].get("mean_range_to_target", 0.0)
                        history["episode_mean_ranges"].append(mean_range)
                        ep_reward_accum[e] = 0.0
                        ep_length[e] = 0

                obs = next_obs
                total_steps += E * N

            # Bootstrap
            if obs_normalizer is not None:
                obs_norm = _safe_normalize(obs, obs_normalizer)
            else:
                obs_norm = obs
            obs_flat = torch.tensor(
                obs_norm.reshape(E * N, -1), dtype=torch.float32, device=device
            )
            last_values = critic(obs_flat[:, :cfg.spread_critic_obs_dim]).reshape(E, N)

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
        ep_angles = history["episode_min_angles"]
        ep_ranges = history["episode_mean_ranges"]

        if writer and len(ep_rewards) > 0:
            writer.add_scalar("Reward/Episode", np.mean(ep_rewards[-10:]), total_steps)
            if ep_angles:
                writer.add_scalar("Spread/MinAngleDeg", np.mean(ep_angles[-10:]), total_steps)
            if ep_ranges:
                writer.add_scalar("Spread/MeanRange", np.mean(ep_ranges[-10:]), total_steps)
            writer.add_scalar("Loss/Policy", loss_info["policy_loss"], total_steps)
            writer.add_scalar("Loss/Value", loss_info["value_loss"], total_steps)
            writer.add_scalar("Loss/Entropy", loss_info["entropy"], total_steps)
            writer.add_scalar("Loss/ClipFrac", loss_info["clip_frac"], total_steps)
            writer.add_scalar("LR", cfg.lr * frac, total_steps)

        if num_rollouts % 5 == 0:
            elapsed = time.time() - t_start
            sps = total_steps / elapsed if elapsed > 0 else 0
            recent_r = np.mean(ep_rewards[-10:]) if ep_rewards else 0
            recent_angle = np.mean(ep_angles[-10:]) if ep_angles else 0
            recent_range = np.mean(ep_ranges[-10:]) if ep_ranges else 0
            print(f"[{total_steps:>8d} | {elapsed:.0f}s | {sps:.0f} sps] "
                  f"r={recent_r:.2f}  min_angle={recent_angle:.1f}°  "
                  f"range={recent_range:.0f}m  "
                  f"ploss={loss_info['policy_loss']:.4f}  "
                  f"ent={loss_info['entropy']:.4f}  "
                  f"eps={len(ep_rewards)}")

        # Eval
        if total_steps % cfg.eval_freq < E * N * cfg.rollout_steps:
            eval_result = run_eval(actor, cfg, device, obs_normalizer)
            history["eval_steps"].append(total_steps)
            history["eval_min_angles"].append(eval_result["min_angle_deg"])
            history["eval_tr_Ps"].append(eval_result["tr_P"])
            history["eval_ranges"].append(eval_result["mean_range"])

            if writer:
                writer.add_scalar("Eval/MinAngleDeg", eval_result["min_angle_deg"], total_steps)
                writer.add_scalar("Eval/TrP", eval_result["tr_P"], total_steps)
                writer.add_scalar("Eval/MeanRange", eval_result["mean_range"], total_steps)

            print(f"  [EVAL @ {total_steps}] "
                  f"min_angle={eval_result['min_angle_deg']:.1f}°  "
                  f"range={eval_result['mean_range']:.0f}m  "
                  f"tr(P)={eval_result['tr_P']:.1f}")

            # CSV logging
            csv_path = os.path.join(save_path, "eval_results.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                if write_header:
                    f.write("step,min_angle_deg,mean_range,tr_P,reward,entropy\n")
                recent_r = np.mean(ep_rewards[-10:]) if ep_rewards else 0
                f.write(f"{total_steps},{eval_result['min_angle_deg']:.4f},"
                        f"{eval_result['mean_range']:.4f},"
                        f"{eval_result['tr_P']:.4f},"
                        f"{recent_r:.4f},{loss_info['entropy']:.4f}\n")

            if eval_result["tr_P"] < best_eval_trP:
                best_eval_trP = eval_result["tr_P"]
                _save_checkpoint(save_path, actor, critic, actor_optimizer, critic_optimizer,
                                 obs_normalizer, value_normalizer, total_steps, history,
                                 cfg, tag="best")

        # Save periodic
        if total_steps % cfg.save_freq < E * N * cfg.rollout_steps:
            _save_checkpoint(save_path, actor, critic, actor_optimizer, critic_optimizer,
                             obs_normalizer, value_normalizer, total_steps, history,
                             cfg, tag=f"step_{total_steps}")

    # Final save
    _save_checkpoint(save_path, actor, critic, actor_optimizer, critic_optimizer,
                     obs_normalizer, value_normalizer, total_steps, history,
                     cfg, tag="final")

    vec_env.close()
    writer.close()
    elapsed = time.time() - t_start
    print(f"[V7] Training complete: {total_steps} steps, "
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
        "phase": "spread",
        "config": {
            "num_drones": cfg.num_drones,
            "actor_obs_dim": cfg.spread_obs_dim,
            "critic_obs_dim": cfg.spread_critic_obs_dim,
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
    parser = argparse.ArgumentParser(description="V7: Train RL Spread with Angular Separation Reward")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = V7Config()

    if args.steps is not None:
        cfg.spread_training_steps = args.steps
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.residual_scale is not None:
        cfg.spread_residual_scale = args.residual_scale
    if args.seed is not None:
        cfg.seed = args.seed
    if args.tag is not None:
        cfg.save_path = f"./output/v7_{args.tag}"

    train(cfg)


if __name__ == "__main__":
    main()
