"""V2 Training — MAPPO with dual observations and asymmetric grad clipping."""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

from src.rl.ppo.tracking_buffer import MultiAgentRolloutBuffer
from src.rl.ppo.tracking_trainer import RunningMeanStd

from .config import V2TrackingConfig
from .networks import BetaActor, CentralizedCritic


def _safe_normalize(obs, normalizer):
    """Normalize obs and clamp to prevent NaN/Inf propagation."""
    out = normalizer.normalize(obs)
    return np.clip(out, -10.0, 10.0)


# ── SubprocVecEnv factory (uses V2 env) ──

class _V2EnvFactory:
    """Picklable factory for V2TrackingEnv in subprocesses."""
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = seed
    def __call__(self):
        from experiments.v2.env import V2TrackingEnv
        return V2TrackingEnv(self.cfg, seed=self.seed)


def _worker(pipe, env_fn):
    """Worker process for vectorized envs."""
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


class V2SubprocVecEnv:
    """Vectorized env for V2 (same pattern as V1 but creates V2 envs)."""

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

    def set_attr(self, attr_name, value):
        for pipe in self._parent_pipes:
            pipe.send(("set_attr", (attr_name, value)))
        for pipe in self._parent_pipes:
            pipe.recv()

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


# ── PPO Update with dual observations ──

def ppo_update(
    actor: BetaActor,
    critic: CentralizedCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    buffer: MultiAgentRolloutBuffer,
    cfg: V2TrackingConfig,
    value_normalizer: RunningMeanStd | None = None,
) -> dict:
    """Custom PPO update with split actor/critic observations.

    Buffer stores critic_obs (23D). Actor gets [:14], critic gets [:23].
    Separate optimizers with asymmetric grad clipping.
    """
    policy_losses = []
    value_losses = []
    entropies = []
    clip_fracs = []

    # Normalize returns if enabled
    if value_normalizer is not None:
        returns_np = buffer.returns.cpu().numpy().flatten()
        value_normalizer.update(returns_np)
        ret_std = np.sqrt(value_normalizer.var) + 1e-8
        buffer.returns = buffer.returns / ret_std

    for _ in range(cfg.ppo_epochs):
        for batch in buffer.get_batches(cfg.mini_batch_size):
            states, actions, old_log_probs, returns, advantages = batch

            # Split observations
            actor_obs = states[:, :cfg.actor_obs_dim]      # (B, 14)
            critic_obs = states[:, :cfg.critic_obs_dim]     # (B, 23)

            # Actor
            new_log_probs, entropy = actor.evaluate(actor_obs, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - cfg.clip_epsilon,
                                  1 + cfg.clip_epsilon) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            entropy_loss = -entropy.mean()

            actor_total = policy_loss + cfg.entropy_coef * entropy_loss
            actor_optimizer.zero_grad()
            actor_total.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm_actor)
            actor_optimizer.step()

            # Critic
            new_values = critic(critic_obs)
            value_loss = nn.functional.mse_loss(new_values, returns)

            critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm_critic)
            critic_optimizer.step()

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

def run_eval(actor, critic, cfg, device, obs_normalizer=None, num_episodes=3):
    """Run deterministic eval episodes."""
    from .env import V2TrackingEnv

    N = cfg.num_drones
    rewards_all, tr_Ps_all, rmses_all = [], [], []

    for ep in range(num_episodes):
        env = V2TrackingEnv(cfg, seed=10000 + ep)
        obs, _ = env.reset()
        ep_reward, ep_tr_P, ep_rmse = 0.0, [], []

        for step in range(cfg.episode_length):
            if obs_normalizer is not None:
                obs_norm = _safe_normalize(obs, obs_normalizer)
            else:
                obs_norm = obs

            actor_obs = torch.tensor(
                obs_norm[:, :cfg.actor_obs_dim].reshape(N, -1),
                dtype=torch.float32, device=device,
            )
            with torch.no_grad():
                action = actor.deterministic_action(actor_obs)

            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            ep_reward += reward.mean()

            if "tr_P_pos" in info:
                ep_tr_P.append(info["tr_P_pos"])
            if info.get("filter_initialized", False) and "result" in info:
                est = env._filter.get_estimate()
                true_pos = info["result"]["target_true_pos"]
                ep_rmse.append(np.linalg.norm(est[:3] - true_pos))

            if terminated or truncated:
                break

        env.close()
        rewards_all.append(ep_reward)
        tr_Ps_all.append(np.mean(ep_tr_P) if ep_tr_P else np.nan)
        rmses_all.append(np.mean(ep_rmse) if ep_rmse else np.nan)

    return {
        "reward": np.mean(rewards_all),
        "tr_P": np.nanmean(tr_Ps_all),
        "rmse": np.nanmean(rmses_all),
    }


# ── Training Loop ──

def train(cfg: V2TrackingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(cfg.save_path, "tb"))

    # Create vectorized envs
    env_fns = [_V2EnvFactory(cfg, seed=cfg.seed + i) for i in range(cfg.num_envs)]
    vec_env = V2SubprocVecEnv(env_fns)
    vec_env.set_attr("spawn_mode", cfg.spawn_mode)

    E, N = cfg.num_envs, cfg.num_drones

    # Networks
    actor = BetaActor(device, obs_dim=cfg.actor_obs_dim, action_dim=3,
                      hidden_sizes=cfg.actor_hidden)
    critic = CentralizedCritic(device, obs_dim=cfg.critic_obs_dim,
                               hidden_sizes=cfg.critic_hidden)

    # Separate optimizers (MAPPO: asymmetric clipping requires separate)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr)

    # Buffer stores full critic_obs (23D)
    buffer = MultiAgentRolloutBuffer(
        cfg.rollout_steps, E, N,
        obs_dim=cfg.critic_obs_dim,
        action_dim=3, device=device,
    )

    # Normalizers
    obs_normalizer = RunningMeanStd(shape=(cfg.critic_obs_dim,)) if cfg.normalize_obs else None
    value_normalizer = RunningMeanStd(shape=()) if cfg.normalize_value_targets else None

    # History
    history = {
        "episode_rewards": [], "episode_tr_Ps": [], "episode_rmses": [],
        "eval_steps": [], "eval_rewards": [], "eval_tr_Ps": [], "eval_rmses": [],
        "policy_losses": [], "value_losses": [], "entropy_losses": [],
    }

    total_steps = 0
    best_eval_trP = float("inf")
    obs = vec_env.reset()  # (E, N, critic_obs_dim)

    ep_reward_accum = np.zeros(E)
    ep_tr_P_accum = np.zeros(E)
    ep_length = np.zeros(E, dtype=int)

    print(f"Training: {cfg.max_training_steps} steps, {E} envs, {N} drones, "
          f"spawn={cfg.spawn_mode}, residual={cfg.residual_scale}")
    t_start = time.time()
    num_rollouts = 0

    while total_steps < cfg.max_training_steps:
        # ── LR annealing ──
        frac = 1.0 - total_steps / cfg.max_training_steps
        for pg in actor_optimizer.param_groups:
            pg["lr"] = cfg.lr * frac
        for pg in critic_optimizer.param_groups:
            pg["lr"] = cfg.lr * frac

        # ── Collect rollout ──
        buffer.reset()
        with torch.no_grad():
            for t in range(cfg.rollout_steps):
                if obs_normalizer is not None:
                    obs_normalizer.update(obs)
                    obs_norm = _safe_normalize(obs, obs_normalizer)
                else:
                    obs_norm = obs

                # Forward pass — actor sees :14, critic sees :23
                obs_flat = torch.tensor(
                    obs_norm.reshape(E * N, -1), dtype=torch.float32, device=device
                )
                actor_obs_flat = obs_flat[:, :cfg.actor_obs_dim]
                critic_obs_flat = obs_flat[:, :cfg.critic_obs_dim]

                actions_flat, log_probs_flat = actor.act(actor_obs_flat)
                values_flat = critic(critic_obs_flat)

                actions_en = actions_flat.reshape(E, N, -1)
                log_probs_en = log_probs_flat.reshape(E, N)
                values_en = values_flat.reshape(E, N)

                actions_np = actions_en.cpu().numpy()
                next_obs, rewards, dones, infos = vec_env.step(actions_np)

                # Store full critic obs in buffer
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

            # Bootstrap last values
            if obs_normalizer is not None:
                obs_norm = _safe_normalize(obs, obs_normalizer)
            else:
                obs_norm = obs
            obs_flat = torch.tensor(
                obs_norm.reshape(E * N, -1), dtype=torch.float32, device=device
            )
            last_values = critic(obs_flat[:, :cfg.critic_obs_dim]).reshape(E, N)

        # ── GAE + PPO Update ──
        buffer.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)
        loss_info = ppo_update(
            actor, critic, actor_optimizer, critic_optimizer,
            buffer, cfg, value_normalizer,
        )
        num_rollouts += 1

        history["policy_losses"].append(loss_info["policy_loss"])
        history["value_losses"].append(loss_info["value_loss"])
        history["entropy_losses"].append(loss_info["entropy"])

        # ── Logging ──
        ep_rewards = history["episode_rewards"]
        ep_tr_Ps = history["episode_tr_Ps"]

        if writer and len(ep_rewards) > 0:
            writer.add_scalar("Reward/Episode", np.mean(ep_rewards[-10:]), total_steps)
            if ep_tr_Ps:
                writer.add_scalar("Filter/MeanTrP", np.mean(ep_tr_Ps[-10:]), total_steps)
            writer.add_scalar("Loss/Policy", loss_info["policy_loss"], total_steps)
            writer.add_scalar("Loss/Value", loss_info["value_loss"], total_steps)
            writer.add_scalar("Loss/Entropy", loss_info["entropy"], total_steps)
            writer.add_scalar("Loss/ClipFrac", loss_info["clip_frac"], total_steps)

        if num_rollouts % 5 == 0:
            elapsed = time.time() - t_start
            sps = total_steps / elapsed if elapsed > 0 else 0
            recent_r = np.mean(ep_rewards[-10:]) if ep_rewards else 0
            recent_trP = np.mean(ep_tr_Ps[-10:]) if ep_tr_Ps else 0
            print(f"[{total_steps:>8d} | {elapsed:.0f}s | {sps:.0f} sps] "
                  f"r={recent_r:.2f}  trP={recent_trP:.1f}  "
                  f"ploss={loss_info['policy_loss']:.4f}  "
                  f"vloss={loss_info['value_loss']:.4f}  "
                  f"ent={loss_info['entropy']:.4f}  "
                  f"clip={loss_info['clip_frac']:.3f}  "
                  f"eps={len(ep_rewards)}")

        # ── Eval ──
        if total_steps % cfg.eval_freq < E * N * cfg.rollout_steps:
            eval_result = run_eval(actor, critic, cfg, device, obs_normalizer)
            history["eval_steps"].append(total_steps)
            history["eval_rewards"].append(eval_result["reward"])
            history["eval_tr_Ps"].append(eval_result["tr_P"])
            history["eval_rmses"].append(eval_result["rmse"])

            if writer:
                writer.add_scalar("Eval/Reward", eval_result["reward"], total_steps)
                writer.add_scalar("Eval/TrP", eval_result["tr_P"], total_steps)
                writer.add_scalar("Eval/RMSE", eval_result["rmse"], total_steps)

            print(f"  [EVAL @ {total_steps}] "
                  f"reward={eval_result['reward']:.2f}  "
                  f"tr(P)={eval_result['tr_P']:.1f}  "
                  f"RMSE={eval_result['rmse']:.2f}")

            # Save best
            if eval_result["tr_P"] < best_eval_trP:
                best_eval_trP = eval_result["tr_P"]
                _save_checkpoint(cfg, actor, critic, actor_optimizer, critic_optimizer,
                                 obs_normalizer, value_normalizer, total_steps, history,
                                 tag="best")

            # CSV
            csv_path = os.path.join(cfg.save_path, "eval_results.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                if write_header:
                    f.write("step,reward,tr_P,rmse\n")
                f.write(f"{total_steps},{eval_result['reward']:.4f},"
                        f"{eval_result['tr_P']:.4f},{eval_result['rmse']:.4f}\n")

        # ── Save periodic ──
        if total_steps % cfg.save_freq < E * N * cfg.rollout_steps:
            _save_checkpoint(cfg, actor, critic, actor_optimizer, critic_optimizer,
                             obs_normalizer, value_normalizer, total_steps, history,
                             tag=f"step_{total_steps}")

    # Final save
    _save_checkpoint(cfg, actor, critic, actor_optimizer, critic_optimizer,
                     obs_normalizer, value_normalizer, total_steps, history,
                     tag="final")

    vec_env.close()
    writer.close()
    elapsed = time.time() - t_start
    print(f"Training complete: {total_steps} steps, {len(history['episode_rewards'])} episodes, "
          f"{elapsed:.0f}s, best eval tr(P)={best_eval_trP:.1f}")
    return history


def _save_checkpoint(cfg, actor, critic, actor_opt, critic_opt,
                     obs_norm, val_norm, step, history, tag=""):
    os.makedirs(cfg.save_path, exist_ok=True)
    path = os.path.join(cfg.save_path, f"checkpoint_{tag}.pt")
    save_dict = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_opt.state_dict(),
        "critic_optimizer": critic_opt.state_dict(),
        "step": step,
        "config": {
            "num_drones": cfg.num_drones,
            "actor_obs_dim": cfg.actor_obs_dim,
            "critic_obs_dim": cfg.critic_obs_dim,
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
    parser = argparse.ArgumentParser(description="V2 MAPPO Training")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--spawn-mode", type=str, default=None,
                        choices=["cluster", "normal", "mixed"])
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = V2TrackingConfig()

    # Override config from CLI
    if args.steps is not None:
        cfg.max_training_steps = args.steps
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.spawn_mode is not None:
        cfg.spawn_mode = args.spawn_mode
    if args.residual_scale is not None:
        cfg.residual_scale = args.residual_scale
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.save_path is not None:
        cfg.save_path = args.save_path
    elif args.tag is not None:
        cfg.save_path = f"./output/v2_{args.tag}"

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Restore config from checkpoint
        if "full_config" in ckpt:
            for k, v in ckpt["full_config"].items():
                if hasattr(cfg, k) and k not in ("max_training_steps", "save_path",
                                                   "num_envs", "spawn_mode"):
                    setattr(cfg, k, v)
        # CLI overrides still apply above

    train(cfg)


if __name__ == "__main__":
    main()
