"""Training loop for RL-based active tracking.

Handles: rollout collection across vectorized multi-agent envs,
observation normalization, GAE computation, PPO updates, logging.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict

from .tracking_config import TrackingConfig
from .tracking_buffer import MultiAgentRolloutBuffer
from .ppo_agent import PPOAgent


class RunningMeanStd:
    """Welford's online algorithm for running mean and variance.

    Used for observation normalization — critical because obs components
    span wildly different scales (positions ~100m, eigenvalues ~10000, flags 0/1).
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # small initial count to avoid div-by-zero

    def update(self, batch: np.ndarray):
        """Update running stats with a batch of observations.

        Args:
            batch: (..., shape) array — flattened over leading dims
        """
        batch = batch.reshape(-1, *self.mean.shape)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        new_var = m2 / total
        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


class TrackingTrainer:
    """Training loop for multi-drone active tracking PPO."""

    def __init__(
        self,
        vec_env,
        agent: PPOAgent,
        buffer: MultiAgentRolloutBuffer,
        cfg: TrackingConfig,
        device: torch.device,
        writer: SummaryWriter | None = None,
        obs_normalizer: RunningMeanStd | None = None,
        start_step: int = 0,
        spawn_mode: str = "mixed",
    ):
        self.vec_env = vec_env
        self.agent = agent
        self.buffer = buffer
        self.cfg = cfg
        self.device = device
        self.writer = writer
        self.obs_normalizer = obs_normalizer
        self.start_step = start_step
        self.spawn_mode = spawn_mode

        self.E = cfg.num_envs
        self.N = cfg.num_drones

        # Return normalizer — stabilizes value targets
        self.return_normalizer = RunningMeanStd(shape=())

        # History for end-of-training plots
        self.history = {
            "episode_rewards": [],
            "episode_tr_Ps": [],
            "episode_rmses": [],
            "eval_steps": [],
            "eval_rewards": [],
            "eval_tr_Ps": [],
            "eval_rmses": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
        }

    def train(self):
        """Main training loop."""
        total_steps = self.start_step
        num_rollouts = 0

        self.vec_env.set_attr("spawn_mode", self.spawn_mode)

        obs = self.vec_env.reset()  # (E, N, obs_dim)

        # Track per-env episode stats
        ep_reward_accum = np.zeros(self.E)
        ep_tr_P_accum = np.zeros(self.E)
        ep_rmse_accum = np.zeros(self.E)
        ep_length = np.zeros(self.E, dtype=int)

        print(f"Starting training: {self.cfg.max_training_steps} steps, "
              f"{self.E} envs, {self.N} drones")
        t_start = time.time()

        while total_steps < self.cfg.max_training_steps:
            # --- Collect rollout ---
            self.buffer.reset()

            with torch.no_grad():
                for t in range(self.cfg.rollout_steps):
                    # Normalize obs if enabled
                    if self.obs_normalizer is not None:
                        self.obs_normalizer.update(obs)
                        obs_norm = self.obs_normalizer.normalize(obs)
                    else:
                        obs_norm = obs

                    # Flatten (E, N) -> (E*N) for batched forward pass
                    obs_flat = torch.tensor(
                        obs_norm.reshape(self.E * self.N, -1),
                        dtype=torch.float32, device=self.device
                    )

                    # Policy forward pass
                    actions_flat, log_probs_flat = self.agent.actor.act(obs_flat)
                    values_flat = self.agent.critic(obs_flat)

                    # Reshape back to (E, N, ...)
                    actions_en = actions_flat.reshape(self.E, self.N, -1)
                    log_probs_en = log_probs_flat.reshape(self.E, self.N)
                    values_en = values_flat.reshape(self.E, self.N)

                    # Step all envs
                    actions_np = actions_en.cpu().numpy()
                    next_obs, rewards, dones, infos = self.vec_env.step(actions_np)

                    # Store in buffer
                    self.buffer.push(
                        states=torch.tensor(obs_norm, dtype=torch.float32, device=self.device),
                        actions=actions_en,
                        rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
                        dones=torch.tensor(dones, dtype=torch.float32, device=self.device),
                        log_probs=log_probs_en,
                        values=values_en,
                    )

                    # Track episode stats
                    ep_reward_accum += rewards.mean(axis=1)  # mean over drones
                    ep_length += 1

                    for e in range(self.E):
                        if "tr_P_pos" in infos[e]:
                            ep_tr_P_accum[e] += infos[e]["tr_P_pos"]
                        if "result" in infos[e] and infos[e].get("filter_initialized", False):
                            result = infos[e]["result"]
                            if "target_true_pos" in result:
                                from src.filters.measurement import bearing_measurement
                                # RMSE from info's tr_P_pos (already computed in env)
                                ep_rmse_accum[e] += np.sqrt(infos[e]["tr_P_pos"] / 3.0)

                    # Handle done episodes
                    for e in range(self.E):
                        if dones[e]:
                            self.history["episode_rewards"].append(ep_reward_accum[e])
                            if ep_length[e] > 0:
                                self.history["episode_tr_Ps"].append(ep_tr_P_accum[e] / ep_length[e])
                                self.history["episode_rmses"].append(ep_rmse_accum[e] / ep_length[e])
                            ep_reward_accum[e] = 0.0
                            ep_tr_P_accum[e] = 0.0
                            ep_rmse_accum[e] = 0.0
                            ep_length[e] = 0

                    obs = next_obs
                    total_steps += self.E * self.N

                # Compute last values for GAE bootstrapping
                if self.obs_normalizer is not None:
                    obs_norm = self.obs_normalizer.normalize(obs)
                else:
                    obs_norm = obs

                obs_flat = torch.tensor(
                    obs_norm.reshape(self.E * self.N, -1),
                    dtype=torch.float32, device=self.device
                )
                last_values = self.agent.critic(obs_flat).reshape(self.E, self.N)

            # --- GAE + PPO Update ---
            self.buffer.compute_gae(last_values, self.cfg.gamma, self.cfg.gae_lambda)

            # Normalize returns to stabilize value learning
            returns_np = self.buffer.returns.cpu().numpy().flatten()
            self.return_normalizer.update(returns_np)
            ret_std = np.sqrt(self.return_normalizer.var) + 1e-8
            self.buffer.returns = self.buffer.returns / ret_std

            loss_info = self._update_with_logging()

            num_rollouts += 1

            # --- Logging ---
            ep_rewards = self.history["episode_rewards"]
            ep_tr_Ps = self.history["episode_tr_Ps"]

            if self.writer and len(ep_rewards) > 0:
                self.writer.add_scalar("Reward/Episode", np.mean(ep_rewards[-10:]), total_steps)
                if len(ep_tr_Ps) > 0:
                    self.writer.add_scalar("Filter/MeanTrP", np.mean(ep_tr_Ps[-10:]), total_steps)
                self.writer.add_scalar("Loss/Policy", loss_info["policy_loss"], total_steps)
                self.writer.add_scalar("Loss/Value", loss_info["value_loss"], total_steps)
                self.writer.add_scalar("Loss/Entropy", loss_info["entropy"], total_steps)
                self.writer.add_scalar("Training/TotalSteps", total_steps, num_rollouts)

            self.history["policy_losses"].append(loss_info["policy_loss"])
            self.history["value_losses"].append(loss_info["value_loss"])
            self.history["entropy_losses"].append(loss_info["entropy"])

            # --- Periodic eval ---
            if total_steps % self.cfg.eval_freq < self.E * self.N * self.cfg.rollout_steps:
                eval_result = self._run_eval()
                self.history["eval_steps"].append(total_steps)
                self.history["eval_rewards"].append(eval_result["reward"])
                self.history["eval_tr_Ps"].append(eval_result["tr_P"])
                self.history["eval_rmses"].append(eval_result["rmse"])

                if self.writer:
                    self.writer.add_scalar("Eval/Reward", eval_result["reward"], total_steps)
                    self.writer.add_scalar("Eval/TrP", eval_result["tr_P"], total_steps)
                    self.writer.add_scalar("Eval/RMSE", eval_result["rmse"], total_steps)

                print(f"  [EVAL @ {total_steps} steps] "
                      f"reward={eval_result['reward']:.2f}  "
                      f"tr(P)={eval_result['tr_P']:.1f}  "
                      f"RMSE={eval_result['rmse']:.2f}")

                # Write results to CSV for easy monitoring
                self._append_eval_csv(total_steps, eval_result)

            if num_rollouts % 5 == 0:
                elapsed = time.time() - t_start
                sps = total_steps / elapsed if elapsed > 0 else 0
                recent_reward = np.mean(ep_rewards[-10:]) if ep_rewards else 0
                recent_trP = np.mean(ep_tr_Ps[-10:]) if ep_tr_Ps else 0
                print(f"[{total_steps:>8d} steps | {elapsed:.0f}s | {sps:.0f} sps] "
                      f"reward={recent_reward:.2f}  tr(P)={recent_trP:.1f}  "
                      f"p_loss={loss_info['policy_loss']:.4f}  "
                      f"v_loss={loss_info['value_loss']:.4f}  "
                      f"entropy={loss_info['entropy']:.4f}  "
                      f"episodes={len(ep_rewards)}")

            # --- Save checkpoint ---
            if total_steps % self.cfg.save_freq < self.E * self.N * self.cfg.rollout_steps:
                self._save_checkpoint(total_steps)

        self._save_checkpoint(total_steps, final=True)
        elapsed = time.time() - t_start
        print(f"Training complete: {total_steps} steps, {len(ep_rewards)} episodes, {elapsed:.0f}s")
        return self.history

    def _update_with_logging(self) -> dict:
        """Run PPO update and return loss components for logging."""
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(self.agent.num_epochs):
            for batch in self.buffer.get_batches(self.agent.mini_batch_size):
                states, actions, old_log_probs, returns, advantages = batch

                new_log_probs, entropy = self.agent.actor.evaluate(states, actions)
                new_values = self.agent.critic.forward(states)
                value_loss = nn.functional.mse_loss(new_values, returns)

                ratio = torch.exp(new_log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1 - self.agent.clip_epsilon,
                                      1 + self.agent.clip_epsilon) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                entropy_loss = -entropy.mean()

                total_loss = (policy_loss
                              + self.agent.value_coef * value_loss
                              + self.agent.entropy_coef * entropy_loss)

                self.agent.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.agent.actor.parameters()) + list(self.agent.critic.parameters()),
                    self.agent.max_grad_norm)
                self.agent.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(-entropy_loss.item())

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
        }

    def _run_eval(self, num_episodes: int = 3) -> dict:
        """Run deterministic eval episodes (single env, no SubprocVecEnv)."""
        from .tracking_env import MultiDroneTrackingEnv

        rewards = []
        tr_Ps = []
        rmses = []

        for ep in range(num_episodes):
            env = MultiDroneTrackingEnv(self.cfg, seed=10000 + ep)
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_tr_P = []
            ep_rmse = []

            for step in range(self.cfg.episode_length):
                if self.obs_normalizer is not None:
                    obs_norm = self.obs_normalizer.normalize(obs)
                else:
                    obs_norm = obs

                obs_t = torch.tensor(
                    obs_norm.reshape(self.N, -1),
                    dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    dist = self.agent.actor.forward(obs_t)
                    action = torch.tanh(dist.mean)

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
            rewards.append(ep_reward)
            tr_Ps.append(np.mean(ep_tr_P) if ep_tr_P else np.nan)
            rmses.append(np.mean(ep_rmse) if ep_rmse else np.nan)

        return {
            "reward": np.mean(rewards),
            "tr_P": np.nanmean(tr_Ps),
            "rmse": np.nanmean(rmses),
        }

    def _append_eval_csv(self, step: int, result: dict):
        """Append eval result to a CSV file in the save directory."""
        csv_path = os.path.join(self.cfg.save_path, "eval_results.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if write_header:
                f.write("step,reward,tr_P,rmse\n")
            f.write(f"{step},{result['reward']:.4f},{result['tr_P']:.4f},{result['rmse']:.4f}\n")

    def _save_checkpoint(self, step: int, final: bool = False):
        os.makedirs(self.cfg.save_path, exist_ok=True)
        tag = "final" if final else f"step_{step}"
        path = os.path.join(self.cfg.save_path, f"checkpoint_{tag}.pt")

        save_dict = {
            "actor": self.agent.actor.state_dict(),
            "critic": self.agent.critic.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "step": step,
            "config": {
                "num_drones": self.cfg.num_drones,
                "obs_dim": self.buffer.states.shape[-1],
                "action_dim": self.buffer.actions.shape[-1],
            },
            "full_config": asdict(self.cfg),
            "history": self.history,
        }

        if self.obs_normalizer is not None:
            save_dict["obs_normalizer"] = self.obs_normalizer.state_dict()

        torch.save(save_dict, path)
        print(f"  Saved checkpoint: {path}")
