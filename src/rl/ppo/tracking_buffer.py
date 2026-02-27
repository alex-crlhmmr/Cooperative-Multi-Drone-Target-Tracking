"""Multi-agent rollout buffer for PPO training.

Stores transitions with shape (rollout_steps, num_envs, num_drones, ...).
Flattens to (T*E*N, ...) for batching — all drones share the same policy.
"""

import torch


class MultiAgentRolloutBuffer:
    """Rollout buffer for multi-agent PPO with shared policy.

    Storage shape: (T, E, N, feature_dim) where
        T = rollout_steps
        E = num_envs
        N = num_drones
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        num_drones: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        self.T = rollout_steps
        self.E = num_envs
        self.N = num_drones
        self.device = device
        self.ptr = 0

        # (T, E, N, ...)
        self.states = torch.zeros(rollout_steps, num_envs, num_drones, obs_dim, device=device)
        self.actions = torch.zeros(rollout_steps, num_envs, num_drones, action_dim, device=device)
        self.rewards = torch.zeros(rollout_steps, num_envs, num_drones, device=device)
        self.log_probs = torch.zeros(rollout_steps, num_envs, num_drones, device=device)
        self.values = torch.zeros(rollout_steps, num_envs, num_drones, device=device)

        # (T, E) — episode-level dones shared across drones
        self.dones = torch.zeros(rollout_steps, num_envs, device=device)

        # Computed after rollout
        self.advantages = torch.zeros(rollout_steps, num_envs, num_drones, device=device)
        self.returns = torch.zeros(rollout_steps, num_envs, num_drones, device=device)

    def push(self, states, actions, rewards, dones, log_probs, values):
        """Store one timestep of transitions.

        Args:
            states: (E, N, obs_dim)
            actions: (E, N, action_dim)
            rewards: (E, N)
            dones: (E,) episode-level
            log_probs: (E, N)
            values: (E, N)
        """
        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.ptr += 1

    def compute_gae(self, last_values, gamma, gae_lambda):
        """Compute GAE advantages and returns.

        Args:
            last_values: (E, N) value estimates for state after last step
            gamma: discount factor
            gae_lambda: GAE lambda
        """
        gae = torch.zeros(self.E, self.N, device=self.device)

        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_values = last_values  # (E, N)
            else:
                next_values = self.values[t + 1]  # (E, N)

            # Done mask: expand from (E,) to (E, N)
            not_done = (1 - self.dones[t]).unsqueeze(-1)  # (E, 1)

            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

        # Normalize advantages across all T*E*N transitions
        flat_adv = self.advantages.reshape(-1)
        self.advantages = (self.advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)

    def get_batches(self, mini_batch_size):
        """Yield shuffled mini-batches from flattened transitions.

        Flattens (T, E, N, ...) -> (T*E*N, ...) and shuffles.

        Yields:
            (states, actions, log_probs, returns, advantages) tuples
        """
        total = self.T * self.E * self.N

        # Flatten
        flat_states = self.states.reshape(total, -1)
        flat_actions = self.actions.reshape(total, -1)
        flat_log_probs = self.log_probs.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_advantages = self.advantages.reshape(total)

        # Shuffle
        indices = torch.randperm(total, device=self.device)

        for start in range(0, total, mini_batch_size):
            batch_idx = indices[start:start + mini_batch_size]
            yield (
                flat_states[batch_idx],
                flat_actions[batch_idx],
                flat_log_probs[batch_idx],
                flat_returns[batch_idx],
                flat_advantages[batch_idx],
            )

    def reset(self):
        self.ptr = 0
