import torch
import torch.nn as nn
import torch.optim as optim


class PPOAgent:
    def __init__(self,
                actor,
                critic,
                lr,
                clip_epsilon,
                entropy_coef,
                value_coef,
                num_epochs,
                mini_batch_size,
                max_grad_norm):
        self.actor = actor
        self.critic = critic
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=lr
        )
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

    def update(self, rollout_buffer):
        for _ in range(self.num_epochs):
          for batch in rollout_buffer.get_batches(self.mini_batch_size):
                # Unpack batch
                states, actions, old_log_probs, returns, advantages = batch

                # Compute new log probs, values, and ratios
                new_log_probs, entropy = self.actor.evaluate(states, actions)

                # Get critic values for states
                new_values = self.critic.forward(states)
                value_loss = nn.functional.mse_loss(new_values, returns)

                # Compute policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()
