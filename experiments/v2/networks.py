"""V2 Networks — BetaActor + CentralizedCritic for MAPPO."""

import torch
import torch.nn as nn
from torch.distributions import Beta
from typing import Tuple

from src.rl.ppo.actor import BaseActor
from src.rl.ppo.network_utils import build_mlp


class BetaActor(BaseActor):
    """Actor with Beta distribution — natural [0,1] support, no clipping bias.

    Actions mapped to [-1, 1] via 2*x - 1. The Jacobian is a constant (2^D)
    which cancels in the PPO ratio, so we skip the correction.
    """

    def __init__(
        self,
        device: torch.device,
        obs_dim: int = 14,
        action_dim: int = 3,
        hidden_sizes: tuple = (64, 64),
    ):
        super().__init__(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.encoder = build_mlp(obs_dim, list(hidden_sizes), output_dim=None)
        encoder_out = hidden_sizes[-1]

        # Alpha and beta heads
        self.alpha_head = nn.Linear(encoder_out, action_dim)
        self.beta_head = nn.Linear(encoder_out, action_dim)

        self.to(device)

    def _get_dist(self, obs: torch.Tensor) -> Beta:
        # Clamp obs to prevent NaN from extreme normalized values
        obs = torch.clamp(obs, -10.0, 10.0)
        h = self.encoder(obs)
        alpha = nn.functional.softplus(self.alpha_head(h)) + 1.0
        beta = nn.functional.softplus(self.beta_head(h)) + 1.0
        return Beta(alpha, beta)

    def forward(self, obs: torch.Tensor) -> Beta:
        return self._get_dist(obs)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._get_dist(obs)
        raw = dist.rsample()                         # [0, 1]
        action = 2.0 * raw - 1.0                     # [-1, 1]
        log_prob = dist.log_prob(raw).sum(-1)         # (batch,)
        return action, log_prob

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._get_dist(obs)
        raw = (actions + 1.0) / 2.0                   # map [-1,1] -> [0,1]
        raw = torch.clamp(raw, 1e-6, 1.0 - 1e-6)     # numerical safety
        log_prob = dist.log_prob(raw).sum(-1)          # (batch,)
        entropy = dist.entropy().sum(-1)               # (batch,)
        return log_prob, entropy

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean action for eval (no sampling)."""
        dist = self._get_dist(obs)
        raw = dist.mean                               # alpha / (alpha + beta)
        return 2.0 * raw - 1.0


class CentralizedCritic(nn.Module):
    """Centralized critic for MAPPO — sees actor obs + global state.

    Input: (batch, critic_obs_dim) where critic_obs_dim = actor_obs(14) + consensus_est(6) + log_P_eigs(3)
    Output: (batch,) scalar value
    """

    def __init__(
        self,
        device: torch.device,
        obs_dim: int = 23,
        hidden_sizes: tuple = (128, 128),
    ):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim

        self.net = build_mlp(obs_dim, list(hidden_sizes), output_dim=1)
        self.to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)  # (batch,)
