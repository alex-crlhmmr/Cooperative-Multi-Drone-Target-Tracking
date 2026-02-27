"""Neural networks for RL-based active tracking.

TrackingActor: encodes ego features + neighbor features, outputs squashed Gaussian actions.
TrackingCritic: same encoder architecture (separate weights), outputs scalar value.
NeighborEncoder: permutation-invariant pooling over neighbor relative positions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torch.distributions import Normal

from .network_utils import build_mlp
from .actor import BaseActor


class NeighborEncoder(nn.Module):
    """Permutation-invariant encoder for neighbor relative positions.

    Input: (batch, num_neighbors, feat_dim)
    Output: (batch, 2 * embed_dim) via mean + max pooling.
    """

    def __init__(self, feat_dim: int, hidden_sizes: tuple, embed_dim: int):
        super().__init__()
        self.encoder = build_mlp(feat_dim, list(hidden_sizes), embed_dim)
        self.output_dim = 2 * embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_neighbors, feat_dim)
        Returns:
            (batch, 2 * embed_dim)
        """
        # Encode each neighbor
        encoded = self.encoder(x)  # (batch, num_neighbors, embed_dim)
        # Pooling
        mean_pool = encoded.mean(dim=1)  # (batch, embed_dim)
        max_pool = encoded.max(dim=1).values  # (batch, embed_dim)
        return torch.cat([mean_pool, max_pool], dim=-1)


class TrackingActor(BaseActor):
    """Actor for multi-drone tracking with tanh-squashed Gaussian policy.

    Observation layout (31D per drone):
        [0:19]  ego features (rel_pos, vel_est, eigenvals, eigvecs, detect_flag)
        [19:31] neighbor features (mean, max, min, std of relative positions)
    """

    EGO_DIM = 19
    NEIGHBOR_DIM = 12  # 4 stats x 3D

    def __init__(
        self,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
        ego_hidden: tuple = (128, 128),
        neighbor_hidden: tuple = (64, 64),
        neighbor_embed_dim: int = 32,
        policy_hidden: tuple = (128, 128),
    ):
        super().__init__(device)
        self.action_dim = action_dim

        # Ego encoder
        self.ego_encoder = build_mlp(self.EGO_DIM, list(ego_hidden), ego_hidden[-1])

        # Neighbor encoder (processes pre-pooled stats, not raw neighbors)
        self.neighbor_encoder = build_mlp(
            self.NEIGHBOR_DIM, list(neighbor_hidden), neighbor_embed_dim
        )

        # Policy head
        combined_dim = ego_hidden[-1] + neighbor_embed_dim
        self.policy_net = build_mlp(combined_dim, list(policy_hidden), action_dim)

        # Scale up the output layer so initial mean actions are ~0.5 not ~0.01
        with torch.no_grad():
            last_layer = list(self.policy_net.children())[-1]
            if hasattr(last_layer, 'weight'):
                last_layer.weight.mul_(5.0)
                if last_layer.bias is not None:
                    last_layer.bias.mul_(5.0)

        # State-independent log_std (init -0.5 → std≈0.6 for moderate exploration)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, device=device))

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation into feature vector.

        Args:
            obs: (batch, obs_dim) or (obs_dim,)
        Returns:
            (batch, combined_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        ego = obs[:, :self.EGO_DIM]
        neighbor = obs[:, self.EGO_DIM:]

        ego_feat = self.ego_encoder(ego)
        neighbor_feat = self.neighbor_encoder(neighbor)

        return torch.cat([ego_feat, neighbor_feat], dim=-1)

    def forward(self, obs: torch.Tensor) -> Normal:
        """Return Gaussian distribution (pre-squashing)."""
        features = self._encode(obs)
        mean = self.policy_net(features)
        # Clamp log_std to prevent entropy explosion or collapse
        log_std = torch.clamp(self.log_std, -3.0, 0.5)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with tanh squashing and corrected log_prob.

        Returns:
            action: (batch, action_dim) in [-1, 1]
            log_prob: (batch,) corrected for tanh Jacobian
        """
        dist = self.forward(obs)
        raw = dist.rsample()

        # Tanh squashing
        action = torch.tanh(raw)

        # Log prob with Jacobian correction: log_prob -= sum(log(1 - tanh^2(raw)))
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate stored actions under current policy.

        Args:
            obs: (batch, obs_dim)
            actions: (batch, action_dim) squashed actions in [-1, 1]

        Returns:
            log_probs: (batch,)
            entropy: (batch,)
        """
        dist = self.forward(obs)

        # Inverse squashing: raw = atanh(action)
        clamped = torch.clamp(actions, -0.999, 0.999)
        raw = torch.atanh(clamped)

        # Log prob with Jacobian correction
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1)

        # Entropy of the squashed distribution (approximate using base entropy)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class TrackingCritic(nn.Module):
    """Critic for multi-drone tracking.

    Same encoder architecture as actor (separate weights), outputs scalar value.
    """

    EGO_DIM = 19
    NEIGHBOR_DIM = 12

    def __init__(
        self,
        device: torch.device,
        obs_dim: int,
        ego_hidden: tuple = (128, 128),
        neighbor_hidden: tuple = (64, 64),
        neighbor_embed_dim: int = 32,
        value_hidden: tuple = (128, 128),
    ):
        super().__init__()
        self.device = device

        # Ego encoder
        self.ego_encoder = build_mlp(self.EGO_DIM, list(ego_hidden), ego_hidden[-1])

        # Neighbor encoder
        self.neighbor_encoder = build_mlp(
            self.NEIGHBOR_DIM, list(neighbor_hidden), neighbor_embed_dim
        )

        # Value head
        combined_dim = ego_hidden[-1] + neighbor_embed_dim
        self.value_net = build_mlp(combined_dim, list(value_hidden), 1)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        ego = obs[:, :self.EGO_DIM]
        neighbor = obs[:, self.EGO_DIM:]

        ego_feat = self.ego_encoder(ego)
        neighbor_feat = self.neighbor_encoder(neighbor)

        return torch.cat([ego_feat, neighbor_feat], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate.

        Args:
            obs: (batch, obs_dim)
        Returns:
            (batch,) values
        """
        features = self._encode(obs)
        return self.value_net(features).squeeze(-1)
