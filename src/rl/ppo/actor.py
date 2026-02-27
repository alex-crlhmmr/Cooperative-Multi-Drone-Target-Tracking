from abc import ABC, abstractmethod
from torch.distributions import Distribution
from typing import Tuple
import torch.nn as nn
import torch


class BaseActor(nn.Module, ABC):

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    @abstractmethod
    def act(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Given an observation, return an action and the log probability of that action."""
        pass

    @abstractmethod
    def forward(self, obs) -> Distribution:
        """ Given an observation, return a distribution over actions."""
        pass

    @abstractmethod
    def evaluate(self, obs, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given stored obs and actions, return log_probs and entropy under current policy."""
        pass
