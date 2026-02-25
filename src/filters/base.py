"""Abstract base class for Bayesian filters."""

from abc import ABC, abstractmethod
import numpy as np


class BayesianFilter(ABC):
    """Common interface for EKF, UKF, and PF."""

    @abstractmethod
    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Set initial state estimate and covariance."""

    @abstractmethod
    def predict(self) -> None:
        """Propagate state forward one timestep using the dynamics model."""

    @abstractmethod
    def update(self, measurements: list, drone_positions: np.ndarray) -> None:
        """Fuse bearing measurements from all drones.

        Args:
            measurements: list of (az, el) tuples or None per drone
            drone_positions: (N, 3) array of drone positions
        """

    @abstractmethod
    def get_estimate(self) -> np.ndarray:
        """Return (6,) state estimate [px, py, pz, vx, vy, vz]."""

    @abstractmethod
    def get_covariance(self) -> np.ndarray:
        """Return (6, 6) state covariance matrix."""

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """Whether the filter has been initialized."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Filter name for display."""
