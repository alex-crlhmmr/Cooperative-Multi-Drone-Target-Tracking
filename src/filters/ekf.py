"""Extended Kalman Filter for bearing-only target tracking.

Uses CV dynamics model with linearized bearing measurement model.
Joseph-form covariance update for numerical stability.
Auto-initializes from triangulation on first update with >= 2 measurements.
"""

import numpy as np
from .base import BayesianFilter
from .measurement import (
    stack_measurements, wrap_innovation, initialize_filter_state,
)
from ..dynamics.target import ConstantVelocityModel


class EKF(BayesianFilter):

    def __init__(
        self,
        dt: float,
        sigma_a: float,
        sigma_bearing: float,
        range_ref: float,
        P0_pos: float = 10000.0,
        P0_vel: float = 100.0,
    ):
        self._dt = dt
        self._model = ConstantVelocityModel(dt, sigma_a)
        self._sigma_bearing = sigma_bearing
        self._range_ref = range_ref
        self._P0_pos = P0_pos
        self._P0_vel = P0_vel

        self._x = np.zeros(6)
        self._P = np.eye(6) * P0_pos
        self._initialized = False

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self._x = x0.copy()
        self._P = P0.copy()
        self._initialized = True

    def predict(self) -> None:
        if not self._initialized:
            return
        F = self._model.F()
        Q = self._model.Q()
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def update(self, measurements: list, drone_positions: np.ndarray) -> None:
        # Auto-initialize on first update with sufficient measurements
        if not self._initialized:
            result = initialize_filter_state(
                measurements, drone_positions, self._P0_pos, self._P0_vel,
            )
            if result is not None:
                self.initialize(*result)
            return

        z, h, H, R, _ = stack_measurements(
            measurements, drone_positions, self._x,
            self._sigma_bearing, self._range_ref,
        )
        if len(z) == 0:
            return

        # Innovation with angle wrapping
        y = wrap_innovation(z - h)

        # Kalman gain
        with np.errstate(all="ignore"):
            S = H @ self._P @ H.T + R
            try:
                K = self._P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return  # skip update if singular

            # State update
            self._x = self._x + K @ y

            # Joseph-form covariance update: P = (I-KH)P(I-KH)' + KRK'
            I_KH = np.eye(6) - K @ H
            self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

        # Symmetrize
        self._P = 0.5 * (self._P + self._P.T)

    def get_estimate(self) -> np.ndarray:
        return self._x.copy()

    def get_covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def name(self) -> str:
        return "EKF"
