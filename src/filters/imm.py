"""Interacting Multiple Model (IMM) filter for bearing-only target tracking.

Runs M parallel EKF sub-filters with different process noise levels (CV models).
Model probabilities are updated each step based on measurement likelihoods.
Output is a probability-weighted blend of the sub-filter estimates.

Two modes: CV-gentle (low sigma_a) and CV-aggressive (high sigma_a).
"""

import numpy as np
from .base import BayesianFilter
from .measurement import (
    stack_measurements, wrap_innovation, initialize_filter_state,
)
from ..dynamics.target import ConstantVelocityModel


class IMM(BayesianFilter):
    """Interacting Multiple Model filter with M constant-velocity sub-filters."""

    def __init__(
        self,
        dt: float,
        sigma_a_modes: list[float],
        sigma_bearing: float,
        range_ref: float,
        transition_matrix: np.ndarray,
        P0_pos: float = 10000.0,
        P0_vel: float = 100.0,
    ):
        self._dt = dt
        self._M = len(sigma_a_modes)
        self._sigma_bearing = sigma_bearing
        self._range_ref = range_ref
        self._Pi = np.asarray(transition_matrix, dtype=float)  # (M, M)
        self._P0_pos = P0_pos
        self._P0_vel = P0_vel

        # Sub-filter dynamics models
        self._models = [ConstantVelocityModel(dt, sa) for sa in sigma_a_modes]

        # Sub-filter states (covariance form)
        self._x = [np.zeros(6) for _ in range(self._M)]
        self._P = [np.eye(6) * P0_pos for _ in range(self._M)]

        # Model probabilities (uniform prior)
        self._mu = np.ones(self._M) / self._M

        self._initialized = False

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        for j in range(self._M):
            self._x[j] = x0.copy()
            self._P[j] = P0.copy()
        self._mu = np.ones(self._M) / self._M
        self._initialized = True

    def predict(self) -> None:
        if not self._initialized:
            return

        M = self._M
        Pi = self._Pi  # Pi[i,j] = P(switch from model i to model j)

        # --- Mixing step ---
        # Mixing weights: c_j = sum_i Pi[i,j] * mu[i]
        c = Pi.T @ self._mu  # (M,)
        c = np.maximum(c, 1e-30)

        # Mixing probabilities: mu_mix[i,j] = Pi[i,j] * mu[i] / c[j]
        mu_mix = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                mu_mix[i, j] = Pi[i, j] * self._mu[i] / c[j]

        # Mixed initial conditions for each model j
        x_mixed = [np.zeros(6) for _ in range(M)]
        P_mixed = [np.zeros((6, 6)) for _ in range(M)]

        for j in range(M):
            # Mixed state
            for i in range(M):
                x_mixed[j] += mu_mix[i, j] * self._x[i]
            # Mixed covariance (with spread-of-means)
            for i in range(M):
                diff = self._x[i] - x_mixed[j]
                P_mixed[j] += mu_mix[i, j] * (self._P[i] + np.outer(diff, diff))

        # --- Per-model prediction ---
        for j in range(M):
            F = self._models[j].F()
            Q = self._models[j].Q()
            self._x[j] = F @ x_mixed[j]
            self._P[j] = F @ P_mixed[j] @ F.T + Q
            self._P[j] = 0.5 * (self._P[j] + self._P[j].T)

        # Store normalizing constants for probability update
        self._c = c

    def update(self, measurements: list, drone_positions: np.ndarray) -> None:
        if not self._initialized:
            result = initialize_filter_state(
                measurements, drone_positions, self._P0_pos, self._P0_vel,
            )
            if result is not None:
                self.initialize(*result)
            return

        M = self._M

        # Stack measurements once (shared across all sub-filters — we recompute
        # h and H per model since linearization point differs)
        # Check if any measurements are valid
        any_valid = any(m is not None for m in measurements)
        if not any_valid:
            return

        # Per-model EKF update + likelihood computation
        lambdas = np.ones(M)

        with np.errstate(all="ignore"):
            for j in range(M):
                z, h, H, R, _ = stack_measurements(
                    measurements, drone_positions, self._x[j],
                    self._sigma_bearing, self._range_ref,
                )
                if len(z) == 0:
                    continue

                y = wrap_innovation(z - h)

                # Innovation covariance
                S = H @ self._P[j] @ H.T + R
                S = 0.5 * (S + S.T)

                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    continue

                # Kalman gain
                K = self._P[j] @ H.T @ S_inv

                # State update
                self._x[j] = self._x[j] + K @ y

                # Joseph-form covariance update
                I_KH = np.eye(6) - K @ H
                self._P[j] = I_KH @ self._P[j] @ I_KH.T + K @ R @ K.T
                self._P[j] = 0.5 * (self._P[j] + self._P[j].T)

                # Measurement likelihood (multivariate Gaussian)
                n_z = len(z)
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0:
                    continue
                log_likelihood = -0.5 * (n_z * np.log(2 * np.pi) + logdet + y @ S_inv @ y)
                # Clamp to avoid overflow/underflow
                log_likelihood = np.clip(log_likelihood, -500, 500)
                lambdas[j] = np.exp(log_likelihood)

        # Model probability update
        mu_bar = lambdas * self._c
        total = np.sum(mu_bar)
        if total > 0:
            self._mu = mu_bar / total
        # else keep previous probabilities

    def get_estimate(self) -> np.ndarray:
        x_hat = np.zeros(6)
        for j in range(self._M):
            x_hat += self._mu[j] * self._x[j]
        return x_hat

    def get_covariance(self) -> np.ndarray:
        x_hat = self.get_estimate()
        P_hat = np.zeros((6, 6))
        for j in range(self._M):
            diff = self._x[j] - x_hat
            P_hat += self._mu[j] * (self._P[j] + np.outer(diff, diff))
        return 0.5 * (P_hat + P_hat.T)

    def get_mode_probabilities(self) -> np.ndarray:
        """Return (M,) vector of current model probabilities."""
        return self._mu.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def name(self) -> str:
        return "IMM"
