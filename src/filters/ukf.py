"""Unscented Kalman Filter for bearing-only target tracking.

Sigma-point propagation through the nonlinear measurement model.
Handles the bearing nonlinearity better than EKF linearization.
Cholesky decomposition with regularization fallback.
"""

import numpy as np
from .base import BayesianFilter
from .measurement import (
    bearing_measurement, measurement_noise_cov, wrap_angle,
    wrap_innovation, initialize_filter_state,
)
from ..dynamics.target import ConstantVelocityModel


class UKF(BayesianFilter):

    def __init__(
        self,
        dt: float,
        sigma_a: float,
        sigma_bearing: float,
        range_ref: float,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        P0_pos: float = 10000.0,
        P0_vel: float = 100.0,
    ):
        self._dt = dt
        self._model = ConstantVelocityModel(dt, sigma_a)
        self._sigma_bearing = sigma_bearing
        self._range_ref = range_ref
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._P0_pos = P0_pos
        self._P0_vel = P0_vel

        self._n = 6  # state dimension
        self._x = np.zeros(self._n)
        self._P = np.eye(self._n) * P0_pos
        self._initialized = False

        # Compute sigma point weights
        self._compute_weights()

    def _compute_weights(self):
        n = self._n
        lam = self._alpha**2 * (n + self._kappa) - n
        self._lambda = lam

        self._Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self._Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self._Wm[0] = lam / (n + lam)
        self._Wc[0] = lam / (n + lam) + (1 - self._alpha**2 + self._beta)

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate 2n+1 sigma points."""
        n = self._n
        S = self._safe_cholesky(P * (n + self._lambda))
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + S[i]
            sigmas[n + i + 1] = x - S[i]
        return sigmas

    def _safe_cholesky(self, M: np.ndarray) -> np.ndarray:
        """Cholesky with regularization fallback."""
        M = 0.5 * (M + M.T)  # symmetrize
        try:
            return np.linalg.cholesky(M).T  # rows = sqrt(P) columns
        except np.linalg.LinAlgError:
            # Add small diagonal regularization
            eigvals = np.linalg.eigvalsh(M)
            eps = max(1e-6, -eigvals.min() + 1e-6)
            return np.linalg.cholesky(M + eps * np.eye(len(M))).T

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self._x = x0.copy()
        self._P = P0.copy()
        self._initialized = True

    def predict(self) -> None:
        if not self._initialized:
            return
        F = self._model.F()
        Q = self._model.Q()

        # Propagate sigma points through linear dynamics
        sigmas = self._sigma_points(self._x, self._P)
        sigmas_pred = (F @ sigmas.T).T  # (2n+1, 6)

        # Recover predicted mean and covariance
        self._x = self._Wm @ sigmas_pred
        diff = sigmas_pred - self._x
        self._P = (diff.T * self._Wc) @ diff + Q
        self._P = 0.5 * (self._P + self._P.T)

    def update(self, measurements: list, drone_positions: np.ndarray) -> None:
        if not self._initialized:
            result = initialize_filter_state(
                measurements, drone_positions, self._P0_pos, self._P0_vel,
            )
            if result is not None:
                self.initialize(*result)
            return

        # Collect valid measurements
        valid_idx = [i for i, m in enumerate(measurements) if m is not None]
        if len(valid_idx) == 0:
            return

        z_list = [np.array(measurements[i]) for i in valid_idx]
        z = np.concatenate(z_list)
        nz = len(z)  # 2k

        # Build R block-diagonal
        R_blocks = []
        for i in valid_idx:
            R_blocks.append(measurement_noise_cov(
                drone_positions[i], self._x[:3],
                self._sigma_bearing, self._range_ref,
            ))
        R = np.block([
            [R_blocks[i] if i == j else np.zeros((2, 2))
             for j in range(len(R_blocks))]
            for i in range(len(R_blocks))
        ])

        # Generate sigma points
        sigmas = self._sigma_points(self._x, self._P)
        n_sig = sigmas.shape[0]

        # Propagate sigma points through measurement model
        Z_sig = np.zeros((n_sig, nz))
        for s in range(n_sig):
            z_pred = []
            for i in valid_idx:
                z_pred.append(bearing_measurement(sigmas[s], drone_positions[i]))
            Z_sig[s] = np.concatenate(z_pred)

        # Predicted measurement mean (with angle wrapping)
        z_pred_mean = np.zeros(nz)
        # Use circular mean for azimuth components
        for dim in range(nz):
            if dim % 2 == 0:  # azimuth
                sin_sum = np.sum(self._Wm * np.sin(Z_sig[:, dim]))
                cos_sum = np.sum(self._Wm * np.cos(Z_sig[:, dim]))
                z_pred_mean[dim] = np.arctan2(sin_sum, cos_sum)
            else:  # elevation
                z_pred_mean[dim] = self._Wm @ Z_sig[:, dim]

        # Innovation covariance and cross-covariance
        Pzz = R.copy()
        Pxz = np.zeros((self._n, nz))

        for s in range(n_sig):
            dz = Z_sig[s] - z_pred_mean
            # Wrap azimuth differences
            for dim in range(0, nz, 2):
                dz[dim] = wrap_angle(dz[dim])
            dx = sigmas[s] - self._x
            Pzz += self._Wc[s] * np.outer(dz, dz)
            Pxz += self._Wc[s] * np.outer(dx, dz)

        # Kalman gain
        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            return

        # Innovation
        innov = z - z_pred_mean
        innov = wrap_innovation(innov)

        # State update
        self._x = self._x + K @ innov
        self._P = self._P - K @ Pzz @ K.T
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
        return "UKF"
