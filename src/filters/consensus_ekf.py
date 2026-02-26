"""Distributed Consensus EKF for cooperative multi-drone target tracking.

Each drone runs a local EKF in information form. After local measurement
updates, drones exchange information with neighbors via consensus averaging.
The N-scaling trick ensures consensus converges to the centralized solution.

Prediction is done in covariance form (avoids Q⁻¹ issues with near-singular
process noise at high control frequencies), then converted back to information form.
"""

import numpy as np
from .base import BayesianFilter
from .measurement import (
    bearing_measurement,
    bearing_jacobian,
    measurement_noise_cov,
    wrap_angle,
    initialize_filter_state,
)
from .topology import apply_dropout
from ..dynamics.target import ConstantVelocityModel


class ConsensusEKF(BayesianFilter):
    """Distributed consensus EKF with N local information-form filters."""

    def __init__(
        self,
        dt: float,
        sigma_a: float,
        sigma_bearing: float,
        range_ref: float,
        num_drones: int,
        adjacency: np.ndarray,
        num_consensus_iters: int = 5,
        consensus_step_size: float = 0.1,
        dropout_prob: float = 0.0,
        P0_pos: float = 10000.0,
        P0_vel: float = 100.0,
        rng: np.random.Generator | None = None,
        metropolis: bool = False,
    ):
        self._dt = dt
        self._model = ConstantVelocityModel(dt, sigma_a)
        self._sigma_bearing = sigma_bearing
        self._range_ref = range_ref
        self._N = num_drones
        self._adj = adjacency.copy()
        self._L = num_consensus_iters
        self._eps = consensus_step_size
        self._dropout = dropout_prob
        self._P0_pos = P0_pos
        self._P0_vel = P0_vel
        self._rng = rng or np.random.default_rng()
        self._metropolis = metropolis

        # Local information-form states: Y_i = P_i^{-1}, y_i = P_i^{-1} x_i
        self._Y = [np.zeros((6, 6)) for _ in range(self._N)]
        self._y = [np.zeros(6) for _ in range(self._N)]

        self._initialized = False
        # Last effective adjacency (for visualization of dropout)
        self._last_adj_eff = adjacency.copy()

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Initialize all local filters to the same state."""
        P0_inv = np.linalg.inv(P0)
        for i in range(self._N):
            self._Y[i] = P0_inv.copy()
            self._y[i] = P0_inv @ x0.copy()
        self._initialized = True

    def predict(self) -> None:
        if not self._initialized:
            return

        F = self._model.F()
        Q = self._model.Q()

        for i in range(self._N):
            # Convert info → covariance for prediction (avoids Q⁻¹)
            try:
                P_i = np.linalg.inv(self._Y[i])
            except np.linalg.LinAlgError:
                P_i = np.linalg.pinv(self._Y[i])

            x_i = P_i @ self._y[i]

            # CV prediction in covariance form
            x_i = F @ x_i
            P_i = F @ P_i @ F.T + Q

            # Convert back to information form
            try:
                Y_i = np.linalg.inv(P_i)
            except np.linalg.LinAlgError:
                Y_i = np.linalg.pinv(P_i)

            self._Y[i] = 0.5 * (Y_i + Y_i.T)  # symmetrize
            self._y[i] = self._Y[i] @ x_i

    def update(self, measurements: list, drone_positions: np.ndarray) -> None:
        # Auto-initialize on first update with sufficient measurements
        if not self._initialized:
            result = initialize_filter_state(
                measurements, drone_positions, self._P0_pos, self._P0_vel,
            )
            if result is not None:
                self.initialize(*result)
            return

        N = self._N

        # Stash raw info contributions for MH degree-based scaling
        dY_raw = [None] * N
        dy_raw = [None] * N

        # --- Local measurement update (information form) ---
        for i in range(N):
            # Get local state estimate for linearization
            try:
                P_i = np.linalg.inv(self._Y[i])
            except np.linalg.LinAlgError:
                P_i = np.linalg.pinv(self._Y[i])
            x_i = P_i @ self._y[i]

            if measurements[i] is not None:
                z_i = np.array(measurements[i])  # (2,)
                dp_i = drone_positions[i]  # (3,)

                H_i = bearing_jacobian(x_i, dp_i)  # (2, 6)
                R_i = measurement_noise_cov(dp_i, x_i[:3], self._sigma_bearing, self._range_ref)  # (2, 2)

                try:
                    R_inv = np.linalg.inv(R_i)
                except np.linalg.LinAlgError:
                    continue  # skip if R singular

                # Predicted measurement
                h_i = bearing_measurement(x_i, dp_i)  # (2,)

                # Innovation with angle wrapping
                innov = z_i - h_i
                innov[0] = wrap_angle(innov[0])  # azimuth

                # Information contributions
                dY_i = H_i.T @ R_inv @ H_i  # (6, 6)
                # Pseudo-measurement: z_i - h_i + H_i @ x_i (linearized)
                dy_i = H_i.T @ R_inv @ (innov + H_i @ x_i)  # (6,)

                if self._metropolis:
                    # Defer scaling — degree computed after dropout
                    dY_raw[i] = dY_i
                    dy_raw[i] = dy_i
                else:
                    # N-scaling: scale local contribution by N so consensus average = sum
                    self._Y[i] = self._Y[i] + N * dY_i
                    self._y[i] = self._y[i] + N * dy_i

        # --- Consensus iterations ---
        # Track which edges were active (union across all L iterations)
        adj_union = np.zeros_like(self._adj)
        for ell in range(self._L):
            # Apply dropout to get effective adjacency this iteration
            adj_eff = apply_dropout(self._adj, self._dropout, self._rng)
            adj_union = np.maximum(adj_union, adj_eff)

            if self._metropolis:
                # Compute per-node degree from effective adjacency
                degrees = np.sum(adj_eff, axis=1).astype(int)  # d_i

                # On first consensus iteration, apply degree-based measurement scaling
                if ell == 0:
                    for i in range(N):
                        if dY_raw[i] is not None:
                            scale = degrees[i] + 1  # d_i + 1
                            self._Y[i] = self._Y[i] + scale * dY_raw[i]
                            self._y[i] = self._y[i] + scale * dy_raw[i]

                # Snapshot current values (consensus uses old values)
                Y_old = [Yi.copy() for Yi in self._Y]
                y_old = [yi.copy() for yi in self._y]

                for i in range(N):
                    d_i = degrees[i]
                    for j in range(N):
                        if adj_eff[i, j]:
                            d_j = degrees[j]
                            w_ij = 1.0 / (1.0 + max(d_i, d_j))
                            self._Y[i] = self._Y[i] + w_ij * (Y_old[j] - Y_old[i])
                            self._y[i] = self._y[i] + w_ij * (y_old[j] - y_old[i])
            else:
                # Snapshot current values (consensus uses old values)
                Y_old = [Yi.copy() for Yi in self._Y]
                y_old = [yi.copy() for yi in self._y]

                for i in range(N):
                    for j in range(N):
                        if adj_eff[i, j]:
                            self._Y[i] = self._Y[i] + self._eps * (Y_old[j] - Y_old[i])
                            self._y[i] = self._y[i] + self._eps * (y_old[j] - y_old[i])

            # Symmetrize information matrices
            for i in range(N):
                self._Y[i] = 0.5 * (self._Y[i] + self._Y[i].T)

        self._last_adj_eff = adj_union

    def get_estimate(self) -> np.ndarray:
        """Return consensus average estimate across all drones."""
        if not self._initialized:
            return np.zeros(6)
        Y_avg = np.mean(self._Y, axis=0)
        y_avg = np.mean(self._y, axis=0)
        try:
            x_avg = np.linalg.solve(Y_avg, y_avg)
        except np.linalg.LinAlgError:
            x_avg = np.linalg.pinv(Y_avg) @ y_avg
        return x_avg

    def get_covariance(self) -> np.ndarray:
        """Return covariance from consensus average information matrix."""
        if not self._initialized:
            return np.eye(6) * self._P0_pos
        Y_avg = np.mean(self._Y, axis=0)
        try:
            P_avg = np.linalg.inv(Y_avg)
        except np.linalg.LinAlgError:
            P_avg = np.linalg.pinv(Y_avg)
        return 0.5 * (P_avg + P_avg.T)

    def get_local_estimate(self, i: int) -> np.ndarray:
        """Return drone i's local state estimate."""
        try:
            P_i = np.linalg.inv(self._Y[i])
        except np.linalg.LinAlgError:
            P_i = np.linalg.pinv(self._Y[i])
        return P_i @ self._y[i]

    def get_local_covariance(self, i: int) -> np.ndarray:
        """Return drone i's local covariance."""
        try:
            P_i = np.linalg.inv(self._Y[i])
        except np.linalg.LinAlgError:
            P_i = np.linalg.pinv(self._Y[i])
        return 0.5 * (P_i + P_i.T)

    def get_disagreement(self) -> float:
        """RMS spread of local estimates (consensus quality metric)."""
        if not self._initialized:
            return 0.0
        local_ests = np.array([self.get_local_estimate(i) for i in range(self._N)])
        mean_est = np.mean(local_ests, axis=0)
        deviations = local_ests - mean_est
        return np.sqrt(np.mean(np.sum(deviations**2, axis=1)))

    def get_active_edges(self) -> np.ndarray:
        """Return effective adjacency from last update (union over L iterations)."""
        return self._last_adj_eff.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def name(self) -> str:
        return "ConsensusEKF"
