"""Distributed Consensus IMM for cooperative multi-drone target tracking.

Each drone runs a local IMM filter (multiple CV models with different process
noise). After local IMM predict+update, the blended output is converted to
information form for consensus averaging across neighbors.

This combines the multi-model adaptivity of IMM with the distributed fusion
of consensus filtering.
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


class ConsensusIMM(BayesianFilter):
    """Distributed consensus IMM with N local IMM filters."""

    def __init__(
        self,
        dt: float,
        sigma_a_modes: list[float],
        sigma_bearing: float,
        range_ref: float,
        transition_matrix: np.ndarray,
        num_drones: int,
        adjacency: np.ndarray,
        num_consensus_iters: int = 5,
        consensus_step_size: float = 0.1,
        dropout_prob: float = 0.0,
        P0_pos: float = 10000.0,
        P0_vel: float = 100.0,
        rng: np.random.Generator | None = None,
    ):
        self._dt = dt
        self._M = len(sigma_a_modes)
        self._sigma_bearing = sigma_bearing
        self._range_ref = range_ref
        self._Pi = np.asarray(transition_matrix, dtype=float)
        self._N = num_drones
        self._adj = adjacency.copy()
        self._L = num_consensus_iters
        self._eps = consensus_step_size
        self._dropout = dropout_prob
        self._P0_pos = P0_pos
        self._P0_vel = P0_vel
        self._rng = rng or np.random.default_rng()

        # Per-drone sub-filter models
        self._models = [ConstantVelocityModel(dt, sa) for sa in sigma_a_modes]

        # Per-drone, per-mode states (covariance form for IMM operations)
        # _xm[i][j] = state of drone i, mode j
        self._xm = [[np.zeros(6) for _ in range(self._M)] for _ in range(self._N)]
        self._Pm = [[np.eye(6) * P0_pos for _ in range(self._M)] for _ in range(self._N)]

        # Per-drone model probabilities
        self._mu = [np.ones(self._M) / self._M for _ in range(self._N)]

        # Per-drone consensus info-form state (post-IMM blended output)
        self._Y = [np.zeros((6, 6)) for _ in range(self._N)]
        self._y = [np.zeros(6) for _ in range(self._N)]

        self._initialized = False
        self._last_adj_eff = adjacency.copy()

        # Store mixing normalizers per drone
        self._c = [np.ones(self._M) / self._M for _ in range(self._N)]

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        P0_inv = np.linalg.inv(P0)
        for i in range(self._N):
            for j in range(self._M):
                self._xm[i][j] = x0.copy()
                self._Pm[i][j] = P0.copy()
            self._mu[i] = np.ones(self._M) / self._M
            self._Y[i] = P0_inv.copy()
            self._y[i] = P0_inv @ x0.copy()
        self._initialized = True

    def predict(self) -> None:
        if not self._initialized:
            return

        M = self._M
        Pi = self._Pi

        for i in range(self._N):
            # --- Reconstruct sub-filter states from consensus info form ---
            # After consensus, the blended info (Y[i], y[i]) represents the
            # fused estimate. We need to update sub-filter states to reflect
            # the consensus result before doing IMM mixing.
            #
            # Strategy: shift all sub-filter means by the consensus correction,
            # and replace covariances with the consensus covariance.
            try:
                P_cons = np.linalg.inv(self._Y[i])
            except np.linalg.LinAlgError:
                P_cons = np.linalg.pinv(self._Y[i])
            x_cons = P_cons @ self._y[i]

            # Compute current blended IMM estimate for this drone
            x_blend = np.zeros(6)
            for j in range(M):
                x_blend += self._mu[i][j] * self._xm[i][j]

            # Shift sub-filter states by consensus correction
            correction = x_cons - x_blend
            for j in range(M):
                self._xm[i][j] = self._xm[i][j] + correction

            # --- IMM mixing ---
            c = Pi.T @ self._mu[i]
            c = np.maximum(c, 1e-30)
            self._c[i] = c

            mu_mix = np.zeros((M, M))
            for mi in range(M):
                for mj in range(M):
                    mu_mix[mi, mj] = Pi[mi, mj] * self._mu[i][mi] / c[mj]

            x_mixed = [np.zeros(6) for _ in range(M)]
            P_mixed = [np.zeros((6, 6)) for _ in range(M)]

            for j in range(M):
                for mi in range(M):
                    x_mixed[j] += mu_mix[mi, j] * self._xm[i][mi]
                for mi in range(M):
                    diff = self._xm[i][mi] - x_mixed[j]
                    P_mixed[j] += mu_mix[mi, j] * (self._Pm[i][mi] + np.outer(diff, diff))

            # --- Per-model prediction ---
            for j in range(M):
                F = self._models[j].F()
                Q = self._models[j].Q()
                self._xm[i][j] = F @ x_mixed[j]
                self._Pm[i][j] = F @ P_mixed[j] @ F.T + Q
                self._Pm[i][j] = 0.5 * (self._Pm[i][j] + self._Pm[i][j].T)

            # Update info-form prediction from blended IMM output
            x_blend_pred = np.zeros(6)
            for j in range(M):
                x_blend_pred += self._mu[i][j] * self._xm[i][j]
            P_blend_pred = np.zeros((6, 6))
            for j in range(M):
                diff = self._xm[i][j] - x_blend_pred
                P_blend_pred += self._mu[i][j] * (self._Pm[i][j] + np.outer(diff, diff))
            P_blend_pred = 0.5 * (P_blend_pred + P_blend_pred.T)

            try:
                Y_pred = np.linalg.inv(P_blend_pred)
            except np.linalg.LinAlgError:
                Y_pred = np.linalg.pinv(P_blend_pred)

            self._Y[i] = 0.5 * (Y_pred + Y_pred.T)
            self._y[i] = self._Y[i] @ x_blend_pred

    def update(self, measurements: list, drone_positions: np.ndarray) -> None:
        if not self._initialized:
            result = initialize_filter_state(
                measurements, drone_positions, self._P0_pos, self._P0_vel,
            )
            if result is not None:
                self.initialize(*result)
            return

        N = self._N
        M = self._M

        # --- Per-drone local IMM update ---
        for i in range(N):
            if measurements[i] is None:
                continue

            z_i = np.array(measurements[i])  # (2,)
            dp_i = drone_positions[i]  # (3,)

            lambdas = np.ones(M)

            for j in range(M):
                x_j = self._xm[i][j]
                P_j = self._Pm[i][j]

                H = bearing_jacobian(x_j, dp_i)  # (2, 6)
                R = measurement_noise_cov(dp_i, x_j[:3], self._sigma_bearing, self._range_ref)
                h = bearing_measurement(x_j, dp_i)  # (2,)

                innov = z_i - h
                innov[0] = wrap_angle(innov[0])

                S = H @ P_j @ H.T + R
                S = 0.5 * (S + S.T)

                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    continue

                K = P_j @ H.T @ S_inv
                self._xm[i][j] = x_j + K @ innov

                I_KH = np.eye(6) - K @ H
                self._Pm[i][j] = I_KH @ P_j @ I_KH.T + K @ R @ K.T
                self._Pm[i][j] = 0.5 * (self._Pm[i][j] + self._Pm[i][j].T)

                # Likelihood
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0:
                    continue
                log_lik = -0.5 * (2 * np.log(2 * np.pi) + logdet + innov @ S_inv @ innov)
                log_lik = np.clip(log_lik, -500, 500)
                lambdas[j] = np.exp(log_lik)

            # Model probability update
            mu_bar = lambdas * self._c[i]
            total = np.sum(mu_bar)
            if total > 0:
                self._mu[i] = mu_bar / total

            # --- Convert blended IMM output to info form with N-scaling ---
            # Compute information contribution from this drone's measurement
            # Using the blended estimate for linearization
            x_blend = np.zeros(6)
            for j in range(M):
                x_blend += self._mu[i][j] * self._xm[i][j]

            H_blend = bearing_jacobian(x_blend, dp_i)
            R_blend = measurement_noise_cov(dp_i, x_blend[:3], self._sigma_bearing, self._range_ref)

            try:
                R_inv = np.linalg.inv(R_blend)
            except np.linalg.LinAlgError:
                continue

            h_blend = bearing_measurement(x_blend, dp_i)
            innov_blend = z_i - h_blend
            innov_blend[0] = wrap_angle(innov_blend[0])

            dY = H_blend.T @ R_inv @ H_blend
            dy = H_blend.T @ R_inv @ (innov_blend + H_blend @ x_blend)

            # N-scaling for consensus
            self._Y[i] = self._Y[i] + N * dY
            self._y[i] = self._y[i] + N * dy

        # --- Consensus iterations ---
        adj_union = np.zeros_like(self._adj)
        for _ in range(self._L):
            adj_eff = apply_dropout(self._adj, self._dropout, self._rng)
            adj_union = np.maximum(adj_union, adj_eff)

            Y_old = [Yi.copy() for Yi in self._Y]
            y_old = [yi.copy() for yi in self._y]

            for i in range(N):
                for j_nbr in range(N):
                    if adj_eff[i, j_nbr]:
                        self._Y[i] = self._Y[i] + self._eps * (Y_old[j_nbr] - Y_old[i])
                        self._y[i] = self._y[i] + self._eps * (y_old[j_nbr] - y_old[i])

            for i in range(N):
                self._Y[i] = 0.5 * (self._Y[i] + self._Y[i].T)

        self._last_adj_eff = adj_union

    def get_estimate(self) -> np.ndarray:
        if not self._initialized:
            return np.zeros(6)
        Y_avg = np.mean(self._Y, axis=0)
        y_avg = np.mean(self._y, axis=0)
        try:
            return np.linalg.solve(Y_avg, y_avg)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(Y_avg) @ y_avg

    def get_covariance(self) -> np.ndarray:
        if not self._initialized:
            return np.eye(6) * self._P0_pos
        Y_avg = np.mean(self._Y, axis=0)
        try:
            P_avg = np.linalg.inv(Y_avg)
        except np.linalg.LinAlgError:
            P_avg = np.linalg.pinv(Y_avg)
        return 0.5 * (P_avg + P_avg.T)

    def get_local_estimate(self, i: int) -> np.ndarray:
        try:
            P_i = np.linalg.inv(self._Y[i])
        except np.linalg.LinAlgError:
            P_i = np.linalg.pinv(self._Y[i])
        return P_i @ self._y[i]

    def get_local_covariance(self, i: int) -> np.ndarray:
        try:
            P_i = np.linalg.inv(self._Y[i])
        except np.linalg.LinAlgError:
            P_i = np.linalg.pinv(self._Y[i])
        return 0.5 * (P_i + P_i.T)

    def get_disagreement(self) -> float:
        if not self._initialized:
            return 0.0
        local_ests = np.array([self.get_local_estimate(i) for i in range(self._N)])
        mean_est = np.mean(local_ests, axis=0)
        deviations = local_ests - mean_est
        return np.sqrt(np.mean(np.sum(deviations**2, axis=1)))

    def get_active_edges(self) -> np.ndarray:
        return self._last_adj_eff.copy()

    def get_mode_probabilities(self) -> np.ndarray:
        """Return (N, M) array of mode probabilities per drone."""
        return np.array([mu.copy() for mu in self._mu])

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def name(self) -> str:
        return "ConsensusIMM"
