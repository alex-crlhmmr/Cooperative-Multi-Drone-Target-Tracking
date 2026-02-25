"""Particle Filter for bearing-only target tracking.

Key advantages over EKF/UKF:
- Ray-spread initialization handles range ambiguity naturally
- No linearization — handles measurement nonlinearity exactly
- Robust to non-Gaussian posteriors and model mismatch

Uses systematic resampling when effective sample size drops below threshold.
Log-space weight computation for numerical stability.
"""

import numpy as np
from .base import BayesianFilter
from .measurement import (
    bearing_measurement, wrap_angle, triangulate_position,
)
from ..dynamics.target import ConstantVelocityModel


class PF(BayesianFilter):

    def __init__(
        self,
        dt: float,
        sigma_a: float,
        sigma_bearing: float,
        range_ref: float,
        num_particles: int = 500,
        resample_threshold: float = 0.5,
        range_min: float = 10.0,
        range_max: float = 500.0,
        P0_pos: float = 10000.0,
        P0_vel: float = 100.0,
        process_noise_factor: float = 5.0,
        jitter_pos: float = 1.0,
        jitter_vel: float = 0.5,
    ):
        self._dt = dt
        # PF uses inflated process noise to prevent particle degeneracy.
        self._model = ConstantVelocityModel(dt, sigma_a * process_noise_factor)
        self._true_model = ConstantVelocityModel(dt, sigma_a)  # for covariance floor
        # Direct position/velocity jitter per step (prevents collapse at high ctrl_freq).
        # At 48 Hz the CV model's Q gives dt^4 ~ 10^-7 m² position noise per step,
        # far too small for particles to track a 10 m/s target. We add independent
        # noise to keep the particle cloud alive.
        self._jitter_pos = jitter_pos
        self._jitter_vel = jitter_vel
        self._sigma_bearing = sigma_bearing
        self._range_ref = range_ref
        self._N = num_particles
        self._resample_thresh = resample_threshold
        self._range_min = range_min
        self._range_max = range_max
        self._P0_pos = P0_pos
        self._P0_vel = P0_vel

        self._particles = np.zeros((num_particles, 6))
        self._weights = np.ones(num_particles) / num_particles
        self._initialized = False
        self._rng = np.random.default_rng()

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Initialize particles from Gaussian around x0."""
        self._particles = self._rng.multivariate_normal(x0, P0, size=self._N)
        self._weights = np.ones(self._N) / self._N
        self._initialized = True

    def initialize_from_rays(
        self,
        measurements: list,
        drone_positions: np.ndarray,
    ) -> bool:
        """Spread particles along bearing rays from drones.

        For each valid measurement, sample particles at random ranges
        along the ray. This captures bearing-only range ambiguity.
        """
        valid = [(i, m) for i, m in enumerate(measurements) if m is not None]
        if len(valid) < 2:
            return False

        # Get rough center from triangulation
        center = triangulate_position(measurements, drone_positions)
        if center is None:
            return False

        # Distribute particles: some along rays, some around triangulated center
        n_center = self._N // 3
        n_rays = self._N - n_center

        particles = np.zeros((self._N, 6))

        # Particles around triangulated center
        pos_std = 50.0
        vel_std = 5.0
        for i in range(n_center):
            particles[i, :3] = center + self._rng.normal(0, pos_std, 3)
            particles[i, 3:] = self._rng.normal(0, vel_std, 3)

        # Particles along bearing rays
        rays_per_drone = n_rays // len(valid)
        idx = n_center
        for di, (drone_i, m) in enumerate(valid):
            az, el = m
            direction = np.array([
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ])
            dp = drone_positions[drone_i]
            n_this = rays_per_drone if di < len(valid) - 1 else (self._N - idx)

            for j in range(n_this):
                if idx >= self._N:
                    break
                r = self._rng.uniform(self._range_min, self._range_max)
                # Add angular noise
                az_noisy = az + self._rng.normal(0, self._sigma_bearing * 2)
                el_noisy = el + self._rng.normal(0, self._sigma_bearing * 2)
                d_noisy = np.array([
                    np.cos(el_noisy) * np.cos(az_noisy),
                    np.cos(el_noisy) * np.sin(az_noisy),
                    np.sin(el_noisy),
                ])
                particles[idx, :3] = dp + r * d_noisy
                particles[idx, 3:] = self._rng.normal(0, vel_std, 3)
                idx += 1

        self._particles = particles
        self._weights = np.ones(self._N) / self._N
        self._initialized = True
        return True

    def predict(self) -> None:
        if not self._initialized:
            return
        F = self._model.F()
        Q = self._model.Q()

        # Skip if particles contain NaN (can happen before proper initialization)
        if np.any(np.isnan(self._particles)):
            return

        # Propagate through dynamics + model process noise
        with np.errstate(all="ignore"):
            try:
                noise = self._rng.multivariate_normal(np.zeros(6), Q, size=self._N)
            except (np.linalg.LinAlgError, ValueError):
                noise = np.zeros((self._N, 6))
            self._particles = (F @ self._particles.T).T + noise

        # Add direct jitter to prevent particle collapse at high frequencies
        if self._jitter_pos > 0:
            self._particles[:, :3] += self._rng.normal(0, self._jitter_pos, (self._N, 3))
        if self._jitter_vel > 0:
            self._particles[:, 3:] += self._rng.normal(0, self._jitter_vel, (self._N, 3))

    def update(self, measurements: list, drone_positions: np.ndarray) -> None:
        if not self._initialized:
            if self.initialize_from_rays(measurements, drone_positions):
                return
            return

        valid = [(i, m) for i, m in enumerate(measurements) if m is not None]
        if len(valid) == 0:
            return

        # Log-likelihood for each particle
        log_weights = np.log(self._weights + 1e-300)

        for drone_i, m in valid:
            z = np.array(m)
            dp = drone_positions[drone_i]
            r_drone = np.linalg.norm(self._particles[:, :3] - dp, axis=1)
            r_drone = np.maximum(r_drone, 1e-6)
            sigma = self._sigma_bearing * (r_drone / self._range_ref)
            sigma = np.maximum(sigma, 1e-6)

            for p_idx in range(self._N):
                z_pred = bearing_measurement(self._particles[p_idx], dp)
                daz = wrap_angle(z[0] - z_pred[0])
                del_ = z[1] - z_pred[1]
                s = sigma[p_idx]
                # 2D Gaussian log-likelihood: -0.5*||dz||²/σ² - 2*log(σ)
                log_lik = -0.5 * (daz**2 + del_**2) / s**2 - 2.0 * np.log(s)
                log_weights[p_idx] += log_lik

        # Normalize in log space
        max_lw = np.max(log_weights)
        log_weights -= max_lw
        weights = np.exp(log_weights)
        w_sum = weights.sum()
        if w_sum < 1e-300:
            # All weights collapsed — reinitialize
            self._weights = np.ones(self._N) / self._N
            return
        self._weights = weights / w_sum

        # Resample if needed
        n_eff = 1.0 / np.sum(self._weights**2)
        if n_eff < self._resample_thresh * self._N:
            self._systematic_resample()

    def _systematic_resample(self):
        """Systematic resampling with roughening.

        After resampling, add small jitter proportional to the particle spread
        to prevent covariance collapse from duplicate particles.
        """
        N = self._N
        positions = (np.arange(N) + self._rng.random()) / N
        cumsum = np.cumsum(self._weights)
        cumsum[-1] = 1.0  # ensure exact sum

        indices = np.searchsorted(cumsum, positions)
        self._particles = self._particles[indices].copy()
        self._weights = np.ones(N) / N

        # Roughening: add small jitter to break particle degeneracy
        # Scale proportional to particle spread (empirical bandwidth)
        spread = np.std(self._particles, axis=0)
        K = 0.1  # roughening factor (typical: 0.05-0.2)
        jitter_std = K * spread * N ** (-1.0 / 6)  # optimal bandwidth for 6D
        self._particles += self._rng.normal(0, 1, self._particles.shape) * jitter_std

    def get_estimate(self) -> np.ndarray:
        """Weighted mean of particles."""
        if not self._initialized:
            return self._particles[0].copy()  # placeholder
        return self._weights @ self._particles

    def get_covariance(self) -> np.ndarray:
        """Weighted scatter matrix of particles with floor.

        Particle scatter underestimates covariance after resampling
        (duplicate particles). We floor at the process noise Q to
        prevent unrealistically small covariance estimates.
        """
        if not self._initialized:
            return np.eye(6) * self._P0_pos
        mean = self.get_estimate()
        diff = self._particles - mean
        with np.errstate(all="ignore"):
            P = (diff.T * self._weights) @ diff
        if not np.all(np.isfinite(P)):
            return np.eye(6) * self._P0_pos
        # Floor: covariance can't be smaller than one step of true process noise
        Q = self._true_model.Q()
        P = np.maximum(P, Q)
        return 0.5 * (P + P.T)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def name(self) -> str:
        return "PF"
