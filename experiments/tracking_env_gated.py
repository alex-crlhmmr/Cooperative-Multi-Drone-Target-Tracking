"""MultiDroneTrackingEnv with two-phase control.

Phase 1 (steps 0 to spread_steps): Heuristic spreading.
    Each drone flies away from swarm centroid at max speed.
    Guarantees angular diversity from any cluster orientation.

Phase 2 (steps spread_steps+): RL + gated baseline.
    Inverted gating blends RL with chase-offset baseline.
    RL optimizes geometry for tracking.

Gating (inverted, phase 2 only):
    rl_weight = clip(tr(P) / gate_threshold, min_rl_authority, max_rl_authority)
    High uncertainty → RL dominates
    Low uncertainty  → baseline dominates
"""
import numpy as np
from src.rl.ppo.tracking_env import MultiDroneTrackingEnv
from src.rl.ppo.tracking_config import TrackingConfig


class GatedTrackingEnv(MultiDroneTrackingEnv):
    """Two-phase tracking env: heuristic spread → RL + gated baseline."""

    def __init__(
        self,
        cfg: TrackingConfig,
        seed: int = 0,
        gate_threshold: float = 500.0,
        max_rl_authority: float = 1.0,
        min_rl_authority: float = 0.5,
        base_R_desired: float = 60.0,
        base_Kp: float = 2.0,
        augment_obs: bool = False,
        spread_steps: int = 200,
    ):
        super().__init__(cfg, seed=seed)
        self.gate_threshold = gate_threshold
        self.max_rl_authority = max_rl_authority
        self.min_rl_authority = min_rl_authority
        self.base_R_desired = base_R_desired
        self.base_Kp = base_Kp
        self.augment_obs = augment_obs
        self.spread_steps = spread_steps
        self._aug_angle = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.augment_obs:
            self._aug_angle = self._rng.uniform(0, 2 * np.pi)
        return obs, info

    def _rotate_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply random yaw rotation to all 3D vector components in obs."""
        if not self.augment_obs or self._aug_angle == 0.0:
            return obs
        c, s = np.cos(self._aug_angle), np.sin(self._aug_angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        obs_rot = obs.copy()
        for i in range(self.N):
            obs_rot[i, 0:3] = R @ obs[i, 0:3]
            obs_rot[i, 3:6] = R @ obs[i, 3:6]
            evecs = obs[i, 9:18].reshape(3, 3)
            obs_rot[i, 9:18] = (R @ evecs).flatten()
            for start in [19, 22, 25, 28]:
                obs_rot[i, start:start+3] = R @ obs[i, start:start+3]
        return obs_rot

    def _get_tr_P_pos(self) -> float:
        if not self._filter.initialized:
            return float("inf")
        P = self._filter.get_covariance()
        return float(np.trace(P[:3, :3]))

    def _compute_rl_weight(self) -> float:
        """INVERTED gating: high uncertainty → more RL authority."""
        tr_P = self._get_tr_P_pos()
        if not np.isfinite(tr_P):
            return self.max_rl_authority
        raw = tr_P / self.gate_threshold
        return float(np.clip(raw, self.min_rl_authority, self.max_rl_authority))

    def _compute_spread_velocity(self, drone_positions: np.ndarray) -> np.ndarray:
        """Phase 1: fly away from swarm centroid at max speed."""
        centroid = drone_positions.mean(axis=0)
        velocities = np.zeros((self.N, 3))

        for i in range(self.N):
            away = drone_positions[i] - centroid
            dist = np.linalg.norm(away)
            if dist > 1e-3:
                direction = away / dist
            else:
                # If exactly at centroid, pick a unique direction per drone
                angle = 2 * np.pi * i / self.N
                direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            velocities[i] = direction * self.cfg.v_max

        return velocities

    def _compute_base_velocity(self, drone_positions: np.ndarray) -> np.ndarray:
        """Chase-with-offset baseline."""
        base_vel = np.zeros((self.N, 3))

        if not self._filter.initialized:
            return base_vel

        for i in range(self.N):
            local_est = self._filter.get_local_estimate(i)
            est_pos = local_est[:3]
            est_vel = local_est[3:6]

            radial = drone_positions[i] - est_pos
            rdist = np.linalg.norm(radial)

            if rdist > 1e-3:
                rdir = radial / rdist
            else:
                rdir = self._rng.standard_normal(3)
                rdir /= np.linalg.norm(rdir) + 1e-8

            desired_pos = est_pos + rdir * self.base_R_desired

            vel = self.base_Kp * (desired_pos - drone_positions[i]) + est_vel
            speed = np.linalg.norm(vel)
            if speed > self.cfg.v_max:
                vel = vel / speed * self.cfg.v_max

            base_vel[i] = vel

        return base_vel

    def _compute_spread_reward(self, drone_positions: np.ndarray) -> np.ndarray:
        """Per-drone reward for inter-drone distance."""
        reward = np.zeros(self.N, dtype=np.float32)
        target_spread = self.base_R_desired

        for i in range(self.N):
            dists = []
            for j in range(self.N):
                if j != i:
                    dists.append(np.linalg.norm(drone_positions[i] - drone_positions[j]))
            mean_dist = np.mean(dists)
            reward[i] = np.exp(-((mean_dist - target_spread) / target_spread) ** 2)

        return reward

    def step(self, action: np.ndarray):
        """Two-phase step: heuristic spread → RL + gated baseline."""
        action = np.asarray(action, dtype=np.float32).reshape(self.N, 3)
        action = np.clip(action, -1.0, 1.0)

        # Inverse-rotate RL actions back to world frame
        if self.augment_obs and self._aug_angle != 0.0:
            c, s = np.cos(-self._aug_angle), np.sin(-self._aug_angle)
            R_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            action = (R_inv @ action.T).T

        self._step_count += 1

        drone_positions = np.array([
            self._aviary._getDroneStateVector(i)[:3] for i in range(self.N)
        ])

        # === PHASE 1: Heuristic spread (steps 0 to spread_steps) ===
        if self._step_count <= self.spread_steps:
            velocity = self._compute_spread_velocity(drone_positions)
            rl_weight = 0.0  # RL not active

        # === PHASE 2: RL + gated baseline ===
        else:
            rl_weight = self._compute_rl_weight()
            base_weight = 1.0 - rl_weight
            base_velocity = self._compute_base_velocity(drone_positions)
            rl_velocity = action * self.cfg.v_max
            velocity = base_weight * base_velocity + rl_weight * rl_velocity

        # Clamp total velocity
        for i in range(self.N):
            speed = np.linalg.norm(velocity[i])
            if speed > self.cfg.v_max:
                velocity[i] *= self.cfg.v_max / speed

        # Compute waypoints
        waypoints = drone_positions + velocity * self.cfg.dt
        waypoints[:, 2] = np.maximum(waypoints[:, 2], self.cfg.min_altitude)

        # Gimbal: point at per-drone local estimates
        per_drone_estimates = None
        if self._filter.initialized:
            per_drone_estimates = np.array([
                self._filter.get_local_estimate(i)[:3] for i in range(self.N)
            ])

        # Step physics
        result = self._aviary.step_tracking(
            waypoints, velocity, per_drone_estimates=per_drone_estimates
        )

        # Update filter
        drone_positions = np.array([
            self._aviary._getDroneStateVector(i)[:3] for i in range(self.N)
        ])
        measurements = result["measurements"]
        self._filter.predict()
        self._filter.update(measurements, drone_positions)

        # Build observation + rotation augmentation
        obs = self._build_obs(result)
        obs = self._rotate_obs(obs)

        # Reward: parent reward + spreading bonus
        base_reward = self._compute_reward(result)
        spread_reward = self._compute_spread_reward(drone_positions)
        reward = base_reward + 1.0 * spread_reward

        terminated = self._step_count >= self.cfg.episode_length
        truncated = False

        # Early termination
        if not terminated and self._step_count > self._early_term_grace:
            tr_P = self._get_tr_P_pos()
            if tr_P > self._early_term_threshold:
                self._early_term_counter += 1
                if self._early_term_counter >= self._early_term_patience:
                    truncated = True
            else:
                self._early_term_counter = 0

        info = {
            "result": result,
            "filter_initialized": self._filter.initialized,
            "rl_weight": rl_weight,
            "phase": "spread" if self._step_count <= self.spread_steps else "track",
        }
        if self._filter.initialized:
            P = self._filter.get_covariance()
            info["tr_P_pos"] = float(np.trace(P[:3, :3]))

        return obs, reward, terminated, truncated, info
