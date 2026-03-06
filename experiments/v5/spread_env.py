"""V5 Phase 1: SpreadEnv — FIM-based reward for optimal bearing geometry.

Key change from V4: reward is FIM-based (log det + difference FIM + range bonus)
instead of nearest-neighbor distance. Directly optimizes for tracking quality.
"""

import gymnasium as gym
import numpy as np
from itertools import combinations
from gymnasium import spaces

from src.env.tracking_aviary import TrackingAviary
from .config import V5Config
from .fim import compute_bearing_fim, compute_difference_fim_rewards


class SpreadEnv(gym.Env):
    """Phase 1 environment: spread drones from cluster using FIM reward.

    Actor obs (13D per drone): bearing, neighbor info, separation metrics, time
    Critic obs (19D per drone): actor obs + swarm centroid + target bearing
    Action (3D per drone): velocity command (full RL or residual on repulsion)
    Reward: FIM-based (shared log_det + per-agent difference + range bonus)
    """

    def __init__(self, cfg: V5Config, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.num_drones
        self._seed = seed or cfg.seed
        self._rng = np.random.default_rng(self._seed)

        self.actor_obs_dim = cfg.spread_obs_dim          # 13
        self.critic_obs_dim = cfg.spread_critic_obs_dim  # 19
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.N, self.critic_obs_dim), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.N, 3), dtype=np.float32,
        )

        self._aviary = None
        self._step_count = 0
        self._initial_target_bearing = None
        self._target_pos_init = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        if self._aviary is not None:
            self._aviary.close()

        target_pos = np.array([0.0, 0.0, 50.0])
        self._target_pos_init = target_pos.copy()

        # Cluster spawn: all drones within +/-2m of a point 100m from target
        direction = self._rng.standard_normal(3)
        direction[2] = abs(direction[2])
        direction /= np.linalg.norm(direction) + 1e-8
        cluster_center = target_pos + direction * 100.0
        cluster_center[2] = max(cluster_center[2], self.cfg.min_altitude + 10.0)
        positions = np.array([
            cluster_center + self._rng.uniform(-2, 2, size=3)
            for _ in range(self.N)
        ])
        positions[:, 2] = np.maximum(positions[:, 2], self.cfg.min_altitude)

        self._aviary = TrackingAviary(
            num_trackers=self.N,
            tracker_positions=positions,
            target_initial_pos=target_pos,
            target_speed=self.cfg.target_speed,
            target_trajectory=self.cfg.target_trajectory,
            target_sigma_a=self.cfg.target_sigma_a,
            episode_length=self.cfg.spread_steps,
            sensor_config=self.cfg.sensor_config,
            pyb_freq=self.cfg.pyb_freq,
            ctrl_freq=self.cfg.ctrl_freq,
            gui=False,
            rng=self._rng,
        )
        self._aviary.reset()

        drone_positions = self._get_drone_positions()
        self._initial_target_bearing = self._compute_bearings(
            drone_positions, target_pos
        )

        result = self._aviary.step_tracking(
            tracker_targets=drone_positions,
            tracker_target_vels=np.zeros((self.N, 3)),
        )
        self._last_result = result
        self._step_count = 0

        obs = self._build_obs(result)
        return obs, {"result": result}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.N, 3)
        action = np.clip(action, -1.0, 1.0)

        drone_positions = self._get_drone_positions()

        if self.cfg.spread_control == "residual_repulsion":
            base_vel = self._repulsion_base(drone_positions)
            correction = action * self.cfg.v_max * self.cfg.spread_residual_scale
            velocity = base_vel + correction
        else:  # "full_rl"
            velocity = action * self.cfg.v_max

        for i in range(self.N):
            speed = np.linalg.norm(velocity[i])
            if speed > self.cfg.v_max:
                velocity[i] *= self.cfg.v_max / speed

        dt = self.cfg.dt
        waypoints = drone_positions + velocity * dt
        waypoints[:, 2] = np.maximum(waypoints[:, 2], self.cfg.min_altitude)

        result = self._aviary.step_tracking(
            tracker_targets=waypoints,
            tracker_target_vels=velocity,
        )
        self._last_result = result

        if result.get("done", False) and "drone_positions" not in result:
            obs = np.zeros((self.N, self.critic_obs_dim), dtype=np.float32)
            reward = np.full(self.N, 0.0, dtype=np.float32)
            return obs, reward, True, False, {"result": result}

        obs = self._build_obs(result)
        reward = self._compute_reward(result)

        self._step_count += 1
        terminated = result.get("done", False)
        truncated = self._step_count >= self.cfg.spread_steps

        info = {
            "result": result,
            "min_angular_sep": self._last_min_angle,
            "mean_spread_dist": self._last_mean_dist,
            "log_det_fim": self._last_log_det_fim,
        }
        return obs, reward, terminated, truncated, info

    def _repulsion_base(self, drone_positions: np.ndarray) -> np.ndarray:
        """Fly away from nearest neighbor."""
        base_vel = np.zeros((self.N, 3))
        for i in range(self.N):
            min_dist = np.inf
            nearest_dir = np.zeros(3)
            for j in range(self.N):
                if j == i:
                    continue
                diff = drone_positions[i] - drone_positions[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    min_dist = dist
                    nearest_dir = diff / (dist + 1e-8)
            base_vel[i] = nearest_dir * self.cfg.spread_repulsion_gain * self.cfg.v_max * 0.3
        return base_vel

    def _build_obs(self, result: dict) -> np.ndarray:
        """Build 19D observation (13D actor + 6D critic extra)."""
        obs = np.zeros((self.N, self.critic_obs_dim), dtype=np.float32)
        drone_positions = result["drone_positions"]
        centroid = drone_positions.mean(axis=0)

        angular_seps = self._compute_angular_separations(drone_positions, centroid)
        optimal_angle = 2.0 * np.pi / self.N

        for i in range(self.N):
            obs[i, 0:2] = self._initial_target_bearing[i]

            offsets = []
            for j in range(self.N):
                if j != i:
                    offsets.append(drone_positions[j] - drone_positions[i])
            obs[i, 2:5] = np.mean(offsets, axis=0)

            obs[i, 5] = (self.N - 1) / max(self.N - 1, 1)

            vel = np.zeros(3)
            if hasattr(self._aviary, '_getDroneStateVector'):
                state = self._aviary._getDroneStateVector(i)
                vel = state[10:13] if len(state) > 12 else np.zeros(3)
            obs[i, 6:9] = vel / self.cfg.v_max

            min_sep = np.pi
            vi = drone_positions[i] - centroid
            vi_norm = np.linalg.norm(vi)
            if vi_norm > 1e-6:
                for j in range(self.N):
                    if j == i:
                        continue
                    vj = drone_positions[j] - centroid
                    vj_norm = np.linalg.norm(vj)
                    if vj_norm > 1e-6:
                        cos_angle = np.clip(
                            np.dot(vi, vj) / (vi_norm * vj_norm), -1.0, 1.0
                        )
                        angle = np.arccos(cos_angle)
                        min_sep = min(min_sep, angle)
            obs[i, 9] = min_sep / np.pi

            if angular_seps:
                obs[i, 10] = np.mean(angular_seps) / optimal_angle
            else:
                obs[i, 10] = 0.0

            obs[i, 11] = np.linalg.norm(drone_positions[i] - centroid) / self.cfg.spread_target_radius

            obs[i, 12] = self._step_count / self.cfg.spread_steps

            # Critic extra (6D)
            obs[i, 13:16] = centroid

            dir_to_target = self._target_pos_init - centroid
            norm = np.linalg.norm(dir_to_target)
            if norm > 1e-6:
                obs[i, 16:19] = dir_to_target / norm
            else:
                obs[i, 16:19] = np.array([1.0, 0.0, 0.0])

        return obs

    def _compute_reward(self, result: dict) -> np.ndarray:
        """FIM-based spread reward (MAIN V5 CHANGE).

        Components:
          r_fim:   (log det(FIM) + 4.0) / 10.0   — shared, geometric info quality
          r_diff:  per-agent difference FIM        — per-drone credit assignment
          r_range: +0.1 if within max_range, -0.1 otherwise — stay in sensor range
        """
        drone_positions = result["drone_positions"]
        target_pos = result["target_true_pos"]

        # All drones treated as "detecting" for geometric potential
        detections = [True] * self.N

        # Shared FIM reward
        fim = compute_bearing_fim(
            drone_positions, target_pos,
            self.cfg.sigma_bearing_rad, self.cfg.range_ref, detections,
        )
        det_fim = max(np.linalg.det(fim), 1e-30)
        log_det = np.log(det_fim)
        r_fim = (log_det + 4.0) / 10.0
        self._last_log_det_fim = log_det

        # Per-agent difference FIM
        r_diff = compute_difference_fim_rewards(
            drone_positions, target_pos,
            self.cfg.sigma_bearing_rad, self.cfg.range_ref, detections,
        )  # (N,)

        # Range bonus: stay within sensor range of target
        r_range = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            dist = np.linalg.norm(drone_positions[i] - target_pos)
            r_range[i] = 0.1 if dist < self.cfg.max_range else -0.1

        # Combine
        reward = r_fim + r_diff + r_range
        reward = np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip)

        # Track metrics for logging
        centroid = drone_positions.mean(axis=0)
        dists = np.array([np.linalg.norm(drone_positions[i] - centroid) for i in range(self.N)])
        self._last_mean_dist = np.mean(dists)

        angles = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                vi = drone_positions[i] - target_pos
                vj = drone_positions[j] - target_pos
                vi_n, vj_n = np.linalg.norm(vi), np.linalg.norm(vj)
                if vi_n > 1e-6 and vj_n > 1e-6:
                    cos_a = np.clip(np.dot(vi, vj) / (vi_n * vj_n), -1.0, 1.0)
                    angles.append(np.arccos(cos_a))
        self._last_min_angle = min(angles) if angles else 0.0

        return reward.astype(np.float32)

    def _compute_angular_separations(
        self, drone_positions: np.ndarray, centroid: np.ndarray
    ) -> list[float]:
        angles = []
        for i, j in combinations(range(self.N), 2):
            vi = drone_positions[i] - centroid
            vj = drone_positions[j] - centroid
            vi_norm = np.linalg.norm(vi)
            vj_norm = np.linalg.norm(vj)
            if vi_norm < 1e-6 or vj_norm < 1e-6:
                angles.append(0.0)
                continue
            cos_angle = np.clip(np.dot(vi, vj) / (vi_norm * vj_norm), -1.0, 1.0)
            angles.append(np.arccos(cos_angle))
        return angles

    def _compute_bearings(
        self, drone_positions: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray:
        bearings = np.zeros((self.N, 2), dtype=np.float32)
        for i in range(self.N):
            diff = target_pos - drone_positions[i]
            r = np.linalg.norm(diff)
            if r < 1e-6:
                continue
            bearings[i, 0] = np.arctan2(diff[1], diff[0])
            bearings[i, 1] = np.arcsin(np.clip(diff[2] / r, -1.0, 1.0))
        return bearings

    def _get_drone_positions(self) -> np.ndarray:
        return np.array([
            self._aviary._getDroneStateVector(i)[:3] for i in range(self.N)
        ])

    def get_drone_positions(self) -> np.ndarray:
        """Public accessor for final positions (used by Phase 2)."""
        return self._get_drone_positions()

    def close(self):
        if self._aviary is not None:
            self._aviary.close()
            self._aviary = None
