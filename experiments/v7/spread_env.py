"""V7 SpreadEnv — RL residual on repulsion heuristic, angular separation reward.

Key changes from V5:
- Action: residual on repulsion heuristic (not full RL)
- Reward: angular separation relative to TARGET (not FIM)
- Obs: expanded (15D) with per-drone angular info relative to target
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.tracking_aviary import TrackingAviary
from .config import V7Config


class SpreadEnv(gym.Env):
    """Spread drones from cluster with RL residual on repulsion heuristic.

    Actor obs (15D per drone):
      [0:2]  initial target bearing (az, el)
      [2:5]  neighbor mean offset
      [5]    neighbor count (normalized)
      [6:9]  own velocity (normalized)
      [9]    min angular sep from target (this drone)
      [10]   mean angular sep from target (all pairs)
      [11]   range to target (normalized)
      [12]   range to centroid (normalized)
      [13]   angular rank (sorted by angle around target)
      [14]   normalized step

    Critic extra (6D):
      [15:18] swarm centroid
      [18:21] centroid-to-target unit vector
    """

    def __init__(self, cfg: V7Config, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.num_drones
        self._seed = seed or cfg.seed
        self._rng = np.random.default_rng(self._seed)

        self.actor_obs_dim = cfg.spread_obs_dim          # 15
        self.critic_obs_dim = cfg.spread_critic_obs_dim  # 21
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
        target_pos = self._last_result["target_true_pos"]

        # Base: repulsion heuristic with surround bias
        base_vel = self._repulsion_base(drone_positions, target_pos)
        # Residual: RL correction
        correction = action * self.cfg.v_max * self.cfg.spread_residual_scale
        velocity = base_vel + correction

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
            "mean_angular_sep": self._last_mean_angle,
            "mean_range_to_target": self._last_mean_range,
        }
        return obs, reward, terminated, truncated, info

    def _repulsion_base(self, drone_positions: np.ndarray,
                        target_pos: np.ndarray) -> np.ndarray:
        """Repulsion heuristic with surround bias toward target."""
        base_vel = np.zeros((self.N, 3))

        for i in range(self.N):
            # Repulsion: fly away from nearest neighbor
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

            repulsion_vel = nearest_dir

            # Surround bias: tangential orbit around target
            to_target = target_pos - drone_positions[i]
            radial_dist = np.linalg.norm(to_target)

            if radial_dist > 1e-6:
                radial_unit = to_target / radial_dist
                up = np.array([0.0, 0.0, 1.0])
                tangent = np.cross(radial_unit, up)
                tang_norm = np.linalg.norm(tangent)
                if tang_norm > 1e-6:
                    tangent /= tang_norm
                else:
                    tangent = np.cross(radial_unit, np.array([1.0, 0.0, 0.0]))
                    tangent /= np.linalg.norm(tangent) + 1e-8

                if i % 2 == 1:
                    tangent = -tangent

                sw = self.cfg.spread_surround_weight
                vel = (1.0 - sw) * repulsion_vel + sw * tangent
            else:
                vel = repulsion_vel

            # Scale
            speed = np.linalg.norm(vel)
            if speed > 1e-6:
                vel = vel / speed * self.cfg.v_max * self.cfg.spread_repulsion_gain
            else:
                vel = np.zeros(3)

            # Leave room for residual
            max_base = self.cfg.v_max * (1.0 - self.cfg.spread_residual_scale)
            speed = np.linalg.norm(vel)
            if speed > max_base:
                vel *= max_base / speed

            base_vel[i] = vel

        return base_vel

    def _build_obs(self, result: dict) -> np.ndarray:
        """Build 21D obs (15D actor + 6D critic extra)."""
        obs = np.zeros((self.N, self.critic_obs_dim), dtype=np.float32)
        drone_positions = result["drone_positions"]
        target_pos = result["target_true_pos"]
        centroid = drone_positions.mean(axis=0)

        # Pairwise angles relative to TARGET
        pairwise_angles = self._compute_pairwise_angles(drone_positions, target_pos)
        optimal_angle = 2.0 * np.pi / self.N  # 72 deg for 5 drones

        # Angular rank around target
        ranks = self._compute_angular_ranks(drone_positions, target_pos)

        for i in range(self.N):
            # [0:2] initial target bearing (az, el)
            obs[i, 0:2] = self._initial_target_bearing[i]

            # [2:5] neighbor mean offset
            offsets = []
            for j in range(self.N):
                if j != i:
                    offsets.append(drone_positions[j] - drone_positions[i])
            obs[i, 2:5] = np.mean(offsets, axis=0)

            # [5] neighbor count (normalized)
            obs[i, 5] = (self.N - 1) / max(self.N - 1, 1)

            # [6:9] own velocity (normalized)
            vel = np.zeros(3)
            if hasattr(self._aviary, '_getDroneStateVector'):
                state = self._aviary._getDroneStateVector(i)
                vel = state[10:13] if len(state) > 12 else np.zeros(3)
            obs[i, 6:9] = vel / self.cfg.v_max

            # [9] min angular sep from target (this drone's min angle to any neighbor)
            min_sep = np.pi
            vi = drone_positions[i] - target_pos
            vi_norm = np.linalg.norm(vi)
            if vi_norm > 1e-6:
                for j in range(self.N):
                    if j == i:
                        continue
                    vj = drone_positions[j] - target_pos
                    vj_norm = np.linalg.norm(vj)
                    if vj_norm > 1e-6:
                        cos_a = np.clip(np.dot(vi, vj) / (vi_norm * vj_norm), -1.0, 1.0)
                        min_sep = min(min_sep, np.arccos(cos_a))
            obs[i, 9] = min_sep / optimal_angle  # >1 means better than equiangular

            # [10] mean angular sep (all pairs, normalized by optimal)
            if pairwise_angles:
                obs[i, 10] = np.mean(pairwise_angles) / optimal_angle
            else:
                obs[i, 10] = 0.0

            # [11] range to target (normalized)
            obs[i, 11] = np.linalg.norm(drone_positions[i] - target_pos) / self.cfg.max_range

            # [12] range to centroid (normalized)
            obs[i, 12] = np.linalg.norm(drone_positions[i] - centroid) / 100.0

            # [13] angular rank
            obs[i, 13] = ranks[i]

            # [14] normalized step (time pressure)
            obs[i, 14] = self._step_count / self.cfg.spread_steps

            # ── Critic extra (6D) ──
            # [15:18] swarm centroid
            obs[i, 15:18] = centroid

            # [18:21] centroid-to-target unit vector
            dir_to_target = target_pos - centroid
            norm = np.linalg.norm(dir_to_target)
            if norm > 1e-6:
                obs[i, 18:21] = dir_to_target / norm
            else:
                obs[i, 18:21] = np.array([1.0, 0.0, 0.0])

        return obs

    def _compute_reward(self, result: dict) -> np.ndarray:
        """Angular separation reward — dense, clear, per-drone.

        Components:
          r_min_angle: per-drone min angular sep / optimal_angle
          r_mean_angle: shared mean angular sep / optimal_angle
          r_range: per-drone bonus for being in [range_min, range_max] from target
        """
        drone_positions = result["drone_positions"]
        target_pos = result["target_true_pos"]
        optimal_angle = 2.0 * np.pi / self.N

        rewards = np.zeros(self.N, dtype=np.float32)

        # Per-drone min angular separation from target
        per_drone_min_angle = np.zeros(self.N)
        all_pairwise = []

        for i in range(self.N):
            min_angle = np.pi
            vi = drone_positions[i] - target_pos
            vi_norm = np.linalg.norm(vi)

            for j in range(self.N):
                if j == i:
                    continue
                vj = drone_positions[j] - target_pos
                vj_norm = np.linalg.norm(vj)
                if vi_norm > 1e-6 and vj_norm > 1e-6:
                    cos_a = np.clip(np.dot(vi, vj) / (vi_norm * vj_norm), -1.0, 1.0)
                    angle = np.arccos(cos_a)
                    min_angle = min(min_angle, angle)
                    if j > i:
                        all_pairwise.append(angle)

            per_drone_min_angle[i] = min_angle

        # r_min_angle: per-drone, normalized by optimal
        r_min_angle = per_drone_min_angle / optimal_angle  # >1 is good
        rewards += self.cfg.reward_min_angle * r_min_angle

        # r_mean_angle: shared, log of global min pairwise angle
        global_min_angle = min(all_pairwise) if all_pairwise else 0.0
        r_mean = global_min_angle / optimal_angle
        rewards += self.cfg.reward_mean_angle * r_mean

        # r_range: per-drone, bonus for good range from target
        for i in range(self.N):
            dist = np.linalg.norm(drone_positions[i] - target_pos)
            if self.cfg.reward_range_min <= dist <= self.cfg.reward_range_max:
                rewards[i] += self.cfg.reward_range * 1.0
            else:
                # Penalty proportional to how far outside the range
                if dist < self.cfg.reward_range_min:
                    excess = (self.cfg.reward_range_min - dist) / self.cfg.reward_range_min
                else:
                    excess = (dist - self.cfg.reward_range_max) / self.cfg.reward_range_max
                rewards[i] -= self.cfg.reward_range * excess

        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)

        # Track metrics for logging
        self._last_min_angle = global_min_angle
        self._last_mean_angle = np.mean(all_pairwise) if all_pairwise else 0.0
        ranges = [np.linalg.norm(drone_positions[i] - target_pos) for i in range(self.N)]
        self._last_mean_range = np.mean(ranges)

        return rewards.astype(np.float32)

    def _compute_pairwise_angles(self, drone_positions, target_pos):
        angles = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                vi = drone_positions[i] - target_pos
                vj = drone_positions[j] - target_pos
                vi_n, vj_n = np.linalg.norm(vi), np.linalg.norm(vj)
                if vi_n > 1e-6 and vj_n > 1e-6:
                    cos_a = np.clip(np.dot(vi, vj) / (vi_n * vj_n), -1.0, 1.0)
                    angles.append(np.arccos(cos_a))
        return angles

    def _compute_angular_ranks(self, drone_positions, target_pos):
        """Rank drones by angle around target (in a reference plane)."""
        ref_dir = drone_positions.mean(axis=0) - target_pos
        ref_norm = np.linalg.norm(ref_dir)
        if ref_norm < 1e-6:
            ref_dir = np.array([1.0, 0.0, 0.0])
        else:
            ref_dir /= ref_norm

        perp1 = np.cross(ref_dir, [0, 0, 1])
        if np.linalg.norm(perp1) < 1e-6:
            perp1 = np.cross(ref_dir, [0, 1, 0])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(ref_dir, perp1)

        angles = []
        for i in range(self.N):
            d = drone_positions[i] - target_pos
            angles.append(np.arctan2(np.dot(d, perp2), np.dot(d, perp1)))

        rank_order = np.argsort(angles)
        ranks = np.zeros(self.N, dtype=np.float32)
        for order_idx, drone_idx in enumerate(rank_order):
            ranks[drone_idx] = order_idx / max(self.N - 1, 1)
        return ranks

    def _compute_bearings(self, drone_positions, target_pos):
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
        return self._get_drone_positions()

    def close(self):
        if self._aviary is not None:
            self._aviary.close()
            self._aviary = None
