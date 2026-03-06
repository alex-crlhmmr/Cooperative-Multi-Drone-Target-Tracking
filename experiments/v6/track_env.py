"""V6 Phase 2: TrackEnv — Angular-repulsion tracking with RL residual.

Key change from V4: _tracking_controller replaces _chase_controller.
Adds angular repulsion between drone pairs (relative to target estimate)
to maintain bearing geometry diversity during tracking.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from .config import V6Config
from .fim import compute_bearing_fim, compute_difference_fim_rewards


class TrackEnv(gym.Env):
    """Phase 2 environment: tracking with angular-repulsion base controller.

    Actor obs (14D per drone): rel_pos, vel_est, log_tr_P, detected,
                                neighbor_mean_offset, neighbor_count,
                                angular_rank, range_to_est
    Critic obs (23D per drone): actor obs + consensus estimate + log P eigenvalues
    Action (3D per drone): residual correction to angular-repulsion tracking
    Reward: r_fim + r_diff + r_cov_rate + r_detect (per-agent)
    """

    def __init__(self, cfg: V6Config, seed: int | None = None,
                 initial_positions: np.ndarray | None = None):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.num_drones
        self._seed = seed or cfg.seed
        self._rng = np.random.default_rng(self._seed)

        self.actor_obs_dim = cfg.track_actor_obs_dim     # 14
        self.critic_obs_dim = cfg.track_critic_obs_dim   # 23
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.N, self.critic_obs_dim), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.N, 3), dtype=np.float32,
        )

        self._aviary = None
        self._filter = None
        self._adj = None
        self._step_count = 0
        self._prev_tr_P = None
        self._initial_positions = initial_positions
        self._early_term_counter = 0
        self.spawn_mode = "normal"

    def set_initial_positions(self, positions: np.ndarray):
        self._initial_positions = positions.copy()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        if self._aviary is not None:
            self._aviary.close()

        target_pos = np.array([0.0, 0.0, 50.0])

        if self._initial_positions is not None:
            tracker_positions = self._initial_positions.copy()
        elif self.spawn_mode == "cluster":
            tracker_positions = self._cluster_spawn(target_pos)
        else:
            tracker_positions = spawn_in_hollow_sphere(
                target_pos, r_min=50.0, r_max=75.0, n=self.N,
                rng=self._rng, min_altitude=self.cfg.min_altitude,
            )

        self._aviary = TrackingAviary(
            num_trackers=self.N,
            tracker_positions=tracker_positions,
            target_initial_pos=target_pos,
            target_speed=self.cfg.target_speed,
            target_trajectory=self.cfg.target_trajectory,
            target_sigma_a=self.cfg.target_sigma_a,
            episode_length=self.cfg.track_episode_length,
            sensor_config=self.cfg.sensor_config,
            pyb_freq=self.cfg.pyb_freq,
            ctrl_freq=self.cfg.ctrl_freq,
            gui=False,
            rng=self._rng,
        )
        self._aviary.reset()

        self._adj = generate_adjacency(self.N, self.cfg.topology)
        self._filter = ConsensusIMM(
            dt=self.cfg.dt,
            sigma_a_modes=list(self.cfg.imm_sigma_a_modes),
            sigma_bearing=self.cfg.sigma_bearing_rad,
            range_ref=self.cfg.range_ref,
            transition_matrix=self.cfg.transition_matrix,
            num_drones=self.N,
            adjacency=self._adj,
            num_consensus_iters=self.cfg.consensus_iters,
            consensus_step_size=self.cfg.consensus_step_size,
            dropout_prob=self.cfg.dropout_prob,
            P0_pos=self.cfg.P0_pos,
            P0_vel=self.cfg.P0_vel,
            rng=self._rng,
        )

        self._step_count = 0
        self._prev_tr_P = None
        self._early_term_counter = 0

        obs, info = self._initial_step()
        return obs, info

    def _cluster_spawn(self, target_pos):
        direction = self._rng.standard_normal(3)
        direction[2] = abs(direction[2])
        direction /= np.linalg.norm(direction) + 1e-8
        cluster_center = target_pos + direction * 100.0
        cluster_center[2] = max(cluster_center[2], self.cfg.min_altitude + 10.0)
        positions = np.array([
            cluster_center + self._rng.uniform(-10, 10, size=3)
            for _ in range(self.N)
        ])
        positions[:, 2] = np.maximum(positions[:, 2], self.cfg.min_altitude)
        return positions

    def _initial_step(self):
        drone_positions = self._get_drone_positions()
        result = self._aviary.step_tracking(
            tracker_targets=drone_positions,
            tracker_target_vels=np.zeros((self.N, 3)),
        )
        self._last_result = result

        if not self._filter.initialized:
            self._filter.update(result["measurements"], result["drone_positions"])

        if self._filter.initialized:
            self._prev_tr_P = self._get_tr_P_pos()

        obs = self._build_obs(result)
        return obs, {"result": result}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.N, 3)
        action = np.clip(action, -1.0, 1.0)

        drone_positions = self._get_drone_positions()

        # V6: angular-repulsion tracking + RL residual
        base_velocity = self._tracking_controller(drone_positions)
        correction = action * self.cfg.v_max * self.cfg.track_residual_scale
        velocity = base_velocity + correction

        for i in range(self.N):
            speed = np.linalg.norm(velocity[i])
            if speed > self.cfg.v_max:
                velocity[i] *= self.cfg.v_max / speed

        dt = self.cfg.dt
        waypoints = drone_positions + velocity * dt
        waypoints[:, 2] = np.maximum(waypoints[:, 2], self.cfg.min_altitude)

        per_drone_estimates = None
        if self._filter.initialized:
            per_drone_estimates = np.array([
                self._filter.get_local_estimate(i)[:3] for i in range(self.N)
            ])

        result = self._aviary.step_tracking(
            tracker_targets=waypoints,
            tracker_target_vels=velocity,
            per_drone_estimates=per_drone_estimates,
        )
        self._last_result = result

        if result.get("done", False) and "drone_positions" not in result:
            obs = np.zeros((self.N, self.critic_obs_dim), dtype=np.float32)
            reward = np.full(self.N, -1.0, dtype=np.float32)
            return obs, reward, True, False, {"result": result}

        if self._filter.initialized:
            self._filter.predict()
        self._filter.update(result["measurements"], result["drone_positions"])

        obs = self._build_obs(result)
        reward = self._compute_reward(result)

        self._step_count += 1
        terminated = result.get("done", False)
        truncated = False

        if not terminated and self._step_count > self.cfg.early_term_grace:
            tr_P = self._get_tr_P_pos()
            if tr_P > self.cfg.early_term_threshold:
                self._early_term_counter += 1
                if self._early_term_counter >= self.cfg.early_term_patience:
                    truncated = True
            else:
                self._early_term_counter = 0

        info = {
            "result": result,
            "filter_initialized": self._filter.initialized,
            "tr_P_pos": self._get_tr_P_pos(),
        }
        return obs, reward, terminated, truncated, info

    def _tracking_controller(self, drone_positions: np.ndarray) -> np.ndarray:
        """Angular-repulsion tracking controller (V6 key change).

        Blends chase (PD toward filter estimate at R_desired) with
        angular repulsion (tangential push away from angularly-close drones).
        """
        base_vel = np.zeros((self.N, 3))
        if not self._filter.initialized:
            return base_vel

        # Get consensus estimate for angular repulsion reference
        consensus_est = self._filter.get_estimate()[:3]

        for i in range(self.N):
            local_est = self._filter.get_local_estimate(i)

            # 1. Chase component (same as V4)
            dir_to_est = local_est[:3] - drone_positions[i]
            dist = np.linalg.norm(dir_to_est)

            if dist < 1e-3:
                chase_vel = local_est[3:6].copy()
            else:
                unit = dir_to_est / dist
                radial_error = dist - self.cfg.track_base_R_desired
                approach_speed = np.clip(
                    self.cfg.track_base_Kp * radial_error,
                    -self.cfg.v_max * 0.3,
                    self.cfg.v_max * 0.5,
                )
                chase_vel = unit * approach_speed + local_est[3:6]

            # 2. Angular repulsion (NEW)
            angular_vel = np.zeros(3)
            vi = drone_positions[i] - consensus_est
            vi_norm = np.linalg.norm(vi)

            if vi_norm > 1e-6:
                for j in range(self.N):
                    if j == i:
                        continue
                    vj = drone_positions[j] - consensus_est
                    vj_norm = np.linalg.norm(vj)

                    if vj_norm < 1e-6:
                        continue

                    # Angle between radial vectors
                    cos_angle = np.clip(
                        np.dot(vi, vj) / (vi_norm * vj_norm), -1.0, 1.0
                    )
                    angle_ij = np.arccos(cos_angle)

                    # Tangential push: cross product gives perpendicular direction
                    tangent = np.cross(vi, vj)
                    tang_norm = np.linalg.norm(tangent)
                    if tang_norm > 1e-8:
                        tangent /= tang_norm
                    else:
                        # Drones are collinear — push in arbitrary perpendicular dir
                        perp = np.cross(vi, np.array([0.0, 0.0, 1.0]))
                        pn = np.linalg.norm(perp)
                        if pn > 1e-8:
                            tangent = perp / pn
                        else:
                            perp = np.cross(vi, np.array([0.0, 1.0, 0.0]))
                            tangent = perp / (np.linalg.norm(perp) + 1e-8)

                    # Stronger repulsion when drones are close in angle
                    angular_vel += tangent / (angle_ij + 0.1) * self.cfg.angular_gain

            # 3. Blend chase and angular repulsion
            w = self.cfg.angular_weight
            blended = (1.0 - w) * chase_vel + w * angular_vel

            # Clamp to leave room for residual
            max_base = self.cfg.v_max * (1.0 - self.cfg.track_residual_scale)
            speed = np.linalg.norm(blended)
            if speed > max_base:
                blended *= max_base / speed

            base_vel[i] = blended

        return base_vel

    def _build_obs(self, result: dict) -> np.ndarray:
        """Build 23D observation (14D actor + 9D critic extra)."""
        obs = np.zeros((self.N, self.critic_obs_dim), dtype=np.float32)

        if not self._filter.initialized:
            return obs

        drone_positions = result["drone_positions"]

        neighbors = {}
        for i in range(self.N):
            neighbors[i] = [j for j in range(self.N) if j != i and self._adj[i, j] > 0]

        # Angular rank
        consensus_est = self._filter.get_estimate()[:3]
        centroid = drone_positions.mean(axis=0)
        ref_dir = consensus_est - centroid
        ref_norm = np.linalg.norm(ref_dir)
        if ref_norm > 1e-6:
            ref_dir /= ref_norm
        else:
            ref_dir = np.array([1.0, 0.0, 0.0])

        perp1 = np.cross(ref_dir, [0, 0, 1])
        if np.linalg.norm(perp1) < 1e-6:
            perp1 = np.cross(ref_dir, [0, 1, 0])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(ref_dir, perp1)

        angles = []
        for i in range(self.N):
            d = drone_positions[i] - centroid
            angles.append(np.arctan2(np.dot(d, perp2), np.dot(d, perp1)))
        rank_order = np.argsort(angles)
        ranks = np.zeros(self.N)
        for order_idx, drone_idx in enumerate(rank_order):
            ranks[drone_idx] = order_idx / max(self.N - 1, 1)

        for i in range(self.N):
            local_est = self._filter.get_local_estimate(i)
            local_cov = self._filter.get_local_covariance(i)

            # Actor obs (14D)
            obs[i, 0:3] = local_est[:3] - drone_positions[i]
            obs[i, 3:6] = local_est[3:6]
            P_pos = local_cov[:3, :3]
            obs[i, 6] = np.log(np.trace(P_pos) + 1.0)
            obs[i, 7] = 1.0 if result["measurements"][i] is not None else 0.0
            nbrs = neighbors[i]
            if len(nbrs) > 0:
                offsets = np.array([drone_positions[j] - drone_positions[i] for j in nbrs])
                obs[i, 8:11] = offsets.mean(axis=0)
            obs[i, 11] = len(nbrs) / max(self.N - 1, 1)
            obs[i, 12] = ranks[i]
            obs[i, 13] = np.linalg.norm(drone_positions[i] - local_est[:3]) / self.cfg.max_range

            # Critic extra (9D)
            obs[i, 14:20] = self._filter.get_estimate()
            P_global = self._filter.get_covariance()[:3, :3]
            P_global = 0.5 * (P_global + P_global.T)
            try:
                eigvals = np.linalg.eigvalsh(P_global)
                eigvals = np.maximum(eigvals, 1e-10)
            except np.linalg.LinAlgError:
                eigvals = np.diag(P_global)
                eigvals = np.maximum(eigvals, 1e-10)
            obs[i, 20:23] = np.log(eigvals + 1.0)

        return obs

    def _compute_reward(self, result: dict) -> np.ndarray:
        """FIM-primary + difference rewards (same as V4)."""
        if not self._filter.initialized:
            return np.full(self.N, -1.0, dtype=np.float32)

        drone_positions = result["drone_positions"]
        detections = [m is not None for m in result["measurements"]]
        est = self._filter.get_estimate()[:3]

        fim = compute_bearing_fim(
            drone_positions, est,
            self.cfg.sigma_bearing_rad, self.cfg.range_ref, detections,
        )
        det_fim = max(np.linalg.det(fim), 1e-30)
        r_fim = (np.log(det_fim) + 4.0) / 10.0

        r_diff = compute_difference_fim_rewards(
            drone_positions, est,
            self.cfg.sigma_bearing_rad, self.cfg.range_ref, detections,
        )

        tr_P = self._get_tr_P_pos()
        r_cov_rate = 0.0
        if self._prev_tr_P is not None:
            r_cov_rate = (self._prev_tr_P - tr_P) / 100.0
        self._prev_tr_P = tr_P

        r_detect = np.array([0.1 if d else 0.0 for d in detections], dtype=np.float32)

        reward = r_fim + r_diff + r_cov_rate + r_detect
        reward = np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip)
        return reward.astype(np.float32)

    def _get_tr_P_pos(self) -> float:
        if not self._filter.initialized:
            return self.cfg.P0_pos * 3.0
        P = self._filter.get_covariance()
        return float(np.trace(P[:3, :3]))

    def _get_drone_positions(self) -> np.ndarray:
        return np.array([
            self._aviary._getDroneStateVector(i)[:3] for i in range(self.N)
        ])

    def close(self):
        if self._aviary is not None:
            self._aviary.close()
            self._aviary = None
