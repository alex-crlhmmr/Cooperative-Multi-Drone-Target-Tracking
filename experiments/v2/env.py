"""V2 Tracking Environment — MAPPO with PBRS reward and residual actions.

Key differences from V1:
- 14D actor obs (minimal, topology-aware) + 23D critic obs (global state)
- PBRS reward using FIM potential (no distance penalty)
- Residual actions with 15% authority over chase+offset baseline
- Returns (actor_obs, critic_obs) tuple from step/reset
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from .config import V2TrackingConfig
from .fim import compute_bearing_fim


class V2TrackingEnv(gym.Env):
    """Multi-agent tracking env with dual observations (actor + critic).

    Actor obs (14D per drone): local, topology-safe
    Critic obs (23D per drone): actor obs + global filter state (training only)
    Action (3D per drone): residual correction to chase+offset baseline
    Reward: PBRS with FIM potential (shared across drones)
    """

    def __init__(self, cfg: V2TrackingConfig, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.num_drones
        self._seed = seed or cfg.seed
        self._rng = np.random.default_rng(self._seed)

        # Spaces — externally we expose actor_obs shape
        self.actor_obs_dim = cfg.actor_obs_dim  # 14
        self.critic_obs_dim = cfg.critic_obs_dim  # 23
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
        self._prev_fim = np.zeros((3, 3))
        self.spawn_mode = cfg.spawn_mode

        # Early termination
        self._early_term_counter = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        if self._aviary is not None:
            self._aviary.close()

        # Spawn positions
        target_pos = np.array([0.0, 0.0, 50.0])
        tracker_positions = self._compute_spawn(target_pos)

        self._aviary = TrackingAviary(
            num_trackers=self.N,
            tracker_positions=tracker_positions,
            target_initial_pos=target_pos,
            target_speed=self.cfg.target_speed,
            target_trajectory=self.cfg.target_trajectory,
            target_sigma_a=self.cfg.target_sigma_a,
            episode_length=self.cfg.episode_length,
            sensor_config=self.cfg.sensor_config,
            pyb_freq=self.cfg.pyb_freq,
            ctrl_freq=self.cfg.ctrl_freq,
            gui=False,
            rng=self._rng,
        )
        self._aviary.reset()

        # Adjacency and filter
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
        self._prev_fim = np.zeros((3, 3))
        self._early_term_counter = 0

        # Initial step with zero velocity
        obs, info = self._initial_step()
        return obs, info

    def _compute_spawn(self, target_pos):
        """Compute tracker spawn positions based on spawn_mode."""
        if self.spawn_mode in ("mixed", "cluster"):
            do_cluster = (self.spawn_mode == "cluster") or (self._rng.random() < 0.5)
            if do_cluster:
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
        # Normal: hollow sphere 50-75m
        return spawn_in_hollow_sphere(
            target_pos, r_min=50.0, r_max=75.0, n=self.N,
            rng=self._rng, min_altitude=self.cfg.min_altitude,
        )

    def _initial_step(self):
        """Zero-velocity step to get first measurements and attempt filter init."""
        drone_positions = np.array([
            self._aviary._getDroneStateVector(i)[:3] for i in range(self.N)
        ])
        result = self._aviary.step_tracking(
            tracker_targets=drone_positions,
            tracker_target_vels=np.zeros((self.N, 3)),
        )
        self._last_result = result

        if not self._filter.initialized:
            self._filter.update(result["measurements"], result["drone_positions"])

        # Initialize FIM
        if self._filter.initialized:
            detections = [m is not None for m in result["measurements"]]
            est = self._filter.get_estimate()[:3]
            self._prev_fim = compute_bearing_fim(
                result["drone_positions"], est,
                self.cfg.sigma_bearing_rad, self.cfg.range_ref, detections,
            )

        obs = self._build_obs(result)
        return obs, {"result": result}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.N, 3)
        action = np.clip(action, -1.0, 1.0)

        drone_positions = np.array([
            self._aviary._getDroneStateVector(i)[:3] for i in range(self.N)
        ])

        # Residual policy: base chase+offset + small RL correction
        base_velocity = self._chase_controller(drone_positions)
        correction = action * self.cfg.v_max * self.cfg.residual_scale
        velocity = base_velocity + correction

        # Clamp total velocity
        for i in range(self.N):
            speed = np.linalg.norm(velocity[i])
            if speed > self.cfg.v_max:
                velocity[i] *= self.cfg.v_max / speed

        # Compute waypoints
        dt = self.cfg.dt
        waypoints = drone_positions + velocity * dt
        waypoints[:, 2] = np.maximum(waypoints[:, 2], self.cfg.min_altitude)

        # Per-drone gimbal estimates
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

        # Filter predict + update
        if self._filter.initialized:
            self._filter.predict()
        self._filter.update(result["measurements"], result["drone_positions"])

        obs = self._build_obs(result)
        reward = self._compute_reward(result)

        self._step_count += 1
        terminated = result.get("done", False)
        truncated = False

        # Early termination
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

    def _chase_controller(self, drone_positions: np.ndarray) -> np.ndarray:
        """Base chase+offset controller. Flies toward estimate with radial offset."""
        base_vel = np.zeros((self.N, 3))
        if not self._filter.initialized:
            return base_vel

        for i in range(self.N):
            local_est = self._filter.get_local_estimate(i)
            dir_to_est = local_est[:3] - drone_positions[i]
            dist = np.linalg.norm(dir_to_est)

            if dist < 1e-3:
                # Already at estimate — add feedforward only
                base_vel[i] = local_est[3:6]
                continue

            unit = dir_to_est / dist

            # Proportional approach with desired standoff distance
            # If farther than R_desired, close the gap; if closer, back off slightly
            radial_error = dist - self.cfg.base_R_desired
            approach_speed = np.clip(
                self.cfg.base_Kp * radial_error,
                -self.cfg.v_max * 0.3,
                self.cfg.v_max * 0.5,
            )
            base_vel[i] = unit * approach_speed + local_est[3:6]  # feedforward

            # Clamp base velocity to leave room for RL correction
            max_base = self.cfg.v_max * (1.0 - self.cfg.residual_scale)
            speed = np.linalg.norm(base_vel[i])
            if speed > max_base:
                base_vel[i] *= max_base / speed

        return base_vel

    def _build_obs(self, result: dict) -> np.ndarray:
        """Build dual observations: full critic_obs (23D), actor_obs is [:14].

        Returns: (N, critic_obs_dim) with actor_obs in first 14 columns.
        """
        obs = np.zeros((self.N, self.critic_obs_dim), dtype=np.float32)

        if not self._filter.initialized:
            return obs

        drone_positions = result["drone_positions"]  # (N, 3)

        # Precompute neighbor info from adjacency
        neighbors = {}
        for i in range(self.N):
            neighbors[i] = [j for j in range(self.N) if j != i and self._adj[i, j] > 0]

        # Angular rank (physical observation, all drones)
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

            # ── Actor obs (14D) ──
            # [0:3] rel_pos
            obs[i, 0:3] = local_est[:3] - drone_positions[i]
            # [3:6] vel_est
            obs[i, 3:6] = local_est[3:6]
            # [6] log_tr_P
            P_pos = local_cov[:3, :3]
            tr_P = np.trace(P_pos)
            obs[i, 6] = np.log(tr_P + 1.0)
            # [7] detected
            obs[i, 7] = 1.0 if result["measurements"][i] is not None else 0.0
            # [8:11] neighbor_mean_offset
            nbrs = neighbors[i]
            if len(nbrs) > 0:
                offsets = np.array([drone_positions[j] - drone_positions[i] for j in nbrs])
                obs[i, 8:11] = offsets.mean(axis=0)
            # [11] neighbor_count (normalized)
            obs[i, 11] = len(nbrs) / max(self.N - 1, 1)
            # [12] angular_rank
            obs[i, 12] = ranks[i]
            # [13] range_to_est (normalized)
            obs[i, 13] = np.linalg.norm(drone_positions[i] - local_est[:3]) / self.cfg.max_range

            # ── Critic extra (9D) ──
            # [14:20] consensus estimate
            obs[i, 14:20] = self._filter.get_estimate()
            # [20:23] log eigenvalues of global P_pos
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
        """PBRS reward: bounded log-covariance + FIM potential shaping.

        All drones get the SAME reward (fully cooperative).
        """
        reward_scalar = 0.0

        if not self._filter.initialized:
            return np.full(self.N, -1.0, dtype=np.float32)

        # Base reward: bounded log-covariance
        tr_P = self._get_tr_P_pos()
        log_norm = np.log(1000.0 + 1.0)  # normalization constant
        r_base = -np.clip(np.log(tr_P + 1.0) / log_norm, -1.0, 1.0)

        # PBRS shaping: F = γ·Φ(s') - Φ(s) where Φ = log(det(FIM))
        drone_positions = result["drone_positions"]
        detections = [m is not None for m in result["measurements"]]
        est = self._filter.get_estimate()[:3]

        fim_new = compute_bearing_fim(
            drone_positions, est,
            self.cfg.sigma_bearing_rad, self.cfg.range_ref, detections,
        )

        det_new = max(np.linalg.det(fim_new), 1e-20)
        det_old = max(np.linalg.det(self._prev_fim), 1e-20)
        phi_new = np.log(det_new)
        phi_old = np.log(det_old)
        phi_new_clip = np.clip(phi_new, -20.0, 5.0)
        phi_old_clip = np.clip(phi_old, -20.0, 5.0)
        r_shaping = self.cfg.gamma * phi_new_clip - phi_old_clip

        self._prev_fim = fim_new

        reward_scalar = np.clip(
            r_base + r_shaping,
            -self.cfg.reward_clip, self.cfg.reward_clip,
        )

        # All drones get same reward
        return np.full(self.N, reward_scalar, dtype=np.float32)

    def _get_tr_P_pos(self) -> float:
        if not self._filter.initialized:
            return self.cfg.P0_pos * 3.0
        P = self._filter.get_covariance()
        return float(np.trace(P[:3, :3]))

    def close(self):
        if self._aviary is not None:
            self._aviary.close()
            self._aviary = None
