"""Gym wrapper for multi-drone active tracking with ConsensusIMM.

Wraps TrackingAviary + ConsensusIMM into a standard gym.Env interface.
Each step: RL outputs 3D velocities per drone -> PID tracks waypoints ->
filter updates -> obs/reward computed from filter state.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from .tracking_config import TrackingConfig


class MultiDroneTrackingEnv(gym.Env):
    """Multi-agent tracking environment.

    Observation per drone (31D):
        [0:3]   relative position to local estimate (egocentric)
        [3:6]   velocity estimate from local filter
        [6:9]   eigenvalues of position covariance (3)
        [9:18]  eigenvectors of position covariance (9 = 3x3 flattened)
        [18]    detection flag (0 or 1)
        [19:31] neighbor features: mean(3) + max(3) + min(3) + std(3) of relative positions

    Action per drone (3D): raw values in [-1, 1], squashed via tanh * v_max.
    Reward per drone: -w_cov * tr(P_pos) / scale - w_dist * dist / dist_scale + w_detection * detected
    """

    def __init__(self, cfg: TrackingConfig, seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.num_drones
        self._seed = seed or cfg.seed
        self._rng = np.random.default_rng(self._seed)

        # Observation and action spaces
        self.obs_dim = 31
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.N, self.obs_dim), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.N, 3), dtype=np.float32,
        )

        # Will be created in reset()
        self._aviary = None
        self._filter = None
        self._adj = None
        self._step_count = 0

        # Spawn mode: "normal" = hollow sphere, "mixed" = random mix of cluster + normal
        self.spawn_mode = "normal"

        # Early termination: if tr(P) > threshold for patience steps after grace period
        self._early_term_grace = 1000     # no termination for first N steps
        self._early_term_threshold = 10000.0
        self._early_term_patience = 200
        self._early_term_counter = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)

        # Close previous aviary if exists
        if self._aviary is not None:
            self._aviary.close()

        # Compute tracker spawn positions
        target_pos = np.array([0.0, 0.0, 50.0])
        if self.spawn_mode in ("mixed", "cluster"):
            # mixed: 50% cluster, 50% normal. cluster: always cluster.
            do_cluster = (self.spawn_mode == "cluster") or (self._rng.random() < 0.5)
            if do_cluster:
                # Cluster: all drones near one point, random radius 5-30m
                r = self._rng.uniform(5.0, 30.0)
                direction = self._rng.standard_normal(3)
                direction[2] = abs(direction[2])  # bias upward
                direction /= np.linalg.norm(direction) + 1e-8
                cluster_center = target_pos + direction * 100.0
                cluster_center[2] = max(cluster_center[2], self.cfg.min_altitude + 10.0)
                tracker_positions = np.array([
                    cluster_center + self._rng.uniform(-r, r, size=3) for _ in range(self.N)
                ])
                tracker_positions[:, 2] = np.maximum(tracker_positions[:, 2], self.cfg.min_altitude)
            else:
                tracker_positions = None  # normal hollow sphere (50-75m)
        else:
            tracker_positions = None  # default hollow sphere

        # Create aviary
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

        # Create filter
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
        self._last_result = None
        self._early_term_counter = 0

        # Take one initial step with zero velocity to get first measurements
        obs, info = self._initial_step()
        return obs, info

    def _initial_step(self):
        """Take a zero-velocity step to initialize sensor readings and attempt filter init."""
        drone_positions = np.array([
            self._aviary._getDroneStateVector(i)[:3] for i in range(self.N)
        ])
        result = self._aviary.step_tracking(
            tracker_targets=drone_positions,
            tracker_target_vels=np.zeros((self.N, 3)),
        )
        self._last_result = result

        # Try to initialize filter
        if not self._filter.initialized:
            self._filter.update(result["measurements"], result["drone_positions"])

        obs = self._build_obs(result)
        return obs, {"result": result}

    def step(self, action: np.ndarray):
        """Step the environment with per-drone velocity commands.

        Args:
            action: (N, 3) raw actions in [-1, 1]

        Returns:
            obs: (N, obs_dim) observations
            reward: (N,) per-drone rewards
            terminated: bool
            truncated: bool
            info: dict
        """
        action = np.asarray(action, dtype=np.float32).reshape(self.N, 3)
        action = np.clip(action, -1.0, 1.0)

        # Current drone positions
        drone_positions = np.array([
            self._aviary._getDroneStateVector(i)[:3] for i in range(self.N)
        ])

        # Residual policy: action is a correction to a "chase the estimate" baseline
        # Base velocity: fly toward the local filter estimate (or hold if not init)
        base_velocity = np.zeros((self.N, 3))
        if self._filter.initialized:
            for i in range(self.N):
                local_est = self._filter.get_local_estimate(i)
                dir_to_est = local_est[:3] - drone_positions[i]
                dist = np.linalg.norm(dir_to_est)
                if dist > 1e-3:
                    # Chase at target's estimated speed + close the gap
                    base_velocity[i] = (dir_to_est / dist) * min(dist * 0.5, self.cfg.v_max * 0.5)
                    base_velocity[i] += local_est[3:6]  # feedforward target velocity
                    speed = np.linalg.norm(base_velocity[i])
                    if speed > self.cfg.v_max * 0.5:
                        base_velocity[i] *= self.cfg.v_max * 0.5 / speed

        # RL correction: scales up to 50% of v_max
        correction = action * self.cfg.v_max * 0.5
        velocity = base_velocity + correction

        # Clamp total velocity
        for i in range(self.N):
            speed = np.linalg.norm(velocity[i])
            if speed > self.cfg.v_max:
                velocity[i] *= self.cfg.v_max / speed

        # Compute waypoints from velocity
        dt = self.cfg.dt
        waypoints = drone_positions + velocity * dt

        # Clamp altitude
        waypoints[:, 2] = np.maximum(waypoints[:, 2], self.cfg.min_altitude)

        # Get per-drone gimbal estimates (use local filter estimates if initialized)
        per_drone_estimates = None
        if self._filter.initialized:
            per_drone_estimates = np.array([
                self._filter.get_local_estimate(i)[:3] for i in range(self.N)
            ])

        # Step aviary
        result = self._aviary.step_tracking(
            tracker_targets=waypoints,
            tracker_target_vels=velocity,
            per_drone_estimates=per_drone_estimates,
        )
        self._last_result = result

        if result.get("done", False) and "drone_positions" not in result:
            # Episode ended before we got data
            obs = np.zeros((self.N, self.obs_dim), dtype=np.float32)
            reward = np.full(self.N, -1.0, dtype=np.float32)
            return obs, reward, True, False, {"result": result}

        # Filter predict + update
        if self._filter.initialized:
            self._filter.predict()
        self._filter.update(result["measurements"], result["drone_positions"])

        # Build observation and reward
        obs = self._build_obs(result)
        reward = self._compute_reward(result)

        self._step_count += 1
        terminated = result.get("done", False)
        truncated = False

        # Early termination: if tr(P) stays above threshold after grace period
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
            "tr_P_pos": self._get_tr_P_pos(),
        }

        return obs, reward, terminated, truncated, info

    def _build_obs(self, result: dict) -> np.ndarray:
        """Build per-drone egocentric observations.

        Returns: (N, obs_dim) float32 array.
        """
        obs = np.zeros((self.N, self.obs_dim), dtype=np.float32)

        if not self._filter.initialized:
            return obs

        drone_positions = result["drone_positions"]  # (N, 3)

        for i in range(self.N):
            local_est = self._filter.get_local_estimate(i)  # (6,)
            local_cov = self._filter.get_local_covariance(i)  # (6, 6)

            # Relative position to local estimate (egocentric)
            rel_pos = local_est[:3] - drone_positions[i]
            obs[i, 0:3] = rel_pos

            # Velocity estimate
            obs[i, 3:6] = local_est[3:6]

            # Eigendecomposition of position covariance block
            P_pos = local_cov[:3, :3]
            P_pos = 0.5 * (P_pos + P_pos.T)  # ensure symmetric
            try:
                eigvals, eigvecs = np.linalg.eigh(P_pos)
                eigvals = np.maximum(eigvals, 0.0)  # clip negative eigenvalues
            except np.linalg.LinAlgError:
                eigvals = np.diag(P_pos)
                eigvecs = np.eye(3)

            obs[i, 6:9] = eigvals
            obs[i, 9:18] = eigvecs.flatten()

            # Detection flag
            obs[i, 18] = 1.0 if result["measurements"][i] is not None else 0.0

            # Neighbor relative positions (permutation-invariant pooling)
            neighbor_rel = []
            for j in range(self.N):
                if j != i:
                    neighbor_rel.append(drone_positions[j] - drone_positions[i])

            if len(neighbor_rel) > 0:
                neighbor_arr = np.array(neighbor_rel)  # (N-1, 3)
                obs[i, 19:22] = np.mean(neighbor_arr, axis=0)   # mean
                obs[i, 22:25] = np.max(neighbor_arr, axis=0)    # max
                obs[i, 25:28] = np.min(neighbor_arr, axis=0)    # min
                obs[i, 28:31] = np.std(neighbor_arr, axis=0)    # std

        return obs

    def _compute_reward(self, result: dict) -> np.ndarray:
        """Compute per-drone reward.

        Components:
        - Covariance:  -w_cov * tr(P_pos) / cov_scale  (shared, minimize filter uncertainty)
        - Distance:    -w_dist * dist / dist_scale  (per-drone, smooth gradient to follow target)
        - Detection:   +w_detection * detected  (per-drone, reward for seeing target)
        - Separation:  +w_separation * min_angle / 90°  (per-drone, angular diversity)

        Returns: (N,) float32 array.
        """
        reward = np.zeros(self.N, dtype=np.float32)

        if not self._filter.initialized:
            reward[:] = -1.0  # small negative constant before init
            return reward

        # Shared covariance-based reward (same for all drones)
        tr_P_pos = self._get_tr_P_pos()
        cov_reward = -self.cfg.w_cov * tr_P_pos / self.cfg.cov_scale

        drone_positions = result["drone_positions"]

        # Precompute unit vectors from target estimate to each drone for angular separation
        consensus_est = self._filter.get_estimate()[:3]
        unit_vecs = []
        for i in range(self.N):
            d = drone_positions[i] - consensus_est
            norm = np.linalg.norm(d)
            unit_vecs.append(d / norm if norm > 1e-3 else np.zeros(3))

        for i in range(self.N):
            local_est = self._filter.get_local_estimate(i)[:3]
            dist = np.linalg.norm(drone_positions[i] - local_est)

            # Smooth distance penalty — gives direct gradient to follow target
            dist_penalty = -self.cfg.w_dist * dist / self.cfg.dist_scale

            # Detection bonus — rewards maintaining visual contact
            detected = 1.0 if result["measurements"][i] is not None else 0.0

            # Angular separation: min angle to any other drone (w.r.t. target estimate)
            min_angle = 180.0
            for j in range(self.N):
                if j != i:
                    cos_a = np.clip(np.dot(unit_vecs[i], unit_vecs[j]), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_a))
                    min_angle = min(min_angle, angle)
            sep_bonus = self.cfg.w_separation * min_angle / 90.0

            reward[i] = (cov_reward
                         + dist_penalty
                         + self.cfg.w_detection * detected
                         + sep_bonus)

        return reward

    def _get_tr_P_pos(self) -> float:
        """Get trace of position covariance block."""
        if not self._filter.initialized:
            return self.cfg.P0_pos * 3.0
        P = self._filter.get_covariance()
        return np.trace(P[:3, :3])

    def close(self):
        if self._aviary is not None:
            self._aviary.close()
            self._aviary = None
