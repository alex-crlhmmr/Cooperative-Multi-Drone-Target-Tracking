"""V7 TrackEnv — plain chase+offset tracking for eval (no RL, no angular repulsion).

Used only during eval: spread with RL, then track with chase+offset.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from .config import V7Config


class TrackEnv(gym.Env):
    """Tracking env with plain chase+offset. Used for eval only."""

    def __init__(self, cfg: V7Config, seed: int | None = None,
                 initial_positions: np.ndarray | None = None):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.num_drones
        self._seed = seed or cfg.seed
        self._rng = np.random.default_rng(self._seed)

        self.critic_obs_dim = cfg.track_critic_obs_dim
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
        self._initial_positions = initial_positions
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

        # Initial step
        drone_positions = self._get_drone_positions()
        result = self._aviary.step_tracking(
            tracker_targets=drone_positions,
            tracker_target_vels=np.zeros((self.N, 3)),
        )
        self._last_result = result

        if not self._filter.initialized:
            self._filter.update(result["measurements"], result["drone_positions"])

        obs = np.zeros((self.N, self.critic_obs_dim), dtype=np.float32)
        return obs, {"result": result}

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

    def step(self, action: np.ndarray):
        drone_positions = self._get_drone_positions()

        # Pure chase+offset (action ignored — zero action expected)
        velocity = self._chase_controller(drone_positions)

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
            return obs, np.zeros(self.N), True, False, {"result": result}

        if self._filter.initialized:
            self._filter.predict()
        self._filter.update(result["measurements"], result["drone_positions"])

        self._step_count += 1
        terminated = result.get("done", False)

        obs = np.zeros((self.N, self.critic_obs_dim), dtype=np.float32)
        info = {
            "result": result,
            "filter_initialized": self._filter.initialized,
            "tr_P_pos": self._get_tr_P_pos(),
        }
        return obs, np.zeros(self.N), terminated, False, info

    def _chase_controller(self, drone_positions: np.ndarray) -> np.ndarray:
        """Plain chase+offset PD controller."""
        base_vel = np.zeros((self.N, 3))
        if not self._filter.initialized:
            return base_vel

        for i in range(self.N):
            local_est = self._filter.get_local_estimate(i)
            dir_to_est = local_est[:3] - drone_positions[i]
            dist = np.linalg.norm(dir_to_est)

            if dist < 1e-3:
                base_vel[i] = local_est[3:6]
                continue

            unit = dir_to_est / dist
            radial_error = dist - self.cfg.track_base_R_desired
            approach_speed = np.clip(
                self.cfg.track_base_Kp * radial_error,
                -self.cfg.v_max * 0.3,
                self.cfg.v_max * 0.5,
            )
            base_vel[i] = unit * approach_speed + local_est[3:6]

            speed = np.linalg.norm(base_vel[i])
            if speed > self.cfg.v_max:
                base_vel[i] *= self.cfg.v_max / speed

        return base_vel

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
