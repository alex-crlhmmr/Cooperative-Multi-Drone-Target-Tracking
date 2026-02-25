"""Custom tracking aviary: N tracker drones + 1 target drone.

Subclasses CtrlAviary from gym-pybullet-drones.
Target follows a scripted trajectory via PID.
Trackers are controlled externally (PID waypoints for Phase 0-2, RL for Phase 3).
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

# Auto-patch gym-pybullet-drones with surveillance drone before any imports that use DroneModel
from src.env.patch_drone_model import ensure_surveillance_drone
ensure_surveillance_drone()

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from src.control.surveillance_pid import SurveillancePIDControl
from src.dynamics.trajectories import generate_trajectory
from src.sensors.bearing import BearingSensor


def spawn_in_hollow_sphere(
    center: np.ndarray,
    r_min: float,
    r_max: float,
    n: int,
    rng: np.random.Generator,
    min_altitude: float = 20.0,
) -> np.ndarray:
    """Spawn n points uniformly in a hollow sphere around center.

    Points are guaranteed to have z >= min_altitude.
    Also enforces that drones spawn at least min_altitude above ground
    regardless of center height (prevents spawning too close to floor
    even when target is low).
    """
    positions = np.zeros((n, 3))
    for i in range(n):
        while True:
            # Uniform in shell: r³ uniform between r_min³ and r_max³
            r = (rng.uniform(r_min**3, r_max**3)) ** (1.0 / 3.0)
            # Uniform direction on sphere
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            dx = r * sin_phi * np.cos(theta)
            dy = r * sin_phi * np.sin(theta)
            dz = r * cos_phi
            pos = center + np.array([dx, dy, dz])
            if pos[2] >= min_altitude:
                positions[i] = pos
                break
    return positions


class TrackingAviary(CtrlAviary):
    """Multi-drone tracking environment.

    Drone indices:
        0..num_trackers-1 = tracker drones
        num_trackers = target drone (follows scripted trajectory)
    """

    def __init__(
        self,
        num_trackers: int = 3,
        tracker_positions: np.ndarray | None = None,
        target_initial_pos: np.ndarray = np.array([0.0, 0.0, 50.0]),
        target_speed: float = 12.0,
        target_trajectory: str = "multi_segment",
        target_sigma_a: float = 0.3,
        evasive_params: dict | None = None,
        episode_length: int = 200,
        sensor_config: dict | None = None,
        pyb_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = True,
        physics: Physics = Physics.PYB_GND_DRAG_DW,
        rng: np.random.Generator | None = None,
    ):
        self.num_trackers = num_trackers
        self.target_speed = target_speed
        self.target_trajectory_type = target_trajectory
        self.target_sigma_a = target_sigma_a
        self.evasive_params = evasive_params or {}
        self.target_initial_pos = np.array(target_initial_pos, dtype=np.float64)
        self.episode_length = episode_length
        self._rng = rng or np.random.default_rng()

        # Tracker initial positions: random hollow sphere if not provided
        if tracker_positions is None:
            tracker_positions = spawn_in_hollow_sphere(
                center=self.target_initial_pos,
                r_min=50.0, r_max=75.0,
                n=num_trackers,
                rng=self._rng,
                min_altitude=5.0,
            )

        # All drone positions: trackers + target
        num_drones = num_trackers + 1
        initial_xyzs = np.vstack([tracker_positions, target_initial_pos.reshape(1, 3)])
        initial_rpys = np.zeros((num_drones, 3))

        super().__init__(
            drone_model=DroneModel.SURVEILLANCE,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )

        # PID controllers for all drones
        self._controllers = [
            SurveillancePIDControl(drone_model=DroneModel.SURVEILLANCE)
            for _ in range(num_drones)
        ]

        # Bearing sensor
        sc = sensor_config or {}
        self.sensor = BearingSensor(
            fov_half_deg=sc.get("fov_half_deg", 60.0),
            sigma_bearing_deg=sc.get("sigma_bearing_deg", 2.0),
            max_range=sc.get("max_range", 50.0),
            range_ref=sc.get("range_ref", 10.0),
            p_detect_max=sc.get("p_detect_max", 0.99),
            p_detect_range_half=sc.get("p_detect_range_half", 40.0),
            detection_model=sc.get("detection_prob_model", "range_dependent"),
            rng=self._rng,
        )

        # Generate target trajectory
        self.target_traj = None  # set in reset
        self._step_count = 0

        # Viz line IDs for PyBullet debug drawing
        self._bearing_lines = []  # persistent IDs, updated in-place via replaceItemUniqueId
        self._target_trail = []   # position history for trail segments

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._step_count = 0

        # Generate fresh target trajectory
        dt = self.CTRL_TIMESTEP
        self.target_traj = generate_trajectory(
            traj_type=self.target_trajectory_type,
            num_steps=self.episode_length + 50,  # extra buffer
            dt=dt,
            initial_pos=self.target_initial_pos,
            speed=self.target_speed,
            sigma_a=self.target_sigma_a,
            rng=self._rng,
            evasive_params=self.evasive_params,
        )

        # Clear debug lines
        self._clear_debug_lines()

        info["target_traj"] = self.target_traj
        return obs, info

    def step_tracking(
        self,
        tracker_targets: np.ndarray | None = None,
        tracker_target_vels: np.ndarray | None = None,
        target_estimate: np.ndarray | None = None,
    ) -> dict:
        """High-level step: move drones via PID, get bearing measurements.

        Args:
            tracker_targets: (num_trackers, 3) target positions for tracker drones.
                If None, trackers hold their current position.
            tracker_target_vels: (num_trackers, 3) feedforward velocities for trackers.
                If None, zero velocity (position-only control).
            target_estimate: (3,) filter's estimate of target position (for gimbal).
                If None, uses true target physical position (perfect gimbal).

        Returns:
            Dictionary with step results.
        """
        if self._step_count >= len(self.target_traj):
            return {"done": True}

        # Target waypoint from trajectory
        target_waypoint = self.target_traj[self._step_count, :3].copy()

        # Default: trackers hold position
        if tracker_targets is None:
            tracker_targets = np.array([
                self._getDroneStateVector(i)[:3] for i in range(self.num_trackers)
            ])

        # Default tracker velocities: zero (position-only)
        if tracker_target_vels is None:
            tracker_target_vels = np.zeros((self.num_trackers, 3))

        # Compute RPMs via PID for all drones
        action = np.zeros((self.NUM_DRONES, 4))

        for i in range(self.num_trackers):
            state = self._getDroneStateVector(i)
            action[i, :], _, _ = self._controllers[i].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=tracker_targets[i],
                target_vel=tracker_target_vels[i],
            )

        # Target drone follows trajectory
        target_state = self._getDroneStateVector(self.num_trackers)
        action[self.num_trackers, :], _, _ = self._controllers[self.num_trackers].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=target_state[:3],
            cur_quat=target_state[3:7],
            cur_vel=target_state[10:13],
            cur_ang_vel=target_state[13:16],
            target_pos=target_waypoint,
            target_vel=self.target_traj[self._step_count, 3:6] if self._step_count < len(self.target_traj) else np.zeros(3),
        )

        # Step simulation
        obs, reward, terminated, truncated, info = self.step(action)

        # Get current positions
        drone_positions = np.array([
            self._getDroneStateVector(i)[:3] for i in range(self.num_trackers)
        ])
        target_true_pos = self._getDroneStateVector(self.num_trackers)[:3]

        # Take bearing measurements
        # Gimbal: use provided estimate, or actual physical target position
        gimbal_target = target_estimate if target_estimate is not None else target_true_pos
        measurements = []
        for i in range(self.num_trackers):
            m = self.sensor.measure(
                drone_pos=drone_positions[i],
                target_true_pos=target_true_pos,
                target_estimated_pos=gimbal_target,
            )
            measurements.append(m)

        # Draw debug visualization
        if self.GUI:
            self._draw_debug(drone_positions, target_true_pos, measurements)

        self._step_count += 1

        return {
            "done": self._step_count >= self.episode_length,
            "step": self._step_count,
            "drone_positions": drone_positions,
            "target_true_pos": target_true_pos,
            "target_true_state": self.target_traj[min(self._step_count, len(self.target_traj) - 1)],
            "measurements": measurements,
            "obs": obs,
        }

    def _draw_debug(
        self,
        drone_positions: np.ndarray,
        target_pos: np.ndarray,
        measurements: list,
    ):
        """Draw bearing lines and trails in PyBullet GUI.

        Uses replaceItemUniqueId for per-frame lines (zero allocation after first frame)
        and lifeTime-based trail segments (no manual cleanup needed).
        """
        colors = [
            [0, 0.7, 1],    # cyan for drone 0
            [1, 0.5, 0],    # orange for drone 1
            [0, 1, 0.3],    # green for drone 2
        ]

        # First frame: create persistent line IDs for bearing rays + gray lines
        if not self._bearing_lines:
            for i in range(self.num_trackers):
                # Bearing ray
                bid = p.addUserDebugLine(
                    [0, 0, 0], [0, 0, 0],
                    lineColorRGB=colors[i % len(colors)],
                    lineWidth=1.5, lifeTime=0,
                    physicsClientId=self.CLIENT)
                self._bearing_lines.append(bid)
                # Gray line to target
                gid = p.addUserDebugLine(
                    [0, 0, 0], [0, 0, 0],
                    lineColorRGB=[0.5, 0.5, 0.5],
                    lineWidth=0.5, lifeTime=0,
                    physicsClientId=self.CLIENT)
                self._bearing_lines.append(gid)

        # Update per-frame lines in-place (no remove/add overhead)
        for i in range(self.num_trackers):
            bearing_id = self._bearing_lines[i * 2]
            gray_id = self._bearing_lines[i * 2 + 1]

            if measurements[i] is not None:
                az, el = measurements[i]
                ray_len = np.linalg.norm(target_pos - drone_positions[i]) * 1.2
                ray_end = drone_positions[i] + ray_len * np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el),
                ])
                p.addUserDebugLine(
                    drone_positions[i].tolist(), ray_end.tolist(),
                    lineColorRGB=colors[i % len(colors)],
                    lineWidth=1.5, lifeTime=0,
                    replaceItemUniqueId=bearing_id,
                    physicsClientId=self.CLIENT)
            else:
                # Hide: zero-length line
                p.addUserDebugLine(
                    [0, 0, 0], [0, 0, 0],
                    replaceItemUniqueId=bearing_id,
                    physicsClientId=self.CLIENT)

            # Gray line from drone to true target
            p.addUserDebugLine(
                drone_positions[i].tolist(), target_pos.tolist(),
                lineColorRGB=[0.5, 0.5, 0.5],
                lineWidth=0.5, lifeTime=0,
                replaceItemUniqueId=gray_id,
                physicsClientId=self.CLIENT)

        # Target trail — use lifeTime so segments auto-expire, no cleanup needed
        if len(self._target_trail) > 0:
            p.addUserDebugLine(
                self._target_trail[-1].tolist(), target_pos.tolist(),
                lineColorRGB=[1, 0, 0], lineWidth=2,
                lifeTime=3.0,  # auto-expire after 3 seconds
                physicsClientId=self.CLIENT)
        self._target_trail.append(target_pos.copy())

    def _clear_debug_lines(self):
        """Remove all debug drawings."""
        for line_id in self._bearing_lines:
            p.removeUserDebugItem(line_id, physicsClientId=self.CLIENT)
        self._bearing_lines = []
        self._target_trail = []

    @property
    def target_index(self) -> int:
        return self.num_trackers
