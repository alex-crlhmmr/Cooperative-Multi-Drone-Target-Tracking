"""PID controller for the surveillance quadrotor (~4kg).

Based on DSLPIDControl from gym-pybullet-drones but with:
- Gains tuned for 4kg, 0.40m arm, kf=5e-7 platform
- PWM-to-RPM mapping for lower RPM range (~2500-7000)
- X-configuration mixer matrix
"""

import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


class SurveillancePIDControl(BaseControl):
    """PID controller for the surveillance drone.

    Same cascaded position→attitude→RPM structure as DSLPIDControl,
    with gains and motor mapping tuned for a 4kg quadrotor.
    """

    def __init__(self, drone_model: DroneModel = DroneModel.SURVEILLANCE, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)

        # --- Position PID gains ---
        # Tuned via step-response sweep (scripts/pid_tuning.py).
        # XY: P=5, D=4 — best trade-off of rise time (1.6s), overshoot (13%),
        #     settling (5.6s), z-coupling (0.5m). Higher P causes attitude saturation.
        # Z:  P=10, D=10 — rise 1.9s, OS 1.7%, settle 3.1s. Fast and well-damped.
        self.P_COEFF_FOR = np.array([5.0, 5.0, 10.0])
        self.I_COEFF_FOR = np.array([0.3, 0.3, 0.5])
        self.D_COEFF_FOR = np.array([4.0, 4.0, 10.0])

        # --- Attitude PID gains ---
        # Scaled from CF2X by angular-acceleration-per-PWM ratio (~2x needed).
        # CF2X: P=[70k, 70k, 60k], D=[20k, 20k, 12k]
        self.P_COEFF_TOR = np.array([140000., 140000., 120000.])
        self.I_COEFF_TOR = np.array([0., 0., 1000.])
        self.D_COEFF_TOR = np.array([40000., 40000., 24000.])

        # --- PWM to RPM mapping ---
        # RPM = SCALE * PWM + CONST
        # Hover RPM ≈ 4427 (at ~39270 PWM → mid-range)
        # Max RPM ≈ 7054 (at 65535 PWM)
        # Min RPM ≈ 2500 (at 20000 PWM)
        self.PWM2RPM_SCALE = 0.1
        self.PWM2RPM_CONST = 500.0
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        # X-configuration mixer (same geometry as CF2X)
        self.MIXER_MATRIX = np.array([
            [-.5, -.5, -1],
            [-.5,  .5,  1],
            [ .5,  .5, -1],
            [ .5, -.5,  1],
        ])

        self.reset()

    def reset(self):
        super().reset()
        self.last_rpy = np.zeros(3)
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._positionControl(
            control_timestep, cur_pos, cur_quat, cur_vel,
            target_pos, target_rpy, target_vel,
        )
        rpm = self._attitudeControl(
            control_timestep, thrust, cur_quat,
            computed_target_rpy, target_rpy_rates,
        )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    def _positionControl(self, control_timestep, cur_pos, cur_quat,
                         cur_vel, target_pos, target_rpy, target_vel):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel

        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -10., 10.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -5., 5.)

        # PID target thrust (in Newtons)
        target_thrust = (np.multiply(self.P_COEFF_FOR, pos_e)
                         + np.multiply(self.I_COEFF_FOR, self.integral_pos_e)
                         + np.multiply(self.D_COEFF_FOR, vel_e)
                         + np.array([0, 0, self.GRAVITY]))

        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        # Desired orientation from thrust direction
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)

        if np.any(np.abs(target_euler) > math.pi):
            print(f"\n[WARN] SurveillancePID step {self.control_counter}: euler angles outside [-pi,pi]")

        return thrust, target_euler, pos_e

    def _attitudeControl(self, control_timestep, thrust, cur_quat,
                         target_euler, target_rpy_rates):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))

        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()

        rot_matrix_e = np.dot(target_rotation.transpose(), cur_rotation) - np.dot(cur_rotation.transpose(), target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])

        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy

        self.integral_rpy_e = self.integral_rpy_e - rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)

        # PID target torques
        target_torques = (-np.multiply(self.P_COEFF_TOR, rot_e)
                          + np.multiply(self.D_COEFF_TOR, rpy_rates_e)
                          + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e))
        target_torques = np.clip(target_torques, -6400, 6400)

        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
