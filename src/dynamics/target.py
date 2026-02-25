"""Target motion models for 3D tracking.

State vector: x = [px, py, pz, vx, vy, vz] (6D)
"""

import numpy as np


class ConstantVelocityModel:
    """Constant velocity (CV) motion model in 3D.

    F_cv = [[I3, dt*I3],
            [0,   I3  ]]
    """

    def __init__(self, dt: float, sigma_a: float):
        self.dt = dt
        self.sigma_a = sigma_a
        self.state_dim = 6

    def F(self, dt: float | None = None) -> np.ndarray:
        dt = dt if dt is not None else self.dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def Q(self, dt: float | None = None) -> np.ndarray:
        """Piecewise constant white noise acceleration model."""
        dt = dt if dt is not None else self.dt
        q = self.sigma_a ** 2
        # Block structure for each axis
        q11 = q * dt**4 / 4
        q12 = q * dt**3 / 2
        q22 = q * dt**2
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q11
            Q[i, i + 3] = q12
            Q[i + 3, i] = q12
            Q[i + 3, i + 3] = q22
        return Q

    def predict(self, x: np.ndarray, dt: float | None = None) -> np.ndarray:
        return self.F(dt) @ x

    def sample(self, x: np.ndarray, dt: float | None = None) -> np.ndarray:
        dt = dt if dt is not None else self.dt
        x_pred = self.predict(x, dt)
        noise = np.random.multivariate_normal(np.zeros(6), self.Q(dt))
        return x_pred + noise


class CoordinatedTurnModel:
    """Coordinated turn (CT) model in 3D.

    Rotation in the horizontal (xy) plane with turn rate omega.
    Vertical motion remains constant velocity.
    """

    def __init__(self, dt: float, sigma_a: float):
        self.dt = dt
        self.sigma_a = sigma_a
        self.state_dim = 6

    def F(self, omega: float, dt: float | None = None) -> np.ndarray:
        dt = dt if dt is not None else self.dt
        F = np.zeros((6, 6))

        if abs(omega) < 1e-10:
            # Degenerate to CV
            F = np.eye(6)
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt
        else:
            sw = np.sin(omega * dt)
            cw = np.cos(omega * dt)
            # x
            F[0, 0] = 1.0
            F[0, 3] = sw / omega
            F[0, 4] = -(1 - cw) / omega
            # y
            F[1, 1] = 1.0
            F[1, 3] = (1 - cw) / omega
            F[1, 4] = sw / omega
            # z (constant velocity)
            F[2, 2] = 1.0
            F[2, 5] = dt
            # vx
            F[3, 3] = cw
            F[3, 4] = -sw
            # vy
            F[4, 3] = sw
            F[4, 4] = cw
            # vz
            F[5, 5] = 1.0
        return F

    def Q(self, dt: float | None = None) -> np.ndarray:
        dt = dt if dt is not None else self.dt
        q = self.sigma_a ** 2
        Q = np.zeros((6, 6))
        q11 = q * dt**4 / 4
        q12 = q * dt**3 / 2
        q22 = q * dt**2
        for i in range(3):
            Q[i, i] = q11
            Q[i, i + 3] = q12
            Q[i + 3, i] = q12
            Q[i + 3, i + 3] = q22
        return Q

    def predict(self, x: np.ndarray, omega: float, dt: float | None = None) -> np.ndarray:
        return self.F(omega, dt) @ x

    def sample(self, x: np.ndarray, omega: float, dt: float | None = None) -> np.ndarray:
        dt = dt if dt is not None else self.dt
        x_pred = self.predict(x, omega, dt)
        noise = np.random.multivariate_normal(np.zeros(6), self.Q(dt))
        return x_pred + noise
