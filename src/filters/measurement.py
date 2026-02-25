"""Measurement model for bearing-only target tracking.

Shared by EKF, UKF, and PF. Provides:
- Bearing measurement function h(x, drone_pos) -> (az, el)
- Jacobian H (2x6) for EKF linearization
- Range-dependent measurement noise covariance R
- Multi-sensor stacking (all drones -> single measurement vector)
- Angle wrapping utilities
- Triangulation for filter initialization
"""

import numpy as np


def bearing_measurement(x: np.ndarray, drone_pos: np.ndarray) -> np.ndarray:
    """Compute bearing (azimuth, elevation) from drone to target.

    Args:
        x: (6,) target state [px, py, pz, vx, vy, vz]
        drone_pos: (3,) drone position

    Returns:
        (2,) array [azimuth, elevation] in radians
    """
    dx = x[0] - drone_pos[0]
    dy = x[1] - drone_pos[1]
    dz = x[2] - drone_pos[2]
    r_xy = np.sqrt(dx**2 + dy**2)
    az = np.arctan2(dy, dx)
    el = np.arctan2(dz, r_xy)
    return np.array([az, el])


def bearing_jacobian(x: np.ndarray, drone_pos: np.ndarray) -> np.ndarray:
    """Jacobian of bearing measurement w.r.t. state (2x6).

    H = d[az,el]/d[px,py,pz,vx,vy,vz]
    Only position columns are nonzero.
    """
    dx = x[0] - drone_pos[0]
    dy = x[1] - drone_pos[1]
    dz = x[2] - drone_pos[2]

    r_xy_sq = dx**2 + dy**2
    r_xy = np.sqrt(r_xy_sq)
    r_sq = r_xy_sq + dz**2

    # Guard against degenerate geometry
    if r_xy_sq < 1e-12 or r_sq < 1e-12:
        return np.zeros((2, 6))

    H = np.zeros((2, 6))
    # d(az)/d(px) = -dy / r_xy^2
    H[0, 0] = -dy / r_xy_sq
    # d(az)/d(py) = dx / r_xy^2
    H[0, 1] = dx / r_xy_sq

    # d(el)/d(px) = -dx*dz / (r^2 * r_xy)
    H[1, 0] = -dx * dz / (r_sq * r_xy)
    # d(el)/d(py) = -dy*dz / (r^2 * r_xy)
    H[1, 1] = -dy * dz / (r_sq * r_xy)
    # d(el)/d(pz) = r_xy / r^2
    H[1, 2] = r_xy / r_sq

    return H


def measurement_noise_cov(
    drone_pos: np.ndarray,
    target_pos: np.ndarray,
    sigma_base: float,
    range_ref: float,
) -> np.ndarray:
    """Range-dependent measurement noise covariance R (2x2).

    sigma(r) = sigma_base * (r / range_ref)
    R = diag([sigma^2, sigma^2])
    """
    r = np.linalg.norm(target_pos[:3] - drone_pos[:3])
    r = max(r, 1e-6)  # avoid zero
    sigma = sigma_base * (r / range_ref)
    return np.diag([sigma**2, sigma**2])


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wrap_innovation(innovation: np.ndarray) -> np.ndarray:
    """Wrap measurement innovation: azimuth to [-pi, pi], elevation left as-is.

    For stacked innovations (2k,), wraps every even index (azimuth).
    """
    out = innovation.copy()
    for i in range(0, len(out), 2):
        out[i] = wrap_angle(out[i])
    return out


def stack_measurements(
    measurements: list,
    drone_positions: np.ndarray,
    x: np.ndarray,
    sigma_base: float,
    range_ref: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack all valid measurements from multiple drones.

    Args:
        measurements: list of (az, el) tuples or None per drone
        drone_positions: (N, 3) drone positions
        x: (6,) current state estimate
        sigma_base: base bearing noise (rad)
        range_ref: reference range for noise scaling

    Returns:
        z_stack: (2k,) stacked measurements
        h_stack: (2k,) stacked predicted measurements h(x)
        H_stack: (2k, 6) stacked Jacobians
        R_stack: (2k, 2k) block-diagonal noise covariance
        valid_positions: (k, 3) positions of drones with valid measurements
    """
    z_list = []
    h_list = []
    H_list = []
    R_list = []
    pos_list = []

    for i, m in enumerate(measurements):
        if m is None:
            continue
        dp = drone_positions[i]
        z_list.append(np.array(m))
        h_list.append(bearing_measurement(x, dp))
        H_list.append(bearing_jacobian(x, dp))
        R_list.append(measurement_noise_cov(dp, x[:3], sigma_base, range_ref))
        pos_list.append(dp)

    if len(z_list) == 0:
        return (np.array([]), np.array([]), np.zeros((0, 6)),
                np.zeros((0, 0)), np.zeros((0, 3)))

    z_stack = np.concatenate(z_list)
    h_stack = np.concatenate(h_list)
    H_stack = np.vstack(H_list)
    R_stack = np.block([
        [R_list[i] if i == j else np.zeros((2, 2))
         for j in range(len(R_list))]
        for i in range(len(R_list))
    ])
    valid_positions = np.array(pos_list)

    return z_stack, h_stack, H_stack, R_stack, valid_positions


def triangulate_position(
    measurements: list,
    drone_positions: np.ndarray,
) -> np.ndarray | None:
    """Least-squares ray intersection from bearing measurements.

    Each (az, el) measurement defines a ray from the drone position.
    We find the 3D point closest to all rays (least-squares).

    Requires >= 2 valid measurements for a solution.

    Returns:
        (3,) estimated target position, or None if insufficient measurements.
    """
    rays = []
    origins = []

    for i, m in enumerate(measurements):
        if m is None:
            continue
        az, el = m
        # Unit direction vector from bearing
        d = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])
        origins.append(drone_positions[i])
        rays.append(d)

    if len(rays) < 2:
        return None

    # Least-squares closest point to all rays
    # For each ray: p = o + t*d, minimize sum ||p - (o_i + t_i * d_i)||^2
    # Equivalent to: sum (I - d_i d_i^T)(p - o_i) = 0
    # => (sum (I - d_i d_i^T)) p = sum (I - d_i d_i^T) o_i
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I3 = np.eye(3)

    for o, d in zip(origins, rays):
        proj = I3 - np.outer(d, d)
        A += proj
        b += proj @ o

    # Regularize in case of near-collinear rays
    A += 1e-6 * I3

    try:
        pos = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: average of origins
        pos = np.mean(origins, axis=0)

    return pos


def initialize_filter_state(
    measurements: list,
    drone_positions: np.ndarray,
    P0_pos: float = 10000.0,
    P0_vel: float = 100.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Initialize filter state from bearing measurements.

    Uses triangulation for position, zero velocity assumption.

    Returns:
        (x0, P0) or None if triangulation fails.
    """
    pos = triangulate_position(measurements, drone_positions)
    if pos is None:
        return None

    x0 = np.zeros(6)
    x0[:3] = pos

    P0 = np.diag([P0_pos, P0_pos, P0_pos, P0_vel, P0_vel, P0_vel])

    return x0, P0
