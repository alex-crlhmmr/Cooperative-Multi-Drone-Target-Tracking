"""Fisher Information Matrix computation for bearing geometry (shared with V2)."""

import numpy as np
from src.filters.measurement import bearing_jacobian, measurement_noise_cov


def compute_bearing_fim(
    drone_positions: np.ndarray,
    target_est: np.ndarray,
    sigma_base: float,
    range_ref: float,
    detections: list[bool],
) -> np.ndarray:
    """Compute 3x3 position FIM from bearing measurements.

    Args:
        drone_positions: (N, 3)
        target_est: (3,) estimated target position
        sigma_base: bearing noise in radians
        range_ref: reference range for noise scaling
        detections: list of bool — which drones detected the target

    Returns:
        J: (3, 3) Fisher Information Matrix for position
    """
    x_full = np.zeros(6)
    x_full[:3] = target_est
    J = np.zeros((3, 3))

    for i in range(len(drone_positions)):
        if not detections[i]:
            continue
        H = bearing_jacobian(x_full, drone_positions[i])  # (2, 6)
        H_pos = H[:, :3]                                   # (2, 3)
        R = measurement_noise_cov(drone_positions[i], target_est, sigma_base, range_ref)
        R_inv = np.linalg.inv(R)                           # (2, 2)
        J += H_pos.T @ R_inv @ H_pos                       # (3, 3)

    return J
