"""V4 Fisher Information Matrix — bearing FIM + difference FIM for per-agent rewards."""

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


def compute_collinear_fim_baseline(
    drone_positions: np.ndarray,
    target_est: np.ndarray,
    sigma_base: float,
    range_ref: float,
    detections: list[bool],
) -> float:
    """Log-det of FIM if all drones were at the centroid (no geometric diversity).

    This baseline represents N collocated drones — they add measurement noise
    reduction (N copies of same measurement) but zero geometric benefit.
    Good multi-drone geometry will exceed this; collinear will match it.
    """
    x_full = np.zeros(6)
    x_full[:3] = target_est

    # Centroid of detecting drones
    detecting_pos = [drone_positions[i] for i in range(len(drone_positions)) if detections[i]]
    if len(detecting_pos) == 0:
        return -30.0
    centroid = np.mean(detecting_pos, axis=0)

    # FIM as if all N detecting drones were at centroid
    H = bearing_jacobian(x_full, centroid)[:, :3]
    R = measurement_noise_cov(centroid, target_est, sigma_base, range_ref)
    R_inv = np.linalg.inv(R)
    J_single = H.T @ R_inv @ H
    J_collinear = J_single * len(detecting_pos)  # N copies of same measurement

    det_col = max(np.linalg.det(J_collinear), 1e-30)
    return float(np.log(det_col))


def compute_difference_fim_rewards(
    drone_positions: np.ndarray,
    target_est: np.ndarray,
    sigma_base: float,
    range_ref: float,
    detections: list[bool],
) -> np.ndarray:
    """Per-agent difference reward: log_det(FIM_all) - log_det(FIM_without_j).

    Measures each drone's marginal information contribution.

    Args:
        drone_positions: (N, 3)
        target_est: (3,) estimated target position
        sigma_base: bearing noise in radians
        range_ref: reference range for noise scaling
        detections: list of bool — which drones detected the target

    Returns:
        rewards: (N,) per-drone marginal FIM contribution
    """
    N = len(drone_positions)
    x_full = np.zeros(6)
    x_full[:3] = target_est

    # Compute per-drone FIM contributions
    per_drone_fim = []
    for i in range(N):
        if not detections[i]:
            per_drone_fim.append(np.zeros((3, 3)))
            continue
        H = bearing_jacobian(x_full, drone_positions[i])[:, :3]  # (2, 3)
        R = measurement_noise_cov(drone_positions[i], target_est, sigma_base, range_ref)
        R_inv = np.linalg.inv(R)
        per_drone_fim.append(H.T @ R_inv @ H)

    # Full FIM = sum of all per-drone contributions
    fim_all = sum(per_drone_fim)
    det_all = np.linalg.det(fim_all)
    log_det_all = np.log(max(det_all, 1e-30))

    # Per-agent difference: log_det(FIM_all) - log_det(FIM_without_j)
    rewards = np.zeros(N, dtype=np.float32)
    for j in range(N):
        fim_without_j = fim_all - per_drone_fim[j]
        det_without = np.linalg.det(fim_without_j)
        log_det_without = np.log(max(det_without, 1e-30))
        rewards[j] = log_det_all - log_det_without

    return rewards
