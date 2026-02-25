"""Bearing sensor model for 3D target tracking.

Returns azimuth and elevation angles from a drone to a target,
with range-dependent noise and FOV masking.
"""

import numpy as np
from .noise_models import range_dependent_sigma, detection_probability


class BearingSensor:
    """Geometric bearing sensor (fast, for MC runs).

    Camera is gimballed: it points at the filter's estimate of target position.
    Measurement is (azimuth, elevation) of the TRUE target relative to the drone.
    Returns None if target is outside FOV or not detected.
    """

    def __init__(
        self,
        fov_half_deg: float = 60.0,
        sigma_bearing_deg: float = 2.0,
        max_range: float = 50.0,
        range_ref: float = 10.0,
        p_detect_max: float = 0.99,
        p_detect_range_half: float = 40.0,
        detection_model: str = "range_dependent",
        rng: np.random.Generator | None = None,
    ):
        self.fov_half = np.deg2rad(fov_half_deg)
        self.sigma_base = np.deg2rad(sigma_bearing_deg)
        self.max_range = max_range
        self.range_ref = range_ref
        self.p_detect_max = p_detect_max
        self.p_detect_range_half = p_detect_range_half
        self.detection_model = detection_model
        self.rng = rng or np.random.default_rng()

    def measure(
        self,
        drone_pos: np.ndarray,
        target_true_pos: np.ndarray,
        target_estimated_pos: np.ndarray,
    ) -> tuple[float, float] | None:
        """Take a bearing measurement.

        Args:
            drone_pos: (3,) drone position
            target_true_pos: (3,) true target position
            target_estimated_pos: (3,) filter's estimate of target position (for gimbal)

        Returns:
            (azimuth, elevation) in radians, or None if not detected.
        """
        # Direction vectors
        true_dir = target_true_pos - drone_pos
        cam_dir = target_estimated_pos - drone_pos

        true_range = np.linalg.norm(true_dir)
        cam_range = np.linalg.norm(cam_dir)

        if true_range < 1e-6 or cam_range < 1e-6:
            return None

        # Normalize
        true_dir_n = true_dir / true_range
        cam_dir_n = cam_dir / cam_range

        # FOV check: angle between camera pointing direction and true target
        cos_angle = np.clip(np.dot(cam_dir_n, true_dir_n), -1.0, 1.0)
        angle_off = np.arccos(cos_angle)
        if angle_off > self.fov_half:
            return None

        # Range check
        if true_range > self.max_range:
            return None

        # Detection probability
        p_det = detection_probability(
            true_range, self.p_detect_max, self.p_detect_range_half,
            model=self.detection_model,
        )
        if self.rng.random() > p_det:
            return None

        # True bearing angles
        az = np.arctan2(true_dir[1], true_dir[0])
        el = np.arctan2(true_dir[2], np.sqrt(true_dir[0]**2 + true_dir[1]**2))

        # Range-dependent noise
        sigma = range_dependent_sigma(true_range, self.sigma_base, self.range_ref)
        az_noisy = az + self.rng.normal(0, sigma)
        el_noisy = el + self.rng.normal(0, sigma)

        return (az_noisy, el_noisy)

    def true_bearing(
        self, drone_pos: np.ndarray, target_pos: np.ndarray
    ) -> tuple[float, float]:
        """Compute noiseless bearing (for testing/viz)."""
        d = target_pos - drone_pos
        az = np.arctan2(d[1], d[0])
        el = np.arctan2(d[2], np.sqrt(d[0]**2 + d[1]**2))
        return (az, el)
