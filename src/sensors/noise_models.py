"""Noise and detection probability models for bearing sensors."""

import numpy as np


def range_dependent_sigma(
    range_m: float, sigma_base: float, range_ref: float
) -> float:
    """Bearing noise standard deviation that scales linearly with range.

    sigma(r) = sigma_base * (r / range_ref)
    """
    return sigma_base * (range_m / range_ref)


def detection_probability(
    range_m: float,
    p_max: float = 0.99,
    range_half: float = 40.0,
    model: str = "range_dependent",
) -> float:
    """Probability of detecting the target at a given range.

    Args:
        range_m: distance to target in meters
        p_max: maximum detection probability (at zero range)
        range_half: range at which detection probability drops to p_max/2
        model: "range_dependent" or "constant"
    """
    if model == "constant":
        return p_max

    # Sigmoid-like decay: p(r) = p_max * exp(-k*r^2) where k chosen so p(range_half) = p_max/2
    k = np.log(2) / (range_half**2)
    return p_max * np.exp(-k * range_m**2)
