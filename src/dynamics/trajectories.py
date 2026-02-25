"""Trajectory generators for the target in full 3D.

Each generator returns a (T, 6) array of true states [px, py, pz, vx, vy, vz].
All trajectories maneuver in 3D — climbs, dives, banking turns, altitude changes.
"""

import numpy as np
from .target import ConstantVelocityModel, CoordinatedTurnModel


# Minimum altitude — target trajectory won't go below this.
# Must be high enough that PID tracking lag doesn't cause ground contact.
# At 12 m/s with climb_rate_factor=0.3, max dive ≈ 3.6 m/s → ~15m overshoot.
MIN_ALT = 50.0


def generate_trajectory(
    traj_type: str,
    num_steps: int,
    dt: float,
    initial_pos: np.ndarray,
    speed: float,
    sigma_a: float = 0.0,
    rng: np.random.Generator | None = None,
    evasive_params: dict | None = None,
) -> np.ndarray:
    """Dispatch to the appropriate trajectory generator."""
    rng = rng or np.random.default_rng()
    generators = {
        "straight": _straight_line,
        "single_turn": _single_turn,
        "multi_segment": _multi_segment,
        "evasive": _evasive,
    }
    if traj_type not in generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}. Choose from {list(generators)}")
    if traj_type == "evasive" and evasive_params:
        traj = _evasive(num_steps, dt, initial_pos, speed, sigma_a, rng, **evasive_params)
    else:
        traj = generators[traj_type](num_steps, dt, initial_pos, speed, sigma_a, rng)
    _clamp_altitude(traj)
    return traj


def _random_3d_velocity(speed: float, rng: np.random.Generator,
                        pitch_range_deg: tuple[float, float] = (-25.0, 25.0)) -> np.ndarray:
    """Random 3D velocity vector with bounded pitch angle."""
    heading = rng.uniform(0, 2 * np.pi)
    pitch = rng.uniform(np.deg2rad(pitch_range_deg[0]), np.deg2rad(pitch_range_deg[1]))
    vx = speed * np.cos(pitch) * np.cos(heading)
    vy = speed * np.cos(pitch) * np.sin(heading)
    vz = speed * np.sin(pitch)
    return np.array([vx, vy, vz])


def _clamp_altitude(states: np.ndarray):
    """Enforce minimum altitude and limit descent rate near floor.

    Two-layer protection:
    1. Hard floor at MIN_ALT with velocity bounce
    2. Soft ceiling on descent rate: as altitude approaches MIN_ALT,
       vz is smoothly clamped upward to prevent PID overshoot.
    """
    # Buffer zone: start limiting descent rate at 2x MIN_ALT
    buffer_alt = MIN_ALT * 2.0

    for t in range(states.shape[0]):
        alt = states[t, 2]
        vz = states[t, 5]

        # Soft descent rate limit in buffer zone
        if alt < buffer_alt and vz < 0:
            # Linear taper: at buffer_alt, full vz allowed; at MIN_ALT, vz=0
            frac = max(0.0, (alt - MIN_ALT) / (buffer_alt - MIN_ALT))
            states[t, 5] = vz * frac

        # Hard floor
        if states[t, 2] < MIN_ALT:
            states[t, 2] = MIN_ALT
            if states[t, 5] < 0:
                states[t, 5] = abs(states[t, 5]) * 0.3  # soft bounce


def _step_cv(cv: ConstantVelocityModel, x: np.ndarray, sigma_a: float) -> np.ndarray:
    return cv.sample(x) if sigma_a > 0 else cv.predict(x)


def _step_ct(ct: CoordinatedTurnModel, x: np.ndarray, omega: float,
             sigma_a: float) -> np.ndarray:
    return ct.sample(x, omega) if sigma_a > 0 else ct.predict(x, omega)


def _straight_line(
    num_steps: int, dt: float, initial_pos: np.ndarray, speed: float,
    sigma_a: float, rng: np.random.Generator,
) -> np.ndarray:
    """Constant velocity in a random 3D direction — climbs/dives included."""
    vel = _random_3d_velocity(speed, rng)
    x0 = np.array([*initial_pos, *vel])
    cv = ConstantVelocityModel(dt, sigma_a)

    states = np.zeros((num_steps, 6))
    states[0] = x0
    for t in range(1, num_steps):
        states[t] = _step_cv(cv, states[t - 1], sigma_a)
    return states


def _single_turn(
    num_steps: int, dt: float, initial_pos: np.ndarray, speed: float,
    sigma_a: float, rng: np.random.Generator,
) -> np.ndarray:
    """Straight → 3D banking turn → straight, with altitude change during turn."""
    vel = _random_3d_velocity(speed, rng, pitch_range_deg=(-15.0, 15.0))
    x0 = np.array([*initial_pos, *vel])

    cv = ConstantVelocityModel(dt, sigma_a)
    ct = CoordinatedTurnModel(dt, sigma_a)
    turn_rate = rng.choice([-1, 1]) * rng.uniform(0.05, 0.3)

    # Vertical rate change during the turn
    vz_delta = rng.uniform(-0.3, 0.3) * speed

    t_straight1 = num_steps // 3
    t_turn = num_steps // 3

    states = np.zeros((num_steps, 6))
    states[0] = x0

    for t in range(1, t_straight1):
        states[t] = _step_cv(cv, states[t - 1], sigma_a)

    # Add vz change at turn start
    if t_straight1 > 0:
        states[t_straight1 - 1, 5] += vz_delta

    for t in range(t_straight1, t_straight1 + t_turn):
        states[t] = _step_ct(ct, states[t - 1], turn_rate, sigma_a)

    # Remove vz offset after turn
    if t_straight1 + t_turn - 1 > 0 and t_straight1 + t_turn - 1 < num_steps:
        states[t_straight1 + t_turn - 1, 5] -= vz_delta * 0.5

    for t in range(t_straight1 + t_turn, num_steps):
        states[t] = _step_cv(cv, states[t - 1], sigma_a)

    return states


def _multi_segment(
    num_steps: int, dt: float, initial_pos: np.ndarray, speed: float,
    sigma_a: float, rng: np.random.Generator,
) -> np.ndarray:
    """Random 3D waypoints with banking turns between them.

    Each segment has a random heading, pitch, and speed.
    Transitions use coordinated turns in XY and linear vz ramps.
    """
    num_segments = rng.integers(3, 7)

    # Divide time into segments
    cuts = sorted(rng.choice(range(20, num_steps - 10), size=num_segments - 1, replace=False))
    boundaries = [0] + list(cuts) + [num_steps]
    segment_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

    cv = ConstantVelocityModel(dt, sigma_a)
    ct = CoordinatedTurnModel(dt, sigma_a)

    # Per-segment: random 3D heading + speed
    vels = [_random_3d_velocity(speed * rng.uniform(0.5, 1.5), rng) for _ in segment_lengths]

    states = np.zeros((num_steps, 6))
    states[0] = np.array([*initial_pos, *vels[0]])

    idx = 1
    for seg_i, seg_len in enumerate(segment_lengths):
        if seg_i > 0:
            # Transition: turn in XY + ramp vz
            prev_vel = states[idx - 1, 3:6]
            target_vel = vels[seg_i]

            # XY heading change → turn rate
            prev_heading = np.arctan2(prev_vel[1], prev_vel[0])
            target_heading = np.arctan2(target_vel[1], target_vel[0])
            dh = (target_heading - prev_heading + np.pi) % (2 * np.pi) - np.pi

            turn_steps = min(max(int(seg_len * 0.3), 8), seg_len)
            omega = dh / (turn_steps * dt) if turn_steps > 0 else 0.0

            # Vz ramp
            vz_start = prev_vel[2]
            vz_end = target_vel[2]

            for k in range(turn_steps):
                if idx >= num_steps:
                    break
                states[idx] = _step_ct(ct, states[idx - 1], omega, sigma_a)
                # Linearly ramp vz during turn
                frac = k / max(turn_steps - 1, 1)
                states[idx, 5] = vz_start + (vz_end - vz_start) * frac
                idx += 1

            # Set the target XY speed after turn
            if idx < num_steps:
                xy_speed = np.linalg.norm(target_vel[:2])
                h = target_heading
                states[idx - 1, 3] = xy_speed * np.cos(h)
                states[idx - 1, 4] = xy_speed * np.sin(h)
                states[idx - 1, 5] = vz_end

        # Straight portion
        seg_end = boundaries[seg_i + 1] if seg_i + 1 < len(boundaries) else num_steps
        while idx < min(seg_end, num_steps):
            states[idx] = _step_cv(cv, states[idx - 1], sigma_a)
            idx += 1

    # Fill remaining
    while idx < num_steps:
        states[idx] = _step_cv(cv, states[idx - 1], sigma_a)
        idx += 1

    return states


def _evasive(
    num_steps: int, dt: float, initial_pos: np.ndarray, speed: float,
    sigma_a: float, rng: np.random.Generator,
    speed_var: float = 0.3,
    turn_rate_max: float = 1.0,
    climb_rate_factor: float = 0.3,
    pitch_range_deg: float = 20.0,
    maneuver_density: float = 0.04,
) -> np.ndarray:
    """Rapid random 3D maneuvers — banking turns, climbs/dives, speed changes.

    All aggressiveness controlled via keyword args (from config.target.evasive).
    """
    cv = ConstantVelocityModel(dt, sigma_a)
    ct = CoordinatedTurnModel(dt, sigma_a)

    # Random initial speed
    init_speed = speed * rng.uniform(1.0 - speed_var, 1.0 + speed_var)
    vel = _random_3d_velocity(init_speed, rng,
                              pitch_range_deg=(-pitch_range_deg, pitch_range_deg))
    states = np.zeros((num_steps, 6))
    states[0] = np.array([*initial_pos, *vel])

    # Random maneuver schedule based on density
    n_maneuvers = max(2, int(num_steps * maneuver_density))
    n_maneuvers = min(n_maneuvers, num_steps - 10)
    maneuver_times = sorted(rng.choice(range(5, num_steps - 5),
                                       size=n_maneuvers, replace=False))
    maneuver_set = set(maneuver_times)
    current_omega = 0.0
    vz_target = vel[2]
    speed_target = init_speed
    maneuver_end = 0

    for t in range(1, num_steps):
        if t in maneuver_set:
            # New maneuver: turn + climb/dive + speed change
            current_omega = rng.choice([-1, 1]) * rng.uniform(0.15, turn_rate_max)
            vz_target = rng.uniform(-climb_rate_factor, climb_rate_factor) * speed
            speed_target = speed * rng.uniform(1.0 - speed_var, 1.0 + speed_var)
            maneuver_end = t + rng.integers(15, 40)

        if t > maneuver_end:
            current_omega *= 0.9
            vz_target *= 0.97
            speed_target += 0.03 * (speed - speed_target)

        if abs(current_omega) > 1e-10:
            states[t] = _step_ct(ct, states[t - 1], current_omega, sigma_a)
        else:
            states[t] = _step_cv(cv, states[t - 1], sigma_a)

        # Smoothly ramp vz toward target
        states[t, 5] = 0.9 * states[t, 5] + 0.1 * vz_target

        # Smoothly scale XY speed toward target (preserve heading)
        vxy = states[t, 3:5]
        cur_xy_speed = np.linalg.norm(vxy)
        if cur_xy_speed > 1e-6:
            blended_speed = 0.92 * cur_xy_speed + 0.08 * speed_target
            states[t, 3:5] = vxy * (blended_speed / cur_xy_speed)

    return states
