"""V6 Baselines — heuristic controllers including angular-repulsion tracking."""

import numpy as np
import torch

from .config import V6Config


def chase_offset_controller(
    drone_pos: np.ndarray,
    filter_estimate: np.ndarray,
    R_desired: float = 60.0,
    Kp: float = 2.0,
    v_max: float = 15.0,
) -> np.ndarray:
    """Chase+offset PD controller (V4 baseline)."""
    dir_to_est = filter_estimate[:3] - drone_pos
    dist = np.linalg.norm(dir_to_est)

    if dist < 1e-3:
        return filter_estimate[3:6].copy()

    unit = dir_to_est / dist
    radial_error = dist - R_desired
    approach_speed = np.clip(Kp * radial_error, -v_max * 0.3, v_max * 0.5)
    vel = unit * approach_speed + filter_estimate[3:6]

    speed = np.linalg.norm(vel)
    if speed > v_max:
        vel *= v_max / speed

    return vel


def repulsion_controller(
    drone_pos: np.ndarray,
    all_positions: np.ndarray,
    drone_idx: int,
    v_max: float = 15.0,
    gain: float = 1.0,
    target_pos: np.ndarray | None = None,
    surround_weight: float = 0.3,
) -> np.ndarray:
    """Spread heuristic: repel from neighbors + bias toward surrounding target."""
    min_dist = np.inf
    nearest_dir = np.zeros(3)

    for j in range(len(all_positions)):
        if j == drone_idx:
            continue
        diff = drone_pos - all_positions[j]
        dist = np.linalg.norm(diff)
        if dist < min_dist:
            min_dist = dist
            nearest_dir = diff / (dist + 1e-8)

    repulsion_vel = nearest_dir

    if target_pos is not None:
        to_target = target_pos - drone_pos
        radial_dist = np.linalg.norm(to_target)

        if radial_dist > 1e-6:
            radial_unit = to_target / radial_dist

            up = np.array([0.0, 0.0, 1.0])
            tangent = np.cross(radial_unit, up)
            tang_norm = np.linalg.norm(tangent)
            if tang_norm > 1e-6:
                tangent /= tang_norm
            else:
                tangent = np.cross(radial_unit, np.array([1.0, 0.0, 0.0]))
                tangent /= np.linalg.norm(tangent) + 1e-8

            if drone_idx % 2 == 1:
                tangent = -tangent

            vel = (1.0 - surround_weight) * repulsion_vel + surround_weight * tangent
        else:
            vel = repulsion_vel
    else:
        vel = repulsion_vel

    speed = np.linalg.norm(vel)
    if speed > 1e-6:
        vel = vel / speed * v_max * gain
    else:
        vel = np.zeros(3)

    return vel


def run_heuristic_twophase(env_spread, env_track, cfg: V6Config,
                           spread_steps: int = 400, track_steps: int = 4600):
    """V6 heuristic: repulsion spread (gain=1.0, surround) + angular-repulsion tracking."""
    N = cfg.num_drones

    drone_trajectories = [[] for _ in range(N)]
    target_trajectory = []
    measurements_list = []
    tr_P_history = []

    # Phase 1: Repulsion spread with full gain + surround
    for step in range(spread_steps):
        drone_positions = env_spread.get_drone_positions()
        target_pos = np.array([0.0, 0.0, 50.0])
        action = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            vel = repulsion_controller(
                drone_positions[i], drone_positions, i,
                v_max=cfg.v_max,
                gain=cfg.spread_gain,
                target_pos=target_pos,
                surround_weight=cfg.spread_surround_weight,
            )
            action[i] = np.clip(vel / cfg.v_max, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env_spread.step(action)

        if "result" in info:
            result = info["result"]
            for i in range(N):
                drone_trajectories[i].append(result["drone_positions"][i].copy())
            target_trajectory.append(result["target_true_pos"].copy())
            measurements_list.append(result["measurements"])

        if terminated or truncated:
            break

    final_positions = env_spread.get_drone_positions()

    # Phase 2: Angular-repulsion tracking (zero RL residual)
    env_track.set_initial_positions(final_positions)
    obs, _ = env_track.reset()

    for step in range(track_steps):
        action = np.zeros((N, 3), dtype=np.float32)
        obs, reward, terminated, truncated, info = env_track.step(action)

        if "result" in info:
            result = info["result"]
            for i in range(N):
                drone_trajectories[i].append(result["drone_positions"][i].copy())
            target_trajectory.append(result["target_true_pos"].copy())
            measurements_list.append(result["measurements"])

        if "tr_P_pos" in info:
            tr_P_history.append(info["tr_P_pos"])

        if terminated or truncated:
            break

    return {
        "drone_trajectories": [np.array(t) for t in drone_trajectories],
        "target_trajectory": np.array(target_trajectory),
        "measurements": measurements_list,
        "tr_P_history": tr_P_history,
    }


def run_baseline_chase_only(env_track, cfg: V6Config, total_steps: int = 5000):
    """Baseline: cluster spawn + plain chase+offset (no spread, no angular repulsion).

    Uses a V4-style chase controller by setting angular_weight=0.
    """
    N = cfg.num_drones
    drone_trajectories = [[] for _ in range(N)]
    target_trajectory = []
    measurements_list = []
    tr_P_history = []

    # Temporarily disable angular repulsion
    saved_aw = cfg.angular_weight
    cfg.angular_weight = 0.0

    obs, _ = env_track.reset()

    for step in range(total_steps):
        action = np.zeros((N, 3), dtype=np.float32)
        obs, reward, terminated, truncated, info = env_track.step(action)

        if "result" in info:
            result = info["result"]
            for i in range(N):
                drone_trajectories[i].append(result["drone_positions"][i].copy())
            target_trajectory.append(result["target_true_pos"].copy())
            measurements_list.append(result["measurements"])

        if "tr_P_pos" in info:
            tr_P_history.append(info["tr_P_pos"])

        if terminated or truncated:
            break

    # Restore
    cfg.angular_weight = saved_aw

    return {
        "drone_trajectories": [np.array(t) for t in drone_trajectories],
        "target_trajectory": np.array(target_trajectory),
        "measurements": measurements_list,
        "tr_P_history": tr_P_history,
    }


def run_baseline_no_spread(env_spread, env_track, cfg: V6Config,
                           spread_steps: int = 400, track_steps: int = 4600):
    """V4-style heuristic: repulsion spread + plain chase+offset (no angular repulsion)."""
    N = cfg.num_drones

    drone_trajectories = [[] for _ in range(N)]
    target_trajectory = []
    measurements_list = []
    tr_P_history = []

    # Phase 1: Repulsion spread (same as V6 heuristic)
    for step in range(spread_steps):
        drone_positions = env_spread.get_drone_positions()
        target_pos = np.array([0.0, 0.0, 50.0])
        action = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            vel = repulsion_controller(
                drone_positions[i], drone_positions, i,
                v_max=cfg.v_max,
                gain=cfg.spread_gain,
                target_pos=target_pos,
                surround_weight=cfg.spread_surround_weight,
            )
            action[i] = np.clip(vel / cfg.v_max, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env_spread.step(action)

        if "result" in info:
            result = info["result"]
            for i in range(N):
                drone_trajectories[i].append(result["drone_positions"][i].copy())
            target_trajectory.append(result["target_true_pos"].copy())
            measurements_list.append(result["measurements"])

        if terminated or truncated:
            break

    final_positions = env_spread.get_drone_positions()

    # Phase 2: Plain chase+offset (angular_weight=0)
    saved_aw = cfg.angular_weight
    cfg.angular_weight = 0.0
    env_track.set_initial_positions(final_positions)
    obs, _ = env_track.reset()

    for step in range(track_steps):
        action = np.zeros((N, 3), dtype=np.float32)
        obs, reward, terminated, truncated, info = env_track.step(action)

        if "result" in info:
            result = info["result"]
            for i in range(N):
                drone_trajectories[i].append(result["drone_positions"][i].copy())
            target_trajectory.append(result["target_true_pos"].copy())
            measurements_list.append(result["measurements"])

        if "tr_P_pos" in info:
            tr_P_history.append(info["tr_P_pos"])

        if terminated or truncated:
            break

    cfg.angular_weight = saved_aw

    return {
        "drone_trajectories": [np.array(t) for t in drone_trajectories],
        "target_trajectory": np.array(target_trajectory),
        "measurements": measurements_list,
        "tr_P_history": tr_P_history,
    }


def run_rl_twophase(
    track_actor, env_spread, env_track, cfg: V6Config,
    track_obs_normalizer=None, device=None,
    spread_steps: int = 400, track_steps: int = 4600,
):
    """V6 RL: heuristic spread + RL residual on angular-repulsion tracking."""
    if device is None:
        device = torch.device("cpu")

    N = cfg.num_drones
    drone_trajectories = [[] for _ in range(N)]
    target_trajectory = []
    measurements_list = []
    tr_P_history = []

    # Phase 1: Heuristic repulsion spread
    for step in range(spread_steps):
        drone_positions = env_spread.get_drone_positions()
        target_pos = np.array([0.0, 0.0, 50.0])
        action = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            vel = repulsion_controller(
                drone_positions[i], drone_positions, i,
                v_max=cfg.v_max,
                gain=cfg.spread_gain,
                target_pos=target_pos,
                surround_weight=cfg.spread_surround_weight,
            )
            action[i] = np.clip(vel / cfg.v_max, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env_spread.step(action)

        if "result" in info:
            result = info["result"]
            for i in range(N):
                drone_trajectories[i].append(result["drone_positions"][i].copy())
            target_trajectory.append(result["target_true_pos"].copy())
            measurements_list.append(result["measurements"])

        if terminated or truncated:
            break

    final_positions = env_spread.get_drone_positions()

    # Phase 2: RL residual on angular-repulsion tracking
    env_track.set_initial_positions(final_positions)
    obs, _ = env_track.reset()

    for step in range(track_steps):
        if track_obs_normalizer is not None:
            obs_norm = np.clip(track_obs_normalizer.normalize(obs), -10.0, 10.0)
        else:
            obs_norm = obs

        actor_obs = torch.tensor(
            obs_norm[:, :cfg.track_actor_obs_dim].reshape(N, -1),
            dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            action = track_actor.deterministic_action(actor_obs).cpu().numpy()

        obs, reward, terminated, truncated, info = env_track.step(action)

        if "result" in info:
            result = info["result"]
            for i in range(N):
                drone_trajectories[i].append(result["drone_positions"][i].copy())
            target_trajectory.append(result["target_true_pos"].copy())
            measurements_list.append(result["measurements"])

        if "tr_P_pos" in info:
            tr_P_history.append(info["tr_P_pos"])

        if terminated or truncated:
            break

    return {
        "drone_trajectories": [np.array(t) for t in drone_trajectories],
        "target_trajectory": np.array(target_trajectory),
        "measurements": measurements_list,
        "tr_P_history": tr_P_history,
    }
