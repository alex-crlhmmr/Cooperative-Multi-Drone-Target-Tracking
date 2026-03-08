"""V7 Baselines — heuristic spread controllers for comparison."""

import numpy as np
import torch

from .config import V7Config


def repulsion_controller(
    drone_pos: np.ndarray,
    all_positions: np.ndarray,
    drone_idx: int,
    v_max: float = 15.0,
    gain: float = 1.0,
    target_pos: np.ndarray | None = None,
    surround_weight: float = 0.3,
) -> np.ndarray:
    """Spread heuristic: repel from neighbors + tangential surround bias."""
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


def run_heuristic_spread_chase(env_spread, env_track, cfg: V7Config,
                                spread_steps: int = 400, track_steps: int = 4600):
    """Heuristic baseline: repulsion spread + chase+offset tracking."""
    N = cfg.num_drones

    drone_trajectories = [[] for _ in range(N)]
    target_trajectory = []
    measurements_list = []
    tr_P_history = []

    # Phase 1: Repulsion spread
    for step in range(spread_steps):
        drone_positions = env_spread.get_drone_positions()
        target_pos = np.array([0.0, 0.0, 50.0])
        action = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            vel = repulsion_controller(
                drone_positions[i], drone_positions, i,
                v_max=cfg.v_max,
                gain=cfg.spread_repulsion_gain,
                target_pos=target_pos,
                surround_weight=cfg.spread_surround_weight,
            )
            # Map to action space: residual env expects [-1, 1]
            # but for heuristic we bypass residual, so scale to full [-1, 1]
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

    # Phase 2: Chase+offset tracking
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


def run_baseline_no_spread(env_track, cfg: V7Config, total_steps: int = 5000):
    """Baseline: cluster spawn + chase+offset, no spread."""
    N = cfg.num_drones
    drone_trajectories = [[] for _ in range(N)]
    target_trajectory = []
    measurements_list = []
    tr_P_history = []

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

    return {
        "drone_trajectories": [np.array(t) for t in drone_trajectories],
        "target_trajectory": np.array(target_trajectory),
        "measurements": measurements_list,
        "tr_P_history": tr_P_history,
    }


def run_rl_spread_chase(
    spread_actor, env_spread, env_track, cfg: V7Config,
    spread_obs_normalizer=None, device=None,
    spread_steps: int = 400, track_steps: int = 4600,
):
    """V7 RL: learned spread + chase+offset tracking."""
    if device is None:
        device = torch.device("cpu")

    N = cfg.num_drones
    drone_trajectories = [[] for _ in range(N)]
    target_trajectory = []
    measurements_list = []
    tr_P_history = []

    # Phase 1: RL spread
    obs, _ = env_spread.reset()
    for step in range(spread_steps):
        if spread_obs_normalizer is not None:
            obs_norm = np.clip(spread_obs_normalizer.normalize(obs), -10.0, 10.0)
        else:
            obs_norm = obs

        actor_obs = torch.tensor(
            obs_norm[:, :cfg.spread_obs_dim].reshape(N, -1),
            dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            action = spread_actor.deterministic_action(actor_obs).cpu().numpy()

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

    # Phase 2: Chase+offset tracking
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
