"""Demo: N tracker drones + 1 target drone with bearing measurements and 3D viz.

Usage:
    conda activate aa273
    python demo.py                           # Live 3D + replay
    python demo.py --traj evasive            # Choose trajectory type
    python demo.py --steps 300               # Shorter run
    python demo.py --no-gui --steps 2000     # Headless (fast), save plots
    python demo.py --replay results/run.npz  # Replay a saved run (no re-sim)
"""

import argparse
import os
import sys
import time
import numpy as np
import pybullet as p
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.viz.plotting import plot_3d_trajectories, plot_2d_topdown, plot_distances
from src.viz.animation import animate_tracking


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_simulation(cfg, gui, traj_type, episode_length, seed):
    """Run the live simulation, return recorded data."""
    rng = np.random.default_rng(seed)

    num_trackers = cfg["drones"]["num_trackers"]
    r_min = cfg["drones"]["tracker_min_radius"]
    r_max = cfg["drones"]["tracker_max_radius"]

    # Random target spawn altitude
    target_xy = cfg["target"]["initial_xy"]
    target_alt = rng.uniform(cfg["target"]["min_altitude"], cfg["target"]["max_altitude"])
    target_initial_pos = np.array([target_xy[0], target_xy[1], target_alt])

    drone_min_alt = cfg.get("drones_min_altitude", 5.0)

    # Random tracker spawning in hollow sphere around target
    tracker_positions = spawn_in_hollow_sphere(
        center=target_initial_pos,
        r_min=r_min,
        r_max=r_max,
        n=num_trackers,
        rng=rng,
        min_altitude=drone_min_alt,
    )

    print(f"  Target spawn: [{target_initial_pos[0]:.0f}, {target_initial_pos[1]:.0f}, {target_initial_pos[2]:.0f}]m")
    for i in range(num_trackers):
        d = np.linalg.norm(tracker_positions[i] - target_initial_pos)
        print(f"  Tracker {i}: [{tracker_positions[i][0]:.0f}, {tracker_positions[i][1]:.0f}, {tracker_positions[i][2]:.0f}]m  "
              f"(dist={d:.0f}m)")

    env = TrackingAviary(
        num_trackers=num_trackers,
        tracker_positions=tracker_positions,
        target_initial_pos=target_initial_pos,
        target_speed=cfg["target"]["speed"],
        target_trajectory=traj_type,
        target_sigma_a=cfg["target"]["process_noise_accel"],
        evasive_params=cfg["target"].get("evasive"),
        episode_length=episode_length,
        sensor_config=cfg["sensor"],
        pyb_freq=cfg["sim"]["pyb_freq"],
        ctrl_freq=cfg["sim"]["ctrl_freq"],
        gui=gui,
        rng=rng,
    )

    obs, info = env.reset()

    # Camera setup
    if gui:
        cam_dist = r_max * 2.5
        p.resetDebugVisualizerCamera(
            cameraDistance=cam_dist,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=target_initial_pos.tolist(),
            physicsClientId=env.CLIENT,
        )

    drone_trajs = [[] for _ in range(num_trackers)]
    target_traj_actual = []
    all_measurements = []

    # Formation offset: each tracker's relative position to target at spawn
    formation_offset = tracker_positions - target_initial_pos

    ctrl_dt = 1.0 / cfg["sim"]["ctrl_freq"]

    print(f"\nRunning simulation" + (" (live 3D view)" if gui else "") + "...")
    start_time = time.time()

    for step_i in range(episode_length):
        target_pos_true = env.target_traj[step_i, :3]
        target_vel = env.target_traj[step_i, 3:6]

        # Formation waypoints: maintain relative offset to target
        tracker_waypoints = np.zeros((num_trackers, 3))
        tracker_vels = np.tile(target_vel, (num_trackers, 1))  # match target velocity
        for i in range(num_trackers):
            tracker_waypoints[i] = target_pos_true + formation_offset[i]
            tracker_waypoints[i, 2] = max(drone_min_alt, tracker_waypoints[i, 2])

        result = env.step_tracking(
            tracker_targets=tracker_waypoints,
            tracker_target_vels=tracker_vels,
            target_estimate=None,  # use actual physical target pos for gimbal
        )

        if result["done"]:
            break

        for i in range(num_trackers):
            drone_trajs[i].append(result["drone_positions"][i].copy())
        target_traj_actual.append(result["target_true_pos"].copy())
        all_measurements.append(result["measurements"])

        # Camera follows target
        if gui and step_i % 20 == 0:
            cam_target = result["target_true_pos"]
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist,
                cameraYaw=45 + step_i * 0.05,
                cameraPitch=-25,
                cameraTargetPosition=cam_target.tolist(),
                physicsClientId=env.CLIENT,
            )

        if (step_i + 1) % 100 == 0:
            n_det = sum(1 for m in result["measurements"] if m is not None)
            print(f"  Step {step_i + 1}/{episode_length} | "
                  f"Detections: {n_det}/{num_trackers} | "
                  f"Target: [{target_pos_true[0]:.1f}, {target_pos_true[1]:.1f}, {target_pos_true[2]:.1f}]")

    wall_time = time.time() - start_time
    actual_steps = len(target_traj_actual)
    sim_time = actual_steps * ctrl_dt
    print(f"\nDone: {actual_steps} steps, {sim_time:.1f}s sim time, "
          f"{wall_time:.1f}s wall time ({actual_steps / wall_time:.0f} steps/s)")

    env.close()

    drone_trajs = [np.array(t) for t in drone_trajs]
    target_traj_actual = np.array(target_traj_actual)

    return drone_trajs, target_traj_actual, all_measurements, ctrl_dt


def print_summary(drone_trajs, target_traj_actual, all_measurements, num_trackers):
    print("\n--- Summary ---")
    for i in range(num_trackers):
        dist = np.linalg.norm(drone_trajs[i] - target_traj_actual, axis=1)
        print(f"  Tracker {i}: avg dist = {dist.mean():.1f}m, "
              f"min = {dist.min():.1f}m, max = {dist.max():.1f}m")
    n_meas = sum(sum(1 for m in sm if m is not None) for sm in all_measurements)
    total = len(all_measurements) * num_trackers
    print(f"  Detection rate: {n_meas}/{total} ({100 * n_meas / total:.1f}%)")


def save_run(path, drone_trajs, target_traj, measurements, dt, traj_type):
    np.savez_compressed(
        path,
        **{f"drone_{i}": t for i, t in enumerate(drone_trajs)},
        target=target_traj,
        dt=np.array([dt]),
        traj_type=np.array([traj_type]),
        num_trackers=np.array([len(drone_trajs)]),
    )
    print(f"  Run saved to {path}")


def load_run(path):
    data = np.load(path, allow_pickle=True)
    num_trackers = int(data["num_trackers"][0])
    drone_trajs = [data[f"drone_{i}"] for i in range(num_trackers)]
    target_traj = data["target"]
    dt = float(data["dt"][0])
    traj_type = str(data["traj_type"][0])
    return drone_trajs, target_traj, dt, traj_type


def main():
    parser = argparse.ArgumentParser(description="Multi-Drone Tracking Demo")
    parser.add_argument("--no-gui", action="store_true", help="Headless sim, save plots")
    parser.add_argument("--traj", default=None,
                        help="Trajectory: straight, single_turn, multi_segment, evasive")
    parser.add_argument("--steps", type=int, default=500, help="Episode length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-replay", action="store_true", help="Skip matplotlib replay")
    parser.add_argument("--replay", type=str, default=None,
                        help="Replay a saved .npz run (skip simulation)")
    args = parser.parse_args()

    cfg = load_config()
    gui = not args.no_gui
    traj_type = args.traj or cfg["target"]["trajectory"]
    num_trackers = cfg["drones"]["num_trackers"]

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.replay:
        print(f"Loading saved run from {args.replay}...")
        drone_trajs, target_traj_actual, dt, traj_type = load_run(args.replay)
        all_measurements = []
        num_trackers = len(drone_trajs)
    else:
        print("=" * 60)
        print("  Cooperative Multi-Drone Target Tracking")
        print("=" * 60)
        sim_time_s = args.steps / cfg["sim"]["ctrl_freq"]
        print(f"  Trackers: {num_trackers}  |  Trajectory: {traj_type}")
        print(f"  Steps: {args.steps}  ({sim_time_s:.1f}s sim time)")
        print(f"  Arena: {cfg['arena']['size']}m  |  Target speed: {cfg['target']['speed']} m/s")
        if gui:
            wall_est = args.steps * 0.028
            print(f"  GUI render: ~{wall_est:.0f}s wall time")
        print("=" * 60)

        drone_trajs, target_traj_actual, all_measurements, dt = run_simulation(
            cfg, gui, traj_type, args.steps, args.seed,
        )
        print_summary(drone_trajs, target_traj_actual, all_measurements, num_trackers)

        save_path = os.path.join(results_dir, f"run_{traj_type}_s{args.seed}.npz")
        save_run(save_path, drone_trajs, target_traj_actual, all_measurements, dt, traj_type)

    # Static plots
    save_prefix = os.path.join(results_dir, f"demo_{traj_type}")
    plot_box = cfg["arena"].get("plot_box", 2000)
    print(f"\nSaving plots to {results_dir}/  (plot box: ±{plot_box}m)")

    plot_3d_trajectories(
        drone_trajs, target_traj_actual,
        title=f"Multi-Drone Tracking — {traj_type}",
        save_path=f"{save_prefix}_3d.png",
        plot_box=plot_box,
    )
    plot_2d_topdown(
        drone_trajs, target_traj_actual,
        title=f"Top-Down View — {traj_type}",
        save_path=f"{save_prefix}_topdown.png",
        plot_box=plot_box,
    )
    plot_distances(
        drone_trajs, target_traj_actual, dt=dt,
        save_path=f"{save_prefix}_distances.png",
    )

    # Interactive replay
    if not args.no_replay:
        print("\n>> Opening interactive replay <<")
        animate_tracking(
            drone_trajs, target_traj_actual,
            measurements=all_measurements if all_measurements else None,
            dt=dt,
            interval_ms=max(10, int(dt * 1000)),
            title=f"Replay — {traj_type}",
            plot_box=plot_box,
        )


if __name__ == "__main__":
    main()
