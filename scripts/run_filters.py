"""Single-run filter comparison: EKF, UKF, PF on bearing-only tracking.

Runs the simulation (or replays saved data) and applies all three filters
in parallel on the same measurement stream for direct comparison.

Usage:
    python scripts/run_filters.py                        # default config
    python scripts/run_filters.py --traj evasive --steps 500
    python scripts/run_filters.py --no-gui --save
    python scripts/run_filters.py --replay results/run.npz
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.filters import EKF, UKF, PF
from src.viz.filter_plots import (
    plot_state_estimates, plot_rmse, plot_nees,
    plot_covariance_trace, plot_3d_estimate_trajectory,
)
from src.viz.animation import animate_filter_tracking


def load_config(path: str = "config/default.yaml") -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_filters(cfg: dict, dt: float) -> list:
    """Instantiate EKF, UKF, PF from config."""
    sigma_a = cfg["target"]["process_noise_accel"]
    sigma_bearing = np.deg2rad(cfg["sensor"]["sigma_bearing_deg"])
    range_ref = cfg["sensor"]["range_ref"]

    finit = cfg["filters"].get("init", {})
    P0_pos = finit.get("P0_pos", 10000.0)
    P0_vel = finit.get("P0_vel", 100.0)

    ekf = EKF(dt, sigma_a, sigma_bearing, range_ref, P0_pos, P0_vel)
    ukf_cfg = cfg["filters"].get("ukf", {})
    ukf = UKF(
        dt, sigma_a, sigma_bearing, range_ref,
        alpha=ukf_cfg.get("alpha", 1e-3),
        beta=ukf_cfg.get("beta", 2.0),
        kappa=ukf_cfg.get("kappa", 0.0),
        P0_pos=P0_pos, P0_vel=P0_vel,
    )
    pf_cfg = cfg["filters"].get("pf", {})
    pf = PF(
        dt, sigma_a, sigma_bearing, range_ref,
        num_particles=pf_cfg.get("num_particles", 2000),
        resample_threshold=pf_cfg.get("resample_threshold", 0.5),
        range_min=finit.get("pf_range_min", 10.0),
        range_max=finit.get("pf_range_max", 500.0),
        P0_pos=P0_pos, P0_vel=P0_vel,
        process_noise_factor=pf_cfg.get("process_noise_factor", 5.0),
        jitter_pos=pf_cfg.get("jitter_pos", 1.0),
        jitter_vel=pf_cfg.get("jitter_vel", 0.5),
    )
    return [ekf, ukf, pf]


def run_sim_with_filters(cfg, filters, gui, traj_type, episode_length, seed,
                         gimbal_filter_idx: int = 0):
    """Run simulation with filters in the loop (online).

    The gimbal on each drone points at the estimate from filters[gimbal_filter_idx].
    This creates a realistic feedback loop: bad estimate → bad pointing → missed
    detections → worse estimate.

    Args:
        gimbal_filter_idx: which filter's estimate drives the gimbal (default: EKF=0).
            Use -1 for perfect gimbal (true target position, no feedback).
    """
    rng = np.random.default_rng(seed)

    num_trackers = cfg["drones"]["num_trackers"]
    r_min = cfg["drones"]["tracker_min_radius"]
    r_max = cfg["drones"]["tracker_max_radius"]

    target_xy = cfg["target"]["initial_xy"]
    target_alt = rng.uniform(cfg["target"]["min_altitude"], cfg["target"]["max_altitude"])
    target_initial_pos = np.array([target_xy[0], target_xy[1], target_alt])
    drone_min_alt = cfg.get("drones_min_altitude", 5.0)

    tracker_positions = spawn_in_hollow_sphere(
        center=target_initial_pos,
        r_min=r_min, r_max=r_max,
        n=num_trackers, rng=rng,
        min_altitude=drone_min_alt,
    )

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

    env.reset()
    formation_offset = tracker_positions - target_initial_pos

    n_filters = len(filters)
    T = episode_length

    record = {
        "drone_positions": [],
        "target_true_state": [],
        "measurements": [],
    }
    estimates = np.zeros((n_filters, T, 6))
    covariances = np.zeros((n_filters, T, 6, 6))
    nees = np.zeros((n_filters, T))
    init_steps = np.zeros(n_filters, dtype=int)

    import pybullet as p
    if gui:
        cam_dist = r_max * 2.5
        p.resetDebugVisualizerCamera(
            cameraDistance=cam_dist, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=target_initial_pos.tolist(),
            physicsClientId=env.CLIENT,
        )

    actual_steps = 0
    for step_i in range(episode_length):
        target_pos_true = env.target_traj[step_i, :3]
        target_vel = env.target_traj[step_i, 3:6]

        tracker_waypoints = np.zeros((num_trackers, 3))
        tracker_vels = np.tile(target_vel, (num_trackers, 1))
        for i in range(num_trackers):
            tracker_waypoints[i] = target_pos_true + formation_offset[i]
            tracker_waypoints[i, 2] = max(drone_min_alt, tracker_waypoints[i, 2])

        # Gimbal: feed filter estimate back to sensor pointing
        gimbal_est = None  # None = perfect gimbal (use true position)
        if gimbal_filter_idx >= 0 and gimbal_filter_idx < n_filters:
            filt = filters[gimbal_filter_idx]
            if filt.initialized:
                gimbal_est = filt.get_estimate()[:3]

        result = env.step_tracking(
            tracker_targets=tracker_waypoints,
            tracker_target_vels=tracker_vels,
            target_estimate=gimbal_est,
        )

        if result["done"]:
            break

        meas = result["measurements"]
        drone_pos = result["drone_positions"]
        true_state = result["target_true_state"]

        record["drone_positions"].append(drone_pos.copy())
        record["target_true_state"].append(true_state.copy())
        record["measurements"].append(meas)

        # Run all filters on this step's measurements
        for fi, filt in enumerate(filters):
            filt.predict()
            filt.update(meas, drone_pos)

            x_hat = filt.get_estimate()
            P_mat = filt.get_covariance()
            estimates[fi, step_i] = x_hat
            covariances[fi, step_i] = P_mat

            if not filt.initialized:
                continue
            if init_steps[fi] == 0 and filt.initialized:
                init_steps[fi] = step_i

            err = true_state[:6] - x_hat
            try:
                nees[fi, step_i] = err @ np.linalg.inv(P_mat) @ err
            except np.linalg.LinAlgError:
                nees[fi, step_i] = np.nan

        actual_steps = step_i + 1

        if gui and step_i % 20 == 0:
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist, cameraYaw=45 + step_i * 0.05,
                cameraPitch=-25,
                cameraTargetPosition=result["target_true_pos"].tolist(),
                physicsClientId=env.CLIENT,
            )

    env.close()

    # Trim to actual steps
    record["drone_positions"] = np.array(record["drone_positions"])
    record["target_true_state"] = np.array(record["target_true_state"])
    estimates = estimates[:, :actual_steps]
    covariances = covariances[:, :actual_steps]
    nees = nees[:, :actual_steps]

    return record, estimates, covariances, nees, init_steps


def compute_metrics(estimates, true_states, nees, init_steps, filter_names):
    """Compute summary metrics per filter."""
    T = true_states.shape[0]
    state_dim = 6

    print("\n" + "=" * 70)
    print(f"  {'Filter':<8} | {'Pos RMSE':>10} | {'Vel RMSE':>10} | "
          f"{'ANEES':>8} | {'Conv Step':>10} | {'Final Err':>10}")
    print("-" * 70)

    for fi, name in enumerate(filter_names):
        t0 = max(init_steps[fi], 1)
        err = true_states[t0:, :6] - estimates[fi, t0:]
        pos_err = np.linalg.norm(err[:, :3], axis=1)
        vel_err = np.linalg.norm(err[:, 3:6], axis=1)

        pos_rmse = np.sqrt(np.mean(pos_err**2))
        vel_rmse = np.sqrt(np.mean(vel_err**2))

        valid_nees = nees[fi, t0:]
        valid_nees = valid_nees[~np.isnan(valid_nees)]
        anees = np.mean(valid_nees) if len(valid_nees) > 0 else np.nan

        # Convergence: first time pos error < 20m (sustained for 5 steps)
        conv_step = T
        for t in range(t0, T - 5):
            if np.all(pos_err[t - t0:t - t0 + 5] < 20.0):
                conv_step = t
                break

        final_err = pos_err[-1] if len(pos_err) > 0 else np.nan

        print(f"  {name:<8} | {pos_rmse:>8.2f} m | {vel_rmse:>8.2f} m/s | "
              f"{anees:>8.2f} | {conv_step:>10d} | {final_err:>8.2f} m")

    print("=" * 70)
    print(f"  NEES target: {state_dim:.0f} (state_dim)  |  "
          f"95% bounds: [{state_dim - 2*np.sqrt(2*state_dim):.1f}, "
          f"{state_dim + 2*np.sqrt(2*state_dim):.1f}]")


def main():
    parser = argparse.ArgumentParser(description="Filter Comparison Experiment")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--traj", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="Save plots to results/")
    parser.add_argument("--no-replay", action="store_true", help="Skip interactive replay")
    parser.add_argument("--perfect-gimbal", action="store_true",
                        help="Use true target position for gimbal (no feedback)")
    args = parser.parse_args()

    cfg = load_config()
    gui = not args.no_gui
    traj_type = args.traj or cfg["target"]["trajectory"]
    dt = 1.0 / cfg["sim"]["ctrl_freq"]

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create filters
    filters = create_filters(cfg, dt)
    filter_names = [f.name for f in filters]

    # Run simulation with filters in the loop (online)
    # gimbal_filter_idx=0 → EKF drives gimbal; -1 → perfect gimbal
    gimbal_idx = -1 if args.perfect_gimbal else 0
    gimbal_label = "perfect (true pos)" if gimbal_idx == -1 else f"{filter_names[gimbal_idx]}"
    print(f"Running simulation: traj={traj_type}, steps={args.steps}, "
          f"seed={args.seed}, gimbal={gimbal_label}")

    t0 = time.time()
    record, estimates, covariances, nees, init_steps = run_sim_with_filters(
        cfg, filters, gui, traj_type, args.steps, args.seed,
        gimbal_filter_idx=gimbal_idx,
    )
    print(f"Done in {time.time() - t0:.2f}s")

    true_states = record["target_true_state"]
    T = true_states.shape[0]
    time_axis = np.arange(T) * dt

    # Metrics
    compute_metrics(estimates, true_states, nees, init_steps, filter_names)

    # Plots
    save_prefix = os.path.join(results_dir, f"filters_{traj_type}") if args.save else None

    plot_state_estimates(
        time_axis, true_states, estimates, covariances, filter_names,
        save_path=f"{save_prefix}_states.png" if save_prefix else None,
    )
    plot_rmse(
        time_axis, true_states, estimates, filter_names,
        save_path=f"{save_prefix}_rmse.png" if save_prefix else None,
    )
    plot_nees(
        time_axis, nees, filter_names, state_dim=6,
        save_path=f"{save_prefix}_nees.png" if save_prefix else None,
    )
    plot_covariance_trace(
        time_axis, covariances, filter_names,
        save_path=f"{save_prefix}_covtrace.png" if save_prefix else None,
    )
    plot_3d_estimate_trajectory(
        true_states, estimates, filter_names,
        save_path=f"{save_prefix}_3d.png" if save_prefix else None,
    )

    # Interactive replay
    if not args.no_replay:
        plot_box = cfg["arena"].get("plot_box", 2000)
        print("\n>> Opening interactive filter replay <<")
        animate_filter_tracking(
            drone_positions=record["drone_positions"],
            target_true_states=true_states,
            estimates=estimates,
            filter_names=filter_names,
            measurements=record["measurements"],
            dt=dt,
            interval_ms=max(10, int(dt * 1000)),
            title=f"Filter Tracking — {traj_type}",
            plot_box=plot_box,
        )

    # Save filter data
    if args.save:
        np.savez_compressed(
            os.path.join(results_dir, f"filter_data_{traj_type}_s{args.seed}.npz"),
            estimates=estimates,
            covariances=covariances,
            nees=nees,
            true_states=true_states,
            drone_positions=record["drone_positions"],
            measurements=np.array(record["measurements"], dtype=object),
            filter_names=filter_names,
            dt=dt,
            traj_type=traj_type,
        )
        print(f"  Data saved to {results_dir}/")


if __name__ == "__main__":
    main()
