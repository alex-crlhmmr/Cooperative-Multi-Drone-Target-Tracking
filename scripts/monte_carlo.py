"""Monte Carlo filter comparison sweep.

Runs N trials for each (filter, trajectory) combination and computes
statistical metrics: RMSE, ANEES, convergence time, track loss rate.

Uses multiprocessing for parallel execution (~900 steps/s per core headless).

Usage:
    python scripts/monte_carlo.py                    # 100 runs x 12 combos
    python scripts/monte_carlo.py --runs 10          # quick test
    python scripts/monte_carlo.py --filter ekf       # single filter
    python scripts/monte_carlo.py --traj evasive     # single trajectory
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.filters import EKF, UKF, PF
from src.viz.filter_plots import (
    plot_monte_carlo_comparison, plot_track_loss_bar,
)


def load_config(path: str = "config/default.yaml") -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_filter(filter_name: str, cfg: dict, dt: float):
    """Create a single filter by name."""
    sigma_a = cfg["target"]["process_noise_accel"]
    sigma_bearing = np.deg2rad(cfg["sensor"]["sigma_bearing_deg"])
    range_ref = cfg["sensor"]["range_ref"]
    finit = cfg["filters"].get("init", {})
    P0_pos = finit.get("P0_pos", 10000.0)
    P0_vel = finit.get("P0_vel", 100.0)

    if filter_name == "EKF":
        return EKF(dt, sigma_a, sigma_bearing, range_ref, P0_pos, P0_vel)
    elif filter_name == "UKF":
        ukf_cfg = cfg["filters"].get("ukf", {})
        return UKF(dt, sigma_a, sigma_bearing, range_ref,
                    alpha=ukf_cfg.get("alpha", 1e-3),
                    beta=ukf_cfg.get("beta", 2.0),
                    kappa=ukf_cfg.get("kappa", 0.0),
                    P0_pos=P0_pos, P0_vel=P0_vel)
    elif filter_name == "PF":
        pf_cfg = cfg["filters"].get("pf", {})
        return PF(dt, sigma_a, sigma_bearing, range_ref,
                   num_particles=pf_cfg.get("num_particles", 2000),
                   resample_threshold=pf_cfg.get("resample_threshold", 0.5),
                   range_min=finit.get("pf_range_min", 10.0),
                   range_max=finit.get("pf_range_max", 500.0),
                   P0_pos=P0_pos, P0_vel=P0_vel,
                   process_noise_factor=pf_cfg.get("process_noise_factor", 5.0),
                   jitter_pos=pf_cfg.get("jitter_pos", 1.0),
                   jitter_vel=pf_cfg.get("jitter_vel", 0.5))
    else:
        raise ValueError(f"Unknown filter: {filter_name}")


def run_single_trial(args):
    """Run a single MC trial. Returns metrics dict.

    Designed for multiprocessing — imports inside function to avoid pickling issues.
    """
    filter_name, traj_type, seed, cfg_dict = args

    from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere

    rng = np.random.default_rng(seed)
    dt = 1.0 / cfg_dict["sim"]["ctrl_freq"]
    episode_length = cfg_dict["monte_carlo"]["episode_length"]

    num_trackers = cfg_dict["drones"]["num_trackers"]
    r_min = cfg_dict["drones"]["tracker_min_radius"]
    r_max = cfg_dict["drones"]["tracker_max_radius"]

    target_xy = cfg_dict["target"]["initial_xy"]
    target_alt = rng.uniform(cfg_dict["target"]["min_altitude"],
                             cfg_dict["target"]["max_altitude"])
    target_initial_pos = np.array([target_xy[0], target_xy[1], target_alt])
    drone_min_alt = cfg_dict.get("drones_min_altitude", 5.0)

    tracker_positions = spawn_in_hollow_sphere(
        center=target_initial_pos,
        r_min=r_min, r_max=r_max,
        n=num_trackers, rng=rng,
        min_altitude=drone_min_alt,
    )

    try:
        env = TrackingAviary(
            num_trackers=num_trackers,
            tracker_positions=tracker_positions,
            target_initial_pos=target_initial_pos,
            target_speed=cfg_dict["target"]["speed"],
            target_trajectory=traj_type,
            target_sigma_a=cfg_dict["target"]["process_noise_accel"],
            evasive_params=cfg_dict["target"].get("evasive"),
            episode_length=episode_length,
            sensor_config=cfg_dict["sensor"],
            pyb_freq=cfg_dict["sim"]["pyb_freq"],
            ctrl_freq=cfg_dict["sim"]["ctrl_freq"],
            gui=False,
            rng=rng,
        )
        obs, info = env.reset()
    except Exception as e:
        return {"error": str(e), "filter": filter_name, "traj": traj_type, "seed": seed}

    formation_offset = tracker_positions - target_initial_pos
    filt = create_filter(filter_name, cfg_dict, dt)

    blackout_threshold = cfg_dict["monte_carlo"].get("blackout_steps", 20)

    pos_errors = []
    vel_errors = []
    nees_vals = []
    consecutive_blind = 0   # steps with all drones returning None
    max_blind_streak = 0
    sensor_blackout = False  # flagged if blind streak >= threshold

    for step_i in range(episode_length):
        target_pos_true = env.target_traj[step_i, :3]
        target_vel = env.target_traj[step_i, 3:6]

        tracker_waypoints = np.zeros((num_trackers, 3))
        tracker_vels = np.tile(target_vel, (num_trackers, 1))
        for i in range(num_trackers):
            tracker_waypoints[i] = target_pos_true + formation_offset[i]
            tracker_waypoints[i, 2] = max(drone_min_alt, tracker_waypoints[i, 2])

        result = env.step_tracking(
            tracker_targets=tracker_waypoints,
            tracker_target_vels=tracker_vels,
            target_estimate=None,
        )
        if result["done"]:
            break

        meas = result["measurements"]
        drone_pos = result["drone_positions"]
        true_state = result["target_true_state"]

        # Track sensor blackout (all drones None)
        if all(m is None for m in meas):
            consecutive_blind += 1
            max_blind_streak = max(max_blind_streak, consecutive_blind)
            if consecutive_blind >= blackout_threshold:
                sensor_blackout = True
        else:
            consecutive_blind = 0

        filt.predict()
        filt.update(meas, drone_pos)

        if filt.initialized:
            x_hat = filt.get_estimate()
            P = filt.get_covariance()
            err = true_state[:6] - x_hat
            pos_errors.append(np.linalg.norm(err[:3]))
            vel_errors.append(np.linalg.norm(err[3:6]))

            try:
                P_inv = np.linalg.inv(P)
                nees_vals.append(err @ P_inv @ err)
            except np.linalg.LinAlgError:
                nees_vals.append(np.nan)

    env.close()

    pos_errors = np.array(pos_errors) if pos_errors else np.array([np.nan])
    vel_errors = np.array(vel_errors) if vel_errors else np.array([np.nan])
    nees_vals = np.array(nees_vals) if nees_vals else np.array([np.nan])

    # Metrics
    pos_rmse = np.sqrt(np.nanmean(pos_errors**2))
    vel_rmse = np.sqrt(np.nanmean(vel_errors**2))
    valid_nees = nees_vals[~np.isnan(nees_vals)]
    anees = np.nanmean(valid_nees) if len(valid_nees) > 0 else np.nan

    # Convergence time
    conv_time = len(pos_errors)
    for t in range(len(pos_errors) - 5):
        if np.all(pos_errors[t:t + 5] < 20.0):
            conv_time = t
            break

    # Track loss: final error > 100m OR sensor blackout (all drones blind for N steps)
    error_loss = 1 if (len(pos_errors) > 0 and pos_errors[-1] > 100.0) else 0
    blackout_loss = 1 if sensor_blackout else 0
    track_loss = 1 if (error_loss or blackout_loss) else 0

    return {
        "filter": filter_name,
        "traj": traj_type,
        "seed": seed,
        "pos_rmse": pos_rmse,
        "vel_rmse": vel_rmse,
        "anees": anees,
        "convergence_time": conv_time,
        "track_loss": track_loss,
        "error_loss": error_loss,
        "blackout_loss": blackout_loss,
        "max_blind_streak": max_blind_streak,
    }


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Filter Comparison")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--filter", type=str, default=None,
                        help="Single filter: ekf, ukf, pf")
    parser.add_argument("--traj", type=str, default=None,
                        help="Single trajectory type")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    num_runs = args.runs or cfg["monte_carlo"]["num_runs"]

    filter_names = [args.filter.upper()] if args.filter else ["EKF", "UKF", "PF"]
    traj_types = [args.traj] if args.traj else ["straight", "single_turn", "multi_segment", "evasive"]

    n_workers = args.workers or max(1, cpu_count() - 1)
    total_trials = len(filter_names) * len(traj_types) * num_runs

    print("=" * 60)
    print("  Monte Carlo Filter Comparison")
    print("=" * 60)
    print(f"  Filters: {filter_names}")
    print(f"  Trajectories: {traj_types}")
    print(f"  Runs per combo: {num_runs}")
    print(f"  Total trials: {total_trials}")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    # Build task list
    tasks = []
    for fname in filter_names:
        for traj in traj_types:
            for run_i in range(num_runs):
                seed = 1000 * hash((fname, traj)) % 100000 + run_i
                tasks.append((fname, traj, seed, cfg))

    # Run
    t0 = time.time()
    if n_workers <= 1:
        all_results = [run_single_trial(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            all_results = pool.map(run_single_trial, tasks)
    elapsed = time.time() - t0

    print(f"\nCompleted {len(all_results)} trials in {elapsed:.1f}s "
          f"({elapsed / max(len(all_results), 1):.2f}s/trial)")

    # Aggregate
    results = {}
    errors = []
    for r in all_results:
        if "error" in r:
            errors.append(r)
            continue
        key = (r["filter"], r["traj"])
        if key not in results:
            results[key] = {
                "pos_rmse": [], "vel_rmse": [], "anees": [],
                "convergence_time": [], "track_loss": [],
                "error_loss": [], "blackout_loss": [], "max_blind_streak": [],
            }
        results[key]["pos_rmse"].append(r["pos_rmse"])
        results[key]["vel_rmse"].append(r["vel_rmse"])
        results[key]["anees"].append(r["anees"])
        results[key]["convergence_time"].append(r["convergence_time"])
        results[key]["track_loss"].append(r["track_loss"])
        results[key]["error_loss"].append(r["error_loss"])
        results[key]["blackout_loss"].append(r["blackout_loss"])
        results[key]["max_blind_streak"].append(r["max_blind_streak"])

    if errors:
        print(f"\n  {len(errors)} trials failed:")
        for e in errors[:5]:
            print(f"    {e['filter']}/{e['traj']} seed={e['seed']}: {e['error']}")

    # Print summary table
    blackout_n = cfg["monte_carlo"].get("blackout_steps", 20)
    print("\n" + "=" * 105)
    print(f"  {'Filter':<6} | {'Trajectory':<14} | {'Pos RMSE':>10} | {'Vel RMSE':>10} | "
          f"{'ANEES':>8} | {'Conv':>6} | {'Loss%':>6} | {'Err%':>5} | {'Blind%':>6} | {'MaxBlind':>8}")
    print("-" * 105)

    for traj in traj_types:
        for fname in filter_names:
            key = (fname, traj)
            if key not in results:
                continue
            r = results[key]
            pos = np.nanmean(r["pos_rmse"])
            vel = np.nanmean(r["vel_rmse"])
            anees = np.nanmean(r["anees"])
            conv = np.nanmean(r["convergence_time"])
            loss = np.mean(r["track_loss"]) * 100
            err_loss = np.mean(r["error_loss"]) * 100
            blind_loss = np.mean(r["blackout_loss"]) * 100
            max_blind = np.mean(r["max_blind_streak"])
            print(f"  {fname:<6} | {traj:<14} | {pos:>8.2f} m | {vel:>8.2f} m/s | "
                  f"{anees:>8.2f} | {conv:>6.0f} | {loss:>5.1f}% | {err_loss:>4.1f}% | "
                  f"{blind_loss:>5.1f}% | {max_blind:>7.1f}")
        print("-" * 105)
    print("=" * 105)
    print(f"  Loss = error_loss OR blackout_loss  |  "
          f"Blackout = all drones None for {blackout_n}+ consecutive steps")

    # Convert lists to arrays for plotting
    for key in results:
        for metric in results[key]:
            results[key][metric] = np.array(results[key][metric])

    # Plots
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_prefix = os.path.join(results_dir, "mc") if args.save else None

    plot_monte_carlo_comparison(
        results, filter_names, traj_types,
        save_path=f"{save_prefix}_comparison.png" if save_prefix else None,
    )
    plot_track_loss_bar(
        results, filter_names, traj_types,
        save_path=f"{save_prefix}_track_loss.png" if save_prefix else None,
    )

    # Save raw results
    if args.save:
        save_data = {}
        for (fname, traj), metrics in results.items():
            for metric_name, values in metrics.items():
                save_data[f"{fname}_{traj}_{metric_name}"] = values
        np.savez_compressed(
            os.path.join(results_dir, "mc_results.npz"),
            **save_data,
            filter_names=filter_names,
            traj_types=traj_types,
            num_runs=num_runs,
        )
        print(f"\n  Results saved to {results_dir}/mc_results.npz")


if __name__ == "__main__":
    main()
