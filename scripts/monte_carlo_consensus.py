"""Monte Carlo sweep for distributed consensus EKF.

Runs N trials for each (topology, dropout, trajectory) combination,
paired with a centralized EKF baseline using the same seed.

Usage:
    python scripts/monte_carlo_consensus.py --runs 50
    python scripts/monte_carlo_consensus.py --runs 10 --topology full
    python scripts/monte_carlo_consensus.py --runs 10 --traj multi_segment --save
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.filters import EKF, ConsensusEKF, generate_adjacency
from src.viz.consensus_plots import (
    plot_mc_dropout_comparison,
    plot_mc_topology_boxes,
)


def load_config(path: str = "config/default.yaml") -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_trial(args):
    """Run one paired trial: centralized EKF + ConsensusEKF with same seed.

    Returns dict with metrics for both filters.
    """
    topology, dropout, traj_type, seed, cfg_dict = args

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
        env.reset()
    except Exception as e:
        return {"error": str(e), "topology": topology, "dropout": dropout,
                "traj": traj_type, "seed": seed}

    formation_offset = tracker_positions - target_initial_pos

    # Create filters
    sigma_a = cfg_dict["target"]["process_noise_accel"]
    sigma_bearing = np.deg2rad(cfg_dict["sensor"]["sigma_bearing_deg"])
    range_ref = cfg_dict["sensor"]["range_ref"]
    finit = cfg_dict["filters"].get("init", {})
    P0_pos = finit.get("P0_pos", 10000.0)
    P0_vel = finit.get("P0_vel", 100.0)
    cons_cfg = cfg_dict.get("consensus", {})

    ekf = EKF(dt, sigma_a, sigma_bearing, range_ref, P0_pos, P0_vel)

    adj = generate_adjacency(num_trackers, topology)
    consensus_rng = np.random.default_rng(seed + 999999)
    cekf = ConsensusEKF(
        dt=dt, sigma_a=sigma_a, sigma_bearing=sigma_bearing, range_ref=range_ref,
        num_drones=num_trackers, adjacency=adj,
        num_consensus_iters=cons_cfg.get("num_iterations", 5),
        consensus_step_size=cons_cfg.get("step_size", 0.1),
        dropout_prob=dropout, P0_pos=P0_pos, P0_vel=P0_vel,
        rng=consensus_rng,
    )

    blackout_threshold = cfg_dict["monte_carlo"].get("blackout_steps", 20)

    # Track metrics for both filters
    ekf_pos_errors = []
    ekf_vel_errors = []
    ekf_nees = []
    cekf_pos_errors = []
    cekf_vel_errors = []
    cekf_nees = []
    disagreements = []

    consecutive_blind = 0
    max_blind_streak = 0

    for step_i in range(episode_length):
        target_pos_true = env.target_traj[step_i, :3]
        target_vel = env.target_traj[step_i, 3:6]

        tracker_waypoints = np.zeros((num_trackers, 3))
        tracker_vels = np.tile(target_vel, (num_trackers, 1))
        for i in range(num_trackers):
            tracker_waypoints[i] = target_pos_true + formation_offset[i]
            tracker_waypoints[i, 2] = max(drone_min_alt, tracker_waypoints[i, 2])

        # Gimbal from centralized EKF
        gimbal_est = None
        if ekf.initialized:
            gimbal_est = ekf.get_estimate()[:3]

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

        # Blackout tracking
        if all(m is None for m in meas):
            consecutive_blind += 1
            max_blind_streak = max(max_blind_streak, consecutive_blind)
        else:
            consecutive_blind = 0

        # Update both filters
        for filt in [ekf, cekf]:
            filt.predict()
            filt.update(meas, drone_pos)

        if ekf.initialized:
            x_hat = ekf.get_estimate()
            P_mat = ekf.get_covariance()
            err = true_state[:6] - x_hat
            ekf_pos_errors.append(np.linalg.norm(err[:3]))
            ekf_vel_errors.append(np.linalg.norm(err[3:6]))
            try:
                ekf_nees.append(err @ np.linalg.inv(P_mat) @ err)
            except np.linalg.LinAlgError:
                ekf_nees.append(np.nan)

        if cekf.initialized:
            x_hat = cekf.get_estimate()
            P_mat = cekf.get_covariance()
            err = true_state[:6] - x_hat
            cekf_pos_errors.append(np.linalg.norm(err[:3]))
            cekf_vel_errors.append(np.linalg.norm(err[3:6]))
            try:
                cekf_nees.append(err @ np.linalg.inv(P_mat) @ err)
            except np.linalg.LinAlgError:
                cekf_nees.append(np.nan)
            disagreements.append(cekf.get_disagreement())

    env.close()

    def _compute_metrics(pos_errors, vel_errors, nees_vals):
        pos_errors = np.array(pos_errors) if pos_errors else np.array([np.nan])
        vel_errors = np.array(vel_errors) if vel_errors else np.array([np.nan])
        nees_vals = np.array(nees_vals) if nees_vals else np.array([np.nan])

        pos_rmse = np.sqrt(np.nanmean(pos_errors**2))
        vel_rmse = np.sqrt(np.nanmean(vel_errors**2))
        valid_nees = nees_vals[~np.isnan(nees_vals)]
        anees = np.nanmean(valid_nees) if len(valid_nees) > 0 else np.nan

        conv_time = len(pos_errors)
        for t in range(len(pos_errors) - 5):
            if np.all(pos_errors[t:t + 5] < 20.0):
                conv_time = t
                break

        error_loss = 1 if (len(pos_errors) > 0 and pos_errors[-1] > 100.0) else 0
        blackout_loss = 1 if max_blind_streak >= blackout_threshold else 0
        track_loss = 1 if (error_loss or blackout_loss) else 0

        return {
            "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, "anees": anees,
            "convergence_time": conv_time, "track_loss": track_loss,
            "error_loss": error_loss, "blackout_loss": blackout_loss,
        }

    ekf_metrics = _compute_metrics(ekf_pos_errors, ekf_vel_errors, ekf_nees)
    cekf_metrics = _compute_metrics(cekf_pos_errors, cekf_vel_errors, cekf_nees)
    cekf_metrics["disagreement"] = np.mean(disagreements) if disagreements else np.nan
    cekf_metrics["rmse_ratio"] = cekf_metrics["pos_rmse"] / max(ekf_metrics["pos_rmse"], 1e-6)

    return {
        "topology": topology, "dropout": dropout, "traj": traj_type, "seed": seed,
        "ekf": ekf_metrics, "cekf": cekf_metrics,
        "max_blind_streak": max_blind_streak,
    }


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Consensus EKF Sweep")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--topology", type=str, default=None,
                        help="Single topology: full, ring, star")
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    cons_cfg = cfg.get("consensus", {})
    num_runs = args.runs or cfg["monte_carlo"]["num_runs"]

    topologies = [args.topology] if args.topology else cons_cfg.get("topologies", ["full", "ring", "star"])
    dropout_probs = cons_cfg.get("dropout_probs", [0.0, 0.1, 0.2, 0.3, 0.5])
    traj_types = [args.traj] if args.traj else ["multi_segment", "evasive"]

    n_workers = args.workers or max(1, cpu_count() - 1)
    total_trials = len(topologies) * len(dropout_probs) * len(traj_types) * num_runs

    print("=" * 70)
    print("  Monte Carlo Consensus EKF Sweep")
    print("=" * 70)
    print(f"  Topologies: {topologies}")
    print(f"  Dropout probs: {dropout_probs}")
    print(f"  Trajectories: {traj_types}")
    print(f"  Runs per combo: {num_runs}")
    print(f"  Total trials: {total_trials}")
    print(f"  Workers: {n_workers}")
    print("=" * 70)

    # Build task list
    tasks = []
    for topo in topologies:
        for dp in dropout_probs:
            for traj in traj_types:
                for run_i in range(num_runs):
                    seed = abs(hash((topo, dp, traj))) % 100000 + run_i
                    tasks.append((topo, dp, traj, seed, cfg))

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

    # Aggregate results
    consensus_results = {}   # (topology, dropout, traj) -> lists of metrics
    centralized_results = {} # traj -> lists of metrics
    errors = []

    for r in all_results:
        if "error" in r:
            errors.append(r)
            continue

        topo, dp, traj = r["topology"], r["dropout"], r["traj"]

        # Consensus
        ckey = (topo, dp, traj)
        if ckey not in consensus_results:
            consensus_results[ckey] = {
                "pos_rmse": [], "vel_rmse": [], "anees": [],
                "convergence_time": [], "track_loss": [],
                "disagreement": [], "rmse_ratio": [],
            }
        for k in consensus_results[ckey]:
            consensus_results[ckey][k].append(r["cekf"].get(k, np.nan))

        # Centralized (aggregate across all trials regardless of topo/dropout)
        if traj not in centralized_results:
            centralized_results[traj] = {
                "pos_rmse": [], "vel_rmse": [], "anees": [],
                "convergence_time": [], "track_loss": [],
            }
        for k in centralized_results[traj]:
            centralized_results[traj][k].append(r["ekf"].get(k, np.nan))

    if errors:
        print(f"\n  {len(errors)} trials failed:")
        for e in errors[:5]:
            print(f"    {e['topology']}/{e['dropout']}/{e['traj']} seed={e['seed']}: {e['error']}")

    # Print summary tables
    for traj in traj_types:
        print(f"\n{'=' * 110}")
        print(f"  Trajectory: {traj}")
        print(f"{'=' * 110}")

        # Centralized baseline
        if traj in centralized_results:
            cr = centralized_results[traj]
            pos = np.nanmean(cr["pos_rmse"])
            vel = np.nanmean(cr["vel_rmse"])
            anees = np.nanmean(cr["anees"])
            loss = np.nanmean(cr["track_loss"]) * 100
            print(f"  {'Centralized EKF':<25} | Pos: {pos:>7.2f}m | Vel: {vel:>7.2f}m/s | "
                  f"ANEES: {anees:>8.2f} | Loss: {loss:>5.1f}%")
            print("-" * 110)

        print(f"  {'Topology':<10} | {'Dropout':>7} | {'Pos RMSE':>10} | {'Vel RMSE':>10} | "
              f"{'ANEES':>8} | {'Loss%':>6} | {'Disagree':>10} | {'Ratio':>7}")
        print("-" * 110)

        for topo in topologies:
            for dp in dropout_probs:
                key = (topo, dp, traj)
                if key not in consensus_results:
                    continue
                r = consensus_results[key]
                pos = np.nanmean(r["pos_rmse"])
                vel = np.nanmean(r["vel_rmse"])
                anees = np.nanmean(r["anees"])
                loss = np.nanmean(r["track_loss"]) * 100
                disagree = np.nanmean(r["disagreement"])
                ratio = np.nanmean(r["rmse_ratio"])
                print(f"  {topo:<10} | {dp:>7.2f} | {pos:>8.2f} m | {vel:>8.2f} m/s | "
                      f"{anees:>8.2f} | {loss:>5.1f}% | {disagree:>8.2f} m | {ratio:>6.2f}x")
            print("-" * 110)

    # Convert lists to arrays
    for key in consensus_results:
        for metric in consensus_results[key]:
            consensus_results[key][metric] = np.array(consensus_results[key][metric])
    for traj in centralized_results:
        for metric in centralized_results[traj]:
            centralized_results[traj][metric] = np.array(centralized_results[traj][metric])

    # Plots
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_prefix = os.path.join(results_dir, "mc_consensus") if args.save else None

    # Dropout degradation curves (for each trajectory)
    for traj in traj_types:
        # Reformat for plot function: {(topology, dropout): {metric: array}}
        dropout_data = {}
        for topo in topologies:
            for dp in dropout_probs:
                key = (topo, dp, traj)
                if key in consensus_results:
                    dropout_data[(topo, dp)] = consensus_results[key]

        cent_baseline = None
        if traj in centralized_results:
            cent_baseline = np.nanmedian(centralized_results[traj]["pos_rmse"])

        plot_mc_dropout_comparison(
            dropout_data, topologies, dropout_probs,
            centralized_baseline=cent_baseline,
            save_path=f"{save_prefix}_dropout_{traj}.png" if save_prefix else None,
        )

    # Topology box plots (at dropout=0)
    topo_box_data = {}
    for topo in topologies:
        for traj in traj_types:
            key = (topo, 0.0, traj)
            if key in consensus_results:
                topo_box_data[(topo, traj)] = consensus_results[key]

    cent_box = {}
    for traj in traj_types:
        if traj in centralized_results:
            cent_box[traj] = centralized_results[traj]

    plot_mc_topology_boxes(
        topo_box_data, topologies, traj_types,
        centralized_results=cent_box,
        save_path=f"{save_prefix}_topo_boxes.png" if save_prefix else None,
    )

    # Save raw results
    if args.save:
        save_data = {}
        for (topo, dp, traj), metrics in consensus_results.items():
            for metric_name, values in metrics.items():
                save_data[f"{topo}_{dp}_{traj}_{metric_name}"] = values
        for traj, metrics in centralized_results.items():
            for metric_name, values in metrics.items():
                save_data[f"centralized_{traj}_{metric_name}"] = values

        np.savez_compressed(
            os.path.join(results_dir, "mc_consensus_results.npz"),
            **save_data,
            topologies=np.array(topologies),
            dropout_probs=np.array(dropout_probs),
            traj_types=np.array(traj_types),
            num_runs=num_runs,
        )
        print(f"\n  Results saved to {results_dir}/mc_consensus_results.npz")


if __name__ == "__main__":
    main()
