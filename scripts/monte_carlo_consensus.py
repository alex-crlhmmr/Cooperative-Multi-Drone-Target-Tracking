"""Monte Carlo sweep for distributed consensus filters (EKF and/or IMM).

Runs N trials for each (topology, dropout, trajectory) combination,
paired with centralized baselines using the same seed.

Usage:
    python scripts/monte_carlo_consensus.py --runs 50
    python scripts/monte_carlo_consensus.py --runs 10 --topology full
    python scripts/monte_carlo_consensus.py --runs 10 --traj multi_segment --save
    python scripts/monte_carlo_consensus.py --filters consensus-imm --runs 20
    python scripts/monte_carlo_consensus.py --filters both --runs 30
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.filters import EKF, IMM, ConsensusEKF, ConsensusIMM, generate_adjacency
from src.viz.consensus_plots import (
    plot_mc_dropout_comparison,
    plot_mc_topology_boxes,
)

# IMM configuration (matches run_consensus_imm.py)
IMM_SIGMA_A_MODES = [0.3, 3.0]  # gentle, aggressive
IMM_TRANSITION = np.array([
    [0.95, 0.05],   # gentle → gentle=0.95, gentle → aggressive=0.05
    [0.10, 0.90],   # aggressive → gentle=0.10, aggressive → aggressive=0.90
])


def load_config(path: str = "config/default.yaml") -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_trial(args):
    """Run one paired trial: centralized baseline(s) + consensus filter(s).

    Returns dict with metrics for each filter.
    """
    topology, dropout, traj_type, seed, cfg_dict, filter_mode, num_iters, step_size = args

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

    # Common filter params
    sigma_a = cfg_dict["target"]["process_noise_accel"]
    sigma_bearing = np.deg2rad(cfg_dict["sensor"]["sigma_bearing_deg"])
    range_ref = cfg_dict["sensor"]["range_ref"]
    finit = cfg_dict["filters"].get("init", {})
    P0_pos = finit.get("P0_pos", 10000.0)
    P0_vel = finit.get("P0_vel", 100.0)
    cons_cfg = cfg_dict.get("consensus", {})
    adj = generate_adjacency(num_trackers, topology)
    consensus_rng = np.random.default_rng(seed + 999999)

    run_cekf = filter_mode in ("consensus-ekf", "both")
    run_cimm = filter_mode in ("consensus-imm", "both")

    # --- Build filter dict: name -> filter ---
    filters = {}

    # Centralized EKF (always — baseline)
    filters["ekf"] = EKF(dt, sigma_a, sigma_bearing, range_ref, P0_pos, P0_vel)

    # Centralized IMM (baseline when IMM mode active)
    if run_cimm:
        filters["imm"] = IMM(
            dt=dt, sigma_a_modes=IMM_SIGMA_A_MODES, sigma_bearing=sigma_bearing,
            range_ref=range_ref, transition_matrix=IMM_TRANSITION,
            P0_pos=P0_pos, P0_vel=P0_vel,
        )

    # Resolve L and eps (CLI overrides > config)
    L = num_iters if num_iters is not None else cons_cfg.get("num_iterations", 5)
    eps = step_size if step_size is not None else cons_cfg.get("step_size", 0.1)

    # Consensus EKF
    if run_cekf:
        filters["cekf"] = ConsensusEKF(
            dt=dt, sigma_a=sigma_a, sigma_bearing=sigma_bearing, range_ref=range_ref,
            num_drones=num_trackers, adjacency=adj,
            num_consensus_iters=L,
            consensus_step_size=eps,
            dropout_prob=dropout, P0_pos=P0_pos, P0_vel=P0_vel,
            rng=consensus_rng,
        )

    # Consensus IMM
    if run_cimm:
        cimm_rng = np.random.default_rng(seed + 888888)
        filters["cimm"] = ConsensusIMM(
            dt=dt, sigma_a_modes=IMM_SIGMA_A_MODES, sigma_bearing=sigma_bearing,
            range_ref=range_ref, transition_matrix=IMM_TRANSITION,
            num_drones=num_trackers, adjacency=adj,
            num_consensus_iters=L,
            consensus_step_size=eps,
            dropout_prob=dropout, P0_pos=P0_pos, P0_vel=P0_vel,
            rng=cimm_rng,
        )

    blackout_threshold = cfg_dict["monte_carlo"].get("blackout_steps", 20)

    # Per-filter error tracking
    errors_pos = {k: [] for k in filters}
    errors_vel = {k: [] for k in filters}
    nees_vals = {k: [] for k in filters}
    disagreements = {k: [] for k in filters if k in ("cekf", "cimm")}

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
        if filters["ekf"].initialized:
            gimbal_est = filters["ekf"].get_estimate()[:3]

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

        # Update all filters
        for filt in filters.values():
            filt.predict()
            filt.update(meas, drone_pos)

        # Collect metrics
        for key, filt in filters.items():
            if not filt.initialized:
                continue
            x_hat = filt.get_estimate()
            P_mat = filt.get_covariance()
            err = true_state[:6] - x_hat
            errors_pos[key].append(np.linalg.norm(err[:3]))
            errors_vel[key].append(np.linalg.norm(err[3:6]))
            try:
                nees_vals[key].append(err @ np.linalg.inv(P_mat) @ err)
            except np.linalg.LinAlgError:
                nees_vals[key].append(np.nan)

            if key in disagreements:
                disagreements[key].append(filt.get_disagreement())

    env.close()

    def _compute_metrics(pos_errors, vel_errors, nees_list):
        pos_errors = np.array(pos_errors) if pos_errors else np.array([np.nan])
        vel_errors = np.array(vel_errors) if vel_errors else np.array([np.nan])
        nees_list = np.array(nees_list) if nees_list else np.array([np.nan])

        pos_rmse = np.sqrt(np.nanmean(pos_errors**2))
        vel_rmse = np.sqrt(np.nanmean(vel_errors**2))
        valid_nees = nees_list[~np.isnan(nees_list)]
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

    # Compute per-filter metrics
    result_dict = {
        "topology": topology, "dropout": dropout, "traj": traj_type, "seed": seed,
        "max_blind_streak": max_blind_streak,
        "filter_keys": list(filters.keys()),
    }

    ekf_rmse = np.sqrt(np.nanmean(np.array(errors_pos["ekf"])**2)) if errors_pos["ekf"] else 1e-6

    for key in filters:
        m = _compute_metrics(errors_pos[key], errors_vel[key], nees_vals[key])
        if key in disagreements:
            m["disagreement"] = np.mean(disagreements[key]) if disagreements[key] else np.nan
            m["rmse_ratio"] = m["pos_rmse"] / max(ekf_rmse, 1e-6)
        result_dict[key] = m

    return result_dict


# ---------------------------------------------------------------------------
# Display names for printing
# ---------------------------------------------------------------------------
FILTER_DISPLAY = {
    "ekf": "Centralized EKF",
    "imm": "Centralized IMM",
    "cekf": "Consensus EKF",
    "cimm": "Consensus IMM",
}

CONSENSUS_KEYS = ("cekf", "cimm")
CENTRALIZED_KEYS = ("ekf", "imm")


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Consensus Filter Sweep")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--topology", type=str, default=None,
                        help="Single topology: full, ring, star")
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--filters", type=str, default="consensus-ekf",
                        choices=["consensus-ekf", "consensus-imm", "both"],
                        help="Which consensus filter(s) to run (default: consensus-ekf)")
    parser.add_argument("--dropout", type=str, default=None,
                        help="Dropout probs, comma-separated (e.g. '0.0,0.1,0.3'). "
                             "Default: from config")
    parser.add_argument("--L", type=int, default=None,
                        help="Number of consensus iterations (default: from config)")
    parser.add_argument("--eps", type=float, default=None,
                        help="Consensus step size (default: from config)")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    cons_cfg = cfg.get("consensus", {})
    num_runs = args.runs or cfg["monte_carlo"]["num_runs"]
    filter_mode = args.filters

    topologies = [args.topology] if args.topology else cons_cfg.get("topologies", ["full", "ring", "star"])
    if args.dropout is not None:
        dropout_probs = [float(x.strip()) for x in args.dropout.split(",")]
    else:
        dropout_probs = cons_cfg.get("dropout_probs", [0.0, 0.1, 0.2, 0.3, 0.5])
    traj_types = [args.traj] if args.traj else ["multi_segment", "evasive"]
    num_iters = args.L
    step_size = args.eps

    n_workers = args.workers or max(1, cpu_count() - 1)
    total_trials = len(topologies) * len(dropout_probs) * len(traj_types) * num_runs

    # Determine which filter keys will be present
    if filter_mode == "consensus-ekf":
        consensus_keys = ["cekf"]
        centralized_keys = ["ekf"]
    elif filter_mode == "consensus-imm":
        consensus_keys = ["cimm"]
        centralized_keys = ["ekf", "imm"]
    else:  # both
        consensus_keys = ["cekf", "cimm"]
        centralized_keys = ["ekf", "imm"]

    print("=" * 70)
    print("  Monte Carlo Consensus Filter Sweep")
    print("=" * 70)
    print(f"  Filter mode: {filter_mode}")
    L_display = num_iters if num_iters is not None else f"config ({cons_cfg.get('num_iterations', 5)})"
    eps_display = step_size if step_size is not None else f"config ({cons_cfg.get('step_size', 0.1)})"
    print(f"  Consensus: {[FILTER_DISPLAY[k] for k in consensus_keys]}")
    print(f"  Baselines: {[FILTER_DISPLAY[k] for k in centralized_keys]}")
    print(f"  Topologies: {topologies}")
    print(f"  Dropout probs: {dropout_probs}")
    print(f"  L (consensus iters): {L_display}")
    print(f"  eps (step size): {eps_display}")
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
                    tasks.append((topo, dp, traj, seed, cfg, filter_mode,
                                 num_iters, step_size))

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
    # consensus_results: (filter_key, topology, dropout, traj) -> {metric: [values]}
    consensus_results = {}
    # centralized_results: (filter_key, traj) -> {metric: [values]}
    centralized_results = {}
    errors = []

    metric_keys = ["pos_rmse", "vel_rmse", "anees", "convergence_time", "track_loss"]
    consensus_metric_keys = metric_keys + ["disagreement", "rmse_ratio"]

    for r in all_results:
        if "error" in r:
            errors.append(r)
            continue

        topo, dp, traj = r["topology"], r["dropout"], r["traj"]

        # Consensus filters
        for ck in consensus_keys:
            if ck not in r:
                continue
            agg_key = (ck, topo, dp, traj)
            if agg_key not in consensus_results:
                consensus_results[agg_key] = {k: [] for k in consensus_metric_keys}
            for k in consensus_metric_keys:
                consensus_results[agg_key][k].append(r[ck].get(k, np.nan))

        # Centralized baselines
        for bk in centralized_keys:
            if bk not in r:
                continue
            agg_key = (bk, traj)
            if agg_key not in centralized_results:
                centralized_results[agg_key] = {k: [] for k in metric_keys}
            for k in metric_keys:
                centralized_results[agg_key][k].append(r[bk].get(k, np.nan))

    if errors:
        print(f"\n  {len(errors)} trials failed:")
        for e in errors[:5]:
            print(f"    {e['topology']}/{e['dropout']}/{e['traj']} seed={e['seed']}: {e['error']}")

    # Print summary tables
    for traj in traj_types:
        print(f"\n{'=' * 120}")
        print(f"  Trajectory: {traj}")
        print(f"{'=' * 120}")

        # Centralized baselines
        for bk in centralized_keys:
            bkey = (bk, traj)
            if bkey not in centralized_results:
                continue
            cr = centralized_results[bkey]
            pos = np.nanmean(cr["pos_rmse"])
            vel = np.nanmean(cr["vel_rmse"])
            anees = np.nanmean(cr["anees"])
            loss = np.nanmean(cr["track_loss"]) * 100
            print(f"  {FILTER_DISPLAY[bk]:<25} | Pos: {pos:>7.2f}m | Vel: {vel:>7.2f}m/s | "
                  f"ANEES: {anees:>8.2f} | Loss: {loss:>5.1f}%")
        print("-" * 120)

        print(f"  {'Filter':<16} | {'Topology':<8} | {'Dropout':>7} | {'Pos RMSE':>10} | "
              f"{'Vel RMSE':>10} | {'ANEES':>8} | {'Loss%':>6} | {'Disagree':>10} | {'Ratio':>7}")
        print("-" * 120)

        for ck in consensus_keys:
            for topo in topologies:
                for dp in dropout_probs:
                    key = (ck, topo, dp, traj)
                    if key not in consensus_results:
                        continue
                    r = consensus_results[key]
                    pos = np.nanmean(r["pos_rmse"])
                    vel = np.nanmean(r["vel_rmse"])
                    anees = np.nanmean(r["anees"])
                    loss = np.nanmean(r["track_loss"]) * 100
                    disagree = np.nanmean(r["disagreement"])
                    ratio = np.nanmean(r["rmse_ratio"])
                    print(f"  {FILTER_DISPLAY[ck]:<16} | {topo:<8} | {dp:>7.2f} | "
                          f"{pos:>8.2f} m | {vel:>8.2f} m/s | "
                          f"{anees:>8.2f} | {loss:>5.1f}% | {disagree:>8.2f} m | {ratio:>6.2f}x")
                print("-" * 120)

    # Convert lists to arrays
    for key in consensus_results:
        for metric in consensus_results[key]:
            consensus_results[key][metric] = np.array(consensus_results[key][metric])
    for key in centralized_results:
        for metric in centralized_results[key]:
            centralized_results[key][metric] = np.array(centralized_results[key][metric])

    # Plots — one set per consensus filter
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_prefix = os.path.join(results_dir, "mc_consensus") if args.save else None

    for ck in consensus_keys:
        ck_label = ck  # "cekf" or "cimm"

        # Dropout degradation curves (for each trajectory)
        for traj in traj_types:
            dropout_data = {}
            for topo in topologies:
                for dp in dropout_probs:
                    key = (ck, topo, dp, traj)
                    if key in consensus_results:
                        dropout_data[(topo, dp)] = consensus_results[key]

            cent_baseline = None
            bkey = ("ekf", traj)
            if bkey in centralized_results:
                cent_baseline = np.nanmedian(centralized_results[bkey]["pos_rmse"])

            plot_mc_dropout_comparison(
                dropout_data, topologies, dropout_probs,
                centralized_baseline=cent_baseline,
                save_path=(f"{save_prefix}_{ck_label}_dropout_{traj}.png"
                           if save_prefix else None),
            )

        # Topology box plots (at dropout=0)
        topo_box_data = {}
        for topo in topologies:
            for traj in traj_types:
                key = (ck, topo, 0.0, traj)
                if key in consensus_results:
                    topo_box_data[(topo, traj)] = consensus_results[key]

        cent_box = {}
        for traj in traj_types:
            bkey = ("ekf", traj)
            if bkey in centralized_results:
                cent_box[traj] = centralized_results[bkey]

        plot_mc_topology_boxes(
            topo_box_data, topologies, traj_types,
            centralized_results=cent_box,
            save_path=(f"{save_prefix}_{ck_label}_topo_boxes.png"
                       if save_prefix else None),
        )

    # Save raw results
    if args.save:
        save_data = {}
        for (ck, topo, dp, traj), metrics in consensus_results.items():
            for metric_name, values in metrics.items():
                save_data[f"{ck}_{topo}_{dp}_{traj}_{metric_name}"] = values
        for (bk, traj), metrics in centralized_results.items():
            for metric_name, values in metrics.items():
                save_data[f"{bk}_{traj}_{metric_name}"] = values

        np.savez_compressed(
            os.path.join(results_dir, "mc_consensus_results.npz"),
            **save_data,
            topologies=np.array(topologies),
            dropout_probs=np.array(dropout_probs),
            traj_types=np.array(traj_types),
            filter_mode=filter_mode,
            num_runs=num_runs,
        )
        print(f"\n  Results saved to {results_dir}/mc_consensus_results.npz")


if __name__ == "__main__":
    main()
