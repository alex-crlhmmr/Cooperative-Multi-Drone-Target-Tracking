"""Single-run consensus EKF experiment.

Runs centralized EKF + ConsensusEKF(s) on the same measurement stream and
compares performance across topologies. By default, each drone uses its own
local estimate for gimbal pointing (realistic distributed mode).

Usage:
    python scripts/run_consensus.py                              # full topology, local gimbal
    python scripts/run_consensus.py --topology ring --dropout 0.3
    python scripts/run_consensus.py --all-topologies             # compare all 3
    python scripts/run_consensus.py --sweep-L                    # L=1,2,3,5,10,20,50
    python scripts/run_consensus.py --sweep-eps                  # eps=0.01..0.5
    python scripts/run_consensus.py --sweep-dropout              # dropout 0..0.8
    python scripts/run_consensus.py --perfect-gimbal             # cheat mode for comparison
    python scripts/run_consensus.py --no-gui --save
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.filters import EKF, ConsensusEKF, generate_adjacency
from src.viz.consensus_plots import (
    plot_per_drone_estimates,
    plot_topology_comparison,
    plot_consensus_convergence,
    plot_iteration_sweep,
    plot_dropout_degradation,
)
from src.viz.filter_plots import plot_rmse, plot_nees


def load_config(path: str = "config/default.yaml") -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_consensus_ekf(cfg, dt, topology, dropout, rng,
                         num_iters=None, step_size=None):
    """Create a ConsensusEKF with given topology and dropout."""
    sigma_a = cfg["target"]["process_noise_accel"]
    sigma_bearing = np.deg2rad(cfg["sensor"]["sigma_bearing_deg"])
    range_ref = cfg["sensor"]["range_ref"]
    finit = cfg["filters"].get("init", {})
    P0_pos = finit.get("P0_pos", 10000.0)
    P0_vel = finit.get("P0_vel", 100.0)

    cons_cfg = cfg.get("consensus", {})
    num_drones = cfg["drones"]["num_trackers"]
    adj = generate_adjacency(num_drones, topology)

    return ConsensusEKF(
        dt=dt,
        sigma_a=sigma_a,
        sigma_bearing=sigma_bearing,
        range_ref=range_ref,
        num_drones=num_drones,
        adjacency=adj,
        num_consensus_iters=num_iters or cons_cfg.get("num_iterations", 5),
        consensus_step_size=step_size or cons_cfg.get("step_size", 0.1),
        dropout_prob=dropout,
        P0_pos=P0_pos,
        P0_vel=P0_vel,
        rng=rng,
    )


def create_centralized_ekf(cfg, dt):
    """Create a centralized EKF (Layer 1 baseline)."""
    sigma_a = cfg["target"]["process_noise_accel"]
    sigma_bearing = np.deg2rad(cfg["sensor"]["sigma_bearing_deg"])
    range_ref = cfg["sensor"]["range_ref"]
    finit = cfg["filters"].get("init", {})
    P0_pos = finit.get("P0_pos", 10000.0)
    P0_vel = finit.get("P0_vel", 100.0)
    return EKF(dt, sigma_a, sigma_bearing, range_ref, P0_pos, P0_vel)


def run_experiment(cfg, filters, filter_names, gui, traj_type, episode_length, seed,
                   gimbal_mode="centralized"):
    """Run simulation with centralized EKF + consensus filters.

    Args:
        gimbal_mode: "perfect" (true position), "centralized" (EKF estimate),
                     or "local" (each drone uses its own local consensus estimate)
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

    record = {"drone_positions": [], "target_true_state": [], "measurements": []}
    estimates = np.zeros((n_filters, T, 6))
    covariances = np.zeros((n_filters, T, 6, 6))
    nees = np.zeros((n_filters, T))
    init_steps = np.zeros(n_filters, dtype=int)

    # Per-drone local estimates for consensus filters (first consensus filter)
    local_estimates = np.zeros((num_trackers, T, 6)) if n_filters > 1 else None
    disagreements = np.zeros(T) if n_filters > 1 else None
    # Active edges per timestep (for dropout visualization)
    active_edges = [] if n_filters > 1 else None

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

        # Gimbal mode
        gimbal_est = None
        per_drone_est = None
        if gimbal_mode == "centralized" and filters[0].initialized:
            gimbal_est = filters[0].get_estimate()[:3]
        elif gimbal_mode == "local" and n_filters > 1:
            cons_filt = filters[1]
            if isinstance(cons_filt, ConsensusEKF) and cons_filt.initialized:
                per_drone_est = np.array([
                    cons_filt.get_local_estimate(i)[:3]
                    for i in range(num_trackers)
                ])
        # gimbal_mode == "perfect" → both stay None → true position used

        result = env.step_tracking(
            tracker_targets=tracker_waypoints,
            tracker_target_vels=tracker_vels,
            target_estimate=gimbal_est,
            per_drone_estimates=per_drone_est,
        )
        if result["done"]:
            break

        meas = result["measurements"]
        drone_pos = result["drone_positions"]
        true_state = result["target_true_state"]

        record["drone_positions"].append(drone_pos.copy())
        record["target_true_state"].append(true_state.copy())
        record["measurements"].append(meas)

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

        # Record per-drone local estimates from first consensus filter
        if n_filters > 1 and isinstance(filters[1], ConsensusEKF) and filters[1].initialized:
            for di in range(num_trackers):
                local_estimates[di, step_i] = filters[1].get_local_estimate(di)
            disagreements[step_i] = filters[1].get_disagreement()
            active_edges.append(filters[1].get_active_edges())

        actual_steps = step_i + 1
        if gui and step_i % 20 == 0:
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_dist, cameraYaw=45 + step_i * 0.05,
                cameraPitch=-25,
                cameraTargetPosition=result["target_true_pos"].tolist(),
                physicsClientId=env.CLIENT,
            )

    env.close()

    record["drone_positions"] = np.array(record["drone_positions"])
    record["target_true_state"] = np.array(record["target_true_state"])
    estimates = estimates[:, :actual_steps]
    covariances = covariances[:, :actual_steps]
    nees = nees[:, :actual_steps]
    if local_estimates is not None:
        local_estimates = local_estimates[:, :actual_steps]
        disagreements = disagreements[:actual_steps]

    return record, estimates, covariances, nees, init_steps, local_estimates, disagreements, active_edges


def compute_metrics(estimates, true_states, nees, init_steps, filter_names,
                    disagreements=None):
    """Compute and print summary metrics. Returns dict of metrics per filter."""
    T = true_states.shape[0]
    results = {}

    print("\n" + "=" * 95)
    header = (f"  {'Filter':<22} | {'Pos RMSE':>10} | {'Vel RMSE':>10} | "
              f"{'ANEES':>8} | {'Conv':>6} | {'Final Err':>10}")
    if disagreements is not None:
        header += f" | {'Disagree':>10}"
    print(header)
    print("-" * 95)

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

        conv_step = T
        for t in range(t0, T - 5):
            if np.all(pos_err[t - t0:t - t0 + 5] < 20.0):
                conv_step = t
                break

        final_err = pos_err[-1] if len(pos_err) > 0 else np.nan

        line = (f"  {name:<22} | {pos_rmse:>8.2f} m | {vel_rmse:>8.2f} m/s | "
                f"{anees:>8.2f} | {conv_step:>6d} | {final_err:>8.2f} m")
        if disagreements is not None and fi > 0:
            line += f" | {np.mean(disagreements):>8.2f} m"
        elif disagreements is not None:
            line += f" | {'---':>10}"
        print(line)

        results[name] = {
            "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, "anees": anees,
            "convergence_time": conv_step, "final_err": final_err,
        }

    print("=" * 95)
    return results


def run_sweep(cfg, dt, traj_type, episode_length, seed, gui,
              sweep_var, sweep_values, topology, base_dropout=0.0,
              base_L=5, base_eps=0.1, gimbal_mode="local"):
    """Run a parameter sweep. Returns {topology: {value: metrics}}."""
    topologies = ["full", "ring", "star"] if topology == "all" else [topology]
    all_metrics = {}

    for topo in topologies:
        all_metrics[topo] = {}
        for val in sweep_values:
            # Create filters
            rng = np.random.default_rng(seed + 1000)
            filters = [create_centralized_ekf(cfg, dt)]
            filter_names = ["Centralized EKF"]

            if sweep_var == "L":
                filt = create_consensus_ekf(cfg, dt, topo, base_dropout, rng,
                                            num_iters=val, step_size=base_eps)
                label = f"Consensus ({topo}, L={val})"
            elif sweep_var == "eps":
                filt = create_consensus_ekf(cfg, dt, topo, base_dropout, rng,
                                            num_iters=base_L, step_size=val)
                label = f"Consensus ({topo}, eps={val})"
            elif sweep_var == "dropout":
                filt = create_consensus_ekf(cfg, dt, topo, val, rng,
                                            num_iters=base_L, step_size=base_eps)
                label = f"Consensus ({topo}, drop={val})"
            else:
                raise ValueError(f"Unknown sweep var: {sweep_var}")

            filters.append(filt)
            filter_names.append(label)

            print(f"  Running {label}...", end=" ", flush=True)
            t0 = time.time()
            record, estimates, covariances, nees, init_steps, _, _, _ = \
                run_experiment(cfg, filters, filter_names, gui, traj_type,
                               episode_length, seed, gimbal_mode=gimbal_mode)
            true_states = record["target_true_state"]

            # Compute RMSE for consensus filter (index 1)
            t0_idx = max(init_steps[1], 1)
            err = true_states[t0_idx:, :6] - estimates[1, t0_idx:]
            pos_rmse = np.sqrt(np.mean(np.linalg.norm(err[:, :3], axis=1)**2))
            vel_rmse = np.sqrt(np.mean(np.linalg.norm(err[:, 3:6], axis=1)**2))

            # Centralized baseline
            err_c = true_states[t0_idx:, :6] - estimates[0, t0_idx:]
            pos_rmse_c = np.sqrt(np.mean(np.linalg.norm(err_c[:, :3], axis=1)**2))

            all_metrics[topo][val] = {
                "pos_rmse": pos_rmse,
                "vel_rmse": vel_rmse,
                "centralized_pos_rmse": pos_rmse_c,
                "ratio": pos_rmse / max(pos_rmse_c, 1e-6),
            }
            print(f"pos_rmse={pos_rmse:.2f}m (ratio={all_metrics[topo][val]['ratio']:.2f}x) "
                  f"[{time.time() - t0:.1f}s]")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Consensus EKF Experiment")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--traj", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology", default="full", choices=["full", "ring", "star"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-drones", type=int, default=None,
                        help="Override number of drones (default: from config)")
    parser.add_argument("--perfect-gimbal", action="store_true",
                        help="Use true target position for gimbal (no feedback)")
    parser.add_argument("--centralized-gimbal", action="store_true",
                        help="Use god-node centralized EKF for gimbal (unrealistic, for comparison)")
    parser.add_argument("--all-topologies", action="store_true",
                        help="Compare all 3 topologies + centralized")
    parser.add_argument("--sweep-L", action="store_true",
                        help="Sweep consensus iterations L=1,2,3,5,10,20,50")
    parser.add_argument("--sweep-eps", action="store_true",
                        help="Sweep consensus step size eps=0.01..0.5")
    parser.add_argument("--sweep-dropout", action="store_true",
                        help="Sweep dropout probability 0..0.8")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    gui = not args.no_gui
    traj_type = args.traj or cfg["target"]["trajectory"]
    dt = 1.0 / cfg["sim"]["ctrl_freq"]

    # Override num drones if requested
    if args.num_drones is not None:
        cfg["drones"]["num_trackers"] = args.num_drones

    # Gimbal mode: local is default (realistic — each drone uses its own estimate)
    if args.perfect_gimbal:
        gimbal_mode = "perfect"
    elif args.centralized_gimbal:
        gimbal_mode = "centralized"
    else:
        gimbal_mode = "local"

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_prefix = os.path.join(results_dir, f"consensus_{traj_type}") if args.save else None

    # === Parameter sweeps ===
    if args.sweep_L or args.sweep_eps or args.sweep_dropout:
        sweep_gui = False  # always headless for sweeps

        if args.sweep_L:
            print(f"\n=== Sweep: Consensus Iterations (L) [gimbal={gimbal_mode}] ===")
            L_values = [1, 2, 3, 5, 10, 20, 50]
            metrics = run_sweep(cfg, dt, traj_type, args.steps, args.seed, sweep_gui,
                                "L", L_values, "all", gimbal_mode=gimbal_mode)
            # Print table
            print(f"\n{'Topology':<8}", end="")
            for L in L_values:
                print(f" | L={L:>3}", end="")
            print()
            print("-" * (10 + 9 * len(L_values)))
            for topo in metrics:
                print(f"{topo:<8}", end="")
                for L in L_values:
                    if L in metrics[topo]:
                        print(f" | {metrics[topo][L]['pos_rmse']:>5.1f}m", end="")
                    else:
                        print(f" |    ---", end="")
                print()

            plot_iteration_sweep(
                L_values,
                {topo: [metrics[topo][L]["pos_rmse"] for L in L_values]
                 for topo in metrics},
                save_path=f"{save_prefix}_sweep_L.png" if save_prefix else None,
            )

        if args.sweep_eps:
            print(f"\n=== Sweep: Consensus Step Size (epsilon) [gimbal={gimbal_mode}] ===")
            eps_values = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
            metrics = run_sweep(cfg, dt, traj_type, args.steps, args.seed, sweep_gui,
                                "eps", eps_values, "all", gimbal_mode=gimbal_mode)
            print(f"\n{'Topology':<8}", end="")
            for eps in eps_values:
                print(f" | e={eps:.2f}", end="")
            print()
            print("-" * (10 + 10 * len(eps_values)))
            for topo in metrics:
                print(f"{topo:<8}", end="")
                for eps in eps_values:
                    if eps in metrics[topo]:
                        print(f" | {metrics[topo][eps]['pos_rmse']:>6.1f}m", end="")
                    else:
                        print(f" |     ---", end="")
                print()

            plot_iteration_sweep(
                eps_values,
                {topo: [metrics[topo][e]["pos_rmse"] for e in eps_values]
                 for topo in metrics},
                ylabel="Position RMSE (m)",
                save_path=f"{save_prefix}_sweep_eps.png" if save_prefix else None,
            )

        if args.sweep_dropout:
            print(f"\n=== Sweep: Dropout Probability [gimbal={gimbal_mode}] ===")
            dropout_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
            metrics = run_sweep(cfg, dt, traj_type, args.steps, args.seed, sweep_gui,
                                "dropout", dropout_values, "all", gimbal_mode=gimbal_mode)
            print(f"\n{'Topology':<8}", end="")
            for dp in dropout_values:
                print(f" | p={dp:.2f}", end="")
            print()
            print("-" * (10 + 10 * len(dropout_values)))
            for topo in metrics:
                print(f"{topo:<8}", end="")
                for dp in dropout_values:
                    if dp in metrics[topo]:
                        print(f" | {metrics[topo][dp]['pos_rmse']:>6.1f}m", end="")
                    else:
                        print(f" |     ---", end="")
                print()

            plot_dropout_degradation(
                dropout_values, metrics,
                save_path=f"{save_prefix}_sweep_dropout.png" if save_prefix else None,
            )

        return

    # === Standard experiment ===
    if args.all_topologies:
        topologies = ["full", "ring", "star"]
    else:
        topologies = [args.topology]

    # Build filter list: centralized EKF + one ConsensusEKF per topology
    rng = np.random.default_rng(args.seed + 1000)
    filters = [create_centralized_ekf(cfg, dt)]
    filter_names = ["Centralized EKF"]

    for topo in topologies:
        filt = create_consensus_ekf(cfg, dt, topo, args.dropout, rng)
        filters.append(filt)
        label = f"Consensus ({topo}"
        if args.dropout > 0:
            label += f", drop={args.dropout}"
        label += ")"
        filter_names.append(label)

    print(f"Running consensus experiment: traj={traj_type}, steps={args.steps}, "
          f"seed={args.seed}, drones={cfg['drones']['num_trackers']}")
    print(f"  Topologies: {topologies}, dropout={args.dropout}, gimbal={gimbal_mode}")
    print(f"  Filters: {filter_names}")

    t0 = time.time()
    record, estimates, covariances, nees, init_steps, local_estimates, disagreements, active_edges = \
        run_experiment(cfg, filters, filter_names, gui, traj_type, args.steps, args.seed,
                       gimbal_mode=gimbal_mode)
    print(f"Done in {time.time() - t0:.2f}s")

    true_states = record["target_true_state"]
    T = true_states.shape[0]
    time_axis = np.arange(T) * dt

    # Metrics
    compute_metrics(estimates, true_states, nees, init_steps, filter_names, disagreements)

    # Plots
    if local_estimates is not None:
        plot_per_drone_estimates(
            time_axis, true_states, local_estimates, estimates[1],
            estimates[0], filter_names[1],
            save_path=f"{save_prefix}_per_drone.png" if save_prefix else None,
        )

    if len(topologies) > 1:
        plot_topology_comparison(
            time_axis, true_states, estimates, filter_names,
            save_path=f"{save_prefix}_topo_compare.png" if save_prefix else None,
        )

    plot_rmse(
        time_axis, true_states, estimates, filter_names,
        save_path=f"{save_prefix}_rmse.png" if save_prefix else None,
    )
    plot_nees(
        time_axis, nees, filter_names, state_dim=6,
        save_path=f"{save_prefix}_nees.png" if save_prefix else None,
    )

    if disagreements is not None:
        plot_consensus_convergence(
            time_axis, disagreements,
            save_path=f"{save_prefix}_convergence.png" if save_prefix else None,
        )

    # Interactive consensus replay
    if not args.no_replay and local_estimates is not None:
        from src.viz.animation import animate_consensus_tracking

        num_trackers = cfg["drones"]["num_trackers"]
        adj = generate_adjacency(num_trackers, topologies[0])
        plot_box = cfg["arena"].get("plot_box", 2000)
        print("\n>> Opening interactive consensus replay <<")
        animate_consensus_tracking(
            drone_positions=record["drone_positions"],
            target_true_states=true_states,
            centralized_est=estimates[0],
            consensus_est=estimates[1],
            local_estimates=local_estimates,
            adjacency=adj,
            disagreements=disagreements,
            measurements=record["measurements"],
            active_edges=active_edges,
            dt=dt,
            interval_ms=max(10, int(dt * 1000)),
            title=f"Consensus Tracking — {topologies[0]} — {traj_type}",
            plot_box=plot_box,
            topology_name=topologies[0],
        )

    # Save data
    if args.save:
        save_dict = dict(
            estimates=estimates,
            covariances=covariances,
            nees=nees,
            true_states=true_states,
            drone_positions=record["drone_positions"],
            filter_names=np.array(filter_names),
            dt=dt,
            traj_type=traj_type,
            topologies=np.array(topologies),
            dropout=args.dropout,
        )
        if local_estimates is not None:
            save_dict["local_estimates"] = local_estimates
            save_dict["disagreements"] = disagreements
        np.savez_compressed(
            os.path.join(results_dir, f"consensus_data_{traj_type}_s{args.seed}.npz"),
            **save_dict,
        )
        print(f"  Data saved to {results_dir}/")


if __name__ == "__main__":
    main()
