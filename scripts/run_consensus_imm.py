"""IMM filter comparison experiment.

Compares centralized EKF, centralized IMM, consensus EKF, consensus IMM,
and PF (god-node) on the same measurement stream.

Usage:
    python scripts/run_consensus_imm.py                                 # default 200 steps
    python scripts/run_consensus_imm.py --traj evasive --steps 3000     # stress test
    python scripts/run_consensus_imm.py --topology ring --dropout 0.3
    python scripts/run_consensus_imm.py --all-topologies
    python scripts/run_consensus_imm.py --no-gui --save
"""

import argparse
import os
import sys
import time
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.env.tracking_aviary import TrackingAviary, spawn_in_hollow_sphere
from src.filters import (
    EKF, PF, ConsensusEKF, IMM, ConsensusIMM,
    generate_adjacency,
)
from src.viz.filter_plots import plot_rmse, plot_nees
from src.viz.consensus_plots import (
    plot_per_drone_estimates,
    plot_topology_comparison,
    plot_consensus_convergence,
)


# IMM configuration
IMM_SIGMA_A_MODES = [0.3, 3.0]  # gentle, aggressive
IMM_TRANSITION = np.array([
    [0.95, 0.05],   # gentle → gentle=0.95, gentle → aggressive=0.05
    [0.10, 0.90],   # aggressive → gentle=0.10, aggressive → aggressive=0.90
])


def load_config(path: str = "config/default.yaml") -> dict:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_centralized_ekf(cfg, dt):
    sigma_a = cfg["target"]["process_noise_accel"]
    sigma_bearing = np.deg2rad(cfg["sensor"]["sigma_bearing_deg"])
    range_ref = cfg["sensor"]["range_ref"]
    finit = cfg["filters"].get("init", {})
    return EKF(dt, sigma_a, sigma_bearing, range_ref,
               finit.get("P0_pos", 10000.0), finit.get("P0_vel", 100.0))


def create_centralized_imm(cfg, dt):
    sigma_bearing = np.deg2rad(cfg["sensor"]["sigma_bearing_deg"])
    range_ref = cfg["sensor"]["range_ref"]
    finit = cfg["filters"].get("init", {})
    return IMM(
        dt=dt,
        sigma_a_modes=IMM_SIGMA_A_MODES,
        sigma_bearing=sigma_bearing,
        range_ref=range_ref,
        transition_matrix=IMM_TRANSITION,
        P0_pos=finit.get("P0_pos", 10000.0),
        P0_vel=finit.get("P0_vel", 100.0),
    )


def create_pf(cfg, dt):
    sigma_a = cfg["target"]["process_noise_accel"]
    sigma_bearing = np.deg2rad(cfg["sensor"]["sigma_bearing_deg"])
    range_ref = cfg["sensor"]["range_ref"]
    finit = cfg["filters"].get("init", {})
    pf_cfg = cfg["filters"].get("pf", {})
    return PF(
        dt=dt,
        sigma_a=sigma_a,
        sigma_bearing=sigma_bearing,
        range_ref=range_ref,
        num_particles=pf_cfg.get("num_particles", 2000),
        resample_threshold=pf_cfg.get("resample_threshold", 0.5),
        process_noise_factor=pf_cfg.get("process_noise_factor", 5.0),
        jitter_pos=pf_cfg.get("jitter_pos", 1.0),
        jitter_vel=pf_cfg.get("jitter_vel", 0.5),
        P0_pos=finit.get("P0_pos", 10000.0),
        P0_vel=finit.get("P0_vel", 100.0),
        range_min=finit.get("pf_range_min", 10.0),
        range_max=finit.get("pf_range_max", 500.0),
    )


def create_consensus_ekf(cfg, dt, topology, dropout, rng, metropolis=False):
    sigma_a = cfg["target"]["process_noise_accel"]
    sigma_bearing = np.deg2rad(cfg["sensor"]["sigma_bearing_deg"])
    range_ref = cfg["sensor"]["range_ref"]
    finit = cfg["filters"].get("init", {})
    cons_cfg = cfg.get("consensus", {})
    num_drones = cfg["drones"]["num_trackers"]
    adj = generate_adjacency(num_drones, topology)
    return ConsensusEKF(
        dt=dt, sigma_a=sigma_a,
        sigma_bearing=sigma_bearing, range_ref=range_ref,
        num_drones=num_drones, adjacency=adj,
        num_consensus_iters=cons_cfg.get("num_iterations", 5),
        consensus_step_size=cons_cfg.get("step_size", 0.1),
        dropout_prob=dropout,
        P0_pos=finit.get("P0_pos", 10000.0),
        P0_vel=finit.get("P0_vel", 100.0),
        rng=rng,
        metropolis=metropolis,
    )


def create_consensus_imm(cfg, dt, topology, dropout, rng, metropolis=False):
    sigma_bearing = np.deg2rad(cfg["sensor"]["sigma_bearing_deg"])
    range_ref = cfg["sensor"]["range_ref"]
    finit = cfg["filters"].get("init", {})
    cons_cfg = cfg.get("consensus", {})
    num_drones = cfg["drones"]["num_trackers"]
    adj = generate_adjacency(num_drones, topology)
    return ConsensusIMM(
        dt=dt,
        sigma_a_modes=IMM_SIGMA_A_MODES,
        sigma_bearing=sigma_bearing,
        range_ref=range_ref,
        transition_matrix=IMM_TRANSITION,
        num_drones=num_drones,
        adjacency=adj,
        num_consensus_iters=cons_cfg.get("num_iterations", 5),
        consensus_step_size=cons_cfg.get("step_size", 0.1),
        dropout_prob=dropout,
        P0_pos=finit.get("P0_pos", 10000.0),
        P0_vel=finit.get("P0_vel", 100.0),
        rng=rng,
        metropolis=metropolis,
    )


def run_experiment(cfg, filters, filter_names, gui, traj_type, episode_length, seed,
                   gimbal_mode="local"):
    """Run simulation with multiple filters on the same measurement stream."""
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

    # Track mode probabilities for IMM filters
    imm_indices = [i for i, f in enumerate(filters) if isinstance(f, (IMM, ConsensusIMM))]
    mode_probs = {i: np.zeros((T, len(IMM_SIGMA_A_MODES))) for i in imm_indices}

    # Per-drone local estimates for first consensus filter
    consensus_filters = [(i, f) for i, f in enumerate(filters)
                         if isinstance(f, (ConsensusEKF, ConsensusIMM))]
    has_consensus = len(consensus_filters) > 0
    local_estimates = np.zeros((num_trackers, T, 6)) if has_consensus else None
    disagreements = np.zeros(T) if has_consensus else None
    active_edges = [] if has_consensus else None

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
        elif gimbal_mode == "local" and has_consensus:
            ci, cf = consensus_filters[0]
            if cf.initialized:
                per_drone_est = np.array([
                    cf.get_local_estimate(i)[:3]
                    for i in range(num_trackers)
                ])

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

            # Record mode probabilities
            if fi in imm_indices:
                if isinstance(filt, IMM):
                    mode_probs[fi][step_i] = filt.get_mode_probabilities()
                elif isinstance(filt, ConsensusIMM):
                    # Average mode probs across drones
                    mode_probs[fi][step_i] = filt.get_mode_probabilities().mean(axis=0)

        # Record per-drone local estimates from first consensus filter
        if has_consensus:
            ci, cf = consensus_filters[0]
            if cf.initialized:
                for di in range(num_trackers):
                    local_estimates[di, step_i] = cf.get_local_estimate(di)
                disagreements[step_i] = cf.get_disagreement()
                active_edges.append(cf.get_active_edges())

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
    for k in mode_probs:
        mode_probs[k] = mode_probs[k][:actual_steps]
    if local_estimates is not None:
        local_estimates = local_estimates[:, :actual_steps]
        disagreements = disagreements[:actual_steps]

    return (record, estimates, covariances, nees, init_steps,
            local_estimates, disagreements, active_edges, mode_probs)


def compute_metrics(estimates, true_states, nees, init_steps, filter_names,
                    disagreements=None):
    """Compute and print summary metrics."""
    T = true_states.shape[0]
    results = {}

    print("\n" + "=" * 100)
    header = (f"  {'Filter':<25} | {'Pos RMSE':>10} | {'Vel RMSE':>10} | "
              f"{'ANEES':>8} | {'Conv':>6} | {'Final Err':>10}")
    if disagreements is not None:
        header += f" | {'Disagree':>10}"
    print(header)
    print("-" * 100)

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

        line = (f"  {name:<25} | {pos_rmse:>8.2f} m | {vel_rmse:>8.2f} m/s | "
                f"{anees:>8.2f} | {conv_step:>6d} | {final_err:>8.2f} m")
        if disagreements is not None:
            line += f" | {'---':>10}"
        print(line)

        results[name] = {
            "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, "anees": anees,
            "convergence_time": conv_step, "final_err": final_err,
        }

    print("=" * 100)
    return results


def plot_mode_probabilities(time_axis, mode_probs, filter_names, save_path=None):
    """Plot IMM mode probabilities over time."""
    import matplotlib.pyplot as plt

    n_plots = len(mode_probs)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), squeeze=False)
    mode_labels = ["CV-gentle (σ_a=0.3)", "CV-aggressive (σ_a=3.0)"]
    colors = ["tab:blue", "tab:red"]

    for idx, (fi, probs) in enumerate(mode_probs.items()):
        ax = axes[idx, 0]
        for j in range(probs.shape[1]):
            ax.plot(time_axis[:len(probs)], probs[:, j],
                    label=mode_labels[j], color=colors[j], linewidth=1.5)
        ax.set_ylabel("Mode probability")
        ax.set_title(f"Mode probabilities — {filter_names[fi]}")
        ax.legend(loc="upper right")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved mode probabilities: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="IMM Filter Comparison")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--traj", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology", default="full", choices=["full", "ring", "star"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-drones", type=int, default=None)
    parser.add_argument("--perfect-gimbal", action="store_true")
    parser.add_argument("--centralized-gimbal", action="store_true")
    parser.add_argument("--all-topologies", action="store_true")
    parser.add_argument("--no-pf", action="store_true", help="Skip PF (faster)")
    parser.add_argument("--metropolis", action="store_true",
                        help="Use Metropolis-Hastings consensus weights (N-agnostic)")
    parser.add_argument("--all-filters", action="store_true",
                        help="Run all filters (default behavior)")
    parser.add_argument("--only-consensus", action="store_true",
                        help="Run only consensus EKF + consensus IMM")
    parser.add_argument("--filters", type=str, default=None,
                        help="Comma-separated filter list: ekf,imm,consensus-ekf,consensus-imm,pf")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    gui = not args.no_gui
    traj_type = args.traj or cfg["target"]["trajectory"]
    dt = 1.0 / cfg["sim"]["ctrl_freq"]

    if args.num_drones is not None:
        cfg["drones"]["num_trackers"] = args.num_drones

    if args.perfect_gimbal:
        gimbal_mode = "perfect"
    elif args.centralized_gimbal:
        gimbal_mode = "centralized"
    else:
        gimbal_mode = "local"

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_prefix = os.path.join(results_dir, f"imm_{traj_type}") if args.save else None

    if args.all_topologies:
        topologies = ["full", "ring", "star"]
    else:
        topologies = [args.topology]

    # Determine which filters to run
    VALID_FILTER_KEYS = {"ekf", "imm", "consensus-ekf", "consensus-imm", "pf"}
    if args.filters:
        enabled = set(k.strip().lower() for k in args.filters.split(","))
        unknown = enabled - VALID_FILTER_KEYS
        if unknown:
            parser.error(f"Unknown filter(s): {unknown}. Valid: {sorted(VALID_FILTER_KEYS)}")
    elif args.only_consensus:
        enabled = {"consensus-ekf", "consensus-imm"}
    elif args.no_pf:
        enabled = {"ekf", "imm", "consensus-ekf", "consensus-imm"}
    else:
        # --all-filters or default
        enabled = VALID_FILTER_KEYS.copy()

    # Build filter list
    rng = np.random.default_rng(args.seed + 1000)
    filters = []
    filter_names = []

    def _consensus_label(prefix, topo):
        label = f"{prefix} ({topo}"
        if args.metropolis:
            label += ", MH"
        if args.dropout > 0:
            label += f", drop={args.dropout}"
        label += ")"
        return label

    if "ekf" in enabled:
        filters.append(create_centralized_ekf(cfg, dt))
        filter_names.append("Centralized EKF")

    if "imm" in enabled:
        filters.append(create_centralized_imm(cfg, dt))
        filter_names.append("Centralized IMM")

    if "consensus-ekf" in enabled:
        filters.append(create_consensus_ekf(cfg, dt, topologies[0], args.dropout, rng,
                                            metropolis=args.metropolis))
        filter_names.append(_consensus_label("Consensus EKF", topologies[0]))

    if "consensus-imm" in enabled:
        filters.append(create_consensus_imm(cfg, dt, topologies[0], args.dropout, rng,
                                            metropolis=args.metropolis))
        filter_names.append(_consensus_label("Consensus IMM", topologies[0]))

    if "pf" in enabled:
        filters.append(create_pf(cfg, dt))
        filter_names.append("PF (god-node)")

    # Additional topologies for consensus IMM
    if "consensus-imm" in enabled:
        for topo in topologies[1:]:
            filters.append(create_consensus_imm(cfg, dt, topo, args.dropout, rng,
                                                metropolis=args.metropolis))
            filter_names.append(_consensus_label("Consensus IMM", topo))

    if not filters:
        parser.error("No filters selected! Check --filters / --only-consensus flags.")

    print(f"Running IMM experiment: traj={traj_type}, steps={args.steps}, "
          f"seed={args.seed}, drones={cfg['drones']['num_trackers']}")
    print(f"  Topologies: {topologies}, dropout={args.dropout}, gimbal={gimbal_mode}")
    print(f"  Filters: {filter_names}")

    t0 = time.time()
    (record, estimates, covariances, nees, init_steps,
     local_estimates, disagreements, active_edges, mode_probs) = \
        run_experiment(cfg, filters, filter_names, gui, traj_type, args.steps, args.seed,
                       gimbal_mode=gimbal_mode)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s ({estimates.shape[1]} steps)")

    true_states = record["target_true_state"]
    T = true_states.shape[0]
    time_axis = np.arange(T) * dt

    # Metrics
    compute_metrics(estimates, true_states, nees, init_steps, filter_names, disagreements)

    # Plots
    plot_rmse(
        time_axis, true_states, estimates, filter_names,
        save_path=f"{save_prefix}_rmse.png" if save_prefix else None,
    )
    plot_nees(
        time_axis, nees, filter_names, state_dim=6,
        save_path=f"{save_prefix}_nees.png" if save_prefix else None,
    )

    # Mode probability plot
    if mode_probs:
        plot_mode_probabilities(
            time_axis, mode_probs, filter_names,
            save_path=f"{save_prefix}_mode_probs.png" if save_prefix else None,
        )

    # Per-drone estimates from first consensus filter
    if local_estimates is not None:
        # Find the first consensus filter index
        ci = next(i for i, f in enumerate(filters) if isinstance(f, (ConsensusEKF, ConsensusIMM)))
        # Find centralized baseline (EKF preferred, then IMM, else None)
        cent_idx = next((i for i, f in enumerate(filters) if isinstance(f, EKF)), None)
        if cent_idx is None:
            cent_idx = next((i for i, f in enumerate(filters) if isinstance(f, IMM)), None)
        cent_est = estimates[cent_idx] if cent_idx is not None else None
        cent_label = filter_names[cent_idx] if cent_idx is not None else "Centralized"
        plot_per_drone_estimates(
            time_axis, true_states, local_estimates, estimates[ci],
            cent_est, filter_names[ci], cent_label,
            save_path=f"{save_prefix}_per_drone.png" if save_prefix else None,
        )

    if disagreements is not None:
        plot_consensus_convergence(
            time_axis, disagreements,
            save_path=f"{save_prefix}_convergence.png" if save_prefix else None,
        )

    # Interactive replay
    if not args.no_replay and local_estimates is not None:
        from src.viz.animation import animate_consensus_tracking

        num_trackers = cfg["drones"]["num_trackers"]
        adj = generate_adjacency(num_trackers, topologies[0])
        plot_box = cfg["arena"].get("plot_box", 2000)

        # Find consensus filter indices (EKF and IMM)
        cekf_idx = next((i for i, f in enumerate(filters) if isinstance(f, ConsensusEKF)), None)
        cimm_idx = next((i for i, f in enumerate(filters) if isinstance(f, ConsensusIMM)), None)
        ekf_idx = next((i for i, f in enumerate(filters) if isinstance(f, EKF)), None)
        imm_idx = next((i for i, f in enumerate(filters) if isinstance(f, IMM)), None)

        # Replay 1: Consensus EKF vs Centralized EKF
        if cekf_idx is not None:
            cent_est = estimates[ekf_idx] if ekf_idx is not None else None
            cent_name = "Centralized EKF" if ekf_idx is not None else ""
            title_parts = [f"Consensus EKF"]
            if cent_name:
                title_parts.insert(0, cent_name)
            print(f"\n>> Replay: {' vs '.join(title_parts)} <<")
            animate_consensus_tracking(
                drone_positions=record["drone_positions"],
                target_true_states=true_states,
                centralized_est=cent_est,
                consensus_est=estimates[cekf_idx],
                local_estimates=local_estimates,
                adjacency=adj,
                disagreements=disagreements,
                measurements=record["measurements"],
                active_edges=active_edges,
                dt=dt,
                interval_ms=max(10, int(dt * 1000)),
                title=f"{' vs '.join(title_parts)} — {topologies[0]} — {traj_type}",
                plot_box=plot_box,
                topology_name=topologies[0],
            )

        # Replay 2: Consensus IMM vs Centralized IMM
        if cimm_idx is not None:
            cent_est = estimates[imm_idx] if imm_idx is not None else None
            cent_name = "Centralized IMM" if imm_idx is not None else ""
            title_parts = [f"Consensus IMM"]
            if cent_name:
                title_parts.insert(0, cent_name)
            print(f"\n>> Replay: {' vs '.join(title_parts)} <<")
            animate_consensus_tracking(
                drone_positions=record["drone_positions"],
                target_true_states=true_states,
                centralized_est=cent_est,
                consensus_est=estimates[cimm_idx],
                local_estimates=local_estimates,
                adjacency=adj,
                disagreements=disagreements,
                measurements=record["measurements"],
                active_edges=active_edges,
                dt=dt,
                interval_ms=max(10, int(dt * 1000)),
                title=f"{' vs '.join(title_parts)} — {topologies[0]} — {traj_type}",
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
        for k, v in mode_probs.items():
            save_dict[f"mode_probs_{filter_names[k]}"] = v
        np.savez_compressed(
            os.path.join(results_dir, f"imm_data_{traj_type}_s{args.seed}.npz"),
            **save_dict,
        )
        print(f"  Data saved to {results_dir}/")


if __name__ == "__main__":
    main()
