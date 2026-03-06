"""Quick script: replay baseline PD controller on cluster spawn."""
import numpy as np
import pybullet as p

from src.env.tracking_aviary import TrackingAviary
from src.filters.consensus_imm import ConsensusIMM
from src.filters.topology import generate_adjacency
from src.rl.ppo import TrackingConfig
from src.viz.animation import animate_rl_tracking

cfg = TrackingConfig()
cfg.episode_length = 10000
cfg.target_trajectory = "evasive"
N = cfg.num_drones
rng = np.random.default_rng(123)

aviary = TrackingAviary(
    num_trackers=N, target_speed=cfg.target_speed,
    target_trajectory=cfg.target_trajectory, target_sigma_a=cfg.target_sigma_a,
    episode_length=cfg.episode_length, sensor_config=cfg.sensor_config,
    pyb_freq=cfg.pyb_freq, ctrl_freq=cfg.ctrl_freq, gui=False, rng=rng,
)
aviary.reset()

adj = generate_adjacency(N, cfg.topology)
filt = ConsensusIMM(
    dt=cfg.dt, sigma_a_modes=list(cfg.imm_sigma_a_modes),
    sigma_bearing=cfg.sigma_bearing_rad, range_ref=cfg.range_ref,
    transition_matrix=cfg.transition_matrix, num_drones=N, adjacency=adj,
    num_consensus_iters=cfg.consensus_iters, consensus_step_size=cfg.consensus_step_size,
    dropout_prob=cfg.dropout_prob, P0_pos=cfg.P0_pos, P0_vel=cfg.P0_vel, rng=rng,
)

# --- Cluster spawn: all drones within 2m ---
target_pos = np.array([0.0, 0.0, 50.0])
cluster_center = target_pos + np.array([100.0, 0.0, 0.0])
for i in range(N):
    jitter = rng.uniform(-1.0, 1.0, size=3)
    new_pos = cluster_center + jitter
    new_pos[2] = max(new_pos[2], cfg.min_altitude)
    p.resetBasePositionAndOrientation(
        aviary.DRONE_IDS[i], new_pos, [0, 0, 0, 1],
        physicsClientId=aviary.CLIENT,
    )

R_desired = 60.0
Kp = 2.0

all_drone_pos = []
all_target = []
all_consensus_est = []
all_local_est = []  # (T, N, 6)
all_meas = []
all_edges = []
tr_Ps = []

for step in range(cfg.episode_length):
    drone_positions = np.array([aviary._getDroneStateVector(i)[:3] for i in range(N)])

    if filt.initialized:
        velocities = np.zeros((N, 3))
        per_drone_est = np.zeros((N, 3))
        for i in range(N):
            local_est = filt.get_local_estimate(i)
            est_pos, est_vel = local_est[:3], local_est[3:6]
            per_drone_est[i] = est_pos
            radial = drone_positions[i] - est_pos
            rdist = np.linalg.norm(radial)
            if rdist > 1e-3:
                rdir = radial / rdist
            else:
                rdir = rng.standard_normal(3)
                rdir /= np.linalg.norm(rdir)
            desired_pos = est_pos + rdir * R_desired
            vel = Kp * (desired_pos - drone_positions[i]) + est_vel
            speed = np.linalg.norm(vel)
            if speed > cfg.v_max:
                vel = vel / speed * cfg.v_max
            velocities[i] = vel
        waypoints = drone_positions + velocities * cfg.dt
        waypoints[:, 2] = np.maximum(waypoints[:, 2], cfg.min_altitude)
        result = aviary.step_tracking(waypoints, velocities, per_drone_estimates=per_drone_est)
    else:
        waypoints = drone_positions
        velocities = np.zeros((N, 3))
        result = aviary.step_tracking(waypoints, velocities)

    drone_positions = np.array([aviary._getDroneStateVector(i)[:3] for i in range(N)])
    measurements = result["measurements"]
    filt.predict()
    filt.update(measurements, drone_positions)

    all_drone_pos.append(drone_positions.copy())
    all_target.append(result["target_true_state"].copy())

    consensus_est = filt.get_estimate() if filt.initialized else np.zeros(6)
    all_consensus_est.append(consensus_est.copy())

    local_step = []
    for i in range(N):
        if filt.initialized:
            local_step.append(filt.get_local_estimate(i).copy())
        else:
            local_step.append(np.zeros(6))
    all_local_est.append(local_step)

    all_meas.append(measurements)
    all_edges.append(adj.copy())

    if filt.initialized:
        P = filt.get_covariance()
        tr_Ps.append(np.trace(P[:3, :3]))
    else:
        tr_Ps.append(np.nan)

aviary.close()

print(f"Mean tr(P): {np.nanmean(tr_Ps):.1f}, Final tr(P): {tr_Ps[-1]:.1f}")

# Reshape local_estimates to (N, T, 6)
local_est_arr = np.array(all_local_est)  # (T, N, 6)
local_est_arr = local_est_arr.transpose(1, 0, 2)  # (N, T, 6)

animate_rl_tracking(
    drone_positions=np.array(all_drone_pos),
    target_true_states=np.array(all_target),
    consensus_est=np.array(all_consensus_est),
    local_estimates=local_est_arr,
    adjacency=adj,
    tr_P_history=np.array(tr_Ps),
    measurements=all_meas,
    active_edges=all_edges,
    dt=cfg.dt,
    title="Baseline PD (R=60m offset) - Cluster Spawn 10K",
    topology_name=cfg.topology,
    controller_label="Baseline PD",
)
