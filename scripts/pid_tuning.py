"""PID Tuning Test Bench for the Surveillance Drone.

Runs step-response tests (altitude, lateral, combined, waypoint tracking),
computes control metrics, and produces diagnostic plots.

Usage:
    conda activate aa273
    python scripts/pid_tuning.py                  # All tests, headless
    python scripts/pid_tuning.py --gui             # With PyBullet viewer
    python scripts/pid_tuning.py --test altitude   # Single test
    python scripts/pid_tuning.py --sweep           # Gain sweep (altitude channel)
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.env.patch_drone_model import ensure_surveillance_drone
ensure_surveillance_drone()

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.control.surveillance_pid import SurveillancePIDControl


# ── Config ──────────────────────────────────────────────────────────────────

PYB_FREQ = 480
CTRL_FREQ = 48
CTRL_DT = 1.0 / CTRL_FREQ
SETTLE_PCT = 0.02       # ±2% band for settling time
RISE_LO, RISE_HI = 0.1, 0.9  # 10→90% rise time definition

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "pid_tuning")


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_env(start_pos, gui=False):
    """Spawn a single surveillance drone at start_pos."""
    initial_xyzs = np.array(start_pos, dtype=np.float64).reshape(1, 3)
    env = CtrlAviary(
        drone_model=DroneModel.SURVEILLANCE,
        num_drones=1,
        initial_xyzs=initial_xyzs,
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB_GND_DRAG_DW,
        pyb_freq=PYB_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )
    ctrl = SurveillancePIDControl(drone_model=DroneModel.SURVEILLANCE)
    return env, ctrl


def get_state(env, idx=0):
    """Return (pos, quat, vel, ang_vel) for drone idx."""
    s = env._getDroneStateVector(idx)
    return s[:3].copy(), s[3:7].copy(), s[10:13].copy(), s[13:16].copy()


def run_steps(env, ctrl, target_pos_fn, n_steps, target_vel_fn=None):
    """Run n_steps of PID control, recording state at each ctrl step.

    target_pos_fn(t) -> (3,)  target position at time t
    target_vel_fn(t) -> (3,)  optional feedforward velocity
    """
    positions, velocities, rpms_log, quats_log = [], [], [], []
    times = []

    for step in range(n_steps):
        t = step * CTRL_DT
        pos, quat, vel, ang_vel = get_state(env)

        tgt_pos = np.asarray(target_pos_fn(t), dtype=np.float64)
        tgt_vel = np.asarray(target_vel_fn(t), dtype=np.float64) if target_vel_fn else np.zeros(3)

        rpm, _, _ = ctrl.computeControl(
            control_timestep=CTRL_DT,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_vel,
            target_pos=tgt_pos,
            target_vel=tgt_vel,
        )

        action = np.zeros((1, 4))
        action[0, :] = rpm
        env.step(action)

        positions.append(pos)
        velocities.append(vel)
        rpms_log.append(rpm.copy())
        quats_log.append(quat)
        times.append(t)

    return {
        "t": np.array(times),
        "pos": np.array(positions),
        "vel": np.array(velocities),
        "rpm": np.array(rpms_log),
        "quat": np.array(quats_log),
    }


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(t, response, step_start, step_end):
    """Compute rise time, overshoot, settling time, steady-state error for a 1D step.

    response: 1D array (actual values over time)
    step_start: initial value before step
    step_end: commanded final value
    """
    step_size = step_end - step_start
    if abs(step_size) < 1e-9:
        return {"rise_time": np.nan, "overshoot_pct": np.nan,
                "settling_time": np.nan, "ss_error": np.nan}

    # Normalize: 0 = initial, 1 = final
    normed = (response - step_start) / step_size

    # Rise time: first time from 10% to 90%
    t_lo = t[np.argmax(normed >= RISE_LO)] if np.any(normed >= RISE_LO) else np.nan
    t_hi = t[np.argmax(normed >= RISE_HI)] if np.any(normed >= RISE_HI) else np.nan
    rise_time = t_hi - t_lo if not (np.isnan(t_lo) or np.isnan(t_hi)) else np.nan

    # Overshoot
    if step_size > 0:
        peak = np.max(response)
    else:
        peak = np.min(response)
    overshoot_pct = 100.0 * abs(peak - step_end) / abs(step_size)
    # Only count as overshoot if it actually exceeds target
    if step_size > 0 and peak <= step_end:
        overshoot_pct = 0.0
    elif step_size < 0 and peak >= step_end:
        overshoot_pct = 0.0

    # Settling time: last time the signal leaves the ±2% band
    band = abs(step_size) * SETTLE_PCT
    within_band = np.abs(response - step_end) <= band
    if np.any(within_band):
        # Find the last crossing out of band, settling time = first time it stays in
        settled_from = 0
        for i in range(len(within_band) - 1, -1, -1):
            if not within_band[i]:
                settled_from = i + 1
                break
        settling_time = t[settled_from] - t[0] if settled_from < len(t) else np.nan
    else:
        settling_time = np.nan

    # Steady-state error (average of last 10% of signal)
    tail = max(1, len(response) // 10)
    ss_error = abs(np.mean(response[-tail:]) - step_end)

    return {
        "rise_time": rise_time,
        "overshoot_pct": overshoot_pct,
        "settling_time": settling_time,
        "ss_error": ss_error,
    }


# ── Tests ───────────────────────────────────────────────────────────────────

def test_altitude_step(gui=False):
    """Test 1: Altitude step from 50m to 70m."""
    print("\n" + "=" * 60)
    print("  TEST 1: Altitude Step  (z: 50m → 70m)")
    print("=" * 60)

    start = [0.0, 0.0, 50.0]
    target_z = 70.0
    n_steps = int(10.0 / CTRL_DT)  # 10 seconds

    env, ctrl = make_env(start, gui=gui)
    env.reset()

    # Let the drone stabilize for 2 seconds first
    stab_steps = int(2.0 / CTRL_DT)
    run_steps(env, ctrl, lambda t: start, stab_steps)
    ctrl.reset()  # reset integrators

    data = run_steps(env, ctrl,
                     lambda t: [0.0, 0.0, target_z],
                     n_steps)
    env.close()

    # Compute hover RPM for normalization
    mass = 4.0; g = 9.8; kf = 5e-7
    hover_rpm = np.sqrt(mass * g / (4 * kf))

    metrics = compute_metrics(data["t"], data["pos"][:, 2], start[2], target_z)
    metrics["peak_rpm_ratio"] = np.max(data["rpm"]) / hover_rpm

    _print_metrics("Altitude (Z)", metrics)
    return data, metrics, start, [0.0, 0.0, target_z]


def test_lateral_step(gui=False):
    """Test 2: Lateral step from (0,0,50) to (20,0,50)."""
    print("\n" + "=" * 60)
    print("  TEST 2: Lateral Step  (x: 0m → 20m)")
    print("=" * 60)

    start = [0.0, 0.0, 50.0]
    target = [20.0, 0.0, 50.0]
    n_steps = int(15.0 / CTRL_DT)  # 15 seconds (lateral is slower)

    env, ctrl = make_env(start, gui=gui)
    env.reset()

    stab_steps = int(2.0 / CTRL_DT)
    run_steps(env, ctrl, lambda t: start, stab_steps)
    ctrl.reset()

    data = run_steps(env, ctrl, lambda t: target, n_steps)
    env.close()

    mass = 4.0; g = 9.8; kf = 5e-7
    hover_rpm = np.sqrt(mass * g / (4 * kf))

    metrics_x = compute_metrics(data["t"], data["pos"][:, 0], start[0], target[0])
    z_deviation = np.max(np.abs(data["pos"][:, 2] - start[2]))

    metrics = {**metrics_x, "z_deviation": z_deviation,
               "peak_rpm_ratio": np.max(data["rpm"]) / hover_rpm}

    _print_metrics("Lateral (X)", metrics)
    return data, metrics, start, target


def test_combined_step(gui=False):
    """Test 3: Combined diagonal step (+20, +20, +10)."""
    print("\n" + "=" * 60)
    print("  TEST 3: Combined Step  (+20, +20, +10)")
    print("=" * 60)

    start = [0.0, 0.0, 50.0]
    target = [20.0, 20.0, 60.0]
    n_steps = int(15.0 / CTRL_DT)

    env, ctrl = make_env(start, gui=gui)
    env.reset()

    stab_steps = int(2.0 / CTRL_DT)
    run_steps(env, ctrl, lambda t: start, stab_steps)
    ctrl.reset()

    data = run_steps(env, ctrl, lambda t: target, n_steps)
    env.close()

    mass = 4.0; g = 9.8; kf = 5e-7
    hover_rpm = np.sqrt(mass * g / (4 * kf))

    # 3D position error over time
    err_3d = np.linalg.norm(data["pos"] - np.array(target), axis=1)
    step_3d = np.linalg.norm(np.array(target) - np.array(start))

    # Settling: when 3D error stays below 2% of step magnitude
    band = step_3d * SETTLE_PCT
    within = err_3d <= band
    if np.any(within):
        settled_from = 0
        for i in range(len(within) - 1, -1, -1):
            if not within[i]:
                settled_from = i + 1
                break
        settling_3d = data["t"][settled_from] if settled_from < len(data["t"]) else np.nan
    else:
        settling_3d = np.nan

    tail = max(1, len(err_3d) // 10)
    ss_err_3d = np.mean(err_3d[-tail:])

    metrics = {
        "settling_time_3d": settling_3d,
        "ss_error_3d": ss_err_3d,
        "peak_rpm_ratio": np.max(data["rpm"]) / hover_rpm,
    }
    # Also per-axis
    for ax, name in enumerate(["x", "y", "z"]):
        m = compute_metrics(data["t"], data["pos"][:, ax], start[ax], target[ax])
        metrics[f"rise_time_{name}"] = m["rise_time"]
        metrics[f"overshoot_{name}"] = m["overshoot_pct"]

    _print_metrics("Combined 3D", metrics)
    return data, metrics, start, target


def test_waypoint_tracking(gui=False):
    """Test 4: Track a moving waypoint at 12 m/s along X."""
    print("\n" + "=" * 60)
    print("  TEST 4: Waypoint Tracking  (12 m/s along X)")
    print("=" * 60)

    speed = 12.0
    start = [0.0, 0.0, 50.0]
    duration = 20.0  # seconds
    n_steps = int(duration / CTRL_DT)

    env, ctrl = make_env(start, gui=gui)
    env.reset()

    # Stabilize
    stab_steps = int(2.0 / CTRL_DT)
    run_steps(env, ctrl, lambda t: start, stab_steps)
    ctrl.reset()

    def target_pos(t):
        return [speed * t, 0.0, 50.0]

    def target_vel(t):
        return [speed, 0.0, 0.0]

    data = run_steps(env, ctrl, target_pos, n_steps, target_vel_fn=target_vel)
    env.close()

    # Compute tracking error over time
    target_positions = np.array([target_pos(t) for t in data["t"]])
    tracking_err = np.linalg.norm(data["pos"] - target_positions, axis=1)

    # Steady-state tracking lag (last 50% of run, after transient)
    half = len(tracking_err) // 2
    ss_lag = np.mean(tracking_err[half:])
    max_lag = np.max(tracking_err[half:])

    # X-axis lag specifically
    x_lag = np.mean(target_positions[half:, 0] - data["pos"][half:, 0])

    mass = 4.0; g = 9.8; kf = 5e-7
    hover_rpm = np.sqrt(mass * g / (4 * kf))

    metrics = {
        "ss_tracking_lag": ss_lag,
        "max_tracking_lag": max_lag,
        "x_lag": x_lag,
        "peak_rpm_ratio": np.max(data["rpm"]) / hover_rpm,
    }

    _print_metrics("Waypoint Tracking", metrics)
    return data, metrics, target_positions


# ── Printing ────────────────────────────────────────────────────────────────

def _print_metrics(label, metrics):
    print(f"\n  {label} Metrics:")
    print(f"  {'─' * 40}")
    for k, v in metrics.items():
        if "time" in k:
            unit = "s"
        elif "pct" in k or "overshoot" in k:
            unit = "%"
        elif "error" in k or "deviation" in k or "lag" in k:
            unit = "m"
        elif "ratio" in k:
            unit = "x"
        else:
            unit = ""
        if np.isnan(v):
            print(f"    {k:25s}  N/A")
        else:
            print(f"    {k:25s}  {v:8.3f} {unit}")


def print_summary_table(all_metrics):
    """Print a consolidated summary of all tests."""
    print("\n" + "=" * 70)
    print("  PID TUNING SUMMARY")
    print("=" * 70)

    targets = {
        "rise_time": ("< 2.0 s", 2.0),
        "overshoot_pct": ("< 15 %", 15.0),
        "settling_time": ("< 4.0 s", 4.0),
        "ss_error": ("< 0.5 m", 0.5),
    }

    for test_name, metrics in all_metrics.items():
        print(f"\n  {test_name}:")
        for k, v in metrics.items():
            target_str = ""
            if k in targets:
                label, thresh = targets[k]
                passed = v <= thresh if not np.isnan(v) else False
                target_str = f"  [{'PASS' if passed else 'FAIL'}  target {label}]"

            if np.isnan(v):
                print(f"    {k:25s}  {'N/A':>10s}{target_str}")
            else:
                print(f"    {k:25s}  {v:10.3f}{target_str}")

    print("\n" + "=" * 70)


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_step_response(data, metrics, start, target, test_name, save_path=None):
    """4-panel plot: position, velocity, RPM, euler angles."""
    import pybullet as pb

    t = data["t"]
    pos = data["pos"]
    vel = data["vel"]
    rpm = data["rpm"]

    # Euler angles from quaternions
    euler = np.array([pb.getEulerFromQuaternion(q) for q in data["quat"]])
    euler_deg = np.degrees(euler)

    mass = 4.0; g = 9.8; kf = 5e-7
    hover_rpm = np.sqrt(mass * g / (4 * kf))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"PID Step Response — {test_name}", fontsize=14, fontweight="bold")

    # Panel 1: Position
    ax = axes[0, 0]
    labels = ["X", "Y", "Z"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    target_arr = np.array(target)
    for i in range(3):
        ax.plot(t, pos[:, i], color=colors[i], label=f"{labels[i]} actual")
        ax.axhline(target_arr[i], color=colors[i], linestyle="--", alpha=0.5,
                    label=f"{labels[i]} target = {target_arr[i]:.0f}m")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Position Response")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Velocity
    ax = axes[0, 1]
    for i in range(3):
        ax.plot(t, vel[:, i], color=colors[i], label=f"v{labels[i].lower()}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: RPM
    ax = axes[1, 0]
    for m in range(4):
        ax.plot(t, rpm[:, m], alpha=0.7, label=f"Motor {m+1}")
    ax.axhline(hover_rpm, color="k", linestyle="--", alpha=0.5, label=f"Hover RPM = {hover_rpm:.0f}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RPM")
    ax.set_title("Motor RPMs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Euler angles
    ax = axes[1, 1]
    rpy_labels = ["Roll", "Pitch", "Yaw"]
    for i in range(3):
        ax.plot(t, euler_deg[:, i], label=rpy_labels[i])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [deg]")
    ax.set_title("Attitude (Euler)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add metrics text box
    metric_text = "\n".join(
        f"{k}: {v:.3f}" if not np.isnan(v) else f"{k}: N/A"
        for k, v in metrics.items()
    )
    fig.text(0.02, 0.02, metric_text, fontsize=7, family="monospace",
             va="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_tracking(data, metrics, target_positions, save_path=None):
    """Plot for waypoint tracking test."""
    import pybullet as pb

    t = data["t"]
    pos = data["pos"]
    vel = data["vel"]
    rpm = data["rpm"]
    euler = np.array([pb.getEulerFromQuaternion(q) for q in data["quat"]])
    euler_deg = np.degrees(euler)

    mass = 4.0; g = 9.8; kf = 5e-7
    hover_rpm = np.sqrt(mass * g / (4 * kf))

    tracking_err = np.linalg.norm(pos - target_positions, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PID Waypoint Tracking — 12 m/s", fontsize=14, fontweight="bold")

    # Panel 1: X position (actual vs target)
    ax = axes[0, 0]
    ax.plot(t, pos[:, 0], "tab:red", label="X actual")
    ax.plot(t, target_positions[:, 0], "tab:red", linestyle="--", alpha=0.5, label="X target")
    ax.plot(t, pos[:, 2], "tab:blue", label="Z actual")
    ax.axhline(50.0, color="tab:blue", linestyle="--", alpha=0.5, label="Z target = 50m")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.set_title("Position (X tracking + Z hold)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Tracking error
    ax = axes[0, 1]
    ax.plot(t, tracking_err, "k", label="3D tracking error")
    ax.axhline(metrics["ss_tracking_lag"], color="orange", linestyle="--",
               label=f"SS lag = {metrics['ss_tracking_lag']:.2f} m")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [m]")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: RPMs
    ax = axes[1, 0]
    for m in range(4):
        ax.plot(t, rpm[:, m], alpha=0.7, label=f"Motor {m+1}")
    ax.axhline(hover_rpm, color="k", linestyle="--", alpha=0.5, label=f"Hover = {hover_rpm:.0f}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RPM")
    ax.set_title("Motor RPMs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Euler angles
    ax = axes[1, 1]
    for i, lbl in enumerate(["Roll", "Pitch", "Yaw"]):
        ax.plot(t, euler_deg[:, i], label=lbl)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [deg]")
    ax.set_title("Attitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# ── Gain Sweep ──────────────────────────────────────────────────────────────

def sweep_altitude_gains(gui=False):
    """Sweep P_z and D_z gains for the altitude channel, measuring performance."""
    print("\n" + "=" * 60)
    print("  GAIN SWEEP: Altitude Channel (P_z, D_z)")
    print("=" * 60)

    start = [0.0, 0.0, 50.0]
    target_z = 70.0
    n_steps = int(10.0 / CTRL_DT)

    # Parameter grid
    P_z_vals = [2.0, 3.0, 5.0, 7.0, 10.0]
    D_z_vals = [2.0, 4.0, 6.0, 8.0, 10.0]

    results = []

    for pz in P_z_vals:
        for dz in D_z_vals:
            env, ctrl = make_env(start, gui=False)
            env.reset()

            # Override gains
            ctrl.P_COEFF_FOR[2] = pz
            ctrl.D_COEFF_FOR[2] = dz
            # Keep I fixed
            ctrl.I_COEFF_FOR[2] = 0.5

            # Stabilize
            stab_steps = int(2.0 / CTRL_DT)
            run_steps(env, ctrl, lambda t: start, stab_steps)
            ctrl.reset()

            data = run_steps(env, ctrl, lambda t: [0.0, 0.0, target_z], n_steps)
            env.close()

            m = compute_metrics(data["t"], data["pos"][:, 2], start[2], target_z)
            results.append({
                "P_z": pz, "D_z": dz,
                **m,
            })

            status = "OK" if m["settling_time"] < 4.0 and m["overshoot_pct"] < 15.0 else "  "
            print(f"  P_z={pz:5.1f}  D_z={dz:5.1f}  |  "
                  f"rise={m['rise_time']:5.2f}s  OS={m['overshoot_pct']:5.1f}%  "
                  f"settle={m['settling_time']:5.2f}s  sse={m['ss_error']:5.3f}m  {status}")

    # Find best based on weighted score: 0.3*rise + 0.4*settling + 0.3*overshoot_norm
    print("\n  Best candidates (lowest weighted score):")
    for r in results:
        rt = r["rise_time"] if not np.isnan(r["rise_time"]) else 20.0
        st = r["settling_time"] if not np.isnan(r["settling_time"]) else 20.0
        os_norm = r["overshoot_pct"] / 100.0
        r["score"] = 0.3 * rt + 0.4 * st + 0.3 * os_norm

    results.sort(key=lambda r: r["score"])
    for r in results[:5]:
        print(f"    P_z={r['P_z']:5.1f}  D_z={r['D_z']:5.1f}  score={r['score']:.3f}  "
              f"(rise={r['rise_time']:.2f}s, OS={r['overshoot_pct']:.1f}%, settle={r['settling_time']:.2f}s)")

    # Plot sweep heatmap
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Altitude Gain Sweep — P_z vs D_z", fontsize=13, fontweight="bold")

    for ax_idx, metric_name, title, vmax in [
        (0, "rise_time", "Rise Time [s]", 5.0),
        (1, "overshoot_pct", "Overshoot [%]", 30.0),
        (2, "settling_time", "Settling Time [s]", 10.0),
    ]:
        grid = np.full((len(D_z_vals), len(P_z_vals)), np.nan)
        for r in results:
            pi = P_z_vals.index(r["P_z"])
            di = D_z_vals.index(r["D_z"])
            grid[di, pi] = r[metric_name]

        ax = axes[ax_idx]
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn_r",
                        vmin=0, vmax=vmax)
        ax.set_xticks(range(len(P_z_vals)))
        ax.set_xticklabels([f"{v:.0f}" for v in P_z_vals])
        ax.set_yticks(range(len(D_z_vals)))
        ax.set_yticklabels([f"{v:.0f}" for v in D_z_vals])
        ax.set_xlabel("P_z")
        ax.set_ylabel("D_z")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        # Annotate cells
        for di in range(len(D_z_vals)):
            for pi in range(len(P_z_vals)):
                val = grid[di, pi]
                if not np.isnan(val):
                    ax.text(pi, di, f"{val:.1f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "sweep_altitude.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")

    return results


def sweep_lateral_gains(gui=False):
    """Sweep P_xy and D_xy gains for the lateral channel."""
    print("\n" + "=" * 60)
    print("  GAIN SWEEP: Lateral Channel (P_xy, D_xy)")
    print("=" * 60)

    start = [0.0, 0.0, 50.0]
    target = [20.0, 0.0, 50.0]
    n_steps = int(15.0 / CTRL_DT)

    P_xy_vals = [3.0, 5.0, 7.0, 10.0]
    D_xy_vals = [2.5, 4.0, 6.0, 8.0, 10.0]

    # Use the best Z gains from altitude sweep
    best_pz, best_dz = 10.0, 10.0

    results = []

    for pxy in P_xy_vals:
        for dxy in D_xy_vals:
            env, ctrl = make_env(start, gui=False)
            env.reset()

            # Override gains
            ctrl.P_COEFF_FOR = np.array([pxy, pxy, best_pz])
            ctrl.D_COEFF_FOR = np.array([dxy, dxy, best_dz])
            ctrl.I_COEFF_FOR = np.array([0.3, 0.3, 0.5])

            stab_steps = int(2.0 / CTRL_DT)
            run_steps(env, ctrl, lambda t: start, stab_steps)
            ctrl.reset()

            data = run_steps(env, ctrl, lambda t: target, n_steps)
            env.close()

            m = compute_metrics(data["t"], data["pos"][:, 0], start[0], target[0])
            z_dev = np.max(np.abs(data["pos"][:, 2] - start[2]))

            status = "OK" if (m["settling_time"] < 6.0 and m["overshoot_pct"] < 15.0
                              and z_dev < 2.0) else "  "
            print(f"  P_xy={pxy:5.1f}  D_xy={dxy:5.1f}  |  "
                  f"rise={m['rise_time']:5.2f}s  OS={m['overshoot_pct']:5.1f}%  "
                  f"settle={m['settling_time']:5.2f}s  sse={m['ss_error']:5.3f}m  "
                  f"z_dev={z_dev:5.2f}m  {status}")

            results.append({
                "P_xy": pxy, "D_xy": dxy,
                "z_deviation": z_dev,
                **m,
            })

    # Rank by combined score
    print("\n  Best candidates:")
    for r in results:
        rt = r["rise_time"] if not np.isnan(r["rise_time"]) else 20.0
        st = r["settling_time"] if not np.isnan(r["settling_time"]) else 20.0
        os_norm = r["overshoot_pct"] / 100.0
        zd = r["z_deviation"]
        r["score"] = 0.25 * rt + 0.35 * st + 0.25 * os_norm + 0.15 * zd

    results.sort(key=lambda r: r["score"])
    for r in results[:5]:
        print(f"    P_xy={r['P_xy']:5.1f}  D_xy={r['D_xy']:5.1f}  score={r['score']:.3f}  "
              f"(rise={r['rise_time']:.2f}s, OS={r['overshoot_pct']:.1f}%, "
              f"settle={r['settling_time']:.2f}s, z_dev={r['z_deviation']:.2f}m)")

    # Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lateral Gain Sweep — P_xy vs D_xy", fontsize=13, fontweight="bold")

    for ax_idx, metric_name, title, vmax in [
        (0, "rise_time", "Rise Time [s]", 5.0),
        (1, "overshoot_pct", "Overshoot [%]", 30.0),
        (2, "settling_time", "Settling Time [s]", 15.0),
    ]:
        grid = np.full((len(D_xy_vals), len(P_xy_vals)), np.nan)
        for r in results:
            pi = P_xy_vals.index(r["P_xy"])
            di = D_xy_vals.index(r["D_xy"])
            grid[di, pi] = r[metric_name]

        ax = axes[ax_idx]
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn_r",
                        vmin=0, vmax=vmax)
        ax.set_xticks(range(len(P_xy_vals)))
        ax.set_xticklabels([f"{v:.0f}" for v in P_xy_vals])
        ax.set_yticks(range(len(D_xy_vals)))
        ax.set_yticklabels([f"{v:.1f}" for v in D_xy_vals])
        ax.set_xlabel("P_xy")
        ax.set_ylabel("D_xy")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        for di in range(len(D_xy_vals)):
            for pi in range(len(P_xy_vals)):
                val = grid[di, pi]
                if not np.isnan(val):
                    ax.text(pi, di, f"{val:.1f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "sweep_lateral.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PID Tuning Test Bench")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--test", type=str, default=None,
                        choices=["altitude", "lateral", "combined", "tracking"],
                        help="Run a single test (default: all)")
    parser.add_argument("--sweep", type=str, nargs="?", const="altitude",
                        choices=["altitude", "lateral", "both"],
                        help="Run gain sweep (altitude, lateral, or both)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Surveillance Drone PID Tuning Test Bench")
    print(f"  Physics: PYB_GND_DRAG_DW | pyb={PYB_FREQ}Hz ctrl={CTRL_FREQ}Hz")
    print("=" * 60)

    # Print current gains
    ctrl = SurveillancePIDControl(drone_model=DroneModel.SURVEILLANCE)
    print(f"\n  Current Position Gains:")
    print(f"    P = {ctrl.P_COEFF_FOR}")
    print(f"    I = {ctrl.I_COEFF_FOR}")
    print(f"    D = {ctrl.D_COEFF_FOR}")
    print(f"  Current Attitude Gains:")
    print(f"    P = {ctrl.P_COEFF_TOR}")
    print(f"    I = {ctrl.I_COEFF_TOR}")
    print(f"    D = {ctrl.D_COEFF_TOR}")

    all_metrics = {}

    tests = {
        "altitude": test_altitude_step,
        "lateral": test_lateral_step,
        "combined": test_combined_step,
        "tracking": test_waypoint_tracking,
    }

    if args.sweep:
        if args.sweep in ("altitude", "both"):
            sweep_altitude_gains(gui=args.gui)
        if args.sweep in ("lateral", "both"):
            sweep_lateral_gains(gui=args.gui)
        if args.show:
            plt.show()
        return

    tests_to_run = [args.test] if args.test else list(tests.keys())

    for name in tests_to_run:
        test_fn = tests[name]
        t0 = time.time()

        if name == "tracking":
            data, metrics, target_positions = test_fn(gui=args.gui)
            plot_tracking(data, metrics, target_positions,
                          save_path=os.path.join(RESULTS_DIR, f"test_{name}.png"))
        else:
            data, metrics, start, target = test_fn(gui=args.gui)
            plot_step_response(data, metrics, start, target, name.title(),
                               save_path=os.path.join(RESULTS_DIR, f"test_{name}.png"))

        all_metrics[name] = metrics
        print(f"  (completed in {time.time() - t0:.1f}s)")

    if len(all_metrics) > 1:
        print_summary_table(all_metrics)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
