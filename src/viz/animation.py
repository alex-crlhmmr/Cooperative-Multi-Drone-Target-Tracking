"""3D trajectory animation with interactive playback controls."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button


def animate_tracking(
    drone_trajectories: list[np.ndarray],
    target_trajectory: np.ndarray,
    measurements: list[list] | None = None,
    dt: float = 1.0,
    interval_ms: int = 50,
    save_path: str | None = None,
    title: str = "Tracking Animation",
    trail_len: int = 80,
    plot_box: float | None = None,
):
    """Animate 3D drone tracking with trails, bearing lines, and replay controls.

    Controls:
        - Play/Pause button
        - Scrub slider to jump to any timestep
        - Speed buttons (0.5x, 1x, 2x, 4x)
        - Close the window to exit

    Shows live x,y,z coordinates for all drones and target in a side panel.
    """
    num_trackers = len(drone_trajectories)
    T = min(len(target_trajectory), min(len(t) for t in drone_trajectories))
    colors = ['#00B4D8', '#FF6B35', '#2EC4B6', '#9B5DE5', '#F4A261', '#06D6A0', '#EF476F', '#118AB2']
    target_color = '#E63946'

    fig = plt.figure(figsize=(18, 11))

    # 3D axes on the left, coordinate panel on the right
    ax = fig.add_axes([0.02, 0.20, 0.70, 0.75], projection='3d')
    ax_info = fig.add_axes([0.74, 0.20, 0.25, 0.75])
    ax_info.axis('off')

    # Compute bounds from all data
    all_pts = np.vstack([target_trajectory[:T]] + [t[:T] for t in drone_trajectories])
    mid = all_pts.mean(axis=0)
    if plot_box is not None:
        half = plot_box
    else:
        spans = all_pts.max(axis=0) - all_pts.min(axis=0)
        half = spans.max() / 2 * 1.1

    # Plot elements
    drone_dots = []
    drone_trails = []
    for i in range(num_trackers):
        dot, = ax.plot([], [], [], 'o', color=colors[i % len(colors)],
                       markersize=10, label=f'Tracker {i}', zorder=5)
        trail, = ax.plot([], [], [], color=colors[i % len(colors)],
                         linewidth=1.8, alpha=0.5)
        drone_dots.append(dot)
        drone_trails.append(trail)

    target_dot, = ax.plot([], [], [], 'o', color=target_color,
                          markersize=12, label='Target', zorder=5)
    target_trail, = ax.plot([], [], [], color=target_color,
                            linewidth=2.5, alpha=0.6, linestyle='--')

    bearing_lines = []
    for i in range(num_trackers):
        line, = ax.plot([], [], [], color=colors[i % len(colors)],
                        linewidth=0.8, alpha=0.35, linestyle=':')
        bearing_lines.append(line)

    # Gimbal visualization: green = true bearing, red = noisy measurement
    # Uses line + triangle arrowhead marker at the tip
    arrow_len_frac = 0.3  # arrow length as fraction of drone-target distance
    gimbal_true_lines = []   # green: perfect bearing
    gimbal_meas_lines = []   # red: noisy measurement
    for i in range(num_trackers):
        true_line, = ax.plot([], [], [], color='#2ECC40', linewidth=2.0, alpha=0.85)
        meas_line, = ax.plot([], [], [], color='#FF4136', linewidth=2.0, alpha=0.85)
        gimbal_true_lines.append(true_line)
        gimbal_meas_lines.append(meas_line)

    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                          fontsize=13, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(max(0, mid[2] - half), mid[2] + half)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    # Legend entries for gimbal arrows
    ax.plot([], [], [], color='#2ECC40', linewidth=2.5, label='True bearing')
    ax.plot([], [], [], color='#FF4136', linewidth=2.5, label='Measured bearing')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.7)

    # --- Coordinate panel (right side) ---
    # Pre-create text objects for each drone + target
    coord_texts = []
    y_start = 0.95
    line_h = 0.045
    header = ax_info.text(0.05, y_start, 'Live Coordinates',
                          transform=ax_info.transAxes, fontsize=13,
                          fontweight='bold', va='top')
    y_start -= 0.06

    # Target first
    tgt_text = ax_info.text(0.05, y_start, '', transform=ax_info.transAxes,
                            fontsize=10, fontfamily='monospace', va='top',
                            color=target_color, fontweight='bold')
    coord_texts.append(('target', tgt_text))
    y_start -= line_h * 1.3

    # Separator
    ax_info.plot([0.05, 0.95], [y_start + 0.01, y_start + 0.01],
                transform=ax_info.transAxes, color='#CCCCCC', linewidth=0.8, clip_on=False)
    y_start -= 0.02

    # Trackers
    for i in range(num_trackers):
        txt = ax_info.text(0.05, y_start, '', transform=ax_info.transAxes,
                           fontsize=10, fontfamily='monospace', va='top',
                           color=colors[i % len(colors)])
        coord_texts.append((f'tracker_{i}', txt))
        y_start -= line_h

    # Detection info below trackers
    y_start -= 0.02
    ax_info.plot([0.05, 0.95], [y_start + 0.01, y_start + 0.01],
                transform=ax_info.transAxes, color='#CCCCCC', linewidth=0.8, clip_on=False)
    y_start -= 0.02
    detect_text = ax_info.text(0.05, y_start, '', transform=ax_info.transAxes,
                               fontsize=11, fontfamily='monospace', va='top',
                               fontweight='bold')

    # --- Controls ---
    state = {"playing": True, "speed": 1.0, "frame": 0}

    # Slider
    ax_slider = fig.add_axes([0.08, 0.10, 0.84, 0.03])
    slider = Slider(ax_slider, 'Time', 0, T - 1, valinit=0, valstep=1, color='#4A90D9')

    # Buttons
    btn_h = 0.045
    btn_y = 0.02
    ax_play = fig.add_axes([0.08, btn_y, 0.10, btn_h])
    btn_play = Button(ax_play, 'Pause', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_restart = fig.add_axes([0.20, btn_y, 0.10, btn_h])
    btn_restart = Button(ax_restart, 'Restart', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_slow = fig.add_axes([0.42, btn_y, 0.08, btn_h])
    btn_slow = Button(ax_slow, '0.5x', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_normal = fig.add_axes([0.51, btn_y, 0.08, btn_h])
    btn_normal = Button(ax_normal, '1x', color='#C8D8E8', hovercolor='#D0D0D0')

    ax_fast = fig.add_axes([0.60, btn_y, 0.08, btn_h])
    btn_fast = Button(ax_fast, '2x', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_vfast = fig.add_axes([0.69, btn_y, 0.08, btn_h])
    btn_vfast = Button(ax_vfast, '4x', color='#E8E8E8', hovercolor='#D0D0D0')

    def set_frame(frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, T - 1))
        state["frame"] = frame_idx
        _render_frame(frame_idx)
        slider.set_val(frame_idx)

    def _render_frame(t):
        start = max(0, t - trail_len)

        for i in range(num_trackers):
            pos = drone_trajectories[i][t]
            drone_dots[i].set_data_3d([pos[0]], [pos[1]], [pos[2]])
            tr = drone_trajectories[i][start:t + 1]
            drone_trails[i].set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

            tgt = target_trajectory[t]
            bearing_lines[i].set_data_3d(
                [pos[0], tgt[0]], [pos[1], tgt[1]], [pos[2], tgt[2]])

        tgt = target_trajectory[t]
        target_dot.set_data_3d([tgt[0]], [tgt[1]], [tgt[2]])
        tr = target_trajectory[start:t + 1]
        target_trail.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

        time_text.set_text(f't = {t * dt:.2f}s  (step {t}/{T})')

        # Update coordinate panel
        coord_texts[0][1].set_text(
            f'Target   x:{tgt[0]:8.1f}  y:{tgt[1]:8.1f}  z:{tgt[2]:8.1f}')

        for i in range(num_trackers):
            pos = drone_trajectories[i][t]
            dist = np.linalg.norm(pos - tgt)
            coord_texts[i + 1][1].set_text(
                f'T{i}  x:{pos[0]:7.1f}  y:{pos[1]:7.1f}  z:{pos[2]:7.1f}  d:{dist:.0f}m')

        # Gimbal arrows + detection info
        if measurements and t < len(measurements):
            n_det = 0
            for i in range(num_trackers):
                pos = drone_trajectories[i][t]
                d = tgt - pos
                dist = np.linalg.norm(d)
                arrow_len = dist * arrow_len_frac

                if dist > 1e-3:
                    # Green arrow: true bearing (always drawn)
                    true_dir = d / dist
                    end_true = pos + true_dir * arrow_len
                    gimbal_true_lines[i].set_data_3d(
                        [pos[0], end_true[0]], [pos[1], end_true[1]], [pos[2], end_true[2]])
                else:
                    gimbal_true_lines[i].set_data_3d([], [], [])

                m = measurements[t][i]
                if m is not None:
                    n_det += 1
                    # Red arrow: measured (noisy) bearing
                    az, el = m
                    meas_dir = np.array([
                        np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el),
                    ])
                    end_meas = pos + meas_dir * arrow_len
                    gimbal_meas_lines[i].set_data_3d(
                        [pos[0], end_meas[0]], [pos[1], end_meas[1]], [pos[2], end_meas[2]])
                else:
                    gimbal_meas_lines[i].set_data_3d([], [], [])

            detect_text.set_text(f'Detections: {n_det}/{num_trackers}')
        else:
            for i in range(num_trackers):
                gimbal_true_lines[i].set_data_3d([], [], [])
                gimbal_meas_lines[i].set_data_3d([], [], [])
            detect_text.set_text('')

        fig.canvas.draw_idle()

    def on_slider(val):
        state["frame"] = int(val)
        _render_frame(state["frame"])

    def on_play(event):
        state["playing"] = not state["playing"]
        btn_play.label.set_text('Play' if not state["playing"] else 'Pause')
        fig.canvas.draw_idle()

    def on_restart(event):
        state["frame"] = 0
        state["playing"] = True
        btn_play.label.set_text('Pause')
        slider.set_val(0)

    def on_speed(speed):
        def handler(event):
            state["speed"] = speed
            for ax_btn, s in [(ax_slow, 0.5), (ax_normal, 1.0), (ax_fast, 2.0), (ax_vfast, 4.0)]:
                ax_btn.set_facecolor('#C8D8E8' if s == speed else '#E8E8E8')
            fig.canvas.draw_idle()
        return handler

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_restart.on_clicked(on_restart)
    btn_slow.on_clicked(on_speed(0.5))
    btn_normal.on_clicked(on_speed(1.0))
    btn_fast.on_clicked(on_speed(2.0))
    btn_vfast.on_clicked(on_speed(4.0))

    def update(frame_arg):
        if not state["playing"]:
            return []
        step = max(1, int(state["speed"]))
        state["frame"] = (state["frame"] + step) % T
        slider.set_val(state["frame"])
        _render_frame(state["frame"])
        return []

    anim = FuncAnimation(fig, update, frames=None, interval=interval_ms,
                         blit=False, cache_frame_data=False)

    if save_path:
        def save_update(frame):
            _render_frame(frame)
            return []
        save_anim = FuncAnimation(fig, save_update, frames=T, interval=interval_ms, blit=False)
        save_anim.save(save_path, writer='ffmpeg', fps=int(1000 / interval_ms), dpi=100)
        print(f"Animation saved to {save_path}")

    plt.show()
    return anim


def animate_filter_tracking(
    drone_positions: np.ndarray,
    target_true_states: np.ndarray,
    estimates: np.ndarray,
    filter_names: list[str],
    measurements: list | None = None,
    dt: float = 1.0,
    interval_ms: int = 50,
    title: str = "Filter Tracking Replay",
    trail_len: int = 80,
    plot_box: float | None = None,
):
    """Animate 3D tracking with observers, true target, and filter estimates.

    Shows:
        - Observer drones (all same color — white/gray dots with thin trails)
        - True target (red dot + dashed trail)
        - Estimated target per filter (colored dots + trails: EKF=blue, UKF=orange, PF=green)
        - Bearing lines: green=true direction, red=noisy measurement (from demo)
        - Side panel: positions, errors, detections

    Args:
        drone_positions: (T, num_drones, 3) observer positions per step
        target_true_states: (T, 6) ground truth [px,py,pz,vx,vy,vz]
        estimates: (n_filters, T, 6) filter estimates
        filter_names: list of filter names (e.g. ["EKF", "UKF", "PF"])
        measurements: list of measurement lists per step
        dt: timestep
    """
    T = target_true_states.shape[0]
    num_drones = drone_positions.shape[1]
    n_filters = len(filter_names)
    target_traj = target_true_states[:, :3]

    # Colors
    observer_color = '#AAAAAA'
    target_color = '#E63946'
    filter_colors = {
        "EKF": "#2196F3",
        "UKF": "#FF9800",
        "PF": "#4CAF50",
    }
    filter_styles = {
        "EKF": "-",
        "UKF": "--",
        "PF": "-.",
    }

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_axes([0.02, 0.18, 0.62, 0.77], projection='3d')
    ax_info = fig.add_axes([0.66, 0.18, 0.33, 0.77])
    ax_info.axis('off')

    # Compute bounds
    all_pts = [target_traj]
    for fi in range(n_filters):
        all_pts.append(estimates[fi, :, :3])
    for di in range(num_drones):
        all_pts.append(drone_positions[:, di, :])
    all_pts = np.vstack(all_pts)
    mid = all_pts.mean(axis=0)
    if plot_box is not None:
        half = plot_box
    else:
        half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2

    # --- Plot elements ---

    # Observer drones (all same neutral color)
    obs_dots = []
    obs_trails = []
    for i in range(num_drones):
        dot, = ax.plot([], [], [], 'o', color=observer_color,
                       markersize=7, zorder=4, alpha=0.8)
        trail, = ax.plot([], [], [], color=observer_color,
                         linewidth=1.0, alpha=0.3)
        obs_dots.append(dot)
        obs_trails.append(trail)
    # Single legend entry for observers
    ax.plot([], [], [], 'o', color=observer_color, markersize=7, label='Observers')

    # True target
    target_dot, = ax.plot([], [], [], 'o', color=target_color,
                          markersize=14, label='True Target', zorder=6)
    target_trail_line, = ax.plot([], [], [], color=target_color,
                                  linewidth=2.5, alpha=0.6, linestyle='--')

    # Filter estimates
    est_dots = []
    est_trails = []
    for fi, fname in enumerate(filter_names):
        c = filter_colors.get(fname, '#999999')
        dot, = ax.plot([], [], [], 's', color=c, markersize=10,
                       label=f'{fname} Estimate', zorder=5)
        trail, = ax.plot([], [], [], color=c, linewidth=1.5, alpha=0.6,
                         linestyle=filter_styles.get(fname, '-'))
        est_dots.append(dot)
        est_trails.append(trail)

    # Bearing lines (green=true, red=measured)
    arrow_frac = 0.3
    gimbal_true = []
    gimbal_meas = []
    for i in range(num_drones):
        tl, = ax.plot([], [], [], color='#2ECC40', linewidth=1.8, alpha=0.7)
        ml, = ax.plot([], [], [], color='#FF4136', linewidth=1.8, alpha=0.7)
        gimbal_true.append(tl)
        gimbal_meas.append(ml)
    ax.plot([], [], [], color='#2ECC40', linewidth=2, label='True bearing')
    ax.plot([], [], [], color='#FF4136', linewidth=2, label='Measured bearing')

    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                          fontsize=13, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(max(0, mid[2] - half), mid[2] + half)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.7, ncol=2)

    # --- Info panel ---
    y = 0.97
    lh = 0.032

    ax_info.text(0.02, y, 'FILTER TRACKING', transform=ax_info.transAxes,
                 fontsize=14, fontweight='bold', va='top')
    y -= 0.05

    # True target text
    true_text = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                             fontsize=10, fontfamily='monospace', va='top',
                             color=target_color, fontweight='bold')
    y -= lh * 1.5

    # Filter estimate texts
    est_texts = []
    for fi, fname in enumerate(filter_names):
        c = filter_colors.get(fname, '#999999')
        txt = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                           fontsize=10, fontfamily='monospace', va='top',
                           color=c, fontweight='bold')
        est_texts.append(txt)
        y -= lh

    y -= 0.01
    ax_info.plot([0.02, 0.98], [y + 0.005, y + 0.005],
                 transform=ax_info.transAxes, color='#CCCCCC', linewidth=0.8, clip_on=False)
    y -= 0.02

    # Observer texts
    obs_texts = []
    for i in range(num_drones):
        txt = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                           fontsize=9, fontfamily='monospace', va='top',
                           color=observer_color)
        obs_texts.append(txt)
        y -= lh * 0.85

    y -= 0.01
    ax_info.plot([0.02, 0.98], [y + 0.005, y + 0.005],
                 transform=ax_info.transAxes, color='#CCCCCC', linewidth=0.8, clip_on=False)
    y -= 0.02

    detect_text = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                               fontsize=11, fontfamily='monospace', va='top',
                               fontweight='bold')

    # --- Controls ---
    state = {"playing": True, "speed": 1.0, "frame": 0}

    ax_slider = fig.add_axes([0.08, 0.08, 0.84, 0.03])
    slider = Slider(ax_slider, 'Time', 0, T - 1, valinit=0, valstep=1, color='#4A90D9')

    btn_h = 0.045
    btn_y = 0.015
    ax_play = fig.add_axes([0.08, btn_y, 0.10, btn_h])
    btn_play = Button(ax_play, 'Pause', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_restart = fig.add_axes([0.20, btn_y, 0.10, btn_h])
    btn_restart = Button(ax_restart, 'Restart', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_slow = fig.add_axes([0.42, btn_y, 0.08, btn_h])
    btn_slow = Button(ax_slow, '0.5x', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_normal = fig.add_axes([0.51, btn_y, 0.08, btn_h])
    btn_normal = Button(ax_normal, '1x', color='#C8D8E8', hovercolor='#D0D0D0')

    ax_fast = fig.add_axes([0.60, btn_y, 0.08, btn_h])
    btn_fast = Button(ax_fast, '2x', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_vfast = fig.add_axes([0.69, btn_y, 0.08, btn_h])
    btn_vfast = Button(ax_vfast, '4x', color='#E8E8E8', hovercolor='#D0D0D0')

    def _render_frame(t):
        start = max(0, t - trail_len)
        tgt = target_traj[t]

        # Observers
        for i in range(num_drones):
            pos = drone_positions[t, i]
            obs_dots[i].set_data_3d([pos[0]], [pos[1]], [pos[2]])
            tr = drone_positions[start:t+1, i]
            obs_trails[i].set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

        # True target
        target_dot.set_data_3d([tgt[0]], [tgt[1]], [tgt[2]])
        tr = target_traj[start:t+1]
        target_trail_line.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

        # Filter estimates
        for fi in range(n_filters):
            est = estimates[fi, t, :3]
            est_dots[fi].set_data_3d([est[0]], [est[1]], [est[2]])
            tr = estimates[fi, start:t+1, :3]
            est_trails[fi].set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

        # Bearing lines
        if measurements and t < len(measurements):
            n_det = 0
            for i in range(num_drones):
                pos = drone_positions[t, i]
                d = tgt - pos
                dist = np.linalg.norm(d)
                arrow_len = dist * arrow_frac

                if dist > 1e-3:
                    true_dir = d / dist
                    end_t = pos + true_dir * arrow_len
                    gimbal_true[i].set_data_3d(
                        [pos[0], end_t[0]], [pos[1], end_t[1]], [pos[2], end_t[2]])
                else:
                    gimbal_true[i].set_data_3d([], [], [])

                m = measurements[t][i] if i < len(measurements[t]) else None
                if m is not None:
                    n_det += 1
                    az, el = m
                    meas_dir = np.array([
                        np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el),
                    ])
                    end_m = pos + meas_dir * arrow_len
                    gimbal_meas[i].set_data_3d(
                        [pos[0], end_m[0]], [pos[1], end_m[1]], [pos[2], end_m[2]])
                else:
                    gimbal_meas[i].set_data_3d([], [], [])

            detect_text.set_text(f'Detections: {n_det}/{num_drones}')
        else:
            for i in range(num_drones):
                gimbal_true[i].set_data_3d([], [], [])
                gimbal_meas[i].set_data_3d([], [], [])
            detect_text.set_text('')

        # Time text
        time_text.set_text(f't = {t * dt:.2f}s  (step {t}/{T})')

        # Info panel
        true_text.set_text(
            f'TRUE   x:{tgt[0]:8.1f}  y:{tgt[1]:8.1f}  z:{tgt[2]:8.1f}')

        for fi, fname in enumerate(filter_names):
            est = estimates[fi, t, :3]
            err = np.linalg.norm(tgt - est)
            est_texts[fi].set_text(
                f'{fname:<5}  x:{est[0]:8.1f}  y:{est[1]:8.1f}  z:{est[2]:8.1f}  err:{err:6.1f}m')

        for i in range(num_drones):
            pos = drone_positions[t, i]
            dist = np.linalg.norm(pos - tgt)
            obs_texts[i].set_text(
                f'Obs{i}  x:{pos[0]:7.1f} y:{pos[1]:7.1f} z:{pos[2]:7.1f} d:{dist:.0f}m')

        fig.canvas.draw_idle()

    def on_slider(val):
        state["frame"] = int(val)
        _render_frame(state["frame"])

    def on_play(event):
        state["playing"] = not state["playing"]
        btn_play.label.set_text('Play' if not state["playing"] else 'Pause')
        fig.canvas.draw_idle()

    def on_restart(event):
        state["frame"] = 0
        state["playing"] = True
        btn_play.label.set_text('Pause')
        slider.set_val(0)

    def on_speed(speed):
        def handler(event):
            state["speed"] = speed
            for ax_btn, s in [(ax_slow, 0.5), (ax_normal, 1.0), (ax_fast, 2.0), (ax_vfast, 4.0)]:
                ax_btn.set_facecolor('#C8D8E8' if s == speed else '#E8E8E8')
            fig.canvas.draw_idle()
        return handler

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_restart.on_clicked(on_restart)
    btn_slow.on_clicked(on_speed(0.5))
    btn_normal.on_clicked(on_speed(1.0))
    btn_fast.on_clicked(on_speed(2.0))
    btn_vfast.on_clicked(on_speed(4.0))

    def update(frame_arg):
        if not state["playing"]:
            return []
        step = max(1, int(state["speed"]))
        state["frame"] = (state["frame"] + step) % T
        slider.set_val(state["frame"])
        _render_frame(state["frame"])
        return []

    anim = FuncAnimation(fig, update, frames=None, interval=interval_ms,
                         blit=False, cache_frame_data=False)

    plt.show()
    return anim


def animate_consensus_tracking(
    drone_positions: np.ndarray,
    target_true_states: np.ndarray,
    centralized_est: np.ndarray,
    consensus_est: np.ndarray,
    local_estimates: np.ndarray,
    adjacency: np.ndarray,
    disagreements: np.ndarray,
    measurements: list | None = None,
    active_edges: list | None = None,
    dt: float = 1.0,
    interval_ms: int = 50,
    title: str = "Consensus EKF Replay",
    trail_len: int = 80,
    plot_box: float | None = None,
    topology_name: str = "full",
):
    """Animate 3D consensus tracking with per-drone local estimates and comm links.

    Shows:
        - Observer drones (colored dots — each drone has its own color)
        - Communication links between drones (cyan lines between connected pairs)
        - Per-drone local target estimates (small colored diamonds matching drone color)
        - Consensus average estimate (blue square)
        - Centralized EKF estimate (red square — the god-node baseline)
        - True target (black dot + dashed trail)
        - Bearing lines: green=true direction, red=noisy measurement
        - Side panel: positions, errors, disagreement, detections

    Args:
        drone_positions: (T, num_drones, 3) observer positions per step
        target_true_states: (T, 6) ground truth
        centralized_est: (T, 6) centralized EKF estimates
        consensus_est: (T, 6) consensus average estimates
        local_estimates: (num_drones, T, 6) per-drone local estimates
        adjacency: (num_drones, num_drones) communication adjacency matrix
        disagreements: (T,) consensus disagreement per step
        measurements: list of measurement lists per step
        dt: timestep
        topology_name: name of topology for display
    """
    T = target_true_states.shape[0]
    num_drones = drone_positions.shape[1]
    target_traj = target_true_states[:, :3]

    # Per-drone colors
    drone_colors = ['#E91E63', '#9C27B0', '#3F51B5', '#009688', '#FF5722',
                    '#795548', '#607D8B', '#FF9800']
    target_color = '#000000'
    central_color = '#F44336'
    consensus_color = '#2196F3'
    comm_color = '#00BCD4'

    fig = plt.figure(figsize=(22, 13))
    ax = fig.add_axes([0.02, 0.18, 0.58, 0.77], projection='3d')
    ax_info = fig.add_axes([0.62, 0.18, 0.37, 0.77])
    ax_info.axis('off')

    # Compute bounds
    all_pts = [target_traj, centralized_est[:, :3], consensus_est[:, :3]]
    for di in range(num_drones):
        all_pts.append(drone_positions[:, di, :])
        all_pts.append(local_estimates[di, :, :3])
    all_pts = np.vstack(all_pts)
    mid = all_pts.mean(axis=0)
    if plot_box is not None:
        half = plot_box
    else:
        half = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2

    # --- Plot elements ---

    # Observer drones (each with unique color)
    obs_dots = []
    obs_trails = []
    for i in range(num_drones):
        c = drone_colors[i % len(drone_colors)]
        dot, = ax.plot([], [], [], 'o', color=c, markersize=9, zorder=4, alpha=0.9)
        trail, = ax.plot([], [], [], color=c, linewidth=1.0, alpha=0.25)
        obs_dots.append(dot)
        obs_trails.append(trail)

    # Communication links (lines between ALL possible drone pairs in topology)
    # Active links shown solid cyan; dropped links shown faded dashed red
    comm_lines = []
    comm_pairs = []
    dropped_color = '#FF5252'
    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            if adjacency[i, j]:
                line, = ax.plot([], [], [], color=comm_color, linewidth=1.8,
                                alpha=0.5, linestyle='-', zorder=2)
                comm_lines.append(line)
                comm_pairs.append((i, j))
    if comm_lines:
        ax.plot([], [], [], color=comm_color, linewidth=2, label='Comm active')
        ax.plot([], [], [], color=dropped_color, linewidth=1.2, linestyle='--',
                alpha=0.4, label='Comm dropped')

    # Per-drone local estimates (small diamonds)
    local_dots = []
    for i in range(num_drones):
        c = drone_colors[i % len(drone_colors)]
        dot, = ax.plot([], [], [], 'D', color=c, markersize=5, alpha=0.6, zorder=3)
        local_dots.append(dot)
    ax.plot([], [], [], 'D', color='#888888', markersize=5, label='Local estimates')

    # True target
    target_dot, = ax.plot([], [], [], 'o', color=target_color,
                          markersize=14, label='True Target', zorder=6)
    target_trail_line, = ax.plot([], [], [], color=target_color,
                                  linewidth=2.5, alpha=0.5, linestyle='--')

    # Centralized estimate
    central_dot, = ax.plot([], [], [], 's', color=central_color, markersize=11,
                           label='Centralized EKF', zorder=5)
    central_trail, = ax.plot([], [], [], color=central_color, linewidth=1.5,
                              alpha=0.5, linestyle='-')

    # Consensus average estimate
    cons_dot, = ax.plot([], [], [], 's', color=consensus_color, markersize=11,
                        label='Consensus Avg', zorder=5)
    cons_trail, = ax.plot([], [], [], color=consensus_color, linewidth=1.5,
                           alpha=0.5, linestyle='--')

    # Bearing lines
    arrow_frac = 0.3
    gimbal_true = []
    gimbal_meas = []
    for i in range(num_drones):
        tl, = ax.plot([], [], [], color='#2ECC40', linewidth=1.5, alpha=0.5)
        ml, = ax.plot([], [], [], color='#FF4136', linewidth=1.5, alpha=0.5)
        gimbal_true.append(tl)
        gimbal_meas.append(ml)
    ax.plot([], [], [], color='#2ECC40', linewidth=2, label='True bearing')
    ax.plot([], [], [], color='#FF4136', linewidth=2, label='Measured bearing')

    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                          fontsize=13, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(max(0, mid[2] - half), mid[2] + half)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7, framealpha=0.7, ncol=3)

    # --- Info panel ---
    y = 0.97
    lh = 0.028

    ax_info.text(0.02, y, f'CONSENSUS TRACKING  [{topology_name.upper()}]',
                 transform=ax_info.transAxes, fontsize=14, fontweight='bold', va='top')
    y -= 0.05

    true_text = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                             fontsize=10, fontfamily='monospace', va='top',
                             color=target_color, fontweight='bold')
    y -= lh * 1.3

    central_text = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                                fontsize=10, fontfamily='monospace', va='top',
                                color=central_color, fontweight='bold')
    y -= lh * 1.3

    cons_text = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                             fontsize=10, fontfamily='monospace', va='top',
                             color=consensus_color, fontweight='bold')
    y -= lh * 1.5

    ax_info.plot([0.02, 0.98], [y + 0.005, y + 0.005],
                 transform=ax_info.transAxes, color='#CCCCCC', linewidth=0.8, clip_on=False)
    y -= 0.015

    ax_info.text(0.02, y, 'LOCAL ESTIMATES (per drone)', transform=ax_info.transAxes,
                 fontsize=10, fontweight='bold', va='top', color='#555555')
    y -= lh * 1.2

    local_texts = []
    for i in range(num_drones):
        c = drone_colors[i % len(drone_colors)]
        txt = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                           fontsize=9, fontfamily='monospace', va='top', color=c)
        local_texts.append(txt)
        y -= lh

    y -= 0.008
    ax_info.plot([0.02, 0.98], [y + 0.005, y + 0.005],
                 transform=ax_info.transAxes, color='#CCCCCC', linewidth=0.8, clip_on=False)
    y -= 0.015

    ax_info.text(0.02, y, 'CONSENSUS METRICS', transform=ax_info.transAxes,
                 fontsize=10, fontweight='bold', va='top', color='#555555')
    y -= lh * 1.2

    disagree_text = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                                  fontsize=10, fontfamily='monospace', va='top',
                                  color='#333333', fontweight='bold')
    y -= lh * 1.2

    central_err_text = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                                     fontsize=10, fontfamily='monospace', va='top',
                                     color=central_color)
    y -= lh
    cons_err_text = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                                  fontsize=10, fontfamily='monospace', va='top',
                                  color=consensus_color)
    y -= lh * 1.5

    ax_info.plot([0.02, 0.98], [y + 0.005, y + 0.005],
                 transform=ax_info.transAxes, color='#CCCCCC', linewidth=0.8, clip_on=False)
    y -= 0.015

    ax_info.text(0.02, y, 'OBSERVERS', transform=ax_info.transAxes,
                 fontsize=10, fontweight='bold', va='top', color='#555555')
    y -= lh * 1.2

    obs_texts = []
    for i in range(num_drones):
        c = drone_colors[i % len(drone_colors)]
        txt = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                           fontsize=8, fontfamily='monospace', va='top', color=c)
        obs_texts.append(txt)
        y -= lh * 0.9

    y -= 0.008
    ax_info.plot([0.02, 0.98], [y + 0.005, y + 0.005],
                 transform=ax_info.transAxes, color='#CCCCCC', linewidth=0.8, clip_on=False)
    y -= 0.015

    detect_text = ax_info.text(0.02, y, '', transform=ax_info.transAxes,
                               fontsize=11, fontfamily='monospace', va='top',
                               fontweight='bold')

    # --- Controls ---
    state = {"playing": True, "speed": 1.0, "frame": 0}

    ax_slider = fig.add_axes([0.08, 0.08, 0.84, 0.03])
    slider = Slider(ax_slider, 'Time', 0, T - 1, valinit=0, valstep=1, color='#4A90D9')

    btn_h = 0.045
    btn_y = 0.015
    ax_play = fig.add_axes([0.08, btn_y, 0.10, btn_h])
    btn_play = Button(ax_play, 'Pause', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_restart = fig.add_axes([0.20, btn_y, 0.10, btn_h])
    btn_restart = Button(ax_restart, 'Restart', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_slow = fig.add_axes([0.42, btn_y, 0.08, btn_h])
    btn_slow = Button(ax_slow, '0.5x', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_normal = fig.add_axes([0.51, btn_y, 0.08, btn_h])
    btn_normal = Button(ax_normal, '1x', color='#C8D8E8', hovercolor='#D0D0D0')

    ax_fast = fig.add_axes([0.60, btn_y, 0.08, btn_h])
    btn_fast = Button(ax_fast, '2x', color='#E8E8E8', hovercolor='#D0D0D0')

    ax_vfast = fig.add_axes([0.69, btn_y, 0.08, btn_h])
    btn_vfast = Button(ax_vfast, '4x', color='#E8E8E8', hovercolor='#D0D0D0')

    def _render_frame(t):
        start = max(0, t - trail_len)
        tgt = target_traj[t]

        for i in range(num_drones):
            pos = drone_positions[t, i]
            obs_dots[i].set_data_3d([pos[0]], [pos[1]], [pos[2]])
            tr = drone_positions[start:t+1, i]
            obs_trails[i].set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

        for li, (i, j) in enumerate(comm_pairs):
            pi = drone_positions[t, i]
            pj = drone_positions[t, j]
            comm_lines[li].set_data_3d(
                [pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]])
            # Show active vs dropped links
            if active_edges is not None and t < len(active_edges):
                edge_active = active_edges[t][i, j] > 0
                if edge_active:
                    comm_lines[li].set_color(comm_color)
                    comm_lines[li].set_linestyle('-')
                    comm_lines[li].set_alpha(0.6)
                    comm_lines[li].set_linewidth(2.0)
                else:
                    comm_lines[li].set_color(dropped_color)
                    comm_lines[li].set_linestyle('--')
                    comm_lines[li].set_alpha(0.25)
                    comm_lines[li].set_linewidth(1.0)

        for i in range(num_drones):
            le = local_estimates[i, t, :3]
            local_dots[i].set_data_3d([le[0]], [le[1]], [le[2]])

        target_dot.set_data_3d([tgt[0]], [tgt[1]], [tgt[2]])
        tr = target_traj[start:t+1]
        target_trail_line.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

        ce = centralized_est[t, :3]
        central_dot.set_data_3d([ce[0]], [ce[1]], [ce[2]])
        tr = centralized_est[start:t+1, :3]
        central_trail.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

        co = consensus_est[t, :3]
        cons_dot.set_data_3d([co[0]], [co[1]], [co[2]])
        tr = consensus_est[start:t+1, :3]
        cons_trail.set_data_3d(tr[:, 0], tr[:, 1], tr[:, 2])

        if measurements and t < len(measurements):
            n_det = 0
            for i in range(num_drones):
                pos = drone_positions[t, i]
                d = tgt - pos
                dist = np.linalg.norm(d)
                arrow_len = dist * arrow_frac

                if dist > 1e-3:
                    true_dir = d / dist
                    end_t = pos + true_dir * arrow_len
                    gimbal_true[i].set_data_3d(
                        [pos[0], end_t[0]], [pos[1], end_t[1]], [pos[2], end_t[2]])
                else:
                    gimbal_true[i].set_data_3d([], [], [])

                m = measurements[t][i] if i < len(measurements[t]) else None
                if m is not None:
                    n_det += 1
                    az, el = m
                    meas_dir = np.array([
                        np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el),
                    ])
                    end_m = pos + meas_dir * arrow_len
                    gimbal_meas[i].set_data_3d(
                        [pos[0], end_m[0]], [pos[1], end_m[1]], [pos[2], end_m[2]])
                else:
                    gimbal_meas[i].set_data_3d([], [], [])

            detect_text.set_text(f'Detections: {n_det}/{num_drones}')
        else:
            for i in range(num_drones):
                gimbal_true[i].set_data_3d([], [], [])
                gimbal_meas[i].set_data_3d([], [], [])
            detect_text.set_text('')

        time_text.set_text(f't = {t * dt:.2f}s  (step {t}/{T})')

        true_text.set_text(
            f'TRUE     x:{tgt[0]:8.1f}  y:{tgt[1]:8.1f}  z:{tgt[2]:8.1f}')

        c_err = np.linalg.norm(tgt - ce)
        central_text.set_text(
            f'CENTRAL  x:{ce[0]:8.1f}  y:{ce[1]:8.1f}  z:{ce[2]:8.1f}  err:{c_err:6.1f}m')

        co_err = np.linalg.norm(tgt - co)
        cons_text.set_text(
            f'CONSENS  x:{co[0]:8.1f}  y:{co[1]:8.1f}  z:{co[2]:8.1f}  err:{co_err:6.1f}m')

        for i in range(num_drones):
            le = local_estimates[i, t, :3]
            le_err = np.linalg.norm(tgt - le)
            local_texts[i].set_text(
                f'D{i} x:{le[0]:7.1f} y:{le[1]:7.1f} z:{le[2]:7.1f} err:{le_err:5.1f}m')

        disagree_text.set_text(f'Disagreement: {disagreements[t]:8.2f} m')
        central_err_text.set_text(f'Centralized err: {c_err:8.2f} m')
        cons_err_text.set_text(f'Consensus err:   {co_err:8.2f} m')

        for i in range(num_drones):
            pos = drone_positions[t, i]
            dist = np.linalg.norm(pos - tgt)
            obs_texts[i].set_text(
                f'Obs{i} x:{pos[0]:7.1f} y:{pos[1]:7.1f} z:{pos[2]:7.1f} d:{dist:.0f}m')

        fig.canvas.draw_idle()

    def on_slider(val):
        state["frame"] = int(val)
        _render_frame(state["frame"])

    def on_play(event):
        state["playing"] = not state["playing"]
        btn_play.label.set_text('Play' if not state["playing"] else 'Pause')
        fig.canvas.draw_idle()

    def on_restart(event):
        state["frame"] = 0
        state["playing"] = True
        btn_play.label.set_text('Pause')
        slider.set_val(0)

    def on_speed(speed):
        def handler(event):
            state["speed"] = speed
            for ax_btn, s in [(ax_slow, 0.5), (ax_normal, 1.0), (ax_fast, 2.0), (ax_vfast, 4.0)]:
                ax_btn.set_facecolor('#C8D8E8' if s == speed else '#E8E8E8')
            fig.canvas.draw_idle()
        return handler

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_restart.on_clicked(on_restart)
    btn_slow.on_clicked(on_speed(0.5))
    btn_normal.on_clicked(on_speed(1.0))
    btn_fast.on_clicked(on_speed(2.0))
    btn_vfast.on_clicked(on_speed(4.0))

    def update(frame_arg):
        if not state["playing"]:
            return []
        step = max(1, int(state["speed"]))
        state["frame"] = (state["frame"] + step) % T
        slider.set_val(state["frame"])
        _render_frame(state["frame"])
        return []

    anim = FuncAnimation(fig, update, frames=None, interval=interval_ms,
                         blit=False, cache_frame_data=False)

    plt.show()
    return anim
