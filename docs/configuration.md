# Configuration Reference

All parameters are in `config/default.yaml`. Scripts load this file automatically.

---

## Simulation

```yaml
sim:
  pyb_freq: 480         # PyBullet physics frequency (Hz)
  ctrl_freq: 48         # Control loop / filter update frequency (Hz)
  gui: true             # Enable PyBullet GUI (overridden by --no-gui)
  physics: "pyb_gnd_drag_dw"  # Physics mode
```

| Parameter | Description |
|-----------|-------------|
| `pyb_freq` | Internal physics substeps. 480 Hz = 10 substeps per control tick at 48 Hz |
| `ctrl_freq` | Determines filter dt = 1/48 ~= 0.0208s. All filters and sensors run at this rate |
| `physics` | `pyb_gnd_drag_dw` is the most realistic mode (ground effect + drag + downwash) |

---

## Arena

```yaml
arena:
  size: [500, 500, 500]   # Arena dimensions in meters (x, y, z)
  dt: 0.1                 # Legacy — actual dt comes from 1/ctrl_freq
  plot_box: 2000           # Half-size of replay/animation bounding box (meters)

drones_min_altitude: 20.0  # Trackers won't go below this (meters)
```

---

## Drones

```yaml
drones:
  num_trackers: 5
  model: "surveillance"       # "surveillance" (4kg) or "cf2x" (27g Crazyflie)
  tracker_min_radius: 100.0   # Inner spawn radius around target (m)
  tracker_max_radius: 300.0   # Outer spawn radius (m)
```

Drones spawn at random positions in a hollow sphere around the target's initial position, with altitude clamped above `drones_min_altitude`.

---

## Target

```yaml
target:
  initial_xy: [0.0, 0.0]
  min_altitude: 50.0          # Min spawn height (m)
  max_altitude: 300.0         # Max spawn height (m)
  speed: 12.0                 # Nominal speed (m/s)
  process_noise_accel: 0.6    # sigma_a for filter's Q matrix
  trajectory: "multi_segment" # Default trajectory type
```

### Trajectory types

| Type | Description |
|------|-------------|
| `straight` | Constant velocity in a random 3D direction |
| `single_turn` | Straight-turn-straight with banking and altitude ramp |
| `multi_segment` | 3-6 random segments with varied headings and speeds |
| `evasive` | Continuous rapid maneuvers — worst case for filters |

### Evasive trajectory tuning

```yaml
target:
  evasive:
    speed_var: 0.3            # Speed varies ±30% during maneuvers
    turn_rate_max: 2.0        # Max turn rate (rad/s)
    climb_rate_factor: 0.3    # Max climb/dive = factor × speed
    pitch_range_deg: 20.0     # Initial heading pitch randomization (deg)
    maneuver_density: 0.04    # Maneuvers per step (0.02=calm, 0.08=chaotic)
```

| Density | Behavior |
|---------|----------|
| 0.02 | Calm — occasional gentle turns |
| 0.04 | Moderate — regular maneuvers (default) |
| 0.08 | Chaotic — nearly continuous aggressive maneuvers |

---

## Sensor

```yaml
sensor:
  mode: "geometric"                 # "geometric" (fast) or "camera" (validation)
  fov_half_deg: 15                  # Camera half-angle FOV (deg)
  sigma_bearing_deg: 2.0            # Bearing noise std (deg)
  max_range: 300.0                  # Maximum detection range (m)
  range_ref: 100.0                  # Reference range for noise scaling
  detection_prob_model: "range_dependent"  # or "constant"
  p_detect_max: 0.99               # Peak detection probability
  p_detect_range_half: 250.0       # Range where P(detect) = 0.5
```

**Noise scaling:** `sigma(r) = sigma_bearing * (r / range_ref)`. At 200m range with the defaults, noise is 2x the baseline.

**Detection probability:** Sigmoid model — nearly 1.0 at close range, drops to 0.5 at 250m, approaches 0 beyond that.

**Sensor modes:**
- `geometric` — Pure geometric bearing computation. Fast, used for Monte Carlo.
- `camera` — Simulates a gimballed camera with FOV constraints. Slower but more realistic.

---

## Filters

```yaml
filters:
  init:
    P0_pos: 10000.0       # Initial position variance (m^2)
    P0_vel: 100.0          # Initial velocity variance (m^2/s^2)
    pf_range_min: 10.0     # Min range for PF ray initialization (m)
    pf_range_max: 500.0    # Max range for PF ray initialization (m)
  ekf: {}
  ukf:
    alpha: 0.5
    beta: 2.0
    kappa: 0.0
  pf:
    num_particles: 2000
    resample_threshold: 0.5     # N_eff / N threshold for resampling
    process_noise_factor: 5.0   # Q inflation to prevent particle degeneracy
    jitter_pos: 1.0             # Position roughening (m/step)
    jitter_vel: 0.5             # Velocity roughening (m/s/step)
```

### Initialization

All filters auto-initialize on the first timestep with >= 2 valid bearing measurements:
- **EKF/UKF/IMM:** Triangulation from bearing intersections
- **PF:** Particles spread along bearing rays at random ranges in `[pf_range_min, pf_range_max]`

### UKF sigma-point parameters

| Parameter | Effect |
|-----------|--------|
| `alpha` | Controls spread of sigma points (smaller = tighter around mean) |
| `beta` | Prior distribution info (2.0 is optimal for Gaussian) |
| `kappa` | Secondary scaling (usually 0) |

### PF tuning

| Parameter | Too low | Too high |
|-----------|---------|----------|
| `num_particles` | Poor representation, high variance | Slow |
| `process_noise_factor` | Particle degeneracy | Over-diffusion |
| `jitter_pos/vel` | Particle collapse | Excessive noise |
| `resample_threshold` | Degeneracy | Over-resampling |

---

## Consensus

```yaml
consensus:
  num_iterations: 2       # L — consensus rounds per timestep
  step_size: 0.1           # epsilon — averaging step size
  topologies: ["full", "ring", "star"]    # For MC sweeps
  dropout_probs: [0.0, 0.1, 0.2, 0.3, 0.5]  # For MC sweeps
```

| Parameter | Effect of increasing |
|-----------|---------------------|
| `num_iterations` (L) | Better consensus convergence, more computation per step |
| `step_size` (epsilon) | Faster convergence but can overshoot/oscillate if too large |
| `dropout_probs` | Simulates unreliable comms — higher = more degraded |

**Metropolis-Hastings mode** (via `--metropolis` flag): Ignores `step_size` and uses self-tuning MH weights instead. Does not require knowing `num_drones`.

---

## Monte Carlo

```yaml
monte_carlo:
  num_runs: 100            # Trials per combination
  episode_length: 200      # Timesteps per trial
  blackout_steps: 20       # Consecutive all-None steps = track loss
```

**Track loss** is declared when all drones simultaneously fail to detect the target for `blackout_steps` consecutive steps. This typically happens when the target moves beyond `max_range`.

---

## RL (Layer 3 — in progress)

```yaml
rl:
  lr: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  epochs: 10
  batch_size: 64
  hidden_dim: 128
```

These parameters configure the PPO agent for learned sensor placement. Not yet active in the current scripts.
