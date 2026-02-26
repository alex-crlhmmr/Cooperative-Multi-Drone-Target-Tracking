# Filter Reference

All filters share a common interface defined in `src/filters/base.py`:

```python
filter.predict()                              # Time update
filter.update(measurements, drone_positions)  # Measurement update
filter.get_estimate()     # → np.ndarray (6,)   [px, py, pz, vx, vy, vz]
filter.get_covariance()   # → np.ndarray (6, 6)
filter.initialized        # → bool
filter.name               # → str
```

**State vector:** All filters estimate a 6D target state `[px, py, pz, vx, vy, vz]`.

**Measurements:** Each drone produces a 2D bearing vector `[azimuth, elevation]` or `None` (no detection).

**Dynamics model:** Constant velocity (CV) with white-noise acceleration:

```
x_{k+1} = F x_k + w_k,    w_k ~ N(0, Q(sigma_a))
```

---

## Centralized Filters

These filters process **all drone measurements at once** (god-node assumption).

### EKF — Extended Kalman Filter

**File:** `src/filters/ekf.py`

Standard first-order linearization of the bearing measurement model. Joseph-form covariance update for numerical stability.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_a` | 0.6 | Process noise acceleration std (m/s^2) |
| `sigma_bearing` | 2.0 deg | Bearing measurement noise std |
| `range_ref` | 100.0 m | Reference range for noise scaling |
| `P0_pos` | 10000.0 | Initial position variance (m^2) |
| `P0_vel` | 100.0 | Initial velocity variance (m^2/s^2) |

**Initialization:** Triangulation from the first timestep with >= 2 valid bearing measurements.

**When to use:** Fast baseline. Good post-convergence accuracy. Struggles with initialization (range ambiguity) and model mismatch (fixed sigma_a).

---

### UKF — Unscented Kalman Filter

**File:** `src/filters/ukf.py`

Sigma-point propagation through the nonlinear bearing model. Avoids explicit Jacobian computation. Uses circular mean for azimuth components.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.5 | Sigma-point spread |
| `beta` | 2.0 | Prior distribution parameter |
| `kappa` | 0.0 | Secondary scaling |
| *(plus all EKF params)* | | |

**When to use:** Slightly better than EKF for highly nonlinear geometries. Similar computational cost. In practice, EKF and UKF perform nearly identically for this problem.

---

### PF — Particle Filter

**File:** `src/filters/pf.py`

Sequential Monte Carlo with 2000 particles. Handles the bearing-only range ambiguity naturally through ray-spread initialization.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_particles` | 2000 | Number of particles |
| `resample_threshold` | 0.5 | N_eff/N threshold for systematic resampling |
| `process_noise_factor` | 5.0 | Q inflation factor (prevents particle degeneracy) |
| `jitter_pos` | 1.0 m | Position roughening jitter per step |
| `jitter_vel` | 0.5 m/s | Velocity roughening jitter per step |
| `range_min` | 10.0 m | Minimum range for ray initialization |
| `range_max` | 500.0 m | Maximum range for ray initialization |

**Initialization:** Particles spread along bearing rays from each drone at uniformly sampled ranges. This naturally represents the range ambiguity that Gaussian filters cannot capture.

**When to use:** Best initialization performance. Best NEES calibration (honest uncertainty). ~5x slower than EKF/UKF (~0.37s/trial vs 0.07s/trial).

---

### IMM — Interacting Multiple Model

**File:** `src/filters/imm.py`

Runs M parallel EKF sub-filters with different process noise levels. Model probabilities are updated at each step based on measurement likelihood. The blended output adapts sigma_a online.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_a_modes` | [0.3, 3.0] | Process noise per mode (gentle, aggressive) |
| `transition_matrix` | [[0.95, 0.05], [0.10, 0.90]] | Markov mode transition |

**Extra method:**
```python
filter.get_mode_probabilities()  # → np.ndarray (M,)
```

**When to use:** Maneuvering targets where a single sigma_a is insufficient. The mode probabilities reveal when the target is maneuvering vs cruising.

---

## Distributed Filters

These filters run **one local filter per drone** and fuse estimates through neighbor-to-neighbor consensus.

### ConsensusEKF — Distributed Consensus EKF

**File:** `src/filters/consensus_ekf.py`

N local EKFs operating in information form. After each measurement update, drones exchange information matrices with neighbors and average them over L consensus iterations.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_drones` | 5 | Number of local filters |
| `adjacency` | (full graph) | N x N binary adjacency matrix |
| `num_consensus_iters` | 2 | L — consensus rounds per timestep |
| `consensus_step_size` | 0.1 | Epsilon — averaging step size |
| `dropout_prob` | 0.0 | Per-edge dropout probability |
| `metropolis` | False | Use MH weights instead of fixed epsilon |

**Information form:**
```
Y_i = P_i^{-1}           (information matrix)
y_i = P_i^{-1} x_i       (information vector)
```

**Classic consensus (metropolis=False):**
- Measurement scaling: `Y_i += N * H'R^{-1}H`
- Averaging: `Y_i += eps * (Y_j - Y_i)` for each neighbor j

**Metropolis-Hastings (metropolis=True):**
- Measurement scaling: `Y_i += (d_i + 1) * H'R^{-1}H` where d_i is drone i's degree
- Averaging: `Y_i += w_ij * (Y_j - Y_i)` where `w_ij = 1 / (1 + max(d_i, d_j))`
- N-agnostic: each drone only needs its own neighbor count

**Extra methods:**
```python
filter.get_local_estimate(i)      # Drone i's local state estimate
filter.get_local_covariance(i)    # Drone i's local covariance
filter.get_disagreement()         # RMS spread of local estimates
filter.get_active_edges()         # Effective adjacency (after dropout)
```

---

### ConsensusIMM — Distributed Consensus IMM

**File:** `src/filters/consensus_imm.py`

Each drone runs a full local IMM (mixing, per-mode EKF update, probability update). The blended IMM output is converted to information form for consensus averaging. Combines multi-model adaptivity with distributed fusion.

Same consensus parameters as ConsensusEKF, plus the IMM-specific `sigma_a_modes` and `transition_matrix`.

**Extra methods:**
```python
filter.get_mode_probabilities()   # → np.ndarray (N, M) — per-drone mode probs
```

---

## Topologies

Generated by `src/filters/topology.py`:

```python
from src.filters import generate_adjacency
adj = generate_adjacency(n=5, topology="full")  # "full", "ring", or "star"
```

| Topology | Edges | Properties |
|----------|-------|------------|
| **Full** | N(N-1)/2 | Maximum connectivity, fastest consensus convergence |
| **Ring** | N | Minimal connectivity, slowest convergence, symmetric degrees |
| **Star** | N-1 | Hub-dependent, fast through hub, fragile to hub failure |

**Dropout** is applied per-edge per-consensus-iteration via `apply_dropout(adj, prob, rng)`. The effective adjacency varies each iteration, simulating unreliable communication.

---

## Measurement Model

**File:** `src/filters/measurement.py`

Bearing-only measurements from each drone:

```
z = [azimuth, elevation] + noise
```

- **Azimuth:** `atan2(dy, dx)` — horizontal angle to target
- **Elevation:** `atan2(dz, ||dxy||)` — vertical angle to target
- **Noise:** Gaussian with range-dependent scaling: `sigma(r) = sigma_0 * (r / r_ref)`
- **Detection probability:** Range-dependent sigmoid model

**Gimbal:** Cameras are gimballed toward the filter's current estimate. Three modes:
- `perfect` — gimballed toward true target position (oracle)
- `centralized` — gimballed toward centralized filter estimate
- `local` — each drone gimbals toward its own local estimate (realistic)
