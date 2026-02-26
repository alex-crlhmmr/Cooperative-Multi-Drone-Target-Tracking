# Experiment Scripts

All scripts are in the `scripts/` directory. Run them from the project root:

```bash
conda activate aa273
python scripts/<script_name>.py [flags]
```

---

## Table of Contents

- [run_filters.py](#run_filterspy) — Centralized filter comparison
- [run_consensus.py](#run_consensuspy) — Consensus EKF with sweeps
- [run_consensus_imm.py](#run_consensus_immpy) — Full filter comparison
- [monte_carlo.py](#monte_carlopy) — MC sweep for centralized filters
- [monte_carlo_consensus.py](#monte_carlo_consensuspy) — MC sweep for consensus filters
- [pid_tuning.py](#pid_tuningpy) — PID gain tuning

---

## run_filters.py

**Purpose:** Compare centralized EKF, UKF, and PF on the same measurement stream.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--no-gui` | flag | off | Headless mode (no PyBullet window) |
| `--traj` | str | from config | Trajectory: `straight`, `single_turn`, `multi_segment`, `evasive` |
| `--steps` | int | 200 | Episode length in timesteps |
| `--seed` | int | 42 | Random seed |
| `--perfect-gimbal` | flag | off | Gimbal uses true target position |
| `--save` | flag | off | Save plots and data to `results/` |
| `--no-replay` | flag | off | Skip interactive 3D animation |

**Outputs:** State estimates (6-panel), RMSE, NEES, covariance trace, 3D trajectory, interactive replay.

**Examples:**

```bash
# Quick headless run
python scripts/run_filters.py --no-gui --steps 200

# Evasive target, longer run, save results
python scripts/run_filters.py --no-gui --traj evasive --steps 1000 --save

# With PyBullet 3D viewer
python scripts/run_filters.py --steps 200

# Perfect gimbal (oracle) — isolates filter performance from gimbal errors
python scripts/run_filters.py --no-gui --perfect-gimbal --steps 300
```

---

## run_consensus.py

**Purpose:** Consensus EKF experiments — topology comparison and parameter sweeps.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--no-gui` | flag | off | Headless mode |
| `--traj` | str | from config | Trajectory type |
| `--steps` | int | 200 | Episode length |
| `--seed` | int | 42 | Random seed |
| `--topology` | str | `full` | `full`, `ring`, or `star` |
| `--dropout` | float | 0.0 | Link dropout probability (0.0 to 1.0) |
| `--num-drones` | int | from config | Override drone count |
| `--perfect-gimbal` | flag | off | Gimbal uses true position |
| `--centralized-gimbal` | flag | off | Gimbal uses centralized EKF estimate |
| `--all-topologies` | flag | off | Run full + ring + star simultaneously |
| `--sweep-L` | flag | off | Sweep consensus iterations: L = [1, 2, 3, 5, 10, 20, 50] |
| `--sweep-eps` | flag | off | Sweep step size: eps = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5] |
| `--sweep-dropout` | flag | off | Sweep dropout: [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8] |
| `--save` | flag | off | Save results |
| `--no-replay` | flag | off | Skip animation |

**Outputs:** Per-drone estimates, topology comparison, RMSE, NEES, disagreement, sweep plots, animation.

**Examples:**

```bash
# Compare all topologies
python scripts/run_consensus.py --no-gui --all-topologies --steps 300

# Sweep consensus iterations to find optimal L
python scripts/run_consensus.py --no-gui --sweep-L --steps 300

# Sweep step size epsilon
python scripts/run_consensus.py --no-gui --sweep-eps --steps 300

# Dropout degradation across topologies
python scripts/run_consensus.py --no-gui --sweep-dropout --all-topologies --steps 300

# Star topology with heavy dropout
python scripts/run_consensus.py --no-gui --topology star --dropout 0.5 --steps 500

# 10 drones on ring topology
python scripts/run_consensus.py --no-gui --topology ring --num-drones 10 --steps 300
```

---

## run_consensus_imm.py

**Purpose:** Full comparison — centralized (EKF, IMM), distributed (ConsensusEKF, ConsensusIMM), and PF.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--no-gui` | flag | off | Headless mode |
| `--traj` | str | from config | Trajectory type |
| `--steps` | int | 200 | Episode length |
| `--seed` | int | 42 | Random seed |
| `--topology` | str | `full` | Communication topology |
| `--dropout` | float | 0.0 | Link dropout probability |
| `--num-drones` | int | from config | Override drone count |
| `--perfect-gimbal` | flag | off | Oracle gimbal |
| `--centralized-gimbal` | flag | off | Centralized EKF gimbal |
| `--all-topologies` | flag | off | Run all three topologies |
| `--metropolis` | flag | off | Use Metropolis-Hastings consensus weights |
| `--all-filters` | flag | (default) | Run all 5 filters |
| `--no-pf` | flag | off | Skip PF (faster runs) |
| `--only-consensus` | flag | off | Consensus EKF + Consensus IMM only |
| `--filters` | str | all | Comma-separated: `ekf,imm,consensus-ekf,consensus-imm,pf` |
| `--save` | flag | off | Save results |
| `--no-replay` | flag | off | Skip animation |

**Outputs:** RMSE, NEES, mode probabilities, per-drone estimates, disagreement, replay animations.

**Examples:**

```bash
# All filters on evasive target
python scripts/run_consensus_imm.py --no-gui --traj evasive --steps 500

# Only consensus filters — no centralized baselines
python scripts/run_consensus_imm.py --no-gui --only-consensus --traj evasive --steps 500

# Specific filters
python scripts/run_consensus_imm.py --no-gui --filters consensus-imm,pf --steps 500

# MH weights on star topology with dropout
python scripts/run_consensus_imm.py --no-gui --metropolis --topology star --dropout 0.8 --traj evasive --steps 500

# All topologies comparison
python scripts/run_consensus_imm.py --no-gui --all-topologies --traj evasive --steps 500

# Skip PF for faster iteration
python scripts/run_consensus_imm.py --no-gui --no-pf --steps 300
```

### Filter Selection

| Flag | Filters Run |
|------|-------------|
| *(default)* / `--all-filters` | EKF, IMM, ConsensusEKF, ConsensusIMM, PF |
| `--no-pf` | EKF, IMM, ConsensusEKF, ConsensusIMM |
| `--only-consensus` | ConsensusEKF, ConsensusIMM |
| `--filters X,Y` | Only the listed filters |

Valid filter keys for `--filters`: `ekf`, `imm`, `consensus-ekf`, `consensus-imm`, `pf`

---

## monte_carlo.py

**Purpose:** Monte Carlo sweep across filters and trajectories with multiprocessing.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--runs` | int | from config (100) | Trials per (filter, trajectory) combo |
| `--filter` | str | all | Restrict to one filter: `ekf`, `ukf`, `pf` |
| `--traj` | str | all | Restrict to one trajectory |
| `--workers` | int | cpu_count - 1 | Parallel worker processes |
| `--save` | flag | off | Save plots and `.npz` data |

**Default sweep:** 3 filters x 4 trajectories = 12 combinations, 100 trials each.

**Outputs:** Box plots (pos RMSE, vel RMSE, ANEES, convergence time), track loss bar chart.

**Examples:**

```bash
# Full sweep: 400 trials, all filters, all trajectories
python scripts/monte_carlo.py --runs 400 --save

# Quick test: just EKF on evasive
python scripts/monte_carlo.py --runs 20 --filter ekf --traj evasive

# Maximum parallelism
python scripts/monte_carlo.py --runs 200 --workers 16
```

---

## monte_carlo_consensus.py

**Purpose:** Monte Carlo sweep for consensus EKF across topologies, dropout levels, and trajectories.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--runs` | int | from config (100) | Trials per combo |
| `--topology` | str | all | Restrict to one topology: `full`, `ring`, `star` |
| `--traj` | str | all | Restrict to one trajectory |
| `--workers` | int | cpu_count - 1 | Parallel workers |
| `--save` | flag | off | Save results |

**Default sweep:** 3 topologies x 5 dropout levels x 2 trajectories = 30 combinations.

**Outputs:** Dropout degradation curves (median + IQR shading per topology), topology box plots per trajectory.

**Examples:**

```bash
# Full MC sweep
python scripts/monte_carlo_consensus.py --runs 100 --save

# Just star topology
python scripts/monte_carlo_consensus.py --runs 50 --topology star

# Just evasive trajectory
python scripts/monte_carlo_consensus.py --runs 100 --traj evasive --save
```

---

## pid_tuning.py

**Purpose:** PID controller gain tuning for the 4kg surveillance drone.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gui` | flag | off | Enable PyBullet viewer |
| `--test` | str | (all) | Single test: `altitude`, `lateral`, `combined`, `tracking` |
| `--sweep` | str | none | Gain sweep: `altitude`, `lateral`, or `both` |
| `--show` | flag | off | Show plots interactively |

**Examples:**

```bash
# Run all step-response tests
python scripts/pid_tuning.py

# Altitude gain sweep (P_z × D_z grid)
python scripts/pid_tuning.py --sweep altitude

# Visual test with PyBullet viewer
python scripts/pid_tuning.py --gui --test tracking
```

---

## Common Patterns

### Headless batch runs

```bash
# Use --no-gui --no-replay --save for automated batch processing
python scripts/run_filters.py --no-gui --no-replay --save --steps 500
```

### Reproducibility

```bash
# Same --seed produces identical results
python scripts/run_filters.py --no-gui --seed 123 --steps 200
python scripts/run_filters.py --no-gui --seed 123 --steps 200  # identical
```

### Trajectory types

All scripts accept `--traj` with these options:

| Trajectory | Description |
|------------|-------------|
| `straight` | Constant velocity, random 3D heading |
| `single_turn` | Straight-turn-straight with altitude ramp |
| `multi_segment` | 3-6 segments with random headings and speeds |
| `evasive` | Continuous rapid maneuvers (stress test) |
