# Getting Started

This guide walks you through installation, verifying your setup, and running your first experiment.

---

## Prerequisites

- **macOS or Linux** (tested on macOS 14+ with Apple Silicon)
- **Conda** (Miniconda or Anaconda)
- **Python 3.10+**

---

## Installation

### 1. Create the conda environment

```bash
conda create -n aa273 python=3.10 -y
conda activate aa273
```

### 2. Install core dependencies

```bash
pip install numpy scipy matplotlib networkx pyyaml torch
```

### 3. Install simulation dependencies

```bash
pip install pybullet gymnasium
pip install gym-pybullet-drones
```

> **Note:** `gym-pybullet-drones` v2.0 is installed from PyPI. If you need the latest development version, install from GitHub:
> ```bash
> pip install git+https://github.com/utiasDSL/gym-pybullet-drones.git
> ```

### 4. Verify installation

```bash
conda activate aa273
python -c "import pybullet; import gymnasium; from gym_pybullet_drones.envs import BaseAviary; print('All good')"
```

You should see `All good` with no errors.

---

## First Run: Filter Comparison

Run the centralized filter comparison (EKF vs UKF vs PF):

```bash
python scripts/run_filters.py --no-gui --steps 200
```

**What happens:**
1. Five surveillance drones spawn in a hollow sphere (100-300m) around a target
2. The target flies a multi-segment 3D trajectory at 12 m/s
3. Each drone takes bearing-only measurements (azimuth + elevation)
4. Three filters (EKF, UKF, PF) process the same measurement stream
5. After the simulation, you'll see:
   - A metrics table in the terminal
   - RMSE and NEES plots
   - An interactive 3D replay animation

**Expected terminal output:**
```
Running filter comparison: traj=multi_segment, steps=200, seed=42
  Filters: ['EKF', 'UKF', 'PF']
Done in 1.23s (199 steps)

====================================================================
  Filter     |   Pos RMSE |   Vel RMSE |    ANEES |   Conv | Final Err
--------------------------------------------------------------------
  EKF        |     12.5 m |    15.1 m/s |   1050.2 |      1 |   15.5 m
  UKF        |     12.4 m |    15.0 m/s |    980.3 |      1 |   15.3 m
  PF         |     12.0 m |    14.9 m/s |     20.7 |      1 |   14.0 m
====================================================================
```

> **Tip:** Remove `--no-gui` to see the PyBullet 3D viewer with live drone physics. This is much slower (~28ms/step vs ~1ms/step) but visually impressive.

---

## Second Run: Distributed Consensus

Run the consensus EKF across all three topologies:

```bash
python scripts/run_consensus.py --no-gui --all-topologies --steps 200
```

This shows how distributed filtering compares to the centralized baseline, and how topology affects performance.

---

## Third Run: Full Comparison

Run everything — centralized and distributed filters with IMM on an evasive target:

```bash
python scripts/run_consensus_imm.py --no-gui --traj evasive --steps 500
```

---

## Running with PyBullet GUI

Remove the `--no-gui` flag to see the 3D physics simulation:

```bash
python scripts/run_filters.py --steps 200
```

The PyBullet window shows the drones physically flying and tracking the target. The camera auto-follows the action.

> **Performance:** GUI mode runs at ~28ms/step on Apple M4 Max. Headless mode runs at ~1ms/step.

---

## Saving Results

Add `--save` to any script to save plots and data to `results/`:

```bash
python scripts/run_filters.py --no-gui --steps 300 --save
python scripts/monte_carlo.py --runs 100 --save
```

Saved files include PNG plots and `.npz` data files for further analysis.

---

## Next Steps

- **[Filters](filters.md)** — Understand the available filter types and their parameters
- **[Experiments](experiments.md)** — Full reference for every script and CLI flag
- **[Configuration](configuration.md)** — Tune simulation, sensor, and filter parameters
- **[Visualization](visualization.md)** — Available plots and how to save them
