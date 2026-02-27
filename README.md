# Cooperative Multi-Drone Target Tracking

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyBullet](https://img.shields.io/badge/sim-PyBullet-orange.svg)](https://pybullet.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A research toolkit for **distributed Bayesian filtering** applied to cooperative drone-based target tracking. N surveillance quadrotors estimate a maneuvering target's 3D state using bearing-only measurements, with configurable communication topologies, link dropout, and consensus protocols.

Built on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) for realistic 3D physics simulation.

---

## Highlights

- **6 filter implementations** — EKF, UKF, Particle Filter, IMM, Consensus EKF, Consensus IMM
- **Distributed consensus** in information form with N-scaling or Metropolis-Hastings weights
- **Communication modeling** — full/ring/star topologies with stochastic link dropout
- **Rich visualization** — interactive 3D replay with live comm links, bearing lines, per-drone estimates
- **Monte Carlo framework** — parallel multi-trial sweeps across filters, trajectories, and topologies
- **Fully configurable** — single YAML file controls all simulation, sensor, filter, and consensus parameters

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Simulation Layer                             │
│  PyBullet physics (480 Hz) ─► Control (48 Hz) ─► Sensor readings   │
│  4kg surveillance quads    PID waypoint tracking   bearing angles   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │           Estimation Layer               │
              │                                          │
              │  Centralized          Distributed        │
              │  ┌─────────┐         ┌──────────────┐   │
              │  │ EKF     │         │ ConsensusEKF │   │
              │  │ UKF     │         │ ConsensusIMM │   │
              │  │ PF      │         │              │   │
              │  │ IMM     │         │ Topologies:  │   │
              │  └─────────┘         │ full/ring/   │   │
              │   (god-node)         │ star+dropout │   │
              │                      └──────────────┘   │
              └────────────────────┬────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │          Analysis Layer                  │
              │                                          │
              │  RMSE / NEES / Covariance traces         │
              │  Monte Carlo sweeps (multiprocessing)    │
              │  Parameter sweeps (L, epsilon, dropout)  │
              │  3D animated replay with comm topology   │
              └─────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
conda create -n aa273 python=3.10 -y
conda activate aa273
pip install pybullet gymnasium gym-pybullet-drones
pip install numpy scipy matplotlib networkx pyyaml torch
```

### 2. First run — filter comparison

```bash
python scripts/run_filters.py --no-gui --steps 300
```

Compares EKF, UKF, and Particle Filter on a multi-segment 3D trajectory. Outputs RMSE/NEES plots and an interactive 3D replay.

### 3. Distributed consensus

```bash
python scripts/run_consensus.py --no-gui --all-topologies --steps 300
```

Runs consensus EKF across full, ring, and star topologies. Shows per-drone local estimates converging through consensus.

### 4. Full comparison with IMM

```bash
python scripts/run_consensus_imm.py --no-gui --traj evasive --topology star --dropout 0.3 --steps 500
```

All filters head-to-head on an evasive target with degraded comms.

---

## Project Structure

```
.
├── config/
│   └── default.yaml            # All simulation, sensor, filter, consensus parameters
├── scripts/
│   ├── run_filters.py          # Single-run: EKF vs UKF vs PF
│   ├── run_consensus.py        # Consensus EKF with topology/dropout sweeps
│   ├── run_consensus_imm.py    # Full comparison: all filters + IMM + MH weights
│   ├── monte_carlo.py          # MC sweep: centralized filters × trajectories
│   ├── monte_carlo_consensus.py# MC sweep: consensus × topologies × dropout
│   └── pid_tuning.py           # PID gain tuning for surveillance drone
├── src/
│   ├── filters/                # All filter implementations
│   │   ├── ekf.py              #   Extended Kalman Filter
│   │   ├── ukf.py              #   Unscented Kalman Filter
│   │   ├── pf.py               #   Particle Filter (2000 particles)
│   │   ├── imm.py              #   Interacting Multiple Model
│   │   ├── consensus_ekf.py    #   Distributed Consensus EKF
│   │   ├── consensus_imm.py    #   Distributed Consensus IMM
│   │   ├── topology.py         #   Graph generation + dropout
│   │   └── measurement.py      #   Bearing model, Jacobian, initialization
│   ├── dynamics/               # Target motion models + trajectory generation
│   ├── sensors/                # Bearing sensor + noise models
│   ├── env/                    # PyBullet tracking aviary
│   ├── control/                # Surveillance drone PID controller
│   └── viz/                    # Plots, animations, consensus diagrams
├── assets/                     # Drone URDF models
├── results/                    # Saved plots and data (gitignored)
└── docs/                       # Documentation
    ├── getting-started.md      # Installation and first run
    ├── filters.md              # Filter reference
    ├── experiments.md          # Script reference with all CLI flags
    ├── configuration.md        # YAML config reference
    └── visualization.md        # Plots and animations guide
```

---

## Documentation

| Page | Description |
|------|-------------|
| **[Getting Started](docs/getting-started.md)** | Environment setup, dependencies, first run walkthrough |
| **[Filters](docs/filters.md)** | Filter catalog — math, parameters, when to use each |
| **[Experiments](docs/experiments.md)** | Every script, all CLI flags, example commands |
| **[Configuration](docs/configuration.md)** | Full `config/default.yaml` reference |
| **[Visualization](docs/visualization.md)** | Available plots and animations, saving figures |

---

## Example Commands

```bash
# Monte Carlo: 400 trials, all filters × all trajectories
python scripts/monte_carlo.py --runs 400 --save

# Consensus parameter sweep: iterations L = [1..50]
python scripts/run_consensus.py --no-gui --sweep-L --steps 300

# Dropout degradation study
python scripts/run_consensus.py --no-gui --sweep-dropout --all-topologies --steps 300

# Metropolis-Hastings weights on star topology with 80% dropout
python scripts/run_consensus_imm.py --no-gui --metropolis --topology star --dropout 0.8 --traj evasive --steps 500

# Only consensus filters (no centralized baselines)
python scripts/run_consensus_imm.py --no-gui --only-consensus --steps 300

# Pick specific filters
python scripts/run_consensus_imm.py --no-gui --filters consensus-imm,pf --steps 300

# Monte Carlo consensus: both EKF + IMM, custom dropout values
python scripts/monte_carlo_consensus.py --filters both --dropout "0.0,0.2,0.5,0.8" --runs 100 --save

# Monte Carlo consensus: override L and epsilon
python scripts/monte_carlo_consensus.py --L 10 --eps 0.05 --filters both --runs 50
```

---

## Filters at a Glance

| Filter | Type | Key Strength | Distributed |
|--------|------|-------------|-------------|
| **EKF** | Linearized Gaussian | Fast, good post-convergence | No |
| **UKF** | Sigma-point Gaussian | Better nonlinearity handling | No |
| **PF** | Monte Carlo | Handles range ambiguity, multimodal | No |
| **IMM** | Multi-model EKF | Adapts to maneuvering targets | No |
| **ConsensusEKF** | Info-form + consensus | Distributed, topology-aware | Yes |
| **ConsensusIMM** | Info-form IMM + consensus | Distributed + adaptive | Yes |

See [docs/filters.md](docs/filters.md) for mathematical details and parameter descriptions.

---

## Key Concepts

### Bearing-Only Tracking
Each drone measures azimuth and elevation angles to the target — no range information. This creates an inherent **range ambiguity** at initialization that requires either triangulation (EKF/UKF) or ray-spread sampling (PF).

### Information-Form Consensus
Distributed filters operate in information form (Y = P<sup>-1</sup>, y = Y x). Each drone:
1. Computes local measurement updates
2. Scales contributions by N (classic) or d<sub>i</sub>+1 (Metropolis-Hastings)
3. Averages with neighbors over L consensus iterations
4. The consensus average converges to the centralized solution

### Metropolis-Hastings Weights
Classic consensus requires knowing N (total swarm size). MH weights replace this with local degree information: w<sub>ij</sub> = 1 / (1 + max(d<sub>i</sub>, d<sub>j</sub>)). Each drone only needs to know how many neighbors it has — fully N-agnostic.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{carlhammar2026cooperative,
  author       = {Carlhammar, Alexandre},
  title        = {Cooperative Multi-Drone Target Tracking with Distributed Bayesian Filtering},
  year         = {2026},
  url          = {https://github.com/alex-crlhmmr/Cooperative-Multi-Drone-Target-Tracking},
  note         = {AA 273: State Estimation and Filtering, Stanford University}
}
```

**Contact:** Alexandre Carlhammar — [acarlham@stanford.edu](mailto:acarlham@stanford.edu)
