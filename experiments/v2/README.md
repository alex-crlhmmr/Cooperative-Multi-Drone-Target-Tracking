# V2: MAPPO-Style Active Sensing with PBRS Reward

Clean rewrite of the RL active sensing stack. Key improvements over V1:

- **Centralized critic** (MAPPO): critic sees global filter state, actor sees only local obs
- **PBRS reward**: potential-based shaping using FIM, no competing distance penalty
- **Beta actor**: bounded [-1,1] support without clipping bias
- **Residual actions**: 15% authority over chase+offset baseline
- **Simpler observations**: 14D actor (vs 32D in V1), 23D critic

## Quick Start

```bash
# Smoke test (2 envs, 100K steps)
python -m experiments.v2.train --steps 100000 --num-envs 2 --spawn-mode cluster

# Full training (5M steps, cluster spawn)
python -m experiments.v2.train --steps 5000000 --spawn-mode cluster --tag cluster_5M

# Evaluation
python -m experiments.v2.eval --checkpoint output/v2/checkpoint_best.pt --episodes 10

# Replay animation
python -m experiments.v2.replay --checkpoint output/v2/checkpoint_best.pt --cluster
python -m experiments.v2.replay --checkpoint output/v2/checkpoint_best.pt --baseline --cluster
```

## Files

| File | Description |
|------|-------------|
| `config.py` | `V2TrackingConfig` dataclass with all hyperparameters |
| `env.py` | `V2TrackingEnv` — dual obs, PBRS reward, residual actions |
| `networks.py` | `BetaActor` + `CentralizedCritic` |
| `fim.py` | Fisher Information Matrix from bearing geometry |
| `train.py` | MAPPO training loop with LR annealing |
| `eval.py` | Multi-scenario evaluation (RL vs baseline) |
| `replay.py` | Single-episode animation |

## Architecture

```
Actor (14D obs → Beta dist → 3D action):
  local_est_rel(3), vel_est(3), log_tr_P(1), detected(1),
  neighbor_mean_offset(3), neighbor_count(1), angular_rank(1), range_to_est(1)

Critic (23D obs → scalar value):
  actor_obs(14) + consensus_est(6) + log_P_eigenvalues(3)

Reward = clip(r_base + r_shaping, -5, 5)
  r_base = -clip(log(tr_P + 1) / log(1001), -1, 1)
  r_shaping = γ·clip(log(det(FIM')), -20, 5) - clip(log(det(FIM)), -20, 5)
```
