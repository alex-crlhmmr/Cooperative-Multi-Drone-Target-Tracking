"""Configuration for RL-based active sensing (Layer 3)."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrackingConfig:
    # --- Environment ---
    num_drones: int = 5
    episode_length: int = 3000
    ctrl_freq: int = 48
    pyb_freq: int = 240
    v_max: float = 15.0
    target_trajectory: str = "multi_segment"
    target_speed: float = 12.0
    target_sigma_a: float = 0.3
    min_altitude: float = 20.0

    # --- Sensor (matches config/default.yaml) ---
    fov_half_deg: float = 15.0
    sigma_bearing_deg: float = 2.0
    max_range: float = 300.0
    range_ref: float = 100.0
    p_detect_max: float = 0.99
    p_detect_range_half: float = 250.0

    # --- Filter (ConsensusIMM) ---
    imm_sigma_a_modes: tuple = (0.3, 3.0)
    imm_transition_matrix: tuple = (0.95, 0.05, 0.05, 0.95)  # flattened 2x2
    topology: str = "full"
    consensus_iters: int = 2
    consensus_step_size: float = 0.1
    dropout_prob: float = 0.5
    P0_pos: float = 10000.0
    P0_vel: float = 100.0

    # --- Reward ---
    w_cov: float = 1.0
    w_dist: float = 0.5         # continuous distance penalty to encourage following
    dist_scale: float = 100.0   # normalize distance: reward = -w_dist * dist / dist_scale
    w_detection: float = 0.1    # bonus per drone that detects target
    w_separation: float = 0.3   # angular separation bonus (per-drone)
    cov_scale: float = 1000.0

    # --- PPO ---
    lr: float = 3e-4
    clip_epsilon: float = 0.2
    num_epochs: int = 10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.95
    gae_lambda: float = 0.95
    rollout_steps: int = 1500
    mini_batch_size: int = 2048

    # --- Network ---
    ego_hidden: tuple = (128, 128)
    neighbor_hidden: tuple = (64, 64)
    neighbor_embed_dim: int = 32
    policy_hidden: tuple = (128, 128)

    # --- Training ---
    seed: int = 42
    num_envs: int = 16
    max_training_steps: int = 5_000_000
    eval_freq: int = 10000
    save_freq: int = 50000
    save_path: str = "./output/rl_tracking"
    normalize_obs: bool = True

    @property
    def sensor_config(self) -> dict:
        return {
            "fov_half_deg": self.fov_half_deg,
            "sigma_bearing_deg": self.sigma_bearing_deg,
            "max_range": self.max_range,
            "range_ref": self.range_ref,
            "p_detect_max": self.p_detect_max,
            "p_detect_range_half": self.p_detect_range_half,
        }

    @property
    def transition_matrix(self) -> np.ndarray:
        M = len(self.imm_sigma_a_modes)
        return np.array(self.imm_transition_matrix).reshape(M, M)

    @property
    def sigma_bearing_rad(self) -> float:
        return np.deg2rad(self.sigma_bearing_deg)

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_freq
