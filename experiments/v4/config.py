"""V4 Config — Improved hyperparameters for two-phase RL."""

from dataclasses import dataclass
import numpy as np


@dataclass
class V4Config:
    # ── Shared ──
    num_drones: int = 5
    v_max: float = 15.0
    ctrl_freq: int = 48
    pyb_freq: int = 240
    min_altitude: float = 20.0
    target_speed: float = 12.0
    target_trajectory: str = "evasive"
    target_sigma_a: float = 0.3
    topology: str = "full"
    gamma: float = 0.995
    seed: int = 42
    num_envs: int = 16

    # ── Sensor ──
    fov_half_deg: float = 15.0
    sigma_bearing_deg: float = 2.0
    max_range: float = 300.0
    range_ref: float = 100.0
    p_detect_max: float = 0.99
    p_detect_range_half: float = 250.0

    # ── Filter (Phase 2 only) ──
    imm_sigma_a_modes: tuple = (0.3, 3.0)
    imm_transition_matrix: tuple = (0.95, 0.05, 0.05, 0.95)
    consensus_iters: int = 2
    consensus_step_size: float = 0.1
    dropout_prob: float = 0.5
    P0_pos: float = 10000.0
    P0_vel: float = 100.0

    # ── Phase 1 (Spread) ──
    spread_steps: int = 400
    spread_target_radius: float = 50.0
    spread_control: str = "full_rl"         # "full_rl" or "residual_repulsion"
    spread_repulsion_gain: float = 2.0
    spread_residual_scale: float = 0.5
    spread_obs_dim: int = 13
    spread_critic_extra: int = 6

    # ── Phase 2 (Track) ──
    track_episode_length: int = 5000
    track_residual_scale: float = 0.50
    track_base_Kp: float = 2.0
    track_base_R_desired: float = 60.0
    track_actor_obs_dim: int = 14
    track_critic_extra: int = 9

    # ── Reward ──
    reward_clip: float = 5.0

    # ── PPO (shared) ──
    lr: float = 3e-4
    ppo_epochs: int = 5
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm_actor: float = 0.5
    max_grad_norm_critic: float = 10.0
    gae_lambda: float = 0.97
    rollout_steps: int = 1000
    mini_batch_size: int = 512

    # ── Network ──
    actor_hidden: tuple = (128, 128)
    critic_hidden: tuple = (128, 128)

    # ── Training ──
    spread_training_steps: int = 1_000_000
    track_training_steps: int = 5_000_000
    eval_freq: int = 50_000
    save_freq: int = 100_000
    save_path: str = "./output/v4"
    normalize_obs: bool = True
    normalize_value_targets: bool = True

    # ── Early termination (Phase 2) ──
    early_term_grace: int = 1000
    early_term_threshold: float = 10000.0
    early_term_patience: int = 200

    @property
    def spread_critic_obs_dim(self) -> int:
        return self.spread_obs_dim + self.spread_critic_extra  # 13 + 6 = 19

    @property
    def track_critic_obs_dim(self) -> int:
        return self.track_actor_obs_dim + self.track_critic_extra  # 14 + 9 = 23

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
        M = int(np.sqrt(len(self.imm_transition_matrix)))
        return np.array(self.imm_transition_matrix).reshape(M, M)

    @property
    def sigma_bearing_rad(self) -> float:
        return np.radians(self.sigma_bearing_deg)

    @property
    def dt(self) -> float:
        return 1.0 / self.ctrl_freq
