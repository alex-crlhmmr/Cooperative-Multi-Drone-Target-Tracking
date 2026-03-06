"""V2 tracking config — MAPPO with PBRS reward and Beta actor."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class V2TrackingConfig:
    # ── Environment ──
    num_drones: int = 5
    episode_length: int = 5000
    v_max: float = 15.0
    target_trajectory: str = "evasive"
    target_speed: float = 12.0
    target_sigma_a: float = 0.3
    ctrl_freq: int = 48
    pyb_freq: int = 240
    min_altitude: float = 20.0

    # ── Sensor (matches config/default.yaml) ──
    fov_half_deg: float = 15.0
    sigma_bearing_deg: float = 2.0
    max_range: float = 300.0
    range_ref: float = 100.0
    p_detect_max: float = 0.99
    p_detect_range_half: float = 250.0

    # ── Filter (ConsensusIMM) ──
    imm_sigma_a_modes: tuple = (0.3, 3.0)
    imm_transition_matrix: tuple = (0.95, 0.05, 0.05, 0.95)
    topology: str = "full"
    consensus_iters: int = 2
    consensus_step_size: float = 0.1
    dropout_prob: float = 0.5
    P0_pos: float = 10000.0
    P0_vel: float = 100.0

    # ── Action (residual) ──
    residual_scale: float = 0.50       # RL correction = 50% of v_max
    base_Kp: float = 2.0              # chase controller proportional gain
    base_R_desired: float = 60.0      # baseline radial offset from estimate

    # ── Observation ──
    actor_obs_dim: int = 14
    critic_extra_dim: int = 9          # consensus_est(6) + log_P_eigs(3)

    # ── Reward (PBRS) ──
    gamma: float = 0.99
    reward_clip: float = 5.0

    # ── PPO (MAPPO recommendations) ──
    lr: float = 5e-4
    ppo_epochs: int = 5
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm_actor: float = 0.5
    max_grad_norm_critic: float = 10.0
    gae_lambda: float = 0.95
    rollout_steps: int = 1000
    mini_batch_size: int = 1024

    # ── Network ──
    actor_hidden: tuple = (64, 64)
    critic_hidden: tuple = (128, 128)

    # ── Training ──
    seed: int = 42
    num_envs: int = 16
    max_training_steps: int = 5_000_000
    eval_freq: int = 50_000
    save_freq: int = 100_000
    save_path: str = "./output/v2"
    normalize_obs: bool = True
    normalize_value_targets: bool = True
    spawn_mode: str = "cluster"

    # ── Early termination ──
    early_term_grace: int = 1000
    early_term_threshold: float = 10000.0
    early_term_patience: int = 200

    @property
    def critic_obs_dim(self) -> int:
        return self.actor_obs_dim + self.critic_extra_dim

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
