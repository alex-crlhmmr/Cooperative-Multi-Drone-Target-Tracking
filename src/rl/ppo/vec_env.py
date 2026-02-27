"""Subprocess-based vectorized environment for parallel training.

Each worker runs its own MultiDroneTrackingEnv in a separate process
(each PyBullet instance gets its own physics server). Auto-resets on
termination, stashing terminal obs in info for correct value bootstrapping.
"""

import multiprocessing as mp
import numpy as np
import sys
from typing import Callable, List


class _EnvFactory:
    """Picklable factory for creating tracking environments in subprocesses."""

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = seed

    def __call__(self):
        from src.rl.tracking_env import MultiDroneTrackingEnv
        return MultiDroneTrackingEnv(self.cfg, seed=self.seed)


def _worker(pipe, env_fn: Callable):
    """Worker process: creates env, responds to commands via pipe."""
    env = env_fn()

    while True:
        cmd, data = pipe.recv()

        if cmd == "reset":
            obs, info = env.reset(seed=data)
            pipe.send((obs, info))

        elif cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated

            if done:
                # Stash terminal observation for value bootstrapping
                info["terminal_obs"] = obs.copy()
                obs, _ = env.reset()

            pipe.send((obs, reward, done, info))

        elif cmd == "close":
            env.close()
            pipe.close()
            break

        elif cmd == "set_attr":
            attr_name, attr_value = data
            setattr(env, attr_name, attr_value)
            pipe.send(True)

        elif cmd == "get_spaces":
            pipe.send((env.observation_space, env.action_space))

        else:
            raise ValueError(f"Unknown command: {cmd}")


class SubprocVecEnv:
    """Vectorized environment using subprocesses.

    Each environment runs in its own process for true parallelism.
    Auto-resets terminated environments and returns new initial obs.
    """

    def __init__(self, env_fns: List[Callable]):
        self.num_envs = len(env_fns)
        self.closed = False

        # Create pipes and workers
        self._parent_pipes = []
        self._child_pipes = []
        self._processes = []

        # Use 'fork' — faster startup and closures are picklable.
        # 'spawn' requires all env_fns to be top-level picklable objects.
        ctx = mp.get_context("fork")

        for env_fn in env_fns:
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(target=_worker, args=(child_pipe, env_fn), daemon=True)
            process.start()
            child_pipe.close()  # parent doesn't need the child end

            self._parent_pipes.append(parent_pipe)
            self._processes.append(process)

        # Get spaces from first env
        self._parent_pipes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self._parent_pipes[0].recv()

    def reset(self, seeds=None):
        """Reset all environments.

        Args:
            seeds: list of seeds (one per env) or None

        Returns:
            obs: (E, N, obs_dim) stacked observations
        """
        if seeds is None:
            seeds = [None] * self.num_envs

        for pipe, seed in zip(self._parent_pipes, seeds):
            pipe.send(("reset", seed))

        results = [pipe.recv() for pipe in self._parent_pipes]
        obs = np.stack([r[0] for r in results])
        return obs

    def step(self, actions: np.ndarray):
        """Step all environments.

        Args:
            actions: (E, N, 3) actions for all envs

        Returns:
            obs: (E, N, obs_dim) — new obs (auto-reset obs if done)
            rewards: (E, N) per-drone rewards
            dones: (E,) episode-level done flags
            infos: list of info dicts
        """
        for pipe, action in zip(self._parent_pipes, actions):
            pipe.send(("step", action))

        results = [pipe.recv() for pipe in self._parent_pipes]

        obs = np.stack([r[0] for r in results])
        rewards = np.stack([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]

        return obs, rewards, dones, infos

    def set_attr(self, attr_name: str, value):
        """Set an attribute on all worker environments."""
        for pipe in self._parent_pipes:
            pipe.send(("set_attr", (attr_name, value)))
        for pipe in self._parent_pipes:
            pipe.recv()

    def close(self):
        if self.closed:
            return
        self.closed = True

        for pipe in self._parent_pipes:
            try:
                pipe.send(("close", None))
            except BrokenPipeError:
                pass

        for proc in self._processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
