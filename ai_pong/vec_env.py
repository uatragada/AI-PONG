from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict

import numpy as np

from ai_pong.env import PongConfig, PongEnv


def _worker_loop(connection, config_dict: dict, seeds: list[int]) -> None:
    env_config = PongConfig(**config_dict)
    envs = [PongEnv(config=env_config, seed=seed) for seed in seeds]
    try:
        while True:
            command, payload = connection.recv()
            if command == "reset":
                observations = [env.reset() for env in envs]
                left = np.stack([pair[0] for pair in observations]).astype(np.float32)
                right = np.stack([pair[1] for pair in observations]).astype(np.float32)
                connection.send((left, right))
                continue

            if command == "step":
                left_actions, right_actions = payload
                next_left_observations: list[np.ndarray] = []
                next_right_observations: list[np.ndarray] = []
                left_rewards: list[float] = []
                right_rewards: list[float] = []
                dones: list[bool] = []
                infos: list[dict[str, float | int | str]] = []

                for env, left_action, right_action in zip(envs, left_actions, right_actions):
                    (next_left, next_right), (left_reward, right_reward), done, info = env.step(
                        int(left_action),
                        int(right_action),
                    )
                    if done:
                        info = dict(info)
                        info["terminal_left_observation"] = next_left
                        info["terminal_right_observation"] = next_right
                        next_left, next_right = env.reset()

                    next_left_observations.append(next_left)
                    next_right_observations.append(next_right)
                    left_rewards.append(float(left_reward))
                    right_rewards.append(float(right_reward))
                    dones.append(bool(done))
                    infos.append(info)

                connection.send(
                    (
                        np.stack(next_left_observations).astype(np.float32),
                        np.stack(next_right_observations).astype(np.float32),
                        np.asarray(left_rewards, dtype=np.float32),
                        np.asarray(right_rewards, dtype=np.float32),
                        np.asarray(dones, dtype=np.bool_),
                        infos,
                    )
                )
                continue

            if command == "close":
                connection.close()
                break

            raise ValueError(f"Unknown worker command: {command}")
    finally:
        connection.close()


class ParallelPongVecEnv:
    def __init__(
        self,
        num_envs: int,
        num_workers: int,
        config: PongConfig | None = None,
        seed: int = 7,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.num_envs = num_envs
        self.num_workers = max(1, min(num_workers, num_envs))
        self.config = config or PongConfig()
        self.ctx = mp.get_context("spawn")
        self.processes: list[mp.Process] = []
        self.connections = []
        self.worker_slices: list[slice] = []

        env_indexes = np.array_split(np.arange(num_envs), self.num_workers)
        for worker_index, env_indexes_chunk in enumerate(env_indexes):
            if len(env_indexes_chunk) == 0:
                continue
            worker_seed_base = seed + worker_index * 10_000
            seeds = [worker_seed_base + int(offset) for offset in range(len(env_indexes_chunk))]
            parent_conn, child_conn = self.ctx.Pipe()
            process = self.ctx.Process(
                target=_worker_loop,
                args=(child_conn, asdict(self.config), seeds),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.processes.append(process)
            self.connections.append(parent_conn)
            start = int(env_indexes_chunk[0])
            stop = int(env_indexes_chunk[-1]) + 1
            self.worker_slices.append(slice(start, stop))

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for connection in self.connections:
            connection.send(("reset", None))

        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for connection in self.connections:
            left_observations, right_observations = connection.recv()
            left_parts.append(left_observations)
            right_parts.append(right_observations)

        return np.concatenate(left_parts, axis=0), np.concatenate(right_parts, axis=0)

    def step(
        self,
        left_actions: np.ndarray,
        right_actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
        for env_slice, connection in zip(self.worker_slices, self.connections):
            connection.send(
                (
                    "step",
                    (
                        left_actions[env_slice].astype(np.int64, copy=False),
                        right_actions[env_slice].astype(np.int64, copy=False),
                    ),
                )
            )

        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        left_reward_parts: list[np.ndarray] = []
        right_reward_parts: list[np.ndarray] = []
        done_parts: list[np.ndarray] = []
        infos: list[dict[str, float | int | str]] = []

        for connection in self.connections:
            left_obs, right_obs, left_rewards, right_rewards, dones, worker_infos = connection.recv()
            left_parts.append(left_obs)
            right_parts.append(right_obs)
            left_reward_parts.append(left_rewards)
            right_reward_parts.append(right_rewards)
            done_parts.append(dones)
            infos.extend(worker_infos)

        return (
            np.concatenate(left_parts, axis=0),
            np.concatenate(right_parts, axis=0),
            np.concatenate(left_reward_parts, axis=0),
            np.concatenate(right_reward_parts, axis=0),
            np.concatenate(done_parts, axis=0),
            infos,
        )

    def close(self) -> None:
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
