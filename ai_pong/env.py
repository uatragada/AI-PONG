from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class PongConfig:
    width: int = 900
    height: int = 600
    paddle_width: int = 16
    paddle_height: int = 96
    paddle_margin: int = 36
    paddle_speed: float = 540.0
    ball_radius: int = 9
    serve_speed: float = 360.0
    max_ball_speed: float = 920.0
    ball_speedup: float = 1.045
    max_bounce_angle: float = math.radians(65.0)
    dt: float = 1.0 / 60.0
    score_to_win: int = 7
    max_steps: int = 100000
    hit_reward: float = 0.05


class PongEnv:
    action_to_direction = {0: 0.0, 1: -1.0, 2: 1.0}

    def __init__(self, config: PongConfig | None = None, seed: int = 7) -> None:
        self.config = config or PongConfig()
        self.rng = random.Random(seed)
        self.reset()

    @property
    def left_x(self) -> float:
        return self.config.paddle_margin + self.config.paddle_width / 2

    @property
    def right_x(self) -> float:
        return self.config.width - self.config.paddle_margin - self.config.paddle_width / 2

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.left_score = 0
        self.right_score = 0
        self.steps = 0
        self.left_hits = 0
        self.right_hits = 0
        self.left_paddle_y = self.config.height / 2
        self.right_paddle_y = self.config.height / 2
        self._reset_ball()
        return self.get_observation("left"), self.get_observation("right")

    def _reset_ball(self, serve_direction: int | None = None) -> None:
        angle = self.rng.uniform(-0.32, 0.32)
        direction = serve_direction if serve_direction is not None else self.rng.choice([-1, 1])
        self.ball_x = self.config.width / 2
        self.ball_y = self.config.height / 2 + self.rng.uniform(-self.config.height * 0.15, self.config.height * 0.15)
        self.ball_vx = direction * self.config.serve_speed * math.cos(angle)
        self.ball_vy = self.config.serve_speed * math.sin(angle)

    def get_observation(self, side: str) -> np.ndarray:
        width = float(self.config.width)
        height = float(self.config.height)
        max_speed = float(self.config.max_ball_speed)
        score_norm = float(self.config.score_to_win)

        if side == "left":
            self_y = self.left_paddle_y
            opp_y = self.right_paddle_y
            ball_x = self.ball_x
            ball_vx = self.ball_vx
            score_for = self.left_score
            score_against = self.right_score
        else:
            self_y = self.right_paddle_y
            opp_y = self.left_paddle_y
            ball_x = self.config.width - self.ball_x
            ball_vx = -self.ball_vx
            score_for = self.right_score
            score_against = self.left_score

        obs = np.array(
            [
                self_y / height,
                opp_y / height,
                ball_x / width,
                self.ball_y / height,
                ball_vx / max_speed,
                self.ball_vy / max_speed,
                (self.ball_y - self_y) / height,
                (self.ball_y - opp_y) / height,
                (score_for - score_against) / score_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def _move_paddle(self, paddle_y: float, action: int) -> float:
        direction = self.action_to_direction.get(int(action), 0.0)
        paddle_y += direction * self.config.paddle_speed * self.config.dt
        half = self.config.paddle_height / 2
        return max(half, min(self.config.height - half, paddle_y))

    def _reflect_from_paddle(self, paddle_y: float, direction: int) -> None:
        offset = (self.ball_y - paddle_y) / (self.config.paddle_height / 2)
        offset = max(-1.0, min(1.0, offset))
        speed = min(math.hypot(self.ball_vx, self.ball_vy) * self.config.ball_speedup, self.config.max_ball_speed)
        angle = offset * self.config.max_bounce_angle
        self.ball_vx = direction * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)

    def _defensive_alignment(self) -> tuple[float, float]:
        half_height = self.config.height / 2
        left_alignment = 0.0
        right_alignment = 0.0

        if self.ball_vx < 0:
            left_alignment = 1.0 - min(abs(self.ball_y - self.left_paddle_y) / half_height, 1.0)
        elif self.ball_vx > 0:
            right_alignment = 1.0 - min(abs(self.ball_y - self.right_paddle_y) / half_height, 1.0)

        return left_alignment, right_alignment

    def step(
        self,
        left_action: int,
        right_action: int,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[float, float], bool, dict[str, float | int | str]]:
        self.steps += 1
        left_reward = 0.0
        right_reward = 0.0

        self.left_paddle_y = self._move_paddle(self.left_paddle_y, left_action)
        self.right_paddle_y = self._move_paddle(self.right_paddle_y, right_action)

        self.ball_x += self.ball_vx * self.config.dt
        self.ball_y += self.ball_vy * self.config.dt

        left_alignment, right_alignment = self._defensive_alignment()
        shaping_reward = 0.0025 * (left_alignment - right_alignment)
        left_reward += shaping_reward
        right_reward -= shaping_reward

        if self.ball_y - self.config.ball_radius <= 0:
            self.ball_y = self.config.ball_radius
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y + self.config.ball_radius >= self.config.height:
            self.ball_y = self.config.height - self.config.ball_radius
            self.ball_vy = -abs(self.ball_vy)

        left_front = self.left_x + self.config.paddle_width / 2
        right_front = self.right_x - self.config.paddle_width / 2
        half_paddle = self.config.paddle_height / 2

        if self.ball_vx < 0 and self.ball_x - self.config.ball_radius <= left_front:
            if abs(self.ball_y - self.left_paddle_y) <= half_paddle:
                self.ball_x = left_front + self.config.ball_radius
                self._reflect_from_paddle(self.left_paddle_y, direction=1)
                self.left_hits += 1
                left_reward += self.config.hit_reward
                right_reward -= self.config.hit_reward

        if self.ball_vx > 0 and self.ball_x + self.config.ball_radius >= right_front:
            if abs(self.ball_y - self.right_paddle_y) <= half_paddle:
                self.ball_x = right_front - self.config.ball_radius
                self._reflect_from_paddle(self.right_paddle_y, direction=-1)
                self.right_hits += 1
                right_reward += self.config.hit_reward
                left_reward -= self.config.hit_reward

        scored = None
        if self.ball_x < -self.config.ball_radius:
            self.right_score += 1
            left_reward -= 1.0
            right_reward += 1.0
            scored = "right"
        elif self.ball_x > self.config.width + self.config.ball_radius:
            self.left_score += 1
            left_reward += 1.0
            right_reward -= 1.0
            scored = "left"

        done = False
        if scored is not None:
            done = self.left_score >= self.config.score_to_win or self.right_score >= self.config.score_to_win
            if not done:
                serve_direction = 1 if scored == "left" else -1
                self._reset_ball(serve_direction=serve_direction)

        if self.steps >= self.config.max_steps:
            done = True

        info: dict[str, float | int | str] = {
            "left_score": self.left_score,
            "right_score": self.right_score,
            "left_hits": self.left_hits,
            "right_hits": self.right_hits,
        }
        if scored is not None:
            info["scored"] = scored
        if done:
            if self.left_score > self.right_score:
                info["winner"] = "left"
            elif self.right_score > self.left_score:
                info["winner"] = "right"
            else:
                info["winner"] = "draw"

        observations = (self.get_observation("left"), self.get_observation("right"))
        rewards = (left_reward, right_reward)
        return observations, rewards, done, info
