from __future__ import annotations

import math

import torch

from ai_pong.env import PongConfig


class TorchPongVecEnv:
    def __init__(
        self,
        num_envs: int,
        config: PongConfig | None = None,
        device: torch.device | str = "cpu",
        seed: int = 7,
    ) -> None:
        self.num_envs = num_envs
        self.config = config or PongConfig()
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device.type)
        self.generator.manual_seed(seed)

        self.width = float(self.config.width)
        self.height = float(self.config.height)
        self.paddle_half_height = self.config.paddle_height / 2.0
        self.ball_radius = float(self.config.ball_radius)
        self.left_front = self.config.paddle_margin + self.config.paddle_width
        self.right_front = self.config.width - self.config.paddle_margin - self.config.paddle_width
        self.left_x = self.config.paddle_margin + self.config.paddle_width / 2.0
        self.right_x = self.config.width - self.config.paddle_margin - self.config.paddle_width / 2.0
        self.action_directions = torch.tensor([0.0, -1.0, 1.0], dtype=torch.float32, device=self.device)

        self.left_score = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.right_score = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.left_hits = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.right_hits = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.rally_hits = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.bounce_angle_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.bounce_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.left_paddle_y = torch.full((self.num_envs,), self.height / 2.0, dtype=torch.float32, device=self.device)
        self.right_paddle_y = torch.full((self.num_envs,), self.height / 2.0, dtype=torch.float32, device=self.device)
        self.ball_x = torch.full((self.num_envs,), self.width / 2.0, dtype=torch.float32, device=self.device)
        self.ball_y = torch.full((self.num_envs,), self.height / 2.0, dtype=torch.float32, device=self.device)
        self.ball_vx = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.ball_vy = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset()

    def _rand(self, count: int) -> torch.Tensor:
        return torch.rand(count, generator=self.generator, device=self.device)

    def _rand_angle(self, count: int) -> torch.Tensor:
        return (self._rand(count) * 2.0 - 1.0) * 0.32

    def _serve_ball(self, mask: torch.Tensor, serve_direction: torch.Tensor | None = None) -> None:
        count = int(mask.sum().item())
        if count == 0:
            return

        angle = self._rand_angle(count)
        if serve_direction is None:
            direction = torch.where(
                self._rand(count) < 0.5,
                torch.full((count,), -1.0, dtype=torch.float32, device=self.device),
                torch.ones(count, dtype=torch.float32, device=self.device),
            )
        else:
            direction = serve_direction.to(dtype=torch.float32, device=self.device)

        self.ball_x[mask] = self.width / 2.0
        self.ball_y[mask] = self.height / 2.0 + (self._rand(count) * 2.0 - 1.0) * (self.height * 0.15)
        self.ball_vx[mask] = direction * self.config.serve_speed * torch.cos(angle)
        self.ball_vy[mask] = self.config.serve_speed * torch.sin(angle)

    def reset(self, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.left_score[mask] = 0
        self.right_score[mask] = 0
        self.steps[mask] = 0
        self.left_hits[mask] = 0
        self.right_hits[mask] = 0
        self.rally_hits[mask] = 0
        self.bounce_angle_sum[mask] = 0.0
        self.bounce_count[mask] = 0
        self.left_paddle_y[mask] = self.height / 2.0
        self.right_paddle_y[mask] = self.height / 2.0
        self._serve_ball(mask)
        return self.get_observation("left"), self.get_observation("right")

    def get_observation(self, side: str) -> torch.Tensor:
        width = self.width
        height = self.height
        max_speed = float(self.config.max_ball_speed)
        score_norm = float(self.config.score_to_win)

        if side == "left":
            self_y = self.left_paddle_y
            opp_y = self.right_paddle_y
            ball_x = self.ball_x
            ball_vx = self.ball_vx
            score_for = self.left_score.to(torch.float32)
            score_against = self.right_score.to(torch.float32)
        else:
            self_y = self.right_paddle_y
            opp_y = self.left_paddle_y
            ball_x = self.width - self.ball_x
            ball_vx = -self.ball_vx
            score_for = self.right_score.to(torch.float32)
            score_against = self.left_score.to(torch.float32)

        return torch.stack(
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
            dim=1,
        )

    def _reflect(self, mask: torch.Tensor, paddle_y: torch.Tensor, direction: float) -> None:
        if not torch.any(mask):
            return
        offset = (self.ball_y[mask] - paddle_y[mask]) / self.paddle_half_height
        offset = torch.clamp(offset, -1.0, 1.0)
        speed = torch.sqrt(self.ball_vx[mask].square() + self.ball_vy[mask].square())
        speed = torch.clamp(speed * self.config.ball_speedup, max=self.config.max_ball_speed)
        angle = offset * self.config.max_bounce_angle
        self.ball_vx[mask] = direction * speed * torch.cos(angle)
        self.ball_vy[mask] = speed * torch.sin(angle)
        self.bounce_angle_sum[mask] += torch.abs(angle)
        self.bounce_count[mask] += 1

    def step(
        self,
        left_actions: torch.Tensor,
        right_actions: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        left_actions = left_actions.to(device=self.device, dtype=torch.int64)
        right_actions = right_actions.to(device=self.device, dtype=torch.int64)

        self.steps += 1
        left_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        right_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.left_paddle_y += self.action_directions[left_actions] * self.config.paddle_speed * self.config.dt
        self.right_paddle_y += self.action_directions[right_actions] * self.config.paddle_speed * self.config.dt
        self.left_paddle_y.clamp_(self.paddle_half_height, self.height - self.paddle_half_height)
        self.right_paddle_y.clamp_(self.paddle_half_height, self.height - self.paddle_half_height)

        self.ball_x += self.ball_vx * self.config.dt
        self.ball_y += self.ball_vy * self.config.dt

        half_height = self.height / 2.0
        left_alignment = torch.where(
            self.ball_vx < 0,
            1.0 - torch.clamp(torch.abs(self.ball_y - self.left_paddle_y) / half_height, max=1.0),
            torch.zeros_like(self.ball_y),
        )
        right_alignment = torch.where(
            self.ball_vx > 0,
            1.0 - torch.clamp(torch.abs(self.ball_y - self.right_paddle_y) / half_height, max=1.0),
            torch.zeros_like(self.ball_y),
        )
        shaping_reward = 0.0025 * (left_alignment - right_alignment)
        left_rewards += shaping_reward
        right_rewards -= shaping_reward

        top_hit = self.ball_y - self.ball_radius <= 0.0
        bottom_hit = self.ball_y + self.ball_radius >= self.height
        self.ball_y = torch.where(top_hit, torch.full_like(self.ball_y, self.ball_radius), self.ball_y)
        self.ball_y = torch.where(bottom_hit, torch.full_like(self.ball_y, self.height - self.ball_radius), self.ball_y)
        self.ball_vy = torch.where(top_hit, torch.abs(self.ball_vy), self.ball_vy)
        self.ball_vy = torch.where(bottom_hit, -torch.abs(self.ball_vy), self.ball_vy)

        left_contact = (
            (self.ball_vx < 0)
            & (self.ball_x - self.ball_radius <= self.left_front)
            & (torch.abs(self.ball_y - self.left_paddle_y) <= self.paddle_half_height)
        )
        if torch.any(left_contact):
            self.ball_x[left_contact] = self.left_front + self.ball_radius
            self._reflect(left_contact, self.left_paddle_y, direction=1.0)
            self.left_hits[left_contact] += 1
            self.rally_hits[left_contact] += 1
            left_rewards[left_contact] += self.config.hit_reward
            right_rewards[left_contact] -= self.config.hit_reward

        right_contact = (
            (self.ball_vx > 0)
            & (self.ball_x + self.ball_radius >= self.right_front)
            & (torch.abs(self.ball_y - self.right_paddle_y) <= self.paddle_half_height)
        )
        if torch.any(right_contact):
            self.ball_x[right_contact] = self.right_front - self.ball_radius
            self._reflect(right_contact, self.right_paddle_y, direction=-1.0)
            self.right_hits[right_contact] += 1
            self.rally_hits[right_contact] += 1
            right_rewards[right_contact] += self.config.hit_reward
            left_rewards[right_contact] -= self.config.hit_reward

        right_scored = self.ball_x < -self.ball_radius
        left_scored = self.ball_x > self.width + self.ball_radius
        scored_mask = left_scored | right_scored
        self.right_score[right_scored] += 1
        self.left_score[left_scored] += 1
        left_rewards[right_scored] -= 1.0
        right_rewards[right_scored] += 1.0
        left_rewards[left_scored] += 1.0
        right_rewards[left_scored] -= 1.0

        done = (
            (self.left_score >= self.config.score_to_win)
            | (self.right_score >= self.config.score_to_win)
            | (self.steps >= self.config.max_steps)
        )

        winner = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        winner = torch.where(done & (self.left_score > self.right_score), torch.ones_like(winner), winner)
        winner = torch.where(done & (self.right_score > self.left_score), -torch.ones_like(winner), winner)

        terminal_left_score = self.left_score.clone()
        terminal_right_score = self.right_score.clone()
        terminal_left_hits = self.left_hits.clone()
        terminal_right_hits = self.right_hits.clone()
        terminal_rally_hits = self.rally_hits.clone()
        terminal_bounce_angle_sum = self.bounce_angle_sum.clone()
        terminal_bounce_count = self.bounce_count.clone()

        continue_mask = scored_mask & (~done)
        if torch.any(continue_mask):
            serve_direction = torch.where(
                left_scored[continue_mask],
                torch.ones(int(continue_mask.sum().item()), dtype=torch.float32, device=self.device),
                -torch.ones(int(continue_mask.sum().item()), dtype=torch.float32, device=self.device),
            )
            self._serve_ball(continue_mask, serve_direction=serve_direction)
            self.rally_hits[continue_mask] = 0
            self.bounce_angle_sum[continue_mask] = 0.0
            self.bounce_count[continue_mask] = 0

        if torch.any(done):
            self.reset(done)

        observations = (self.get_observation("left"), self.get_observation("right"))
        info = {
            "terminal_left_score": terminal_left_score,
            "terminal_right_score": terminal_right_score,
            "terminal_left_hits": terminal_left_hits,
            "terminal_right_hits": terminal_right_hits,
            "terminal_rally_hits": terminal_rally_hits,
            "terminal_bounce_angle_sum": terminal_bounce_angle_sum,
            "terminal_bounce_count": terminal_bounce_count,
            "winner": winner,
        }
        return observations, (left_rewards, right_rewards), done, info
