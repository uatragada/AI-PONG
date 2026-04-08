from __future__ import annotations

import os

import pygame
import torch

from ai_pong.env import PongEnv
from ai_pong.train import load_checkpoint, resolve_device


def watch_match(
    checkpoint_path: str,
    fps: int = 60,
    episodes: int = 5,
    deterministic: bool = True,
    max_frames: int | None = None,
) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device("cpu")
    left_agent, right_agent, env_config, checkpoint = load_checkpoint(checkpoint_path, device=device)
    env = PongEnv(config=env_config, seed=checkpoint.get("training_config", {}).get("seed", 7))
    left_observation, right_observation = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((env_config.width, env_config.height))
    pygame.display.set_caption("AI Pong Self-Play")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24)
    small_font = pygame.font.SysFont("consolas", 16)

    frames = 0
    episode_index = 0
    running = True

    while running and episode_index < episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        with torch.no_grad():
            left_action, _, _ = left_agent.act(left_observation, device=device, deterministic=deterministic)
            right_action, _, _ = right_agent.act(right_observation, device=device, deterministic=deterministic)

        (left_observation, right_observation), _, done, info = env.step(left_action, right_action)

        screen.fill((16, 18, 24))
        pygame.draw.line(screen, (60, 66, 78), (env_config.width // 2, 0), (env_config.width // 2, env_config.height), 3)

        left_rect = pygame.Rect(
            int(env.left_x - env_config.paddle_width / 2),
            int(env.left_paddle_y - env_config.paddle_height / 2),
            env_config.paddle_width,
            env_config.paddle_height,
        )
        right_rect = pygame.Rect(
            int(env.right_x - env_config.paddle_width / 2),
            int(env.right_paddle_y - env_config.paddle_height / 2),
            env_config.paddle_width,
            env_config.paddle_height,
        )
        ball_rect = pygame.Rect(
            int(env.ball_x - env_config.ball_radius),
            int(env.ball_y - env_config.ball_radius),
            env_config.ball_radius * 2,
            env_config.ball_radius * 2,
        )

        pygame.draw.rect(screen, (237, 242, 244), left_rect, border_radius=4)
        pygame.draw.rect(screen, (237, 242, 244), right_rect, border_radius=4)
        pygame.draw.ellipse(screen, (255, 190, 92), ball_rect)

        score_text = font.render(f"{env.left_score} : {env.right_score}", True, (237, 242, 244))
        status_text = small_font.render(
            f"episode {episode_index + 1}/{episodes} | checkpoint update {checkpoint.get('update', '?')}",
            True,
            (168, 176, 190),
        )
        hit_text = small_font.render(
            f"hits L:{env.left_hits} R:{env.right_hits} | deterministic: {deterministic}",
            True,
            (168, 176, 190),
        )

        screen.blit(score_text, score_text.get_rect(center=(env_config.width // 2, 40)))
        screen.blit(status_text, (20, env_config.height - 44))
        screen.blit(hit_text, (20, env_config.height - 24))
        pygame.display.flip()

        if fps > 0:
            clock.tick(fps)
        frames += 1

        if done:
            episode_index += 1
            if episode_index < episodes:
                left_observation, right_observation = env.reset()

        if max_frames is not None and frames >= max_frames:
            running = False

    pygame.quit()
