from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.backbone(observations)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return Categorical(logits=logits), values

    @torch.no_grad()
    def act(self, observation: np.ndarray, device: torch.device, deterministic: bool = False) -> tuple[int, float, float]:
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        distribution, value = self(obs_tensor)
        if deterministic:
            action = torch.argmax(distribution.probs, dim=-1)
        else:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = 0.0
    extended_values = np.append(values, np.array([last_value], dtype=np.float32))

    for index in range(len(rewards) - 1, -1, -1):
        mask = 1.0 - dones[index]
        delta = rewards[index] + gamma * extended_values[index + 1] * mask - extended_values[index]
        next_advantage = delta + gamma * gae_lambda * mask * next_advantage
        advantages[index] = next_advantage

    returns = advantages + values
    return advantages, returns
