from __future__ import annotations

import importlib.util
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import optim

from ai_pong.env import PongConfig, PongEnv
from ai_pong.models import ActorCritic
from ai_pong.torch_env import TorchPongVecEnv


@dataclass
class TrainingConfig:
    updates: int
    steps_per_update: int
    ppo_epochs: int
    batch_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    entropy_coef: float
    value_coef: float
    hidden_size: int
    num_envs: int
    num_workers: int
    save_every: int
    checkpoint_dir: str
    seed: int
    device: str
    amp: bool
    torch_compile: bool


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_backend(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def maybe_compile(module: ActorCritic, enabled: bool) -> ActorCritic:
    if not enabled or not hasattr(torch, "compile"):
        return module
    if importlib.util.find_spec("triton") is None:
        return module
    return torch.compile(module, mode="max-autotune")  # type: ignore[return-value]


def resolve_num_workers(requested_workers: int, num_envs: int) -> int:
    if requested_workers > 0:
        return min(requested_workers, num_envs)
    cpu_count = os.cpu_count() or 1
    return max(1, min(num_envs, cpu_count))


def create_rollout_tensors(
    steps_per_update: int,
    num_envs: int,
    observation_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "observations": torch.empty((steps_per_update, num_envs, observation_size), dtype=torch.float32, device=device),
        "actions": torch.empty((steps_per_update, num_envs), dtype=torch.int64, device=device),
        "log_probs": torch.empty((steps_per_update, num_envs), dtype=torch.float32, device=device),
        "values": torch.empty((steps_per_update, num_envs), dtype=torch.float32, device=device),
        "rewards": torch.empty((steps_per_update, num_envs), dtype=torch.float32, device=device),
        "dones": torch.empty((steps_per_update, num_envs), dtype=torch.float32, device=device),
    }


@torch.no_grad()
def act_batch(
    agent: ActorCritic,
    observations: torch.Tensor,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.amp.autocast(device_type=observations.device.type, enabled=amp_enabled):
        distribution, values = agent(observations)
    actions = distribution.sample()
    log_probs = distribution.log_prob(actions)
    return actions, log_probs.float(), values.float()


@torch.no_grad()
def value_batch(agent: ActorCritic, observations: torch.Tensor, amp_enabled: bool) -> torch.Tensor:
    with torch.amp.autocast(device_type=observations.device.type, enabled=amp_enabled):
        _, values = agent(observations)
    return values.float()


def optimize_agent(
    agent: ActorCritic,
    optimizer: optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    config: TrainingConfig,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    num_envs = config.num_envs
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)
    next_values = last_values
    next_non_terminal = 1.0 - dones[-1]

    for step_index in range(steps - 1, -1, -1):
        if step_index < steps - 1:
            next_values = values[step_index + 1]
            next_non_terminal = 1.0 - dones[step_index]
        delta = rewards[step_index] + config.gamma * next_values * next_non_terminal - values[step_index]
        next_advantage = delta + config.gamma * config.gae_lambda * next_non_terminal * next_advantage
        advantages[step_index] = next_advantage
    returns = advantages + values

    flat_observations = observations.reshape(steps * num_envs, -1)
    flat_actions = actions.reshape(steps * num_envs)
    flat_old_log_probs = old_log_probs.reshape(steps * num_envs)
    advantages_tensor = advantages.reshape(-1)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    returns_tensor = returns.reshape(-1)

    actor_losses: list[float] = []
    critic_losses: list[float] = []
    entropies: list[float] = []
    sample_count = flat_observations.shape[0]
    amp_enabled = config.amp and flat_observations.device.type == "cuda"

    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(sample_count, device=flat_observations.device)
        for start in range(0, sample_count, config.batch_size):
            indexes = permutation[start : start + config.batch_size]
            with torch.amp.autocast(device_type=flat_observations.device.type, enabled=amp_enabled):
                distribution, predicted_values = agent(flat_observations[indexes])
                new_log_probs = distribution.log_prob(flat_actions[indexes])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_log_probs - flat_old_log_probs[indexes])
                unclipped = ratio * advantages_tensor[indexes]
                clipped = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages_tensor[indexes]
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = torch.nn.functional.mse_loss(predicted_values.float(), returns_tensor[indexes])
                loss = actor_loss + config.value_coef * critic_loss - config.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            actor_losses.append(float(actor_loss.item()))
            critic_losses.append(float(critic_loss.item()))
            entropies.append(float(entropy.item()))

    return {
        "actor_loss": float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "entropy": float(np.mean(entropies)),
        "mean_return": float(returns.mean().item()),
    }


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(
    path: str,
    left_agent: ActorCritic,
    right_agent: ActorCritic,
    env: PongEnv,
    config: TrainingConfig,
    update: int,
) -> None:
    checkpoint = {
        "update": update,
        "timestamp": time.time(),
        "env_config": asdict(env.config),
        "training_config": asdict(config),
        "observation_size": left_agent.observation_size,
        "action_size": left_agent.action_size,
        "hidden_size": left_agent.hidden_size,
        "left_state_dict": left_agent.state_dict(),
        "right_state_dict": right_agent.state_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[ActorCritic, ActorCritic, PongConfig, dict]:
    checkpoint = torch.load(path, map_location=device)
    env_config = PongConfig(**checkpoint["env_config"])
    observation_size = int(checkpoint["observation_size"])
    action_size = int(checkpoint["action_size"])
    hidden_size = int(checkpoint["hidden_size"])

    left_agent = ActorCritic(observation_size, action_size, hidden_size=hidden_size).to(device)
    right_agent = ActorCritic(observation_size, action_size, hidden_size=hidden_size).to(device)
    left_agent.load_state_dict(checkpoint["left_state_dict"])
    right_agent.load_state_dict(checkpoint["right_state_dict"])
    left_agent.eval()
    right_agent.eval()
    return left_agent, right_agent, env_config, checkpoint


def train_self_play(args) -> None:
    config = TrainingConfig(
        updates=args.updates,
        steps_per_update=args.steps_per_update,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        hidden_size=args.hidden_size,
        num_envs=args.num_envs,
        num_workers=resolve_num_workers(args.num_workers, args.num_envs),
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        torch_compile=args.torch_compile,
    )

    set_seed(config.seed)
    device = resolve_device(config.device)
    configure_torch_backend(device)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    sample_env = PongEnv(seed=config.seed)
    observation_size = len(sample_env.get_observation("left"))
    action_size = 3
    amp_enabled = config.amp and device.type == "cuda"

    left_agent = ActorCritic(observation_size, action_size, hidden_size=config.hidden_size).to(device)
    right_agent = ActorCritic(observation_size, action_size, hidden_size=config.hidden_size).to(device)
    compiled_left_agent = maybe_compile(left_agent, config.torch_compile and device.type == "cuda")
    compiled_right_agent = maybe_compile(right_agent, config.torch_compile and device.type == "cuda")
    left_optimizer = optim.Adam(left_agent.parameters(), lr=config.learning_rate, foreach=device.type == "cuda")
    right_optimizer = optim.Adam(right_agent.parameters(), lr=config.learning_rate, foreach=device.type == "cuda")
    left_scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    right_scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    vec_env = TorchPongVecEnv(
        num_envs=config.num_envs,
        config=sample_env.config,
        device=device,
        seed=config.seed,
    )
    left_observations, right_observations = vec_env.reset()
    recent_winners: deque[str] = deque(maxlen=40)
    recent_margins: deque[int] = deque(maxlen=40)
    recent_hits: deque[int] = deque(maxlen=40)
    recent_rally_lengths: deque[float] = deque(maxlen=80)
    recent_bounce_angles: deque[float] = deque(maxlen=80)

    for update in range(1, config.updates + 1):
        maybe_synchronize(device)
        update_start_time = time.perf_counter()
        left_storage = create_rollout_tensors(config.steps_per_update, config.num_envs, observation_size, device)
        right_storage = create_rollout_tensors(config.steps_per_update, config.num_envs, observation_size, device)
        final_dones = torch.zeros(config.num_envs, dtype=torch.float32, device=device)

        for step_index in range(config.steps_per_update):
            left_storage["observations"][step_index].copy_(left_observations)
            right_storage["observations"][step_index].copy_(right_observations)

            left_actions, left_log_probs, left_values = act_batch(compiled_left_agent, left_observations, amp_enabled)
            right_actions, right_log_probs, right_values = act_batch(compiled_right_agent, right_observations, amp_enabled)

            left_storage["actions"][step_index].copy_(left_actions)
            left_storage["log_probs"][step_index].copy_(left_log_probs)
            left_storage["values"][step_index].copy_(left_values)
            right_storage["actions"][step_index].copy_(right_actions)
            right_storage["log_probs"][step_index].copy_(right_log_probs)
            right_storage["values"][step_index].copy_(right_values)

            (left_observations, right_observations), (left_rewards, right_rewards), dones, info = vec_env.step(
                left_actions,
                right_actions,
            )

            left_storage["rewards"][step_index].copy_(left_rewards)
            right_storage["rewards"][step_index].copy_(right_rewards)
            done_tensor = dones.to(dtype=torch.float32)
            left_storage["dones"][step_index].copy_(done_tensor)
            right_storage["dones"][step_index].copy_(done_tensor)

            if torch.any(dones):
                done_indexes = torch.nonzero(dones, as_tuple=False).flatten().detach().cpu().tolist()
                terminal_left_score = info["terminal_left_score"].detach().cpu()
                terminal_right_score = info["terminal_right_score"].detach().cpu()
                terminal_left_hits = info["terminal_left_hits"].detach().cpu()
                terminal_right_hits = info["terminal_right_hits"].detach().cpu()
                terminal_rally_hits = info["terminal_rally_hits"].detach().cpu()
                terminal_bounce_angle_sum = info["terminal_bounce_angle_sum"].detach().cpu()
                terminal_bounce_count = info["terminal_bounce_count"].detach().cpu()
                winners = info["winner"].detach().cpu()
                for env_index in done_indexes:
                    winner_code = int(winners[env_index].item())
                    recent_winners.append("left" if winner_code > 0 else "right" if winner_code < 0 else "draw")
                    recent_margins.append(int(terminal_left_score[env_index].item()) - int(terminal_right_score[env_index].item()))
                    recent_hits.append(int(terminal_left_hits[env_index].item()) + int(terminal_right_hits[env_index].item()))
                    recent_rally_lengths.append(float(terminal_rally_hits[env_index].item()))
                    bounce_count = int(terminal_bounce_count[env_index].item())
                    if bounce_count > 0:
                        recent_bounce_angles.append(float(terminal_bounce_angle_sum[env_index].item()) / bounce_count)

            final_dones = done_tensor

        left_last_values = value_batch(compiled_left_agent, left_observations, amp_enabled) * (1.0 - final_dones)
        right_last_values = value_batch(compiled_right_agent, right_observations, amp_enabled) * (1.0 - final_dones)

        left_metrics = optimize_agent(
            agent=compiled_left_agent,
            optimizer=left_optimizer,
            observations=left_storage["observations"],
            actions=left_storage["actions"],
            old_log_probs=left_storage["log_probs"],
            values=left_storage["values"],
            rewards=left_storage["rewards"],
            dones=left_storage["dones"],
            last_values=left_last_values,
            config=config,
            scaler=left_scaler,
        )
        right_metrics = optimize_agent(
            agent=compiled_right_agent,
            optimizer=right_optimizer,
            observations=right_storage["observations"],
            actions=right_storage["actions"],
            old_log_probs=right_storage["log_probs"],
            values=right_storage["values"],
            rewards=right_storage["rewards"],
            dones=right_storage["dones"],
            last_values=right_last_values,
            config=config,
            scaler=right_scaler,
        )

        latest_path = os.path.join(config.checkpoint_dir, "selfplay_latest.pt")
        save_checkpoint(latest_path, left_agent, right_agent, sample_env, config, update)
        if config.save_every > 0 and update % config.save_every == 0:
            snapshot_path = os.path.join(config.checkpoint_dir, f"selfplay_update_{update:04d}.pt")
            save_checkpoint(snapshot_path, left_agent, right_agent, sample_env, config, update)

        if recent_winners:
            left_win_rate = sum(1 for winner in recent_winners if winner == "left") / len(recent_winners)
            right_win_rate = sum(1 for winner in recent_winners if winner == "right") / len(recent_winners)
            avg_hits = float(np.mean(recent_hits))
            avg_margin = float(np.mean(recent_margins))
        else:
            left_win_rate = 0.0
            right_win_rate = 0.0
            avg_hits = 0.0
            avg_margin = 0.0
        avg_rally_length = float(np.mean(recent_rally_lengths)) if recent_rally_lengths else 0.0
        avg_bounce_angle = float(np.mean(recent_bounce_angles)) if recent_bounce_angles else 0.0

        maybe_synchronize(device)
        update_elapsed = time.perf_counter() - update_start_time
        print(
            f"update {update:04d} | "
            f"envs {config.num_envs} | sim {device.type} | "
            f"left win {left_win_rate:.2f} | right win {right_win_rate:.2f} | "
            f"avg margin {avg_margin:+.2f} | avg hits {avg_hits:.1f} | "
            f"avg rally {avg_rally_length:.1f} | avg angle {avg_bounce_angle:.3f} | "
            f"L entropy {left_metrics['entropy']:.3f} | R entropy {right_metrics['entropy']:.3f} | "
            f"time {update_elapsed:.2f}s"
        )

    print(f"Training complete. Latest checkpoint: {os.path.join(config.checkpoint_dir, 'selfplay_latest.pt')}")
