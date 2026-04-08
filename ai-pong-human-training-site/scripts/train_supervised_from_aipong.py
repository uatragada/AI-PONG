from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from decode_aipong import decode_file, iter_files


class PolicyNet(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, 3)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.backbone(observations))


def load_examples(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    observations: list[list[float]] = []
    actions: list[int] = []

    for file_path in iter_files(path):
        _, records = decode_file(file_path)
        for record in records:
            if not record["visible"]:
                continue
            action = int(record["human_action"])
            if action not in (0, 1, 2):
                continue
            observations.append(record["human_observation"])
            actions.append(action)

    if not observations:
        raise ValueError(f"No usable supervised examples found under {path}")

    return torch.tensor(observations, dtype=torch.float32), torch.tensor(actions, dtype=torch.long)


def export_browser_model(model: PolicyNet, output_path: Path, base_model_path: Path | None, epochs: int) -> None:
    env_config = {
        "width": 900,
        "height": 600,
        "paddle_width": 16,
        "paddle_height": 96,
        "paddle_margin": 36,
        "paddle_speed": 540.0,
        "ball_radius": 9,
        "serve_speed": 360.0,
        "max_ball_speed": 920.0,
        "ball_speedup": 1.045,
        "max_bounce_angle": 1.1344640137963142,
        "dt": 1.0 / 60.0,
        "score_to_win": 7,
        "hit_reward": 0.05,
    }

    if base_model_path and base_model_path.exists():
        base_model = json.loads(base_model_path.read_text(encoding="utf-8"))
        env_config = base_model.get("envConfig", env_config)

    state = model.state_dict()
    payload = {
        "name": "AI Pong human-supervised policy",
        "format": "ai-pong-mlp-json-v1",
        "sourceCheckpoint": "human-supervised",
        "checkpointUpdate": 0,
        "observationSize": 9,
        "actionSize": 3,
        "hiddenSize": 128,
        "actions": ["stay", "up", "down"],
        "side": "relative",
        "deterministicAction": "argmax(logits)",
        "envConfig": env_config,
        "trainingConfig": {
            "algorithm": "supervised-behavior-cloning",
            "epochs": epochs,
        },
        "layers": {
            "backbone0": {
                "weight": state["backbone.0.weight"].detach().cpu().tolist(),
                "bias": state["backbone.0.bias"].detach().cpu().tolist(),
                "activation": "tanh",
            },
            "backbone2": {
                "weight": state["backbone.2.weight"].detach().cpu().tolist(),
                "bias": state["backbone.2.bias"].detach().cpu().tolist(),
                "activation": "tanh",
            },
            "policyHead": {
                "weight": state["policy_head.weight"].detach().cpu().tolist(),
                "bias": state["policy_head.bias"].detach().cpu().tolist(),
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised AI Pong policy from .aipong human logs.")
    parser.add_argument("path", type=Path, help="A .aipong file or directory containing .aipong files.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("web/models/human-supervised-policy.json"))
    parser.add_argument("--base-model", type=Path, default=Path("web/models/right-agent-policy.json"))
    args = parser.parse_args()

    observations, actions = load_examples(args.path)
    dataset = TensorDataset(observations, actions)
    validation_size = max(1, int(len(dataset) * 0.1))
    training_size = len(dataset) - validation_size
    training_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(7),
    )

    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    model = PolicyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_examples = 0

        for batch_observations, batch_actions in train_loader:
            logits = model(batch_observations)
            loss = loss_fn(logits, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * batch_observations.shape[0]
            train_examples += batch_observations.shape[0]

        model.eval()
        correct = 0
        validation_examples = 0
        with torch.no_grad():
            for batch_observations, batch_actions in validation_loader:
                predictions = model(batch_observations).argmax(dim=1)
                correct += int((predictions == batch_actions).sum().item())
                validation_examples += batch_actions.shape[0]

        accuracy = correct / max(validation_examples, 1)
        print(
            f"epoch {epoch:03d} | "
            f"loss {train_loss / max(train_examples, 1):.4f} | "
            f"val action accuracy {accuracy:.3f}"
        )

    export_browser_model(model, args.output, args.base_model, args.epochs)
    print(f"Exported browser model to {args.output}")


if __name__ == "__main__":
    main()
