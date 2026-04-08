from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MAGIC = b"AIPONG1\x00"
HEADER_BYTES = 64
RECORD_BYTES = 48
OBS_SCALE = 32767.0
REWARD_SCALE = 1000.0
HEADER_FORMAT = "<HHHHIIIHHHHHHHHI"


@dataclass
class TrajectoryHeader:
    sample_count: int
    checkpoint_update: int
    duration_ms: int
    final_left_score: int
    final_right_score: int
    human_hits: int
    bot_hits: int
    max_rally: int
    input_changes: int
    visible_samples: int
    flags: int
    chunk_index: int


@dataclass
class LoadStats:
    files_seen: int = 0
    files_loaded: int = 0
    records_seen: int = 0
    records_used: int = 0
    invisible_records: int = 0
    invalid_records: int = 0
    action_counts: Counter[int] | None = None

    def __post_init__(self) -> None:
        if self.action_counts is None:
            self.action_counts = Counter()


class PolicyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(9, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.policy = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(self.backbone(x))


def resolve_path(raw_path: str | os.PathLike[str], base_dir: Path) -> Path:
    expanded = os.path.expandvars(str(raw_path))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf8") as file:
        config = json.load(file)
    return config


def iter_trajectory_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix == ".aipong":
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob("*.aipong"))


def parse_header(data: bytes, source: Path) -> TrajectoryHeader:
    if len(data) < HEADER_BYTES:
        raise ValueError(f"{source} is smaller than the 64-byte header")
    if data[:8] != MAGIC:
        raise ValueError(f"{source} has an invalid AIPONG magic header")

    (
        version,
        header_bytes,
        record_bytes,
        sample_hz,
        sample_count,
        checkpoint_update,
        duration_ms,
        final_left_score,
        final_right_score,
        human_hits,
        bot_hits,
        max_rally,
        input_changes,
        visible_samples,
        flags,
        chunk_index,
    ) = struct.unpack_from(HEADER_FORMAT, data, 8)

    if version != 1:
        raise ValueError(f"{source} uses unsupported format version {version}")
    if header_bytes != HEADER_BYTES:
        raise ValueError(f"{source} has unexpected header size {header_bytes}")
    if record_bytes != RECORD_BYTES:
        raise ValueError(f"{source} has unexpected record size {record_bytes}")
    if sample_hz != 10:
        raise ValueError(f"{source} has unexpected sample rate {sample_hz}")

    expected_bytes = header_bytes + sample_count * record_bytes
    if len(data) != expected_bytes:
        raise ValueError(f"{source} expected {expected_bytes} bytes, got {len(data)}")

    return TrajectoryHeader(
        sample_count=sample_count,
        checkpoint_update=checkpoint_update,
        duration_ms=duration_ms,
        final_left_score=final_left_score,
        final_right_score=final_right_score,
        human_hits=human_hits,
        bot_hits=bot_hits,
        max_rally=max_rally,
        input_changes=input_changes,
        visible_samples=visible_samples,
        flags=flags,
        chunk_index=chunk_index,
    )


def parse_human_example(data: bytes, offset: int) -> tuple[list[float], int, bool]:
    obs = [value / OBS_SCALE for value in struct.unpack_from("<9h", data, offset)]
    action = struct.unpack_from("<B", data, offset + 36)[0]
    visible = struct.unpack_from("<B", data, offset + 47)[0] == 1
    return obs, action, visible


def load_examples(data_dir: Path, max_samples: int, seed: int) -> tuple[list[tuple[list[float], int]], LoadStats]:
    rng = random.Random(seed)
    files = iter_trajectory_files(data_dir)
    examples: list[tuple[list[float], int]] = []
    stats = LoadStats(files_seen=len(files))
    seen_usable = 0

    for file_path in files:
        try:
            data = file_path.read_bytes()
            header = parse_header(data, file_path)
        except ValueError as error:
            stats.invalid_records += 1
            print(f"Skipping invalid chunk: {error}", file=sys.stderr)
            continue

        stats.files_loaded += 1
        for index in range(header.sample_count):
            offset = HEADER_BYTES + index * RECORD_BYTES
            stats.records_seen += 1
            obs, action, visible = parse_human_example(data, offset)
            if not visible:
                stats.invisible_records += 1
                continue
            if action not in (0, 1, 2):
                stats.invalid_records += 1
                continue

            stats.records_used += 1
            stats.action_counts[action] += 1
            seen_usable += 1

            if len(examples) < max_samples:
                examples.append((obs, action))
                continue

            replacement_index = rng.randrange(seen_usable)
            if replacement_index < max_samples:
                examples[replacement_index] = (obs, action)

    return examples, stats


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def split_examples(
    examples: list[tuple[list[float], int]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[tuple[list[float], int]], list[tuple[list[float], int]]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    validation = shuffled[:validation_count]
    training = shuffled[validation_count:]
    if not training:
        training, validation = validation, []
    return training, validation


def tensor_dataset(examples: list[tuple[list[float], int]], device: torch.device) -> TensorDataset:
    observations = torch.tensor([obs for obs, _ in examples], dtype=torch.float32, device=device)
    actions = torch.tensor([action for _, action in examples], dtype=torch.long, device=device)
    return TensorDataset(observations, actions)


def compute_class_weights(
    examples: list[tuple[list[float], int]],
    device: torch.device,
    power: float,
) -> torch.Tensor:
    counts = Counter(action for _, action in examples)
    total = sum(counts.values())
    weights = []
    for action in (0, 1, 2):
        count = counts.get(action, 0)
        if count == 0:
            weights.append(0.0)
            continue
        weights.append((total / (3 * count)) ** power)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_policy(
    examples: list[tuple[list[float], int]],
    config: dict[str, Any],
) -> tuple[PolicyNet, dict[str, Any]]:
    seed = int(config["training"].get("seed", 7))
    torch.manual_seed(seed)
    random.seed(seed)

    device = select_device(str(config["training"].get("device", "auto")))
    validation_fraction = float(config["training"].get("validation_fraction", 0.1))
    train_examples, validation_examples = split_examples(examples, validation_fraction, seed)

    model = PolicyNet().to(device)
    train_dataset = tensor_dataset(train_examples, device)
    val_dataset = tensor_dataset(validation_examples, device) if validation_examples else None

    batch_size = int(config["training"].get("batch_size", 512))
    epochs = int(config["training"].get("epochs", 12))
    learning_rate = float(config["training"].get("learning_rate", 0.0003))
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_balance = bool(config["training"].get("class_balance", True))
    class_weights = None
    if class_balance:
        class_weights = compute_class_weights(
            train_examples,
            device,
            power=float(config["training"].get("class_weight_power", 0.5)),
        )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for obs_batch, action_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(obs_batch)
            loss = loss_fn(logits, action_batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * obs_batch.shape[0]
            total_examples += obs_batch.shape[0]

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, total_examples),
        }
        print(f"epoch {epoch + 1:03d}: train_loss={epoch_metrics['train_loss']:.4f}")
        history.append(epoch_metrics)

    metrics = evaluate_policy(model, val_dataset, device) if val_dataset else {
        "validation_accuracy": 0.0,
        "prediction_entropy": 0.0,
        "prediction_action_fractions": {"0": 0.0, "1": 0.0, "2": 0.0},
        "validation_examples": 0,
    }
    metrics.update({
        "device": str(device),
        "epochs": epochs,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "history": history,
        "class_balance": class_balance,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
    })
    return model, metrics


def evaluate_policy(model: PolicyNet, dataset: TensorDataset, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    correct = 0
    total = 0
    prediction_counts = Counter()

    model.eval()
    with torch.no_grad():
        for obs_batch, action_batch in loader:
            logits = model(obs_batch.to(device))
            predictions = torch.argmax(logits, dim=-1)
            correct += int((predictions == action_batch.to(device)).sum().cpu())
            total += int(action_batch.shape[0])
            prediction_counts.update(int(item) for item in predictions.detach().cpu().tolist())

    fractions = {str(action): prediction_counts[action] / max(1, total) for action in (0, 1, 2)}
    entropy = -sum(value * math.log(value) for value in fractions.values() if value > 0)
    return {
        "validation_accuracy": correct / max(1, total),
        "prediction_entropy": entropy,
        "prediction_action_fractions": fractions,
        "validation_examples": total,
    }


def tensor_to_nested(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def export_browser_model(
    model: PolicyNet,
    base_model_path: Path,
    output_path: Path,
    run_id: str,
    metrics: dict[str, Any],
) -> None:
    with base_model_path.open("r", encoding="utf8") as file:
        base_model = json.load(file)

    exported = {
        **base_model,
        "modelType": "human-supervised-policy",
        "source": "human-supervised-browser-logs",
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "trainingRunId": run_id,
        "trainingMetrics": {
            "validationAccuracy": metrics.get("validation_accuracy"),
            "predictionEntropy": metrics.get("prediction_entropy"),
            "trainExamples": metrics.get("train_examples"),
            "validationExamples": metrics.get("validation_examples"),
        },
        "checkpointUpdate": int(base_model.get("checkpointUpdate", 0)),
        "layers": {
            "backbone0": {
                "weight": tensor_to_nested(model.backbone[0].weight),
                "bias": tensor_to_nested(model.backbone[0].bias),
            },
            "backbone2": {
                "weight": tensor_to_nested(model.backbone[2].weight),
                "bias": tensor_to_nested(model.backbone[2].bias),
            },
            "policyHead": {
                "weight": tensor_to_nested(model.policy.weight),
                "bias": tensor_to_nested(model.policy.bias),
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf8") as file:
        json.dump(exported, file, separators=(",", ":"))


def gate_metrics(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[bool, list[str]]:
    eval_config = config.get("eval", {})
    failures = []
    validation_accuracy = float(metrics.get("validation_accuracy", 0.0))
    prediction_entropy = float(metrics.get("prediction_entropy", 0.0))
    fractions = metrics.get("prediction_action_fractions", {})
    stay_fraction = float(fractions.get("0", 0.0))
    single_action_fraction = max((float(fractions.get(str(action), 0.0)) for action in (0, 1, 2)), default=1.0)

    if validation_accuracy < float(eval_config.get("min_validation_accuracy", 0.38)):
        failures.append(f"validation accuracy {validation_accuracy:.3f} is below threshold")
    if prediction_entropy < float(eval_config.get("min_prediction_entropy", 0.2)):
        failures.append(f"prediction entropy {prediction_entropy:.3f} is too low")
    if stay_fraction > float(eval_config.get("max_stay_prediction_fraction", 0.75)):
        failures.append(f"stay prediction fraction {stay_fraction:.3f} is too high")
    if single_action_fraction > float(eval_config.get("max_single_action_fraction", 0.85)):
        failures.append(f"single-action prediction fraction {single_action_fraction:.3f} is too high")

    return len(failures) == 0, failures


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")


def sync_r2_if_enabled(config: dict[str, Any], data_dir: Path) -> None:
    r2_config = config.get("r2", {})
    if not r2_config.get("enabled", False):
        return

    bucket = r2_config["bucket"]
    prefix = str(r2_config.get("prefix", "")).strip("/")
    endpoint_url = r2_config["endpoint_url"]
    source = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
    data_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "aws",
        "s3",
        "sync",
        source,
        str(data_dir),
        "--endpoint-url",
        endpoint_url,
        "--only-show-errors",
    ]
    print(f"syncing R2 training chunks from {source}")
    subprocess.run(command, check=True)


def promote_model(
    candidate_path: Path,
    config: dict[str, Any],
    config_dir: Path,
    run_id: str,
) -> dict[str, str]:
    promotion_config = config.get("promotion", {})
    model_dir = resolve_path(config.get("model_dir", "../ai-pong-human-training-site/web/models"), config_dir)
    outputs: dict[str, str] = {}

    if promotion_config.get("keep_versioned_copy", True):
        versioned_path = model_dir / f"human-policy-gen-{run_id}.json"
        versioned_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_path, versioned_path)
        outputs["versioned"] = str(versioned_path)

    if promotion_config.get("enabled", False):
        promote_to = resolve_path(promotion_config["promote_to"], config_dir)
        promote_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_path, promote_to)
        outputs["current"] = str(promote_to)

    return outputs


def run_once(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Missing config: {config_path}", file=sys.stderr)
        return 2

    config = load_config(config_path)
    if args.promote:
        config.setdefault("promotion", {})["enabled"] = True
    if args.no_promote:
        config.setdefault("promotion", {})["enabled"] = False

    config_dir = config_path.parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).resolve().parent / "runs" / run_id
    data_dir = resolve_path(config["data_dir"], config_dir)
    base_model = resolve_path(config["base_model"], config_dir)
    candidate_path = run_dir / "candidate-policy.json"
    metrics_path = run_dir / "metrics.json"

    if not args.skip_sync:
        sync_r2_if_enabled(config, data_dir)

    max_samples = int(config["training"].get("max_samples", 1_000_000))
    examples, stats = load_examples(data_dir, max_samples=max_samples, seed=int(config["training"].get("seed", 7)))

    metrics: dict[str, Any] = {
        "runId": run_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dataDir": str(data_dir),
        "baseModel": str(base_model),
        "filesSeen": stats.files_seen,
        "filesLoaded": stats.files_loaded,
        "recordsSeen": stats.records_seen,
        "recordsUsed": stats.records_used,
        "recordsSampledForTraining": len(examples),
        "invisibleRecords": stats.invisible_records,
        "invalidRecords": stats.invalid_records,
        "actionCounts": {str(key): value for key, value in sorted(stats.action_counts.items())},
    }

    min_files = int(config["training"].get("min_files", 5))
    min_samples = int(config["training"].get("min_samples", 5000))
    if stats.files_loaded < min_files or len(examples) < min_samples:
        metrics.update({
            "status": "skipped",
            "reason": f"not enough data: {stats.files_loaded} files and {len(examples)} samples",
        })
        write_metrics(metrics_path, metrics)
        print(metrics["reason"])
        print(f"wrote metrics: {metrics_path}")
        return 0

    if not base_model.exists():
        metrics.update({"status": "failed", "reason": f"missing base model: {base_model}"})
        write_metrics(metrics_path, metrics)
        print(metrics["reason"], file=sys.stderr)
        return 2

    model, train_metrics = train_policy(examples, config)
    metrics.update(train_metrics)

    gate_passed, gate_failures = gate_metrics(metrics, config)
    metrics["gatePassed"] = gate_passed
    metrics["gateFailures"] = gate_failures

    export_browser_model(model, base_model, candidate_path, run_id, metrics)
    metrics["candidateModel"] = str(candidate_path)

    if gate_passed:
        promotion_outputs = promote_model(candidate_path, config, config_dir, run_id)
        metrics["status"] = "promoted" if promotion_outputs.get("current") else "candidate-ready"
        metrics["promotionOutputs"] = promotion_outputs
    else:
        metrics["status"] = "candidate-rejected"
        print("candidate failed eval gates:")
        for failure in gate_failures:
            print(f"- {failure}")

    write_metrics(metrics_path, metrics)
    print(f"wrote metrics: {metrics_path}")
    print(f"candidate model: {candidate_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AI Pong supervised training pass from browser trajectory chunks.")
    parser.add_argument("--config", default="config.json", help="Path to config.json.")
    parser.add_argument("--skip-sync", action="store_true", help="Skip optional R2 aws s3 sync.")
    parser.add_argument("--promote", action="store_true", help="Override config and promote passing candidates.")
    parser.add_argument("--no-promote", action="store_true", help="Override config and do not replace current-policy.json.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_once(parse_args()))
