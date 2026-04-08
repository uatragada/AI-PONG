from __future__ import annotations

import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and watch adversarial AI Pong.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train two PPO agents via self-play.")
    train_parser.add_argument("--updates", type=int, default=300, help="Number of PPO updates.")
    train_parser.add_argument(
        "--steps-per-update",
        type=int,
        default=2048,
        help="Environment steps collected before each PPO update.",
    )
    train_parser.add_argument("--ppo-epochs", type=int, default=8, help="Gradient epochs per update.")
    train_parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    train_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    train_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    train_parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    train_parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon.")
    train_parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    train_parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient.")
    train_parser.add_argument("--hidden-size", type=int, default=128, help="Policy hidden size.")
    train_parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of Pong environments to run in parallel during training.",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Legacy option retained for compatibility. The torch-native simulator ignores worker count.",
    )
    train_parser.add_argument("--save-every", type=int, default=25, help="Snapshot interval in updates.")
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for latest and snapshot checkpoints.",
    )
    train_parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to use: auto, cpu, cuda, ...",
    )
    train_parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training when using CUDA.",
    )
    train_parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Use torch.compile for the policy networks when available.",
    )

    watch_parser = subparsers.add_parser("watch", help="Render a saved AI-vs-AI Pong match.")
    watch_parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/selfplay_latest.pt",
        help="Path to a saved self-play checkpoint.",
    )
    watch_parser.add_argument("--fps", type=int, default=60, help="Render frame rate.")
    watch_parser.add_argument("--episodes", type=int, default=5, help="How many matches to render.")
    watch_parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using greedy play.",
    )
    watch_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap, useful for smoke tests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        from ai_pong.train import train_self_play

        train_self_play(args)
        return
    if args.command == "watch":
        from ai_pong.render import watch_match

        watch_match(
            checkpoint_path=args.checkpoint,
            fps=args.fps,
            episodes=args.episodes,
            deterministic=not args.stochastic,
            max_frames=args.max_frames,
        )
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
