# AI Pong Self-Play

This project trains two Pong agents adversarially using PPO self-play. Both paddles are controlled by separate neural networks that learn from the same zero-sum match.

## Features

- Two AI-controlled paddles trained against each other
- PPO-based adversarial self-play in PyTorch
- `pygame` viewer for watching saved checkpoints battle
- Simple CLI for training and visualization

## Quick Start

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Train the agents:

```bash
python main.py train --num-envs 16 --steps-per-update 2048 --updates 50
```

Watch the latest checkpoint:

```bash
python main.py watch --checkpoint checkpoints/selfplay_latest.pt
```

## Notes

- Training longer produces stronger rallies. A few hundred updates is enough to see coordinated behavior emerge.
- Checkpoints are written to `checkpoints/` during training.
