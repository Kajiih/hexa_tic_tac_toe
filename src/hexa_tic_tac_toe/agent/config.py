"""Structured configuration for AlphaZero training using Hydra."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    radius: int = 50
    win_length: int = 6


@dataclass
class ModelConfig:
    num_channels: int = 64
    num_blocks: int = 5


@dataclass
class MCTSConfig:
    num_simulations: int = 50
    temperature: float = 1.0


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    batch_size: int = 256
    buffer_size: int = 50_000


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    project_name: str = "hexa-tic-tac-toe-alphazero"
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 100
    eval_games: int = 64


@dataclass
class AlphaZeroConfig:
    """Root configuration for the AlphaZero pipeline."""
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    seed: int = 42
    total_steps: int = 10_000
    num_envs: int = 128
    checkpoint_dir: str = "./checkpoints"
    checkpoint_step: Optional[int] = None
