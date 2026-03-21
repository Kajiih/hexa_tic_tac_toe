"""Flashbax implementation of the AlphaZero Replay Buffer."""

from typing import Any

import flashbax as fbx
import jax.numpy as jnp


def create_replay_buffer(max_length: int = 100_000, min_length: int = 10_000, sample_batch_size: int = 256, add_batch_size: int = 256) -> Any:
    """Creates a Flashbax item buffer for storing self-play transitions.
    
    Args:
        max_length: Maximum capacity of the buffer.
        min_length: Minimum items needed before sampling is allowed.
        sample_batch_size: Number of items sampled during training.
        add_batch_size: The number of transition items added per self-play batch.
        
    Returns:
        A Flashbax buffer module.
    """
    buffer = fbx.make_item_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_batches=True,
    )
    return buffer


def init_buffer_state(buffer: Any, dummy_env_obs_shape: tuple[int, ...], action_size: int) -> Any:
    """Initializes the Flashbax buffer state from a dummy transition template."""
    dummy_transition = {
        "observation": jnp.zeros(dummy_env_obs_shape, dtype=jnp.bool_),
        "target_policy": jnp.zeros((action_size,), dtype=jnp.float32),
        "target_value": jnp.zeros((), dtype=jnp.float32),
    }
    return buffer.init(dummy_transition)
