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


class TrajectoryBuffer:
    """Helper to collect and manage trajectories from parallel environments.

    This class handles:
    1. Accumulating transitions for each environment.
    2. Propagating final rewards backward through the trajectory once an episode ends.
    3. Stacking completed transitions into batches for the Flashbax replay buffer.
    """

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.trajectories: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
        self.pending_add: list[dict[str, Any]] = []

    def add_step(
        self,
        observations: jnp.ndarray,
        target_policies: jnp.ndarray,
        current_players: jnp.ndarray,
        is_terminated: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> None:
        """Adds a single step of Transitions from all environments.

        Args:
            observations: (num_envs, ...) JAX array of observations.
            target_policies: (num_envs, action_size) JAX array of MCTS policies.
            current_players: (num_envs,) JAX array of active player indices.
            is_terminated: (num_envs,) JAX array of termination flags.
            rewards: (num_envs, 2) JAX array of rewards for [P0, P1].
        """
        for i in range(self.num_envs):
            # Store the transition for this environment
            self.trajectories[i].append({
                "observation": observations[i],
                "target_policy": target_policies[i],
                "current_player": int(current_players[i]),
            })

            if is_terminated[i]:
                # Episode finished! Propagate reward to all steps.
                final_rewards = rewards[i]
                for trans in self.trajectories[i]:
                    p_id = trans.pop("current_player")
                    trans["target_value"] = float(final_rewards[p_id])
                    self.pending_add.append(trans)
                # Reset trajectory for this environment
                self.trajectories[i] = []

    def get_add_batch(self, batch_size: int) -> dict[str, jnp.ndarray] | None:
        """Returns a stacked batch of completed transitions if enough are available.

        Args:
            batch_size: The number of transitions to return.

        Returns:
            A dictionary of stacked JAX arrays, or None if not enough transitions.
        """
        if len(self.pending_add) < batch_size:
            return None

        batch_data = self.pending_add[:batch_size]
        self.pending_add = self.pending_add[batch_size:]

        return {
            "observation": jnp.stack([t["observation"] for t in batch_data]),
            "target_policy": jnp.stack([t["target_policy"] for t in batch_data]),
            "target_value": jnp.stack([t["target_value"] for t in batch_data]),
        }
