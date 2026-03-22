"""AlphaZero Neural Network Architecture implemented in Flax."""

from typing import Final

import jax
import jax.numpy as jnp
from flax import linen as nn

from hexa_tic_tac_toe.core.constants import GRID_SIZE

ACTION_SPACE_SIZE: Final[int] = GRID_SIZE


class AlphaZeroNet(nn.Module):
    """A ResNet architecture with dual Policy and Value heads.

    This network evaluates a board state and outputs both the legal move probabilities
    (policy) and the expected game outcome from the current player's perspective (value).
    """

    num_actions: int = ACTION_SPACE_SIZE
    num_channels: int = 64
    num_blocks: int = 5

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass of the AlphaZero network.

        Args:
            x: A batch of observations of shape `(B, 3, 106, 106)` or a single
               observation of shape `(3, 106, 106)`. This corresponds to
               `[my_board, opponent_board, valid_mask]`.

        Returns:
            A tuple of (policy_logits, value):
                policy_logits: logits of shape `(B, num_actions)`.
                value: scalar evaluations of shape `(B,)` in range [-1, 1].
        """
        # Handle unbatched single state input
        if x.ndim == 3:
            x = jnp.expand_dims(x, 0)

        # PyTorch/PettingZoo uses (N, C, H, W). JAX Convolutions expect (N, H, W, C).
        # We transpose the channel dimension to the end.
        x = jnp.transpose(x, (0, 2, 3, 1))

        # 1. Initial Convolutional Block
        x = nn.Conv(features=self.num_channels, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
        # Using LayerNorm instead of BatchNorm to avoid `batch_stats` state management complexity
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # 2. Residual Tower
        for _ in range(self.num_blocks):
            y = nn.Conv(features=self.num_channels, kernel_size=(3, 3), padding="SAME", use_bias=False)(x)
            y = nn.LayerNorm()(y)
            y = nn.relu(y)
            y = nn.Conv(features=self.num_channels, kernel_size=(3, 3), padding="SAME", use_bias=False)(y)
            y = nn.LayerNorm()(y)
            x = nn.relu(x + y)

        # 3. Policy Head
        # Compresses the representation and flattens it into action logits
        p = nn.Conv(features=2, kernel_size=(1, 1), use_bias=False)(x)
        p = nn.LayerNorm()(p)
        p = nn.relu(p)

        # Flatten from (B, H, W, C) to (B, H*W*C)
        p = p.reshape((p.shape[0], -1))
        policy_logits = nn.Dense(features=self.num_actions)(p)

        # 4. Value Head
        # Compresses the representation and flattens it into a single scalar value (-1 to 1)
        v = nn.Conv(features=1, kernel_size=(1, 1), use_bias=False)(x)
        v = nn.LayerNorm()(v)
        v = nn.relu(v)

        # Flatten and process via dense layers
        v = v.reshape((v.shape[0], -1))
        v = nn.Dense(features=64)(v)
        v = nn.relu(v)

        value = nn.Dense(features=1)(v)
        value = jnp.tanh(value)
        value = jnp.squeeze(value, axis=-1)  # Squeeze (B, 1) down to (B,)

        return policy_logits, value
