"""Monte Carlo Tree Search implementation using DeepMind's mctx."""

from typing import Any

import jax
import jax.numpy as jnp
import mctx
import pgx
from flax import linen as nn

from hexa_tic_tac_toe.env.pgx_env import HexTicTacToePgx


def mask_invalid_actions(logits: jnp.ndarray, legal_action_mask: jnp.ndarray) -> jnp.ndarray:
    """Masks invalid actions by setting their logits to a very large negative number."""
    # Using -1e9 to ensure softmax probability becomes 0
    return jnp.where(legal_action_mask, logits, -1e9)


def build_recurrent_fn(env: HexTicTacToePgx, network: nn.Module):
    """Builds the recurrent function required by mctx for tree search.

    In AlphaZero, the 'embedding' is just the actual environment state.
    Instead of predicting the next state with a neural network (MuZero),
    we use the actual fast JAX `pgx.Env` to step the state.

    Args:
        env: The Pgx environment instance.
        network: The Flax AlphaZero neural network.

    Returns:
        A recurrent_fn compatible with mctx.
    """

    def recurrent_fn(params: Any, rng_key: jax.Array, action: jnp.ndarray, embedding: pgx.State):
        # 1. Step the environment forward
        # Action needs to be expanded or handled if it's batched.
        # mctx passes batched actions and embeddings: shape (B,) and (B, ...)
        # `pgx.Env.step` is already vmap-compatible or we vmap it.
        # Wait, pgx APIs are often written for single states, but we just vmap them.
        env_step_vmap = jax.vmap(env.step, in_axes=(0, 0, 0))
        
        # We need a batch of keys
        batch_size = action.shape[0]
        step_keys = jax.random.split(rng_key, batch_size)
        
        next_state = env_step_vmap(embedding, action, step_keys)

        # 2. Evaluate the new state using the Neural Network
        policy_logits, value = network.apply({"params": params}, next_state.observation)
        
        # Mask impossible moves out of the policy
        policy_logits = mask_invalid_actions(policy_logits, next_state.legal_action_mask)

        # 3. Handle rewards and discounts for zero-sum alternating turns
        # mctx uses: Q(s, a) = r + discount * V(s')
        # In alternating turn games, V(s') is from the opponent's perspective.
        # Thus, if the game isn't over, discount = -1.0 to flip the opponent's value.
        # If the game is over, discount = 0.0.
        
        # Because pgx step returns rewards from the perspective of both players,
        # we extract the reward for the player who just made the move.
        # In our env, reward[current_player] is for the NEW current player.
        # We want the reward for the player who took the action (the old current player).
        old_player = embedding.current_player
        # Gather the rewards for the `old_player` using advanced indexing
        reward = next_state.rewards[jnp.arange(batch_size), old_player]
        
        discount = jnp.where(next_state.terminated, 0.0, -1.0)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=policy_logits,
            value=value,
        )
        return recurrent_fn_output, next_state

    return recurrent_fn


def run_mcts(
    env: HexTicTacToePgx,
    network: nn.Module,
    params: Any,
    state: pgx.State,
    rng_key: jax.Array,
    num_simulations: int = 50,
) -> mctx.PolicyOutput:
    """Runs Monte Carlo Tree Search for a batch of states.

    Args:
        env: The Pgx environment.
        network: The AlphaZero Neural Net.
        params: Network parameters.
        state: A batched pgx.State of shape (B, ...).
        rng_key: JAX PRNG key.
        num_simulations: Number of MCTS passes per move.

    Returns:
        mctx.PolicyOutput containing action selections and improved policies.
    """
    # Initial network evaluation for the root nodes
    policy_logits, value = network.apply({"params": params}, state.observation)
    policy_logits = mask_invalid_actions(policy_logits, state.legal_action_mask)

    root = mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=value,
        embedding=state,
    )

    recurrent_fn = build_recurrent_fn(env, network)

    # We use muzero_policy (which is functionally AlphaZero when embedding is real state)
    # Gumbel MuZero is also highly recommended by DeepMind for AlphaZero, but standard
    # muzero_policy with a dirichlet distribution on the root is the classic AlphaZero.
    policy_output = mctx.muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=None, # Search until termination
        dirichlet_fraction=0.25, # Standard AlphaZero exploration
        dirichlet_alpha=0.3,
        temperature=1.0, # 1.0 encourages exploration during self-play
    )

    return policy_output
