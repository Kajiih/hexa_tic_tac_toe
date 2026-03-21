"""Optax-based AlphaZero Self-Play and Training Loop."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from hexa_tic_tac_toe.agent.mcts import run_mcts
from hexa_tic_tac_toe.agent.network import AlphaZeroNet
from hexa_tic_tac_toe.env.pgx_env import HexTicTacToePgx


def create_train_state(rng: jax.Array, learning_rate: float = 1e-3) -> TrainState:
    """Initializes the Flax TrainState with Optax optimizer."""
    network = AlphaZeroNet()
    # Dummy input to initialize the network parameters
    dummy_obs = jnp.zeros((1, 3, 106, 106), dtype=jnp.bool_)
    variables = network.init(rng, dummy_obs)
    params = variables["params"]

    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def loss_fn(params: Any, network: AlphaZeroNet, batch: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Computes the AlphaZero loss: Policy Cross-Entropy + Value MSE."""
    obs = batch["observation"]
    target_policy = batch["target_policy"]
    target_value = batch["target_value"]

    policy_logits, value = network.apply({"params": params}, obs)

    # Value Loss (MSE)
    value_loss = jnp.mean(optax.l2_loss(value, target_value))  # type: ignore

    # Policy Loss (Cross Entropy)
    # target_policy from mctx is already probabilities
    policy_loss = jnp.mean(optax.softmax_cross_entropy(logits=policy_logits, labels=target_policy))

    total_loss = value_loss + policy_loss
    return total_loss, {"value_loss": value_loss, "policy_loss": policy_loss}


@jax.jit
def train_step(state: TrainState, batch: dict[str, jnp.ndarray]) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """Performs a single JIT-compiled optimization step."""
    network = AlphaZeroNet()
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, network, batch)
    state = state.apply_gradients(grads=grads)
    metrics["total_loss"] = loss
    return state, metrics


def self_play_step(
    env: HexTicTacToePgx,
    network: AlphaZeroNet,
    params: Any,
    env_state: Any,
    rng_key: jax.Array,
    num_simulations: int = 10,
) -> tuple[Any, dict[str, jnp.ndarray], jax.Array]:
    """Runs MCTS and advances the environment by exactly one step."""
    # 1. Run MCTS to get the improved policy
    mcts_key, step_key, next_rng_key = jax.random.split(rng_key, 3)
    policy_output = run_mcts(env, network, params, env_state, mcts_key, num_simulations)

    # The action weights from MCTS are our target policy
    target_policy = policy_output.action_weights

    # Extract the chosen action
    action = policy_output.action

    # Store the observation before stepping
    observation = env_state.observation

    # 2. Step the environments forward natively using vmapped pgx
    env_step_vmap = jax.vmap(env.step)
    batch_size = action.shape[0]
    step_keys = jax.random.split(step_key, batch_size)
    next_env_state = env_step_vmap(env_state, action, step_keys)

    # Collect transition data (we assign target_value loosely as the outcome seen later, 
    # but for this verified pipeline test, we just use the raw rewards/0.0 as placeholders)
    transition = {
        "observation": observation,
        "target_policy": target_policy,
        "target_value": jnp.zeros(batch_size, dtype=jnp.float32),  # Placeholder for MCTS/game value
    }

    return next_env_state, transition, next_rng_key  # type: ignore
