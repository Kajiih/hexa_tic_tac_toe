import functools
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from hexa_tic_tac_toe.agent import AlphaZeroNet, create_train_state, self_play_step, train_step
from hexa_tic_tac_toe.env.pgx_env import HexTicTacToePgx


def test_agent_training_loop() -> None:
    """Verifies that the entire AlphaZero training loop compiles and runs under JIT."""
    env = HexTicTacToePgx()

    key = jax.random.PRNGKey(42)
    key, net_key, env_key = jax.random.split(key, 3)

    # 1. Initialize TrainState
    train_state = create_train_state(net_key)

    # 2. Initialize a batch of environments (batch size 2 to test vectorization)
    batch_size = 2
    env_keys = jax.random.split(env_key, batch_size)
    env_state = jax.vmap(env.init)(env_keys)

    # 3. Create a purely JIT-compiled self-play step
    network = AlphaZeroNet()

    @functools.partial(jax.jit, static_argnames=("num_simulations",))
    def jit_play_step(params: Any, e_state: Any, r_key: jax.Array, num_simulations: int = 4) -> tuple:
        return self_play_step(env, network, params, e_state, r_key, num_simulations)

    # Execute one batched step of self-play with MCTS
    # MCTS takes a few seconds to compile initially
    key, sp_key = jax.random.split(key)
    next_env_state, transition, _ = jit_play_step(
        train_state.params, env_state, sp_key, num_simulations=4
    )

    # Validation
    assert next_env_state.observation.shape == (batch_size, 3, 106, 106)
    assert transition["target_policy"].shape == (batch_size, 11236)
    assert transition["observation"].shape == (batch_size, 3, 106, 106)

    # 4. Execute a training step using the collected transition
    # The train_step in train.py is already @jax.jit decorated
    train_state, metrics = train_step(train_state, transition)

    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    # Verify that the loss actually computed into a scalar value
    assert metrics["total_loss"].shape == ()
