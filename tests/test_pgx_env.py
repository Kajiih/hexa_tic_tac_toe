import jax
import jax.numpy as jnp
from hexa_tic_tac_toe.env import HexTicTacToePgx


def test_pgx_env_jit() -> None:
    """Verifies that the Pgx environment complies with JAX constraints and compiles."""
    env = HexTicTacToePgx()

    # Verify that the functions can be JIT compiled
    init_fn = jax.jit(env.init)
    step_fn = jax.jit(env.step)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Initialize state
    state = init_fn(subkey)
    assert not state.terminated
    assert jnp.sum(state.boards) == 0

    # Execute a random valid action
    action = jnp.argmax(state.legal_action_mask)
    
    key, subkey = jax.random.split(key)
    state = step_fn(state, action, subkey)

    # Player 1 taking their first turn (only 1 move allowed for turn 1)
    assert state.turn_number == 2
    assert state.current_player == 1  # 1 means Player 2 in 0-indexed terms
    assert jnp.sum(state.boards[0]) == 1
    assert jnp.sum(state.boards[1]) == 0

    # Player 2 taking a move (needs 2 moves for their turn)
    action = jnp.argmax(state.legal_action_mask)
    key, subkey = jax.random.split(key)
    state = step_fn(state, action, subkey)

    assert state.turn_number == 2
    assert state.current_player == 1  # Still Player 2's turn
    assert state.moves_this_turn == 1
    assert jnp.sum(state.boards[1]) == 1
