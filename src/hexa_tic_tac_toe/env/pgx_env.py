"""Pgx (Pure JAX) Environment for Hexagonal Tic-Tac-Toe.

This environment is written entirely in JAX, allowing 100x faster
hardware-accelerated simulations on GPU/TPU for AlphaZero.
"""

from typing import Final
import jax
import jax.numpy as jnp
from flax import struct
import pgx

# Constants for Radius 50
RADIUS: Final[int] = 50
WIN_LENGTH: Final[int] = 6
OFFSET: Final[int] = RADIUS - 1
PADDED_WIDTH: Final[int] = 2 * RADIUS + WIN_LENGTH
GRID_SIZE: Final[int] = PADDED_WIDTH * PADDED_WIDTH

# Max number of pieces that can fit on the board (the valid cells)
MAX_PIECES: Final[int] = 3 * RADIUS * RADIUS - 3 * RADIUS + 1


@struct.dataclass
class HexaState(pgx.State):
    """The strictly immutable JAX PyTree representing the game state."""
    # Internal HexGame tracking
    boards: jnp.ndarray
    turn_number: jnp.ndarray
    moves_this_turn: jnp.ndarray

    @property
    def env_id(self) -> pgx.EnvId | str:
        return "hexa_tic_tac_toe-v0"


class HexTicTacToePgx(pgx.Env):
    """Hexagonal Tic-Tac-Toe environment for Pgx."""

    def __init__(self) -> None:
        super().__init__()

        # Precompute the immutable valid coordinate mask
        q_idx, r_idx = jnp.meshgrid(
            jnp.arange(PADDED_WIDTH), jnp.arange(PADDED_WIDTH), indexing="ij"
        )
        q = q_idx - OFFSET
        r = r_idx - OFFSET

        # Standard hex game valid grid check
        valid = (jnp.abs(q) < RADIUS) & (jnp.abs(r) < RADIUS) & (jnp.abs(q + r) < RADIUS)
        self._valid_flat = valid.flatten()

    def _init(self, key: jax.Array) -> pgx.State:
        """Initializes the empty board state."""
        return HexaState(
            current_player=jnp.int32(0),
            observation=self._get_obs(jnp.zeros((2, GRID_SIZE), dtype=jnp.bool_), jnp.int32(0)),
            rewards=jnp.zeros(2, dtype=jnp.float32),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            legal_action_mask=self._valid_flat,
            _step_count=jnp.int32(0),
            boards=jnp.zeros((2, GRID_SIZE), dtype=jnp.bool_),
            turn_number=jnp.int32(1),
            moves_this_turn=jnp.int32(0),
        )

    def _step(self, state: pgx.State, action: jnp.ndarray, key: jax.Array) -> pgx.State:
        """Executes one action sequentially applying JAX transformations."""
        assert isinstance(state, HexaState)
        
        # Update board
        player = state.current_player
        new_boards = state.boards.at[player, action].set(True)

        # Check Win
        has_win = self._check_win(new_boards[player])

        # Check Draw (board completely full)
        total_pieces = jnp.sum(new_boards[0] | new_boards[1])
        is_draw = total_pieces >= MAX_PIECES

        # Calculate reward
        terminated = has_win | is_draw
        win_reward = jnp.where(
            has_win,
            jnp.array([1.0, -1.0], dtype=jnp.float32).at[player].set(1.0).at[1 - player].set(-1.0),
            jnp.zeros(2, dtype=jnp.float32),
        )
        reward = jnp.where(is_draw & ~has_win, jnp.zeros(2, dtype=jnp.float32), win_reward)

        # Turn Transitions
        new_moves = state.moves_this_turn + 1
        moves_needed = jnp.where(state.turn_number == 1, 1, 2)
        turn_over = new_moves >= moves_needed

        new_player = jnp.where(turn_over, 1 - state.current_player, state.current_player)
        new_turn = state.turn_number + jnp.where(turn_over, 1, 0)
        new_moves = jnp.where(turn_over, 0, new_moves)

        # Recalculate legal moves
        occupied = new_boards[0] | new_boards[1]
        legal_actions = self._valid_flat & ~occupied
        legal_actions = jnp.where(terminated, jnp.zeros_like(legal_actions), legal_actions)

        # If turn is over, current player flips
        # Observation is always from the perspective of the new current player
        observation = self._get_obs(new_boards, new_player)

        return state.replace(  # type: ignore[attr-defined]
            current_player=new_player,
            observation=observation,
            rewards=reward,
            terminated=terminated,
            legal_action_mask=legal_actions,
            boards=new_boards,
            turn_number=new_turn,
            moves_this_turn=new_moves,
        )

    def _observe(self, state: pgx.State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, HexaState)
        return self._get_obs(state.boards, player_id)

    def _get_obs(self, boards: jnp.ndarray, player_id: jnp.ndarray) -> jnp.ndarray:
        """Generates the AlphaZero compatible 3D tensor representing the state."""
        my_board = boards[player_id].reshape((PADDED_WIDTH, PADDED_WIDTH))
        opp_board = boards[1 - player_id].reshape((PADDED_WIDTH, PADDED_WIDTH))
        valid_mask = self._valid_flat.reshape((PADDED_WIDTH, PADDED_WIDTH))

        return jnp.stack([my_board, opp_board, valid_mask], axis=0)

    def _check_win(self, board: jnp.ndarray) -> jnp.ndarray:
        """Checks for win_length in a row natively in JAX arrays."""
        has_win = jnp.bool_(False)
        # Directions: Horizontal, Vertical, Diagonal (in axial coords mapped to 2D matrix)
        for direction in (1, PADDED_WIDTH, PADDED_WIDTH - 1):
            mask = board
            for shift in range(1, WIN_LENGTH):
                # Roll natively shifts the array indices (bitshift equivalent)
                mask = mask & jnp.roll(board, -(shift * direction))
            has_win = has_win | jnp.any(mask)
        return has_win

    @property
    def id(self) -> str:
        return "hexa_tic_tac_toe-v0"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (3, PADDED_WIDTH, PADDED_WIDTH)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def num_actions(self) -> int:
        return GRID_SIZE
