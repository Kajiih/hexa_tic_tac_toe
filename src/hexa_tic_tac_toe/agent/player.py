"""AI Player implementation for Hexagonal Tic-Tac-Toe."""

import os
from typing import Any, cast

import jax
import jax.numpy as jnp
import orbax.checkpoint

from hexa_tic_tac_toe.agent.network import AlphaZeroNet
from hexa_tic_tac_toe.agent.mcts import run_mcts
from hexa_tic_tac_toe.agent.trainer import create_train_state
from hexa_tic_tac_toe.env.pgx_env import HexTicTacToePgx


class AlphaZeroPlayer:
    """A player that uses a trained AlphaZero model to make moves."""

    def __init__(self, checkpoint_dir: str = "./checkpoints", seed: int = 42) -> None:
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.env = HexTicTacToePgx()
        self.network = AlphaZeroNet()
        
        # 1. Initialize PRNG Keys
        self.key = jax.random.PRNGKey(seed)
        self.key, net_key = jax.random.split(self.key)
        
        # 2. Setup TrainState (to hold parameters)
        # Using default learning rate as we are only using it for inference
        self.train_state = create_train_state(net_key)
        
        # 3. Restore from checkpoint
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            self.checkpoint_dir, 
            options=orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2)
        )
        
        latest_step = self.checkpoint_manager.latest_step()
        if latest_step is None:
            print(f"Warning: No checkpoint found in {checkpoint_dir}. Using initial random weights.")
        else:
            print(f"Loading AI model from checkpoint at step {latest_step}...")
            restore_args = orbax.checkpoint.args.StandardRestore(self.train_state)
            self.train_state = self.checkpoint_manager.restore(latest_step, args=restore_args)

        # 4. JIT-compile the search function
        self._setup_search()

    def _setup_search(self) -> None:
        """Compiles the MCTS search function for this env/network."""
        import functools

        @functools.partial(jax.jit, static_argnames=("num_simulations",))
        def search_fn(params, state, key, num_simulations):
            # MCTX requires batched input
            batched_state = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
            output = run_mcts(self.env, self.network, params, batched_state, key, num_simulations)
            return jnp.squeeze(output.action) # return the best action scalar

        self.search_fn = cast(Any, search_fn)

    def decide_move(self, game_state: Any, num_simulations: int = 50) -> int:
        """Computes the best move for the given Pgx state.

        Args:
            game_state: A Pgx HexaState.
            num_simulations: The number of MCTS simulations.

        Returns:
            The integer action index.
        """
        self.key, subkey = jax.random.split(self.key)
        # Access parameters with cast to Any to avoid lint errors
        params = cast(Any, self.train_state).params
        action = self.search_fn(params, game_state, subkey, num_simulations)
        return int(action)
