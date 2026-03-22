"""Evaluation logic for AlphaZero vs Random baseline."""

import jax
import jax.numpy as jnp
from typing import Any
from hexa_tic_tac_toe.env.pgx_env import HexTicTacToePgx
from hexa_tic_tac_toe.agent.mcts import run_mcts

def evaluate_vs_random(
    env: HexTicTacToePgx,
    network: Any,
    params: Any,
    rng_key: jax.Array,
    num_games: int = 64,
    num_simulations: int = 25,
) -> dict[str, jax.Array]:
    """Plays a batch of games: AlphaZero (Player 0) vs Random (Player 1).
    
    Returns win/draw/loss rates from the perspective of AlphaZero.
    """
    
    def step_fn(state, key):
        # Determine current player
        curr_player = state.current_player
        
        # AlphaZero Move (if current_player == 0)
        # Random Move (if current_player == 1)
        
        # We use jax.lax.cond to switch between AlphaZero and Random
        key, az_key, rnd_key = jax.random.split(key, 3)
        
        # AlphaZero action using MCTS (mctx requires batched input)
        # We expand state.observation to (1, ...) and then squeeze the result
        # Actually, mctx requires the embedding itself to be batched
        batched_state = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        az_output = run_mcts(env, network, params, batched_state, az_key, num_simulations)
        az_action = jnp.squeeze(az_output.action) # Squeeze back to scalar
        
        # Random action
        probs = state.legal_action_mask.astype(jnp.float32)
        probs = probs / jnp.sum(probs)
        rnd_action = jax.random.choice(rnd_key, env.num_actions, p=probs)
        
        action = jnp.where(curr_player == 0, az_action, rnd_action)
        
        # Step environment
        next_state = env.step(state, action, key)
        return next_state, key

    # Vectorize the game loop
    def play_game(key):
        state = env.init(key)
        
        def cond_fn(state_key):
            state, _ = state_key
            return ~state.terminated
        
        def body_fn(state_key):
            state, key = state_key
            return step_fn(state, key)
            
        final_state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, key))
        return final_state.rewards[0] # Reward for AlphaZero (Player 0)

    keys = jax.random.split(rng_key, num_games)
    rewards = jax.vmap(play_game)(keys)
    
    wins = jnp.sum(rewards == 1.0)
    draws = jnp.sum(rewards == 0.0)
    losses = jnp.sum(rewards == -1.0)
    
    return {
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "loss_rate": losses / num_games,
    }
