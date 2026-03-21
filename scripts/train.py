"""AlphaZero Distributed Training Orchestration Script.

This script runs continuous AlphaZero training using:
- JAX / Flax for Neural Network execution
- Pgx for hardware-accelerated batch environments
- Flashbax for experiential replay
- Orbax for asynchronous checkpointing
- WandB for logging
"""

import argparse
import functools
import os
import time
from typing import Any, cast

import jax
import orbax.checkpoint
import wandb
from flax.training import orbax_utils

from hexa_tic_tac_toe.agent import (
    AlphaZeroNet,
    create_replay_buffer,
    create_train_state,
    init_buffer_state,
    self_play_step,
    train_step,
)
from hexa_tic_tac_toe.env.pgx_env import HexTicTacToePgx


def main(args: argparse.Namespace) -> None:
    # 1. Initialize Weights & Biases
    if args.use_wandb:
        wandb.init(project="hexa-tic-tac-toe-alphazero", config=vars(args))

    # 2. Setup Checkpointing (Orbax)
    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        ckpt_dir, orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), options=options
    )

    # 3. Environment & Neural Network
    env = HexTicTacToePgx()
    network = AlphaZeroNet()

    key = jax.random.PRNGKey(args.seed)
    key, net_key, env_key = jax.random.split(key, 3)

    # Initialize TrainState (Network parameters + Optax state)
    train_state = create_train_state(net_key, learning_rate=args.learning_rate)

    # Restore from checkpoint if available
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        print(f"Restoring from checkpoint at step {latest_step}...")
        train_state = checkpoint_manager.restore(latest_step, args=orbax_utils.restore_args_from_target(train_state))
    else:
        latest_step = 0

    # 4. Initialize Flashbax Replay Buffer
    buffer = create_replay_buffer(
        max_length=args.buffer_size,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
        add_batch_size=args.num_envs,
    )
    buffer_state = init_buffer_state(buffer, dummy_env_obs_shape=(3, 106, 106), action_size=11236)

    # 5. Initialize Batched Pgx Environments
    env_keys = jax.random.split(env_key, args.num_envs)
    batched_env_state = jax.jit(jax.vmap(env.init))(env_keys)

    # 6. JIT Compile the core heavy operations
    @functools.partial(jax.jit, static_argnames=("num_simulations",))
    def jitted_actor_step(params, e_state, r_key, num_simulations):
        return self_play_step(env, network, params, e_state, r_key, num_simulations)

    jitted_actor_step = cast(Any, jitted_actor_step)

    jitted_buffer_add = jax.jit(buffer.add)
    jitted_buffer_sample = jax.jit(buffer.sample)
    jitted_train_step = jax.jit(train_step)

    print("Starting continuous training loop...")
    for step in range(latest_step + 1, args.total_steps + 1):
        start_time = time.time()

        # ACTOR: Self-Play Phase
        # Run X simultaneous MCTS steps on the batched environments
        key, sp_key = jax.random.split(key)
        batched_env_state, transitions, _ = jitted_actor_step(
            train_state.params, batched_env_state, sp_key, num_simulations=args.num_simulations  # type: ignore
        )

        # STORAGE: Insert transitions into the replay buffer
        buffer_state = jitted_buffer_add(buffer_state, transitions)

        # LEARNER: Training Phase (Only if buffer is sufficiently full)
        metrics = {}
        if buffer.can_sample(buffer_state):
            # Sample a massive batch from history
            key, sample_key = jax.random.split(key)
            batch = jitted_buffer_sample(buffer_state, sample_key)
            
            # Step the Optax optimizer
            train_state, metrics = jitted_train_step(train_state, batch.experience)

        step_duration = time.time() - start_time

        # 7. Logging and Checkpointing
        if step % args.log_interval == 0 and metrics:
            log_data = {
                "step": step,
                "value_loss": float(metrics["value_loss"]),
                "policy_loss": float(metrics["policy_loss"]),
                "total_loss": float(metrics["total_loss"]),
                "steps_per_sec": 1.0 / step_duration,
                "env_steps_per_sec": args.num_envs / step_duration,
            }
            print(f"[{step}/{args.total_steps}] Loss: {log_data['total_loss']:.4f} | Speed: {log_data['env_steps_per_sec']:.1f} env_steps/s")
            
            if args.use_wandb:
                wandb.log(log_data)

        if step % args.save_interval == 0:
            print(f"Saving checkpoint at step {step}...")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(step, train_state, save_kwargs={"save_args": save_args})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero Hexagonal Tic-Tac-Toe Training")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments for self-play")
    parser.add_argument("--num_simulations", type=int, default=25, help="Number of MCTS simulations per step")
    parser.add_argument("--buffer_size", type=int, default=50_000, help="Flashbax replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam optimizer learning rate")
    parser.add_argument("--total_steps", type=int, default=10_000, help="Total number of self-play/train cycles")
    parser.add_argument("--log_interval", type=int, default=10, help="Steps between logging metrics")
    parser.add_argument("--save_interval", type=int, default=1000, help="Steps between Orbax checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for Orbax checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights and Biases")
    parser.add_argument("--seed", type=int, default=42, help="JAX random seed")
    
    main(parser.parse_args())
