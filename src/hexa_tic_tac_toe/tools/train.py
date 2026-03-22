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
import jax.numpy as jnp
import orbax.checkpoint
import wandb


from hexa_tic_tac_toe.agent import (
    AlphaZeroNet,
    create_replay_buffer,
    create_train_state,
    evaluate_vs_random,
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
        ckpt_dir, options=options
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
        restore_args = orbax.checkpoint.args.StandardRestore(train_state)
        train_state = checkpoint_manager.restore(latest_step, args=restore_args)
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

    # 7. Trajectory Tracking
    # We maintain a list of transitions for each parallel environment
    # to back-propagate the final game reward once it terminates.
    trajectories: list[list[dict[str, Any]]] = [[] for _ in range(args.num_envs)]
    pending_buffer_add: list[dict[str, Any]] = []

    jitted_buffer_add = jax.jit(buffer.add)
    jitted_buffer_sample = jax.jit(buffer.sample)
    jitted_train_step = jax.jit(train_step)

    @functools.partial(jax.jit, static_argnames=("num_games", "num_simulations"))
    def jitted_eval_step(params, r_key, num_games, num_simulations):
        return evaluate_vs_random(env, network, params, r_key, num_games, num_simulations)

    print("Starting continuous training loop...")
    for step in range(latest_step + 1, args.total_steps + 1):
        start_time = time.time()

        # ACTOR: Self-Play Phase
        # Run X simultaneous MCTS steps on the batched environments
        key, sp_key = jax.random.split(key)
        batched_env_state, transitions, _, terminated, rewards = jitted_actor_step(
            train_state.params, batched_env_state, sp_key, num_simulations=args.num_simulations  # type: ignore
        )

        # STORAGE: Manage trajectories and insert into replay buffer
        # 1. Store the transitions for each environment
        # We need to move the data from JAX device arrays to Python for trajectory management
        obs_batch = jax.device_get(transitions["observation"])
        policy_batch = jax.device_get(transitions["target_policy"])
        player_batch = jax.device_get(transitions["current_player"])
        term_batch = jax.device_get(terminated)
        reward_batch = jax.device_get(rewards)

        for i in range(args.num_envs):
            trajectories[i].append({
                "observation": obs_batch[i],
                "target_policy": policy_batch[i],
                "current_player": int(player_batch[i]),
            })

            if term_batch[i]:
                # Episode finished! Propagate the reward to all steps.
                final_rewards = reward_batch[i]  # [reward_p0, reward_p1]
                for trans in trajectories[i]:
                    player_id = trans.pop("current_player")
                    trans["target_value"] = final_rewards[player_id]
                    pending_buffer_add.append(trans)
                trajectories[i] = []

        # 2. Push to Replay Buffer in batches of `num_envs` to maintain efficiency
        while len(pending_buffer_add) >= args.num_envs:
            batch_to_add = pending_buffer_add[:args.num_envs]
            pending_buffer_add = pending_buffer_add[args.num_envs:]

            # Stack individual transitions into a batch of arrays
            collated_batch = {
                "observation": jnp.stack([t["observation"] for t in batch_to_add]),
                "target_policy": jnp.stack([t["target_policy"] for t in batch_to_add]),
                "target_value": jnp.array([t["target_value"] for t in batch_to_add], dtype=jnp.float32),
            }
            buffer_state = jitted_buffer_add(buffer_state, collated_batch)

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
            save_args = orbax.checkpoint.args.StandardSave(train_state)
            checkpoint_manager.save(step, args=save_args)

        # 8. Evaluation Phase
        if step % args.eval_interval == 0:
            print(f"[{step}/{args.total_steps}] Evaluating agent against random baseline...")
            key, eval_key = jax.random.split(key)
            # Cast params to Any for mctx/jax compatibility and to avoid lint errors
            eval_metrics = jitted_eval_step(
                cast(Any, train_state).params, eval_key, num_games=args.eval_games, num_simulations=args.num_simulations
            )
            # Convert JAX scalars to Python floats for logging
            eval_results = {k: float(v) for k, v in eval_metrics.items()}
            print(f"Evaluation Results: Win: {eval_results['win_rate']:.2f} | Draw: {eval_results['draw_rate']:.2f} | Loss: {eval_results['loss_rate']:.2f}")
            
            if args.use_wandb:
                wandb.log({f"eval/{k}": v for k, v in eval_results.items()} | {"step": step})


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
    parser.add_argument("--eval_interval", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--eval_games", type=int, default=64, help="Number of games to play during evaluation")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for Orbax checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights and Biases")
    parser.add_argument("--seed", type=int, default=42, help="JAX random seed")
    
    main(parser.parse_args())
