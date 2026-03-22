"""AlphaZero Training Orchestrator.

This module provides the high-level training loop, integrating:
- Self-play with MCTS
- Replay buffer management
- Optimization steps
- Checkpointing with Orbax
- Logging with WandB
"""

import os
import time
import functools
from typing import Any, cast

import jax
import orbax.checkpoint
import wandb

from hexa_tic_tac_toe.agent.trainer import create_train_state, train_step, self_play_step
from hexa_tic_tac_toe.agent.network import AlphaZeroNet
from hexa_tic_tac_toe.agent.buffer import create_replay_buffer, init_buffer_state, TrajectoryBuffer
from hexa_tic_tac_toe.agent.evaluate import evaluate_vs_random
from hexa_tic_tac_toe.agent.config import AlphaZeroConfig
from hexa_tic_tac_toe.env.pgx_env import HexTicTacToePgx


class AlphaZeroOrchestrator:
    """Manages the lifecycle of AlphaZero training."""

    def __init__(self, config: AlphaZeroConfig) -> None:
        self.config = config
        self.env = HexTicTacToePgx()
        self.network = AlphaZeroNet()
        
        # 1. Initialize PRNG Keys
        self.key = jax.random.PRNGKey(config.seed)
        self.key, net_key = jax.random.split(self.key)
        
        # 2. Setup TrainState (Parameters + Optimizer)
        self.train_state = create_train_state(net_key, learning_rate=config.optimizer.learning_rate)
        
        # 3. Setup Checkpointing
        ckpt_dir = os.path.abspath(config.checkpoint_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            ckpt_dir, 
            options=orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        )
        
        # Restore if possible
        latest_step = self.checkpoint_manager.latest_step()
        if latest_step is not None:
            print(f"Restoring from checkpoint at step {latest_step}...")
            restore_args = orbax.checkpoint.args.StandardRestore(self.train_state)
            self.train_state = self.checkpoint_manager.restore(latest_step, args=restore_args)
            self.start_step = latest_step + 1
        elif config.checkpoint_step is not None:
            # Manual step restore if specified but not found in latest_step
             print(f"Restoring from specific checkpoint step {config.checkpoint_step}...")
             restore_args = orbax.checkpoint.args.StandardRestore(self.train_state)
             self.train_state = self.checkpoint_manager.restore(config.checkpoint_step, args=restore_args)
             self.start_step = config.checkpoint_step + 1
        else:
            self.start_step = 1

        # 4. Setup Replay Buffer
        self.buffer = create_replay_buffer(
            max_length=config.optimizer.buffer_size,
            min_length=config.optimizer.batch_size,
            sample_batch_size=config.optimizer.batch_size,
        )
        self.buffer_state = init_buffer_state(
            self.buffer, 
            dummy_env_obs_shape=(3, 106, 106), 
            action_size=11236
        )
        self.trajectory_buffer = TrajectoryBuffer(num_envs=config.num_envs)

        # 5. JIT Compilations
        self._setup_jitted_functions()

    def _setup_jitted_functions(self) -> None:
        """Compiles the core training operations."""
        # actor step
        @functools.partial(jax.jit, static_argnames=("num_simulations",))
        def jitted_actor_step(params, e_state, r_key, num_simulations):
            return self_play_step(self.env, self.network, params, e_state, r_key, num_simulations)
        self.jitted_actor_step = cast(Any, jitted_actor_step)

        # trainer steps
        self.jitted_buffer_add = jax.jit(self.buffer.add)
        self.jitted_buffer_sample = jax.jit(self.buffer.sample)
        self.jitted_train_step = jax.jit(train_step)

        # evaluation
        @functools.partial(jax.jit, static_argnames=("num_games", "num_simulations"))
        def jitted_eval_step(params, r_key, num_games, num_simulations):
            return evaluate_vs_random(self.env, self.network, params, r_key, num_games, num_simulations)
        self.jitted_eval_step = cast(Any, jitted_eval_step)

    def train(self) -> None:
        """Runs the main training loop."""
        if self.config.logging.use_wandb:
            # We use OmegaConf.to_container to log the config to wandb
            from omegaconf import OmegaConf
            wandb.init(
                project=self.config.logging.project_name, 
                config=OmegaConf.to_container(self.config, resolve=True) # type: ignore
            )

        # Initialize batched environments
        self.key, env_key = jax.random.split(self.key)
        env_keys = jax.random.split(env_key, self.config.num_envs)
        batched_env_state = jax.jit(jax.vmap(self.env.init))(env_keys)

        print(f"Starting training from step {self.start_step}...")
        for step in range(self.start_step, self.config.total_steps + 1):
            start_time = time.time()

            # 1. ACT: Self-Play
            self.key, sp_key = jax.random.split(self.key)
            batched_env_state, transitions, _, terminated, rewards = self.jitted_actor_step(
                cast(Any, self.train_state).params, 
                batched_env_state, 
                sp_key, 
                num_simulations=self.config.mcts.num_simulations
            )

            # 2. STORE: Manage trajectories and replay buffer
            # Move to CPU for trajectory management
            self.trajectory_buffer.add_step(
                jax.device_get(transitions["observation"]),
                jax.device_get(transitions["target_policy"]),
                jax.device_get(transitions["current_player"]),
                jax.device_get(terminated),
                jax.device_get(rewards)
            )

            # Push completed trajectories to replay buffer
            while (batch := self.trajectory_buffer.get_add_batch(self.config.num_envs)) is not None:
                self.buffer_state = self.jitted_buffer_add(self.buffer_state, batch)

            # 3. LEARN: Optimize Network
            metrics = {}
            if self.buffer.can_sample(self.buffer_state):
                self.key, sample_key = jax.random.split(self.key)
                batch = self.jitted_buffer_sample(self.buffer_state, sample_key)
                self.train_state, metrics = self.jitted_train_step(self.train_state, batch.experience)

            # 4. LOG & SAVE
            step_duration = time.time() - start_time
            self._handle_logging(step, metrics, step_duration)
            self._handle_checkpointing(step)
            self._handle_evaluation(step)

    def _handle_logging(self, step: int, metrics: dict, duration: float) -> None:
        if step % self.config.logging.log_interval == 0 and metrics:
            log_data = {
                "step": step,
                "total_loss": float(metrics["total_loss"]),
                "policy_loss": float(metrics["policy_loss"]),
                "value_loss": float(metrics["value_loss"]),
                "env_steps_per_sec": self.config.num_envs / duration,
            }
            print(f"[{step}] Loss: {log_data['total_loss']:.4f} | Speed: {log_data['env_steps_per_sec']:.1f} steps/s")
            if self.config.logging.use_wandb:
                wandb.log(log_data)

    def _handle_checkpointing(self, step: int) -> None:
        if step % self.config.logging.save_interval == 0:
            print(f"Saving checkpoint at step {step}...")
            save_args = orbax.checkpoint.args.StandardSave(self.train_state)
            self.checkpoint_manager.save(step, args=save_args)

    def _handle_evaluation(self, step: int) -> None:
        if step % self.config.logging.eval_interval == 0:
            print(f"Evaluating at step {step}...")
            self.key, eval_key = jax.random.split(self.key)
            eval_metrics = self.jitted_eval_step(
                cast(Any, self.train_state).params, 
                eval_key, 
                num_games=self.config.logging.eval_games, 
                num_simulations=self.config.mcts.num_simulations
            )
            eval_results = {k: float(v) for k, v in eval_metrics.items()}
            print(f"Eval results: Win {eval_results['win_rate']:.2f} | Draw {eval_results['draw_rate']:.2f}")
            if self.config.logging.use_wandb:
                wandb.log({f"eval/{k}": v for k, v in eval_results.items()} | {"step": step})
