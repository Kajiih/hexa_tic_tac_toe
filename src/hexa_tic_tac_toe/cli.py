"""Unified Command-Line Interface for Hexagonal Tic-Tac-Toe."""

import json
import random
import time
from pathlib import Path
from typing import Optional, Any, cast

import typer
from typing_extensions import Annotated

from hexa_tic_tac_toe.core.engine import HexGame
from hexa_tic_tac_toe.core.constants import RADIUS, WIN_LENGTH, PLAYER_1, PLAYER_2


app = typer.Typer(help="Hexagonal Tic-Tac-Toe CLI", no_args_is_help=True)


@app.command()
def sync_gui():
    """Synchronizes Python game constants with the React visualizer."""
    config = {
        "radius": RADIUS,
        "winLength": WIN_LENGTH,
        "player1": PLAYER_1,
        "player2": PLAYER_2,
    }
    target_path = Path("visualizer/src/config.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w") as f:
        json.dump(config, f, indent=4)
    typer.echo(f"Synced game config to {target_path}")


@app.command()
def export(
    radius: Annotated[int, typer.Option(help="Radius of the board")] = RADIUS,
    output: Annotated[Path, typer.Option(help="Output JSON path")] = Path("visualizer/public/game_data.json"),
):
    """Plays a random game and exports the history to JSON for the visualizer."""
    game = HexGame(radius=radius)
    coords = list(game.get_all_coordinates())
    random.shuffle(coords)

    winner = None
    for q, r in coords:
        winner = game.make_move(q, r)
        if winner:
            break

    data = {"radius": radius, "history": game.move_history, "winner": winner}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(data, f)
    typer.echo(f"Exported game with {len(game.move_history)} moves to {output}")


@app.command()
def bench(
    num_games: Annotated[int, typer.Option(help="Number of games to simulate")] = 1000,
    radius: Annotated[int, typer.Option(help="Radius of the board")] = RADIUS,
):
    """Measures the performance of the HexGame engine."""
    start_time = time.time()
    total_moves = 0
    wins = {1: 0, 2: 0, None: 0}

    game_instance = HexGame(radius=radius)
    all_coords = list(game_instance.get_all_coordinates())

    typer.echo(f"Starting benchmark: {num_games} games on radius {radius} board ({len(all_coords)} cells)")

    for i in range(num_games):
        game = HexGame(radius=radius)
        coords = all_coords.copy()
        random.shuffle(coords)

        winner = None
        for q, r in coords:
            total_moves += 1
            winner = game.make_move(q, r)
            if winner:
                break
        wins[winner] += 1

        if (i + 1) % 100 == 0:
            typer.echo(f"Finished {i + 1} games...")

    duration = time.time() - start_time
    typer.echo("\nBenchmark Results:")
    typer.echo(f"Total Games: {num_games} | Total Moves: {total_moves}")
    typer.echo(f"Total Time: {duration:.2f}s | Games/s: {num_games / duration:.2f} | Moves/s: {total_moves / duration:.2f}")
    typer.echo(f"Wins: P1: {wins[1]}, P2: {wins[2]}, Draw: {wins[None]}")


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train(ctx: typer.Context):
    """Trains the AlphaZero agent using self-play (Hydra enabled)."""
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from hexa_tic_tac_toe.agent.orchestrator import AlphaZeroOrchestrator
    from hexa_tic_tac_toe.agent.config import AlphaZeroConfig

    # 1. Initialize Hydra from the configs directory
    # We use a context manager to avoid global initialization issues
    with initialize(version_base=None, config_path="../../configs"):
        # 2. Compose config from default and CLI overrides
        # ctx.args contains any extra arguments passed to the command
        cfg_yaml = compose(config_name="config", overrides=ctx.args)
        
        # 3. Convert to structured config (AlphaZeroConfig)
        # We merge with a structured config to ensure type safety
        base_cfg = OmegaConf.structured(AlphaZeroConfig)
        cfg = OmegaConf.merge(base_cfg, cfg_yaml)
        
        # 4. Initialize and run orchestrator
        orchestrator = AlphaZeroOrchestrator(cast(AlphaZeroConfig, cfg))
        orchestrator.train()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def eval(ctx: typer.Context):
    """Evaluates the AlphaZero agent against a random baseline (Hydra enabled)."""
    import jax
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from hexa_tic_tac_toe.agent.orchestrator import AlphaZeroOrchestrator
    from hexa_tic_tac_toe.agent.config import AlphaZeroConfig

    with initialize(version_base=None, config_path="../../configs"):
        cfg_yaml = compose(config_name="config", overrides=ctx.args)
        base_cfg = OmegaConf.structured(AlphaZeroConfig)
        cfg = cast(AlphaZeroConfig, OmegaConf.merge(base_cfg, cfg_yaml))
        
        orchestrator = AlphaZeroOrchestrator(cfg)
        
        key = jax.random.PRNGKey(cfg.seed)
        # We cast to avoid lint errors regarding parameters on TrainState
        params = cast(Any, orchestrator.train_state).params if hasattr(orchestrator.train_state, "params") else None
        
        if params is None:
            typer.echo("Error: Could not load model parameters for evaluation.")
            raise typer.Exit(code=1)
            
        typer.echo(f"Evaluating model against random agent ({cfg.logging.eval_games} games)...")
        metrics = orchestrator.jitted_eval_step(
            params, 
            key, 
            num_games=cfg.logging.eval_games, 
            num_simulations=cfg.mcts.num_simulations
        )
        
        typer.echo(f"Results: Win Rate: {metrics['win_rate']:.2f} | Draw Rate: {metrics['draw_rate']:.2f}")


@app.command()
def play(
    radius: int = RADIUS,
    human_first: bool = True,
    num_simulations: int = 100,
    checkpoint: str = "./checkpoints",
):
    """Starts an interactive game against the AlphaZero agent."""
    import os
    import jax
    from hexa_tic_tac_toe.agent.player import AlphaZeroPlayer
    from hexa_tic_tac_toe.env.pgx_env import HexTicTacToePgx

    def clear_screen():
        os.system("cls" if os.name == "nt" else "clear")

    # 1. Initialize Game & AI
    game = HexGame(radius=radius)
    env = HexTicTacToePgx()
    player = AlphaZeroPlayer(checkpoint_dir=checkpoint)
    
    # Initialize Pgx State
    key = jax.random.PRNGKey(random.randint(0, 10000))
    state = env.init(key)

    human_player_idx = 0 if human_first else 1
    ai_player_idx = 1 - human_player_idx

    while not state.terminated:
        clear_screen()
        typer.echo(f"Hexagonal Tic-Tac-Toe (Radius: {radius})")
        typer.echo(f"Goal: {WIN_LENGTH} in a row")
        typer.echo("-" * 20)
        typer.echo(game)
        typer.echo("-" * 20)
        
        curr_player = int(state.current_player)
        if curr_player == human_player_idx:
            # Human Turn
            valid_move = False
            while not valid_move:
                try:
                    move_input = typer.prompt("Your move (q,r)").strip()
                    if "," not in move_input:
                        typer.echo("Format: q,r (e.g., 0,0)")
                        continue
                    q, r = map(int, move_input.split(","))
                    if not game.is_valid_move(q, r):
                        typer.echo("Invalid move or already occupied.")
                        continue
                    
                    action = game._coord_to_index(q, r)
                    game.make_move(q, r)
                    state = env.step(state, action, key)
                    valid_move = True
                except ValueError:
                    typer.echo("Parsing error. Use integers for q and r.")
        else:
            # AI Turn
            typer.echo("AI is thinking...")
            action = player.decide_move(state, num_simulations=num_simulations)
            
            # Find coordinates for display
            q, r = None, None
            for cq, cr in game.get_all_coordinates():
                if game._coord_to_index(cq, cr) == action:
                    q, r = cq, cr
                    break
            
            typer.echo(f"AI chose: {q},{r}")
            time.sleep(1.0) # brief pause for readability
            assert q is not None and r is not None
            game.make_move(q, r)
            state = env.step(state, action, key)

    # FINAL STATE
    clear_screen()
    typer.echo(game)
    typer.echo("-" * 20)
    winner = int(game.winner) if game.winner else None
    if winner == human_player_idx + 1:
        typer.echo("Congratulations! You won!")
    elif winner == ai_player_idx + 1:
        typer.echo("AI wins! Better luck next time.")
    else:
        typer.echo("It's a draw!")


@app.command()
def replay(
    radius: int = 3,
    delay: float = 0.5,
    moves: Annotated[Optional[list[str]], typer.Argument(help="List of 'q,r' coordinates")] = None,
):
    """Replays a specific game in the terminal."""
    import os
    
    def clear_screen():
        os.system("cls" if os.name == "nt" else "clear")

    game = HexGame(radius=radius)
    
    if not moves:
        typer.echo("No moves provided. Playing a sample game.")
        parsed_moves = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (-1, 0), (-2, 0)]
    else:
        parsed_moves = []
        for m in moves:
            q, r = map(int, m.split(","))
            parsed_moves.append((q, r))

    for i, (q, r) in enumerate(parsed_moves):
        clear_screen()
        typer.echo(f"Replaying Game (Radius: {radius})")
        typer.echo(f"Move {i + 1}/{len(parsed_moves)}: ({q}, {r})")
        typer.echo("-" * 20)
        game.make_move(q, r)
        typer.echo(game)
        if i < len(parsed_moves) - 1:
            time.sleep(delay)
    
    typer.echo("-" * 20)
    typer.echo("Replay Finished.")


def main():
    app()


if __name__ == "__main__":
    main()
