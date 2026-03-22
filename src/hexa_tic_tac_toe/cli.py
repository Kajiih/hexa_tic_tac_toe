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


@app.command()
def train(
    num_envs: int = 128,
    num_simulations: int = 25,
    total_steps: int = 10_000,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    buffer_size: int = 50_000,
    log_interval: int = 10,
    save_interval: int = 1000,
    eval_interval: int = 100,
    eval_games: int = 64,
    checkpoint_dir: str = "./checkpoints",
    use_wandb: bool = False,
    seed: int = 42,
):
    """Trains the AlphaZero agent using self-play."""
    from hexa_tic_tac_toe.agent.orchestrator import AlphaZeroOrchestrator
    orchestrator = AlphaZeroOrchestrator(locals())
    orchestrator.train()


@app.command()
def eval(
    num_games: int = 64,
    num_simulations: int = 25,
    checkpoint: Optional[str] = None,
):
    """Evaluates the AlphaZero agent against a random baseline."""
    from hexa_tic_tac_toe.agent.orchestrator import AlphaZeroOrchestrator
    import jax
    
    config = {"checkpoint_dir": checkpoint if checkpoint else "./checkpoints"}
    orchestrator = AlphaZeroOrchestrator(config)
    
    key = jax.random.PRNGKey(42)
    # We cast to avoid lint errors regarding parameters on TrainState
    params = cast(Any, orchestrator.train_state).params if hasattr(orchestrator.train_state, "params") else None
    
    if params is None:
        typer.echo("Error: Could not retrieve parameters from TrainState.")
        raise typer.Exit(code=1)

    metrics = orchestrator.jitted_eval_step(
        params, key, num_games, num_simulations
    )
    typer.echo(f"Evaluation results: {metrics}")


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
