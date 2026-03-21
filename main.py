"""Main entry point for running Hexagonal Tic Tac Toe simulations.

This module provides functions to play random games and run them in parallel
to measure performance and gather statistics.
"""

import time
import multiprocessing
import random
from game import HexGame


def play_one_random_game(radius: int = 50) -> int | None:
    """Plays a single random game to completion.

    Args:
        radius: The radius of the hexagonal board. Defaults to 50.

    Returns:
        The player number (1 or 2) who won, or None if it was a draw.
    """
    game = HexGame(radius=radius)

    # Generate and shuffle all valid coordinates
    coords = list(game.get_all_coordinates())
    random.shuffle(coords)

    winner: int | None = None
    for q, r in coords:
        winner = game.make_move(q, r)
        if winner:
            break
    return winner


def run_multiprocess_games(total_games: int = 1000) -> None:
    """Runs multiple random games in parallel using multiprocessing.

    Args:
        total_games: The number of games to simulate. Defaults to 1000.
    """
    print(f"Running {total_games} games in parallel across CPU cores...")
    start_time = time.time()

    num_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(play_one_random_game, [50] * total_games)

    duration = time.time() - start_time
    print(f"Finished {total_games} games in {duration:.2f} seconds.")
    print(f"Throughput: {total_games / duration:.2f} games per second.")

    p1_wins = results.count(1)
    p2_wins = results.count(2)
    draws = results.count(None)
    print(f"Results: P1: {p1_wins}, P2: {p2_wins}, Draws: {draws}")


def main() -> None:
    """Main function to run a demonstration of the game simulations."""
    print("=== Hexagonal Tic-Tac-Toe ===")
    print("Grid Radius: 50 (7351 cells)")
    print("Win Condition: 6 in a row")
    print("Turn Pattern: 1, 2, 2, 2...")
    print("-" * 30)

    # Run a small batch in parallel to show performance
    run_multiprocess_games(500)


if __name__ == "__main__":
    main()
