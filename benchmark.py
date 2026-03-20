"""Benchmark script for Hexagonal Tic Tac Toe.

This module provides a benchmark function to measure the performance of the
HexGame engine by playing a large number of random games.
"""

import time
import random
from game import HexGame


def run_benchmark(num_games: int = 1000) -> None:
    """Runs a performance benchmark by playing multiple random games.

    Args:
        num_games: The number of games to simulate. Defaults to 1000.
    """
    start_time = time.time()
    total_moves: int = 0
    wins: dict[int | None, int] = {1: 0, 2: 0, None: 0}

    # Pre-calculate all valid moves to save time in the loop
    # (Though in a real game, moves would be checked dynamically)
    radius: int = 50
    all_coords: list[tuple[int, int]] = []
    for q in range(-radius + 1, radius):
        for r in range(-radius + 1, radius):
            if abs(q + r) < radius:
                all_coords.append((q, r))

    print(
        f"Starting benchmark: {num_games} games on radius {radius} board ({len(all_coords)} cells)"
    )

    for i in range(num_games):
        game = HexGame(radius=radius)
        coords = all_coords.copy()
        random.shuffle(coords)

        winner: int | None = None
        for q, r in coords:
            total_moves += 1
            winner = game.make_move(q, r)
            if winner:
                break

        wins[winner] += 1

        if (i + 1) % 100 == 0:
            print(f"Finished {i + 1} games...")

    end_time = time.time()
    duration = end_time - start_time

    print("\nBenchmark Results:")
    print(f"Total Games: {num_games}")
    print(f"Total Moves: {total_moves}")
    print(f"Total Time: {duration:.2f} seconds")
    if duration > 0:
        print(f"Games per Second: {num_games / duration:.2f}")
        print(f"Moves per Second: {total_moves / duration:.2f}")
    print(f"Wins: P1: {wins[1]}, P2: {wins[2]}, Draw: {wins[None]}")


if __name__ == "__main__":
    run_benchmark(1000)
