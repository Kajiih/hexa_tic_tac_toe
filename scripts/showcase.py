"""Showcase script for Hexagonal Tic-Tac-Toe features."""

import subprocess
import time
from export_game import export_random_game


def run_command(cmd_args: list[str]) -> None:
    print(f"Executing: {' '.join(cmd_args)}")  # noqa: T201
    subprocess.run(cmd_args, check=False)


def main() -> None:
    print("=== Hexagonal Tic-Tac-Toe Showcase ===")

    # 1. Run a small benchmark
    print("\n[1/3] Running a quick performance benchmark...")  # noqa: T201
    run_command(["python3", "benchmark.py"])

    # 2. Export a fresh game for the visualizer
    print("\n[2/3] Exporting a random game for the web visualizer...")  # noqa: T201
    export_random_game(radius=5)

    # 3. Run a terminal replay
    print("\n[3/3] Launching a terminal replay demonstration...")  # noqa: T201
    time.sleep(1)
    # Using small moves for the terminal demo
    sample_moves = ["0,0", "1,0", "2,0", "0,1", "0,2", "-1,0", "-2,0"]
    run_command(["python3", "replay.py", "3"] + sample_moves)

    print("\nShowcase Complete!")
    print("\nTo see the high-end web visualizer:")
    print("1. cd visualizer")
    print("2. npm run dev")


if __name__ == "__main__":
    main()
