"""Terminal-based replay utility for Hexagonal Tic Tac Toe."""

import time
import os
import sys
from game import HexGame


def clear_screen() -> None:
    """Clears the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def replay_game(
    moves: list[tuple[int, int]], radius: int = 3, delay: float = 0.5
) -> None:
    """Replays a game in the terminal.

    Args:
        moves: A list of (q, r) coordinates representing the game history.
        radius: The radius of the board. Defaults to 3.
        delay: Time in seconds between moves. Defaults to 0.5.
    """
    game = HexGame(radius=radius)

    for i, (q, r) in enumerate(moves):
        clear_screen()
        print(f"Replaying Game (Radius: {radius})")
        print(f"Move {i + 1}/{len(moves)}: ({q}, {r})")
        print("-" * 20)

        game.make_move(q, r)
        print(game)

        if i < len(moves) - 1:
            time.sleep(delay)

    print("-" * 20)
    print("Replay Finished.")


def main() -> None:
    """Main entry point for the replay utility."""
    # Sample game if no arguments provided
    if len(sys.argv) < 2:
        print(
            "Usage: python replay.py <radius> <move_q1,move_r1> <move_q2,move_r2> ..."
        )
        print("Example: python replay.py 3 0,0 1,0 0,1")

        # Default sample for demonstration
        sample_moves = [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (-1, 0), (-2, 0)]
        replay_game(sample_moves, radius=3)
        return

    try:
        radius = int(sys.argv[1])
        moves = []
        for move_str in sys.argv[2:]:
            q, r = map(int, move_str.split(","))
            moves.append((q, r))

        replay_game(moves, radius=radius)
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
