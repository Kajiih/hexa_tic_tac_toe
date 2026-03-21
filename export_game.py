"""Utility script to export a random game for the visualizer."""

import json
import random
from pathlib import Path

from game import HexGame


def export_random_game(radius: int = 5) -> None:
    """Plays a random game and exports the history to JSON."""
    game = HexGame(radius=radius)
    coords = list(game.get_all_coordinates())
    random.shuffle(coords)  # noqa: S311

    winner = None
    for q, r in coords:
        winner = game.make_move(q, r)
        if winner:
            break

    data = {"radius": radius, "history": game.move_history, "winner": winner}

    # Save to the visualizer's public folder for easy fetching
    out_path = Path("visualizer/public/game_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f)

    print(  # noqa: T201
        f"Exported game with {len(game.move_history)} moves to {out_path}"
    )


if __name__ == "__main__":
    export_random_game(radius=5)
