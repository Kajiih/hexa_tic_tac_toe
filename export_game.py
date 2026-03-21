import json
import random
from game import HexGame


def export_random_game(radius: int = 5) -> None:
    """Plays a random game and exports the history to JSON."""
    game = HexGame(radius=radius)
    coords = list(game.get_all_coordinates())
    random.shuffle(coords)

    winner = None
    for q, r in coords:
        winner = game.make_move(q, r)
        if winner:
            break

    data = {"radius": radius, "history": game.move_history, "winner": winner}

    # Save to the visualizer's public folder for easy fetching
    with open("visualizer/public/game_data.json", "w") as f:
        json.dump(data, f)

    print(
        f"Exported game with {len(game.move_history)} moves to visualizer/public/game_data.json"
    )


if __name__ == "__main__":
    export_random_game(radius=5)
