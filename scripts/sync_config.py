"""Utility to synchronize Python game constants with the React visualizer."""

import json
import os
from hexa_tic_tac_toe.core.constants import RADIUS, WIN_LENGTH, PLAYER_1, PLAYER_2

def sync_config():
    config = {
        "radius": RADIUS,
        "winLength": WIN_LENGTH,
        "player1": PLAYER_1,
        "player2": PLAYER_2,
    }
    
    # Path to the visualizer's src directory
    # Assuming this script is run from the project root
    target_path = os.path.join("visualizer", "src", "config.json")
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with open(target_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Synced game config to {target_path}")

if __name__ == "__main__":
    sync_config()
