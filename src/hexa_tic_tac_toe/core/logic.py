"""Shared game logic for Hexagonal Tic-Tac-Toe."""

from hexa_tic_tac_toe.core.constants import PLAYER_1, PLAYER_2, Player

def get_player_for_move_index(index: int) -> Player:
    """Calculates the player for a given move index based on the 1-2-2 game rules.
    
    Player 1 plays the first move (index 0).
    Then players alternate taking two moves each.
    
    Args:
        index: The 0-based index of the move in the game history.
        
    Returns:
        1 for Player 1, 2 for Player 2.
    """
    if index == 0:
        return PLAYER_1
    return PLAYER_2 if ((index + 1) // 2) % 2 == 1 else PLAYER_1
