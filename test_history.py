"""Tests for the game history and undo logic."""

from game import HexGame


def test_move_history() -> None:
    """Tests that the game correctly tracks the history of moves."""
    game = HexGame(radius=3)
    moves = [(0, 0), (1, 0), (0, 1)]
    for q, r in moves:
        game.make_move(q, r)

    assert game.move_history == moves


def test_undo_move_logic() -> None:
    """Tests that undoing moves correctly reverts the board and turn state."""
    game = HexGame(radius=3)

    # Initial state
    assert game.current_player == 1
    assert game.turn_number == 1
    assert game.moves_this_turn == 0

    # Move 1 (P1 Turn 1)
    game.make_move(0, 0)
    assert game.current_player == 2
    assert game.turn_number == 2
    assert game.moves_this_turn == 0

    # Undo 1
    game.undo_move()
    assert game.current_player == 1
    assert game.turn_number == 1
    assert game.moves_this_turn == 0
    assert all(game.get_player_at(q, r) is None for q, r in game.get_all_coordinates())
    assert game.move_history == []

    # Re-apply move 1
    game.make_move(0, 0)

    # Move 2 (P2 Turn 2 - move 1 of 2)
    game.make_move(1, 0)
    assert game.current_player == 2
    assert game.turn_number == 2
    assert game.moves_this_turn == 1

    # Move 3 (P2 Turn 2 - move 2 of 2)
    game.make_move(2, 0)
    assert game.current_player == 1
    assert game.turn_number == 3
    assert game.moves_this_turn == 0

    # Undo Move 3
    game.undo_move()
    assert game.current_player == 2
    assert game.turn_number == 2
    assert game.moves_this_turn == 1

    # Undo Move 2
    game.undo_move()
    assert game.current_player == 2
    assert game.turn_number == 2
    assert game.moves_this_turn == 0

    # Undo Move 1
    game.undo_move()
    assert game.current_player == 1
    assert game.turn_number == 1
    assert game.moves_this_turn == 0
    assert all(game.get_player_at(q, r) is None for q, r in game.get_all_coordinates())


def test_reset() -> None:
    """Tests that the reset method clears all state."""
    game = HexGame(radius=3)
    game.make_move(0, 0)
    game.make_move(1, 0)
    game.reset()

    assert all(game.get_player_at(q, r) is None for q, r in game.get_all_coordinates())
    assert game.current_player == 1
    assert game.turn_number == 1
    assert game.moves_this_turn == 0
    assert game.move_history == []
