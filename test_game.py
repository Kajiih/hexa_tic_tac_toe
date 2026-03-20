"""Unit tests for Hexagonal Tic Tac Toe."""

import pytest
from game import HexGame


@pytest.fixture
def game() -> HexGame:
    """Fixture to provide a standard HexGame instance for testing."""
    return HexGame(radius=50)


def test_radius_validation(game: HexGame) -> None:
    """Tests that coordinates are correctly validated against the board radius."""
    # Center is valid
    assert game.is_valid_move(0, 0)
    # Edge is valid
    assert game.is_valid_move(49, 0)
    assert game.is_valid_move(0, 49)
    assert game.is_valid_move(-49, 49)
    # Outside is invalid
    assert not game.is_valid_move(50, 0)
    assert not game.is_valid_move(0, 50)
    assert not game.is_valid_move(25, 25)  # q+r = 50 > 49


def test_turn_pattern(game: HexGame) -> None:
    """Tests the 1-2-2-2... turn pattern of the hexagonal tic-tac-toe game."""
    # Turn 1: P1 plays 1
    assert game.current_player == 1
    game.make_move(0, 0)

    # Turn 2: P2 plays 2
    assert game.current_player == 2
    game.make_move(1, 0)
    assert game.current_player == 2
    game.make_move(2, 0)

    # Turn 3: P1 plays 2
    assert game.current_player == 1
    game.make_move(0, 1)
    assert game.current_player == 1
    game.make_move(0, 2)

    # Turn 4: P2 plays 2
    assert game.current_player == 2


def test_win_r_axis() -> None:
    """Tests a win condition along the r-axis."""
    grid = """
      . . . . . .
     . . . . . . .
    . . . . . . . .
   . . . . . . . . .
  . . . . . . . . . .
 . . . . . X . . . . .
  . . . . O X O . . .
   . . . . O X . . .
    . . . . O X . .
     . . . . O X .
      . . . . O .
    """
    game = HexGame.from_string(grid)
    winner = game.make_move(0, 5)
    assert winner == 1


def test_win_q_axis() -> None:
    """Tests a win condition along the q-axis."""
    grid = """
      . . . . . .
     . . . . . . .
    . . . . . . . .
   . . . . . . . . .
  . . . . . . . . . .
 . . . . . X X X X X .
  . . . . O O O O O O
   . . . . . . . . .
    . . . . . . . .
     . . . . . . .
      . . . . . .
    """
    game = HexGame.from_string(grid)
    winner = game.make_move(5, 0)
    assert winner == 1


def test_win_s_axis() -> None:
    """Tests a win condition along the game_string-axis."""
    grid = """
      . . . . O .
     . . . . O X .
    . . . . O X . .
   . . . . O X . . .
  . . . . O X . . . .
 . . . . O X . . . . .
  . . . . . . . . . .
   . . . . . . . . .
    . . . . . . . .
     . . . . . . .
      . . . . . .
    """
    game = HexGame.from_string(grid)
    winner = game.make_move(5, -5)
    assert winner == 1


def test_grid_creation() -> None:
    """Tests that a HexGame can be correctly created from a string representation."""
    grid = """
      . .
     . X .
      O .
    """
    game = HexGame.from_string(grid)
    assert game.radius == 2
    # Check pieces
    # Center (0,0) is X
    assert not game.is_valid_move(0, 0)
    # (0,1) is O
    assert not game.is_valid_move(-1, 1)

    # Check reconstruction of turn state
    assert game.turn_number == 2
    assert game.current_player == 2
    assert game.moves_this_turn == 1


def test_string_roundtrip(game: HexGame) -> None:
    """Tests that str(game) and from_string(game_string) are consistent."""
    game.make_move(0, 0)  # P1
    game.make_move(1, 0)  # P2
    game_string = str(game)
    game2 = HexGame.from_string(game_string)
    assert str(game) == str(game2)
    assert game.boards == game2.boards
    assert game.current_player == game2.current_player
    assert game.turn_number == game2.turn_number


def test_from_string_vs_manual() -> None:
    """Tests that creating a game from string matches a manually created game."""
    grid = """
      O X
     O X X
      O O
    """
    game_str = HexGame.from_string(grid)

    game_manual = HexGame(radius=2)
    game_manual.make_move(1, -1)  # X (P1 Turn 1)
    game_manual.make_move(0, -1)  # O (P2 Turn 2)
    game_manual.make_move(-1, 0)  # O (P2 Turn 2)
    game_manual.make_move(0, 0)  # X (P1 Turn 3)
    game_manual.make_move(1, 0)  # X (P1 Turn 3)
    game_manual.make_move(0, 1)  # O (P2 Turn 4)
    game_manual.make_move(-1, 1)  # O (P2 Turn 4)
    game_manual.current_player = 1

    assert game_str.boards == game_manual.boards
    assert game_str.current_player == game_manual.current_player
    assert game_str.turn_number == game_manual.turn_number
    assert game_str.moves_this_turn == game_manual.moves_this_turn
    assert str(game_str) == str(game_manual)


def test_occupied_cell(game: HexGame) -> None:
    """Tests that making a move in an already occupied cell raises an error."""
    game.make_move(0, 0)
    with pytest.raises(ValueError, match="already occupied"):
        game.make_move(0, 0)
