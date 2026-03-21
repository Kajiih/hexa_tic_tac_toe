"""Unit tests for Hexagonal Tic Tac Toe."""

import pytest
from game import HexGame


@pytest.fixture
def game() -> HexGame:
    """Fixture to provide a standard HexGame instance for testing."""
    return HexGame(radius=50)


# --- Radius Validation ---
@pytest.mark.parametrize(
    ("q", "r", "expected"),
    [
        pytest.param(0, 0, True, id="center"),
        pytest.param(49, 0, True, id="q_edge"),
        pytest.param(0, 49, True, id="r_edge"),
        pytest.param(-49, 49, True, id="s_edge"),
        pytest.param(50, 0, False, id="q_outside_bounds"),
        pytest.param(0, 50, False, id="r_outside_bounds"),
        pytest.param(25, 25, False, id="s_outside_bounds_positive"),
        pytest.param(-25, -25, False, id="s_outside_bounds_negative"),
        pytest.param(-49, -1, False, id="q_r_negative"),
        pytest.param(49, 1, False, id="q_r_positive"),
        pytest.param(100, 100, False, id="distinctly_outside"),
    ],
)
def test_radius_validation(game: HexGame, q: int, r: int, *, expected: bool) -> None:
    """Tests that coordinates are correctly validated against the board radius."""
    assert game.is_valid_move(q, r) is expected


# --- Invalid Move Validations ---
@pytest.mark.parametrize(
    ("q", "r", "match_text"),
    [
        pytest.param(0, 0, "already occupied", id="already_occupied"),
        pytest.param(50, 0, "outside the radius", id="q_outside_radius"),
        pytest.param(0, 50, "outside the radius", id="r_outside_radius"),
        pytest.param(25, 25, "outside the radius", id="s_outside_radius_positive"),
        pytest.param(-25, -25, "outside the radius", id="s_outside_radius_negative"),
    ],
)
def test_invalid_make_move(game: HexGame, q: int, r: int, match_text: str) -> None:
    """Tests that invalid moves correctly raise appropriate ValueErrors."""
    game.make_move(0, 0)  # Occupy center before testing
    with pytest.raises(ValueError, match=match_text):
        game.make_move(q, r)


# --- Turn Pattern Logic ---
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


# --- Real Win Validations ---
# These grids represent states where a player has already won with 6-in-a-row.
WIN_R_GRID = """
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
      . . . . O X
"""

WIN_Q_GRID = """
      . . . . . .
     . . . . . . .
    . . . . . . . .
   . . . . . . . . .
  . . . . . . . . . .
 . . . . . X X X X X X
  . . . . O O O O O O
   . . . . . . . . .
    . . . . . . . .
     . . . . . . .
      . . . . . .
"""

WIN_S_GRID = """
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

FIVE_IN_A_ROW_Q_GRID = """
      . . . . . .
     . . . . . . .
    . . . . . . . .
   . . . . . . . . .
  . . . . . . . . . .
 . . . . . X X X X X .
  . . . . . . . . . .
   . . . . . . . . .
    . . . . . . . .
     . . . . . . .
      . . . . . .
"""

FIVE_IN_A_ROW_R_GRID = """
      . . . . . .
     . . . . . . .
    . . . . . . . .
   . . . . . . . . .
  . . . . . . . . . .
 . . . . . X . . . . .
  . . . . . X . . . .
   . . . . . X . . .
    . . . . . X . .
     . . . . . X .
      . . . . . .
"""

FIVE_IN_A_ROW_S_GRID = """
      . . . . . .
     . . . . . . .
    . . . . . . . .
   . . . . . . . . .
  . . . . . . . . . .
 . . . . . X . . . . .
  . . . . X . . . . .
   . . . X . . . . .
    . . X . . . . .
     . X . . . . .
      . . . . . .
"""


@pytest.mark.parametrize(
    ("grid", "expected_winner"),
    [
        pytest.param(WIN_R_GRID, 1, id="win_along_r_axis"),
        pytest.param(WIN_Q_GRID, 1, id="win_along_q_axis"),
        pytest.param(WIN_S_GRID, 2, id="win_along_s_axis"),
        pytest.param(FIVE_IN_A_ROW_Q_GRID, None, id="five_in_a_row_q"),
        pytest.param(FIVE_IN_A_ROW_R_GRID, None, id="five_in_a_row_r"),
        pytest.param(FIVE_IN_A_ROW_S_GRID, None, id="five_in_a_row_s"),
    ],
)
def test_static_win_conditions(grid: str, expected_winner: int) -> None:
    """Tests that wins are correctly identified from static board states."""
    game = HexGame.from_string(grid)
    assert game.winner == expected_winner


def test_turn_reconstruction() -> None:
    """Tests that turn state is still correctly reconstructed even if someone won."""
    game = HexGame.from_string(WIN_R_GRID)
    # 6 X's + 6 O's = 12 pieces => Turn 7, P1, 1 move made.
    assert game.turn_number == 7
    assert game.current_player == 1
    assert game.moves_this_turn == 1


def test_winner_detected_during_parsing() -> None:
    """Tests that winners are correctly identified immediately after parsing."""
    # FIVE_IN_A_ROW_Q_GRID only has 5 X's (No winner)
    game = HexGame.from_string(FIVE_IN_A_ROW_Q_GRID)
    assert game.winner is None

    # FIVE_IN_A_ROW_R_GRID only has 5 X's (No winner)
    game = HexGame.from_string(FIVE_IN_A_ROW_R_GRID)
    assert game.winner is None

    # FIVE_IN_A_ROW_S_GRID only has 5 X's (No winner)
    game = HexGame.from_string(FIVE_IN_A_ROW_S_GRID)
    assert game.winner is None


# --- Parsing Exceptions ---
@pytest.mark.parametrize(
    ("grid_str", "match_text"),
    [
        pytest.param("", "Empty grid string", id="empty_string"),
        pytest.param("   \n   ", "Empty grid string", id="whitespace_string"),
        pytest.param(" . . \n  . ", "odd number of rows", id="even_number_of_rows"),
        pytest.param(
            " . . \n . . \n . . ", "has 2 cells, expected 3", id="incorrect_row_length"
        ),
        pytest.param(
            "  X ?\n . . .\n  . .", "Invalid characters '?'", id="invalid_character"
        ),
    ],
)
def test_from_string_invalid(grid_str: str, match_text: str) -> None:
    """Tests that invalid strings raise the correct ValueErrors during parsing."""
    with pytest.raises(ValueError, match=match_text):
        HexGame.from_string(grid_str)


# --- String Roundtrip Operations ---
@pytest.mark.parametrize(
    "move_sequence",
    [
        pytest.param([], id="empty_board"),
        pytest.param([(0, 0)], id="single_piece"),
        pytest.param([(0, 0), (1, 0)], id="two_pieces"),
        pytest.param(
            [(0, 0), (1, 0), (-1, -1), (0, 1), (-1, 0), (2, -1), (0, 2)],
            id="mid_game_sequence",
        ),
    ],
)
def test_string_roundtrip(move_sequence: list[tuple[int, int]]) -> None:
    """Tests that str(game) and from_string(game_string) are consistent."""
    game = HexGame(radius=4)
    for q, r in move_sequence:
        game.make_move(q, r)

    game_string = str(game)
    game2 = HexGame.from_string(game_string)
    assert str(game) == str(game2)
    # Compare public state
    assert game.current_player == game2.current_player
    assert game.turn_number == game2.turn_number
    assert game.moves_this_turn == game2.moves_this_turn
    # Verify boards match by checking every coordinate
    for q, r in game.get_all_coordinates():
        assert game.get_player_at(q, r) == game2.get_player_at(q, r)


# --- String vs Manual Setup Equivalencies ---
@pytest.mark.parametrize(
    ("grid", "move_sequence"),
    [
        pytest.param("  . .\n . . .\n  . .", [], id="empty_board"),
        pytest.param("  X .\n . . .\n  . .", [(0, -1)], id="single_piece_p1"),
        pytest.param(
            "  O X\n O X X\n  O O",
            [(1, -1), (0, -1), (-1, 0), (0, 0), (1, 0), (0, 1), (-1, 1)],
            id="mid_game_7_pieces",
        ),
    ],
)
def test_from_string_vs_manual(grid: str, move_sequence: list[tuple[int, int]]) -> None:
    """Tests that creating a game from string matches a manually created game."""
    game_str = HexGame.from_string(grid)

    game_manual = HexGame(radius=2)
    for q, r in move_sequence:
        game_manual.make_move(q, r)

    if len(move_sequence) == 7:
        game_manual.current_player = 1

    assert game_str.current_player == game_manual.current_player
    assert game_str.turn_number == game_manual.turn_number
    assert game_str.moves_this_turn == game_manual.moves_this_turn
    assert str(game_str) == str(game_manual)
    for q, r in game_manual.get_all_coordinates():
        assert game_str.get_player_at(q, r) == game_manual.get_player_at(q, r)
